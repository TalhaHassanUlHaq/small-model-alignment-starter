"""Task 4: From-scratch Direct Preference Optimization (DPO) loop.

Runs custom DPO on Anthropic HH-RLHF or UltraFeedback preference data starting
from the best SFT checkpoint, logging loss, reward accuracy, and reward margin.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from small_model_alignment.batching import build_preference_batch
from small_model_alignment.data import (
    extract_preference_pairs,
    format_prompt,
    load_hf_split,
)
from small_model_alignment.training import (
    PreferenceDataset,
    dpo_batch_step,
    save_checkpoint,
    save_json,
    set_seed,
    validate_dpo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=["hh_rlhf", "ultrafeedback"],
        required=True,
        help="Which preference optimization track to run",
    )
    parser.add_argument("--sft-checkpoint", default="outputs/checkpoints/sft_best")
    parser.add_argument("--train-pairs", type=int, default=15_000)
    parser.add_argument("--val-pairs", type=int, default=1_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/checkpoints")
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--pilot", action="store_true", help="Run short 10-step pilot")
    return parser.parse_args()


def get_dataset_info(track: str) -> tuple[str, str, str]:
    if track == "hh_rlhf":
        return "Anthropic/hh-rlhf", "train", "test"
    elif track == "ultrafeedback":
        return "HuggingFaceH4/ultrafeedback_binarized", "train_prefs", "test_prefs"
    raise ValueError(f"Unknown track: {track}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        policy_device = torch.device("cuda:0")
        ref_device = torch.device("cuda:1")
        print("Dual-GPU setup detected: Policy on cuda:0, Reference on cuda:1")
    elif num_gpus == 1:
        policy_device = torch.device("cuda:0")
        ref_device = torch.device("cuda:0")
        print("Single-GPU setup detected: Policy and Reference on cuda:0")
    else:
        policy_device = torch.device("cpu")
        ref_device = torch.device("cpu")
        print("CPU setup: Policy and Reference on CPU")

    sft_path = Path(args.sft_checkpoint)
    if not sft_path.exists():
        fallback_model = "Qwen/Qwen2.5-0.5B"
        print(f"Warning: SFT checkpoint {sft_path} not found. Falling back to {fallback_model} for testing.")
        model_source = fallback_model
    else:
        model_source = str(sft_path)

    print(f"Loading tokenizer from {model_source}...")
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading trainable policy model on {policy_device}...")
    policy_model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.float32)
    policy_model.to(policy_device)

    print(f"Loading frozen reference model on {ref_device}...")
    ref_dtype = torch.float16 if ref_device.type == "cuda" else torch.float32
    reference_model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=ref_dtype)
    reference_model.to(ref_device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Load preference dataset
    ds_name, train_split, val_split = get_dataset_info(args.track)
    train_count = 30 if args.pilot else args.train_pairs
    val_count = 15 if args.pilot else args.val_pairs

    print(f"Loading {args.track} dataset ({ds_name})...")
    raw_train = load_hf_split(ds_name, train_split, select=train_count * 2, seed=args.seed)
    train_examples = extract_preference_pairs(raw_train)[:train_count]

    raw_val = load_hf_split(ds_name, val_split, select=val_count * 2, seed=args.seed)
    val_examples = extract_preference_pairs(raw_val)[:val_count]

    print(f"Loaded {len(train_examples)} train pairs, {len(val_examples)} val pairs for track {args.track}")

    def collate_fn(batch_items):
        prompts = [p for p, c, r in batch_items]
        chosen = [c for p, c, r in batch_items]
        rejected = [r for p, c, r in batch_items]
        return build_preference_batch(
            tokenizer,
            prompts,
            chosen,
            rejected,
            max_length=args.max_length,
        )

    train_dataset = PreferenceDataset(train_examples, format_prompt)
    val_dataset = PreferenceDataset(val_examples, format_prompt)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = AdamW(policy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(policy_device.type == "cuda"))

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"dpo_{args.track}_metrics.jsonl"

    best_reward_acc = -1.0
    best_checkpoint_dir = output_dir / f"dpo_{args.track}_best"
    final_checkpoint_dir = output_dir / f"dpo_{args.track}_final"

    total_update_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    print(f"Starting DPO Training: {total_update_steps} update steps (Effective batch size {args.per_device_batch_size * args.grad_accum})...")

    step_idx = 0
    update_idx = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for batch in train_loader:
            step_idx += 1
            is_boundary = (step_idx % args.grad_accum == 0) or (step_idx == len(train_loader))

            step_metrics = dpo_batch_step(
                policy_model=policy_model,
                reference_model=reference_model,
                optimizer=optimizer,
                batch=batch,
                scaler=scaler,
                beta=args.beta,
                accumulation_steps=args.grad_accum,
                is_accumulation_boundary=is_boundary,
                max_grad_norm=args.max_grad_norm,
            )

            if is_boundary:
                update_idx += 1
                if update_idx % 10 == 0 or update_idx == total_update_steps:
                    print(
                        f"Epoch {epoch+1} | Step {update_idx}/{total_update_steps} | "
                        f"Loss: {step_metrics['loss']:.4f} | "
                        f"Reward Acc: {step_metrics['reward_accuracy']:.2f} | "
                        f"Reward Margin: {step_metrics['reward_margin']:.4f}"
                    )

                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "epoch": epoch + 1,
                        "step": update_idx,
                        **step_metrics,
                    }) + "\n")

                if (update_idx % args.eval_every_steps == 0) or (update_idx == total_update_steps):
                    val_metrics = validate_dpo(policy_model, reference_model, val_loader, beta=args.beta)
                    print(
                        f"==> Validation at step {update_idx}: "
                        f"Loss: {val_metrics['val_loss']:.4f} | "
                        f"Reward Acc: {val_metrics['val_reward_accuracy']:.2f} | "
                        f"Reward Margin: {val_metrics['val_reward_margin']:.4f}"
                    )
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1,
                            "step": update_idx,
                            **val_metrics,
                        }) + "\n")

                    if val_metrics["val_reward_accuracy"] > best_reward_acc:
                        best_reward_acc = val_metrics["val_reward_accuracy"]
                        print(f"Saving new best checkpoint to {best_checkpoint_dir} (Reward Acc: {best_reward_acc:.2f})")
                        save_checkpoint(policy_model, tokenizer, best_checkpoint_dir)

    print(f"Saving final checkpoint to {final_checkpoint_dir}...")
    save_checkpoint(policy_model, tokenizer, final_checkpoint_dir)

    config_payload = vars(args)
    config_payload["best_reward_accuracy"] = best_reward_acc
    save_json(output_dir / f"dpo_{args.track}_config.json", config_payload)
    print(f"DPO Training for track {args.track} complete!")


if __name__ == "__main__":
    main()
