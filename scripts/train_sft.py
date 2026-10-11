"""Task 3: Full-model supervised fine-tuning (SFT) loop.

Trains Qwen/Qwen2.5-0.5B on instruction-response examples with response-only
loss masking, gradient accumulation, fp16 autocast, and checkpointing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from small_model_alignment.batching import build_lm_batch
from small_model_alignment.data import extract_sft_pairs, format_prompt, load_hf_split
from small_model_alignment.training import (
    InstructionDataset,
    save_checkpoint,
    save_json,
    set_seed,
    train_sft_step,
    validate_sft,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset-name", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--train-split", default="train_sft")
    parser.add_argument("--test-split", default="test_sft")
    parser.add_argument("--train-examples", type=int, default=30_000)
    parser.add_argument("--val-examples", type=int, default=1_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--output-dir", default="outputs/checkpoints")
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--pilot", action="store_true", help="Run short 10-step pilot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(device)

    # Load datasets
    train_count = 50 if args.pilot else args.train_examples
    val_count = 20 if args.pilot else args.val_examples

    print(f"Loading SFT dataset from {args.dataset_name}...")
    raw_train = load_hf_split(args.dataset_name, args.train_split, select=train_count * 2, seed=args.seed)
    train_pairs = extract_sft_pairs(raw_train)[:train_count]

    raw_val = load_hf_split(args.dataset_name, args.test_split, select=val_count * 2, seed=args.seed)
    val_pairs = extract_sft_pairs(raw_val)[:val_count]

    print(f"Prepared {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs.")

    def collate_fn(batch_items):
        prompts = [p for p, r in batch_items]
        responses = [r for p, r in batch_items]
        batch = build_lm_batch(tokenizer, prompts, responses, max_length=args.max_length)
        return {k: v.to(device) for k, v in batch.items()}

    train_dataset = InstructionDataset(train_pairs, format_prompt)
    val_dataset = InstructionDataset(val_pairs, format_prompt)

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

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    total_update_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "sft_metrics.jsonl"

    best_val_loss = float("inf")
    best_checkpoint_dir = output_dir / "sft_best"
    final_checkpoint_dir = output_dir / "sft_final"

    print(f"Starting SFT Training: {total_update_steps} update steps, effective batch size {args.per_device_batch_size * args.grad_accum}...")

    step_idx = 0
    update_idx = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for batch in train_loader:
            step_idx += 1
            is_boundary = (step_idx % args.grad_accum == 0) or (step_idx == len(train_loader))

            loss_val = train_sft_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                scaler=scaler,
                accumulation_steps=args.grad_accum,
                is_accumulation_boundary=is_boundary,
                max_grad_norm=args.max_grad_norm,
            )

            if is_boundary:
                scheduler.step()
                update_idx += 1

                if update_idx % 20 == 0 or update_idx == total_update_steps:
                    print(f"Epoch {epoch+1} | Step {update_idx}/{total_update_steps} | Train Loss: {loss_val:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

                # Log training step
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "epoch": epoch + 1,
                        "step": update_idx,
                        "train_loss": loss_val,
                        "lr": scheduler.get_last_lr()[0],
                    }) + "\n")

                if (update_idx % args.eval_every_steps == 0) or (update_idx == total_update_steps):
                    val_loss = validate_sft(model, val_loader)
                    print(f"==> Validation at step {update_idx}: Loss = {val_loss:.4f}")
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1,
                            "step": update_idx,
                            "val_loss": val_loss,
                        }) + "\n")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"Saving new best checkpoint to {best_checkpoint_dir} (Loss: {best_val_loss:.4f})")
                        save_checkpoint(model, tokenizer, best_checkpoint_dir)

    print(f"Saving final checkpoint to {final_checkpoint_dir}...")
    save_checkpoint(model, tokenizer, final_checkpoint_dir)

    # Save resolved config
    config_payload = vars(args)
    config_payload["best_val_loss"] = best_val_loss
    save_json(output_dir / "sft_config.json", config_payload)
    print("SFT Training complete!")


if __name__ == "__main__":
    main()
