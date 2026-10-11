"""Task 6: Generate comparable outputs and build blinded pairwise evaluation files.

Generates outputs from SFT and DPO checkpoints using identical decoding configurations,
and compiles randomized, blinded pairs for model-based judging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from small_model_alignment.data import format_prompt, load_alpaca_eval_subset
from small_model_alignment.generation import (
    build_blind_pairs,
    generate_one,
    save_json,
    write_generations,
)
from small_model_alignment.training import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-model", default="outputs/checkpoints/sft_best")
    parser.add_argument("--dpo-uf-model", default="outputs/checkpoints/dpo_ultrafeedback_best")
    parser.add_argument("--dpo-hh-model", default="outputs/checkpoints/dpo_hh_rlhf_best")
    parser.add_argument("--lora-persona-model", default="outputs/checkpoints/lora_dpo_persona")
    parser.add_argument("--base-persona-model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument(
        "--hh-eval-set",
        default="eval_sets/hh_rlhf_eval_300_seed42.json",
    )
    parser.add_argument("--alpaca-eval-count", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/generations")
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--pilot", action="store_true", help="Generate for only 5 prompts")
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    p = Path(model_path)
    if not p.exists():
        fallback = "Qwen/Qwen2.5-0.5B"
        print(f"Notice: Checkpoint {model_path} not found. Using fallback {fallback}")
        model_path = fallback

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    gen_dir = Path(args.output_dir)
    eval_dir = Path(args.eval_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "repetition_penalty": args.repetition_penalty,
    }

    # 1. Load evaluation subsets
    count = 5 if args.pilot else args.alpaca_eval_count

    # UltraFeedback / AlpacaEval prompts
    print("Loading AlpacaEval evaluation subset...")
    alpaca_prompts = load_alpaca_eval_subset(count=count, seed=args.seed)

    # HH-RLHF prompts
    print(f"Loading HH-RLHF evaluation prompts from {args.hh_eval_set}...")
    hh_records = json.loads(Path(args.hh_eval_set).read_text(encoding="utf-8"))
    hh_prompts = []
    for idx, r in enumerate(hh_records[:count]):
        inst = r.get("instruction", r.get("prompt", ""))
        hh_prompts.append({"id": str(idx), "instruction": inst.strip()})

    # Persona prompts
    persona_prompts = [
        {"id": "p0", "instruction": "How do I make a classic mushroom risotto?"},
        {"id": "p1", "instruction": "I boiled the pasta for 45 minutes until it dissolved. Is it ready?"},
        {"id": "p2", "instruction": "Write a quick Python script to calculate Fibonacci numbers."},
        {"id": "p3", "instruction": "How can I prepare a simple lemon herb roasted chicken?"},
        {"id": "p4", "instruction": "What should I do if someone gets a burn while cooking?"},
    ][:count]

    # -------------------------------------------------------------
    # Generate SFT Outputs
    # -------------------------------------------------------------
    print("\n--- Generating SFT Model Outputs ---")
    sft_model, sft_tok = load_model_and_tokenizer(args.sft_model)
    sft_gen_config = {**gen_config, "pad_token_id": sft_tok.pad_token_id, "eos_token_id": sft_tok.eos_token_id}

    # Sanity check
    print("Running sanity check (2 prompts)...")
    for rec in alpaca_prompts[:2]:
        resp = generate_one(sft_model, sft_tok, format_prompt(rec["instruction"]), sft_gen_config)
        print(f"Prompt: {rec['instruction'][:50]}... -> SFT: {resp[:60]}...")

    sft_alpaca_outs = write_generations(
        sft_model, sft_tok, alpaca_prompts, format_prompt, sft_gen_config,
        generator_label="sft", dataset_name="alpaca_eval",
        output_path=gen_dir / "sft_alpaca_eval.json",
    )
    sft_hh_outs = write_generations(
        sft_model, sft_tok, hh_prompts, format_prompt, sft_gen_config,
        generator_label="sft", dataset_name="hh_rlhf",
        output_path=gen_dir / "sft_hh_rlhf.json",
    )
    del sft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Generate UltraFeedback DPO Outputs
    # -------------------------------------------------------------
    print("\n--- Generating UltraFeedback DPO Model Outputs ---")
    uf_model, uf_tok = load_model_and_tokenizer(args.dpo_uf_model)
    uf_gen_config = {**gen_config, "pad_token_id": uf_tok.pad_token_id, "eos_token_id": uf_tok.eos_token_id}

    uf_alpaca_outs = write_generations(
        uf_model, uf_tok, alpaca_prompts, format_prompt, uf_gen_config,
        generator_label="dpo_ultrafeedback", dataset_name="alpaca_eval",
        output_path=gen_dir / "dpo_uf_alpaca_eval.json",
    )
    del uf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build Blind Pairwise JSONL for UltraFeedback DPO vs SFT
    print("Building blind pairwise file: uf_dpo_vs_sft_pairs.jsonl")
    build_blind_pairs(
        records_1=uf_alpaca_outs,
        records_2=sft_alpaca_outs,
        output_path=eval_dir / "uf_dpo_vs_sft_pairs.jsonl",
        seed=args.seed,
    )

    # -------------------------------------------------------------
    # Generate HH-RLHF DPO Outputs
    # -------------------------------------------------------------
    print("\n--- Generating HH-RLHF DPO Model Outputs ---")
    hh_model, hh_tok = load_model_and_tokenizer(args.dpo_hh_model)
    hh_gen_config = {**gen_config, "pad_token_id": hh_tok.pad_token_id, "eos_token_id": hh_tok.eos_token_id}

    hh_dpo_outs = write_generations(
        hh_model, hh_tok, hh_prompts, format_prompt, hh_gen_config,
        generator_label="dpo_hh_rlhf", dataset_name="hh_rlhf",
        output_path=gen_dir / "dpo_hh_eval.json",
    )
    del hh_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build Blind Pairwise JSONL for HH-RLHF DPO vs SFT
    print("Building blind pairwise file: hh_dpo_vs_sft_pairs.jsonl")
    build_blind_pairs(
        records_1=hh_dpo_outs,
        records_2=sft_hh_outs,
        output_path=eval_dir / "hh_dpo_vs_sft_pairs.jsonl",
        seed=args.seed,
    )

    # -------------------------------------------------------------
    # Persona Model Generation (Task 5 Style Check)
    # -------------------------------------------------------------
    if Path(args.lora_persona_model).exists():
        print("\n--- Generating Persona LoRA Model Outputs ---")
        try:
            from peft import AutoPeftModelForCausalLM
            persona_tok = AutoTokenizer.from_pretrained(args.base_persona_model)
            if persona_tok.pad_token_id is None:
                persona_tok.pad_token = persona_tok.eos_token
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            persona_model = AutoPeftModelForCausalLM.from_pretrained(
                args.lora_persona_model,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            persona_model.to(device)
            persona_model.eval()

            p_gen_config = {**gen_config, "pad_token_id": persona_tok.pad_token_id, "eos_token_id": persona_tok.eos_token_id}
            persona_outs = write_generations(
                persona_model, persona_tok, persona_prompts, format_prompt, p_gen_config,
                generator_label="lora_dpo_persona", dataset_name="persona_style",
                output_path=gen_dir / "lora_persona_outputs.json",
            )
            print(f"Generated {len(persona_outs)} persona outputs.")
        except Exception as e:
            print(f"Skipping persona generation: {e}")

    print("\nTask 6 Generation and Pair Construction Complete!")


if __name__ == "__main__":
    main()
