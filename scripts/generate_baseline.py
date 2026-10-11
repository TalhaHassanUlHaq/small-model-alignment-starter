"""Task 2: generate the fixed AlpacaEval baseline.

The script creates the fixed prompt subset, runs a short sanity check, and
then writes the baseline answers and generation settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from small_model_alignment.data import format_prompt, load_alpaca_eval_subset
from small_model_alignment.generation import (
    generate_baseline_outputs,
    generate_one,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Task 2 — location: ``scripts/generate_baseline.py``.

    TODO steps:
    1. Define arguments for model, seed, subset size, sanity count, and paths.
    2. Define arguments for max new tokens and anti-repetition controls.
    3. Use assignment-compatible defaults: 300 prompts and seed 42.
    4. Return the parsed namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=300)
    parser.add_argument("--sanity-count", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--subset-path",
        default=None,
    )
    parser.add_argument(
        "--output-path",
        default=None,
    )
    return parser.parse_args()


def load_baseline_model(model_name: str):
    """Task 2 — location: ``scripts/generate_baseline.py``.

    TODO steps:
    1. Load the tokenizer from ``model_name``.
    2. If no pad token exists, choose and document a safe padding fallback.
    3. Select fp16 for CUDA/T4 inference and fp32 for CPU fallback.
    4. Load the causal LM with the selected dtype.
    5. Move it to the selected device and switch to evaluation mode.
    6. Return model and tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def build_generation_config(args, tokenizer: AutoTokenizer) -> dict:
    """Task 2 — location: ``scripts/generate_baseline.py``.

    TODO steps:
    1. Set ``do_sample`` to false for greedy decoding.
    2. Set ``max_new_tokens`` so prompt length is not counted in the limit.
    3. Include tokenizer pad/EOS IDs to avoid generation warnings.
    4. Add repetition controls only when their values are greater than their
       neutral setting.
    5. Return and later save the complete resolved configuration.
    """

    config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.no_repeat_ngram_size > 0:
        config["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    return config


def main() -> None:
    """Task 2 — location: ``scripts/generate_baseline.py``.

    TODO steps:
    1. Parse model, seed, prompt count, decoding, and output paths.
    2. Reject invalid sanity counts and subset sizes.
    3. Load AlpacaEval and select the fixed subset with the chosen seed.
    4. Save the subset before running model inference.
    5. Load the tokenizer/model and resolved generation configuration.
    6. Generate and print a 2–5 prompt sanity check.
    7. Generate all fixed prompts in order.
    8. Save required answer records and a settings artifact.
    9. Print output count and paths for later tasks.
    """

    args = parse_args()
    if args.subset_size <= 0:
        raise ValueError("subset-size must be positive")
    if not 0 <= args.sanity_count <= min(5, args.subset_size):
        raise ValueError("sanity-count must be between zero and five")

    subset_path = args.subset_path or (
        f"outputs/eval_sets/alpaca_eval_{args.subset_size}_seed{args.seed}.json"
    )
    output_path = args.output_path or (
        f"outputs/generations/baseline_alpaca_eval_{args.subset_size}_seed{args.seed}.json"
    )

    subset = load_alpaca_eval_subset(args.subset_size, args.seed)
    save_json(subset_path, subset)

    model, tokenizer = load_baseline_model(args.model)
    generation_config = build_generation_config(args, tokenizer)

    for record in subset[:args.sanity_count]:
        response = generate_one(
            model,
            tokenizer,
            format_prompt(record["instruction"]),
            generation_config,
        )
        print(
            f"\nSanity check {record['id']}\n"
            f"Instruction: {record['instruction']}\n"
            f"Response: {response}\n"
        )

    outputs = generate_baseline_outputs(
        model,
        tokenizer,
        subset,
        format_prompt,
        generation_config,
    )
    save_json(output_path, outputs)

    settings_path = Path(output_path).with_suffix(".settings.json")
    save_json(
        settings_path,
        {
            "task": "task2_baseline_generation",
            "model": args.model,
            "seed": args.seed,
            "subset_size": args.subset_size,
            "prompt_template": "Instruction: {instruction}\\n\\nAnswer:",
            "generation_config": generation_config,
            "subset_path": str(subset_path),
            "output_path": str(output_path),
        },
    )
    print(json.dumps({"count": len(outputs), "output": output_path}, indent=2))


if __name__ == "__main__":
    main()
