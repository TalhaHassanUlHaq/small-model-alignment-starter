"""Task 5: TRL DPOTrainer with LoRA Persona Tuning (Idiot Sandwich).

Trains SmolLM2-360M-Instruct with PEFT/LoRA and TRL DPOTrainer on the Gordon
Ramsay style preference dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from small_model_alignment.training import save_json, set_seed


def format_for_trl(row: dict) -> dict:
    """Task 5 — location: ``scripts/train_lora_dpo.py``.

    Converts prompt, chosen, and rejected strings into TRL conversational format.
    """
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument(
        "--data-path",
        default="data/persona_lora/ramsay_style_preferences.json",
    )
    parser.add_argument("--output-dir", default="outputs/checkpoints/lora_dpo_persona")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading persona dataset from {args.data_path}...")
    raw_data = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    if args.pilot:
        raw_data = raw_data[:10]

    formatted_rows = [format_for_trl(row) for row in raw_data]
    train_dataset = Dataset.from_list(formatted_rows)
    print(f"Prepared {len(train_dataset)} preference training examples.")

    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=256,
        logging_steps=5,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting LoRA DPO Training with TRL...")
    dpo_trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}...")
    dpo_trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    save_json(
        Path(args.output_dir) / "lora_config_resolved.json",
        {
            "model": args.model,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "beta": args.beta,
            "dataset_rows": len(train_dataset),
        },
    )
    print("Task 5 training complete!")


if __name__ == "__main__":
    main()
