# Task 5 — TRL DPOTrainer with LoRA persona tuning

## Goal

Use `SmolLM2-360M-Instruct`, TRL `DPOTrainer`, PEFT/LoRA, and the provided
persona preference data to test stylistic alignment without sacrificing
usefulness or safety.

## Concepts

Study parameter-efficient fine-tuning, low-rank adapters, target modules,
adapter rank/alpha/dropout, conversational preference schemas, TRL data
collation, and style-versus-capability tradeoffs.

## Functions and pseudocode

### `format_for_trl(row)`

1. Read prompt, chosen, and rejected text.
2. Create a user message for the prompt.
3. Create assistant messages for chosen and rejected responses.
4. Return the conversational fields expected by TRL.
5. Validate a few formatted records before training.

### `main()` in `scripts/train_lora_dpo.py`

1. Load the local persona JSON.
2. Convert all rows into TRL conversational format.
3. Load the instruction model and tokenizer.
4. Choose LoRA rank, alpha, dropout, and target modules.
5. Explain those choices using model architecture and memory limits.
6. Configure TRL `DPOTrainer` and its training arguments.
7. Train and save the adapter/checkpoint.
8. Generate persona prompts and neutral usefulness/safety prompts.
9. Compare style improvement against usefulness and safety behavior.

## Checklist

- This task uses TRL/PEFT; Tasks 3 and 4 remain your own loops.
- Report all LoRA hyperparameters.
- Keep persona evaluation separate from general DPO evaluation.
- Include examples of stylistic improvement and any regression.
