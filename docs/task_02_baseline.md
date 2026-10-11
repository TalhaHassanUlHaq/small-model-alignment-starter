# Task 2 — Baseline generation guide

## Objective

Create a fixed AlpacaEval prompt subset and generate baseline answers from
`Qwen/Qwen2.5-0.5B`. Do not train a model in this task. This baseline becomes
the comparison point for later SFT and DPO experiments.

## Concepts

- **Evaluation control:** every later model must receive exactly the same
  prompts so differences are attributable to training rather than prompt
  selection.
- **Random seed:** seed `42` must select the same 300 prompts every time.
- **Greedy decoding:** use `do_sample=False` for deterministic next-token
  selection.
- **Prompt/completion separation:** decode only tokens generated after the
  input prompt.
- **Artifact reproducibility:** save the prompt subset and generation settings,
  not only the final answers.

## Functions to write

### `format_prompt(instruction)`

English pseudocode:

1. Strip surrounding whitespace from the instruction.
2. Add the user and assistant delimiters you choose.
3. Return the resulting string.
4. Record the exact template in the report.

### `select_instruction_subset(records, count, seed)`

English pseudocode:

1. Verify that `count` is positive and does not exceed the dataset size.
2. Create a local random generator using `seed`.
3. Sample `count` records without replacement.
4. Extract a non-empty instruction from every selected record.
5. Assign stable IDs such as `0`, `1`, … in selected order.
6. Return only the ID and instruction fields for the fixed subset.

### `load_alpaca_eval_subset(count, seed)`

English pseudocode:

1. Download or load the AlpacaEval JSON file.
2. Parse it into a list of records.
3. Call `select_instruction_subset`.
4. Return the selected records without generating text.

### `generate_one(model, tokenizer, prompt, generation_config)`

English pseudocode:

1. Tokenize one formatted prompt.
2. Move token tensors to the same device as the model.
3. Disable gradients because this is inference.
4. Call generation with `do_sample=False`, `max_new_tokens`, and any documented
   deterministic anti-repetition settings.
5. Remove the input-token prefix from the generated sequence.
6. Decode and return only the completion.

### `generate_baseline_outputs(...)`

English pseudocode:

1. Iterate through the fixed subset in order.
2. Format each instruction.
3. Call `generate_one`.
4. Create a record with `id`, `instruction`, `output`,
   `generator="qwen2.5-0.5b-base"`, and `dataset="alpaca_eval"`.
5. Return the list without reordering it.

### `main()` in `scripts/generate_baseline.py`

English pseudocode:

1. Parse command-line arguments.
2. Select and save the 300-prompt subset.
3. Load tokenizer and model.
4. Generate 2–5 prompts first when running a sanity-check option.
5. Inspect those outputs manually.
6. Generate the full subset.
7. Save outputs and settings containing model, seed, prompt template,
   decoding settings, and artifact paths.

## Required output

Each answer record must contain at least:

```text
id
instruction
output
generator
dataset
```

Suggested artifacts:

```text
outputs/eval_sets/alpaca_eval_300_seed42.json
outputs/generations/baseline_alpaca_eval_300_seed42.json
outputs/generations/baseline_alpaca_eval_300_seed42.settings.json
```

## Validation checklist

- Exactly 300 prompts are selected.
- Re-running subset selection with seed 42 produces the same IDs.
- Output order matches subset order.
- Greedy decoding is explicitly recorded.
- The decoded answer does not contain the original prompt prefix.
- The 2–5 prompt sanity check is inspected before the full run.
