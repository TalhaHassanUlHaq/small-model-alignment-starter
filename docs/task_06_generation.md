# Task 6 — Evaluation generations and blind pairs

## Goal

Generate comparable outputs from SFT and each relevant DPO checkpoint using
the same prompts and decoding settings.

## Concepts

Study controlled experiments, decoding confounds, JSONL artifacts, prompt
identity, evaluator blinding, randomized response order, and metadata leakage.

## Functions and pseudocode

### `write_generations(...)`

1. Load the fixed evaluation prompt file.
2. Load one checkpoint and its tokenizer.
3. Use the exact shared decoding configuration.
4. Generate a small sanity sample and inspect it.
5. Generate all prompts in stable order.
6. Save ID, instruction, output, model label, dataset, and settings.

### `build_blind_pairs(...)`

1. Load the SFT and DPO output files.
2. Verify identical IDs and prompt order.
3. Randomly choose which model becomes Response A using a recorded seed.
4. Write only Response A, Response B, ID, and instruction to the judge-facing
   fields.
5. Store model identity in private metadata that is not sent to the judge.

### `main()` in `scripts/generate_outputs.py`

1. Run the sanity check for 2–5 prompts.
2. Generate SFT outputs.
3. Generate the relevant DPO outputs.
4. Build blinded pairwise files.
5. Save decoding settings and artifact paths.

## Checklist

- DPO and SFT use the same prompts.
- DPO and SFT use the same decoding settings.
- Judge prompts never reveal checkpoint or model identity.
- Pair mapping remains recoverable after judging.
