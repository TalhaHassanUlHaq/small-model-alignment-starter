# Task 7 — Kaggle model-based evaluation

## Goal

Use a supported Kaggle judge model to evaluate saved pairwise files reliably
without regenerating model outputs.

## Concepts

Study batch APIs, rate limits, retries, checkpointed jobs, resume behavior,
judge bias, invalid decisions, and win-rate denominators.

## Functions and pseudocode

### `build_judge_prompt(pair)`

1. Include the instruction and Response A/Response B text.
2. Ask for helpfulness, correctness, instruction following, formatting,
   hallucination avoidance, and safety.
3. Never include private model identity metadata.
4. Require a parseable decision format.

### `normalize_decision(value)`

1. Convert the raw response to a normalized string.
2. Recognize A, B, tie, and unknown.
3. Treat ambiguous or malformed answers as unknown.

### `judge_file(...)`

1. Read saved pairs.
2. Load existing judgments when resume is enabled.
3. Skip IDs already completed.
4. Submit at most five pairs per request.
5. Wait at least five seconds between attempts.
6. Retry failed batches up to the configured limit.
7. Save successful judgments after every batch.
8. Write unresolved failures to a separate JSONL file.

### `summarize_judgments(rows)`

1. Count A wins, B wins, ties, unknowns, and failures.
2. Map A/B decisions back to actual model identity using private metadata.
3. Compute win rate with and without ties as appropriate.
4. Report sample count, retries, invalid decisions, and failure rate.

## Checklist

- Judge reads saved files only.
- Judgments are resumable.
- Failures are not silently discarded.
- Model identities are hidden until after judging.
- Report the judge model and all invalid/failure handling.
