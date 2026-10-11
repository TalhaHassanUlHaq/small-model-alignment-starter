# Small-Model Alignment Report (8-page maximum)

This outline is part of the scaffold. Replace brackets with measured results;
do not invent numbers before the runs complete.

## 1. Repository, models, and data

- Repository: `[URL]`
- Base checkpoints: `[Qwen checkpoint]`, `[SmolLM checkpoint]`
- Dataset names, splits, counts, filters, seed, and prompt template: `[fill]`

## 2. Experimental configuration

Include one compact table for SFT, HH-RLHF DPO, UltraFeedback DPO, and LoRA-DPO:
optimizer, learning rate, scheduler, warmup, clipping, batch/accumulation,
effective batch, precision, sequence length, epochs, seed, DPO beta, reference
checkpoint, and generation settings.

## 3. Deviations and parameter choices

For each change from the guide, record guide value, final value, reason, and
evidence from memory, validation behavior, or a pilot run.

## 4. Training behavior

Insert labeled SFT loss and DPO loss/reward-accuracy/reward-margin plots.
Explain convergence, overfitting, instability, and checkpoint selection.

## 5. DPO versus SFT

Report sample count, wins, ties, losses, valid/unknown judgments, retries,
judge model, and win rates for each evaluation track. Keep persona results
separate from general helpfulness/safety results.

## 6. Interpretation, examples, and limitations

Distinguish measured observations from explanations. Include improvement and
regression examples, then discuss judge bias, sample size, seed variance, and
compute limits.
