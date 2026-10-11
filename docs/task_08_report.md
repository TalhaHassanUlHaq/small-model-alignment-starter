# Task 8 — Final report

## Goal

Produce a concise report of no more than eight pages, including plots, tables,
references, and enough information to reproduce the comparisons.

## Concepts

Study scientific reporting, ablations/deviations, metric interpretation,
training diagnostics, controlled comparisons, error analysis, limitations,
and reproducibility.

## Report-writing pseudocode

1. Record repository, model checkpoints, datasets, counts, seeds, filters, and
   prompt templates.
2. Create one configuration table covering SFT, both DPO runs, and LoRA-DPO.
3. For every deviation, write guide value, final value, reason, and evidence.
4. Plot SFT loss and DPO loss/reward diagnostics with labeled axes.
5. Report DPO-versus-SFT wins, ties, losses, failures, judge model, and rates
   separately for each evaluation set.
6. Include representative improvements and regressions.
7. Separate measured observations from interpretations.
8. Discuss judge bias, small sample size, seed variance, and compute limits.
9. List paths to configurations, checkpoints, metrics, generations, and
   judgments.
10. Check that the final document is at most eight pages.

## Checklist

- No result is reported without an artifact or metric source.
- HH-RLHF, UltraFeedback, and persona results are not merged improperly.
- The actual settings used—not only guide settings—are reported.
- Figures are readable and referenced in the text.
