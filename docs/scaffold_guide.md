# Scaffold guide

This document explains the purpose of every scaffold file. No assignment task
is implemented. The Python files contain signatures, TODOs, and English
pseudocode; the task guides contain the detailed implementation plan.

## Core package

- `src/small_model_alignment/__init__.py`: package marker and explanation of
  the intended reusable modules. No assignment logic is implemented.
- `batching.py`: skeleton for prompt-plus-response tokenization, EOS handling,
  padding, attention masks, and `-100` response-only labels. The main concept
  is causal language-model masking: prompt tokens provide context but are not
  prediction targets.
- `losses.py`: skeleton for next-token shifting, response log probabilities,
  response-only SFT loss, and the DPO logistic objective. The reference model
  must eventually be scored without gradients.
- `data.py`: skeleton for prompt formatting, dataset loading, preference-row
  validation, and reproducible subsets. The key concept is keeping train,
  validation, DPO, and generation formatting consistent.
- `config.py`: records model, SFT, DPO, generation, and output-path defaults.
  Every value that can affect reproducibility should be copied into the final
  run configuration.
- `training.py`: skeleton for mixed precision, gradient clipping, SFT/DPO
  steps, gradient accumulation, validation, and reward diagnostics.
- `generation.py`: skeleton for comparable JSONL generations and blind
  DPO-vs-SFT pair files. Model identity must not reach the judge prompt.
- `evaluation.py`: skeleton for judge parsing and counts/rates, including the
  required distinction between wins, losses, ties, unknowns, and failures.

## Assignment entry points

- `scripts/setup_check.py`: Task 1 skeleton for CUDA and package checks.
- `scripts/generate_baseline.py`: Task 2 skeleton for the fixed AlpacaEval
  subset and baseline generation.
- `scripts/train_sft.py`: Task 3 skeleton for 30k-example SFT data loading,
  DataLoaders, accumulation, fp16 GradScaler, validation, and best/final saves.
- `scripts/train_dpo.py`: Task 4 skeleton for the same from-scratch loop on
  15k HH-RLHF and 15k UltraFeedback pairs, each with a frozen SFT reference,
  validation reward accuracy, and reward-margin logging.
- `scripts/train_lora_dpo.py`: Task 5 skeleton to format persona records for
  TRL, choose and justify PEFT/LoRA settings, and evaluate style against
  usefulness and safety.
- `scripts/generate_outputs.py`: Task 6. Sanity-check 2–5 prompts, then use
  identical decoding settings for SFT and DPO on the fixed evaluation sets.
- `scripts/judge_pairwise_kaggle.py`: Task 7 skeleton for the Kaggle model-proxy call,
  five-example batches, delays, retries, checkpointing, resume behavior, and
  a separate failure file.
- `scripts/plot_metrics.py`: Task 8 skeleton for report-ready labeled plots.

Detailed task guides:

- `task_01_setup.md`
- `task_02_baseline.md`
- `task_03_sft.md`
- `task_04_dpo.md`
- `task_05_lora_dpo.md`
- `task_06_generation.md`
- `task_07_judging.md`
- `task_08_report.md`
- `function_index.md` — exact file/function locations and task ownership.

## Reproducibility and reporting

- `configs/default.yaml`: starting values based on the assignment guide. Edit
  it whenever the actual run deviates and save the resolved configuration.
- `reports/final_report_outline.md`: an eight-page report structure covering
  data, configuration, deviations, training curves, comparisons, examples,
  limitations, and reproducibility paths.
- `submission_adapter.py`: the fixed grader interface. It contains stubs only;
  connect it after implementing the core package, without changing its six
  signatures.
- `pyproject.toml`: exposes the `src/` package for editable installation.
- `README.md`: repository-level setup, test instructions, and a short map of
  the scaffold.
