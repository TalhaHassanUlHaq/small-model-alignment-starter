# Task 1 — Repository and Kaggle setup

## Goal

Prepare a reproducible GPU environment without replacing Kaggle's CUDA-ready
PyTorch installation.

## Concepts

Understand package isolation, CUDA device visibility, GPU memory, version
logging, random seeds, and why environment metadata belongs in the experiment
artifacts.

## Functions/script pseudocode

### `main()` in `scripts/setup_check.py`

1. Print Python, PyTorch, Transformers, Datasets, and Accelerate versions.
2. Check whether CUDA is available.
3. Count visible GPUs and print each device name and memory.
4. Confirm the expected two T4 devices or clearly record the actual runtime.
5. Create the output directory.
6. Save the collected environment information as JSON.

## Checklist

- Clone the repository in Kaggle.
- Install dependencies without reinstalling PyTorch.
- Confirm two T4 GPUs.
- Save the environment report before training.
- Record any deviation from the recommended runtime.
