"""Task 1: Kaggle environment and accelerator verification.

Reports Python, PyTorch, Transformers, Datasets, and Accelerate versions,
verifies CUDA and the two T4 GPUs, and saves the environment report.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

try:
    import torch
    torch_version = torch.__version__
except ImportError:
    torch = None
    torch_version = "not installed"

try:
    import transformers
    transformers_version = transformers.__version__
except ImportError:
    transformers = None
    transformers_version = "not installed"

try:
    import datasets
    datasets_version = datasets.__version__
except ImportError:
    datasets = None
    datasets_version = "not installed"

try:
    import accelerate
    accelerate_version = accelerate.__version__
except ImportError:
    accelerate_version = "not installed"

from small_model_alignment.training import save_json


def main() -> None:
    """Task 1 — location: ``scripts/setup_check.py``."""
    cuda_available = torch is not None and torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "datasets_version": datasets_version,
        "accelerate_version": accelerate_version,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpus": [],
    }

    print("=" * 60)
    print("Environment Verification Report (Task 1)")
    print("=" * 60)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch_version}")
    print(f"Transformers: {transformers_version}")
    print(f"Datasets:     {datasets_version}")
    print(f"Accelerate:   {accelerate_version}")
    print(f"CUDA status:  {'Available' if cuda_available else 'Unavailable'}")
    print(f"Device count: {gpu_count}")

    if cuda_available:
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            mem_gib = props.total_memory / (1024**3)
            gpu_info = {
                "index": i,
                "name": props.name,
                "total_memory_gib": round(mem_gib, 2),
                "multi_processor_count": props.multi_processor_count,
            }
            info["gpus"].append(gpu_info)
            print(f"  GPU {i}: {props.name} ({gpu_info['total_memory_gib']:.1f} GiB)")

        if gpu_count != 2 or not any("T4" in g["name"] for g in info["gpus"]):
            print("Notice: Environment does not match the reference two Tesla T4 GPUs.")
    else:
        print("Warning: Running on CPU. CUDA accelerator recommended for full training.")

    output_path = Path("outputs/setup_environment.json")
    save_json(output_path, info)
    print(f"\nSaved environment report to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
