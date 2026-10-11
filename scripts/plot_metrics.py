"""Task 8: Plot training metrics for the final report.

Generates high-resolution plots for SFT loss curves and DPO loss, reward
accuracy, and reward margin curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", default="outputs/logs")
    parser.add_argument("--output_dir", default="reports/figures")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def plot_sft(records: list[dict], out_dir: Path) -> None:
    if not records:
        print("No SFT records found.")
        return

    train_steps = [r["step"] for r in records if "train_loss" in r]
    train_losses = [r["train_loss"] for r in records if "train_loss" in r]

    val_steps = [r["step"] for r in records if "val_loss" in r]
    val_losses = [r["val_loss"] for r in records if "val_loss" in r]

    plt.figure(figsize=(8, 5), dpi=300)
    if train_steps:
        plt.plot(train_steps, train_losses, label="Train Loss", color="#1f77b4", alpha=0.8)
    if val_steps:
        plt.plot(val_steps, val_losses, label="Validation Loss", color="#ff7f0e", marker="o", linewidth=2)

    plt.title("Supervised Fine-Tuning (SFT) Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Optimizer Steps", fontsize=12)
    plt.ylabel("Loss (Negative Log-Likelihood)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_path = out_dir / "sft_loss_curve.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_dpo(records: list[dict], track_name: str, out_dir: Path) -> None:
    if not records:
        print(f"No DPO records found for {track_name}.")
        return

    steps = [r["step"] for r in records if "loss" in r]
    losses = [r["loss"] for r in records if "loss" in r]
    accs = [r["reward_accuracy"] for r in records if "reward_accuracy" in r]
    margins = [r["reward_margin"] for r in records if "reward_margin" in r]

    val_steps = [r["step"] for r in records if "val_loss" in r]
    val_losses = [r["val_loss"] for r in records if "val_loss" in r]
    val_accs = [r["val_reward_accuracy"] for r in records if "val_reward_accuracy" in r]
    val_margins = [r["val_reward_margin"] for r in records if "val_reward_margin" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Loss plot
    if steps:
        ax1.plot(steps, losses, label="Train DPO Loss", color="#2ca02c", alpha=0.7)
    if val_steps:
        ax1.plot(val_steps, val_losses, label="Val DPO Loss", color="#d62728", marker="o", linewidth=2)
    ax1.set_title(f"{track_name.upper()} DPO Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Optimizer Steps", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Reward Diagnostics Plot
    if steps:
        ax2.plot(steps, accs, label="Train Reward Acc", color="#1f77b4", alpha=0.7)
        ax2.plot(steps, margins, label="Train Reward Margin", color="#9467bd", alpha=0.7, linestyle=":")
    if val_steps:
        ax2.plot(val_steps, val_accs, label="Val Reward Acc", color="#ff7f0e", marker="s", linewidth=2)
        ax2.plot(val_margins, val_margins, label="Val Reward Margin", color="#8c564b", marker="^", linewidth=2)
    ax2.set_title(f"{track_name.upper()} Reward Diagnostics", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Optimizer Steps", fontsize=11)
    ax2.set_ylabel("Metric Value", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    out_path = out_dir / f"dpo_{track_name}_curves.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_records = load_jsonl(log_dir / "sft_metrics.jsonl")
    plot_sft(sft_records, out_dir)

    hh_records = load_jsonl(log_dir / "dpo_hh_rlhf_metrics.jsonl")
    plot_dpo(hh_records, "hh_rlhf", out_dir)

    uf_records = load_jsonl(log_dir / "dpo_ultrafeedback_metrics.jsonl")
    plot_dpo(uf_records, "ultrafeedback", out_dir)

    print("Plotting complete!")


if __name__ == "__main__":
    main()
