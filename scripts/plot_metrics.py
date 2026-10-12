"""Task 8: Plot publication-quality training metrics for the final report.

Generates high-resolution, readable plots for:
1. SFT loss curves (with EMA smoothing and validation checkpoint markers).
2. HH-RLHF DPO loss and reward diagnostics (accuracy & margin).
3. UltraFeedback DPO loss and reward diagnostics (accuracy & margin).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def compute_ema(values: list[float], alpha: float = 0.05) -> list[float]:
    """Compute exponential moving average."""
    if not values:
        return []
    ema = []
    curr = values[0]
    for v in values:
        curr = alpha * v + (1.0 - alpha) * curr
        ema.append(curr)
    return ema


def plot_sft(records: list[dict], out_dir: Path) -> None:
    if not records:
        print("No SFT records found.")
        return

    train_steps = [r["step"] for r in records if "train_loss" in r]
    train_losses = [r["train_loss"] for r in records if "train_loss" in r]

    val_steps = [r["step"] for r in records if "val_loss" in r]
    val_losses = [r["val_loss"] for r in records if "val_loss" in r]

    plt.figure(figsize=(9, 5.2), dpi=300)

    if train_steps:
        # Faint raw training loss
        plt.plot(train_steps, train_losses, label="Train Loss (raw)", color="#1f77b4", alpha=0.18, linewidth=0.8)
        # Smoothed EMA training loss
        smoothed = compute_ema(train_losses, alpha=0.04)
        plt.plot(train_steps, smoothed, label="Train Loss (EMA, α=0.04)", color="#1f77b4", linewidth=2.2)

    if val_steps:
        plt.plot(val_steps, val_losses, label="Validation Loss", color="#ff7f0e", marker="o", markersize=6, linewidth=2.4)
        # Annotate final/best checkpoint
        best_idx = int(np.argmin(val_losses))
        best_step = val_steps[best_idx]
        best_val = val_losses[best_idx]
        plt.annotate(
            f"Best Checkpoint (Step {best_step})\nVal Loss = {best_val:.2f}",
            xy=(best_step, best_val),
            xytext=(best_step - 250, best_val + 120),
            arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=7),
            fontsize=9.5,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff2f2", edgecolor="#d62728", alpha=0.9),
        )

    plt.title("Supervised Fine-Tuning (SFT) Convergence Curve", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Optimizer Steps", fontsize=11)
    plt.ylabel("Sequence Negative Log-Likelihood (Summed)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper right")
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    display_title = "Anthropic HH-RLHF" if "hh" in track_name.lower() else "UltraFeedback"

    # --- Subplot 1: DPO Loss ---
    if steps:
        ax1.plot(steps, losses, label="Train Loss (raw)", color="#2ca02c", alpha=0.18, linewidth=0.8)
        smoothed_losses = compute_ema(losses, alpha=0.05)
        ax1.plot(steps, smoothed_losses, label="Train Loss (EMA, α=0.05)", color="#2ca02c", linewidth=2.0)
    if val_steps:
        ax1.plot(val_steps, val_losses, label="Val DPO Loss", color="#d62728", marker="o", markersize=6, linewidth=2.4)
        best_val_idx = int(np.argmin(val_losses))
        best_step = val_steps[best_val_idx]
        best_loss = val_losses[best_val_idx]
        ax1.annotate(
            f"Best: Step {best_step} ({best_loss:.4f})",
            xy=(best_step, best_loss),
            xytext=(best_step - 160, best_loss + 0.15),
            arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.2, headwidth=6),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff2f2", edgecolor="#d62728", alpha=0.9),
        )

    ax1.set_title(f"{display_title} DPO Loss Dynamics", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Optimizer Steps", fontsize=11)
    ax1.set_ylabel("DPO Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9.5, loc="upper right")

    # --- Subplot 2: Reward Diagnostics (Accuracy & Margin) ---
    if steps:
        # Smoothed train accuracy and margin
        smoothed_accs = compute_ema(accs, alpha=0.05)
        smoothed_margins = compute_ema(margins, alpha=0.05)
        ax2.plot(steps, smoothed_accs, label="Train Reward Acc (EMA)", color="#1f77b4", linewidth=1.8)
        ax2.plot(steps, smoothed_margins, label="Train Reward Margin (EMA)", color="#9467bd", linewidth=1.8, linestyle="--")

    if val_steps:
        ax2.plot(val_steps, val_accs, label="Val Reward Accuracy", color="#ff7f0e", marker="s", markersize=6, linewidth=2.2)
        # CRUCIAL BUG FIX: plot against val_steps, NOT val_margins
        ax2.plot(val_steps, val_margins, label="Val Reward Margin", color="#8c564b", marker="^", markersize=6, linewidth=2.2)

    # Reference lines
    ax2.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.axhline(0.5, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)

    ax2.set_title(f"{display_title} Reward Diagnostics", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Optimizer Steps", fontsize=11)
    ax2.set_ylabel("Metric Value", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9.5, loc="best")

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
