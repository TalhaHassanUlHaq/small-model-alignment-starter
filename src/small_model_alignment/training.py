"""Training loops and optimization utilities for SFT and DPO."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from small_model_alignment.losses import (
    compute_dpo_loss_from_logps,
    response_logprobs,
    sft_loss,
)


class InstructionDataset(Dataset):
    """Dataset for Supervised Fine-Tuning instruction-response pairs."""

    def __init__(self, pairs: list[tuple[str, str]], prompt_formatter):
        self.pairs = pairs
        self.prompt_formatter = prompt_formatter

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        instruction, response = self.pairs[idx]
        return self.prompt_formatter(instruction), response


class PreferenceDataset(Dataset):
    """Dataset for Direct Preference Optimization preference pairs."""

    def __init__(self, examples, prompt_formatter):
        self.examples = examples
        self.prompt_formatter = prompt_formatter

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        ex = self.examples[idx]
        return self.prompt_formatter(ex.prompt), ex.chosen, ex.rejected


def set_seed(seed: int) -> None:
    """Tasks 1/3/4 — location: ``src/small_model_alignment/training.py``.

    Seeds Python's random generator, NumPy, CPU torch, and all CUDA devices.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_sft_step(
    model,
    optimizer,
    batch,
    scaler=None,
    accumulation_steps: int = 1,
    is_accumulation_boundary: bool = True,
    max_grad_norm: float = 1.0,
) -> float:
    """Task 3 SFT optimization step with fp16 autocast, scaling, and clipping."""
    model.train()
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"

    with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
        loss = sft_loss(model, batch)
        scaled_loss = loss / accumulation_steps

    if scaler is not None and use_cuda:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    if is_accumulation_boundary:
        if scaler is not None and use_cuda:
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()

    return loss.item()


def validate_sft(model, dataloader) -> float:
    """Evaluate response-only SFT loss on validation dataset."""
    model.eval()
    total_loss = 0.0
    count = 0

    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for batch in dataloader:
            with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                loss = sft_loss(model, batch)
            total_loss += loss.item()
            count += 1

    model.train()
    return total_loss / max(1, count)


def dpo_batch_step(
    policy_model,
    reference_model,
    optimizer,
    batch,
    scaler=None,
    beta: float = 0.1,
    accumulation_steps: int = 1,
    is_accumulation_boundary: bool = True,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Task 4 DPO optimization step scoring policy and frozen reference."""
    policy_model.train()
    policy_device = next(policy_model.parameters()).device
    use_cuda = policy_device.type == "cuda"

    # Forward through policy
    with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
        pi_w = response_logprobs(policy_model, batch["chosen"])
        pi_l = response_logprobs(policy_model, batch["rejected"])

    # Forward through frozen reference model under no_grad
    with torch.no_grad():
        ref_device = next(reference_model.parameters()).device
        ref_cuda = ref_device.type == "cuda"
        with torch.amp.autocast(device_type="cuda" if ref_cuda else "cpu", enabled=ref_cuda):
            ref_w = response_logprobs(reference_model, batch["chosen"]).to(policy_device)
            ref_l = response_logprobs(reference_model, batch["rejected"]).to(policy_device)

    with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
        loss = compute_dpo_loss_from_logps(pi_w, pi_l, ref_w, ref_l, beta)
        scaled_loss = loss / accumulation_steps

    if scaler is not None and use_cuda:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    if is_accumulation_boundary:
        if scaler is not None and use_cuda:
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()

    # Diagnostics: implicit rewards
    with torch.no_grad():
        rw = beta * (pi_w - ref_w)
        rl = beta * (pi_l - ref_l)
        reward_acc = (rw > rl).float().mean().item()
        reward_margin = (rw - rl).mean().item()

    return {
        "loss": loss.item(),
        "reward_accuracy": reward_acc,
        "reward_margin": reward_margin,
    }


def validate_dpo(
    policy_model,
    reference_model,
    dataloader,
    beta: float = 0.1,
) -> dict[str, float]:
    """Compute validation DPO loss, reward accuracy, and reward margin."""
    policy_model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_margin = 0.0
    count = 0

    policy_device = next(policy_model.parameters()).device
    use_cuda = policy_device.type == "cuda"

    with torch.no_grad():
        for batch in dataloader:
            with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                pi_w = response_logprobs(policy_model, batch["chosen"])
                pi_l = response_logprobs(policy_model, batch["rejected"])

            ref_w = response_logprobs(reference_model, batch["chosen"]).to(policy_device)
            ref_l = response_logprobs(reference_model, batch["rejected"]).to(policy_device)

            loss = compute_dpo_loss_from_logps(pi_w, pi_l, ref_w, ref_l, beta)
            rw = beta * (pi_w - ref_w)
            rl = beta * (pi_l - ref_l)
            reward_acc = (rw > rl).float().mean().item()
            reward_margin = (rw - rl).mean().item()

            total_loss += loss.item()
            total_acc += reward_acc
            total_margin += reward_margin
            count += 1

    policy_model.train()
    denom = max(1, count)
    return {
        "val_loss": total_loss / denom,
        "val_reward_accuracy": total_acc / denom,
        "val_reward_margin": total_margin / denom,
    }


def save_checkpoint(model, tokenizer, path: str | Path) -> None:
    """Save model checkpoint and tokenizer."""
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)


def save_json(path: str | Path, payload: Any) -> None:
    """Serialize payload to UTF-8 JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
