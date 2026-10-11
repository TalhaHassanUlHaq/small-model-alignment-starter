"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations

import torch

from small_model_alignment.batching import build_lm_batch, build_preference_batch
from small_model_alignment.losses import (
    compute_dpo_loss_from_logps as _compute_dpo_loss_from_logps,
    response_logprobs,
    sft_loss,
)


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Task 3 — location: ``submission_adapter.py``.

    1. Import your internal SFT batch builder.
    2. Pass tokenizer, prompts, responses, and max length unchanged.
    3. Return its dictionary without changing keys or tensor shapes.
    4. Keep this grading wrapper thin; implement logic in the package.
    """
    return build_lm_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Task 3/4 — location: ``submission_adapter.py``.

    1. Build a response-masked batch with your internal batch builder.
    2. Call your internal response-scoring function.
    3. Return one tensor value per input row.
    4. Ensure prompt and padding tokens do not affect the score.
    """
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    return response_logprobs(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Task 3 — location: ``submission_adapter.py``.

    1. Build the response-masked batch.
    2. Call your internal SFT loss function.
    3. Return a scalar tensor retaining policy gradients.
    4. Use the same reduction documented in your report.
    """
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    return sft_loss(model, batch)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Task 4 — location: ``submission_adapter.py``.

    1. Import your internal preference-batch builder.
    2. Build aligned chosen and rejected response-masked batches.
    3. Return the two-batch dictionary without modifying its structure.
    """
    return build_preference_batch(
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        max_length,
    )


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Task 4 — location: ``submission_adapter.py``.

    1. Delegate the four tensors and beta to your tensor-level DPO function.
    2. Ensure the function computes the policy/reference gap correctly.
    3. Return the scalar mean loss.
    """
    return _compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Task 4 — location: ``submission_adapter.py``.

    1. Build chosen and rejected batches.
    2. Score both responses with the policy model.
    3. Score both responses with the reference model under no-gradient mode.
    4. Delegate all four scores to the tensor-level DPO objective.
    5. Return a scalar loss with gradients only through the policy path.
    """
    dpo_batch = build_preference_batch(
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        max_length,
    )
    policy_chosen_logps = response_logprobs(policy_model, dpo_batch["chosen"])
    policy_rejected_logps = response_logprobs(policy_model, dpo_batch["rejected"])

    with torch.no_grad():
        reference_chosen_logps = response_logprobs(reference_model, dpo_batch["chosen"])
        reference_rejected_logps = response_logprobs(reference_model, dpo_batch["rejected"])

    device = policy_chosen_logps.device
    reference_chosen_logps = reference_chosen_logps.to(device)
    reference_rejected_logps = reference_rejected_logps.to(device)

    return _compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
