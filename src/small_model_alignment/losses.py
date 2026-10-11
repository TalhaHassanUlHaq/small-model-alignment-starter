"""Skeleton for the mathematical core of Tasks 3 and 4.

Implement in this file:

1. Shift causal-LM logits and labels correctly.
2. Gather log probabilities only for unmasked response tokens.
3. Choose and document either summed-per-example or token-mean SFT loss.
4. Implement the DPO logistic objective using policy/reference log-probability
   differences.

Concepts to understand: teacher forcing, next-token shifting, log-softmax,
sequence log probability, reference-policy comparison, beta, and
``-log(sigmoid(...))``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def response_logprobs(model, batch):
    """Tasks 3/4 — location: ``src/small_model_alignment/losses.py``.

    1. Move batch tensors to the model's device.
    2. Run the causal LM and obtain logits.
    3. Shift logits left and labels right so position t predicts token t+1.
    4. Build a mask for labels not equal to ``-100``.
    5. Apply log-softmax over the vocabulary dimension.
    6. Gather the log probability of each target label.
    7. Replace masked contributions with zero.
    8. Sum valid response-token log probabilities per row.
    9. Return one sequence score per input row.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift so that position t predicts token t+1
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    valid_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)

    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    return (token_logps * valid_mask.float()).sum(dim=-1)


def sft_loss(model, batch):
    """Task 3 — location: ``src/small_model_alignment/losses.py``.

    1. Call ``response_logprobs`` for the batch.
    2. Negate the scores to obtain negative log likelihood.
    3. Apply per-example mean reduction.
    4. Return a scalar tensor that retains gradients for the policy model.
    """
    logps = response_logprobs(model, batch)
    return -logps.mean()


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta: float,
):
    """Task 4 — location: ``src/small_model_alignment/losses.py``.

    1. Compute policy chosen-minus-rejected log-probability gaps.
    2. Compute reference chosen-minus-rejected log-probability gaps.
    3. Subtract reference gaps from policy gaps.
    4. Multiply the result by positive beta.
    5. Apply negative log-sigmoid to each preference logit.
    6. Average the per-example losses and return one scalar tensor.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)
    return losses.mean()

