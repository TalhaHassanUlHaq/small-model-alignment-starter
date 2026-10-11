"""Additional edge case tests for submission adapter and batching/losses.

Tests truncation boundaries, mixed sequence lengths, row ordering,
gradient preservation, and numerical correctness.
"""

from __future__ import annotations

import torch
import pytest

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_row_ordering_preserved():
    tokenizer = ToyTokenizer()
    prompts = ["first", "second", "third", "fourth"]
    responses = ["ans1", "ans2", "ans3", "ans4"]
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=32)

    for i, (p, r) in enumerate(zip(prompts, responses)):
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        r_ids = tokenizer.encode(r, add_special_tokens=False) + [tokenizer.eos_token_id]
        expected_ids = p_ids + r_ids
        actual_ids = batch["input_ids"][i, :len(expected_ids)].tolist()
        assert actual_ids == expected_ids, f"Row {i} order mismatch"


def test_truncation_boundary_removes_eos_when_exceeding_max_length():
    tokenizer = ToyTokenizer()
    prompt = "This is a prompt"  # 16 chars
    response = "and a very long response that will get truncated"
    p_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    max_len = p_len + 4  # Only allow 4 response tokens
    batch = adapter.build_sft_batch(tokenizer, [prompt], [response], max_length=max_len)

    assert batch["input_ids"].shape[1] == max_len
    assert batch["labels"][0, :p_len].eq(-100).all()
    # unmasked labels should be exactly 4 tokens long
    unmasked = batch["labels"][0].ne(-100)
    assert unmasked.sum().item() == 4
    # EOS should not be present because it was truncated
    assert batch["input_ids"][0, -1].item() != tokenizer.eos_token_id


def test_prompt_exceeding_max_length_masks_all_tokens():
    tokenizer = ToyTokenizer()
    prompt = "A very long prompt that exceeds the maximum length"
    response = "short"
    batch = adapter.build_sft_batch(tokenizer, [prompt], [response], max_length=5)

    assert batch["input_ids"].shape[1] == 5
    # All 5 tokens should be prompt tokens, hence masked to -100
    assert batch["labels"][0].eq(-100).all()


def test_dpo_loss_symmetry_and_values():
    # If policy prefers chosen by +1, and ref is neutral (0), loss < log(2)
    p_w = torch.tensor([1.0])
    p_l = torch.tensor([0.0])
    r_w = torch.tensor([0.0])
    r_l = torch.tensor([0.0])
    loss_better = adapter.compute_dpo_loss_from_logps(p_w, p_l, r_w, r_l, beta=0.1)

    # If policy prefers rejected by +1, loss > log(2)
    loss_worse = adapter.compute_dpo_loss_from_logps(p_l, p_w, r_w, r_l, beta=0.1)

    log_2 = torch.log(torch.tensor(2.0))
    assert loss_better < log_2
    assert loss_worse > log_2


def test_dpo_loss_gradient_flow():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)

    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    initial_bias = policy.logit_bias.detach().clone()

    loss = adapter.compute_dpo_loss(
        policy,
        reference,
        tokenizer,
        ["Q1: "],
        ["chosen answer"],
        ["rejected answer"],
        beta=0.1,
        max_length=32,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Policy weights updated
    assert not torch.equal(policy.logit_bias, initial_bias)
    # Reference weights unchanged
    assert reference.logit_bias.grad is None
