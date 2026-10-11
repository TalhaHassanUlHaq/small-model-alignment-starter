"""Skeleton for Task 3 and Task 4 batch construction.

Implement in this file:

1. Tokenize ``prompt + response + EOS`` for causal language modeling.
2. Pad rows to a common length and create attention masks.
3. Set prompt and padding labels to ``-100`` so only response tokens affect
   the loss.
4. Reuse the same logic to create chosen and rejected DPO batches.

Concepts to understand: causal next-token prediction, attention masks,
response-only loss masking, EOS tokens, padding, and truncation policy.
"""

from __future__ import annotations

import torch


def build_lm_batch(tokenizer, prompts, responses, max_length):
    """Task 3/4 — location: ``src/small_model_alignment/batching.py``.

    1. Verify that prompt and response lists have equal lengths.
    2. Tokenize prompts and responses without accidentally adding duplicate
       special tokens.
    3. Append the tokenizer EOS token to every response.
    4. Concatenate each prompt with its response and apply sequence truncation.
    5. Pad all rows to a common width no greater than ``max_length``.
    6. Create attention masks: one for real tokens and zero for padding.
    7. Create labels filled with ``-100`` for prompt and padding.
    8. Copy only response-token IDs, including EOS, into the label positions.
    9. Return a dictionary of tensors: ``input_ids``, ``attention_mask``, ``labels``.
    """
    if len(prompts) != len(responses):
        raise ValueError("Prompts and responses must have the same length")

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    all_input_ids = []
    all_labels = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        if eos_token_id is not None:
            response_ids = response_ids + [eos_token_id]

        full_ids = prompt_ids + response_ids
        full_labels = [-100] * len(prompt_ids) + response_ids

        # Truncate to max_length
        full_ids = full_ids[:max_length]
        full_labels = full_labels[:max_length]

        all_input_ids.append(full_ids)
        all_labels.append(full_labels)

    max_len = max((len(seq) for seq in all_input_ids), default=0)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for seq_ids, seq_labels in zip(all_input_ids, all_labels):
        pad_len = max_len - len(seq_ids)
        padded_input_ids.append(seq_ids + [pad_token_id] * pad_len)
        padded_attention_mask.append([1] * len(seq_ids) + [0] * pad_len)
        padded_labels.append(seq_labels + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def build_preference_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Task 4 — location: ``src/small_model_alignment/batching.py``.

    1. Verify that all three input lists have equal length.
    2. Call ``build_lm_batch`` for prompts plus chosen responses.
    3. Call ``build_lm_batch`` for prompts plus rejected responses.
    4. Preserve row alignment between chosen and rejected batches.
    5. Return a dictionary with exactly ``chosen`` and ``rejected`` keys.
    """
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("Prompts, chosen_responses, and rejected_responses must have the same length")

    chosen_batch = build_lm_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected_batch = build_lm_batch(tokenizer, prompts, rejected_responses, max_length)

    return {
        "chosen": chosen_batch,
        "rejected": rejected_batch,
    }

