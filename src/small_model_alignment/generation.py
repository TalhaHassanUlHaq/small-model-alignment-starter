"""Skeleton interfaces for generation in Tasks 2 and 6.

No generation logic is implemented here. The docstrings provide English
pseudocode for the functions you must write.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def generate_one(model, tokenizer, formatted_prompt: str, generation_config: dict):
    """Tasks 2/6 — location: ``src/small_model_alignment/generation.py``.

    TODO steps: generate one answer.

    English pseudocode:
    1. Tokenize the formatted prompt and move tensors to the model device.
    2. Disable gradient tracking and call the model's generation method.
    3. Pass the agreed decoding settings, including maximum new tokens.
    4. Remove the original prompt tokens from the returned sequence.
    5. Decode only newly generated tokens and return the cleaned text.
    """
    # Tokenize the formatted prompt and move tensors to the model device.
    encoded = tokenizer(formatted_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    # Disable gradient tracking and call the model's generation method.
    with torch.no_grad():
        output = model.generate(**encoded, **generation_config)

    # Remove the original prompt tokens from the returned sequence.
    prompt_width = encoded["input_ids"].shape[1]
    generated_tokens = output[0][prompt_width:]

    # Decode only newly generated tokens and return the cleaned text.
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def generate_baseline_outputs(model, tokenizer, subset, prompt_formatter, generation_config):
    """Task 2 — location: ``src/small_model_alignment/generation.py``.

    TODO steps: generate baseline outputs for the fixed subset.

    English pseudocode:
    1. Loop over the saved evaluation subset in its fixed order.
    2. Format each instruction with the shared prompt formatter.
    3. Call ``generate_one``.
    4. Store ID, instruction, output, generator name, and dataset name.
    5. Return all records without changing their order.
    """
    outputs = []
    for record in subset:
        instruction = record["instruction"]
        formatted = prompt_formatter(instruction)
        response = generate_one(model, tokenizer, formatted, generation_config)
        outputs.append({
            "id": record["id"],
            "instruction": instruction,
            "output": response,
            "generator": "qwen2.5-0.5b-base",
            "dataset": "alpaca_eval",
        })
    return outputs


def save_json(path, records):
    """Tasks 2/6 — location: ``src/small_model_alignment/generation.py``.

    TODO steps:
    1. Validate the output extension and parent directory.
    2. Create missing directories.
    3. Serialize records in readable, deterministic JSON.
    4. Write with UTF-8 encoding and report the artifact path.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_generations(
    model,
    tokenizer,
    prompt_records: list[dict],
    prompt_formatter,
    generation_config: dict,
    generator_label: str,
    dataset_name: str,
    output_path: str | Path | None = None,
) -> list[dict]:
    """Task 6 — location: ``src/small_model_alignment/generation.py``.

    1. Loop over prompt records in fixed order.
    2. Format prompt and generate output.
    3. Store ID, instruction, output, generator name, and dataset name.
    4. Save to JSON if output_path is provided.
    """
    outputs = []
    for record in prompt_records:
        instruction = record["instruction"]
        formatted = prompt_formatter(instruction)
        response = generate_one(model, tokenizer, formatted, generation_config)
        outputs.append({
            "id": record["id"],
            "instruction": instruction,
            "output": response,
            "generator": generator_label,
            "dataset": dataset_name,
        })
    if output_path is not None:
        save_json(output_path, outputs)
    return outputs


def build_blind_pairs(
    records_1: list[dict],
    records_2: list[dict],
    output_path: str | Path,
    seed: int = 42,
) -> list[dict]:
    """Task 6 — location: ``src/small_model_alignment/generation.py``.

    1. Verify matching IDs and instructions.
    2. Randomize which output becomes A using a recorded seed.
    3. Store model identity in private metadata (__model_a, __model_b).
    4. Write pairwise JSONL file.
    """
    if len(records_1) != len(records_2):
        raise ValueError(f"Record count mismatch: {len(records_1)} vs {len(records_2)}")

    import random
    rng = random.Random(seed)

    pairs = []
    for r1, r2 in zip(records_1, records_2):
        if r1["id"] != r2["id"]:
            raise ValueError(f"ID mismatch between records: {r1['id']} vs {r2['id']}")

        # Randomize position
        if rng.random() < 0.5:
            resp_a, resp_b = r1["output"], r2["output"]
            mod_a, mod_b = r1["generator"], r2["generator"]
        else:
            resp_a, resp_b = r2["output"], r1["output"]
            mod_a, mod_b = r2["generator"], r1["generator"]

        pairs.append({
            "id": r1["id"],
            "instruction": r1["instruction"],
            "response_a": resp_a,
            "response_b": resp_b,
            "__model_a": mod_a,
            "__model_b": mod_b,
        })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return pairs

