"""Skeleton interfaces for dataset preparation in Tasks 2–6.

This file intentionally contains no assignment implementation. Each function
has English pseudocode describing what the student should write.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random


@dataclass
class PreferenceExample:
    """Represent one prompt and its preferred/non-preferred responses."""

    prompt: str
    chosen: str
    rejected: str


def format_prompt(instruction: str) -> str:
    """Tasks 2/3/4 — location: ``src/small_model_alignment/data.py``.

    TODO steps: define and return one consistent prompt format.

    English pseudocode:
    1. Remove accidental leading and trailing whitespace from the instruction.
    2. Add the chosen user/assistant delimiters.
    3. Return the formatted prompt without generating an answer.
    4. Document this exact template in the final report.
    """

    instruction = instruction.strip()
    return f"Instruction:\n{instruction}\n\nAnswer:\n"


def select_instruction_subset(records, count: int = 300, seed: int = 42):
    """Task 2 — location: ``src/small_model_alignment/data.py``.

    TODO steps: create the fixed AlpacaEval evaluation subset.

    English pseudocode:
    1. Check that the requested count is positive and available.
    2. Create a local random-number generator using the supplied seed.
    3. Sample exactly ``count`` records without replacement.
    4. Extract and validate each instruction.
    5. Assign stable IDs in the selected order and return the records.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(records):
        raise ValueError(f"Requested {count} prompts but only {len(records)} are available")

    rng = random.Random(seed)
    selected = rng.sample(list(records), count)
    normalized = []
    for index, record in enumerate(selected):
        instruction = record.get("instruction", record.get("prompt"))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Record {index} has no non-empty instruction/prompt field")
        normalized.append({"id": str(index), "instruction": instruction.strip()})
    return normalized


def load_alpaca_eval_subset(count: int = 300, seed: int = 42, cache_dir=None):
    """Task 2 — location: ``src/small_model_alignment/data.py``.

    TODO steps: download AlpacaEval and call the selector above.

    English pseudocode:
    1. Download or load the AlpacaEval JSON artifact.
    2. Parse the JSON into a list of records.
    3. Pass the records, count, and seed to ``select_instruction_subset``.
    4. Return the selected records; do not generate model outputs here.
    """
    from huggingface_hub import hf_hub_download

    download_kwargs = {"repo_id": "tatsu-lab/alpaca_eval",
                       "filename": "alpaca_eval.json",
                       "repo_type": "dataset"}
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)
    source_path = hf_hub_download(**download_kwargs)
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected alpaca_eval.json to contain a list of records")
    return select_instruction_subset(payload, count=count, seed=seed)


def load_hf_split(dataset_name: str, split: str, *, select: int | None = None, seed: int = 42):
    """Tasks 3/4 — location: ``src/small_model_alignment/data.py``.

    1. Load the requested dataset and split using datasets.load_dataset.
    2. Shuffle with the provided seed only when a subset is requested.
    3. Select the requested number of rows.
    4. Return the dataset split.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    if select is not None:
        count = min(select, len(ds))
        ds = ds.shuffle(seed=seed).select(range(count))
    return ds


def extract_sft_pairs(dataset) -> list[tuple[str, str]]:
    """Extract single-turn (instruction, response) pairs from an instruction dataset."""
    pairs = []
    for row in dataset:
        instruction = ""
        response = ""
        if "messages" in row and isinstance(row["messages"], list):
            msgs = row["messages"]
            if len(msgs) >= 2:
                instruction = msgs[0].get("content", "")
                response = msgs[1].get("content", "")
        elif "instruction" in row and "response" in row:
            instruction = row["instruction"]
            response = row["response"]
        elif "prompt" in row and "completion" in row:
            instruction = row["prompt"]
            response = row["completion"]

        instruction = instruction.strip()
        response = response.strip()
        if instruction and response:
            pairs.append((instruction, response))
    return pairs


def extract_preference_pairs(dataset) -> list[PreferenceExample]:
    """Extract single-turn PreferenceExample from preference datasets."""
    examples = []
    for row in dataset:
        prompt = ""
        chosen = ""
        rejected = ""

        if "chosen" in row and "rejected" in row:
            raw_chosen = row["chosen"]
            raw_rejected = row["rejected"]

            # Handle Anthropic HH-RLHF format
            if isinstance(raw_chosen, str) and isinstance(raw_rejected, str):
                if "\n\nAssistant:" in raw_chosen and "\n\nAssistant:" in raw_rejected:
                    # Parse single-turn from "\n\nHuman: <prompt>\n\nAssistant: <response>"
                    c_parts = raw_chosen.split("\n\nAssistant:")
                    r_parts = raw_rejected.split("\n\nAssistant:")
                    prompt = c_parts[0].replace("\n\nHuman:", "").strip()
                    chosen = c_parts[1].strip() if len(c_parts) > 1 else ""
                    rejected = r_parts[1].strip() if len(r_parts) > 1 else ""
                else:
                    prompt = row.get("prompt", "").strip()
                    chosen = raw_chosen.strip()
                    rejected = raw_rejected.strip()

            # Handle HuggingFace conversational message lists
            elif isinstance(raw_chosen, list) and isinstance(raw_rejected, list):
                prompt = row.get("prompt", "")
                if not prompt and len(raw_chosen) > 0 and raw_chosen[0].get("role") == "user":
                    prompt = raw_chosen[0].get("content", "")
                chosen_msg = next((m.get("content", "") for m in raw_chosen if m.get("role") == "assistant"), "")
                rejected_msg = next((m.get("content", "") for m in raw_rejected if m.get("role") == "assistant"), "")
                chosen = chosen_msg.strip()
                rejected = rejected_msg.strip()

        if prompt and chosen and rejected:
            examples.append(PreferenceExample(prompt=prompt, chosen=chosen, rejected=rejected))
    return examples


def load_persona_preferences(path) -> list[PreferenceExample]:
    """Task 5 — location: ``src/small_model_alignment/data.py``.

    1. Read the JSON file.
    2. Check every row has prompt, chosen, and rejected fields.
    3. Preserve the original text and return normalized records.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Persona preferences file not found at: {path}")

    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("Expected persona preferences file to contain a list of objects")

    examples = []
    for idx, row in enumerate(raw_data):
        if not all(k in row for k in ("prompt", "chosen", "rejected")):
            raise ValueError(f"Row {idx} missing required fields (prompt, chosen, rejected)")
        examples.append(
            PreferenceExample(
                prompt=row["prompt"],
                chosen=row["chosen"],
                rejected=row["rejected"],
            )
        )
    return examples

