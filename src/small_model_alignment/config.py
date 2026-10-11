"""Central experiment configuration.

Assignment task: make every choice that affects reproducibility explicit:
checkpoint, sequence length, learning rate, precision, batch size, seed,
DPO beta, decoding settings, and output paths. Do not rely on notebook state.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-0.5B"
    max_length: int = 512
    seed: int = 42


@dataclass
class SFTConfig:
    train_examples: int = 30_000
    validation_examples: int = 1_000
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True


@dataclass
class DPOConfig:
    beta: float = 0.1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 1e-6
    epochs: int = 1
    max_grad_norm: float = 1.0


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


@dataclass
class Paths:
    root: Path = field(default_factory=lambda: Path("outputs"))

    @property
    def checkpoints(self) -> Path:
        """Return the planned checkpoint directory under ``Paths.root``."""

        return self.root / "checkpoints"

    @property
    def generations(self) -> Path:
        """Return the planned generation-artifact directory."""

        return self.root / "generations"

    @property
    def evaluations(self) -> Path:
        """Return the planned judgment/evaluation directory."""

        return self.root / "eval"
