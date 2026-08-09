"""Canonical small language model architecture, datasets, and resolved training profiles."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SMOLTALK_DATASET = "HuggingFaceTB/smoltalk"
SMOLTALK_CONFIG = "smol-magpie-ultra"
SMOLTALK_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"

SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
)


class ConfigurationError(ValueError):
    """Raised when a model configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Frozen decoder architecture shared by JAX training and Rust inference."""

    model_name: str = "small-lm-8m"
    architecture: str = "decoder-only-transformer"
    vocab_size: int = 8192
    context_length: int = 512
    num_layers: int = 8
    hidden_size: int = 256
    intermediate_size: int = 704
    num_query_heads: int = 4
    num_key_value_heads: int = 2
    head_dimension: int = 64
    rms_norm_epsilon: float = 1e-5
    rope_theta: float = 10_000.0
    situ_beta_gate: float = 4.0
    situ_beta_up: float = 25.0
    tie_word_embeddings: bool = True
    use_bias: bool = False
    dropout: float = 0.0
    expected_parameter_count: int = 7_999_744

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate architecture invariants and the exact parameter count."""

        if self.architecture != "decoder-only-transformer":
            raise ConfigurationError("architecture must be decoder-only-transformer")
        for name, value in (
            ("vocab_size", self.vocab_size),
            ("context_length", self.context_length),
            ("num_layers", self.num_layers),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_query_heads", self.num_query_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dimension", self.head_dimension),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive")
        if self.hidden_size != self.num_query_heads * self.head_dimension:
            raise ConfigurationError("hidden_size must equal num_query_heads * head_dimension")
        if self.num_query_heads % self.num_key_value_heads:
            raise ConfigurationError("num_query_heads must be divisible by num_key_value_heads")
        if self.vocab_size >= 65_536:
            raise ConfigurationError("uint16 token storage requires vocab_size < 65536")
        if self.rms_norm_epsilon <= 0.0 or self.rope_theta <= 0.0:
            raise ConfigurationError("RMSNorm epsilon and RoPE theta must be positive")
        if self.situ_beta_gate <= 0.0 or self.situ_beta_up <= 0.0:
            raise ConfigurationError("bounded-gate beta values must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigurationError("dropout must be in [0, 1)")
        if not self.tie_word_embeddings or self.use_bias or self.dropout != 0.0:
            raise ConfigurationError("small-lm-v1 requires tied embeddings, no bias, and no dropout")
        if self.calculated_parameter_count != self.expected_parameter_count:
            raise ConfigurationError(
                "expected_parameter_count does not match the configured architecture: "
                f"{self.expected_parameter_count} != {self.calculated_parameter_count}"
            )

    @property
    def calculated_parameter_count(self) -> int:
        """Return the exact dense tied-embedding parameter count."""

        kv_width = self.num_key_value_heads * self.head_dimension
        embedding = self.vocab_size * self.hidden_size
        attention = 2 * self.hidden_size * self.hidden_size + 2 * self.hidden_size * kv_width
        ffn = 3 * self.hidden_size * self.intermediate_size
        norms = 2 * self.hidden_size
        return embedding + self.num_layers * (attention + ffn + norms) + self.hidden_size

    def parameter_count(self) -> int:
        """Compatibility alias used by training checks."""

        return self.calculated_parameter_count

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> Self:
        """Build a validated model configuration from JSON-compatible values."""

        try:
            return cls(**payload)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(f"invalid model configuration: {error}") from error


MODEL = ModelConfig()


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    """Muon/AdamW optimizer settings."""

    muon_learning_rate: float
    adam_learning_rate: float
    weight_decay: float
    warmup_fraction: float
    gradient_clip_norm: float

    def validate(self) -> None:
        for name, value in (
            ("muon_learning_rate", self.muon_learning_rate),
            ("adam_learning_rate", self.adam_learning_rate),
            ("gradient_clip_norm", self.gradient_clip_norm),
        ):
            if value <= 0.0:
                raise ConfigurationError(f"{name} must be positive")
        if self.weight_decay < 0.0:
            raise ConfigurationError("weight_decay must be non-negative")
        if not 0.0 < self.warmup_fraction < 1.0:
            raise ConfigurationError("warmup_fraction must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    """Resolved smoke, pilot, or full pretraining profile."""

    name: str
    seed: int
    train_tokens: int
    validation_tokens: int
    global_sequences: int
    microbatch_sequences: int
    validation_sequences: int
    validate_every_updates: int
    checkpoint_every_updates: int
    log_every_updates: int
    tokenizer_bytes: int
    optimizer: OptimizerSpec

    @property
    def tokens_per_update(self) -> int:
        return self.global_sequences * MODEL.context_length

    @property
    def updates(self) -> int:
        return math.ceil(self.train_tokens / self.tokens_per_update)

    def validate(self) -> None:
        if not self.name:
            raise ConfigurationError("training profile name must not be empty")
        for name, value in (
            ("train_tokens", self.train_tokens),
            ("validation_tokens", self.validation_tokens),
            ("global_sequences", self.global_sequences),
            ("microbatch_sequences", self.microbatch_sequences),
            ("validation_sequences", self.validation_sequences),
            ("validate_every_updates", self.validate_every_updates),
            ("checkpoint_every_updates", self.checkpoint_every_updates),
            ("log_every_updates", self.log_every_updates),
            ("tokenizer_bytes", self.tokenizer_bytes),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive")
        if self.seed < 0:
            raise ConfigurationError("seed must be non-negative")
        if self.global_sequences % self.microbatch_sequences:
            raise ConfigurationError("global_sequences must be divisible by microbatch_sequences")
        self.optimizer.validate()


@dataclass(frozen=True, slots=True)
class SFTProfile:
    """Resolved instruction-tuning profile."""

    name: str
    seed: int
    selected_examples: int
    train_examples: int
    validation_examples: int
    batch_size: int
    microbatch_sequences: int
    validate_every_updates: int
    checkpoint_every_updates: int
    log_every_updates: int
    optimizer: OptimizerSpec

    @property
    def updates(self) -> int:
        return math.ceil(self.train_examples / self.batch_size)

    def validate(self) -> None:
        if self.train_examples + self.validation_examples != self.selected_examples:
            raise ConfigurationError("SFT train + validation must equal selected_examples")
        for name, value in (
            ("selected_examples", self.selected_examples),
            ("train_examples", self.train_examples),
            ("validation_examples", self.validation_examples),
            ("batch_size", self.batch_size),
            ("microbatch_sequences", self.microbatch_sequences),
            ("validate_every_updates", self.validate_every_updates),
            ("checkpoint_every_updates", self.checkpoint_every_updates),
            ("log_every_updates", self.log_every_updates),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive")
        if self.seed < 0:
            raise ConfigurationError("seed must be non-negative")
        if self.batch_size % self.microbatch_sequences:
            raise ConfigurationError("SFT batch_size must be divisible by microbatch_sequences")
        self.optimizer.validate()


def _read_json(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"configuration file does not exist: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"invalid JSON in {config_path} at line {error.lineno}, column {error.colno}"
        ) from error
    if not isinstance(payload, dict):
        raise ConfigurationError(f"configuration must be a JSON object: {config_path}")
    return payload


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate the frozen model JSON."""

    return ModelConfig.from_mapping(_read_json(path))


def _optimizer_from_mapping(payload: object) -> OptimizerSpec:
    if not isinstance(payload, dict):
        raise ConfigurationError("optimizer must be a JSON object")
    try:
        optimizer = OptimizerSpec(
            muon_learning_rate=float(payload["muon_learning_rate"]),
            adam_learning_rate=float(payload["adam_learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            warmup_fraction=float(payload["warmup_fraction"]),
            gradient_clip_norm=float(payload["gradient_clip_norm"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"invalid optimizer configuration: {error}") from error
    optimizer.validate()
    return optimizer


def load_profile(path: str | Path) -> TrainingProfile:
    """Load one resolved pretraining profile."""

    payload = _read_json(path)
    try:
        profile = TrainingProfile(
            name=str(payload["name"]),
            seed=int(payload["seed"]),
            train_tokens=int(payload["train_tokens"]),
            validation_tokens=int(payload["validation_tokens"]),
            global_sequences=int(payload["global_sequences"]),
            microbatch_sequences=int(payload["microbatch_sequences"]),
            validation_sequences=int(payload["validation_sequences"]),
            validate_every_updates=int(payload["validate_every_updates"]),
            checkpoint_every_updates=int(payload["checkpoint_every_updates"]),
            log_every_updates=int(payload["log_every_updates"]),
            tokenizer_bytes=int(payload["tokenizer_bytes"]),
            optimizer=_optimizer_from_mapping(payload["optimizer"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"invalid training profile: {error}") from error
    profile.validate()
    return profile


def load_sft_profile(path: str | Path) -> SFTProfile:
    """Load the resolved SFT profile."""

    payload = _read_json(path)
    try:
        profile = SFTProfile(
            name=str(payload["name"]),
            seed=int(payload["seed"]),
            selected_examples=int(payload["selected_examples"]),
            train_examples=int(payload["train_examples"]),
            validation_examples=int(payload["validation_examples"]),
            batch_size=int(payload["batch_size"]),
            microbatch_sequences=int(payload["microbatch_sequences"]),
            validate_every_updates=int(payload.get("validate_every_updates", 1000)),
            checkpoint_every_updates=int(payload.get("checkpoint_every_updates", 250)),
            log_every_updates=int(payload.get("log_every_updates", 25)),
            optimizer=_optimizer_from_mapping(payload["optimizer"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"invalid SFT profile: {error}") from error
    profile.validate()
    return profile


def workspace_root() -> Path:
    """Return the local or Modal durable workspace root."""

    return Path(os.environ.get("NILEMINI_WORKSPACE", ".")).resolve()


def data_dir(root: Path | None = None) -> Path:
    path = (root or workspace_root()) / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir(root: Path | None = None) -> Path:
    path = (root or workspace_root()) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(root: Path | None = None) -> Path:
    path = (root or workspace_root()) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_dir(root: Path | None = None) -> Path:
    path = (root or workspace_root()) / "artifacts" / MODEL.model_name
    path.mkdir(parents=True, exist_ok=True)
    return path
