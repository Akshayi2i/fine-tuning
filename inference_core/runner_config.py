"""Generation and backend settings, read once and shared.

Evaluation, serving, and testing must generate **identically** or their numbers
do not describe the same system. Generation parameters are as capable of
breaking that as the prompt is, so they come from one config
(``configs/inference/vllm_serving.yaml``) rather than from each caller's
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from common.config import ConfigError, generation_config, resolution_cap_px, serving_config

Backend = Literal["vllm", "hf", "gguf"]


@dataclass(frozen=True)
class RunnerConfig:
    """Everything that determines what the model produces for a given input."""

    backend: Backend = "vllm"
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 4096
    seed: int = 42
    logprobs: bool = True
    top_logprobs: int = 1
    resolution_cap_px: int = 1792
    max_seq_len: int = 8192
    gpu_memory_utilization: float = 0.90
    enable_lora: bool = True
    max_loras: int = 4
    max_lora_rank: int = 64

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    def as_generation_kwargs(self) -> dict[str, Any]:
        """Backend-agnostic generation arguments."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "logprobs": self.logprobs,
        }

    def fingerprint(self) -> str:
        """Short hash of the settings that affect output.

        Recorded alongside a result so two runs that disagree can be checked for
        having actually generated the same way, rather than that being assumed.
        """
        import hashlib

        payload = f"{self.backend}|{self.temperature}|{self.top_p}|{self.max_new_tokens}|{self.seed}"
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_runner_config(backend: Backend | None = None) -> RunnerConfig:
    """Build the config from ``configs/inference/vllm_serving.yaml``.

    Rejects a config with logprobs disabled: per-field confidence is derived from
    token logprobs (arch §5), so a runner that cannot return them produces
    extractions no one can route for review — which is the point of the
    confidence work.
    """
    serving = serving_config()
    generation = generation_config()          # raises when logprobs are off
    serve = serving.get("serving", {})
    sequence = serving.get("sequence", {})

    config = RunnerConfig(
        backend=backend or serve.get("backend", "vllm"),
        temperature=float(generation.get("temperature", 0.0)),
        top_p=float(generation.get("top_p", 1.0)),
        max_new_tokens=int(generation.get("max_new_tokens", 4096)),
        seed=int(generation.get("seed", 42)),
        logprobs=bool(generation.get("logprobs", True)),
        top_logprobs=int(generation.get("top_logprobs", 1)),
        resolution_cap_px=resolution_cap_px(),
        max_seq_len=int(sequence.get("max_seq_len", 8192)),
        gpu_memory_utilization=float(serve.get("gpu_memory_utilization", 0.90)),
        enable_lora=bool(serve.get("enable_lora", True)),
        max_loras=int(serve.get("max_loras", 4)),
        max_lora_rank=int(serve.get("max_lora_rank", 64)),
    )

    if not config.is_greedy:
        # Not fatal, but it costs reproducibility: the same document would
        # extract differently on two runs, and the calibration fitted on one
        # would not describe the other.
        import warnings

        warnings.warn(
            f"temperature is {config.temperature}, not 0.0. Extraction is not a creative task; "
            "sampling makes the same document extract differently across runs and invalidates "
            "the confidence calibration fitted on it.",
            stacklevel=2,
        )
    return config


def assert_lora_rank_supported(config: RunnerConfig, foundation_rank: int) -> None:
    """The serving stack must admit the Foundation adapter's rank.

    Catches the mismatch at config time rather than as a confusing load failure
    on a cold start, when it is much harder to attribute.
    """
    if foundation_rank > config.max_lora_rank:
        raise ConfigError(
            f"the Foundation adapter is rank {foundation_rank} but serving allows at most "
            f"{config.max_lora_rank}. Raise max_lora_rank in configs/inference/vllm_serving.yaml."
        )
