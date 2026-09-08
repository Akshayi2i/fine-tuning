"""Config loading — YAML plus environment, with the cross-file invariants enforced.

Config lives in ``configs/`` and secrets live in the environment; nothing is
hardcoded (master §7). Beyond loading, this module's real job is enforcing the
invariants that span *two* files and would otherwise be maintained by hand:

* **Resolution parity** — the image cap in ``base_model.yaml`` and
  ``inference/vllm_serving.yaml`` must be equal. They are read by different
  stages, so nothing else would notice them drifting apart, and the symptom
  would be a model served a distribution it never trained on.
* **Effective batch consistency** — the declared effective batch must equal
  ``per_device × grad_accum × n_gpu``, or the run manifest records a batch size
  that was never actually used, and the run stops being reproducible.
"""

from __future__ import annotations

import os
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import yaml

from common.constants import DEFAULT_RESOLUTION_CAP_PX, RESOLUTION_CAP_RANGE_PX

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

BASE_MODEL_CONFIG = CONFIG_DIR / "base_model.yaml"
SERVING_CONFIG = CONFIG_DIR / "inference" / "vllm_serving.yaml"


class ConfigError(RuntimeError):
    """Raised on a missing config, a missing required env var, or a broken invariant."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, failing loudly rather than returning ``{}``."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} did not parse to a mapping")
    return data


@lru_cache(maxsize=1)
def base_model_config() -> dict[str, Any]:
    return load_yaml(BASE_MODEL_CONFIG)


@lru_cache(maxsize=1)
def serving_config() -> dict[str, Any]:
    return load_yaml(SERVING_CONFIG)


@cache
def training_config(name: str) -> dict[str, Any]:
    """Load ``configs/training/{name}.yaml`` (``foundation``, ``acord_adapter``, …)."""
    return load_yaml(CONFIG_DIR / "training" / f"{name}.yaml")


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Read an env var. Secrets come from the environment only, never from YAML."""
    value = os.environ.get(name, default)
    if required and not value:
        raise ConfigError(
            f"required environment variable {name} is not set — see .env.example for the full list"
        )
    return value


def resolution_cap_px() -> int:
    """The image long-side cap, validated against the architecture's range.

    Image token count is a direct function of page resolution, making this the
    single biggest cost and latency lever (arch §11).
    """
    cap = int(base_model_config().get("vision", {}).get("max_image_long_side_px", DEFAULT_RESOLUTION_CAP_PX))
    low, high = RESOLUTION_CAP_RANGE_PX
    if not low <= cap <= high:
        raise ConfigError(
            f"resolution cap {cap}px is outside the architecture's {low}-{high}px range (arch §11). "
            "Below it, small print and checkboxes are missed; above it, cost and latency blow out."
        )
    return cap


def assert_resolution_parity() -> None:
    """The cap must be identical in training prep and production inference.

    Not a tuning knob — a mismatch is a correctness bug. The model would be
    served images at a resolution it never saw, and nothing downstream would
    report it as anything but degraded accuracy.
    """
    train_cap = base_model_config().get("vision", {}).get("max_image_long_side_px")
    serve_cap = serving_config().get("vision", {}).get("max_image_long_side_px")
    if train_cap != serve_cap:
        raise ConfigError(
            f"resolution cap mismatch: base_model.yaml says {train_cap}px, "
            f"inference/vllm_serving.yaml says {serve_cap}px. These must be equal — the model is "
            "trained at one resolution and would be served at another (arch §11)."
        )


def assert_effective_batch(config: dict[str, Any], n_gpu: int = 1) -> None:
    """Declared effective batch must match what the settings actually produce."""
    batch = config.get("batch", {})
    declared = batch.get("effective_batch_size")
    if declared is None:
        return
    actual = int(batch.get("per_device_train_batch_size", 1)) * int(batch.get("gradient_accumulation_steps", 1)) * n_gpu
    if actual != declared:
        raise ConfigError(
            f"effective_batch_size is declared {declared} but per_device × grad_accum × n_gpu = {actual}. "
            "The run manifest would record a batch size that was never used."
        )


def assert_model_revision_pinned() -> None:
    """A floating base-model revision makes a regression unattributable."""
    revision = base_model_config().get("model", {}).get("revision")
    if not revision or revision == "PIN_ME":
        raise ConfigError(
            "base_model.yaml model.revision is unpinned. Two runs could silently use different "
            "base weights, which makes 'why did this regress' unanswerable. Pin it (Phase 0)."
        )


def validate_all(*, require_pinned_revision: bool = False) -> None:
    """Run every cross-file invariant. Called at the start of any real run.

    ``require_pinned_revision`` is off by default so the fixture-driven test
    suite runs before Phase 0's infrastructure exists; orchestration turns it on.
    """
    resolution_cap_px()
    assert_resolution_parity()
    for name in ("foundation", "acord_adapter", "policy_adapter", "lossrun_adapter"):
        assert_effective_batch(training_config(name))
    if require_pinned_revision:
        assert_model_revision_pinned()


def generation_config() -> dict[str, Any]:
    """Generation settings, shared by eval, serving, and testing.

    Logprobs are mandatory: they are the entire basis of per-field confidence
    (arch §5), so a config that disables them is rejected here rather than
    producing confidence-free output somewhere downstream.
    """
    gen = dict(serving_config().get("generation", {}))
    if not gen.get("logprobs"):
        raise ConfigError(
            "generation.logprobs is disabled. Per-field confidence is derived from token "
            "logprobs (arch §5); without them the pipeline cannot route low-confidence "
            "extractions to review, which is the point of the confidence work."
        )
    return gen
