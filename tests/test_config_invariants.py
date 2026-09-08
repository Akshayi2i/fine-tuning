"""Cross-file config invariants (SPEC_01).

Each of these spans two files that are read by different stages. Nothing else in
the system would notice them drifting apart, and none of them fail loudly on
their own — they fail as a quietly worse model.
"""

from __future__ import annotations

import pytest

from common import config
from common.constants import RESOLUTION_CAP_RANGE_PX


def test_all_invariants_hold_as_shipped():
    """The configs in the repo must be self-consistent out of the box."""
    config.validate_all()


def test_resolution_cap_is_within_the_architecture_range():
    low, high = RESOLUTION_CAP_RANGE_PX
    assert low <= config.resolution_cap_px() <= high


def test_resolution_parity_between_training_and_serving():
    """The one that matters most.

    The model learns at one resolution and would be served at another — the
    symptom is degraded accuracy with no error anywhere.
    """
    train = config.base_model_config()["vision"]["max_image_long_side_px"]
    serve = config.serving_config()["vision"]["max_image_long_side_px"]
    assert train == serve


def test_resolution_parity_check_fires_on_mismatch(monkeypatch):
    monkeypatch.setattr(
        config, "serving_config", lambda: {"vision": {"max_image_long_side_px": 2048}}
    )
    with pytest.raises(config.ConfigError, match="resolution cap mismatch"):
        config.assert_resolution_parity()


@pytest.mark.parametrize(
    "name", ["foundation", "acord_adapter", "policy_adapter", "lossrun_adapter"]
)
def test_declared_effective_batch_matches_the_arithmetic(name):
    """Otherwise the run manifest records a batch size that was never used."""
    config.assert_effective_batch(config.training_config(name))


def test_effective_batch_check_fires_on_mismatch():
    with pytest.raises(config.ConfigError, match="effective_batch_size is declared"):
        config.assert_effective_batch(
            {"batch": {"per_device_train_batch_size": 1,
                       "gradient_accumulation_steps": 8,
                       "effective_batch_size": 32}}
        )


def test_generation_requires_logprobs():
    """Per-field confidence is derived from token logprobs (arch §5)."""
    assert config.generation_config()["logprobs"] is True


def test_generation_config_rejects_disabled_logprobs(monkeypatch):
    monkeypatch.setattr(config, "serving_config", lambda: {"generation": {"logprobs": False}})
    with pytest.raises(config.ConfigError, match="logprobs is disabled"):
        config.generation_config()


def test_generation_is_greedy_by_default():
    """Sampling would make one document extract differently on two runs, which
    breaks reproducibility and the calibration fitted on it."""
    assert config.generation_config()["temperature"] == 0.0


def test_unpinned_base_model_revision_is_detected():
    """Phase 0 has not pinned it yet — this test documents that, and will start
    passing (as a no-raise) once the revision is set."""
    revision = config.base_model_config()["model"]["revision"]
    if revision == "PIN_ME":
        with pytest.raises(config.ConfigError, match="unpinned"):
            config.assert_model_revision_pinned()
    else:
        config.assert_model_revision_pinned()


def test_foundation_and_per_type_ranks_match_the_architecture():
    """Foundation learns broad behaviour; per-type learns only schema mapping."""
    from common.constants import (
        FOUNDATION_LORA_ALPHA,
        FOUNDATION_LORA_RANK,
        PER_TYPE_LORA_ALPHA,
        PER_TYPE_LORA_RANK,
    )
    foundation = config.training_config("foundation")["lora"]
    assert (foundation["rank"], foundation["alpha"]) == (FOUNDATION_LORA_RANK, FOUNDATION_LORA_ALPHA)

    for name in ("acord_adapter", "policy_adapter", "lossrun_adapter"):
        per_type = config.training_config(name)["lora"]
        assert (per_type["rank"], per_type["alpha"]) == (PER_TYPE_LORA_RANK, PER_TYPE_LORA_ALPHA)


def test_vit_is_frozen_by_default():
    """Unfreezing is an escalation through the arch §3 gate, and even then it is
    LoRA-on-ViT, never a full fine-tune."""
    assert config.base_model_config()["vision"]["train_vit"] is False


def test_per_type_adapters_do_not_pin_a_foundation_version():
    """Resolved at launch from the registry, so an adapter can never be built
    against a stale Foundation by a forgotten config edit (arch §12)."""
    for name in ("acord_adapter", "policy_adapter", "lossrun_adapter"):
        assert config.training_config(name)["foundation_version"] is None
