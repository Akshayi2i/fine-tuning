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


# --------------------------------------------------------------------------
# Per-task vision and sequence budgets (arch v2.1 §7a)
# --------------------------------------------------------------------------

def test_every_task_is_budgeted_in_both_shared_files():
    """The two files are edited separately and read together. A task present in
    one and missing from the other resolves to a default chosen for a different
    shape of input — the silent mismatch these files exist to prevent."""
    config.assert_task_budgets_are_coherent()


def test_thumbnail_tasks_cost_an_order_of_magnitude_less_than_extraction():
    """classify and page_select recognise layout, not small print. Paying
    extraction resolution for that is the most wasteful thing this pipeline
    could do — page_select alone sends up to 60 pages in one call."""
    thumbnail = config.vision_for_task("classify")["max_pixels"]
    extraction = config.vision_for_task("extract")["max_pixels"]
    assert thumbnail * 5 < extraction, (
        f"classify at {thumbnail:,}px vs extract at {extraction:,}px — a thumbnail task "
        "budgeted near extraction resolution multiplies the page_select cost by ~10x"
    )


def test_the_output_budget_always_fits_inside_the_cap():
    """The assistant span is the one part of a sequence that must never be
    truncated: a clipped JSON target trains the model to stop early, which on a
    Loss Run means training it to omit claim rows."""
    from common.tasks import Task

    for task in Task:
        budget = config.sequence_for_task(str(task))
        cap, output = int(budget["max_seq_len"]), int(budget["max_output_tokens"])
        assert output < cap, f"{task} reserves {output} output tokens inside a {cap} cap"
        assert cap - output > 1024, (
            f"{task} leaves only {cap - output} tokens for schema, images and OCR text"
        )


def test_lossrun_rows_reserves_the_largest_output_budget():
    """The one task where OUTPUT, not input, is the binding constraint: the
    window planner shrinks the page window until the estimated row output fits.
    If some other task out-reserved it, that planner would be sizing against the
    wrong limit."""
    from common.tasks import Task

    rows = int(config.sequence_for_task(str(Task.LOSSRUN_ROWS))["max_output_tokens"])
    others = {
        str(t): int(config.sequence_for_task(str(t))["max_output_tokens"])
        for t in Task if t is not Task.LOSSRUN_ROWS
    }
    assert rows > max(others.values()), f"lossrun_rows reserves {rows}, others {others}"


def test_an_unknown_task_is_refused_rather_than_silently_defaulted():
    import pytest

    with pytest.raises(ValueError, match="unknown task"):
        config.vision_for_task("summarise")


def test_the_shared_budgets_are_not_the_v1_single_cap():
    """v1 used one 8192-token cap for everything. At ~2.4k visual tokens per
    US-Letter page it could not hold a one-page ACORD with schema, image, OCR and
    output — and the canonical three-page Loss Run overflowed before a single OCR
    token was added."""
    from common.tasks import Task

    caps = {str(t): config.seq_cap_for_task(str(t)) for t in Task}
    assert len(set(caps.values())) > 1, f"all tasks share one cap: {caps}"
    assert caps["extract"] > 8192, "extraction still cannot hold a page plus its schema"


def test_policy_extraction_gets_a_larger_budget_than_acord():
    """Extraction is ONE task — the model is conditioned on doc_type through the
    prompt, not routed elsewhere. What differs is the budget: a routed policy
    sends up to 6 pages against a larger schema. Modelled as an override so
    `extract` stays one task in the corpus, the prompt and the metrics."""
    acord = config.seq_cap_for_task("extract", "acord")
    policy = config.seq_cap_for_task("extract", "policy")
    assert policy > acord, f"policy {policy} is not above acord {acord}"
    assert config.seq_cap_for_task("extract") == acord, "the bare cap is the ACORD default"


def test_an_unknown_doc_type_falls_back_to_the_task_budget():
    """A new document type must not silently inherit Policy's 32k cap."""
    assert config.seq_cap_for_task("extract", "binder") == config.seq_cap_for_task("extract")
