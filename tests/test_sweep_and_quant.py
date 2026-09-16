"""SPEC_06 §6 and SPEC_10 §3–4 — the sweep protocol and quantization thresholds.

Both were deferred for a reason that still holds about *when* to run them. Both
have acceptance criteria that need no GPU, which is what these cover: the sweep
against a stub trainer, and the thresholds at each format's boundary.

The property both share is the one that matters: **unmeasured is not passing.** A
candidate that crashed must not win a minimise-loss phase, and a format nobody
scored must not reach serving.
"""

from __future__ import annotations

import pytest

from postprocessing.quant_thresholds import (
    DEFAULT_SERVING_FORMAT,
    THRESHOLDS,
    ThresholdError,
    threshold_for,
)
from postprocessing.validate_quant import (
    QuantValidationError,
    assert_servable,
    validate_quant,
)
from registry_utils.models import RunManifest
from training.sweep import (
    Candidate,
    CandidateResult,
    SweepError,
    assert_enough_data,
    candidates_for,
    pick_winner,
    run_phase,
    run_sweep,
)


def _untagged_manifest(run_id: str) -> RunManifest:
    """A manifest a sweep produced but did not tag."""
    from registry_utils.models import DataStats, Dependencies, TrainingConfig

    return RunManifest(
        run_id=run_id,
        run_type="foundation",
        dependencies=Dependencies(
            base_model="qwen3-vl-8b-instruct@abc1234",
            corpus_version="corpus/v1",
            code_git_commit="abc1234",
        ),
        training_config=TrainingConfig(
            technique="LoRA", base_quantization="bf16_frozen_base", optimizer="adamw_torch",
            lora_rank=64, lora_alpha=128, learning_rate=1e-4, epochs=3,
            gradient_accumulation_steps=32, effective_batch_size=32,
            target_modules=["q_proj"], resolution_cap_px=1792, max_seq_len=8192, seed=42,
        ),
        data_stats=DataStats(train_examples=10, val_examples=2, test_examples=2),
        is_sweep_run=False,     # the defect under test
    )


AT_SCALE = {"policy": 400, "lossrun": 350, "acord": 300}

FP16 = {"field_normalized_match": 0.90, "ece_confidence": 0.040, "schema_validity_rate": 1.0}


def _metrics(f1: float, ece: float = 0.040, validity: float = 1.0) -> dict[str, float]:
    return {"field_normalized_match": f1, "ece_confidence": ece, "schema_validity_rate": validity}


# ==========================================================================
# The sweep
# ==========================================================================


def test_the_sweep_refuses_to_run_at_pilot_volume():
    """A sweep that ran and picked a winner produces a config that *looks*
    justified and gets promoted. At 25-30 docs/type the test split is 3-4
    documents and the difference between 1e-4 and 2e-4 is smaller than the
    variance from which documents landed in it."""
    with pytest.raises(SweepError, match="below the 200"):
        assert_enough_data({"policy": 28, "lossrun": 25})


def test_the_sweep_proceeds_at_production_volume():
    assert_enough_data(AT_SCALE)      # does not raise


def test_phase_one_sweeps_learning_rate_first():
    """Highest impact, and every later phase is conditioned on its winner."""
    candidates = candidates_for("lr")
    assert {c.parameter for c in candidates} == {"learning_rate"}
    assert len(candidates) == 3


def test_phase_two_optimises_f1_not_loss():
    """Validation loss can fall while field extraction gets worse, and F1 is what
    the promotion gate reads."""
    from training.sweep import load_phase

    assert load_phase("epochs")["metric"] == {"name": "field_f1", "goal": "maximize"}
    assert load_phase("lr")["metric"]["goal"] == "minimize"


def test_a_winner_is_picked_by_the_phase_s_own_goal():
    lo = CandidateResult(Candidate("lr", "foundation", "learning_rate", 5e-5), {"eval_loss": 0.20})
    hi = CandidateResult(Candidate("lr", "foundation", "learning_rate", 2e-4), {"eval_loss": 0.40})

    assert pick_winner([lo, hi], "eval_loss", "minimize") is lo
    assert pick_winner([lo, hi], "eval_loss", "maximize") is hi


def test_an_unmeasured_candidate_cannot_win():
    """Ranking a crashed run as if it scored zero would let it win a
    minimise-loss phase outright."""
    crashed = CandidateResult(Candidate("lr", "foundation", "learning_rate", 5e-5), {})
    real = CandidateResult(Candidate("lr", "foundation", "learning_rate", 1e-4), {"eval_loss": 0.3})

    assert pick_winner([crashed, real], "eval_loss", "minimize") is real
    assert pick_winner([crashed], "eval_loss", "minimize") is None


def test_a_phase_where_nothing_was_measured_has_no_winner():
    def crashes(c: Candidate) -> CandidateResult:
        return CandidateResult(c, {})

    with pytest.raises(SweepError, match="no measured"):
        run_phase("lr", crashes)


def test_every_candidate_must_be_tagged_as_a_sweep_run():
    """Sweep runs are first-class registry entries; an untagged one would swamp
    default listings and read as a production run."""
    def untagged(c: Candidate) -> CandidateResult:
        return CandidateResult(c, {"eval_loss": 0.2}, manifest=_untagged_manifest(c.run_id))

    with pytest.raises(SweepError, match="is_sweep_run"):
        run_phase("lr", untagged)


def test_phase_two_is_conditioned_on_phase_one_s_winner():
    """An unbounded grid is 27 runs before it says anything; the bound comes from
    each phase fixing the previous winner."""
    seen: list[dict] = []

    def train(c: Candidate) -> CandidateResult:
        seen.append(c.overrides())
        score = {"learning_rate": 1e-4, "num_train_epochs": 3}.get(c.parameter)
        good = c.value == score
        return CandidateResult(c, {"eval_loss": 0.1 if good else 0.5,
                                   "field_f1": 0.9 if good else 0.5})

    result = run_sweep(train, documents_per_type=AT_SCALE)

    epoch_runs = [o for o in seen if "num_train_epochs" in o]
    assert epoch_runs, "phase 2 did not run"
    assert all(o.get("learning_rate") == 1e-4 for o in epoch_runs), \
        "phase 2 did not fix phase 1's winner"
    assert result["winning_config"]["learning_rate"] == 1e-4


def test_rank_is_not_swept_by_default():
    """Least likely bottleneck; the spec runs it only if F1 plateaus."""
    def train(c: Candidate) -> CandidateResult:
        return CandidateResult(c, {"eval_loss": 0.1, "field_f1": 0.9})

    result = run_sweep(train, documents_per_type=AT_SCALE)
    rank = next(p for p in result["phases"] if p["phase"] == "rank")

    assert rank["skipped"]
    assert rank["candidates"] == []
    assert result["total_runs"] == 6      # 3 + 3, not 9


def test_the_full_budget_stays_inside_nine_to_twelve_runs():
    def train(c: Candidate) -> CandidateResult:
        return CandidateResult(c, {"eval_loss": 0.1, "field_f1": 0.9})

    result = run_sweep(train, documents_per_type=AT_SCALE, include_rank=True)
    assert result["total_runs"] == 9


# ==========================================================================
# Quantization thresholds
# ==========================================================================


def test_every_format_has_a_threshold():
    assert set(THRESHOLDS) == {"fp16", "bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m"}


def test_an_unknown_format_is_refused_rather_than_defaulted():
    """A permissive default would let an unlisted format pass by omission."""
    with pytest.raises(ThresholdError, match="no quantization threshold"):
        threshold_for("q2_k")


def test_the_allowance_loosens_as_the_bits_drop():
    order = ["bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m"]
    drops = [threshold_for(f).max_field_f1_drop for f in order]
    assert drops == sorted(drops), "a lower-bit format should not be held to a tighter bar"


def test_json_validity_loosens_far_more_slowly_than_accuracy():
    """Structural discipline is one of the first things quantization costs, and
    a schema-invalid response is an error rather than a degraded result."""
    assert threshold_for("q4_k_m").min_json_validity >= 0.99


def test_a_format_inside_its_allowance_passes():
    report = validate_quant({"fp16": FP16, "q5_k_m": _metrics(0.885)})   # 1.7% drop, allowance 2%
    assert report.servable_formats == ["fp16", "q5_k_m"]
    assert report.all_passed


def test_a_format_over_its_allowance_is_blocked():
    report = validate_quant({"fp16": FP16, "q5_k_m": _metrics(0.86)})    # 4.4% drop
    assert "q5_k_m" in report.blocked_formats
    assert "field F1 dropped" in report.results[-1].describe()


def test_the_boundary_is_evaluated_exactly():
    allowance = threshold_for("q5_k_m").max_field_f1_drop
    at = FP16["field_normalized_match"] * (1 - allowance)

    assert validate_quant({"fp16": FP16, "q5_k_m": _metrics(at)}).all_passed
    assert not validate_quant({"fp16": FP16, "q5_k_m": _metrics(at - 0.01)}).all_passed


def test_calibration_drift_blocks_a_format_on_its_own():
    """A miscalibrated model routes the wrong documents to review, whatever its
    accuracy."""
    report = validate_quant({"fp16": FP16, "q6_k": _metrics(0.899, ece=0.10)})
    assert "q6_k" in report.blocked_formats
    assert any("ECE rose" in r for r in report.results[-1].reasons)


def test_a_format_that_was_not_measured_does_not_pass():
    report = validate_quant({"fp16": FP16, "q4_k_m": {"field_normalized_match": 0.88}})
    assert "q4_k_m" in report.blocked_formats
    assert any("not measured" in r for r in report.results[-1].reasons)


def test_validation_without_the_reference_is_refused():
    """Every threshold is relative to fp16, so without it each format would be
    compared to nothing."""
    with pytest.raises(QuantValidationError, match="no fp16 metrics"):
        validate_quant({"q5_k_m": _metrics(0.88)})


def test_the_reference_must_itself_emit_valid_json():
    """A reference that cannot produce valid JSON makes every comparison against
    it meaningless."""
    report = validate_quant({"fp16": _metrics(0.90, validity=0.90)})
    assert "fp16" in report.blocked_formats


def test_promotion_of_a_blocked_format_is_refused():
    report = validate_quant({"fp16": FP16, "q4_k_m": _metrics(0.70)})
    with pytest.raises(QuantValidationError, match="did not pass"):
        assert_servable(report, ["q4_k_m"])


def test_there_is_no_override_for_a_blocked_format():
    """Every other guarantee becomes advisory the moment one exists."""
    import inspect

    from postprocessing import validate_quant as module

    source = inspect.getsource(module)
    for forbidden in ("force", "override", "skip_validation", "ignore_threshold"):
        assert f"{forbidden}=" not in source and f"--{forbidden}" not in source


def test_the_result_is_recorded_for_the_manifest():
    entry = validate_quant({"fp16": FP16, "q5_k_m": _metrics(0.89)}).manifest_entry()
    assert entry["reference_format"] == "fp16"
    assert entry["servable"] and "per_format" in entry


def test_the_default_serving_target_is_q5_k_m():
    assert DEFAULT_SERVING_FORMAT == "q5_k_m"
