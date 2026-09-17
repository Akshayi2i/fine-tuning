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

#: The reference every serving format is measured against (arch v2.1 §13b).
#: bf16, not fp16: fp16 is a GGUF format, so v1 measured against an artifact
#: the serving endpoint could not load.
BF16 = {"field_normalized_match": 0.90, "ece_confidence": 0.040, "schema_validity_rate": 1.0}


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

    assert load_phase("epochs")["metric"] == {"name": "field_normalized_match", "goal": "maximize"}
    assert load_phase("rank")["metric"]["name"] == "field_normalized_match"
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
                                   "field_normalized_match": 0.9 if good else 0.5})

    result = run_sweep(train, documents_per_type=AT_SCALE)

    epoch_runs = [o for o in seen if "num_train_epochs" in o]
    assert epoch_runs, "phase 2 did not run"
    assert all(o.get("learning_rate") == 1e-4 for o in epoch_runs), \
        "phase 2 did not fix phase 1's winner"
    assert result["winning_config"]["learning_rate"] == 1e-4


def test_rank_is_not_swept_by_default():
    """Least likely bottleneck; the spec runs it only if F1 plateaus."""
    def train(c: Candidate) -> CandidateResult:
        return CandidateResult(c, {"eval_loss": 0.1, "field_normalized_match": 0.9})

    result = run_sweep(train, documents_per_type=AT_SCALE)
    rank = next(p for p in result["phases"] if p["phase"] == "rank")

    assert rank["skipped"]
    assert rank["candidates"] == []
    assert result["total_runs"] == 6      # 3 + 3, not 9


def test_the_full_budget_stays_inside_nine_to_twelve_runs():
    def train(c: Candidate) -> CandidateResult:
        return CandidateResult(c, {"eval_loss": 0.1, "field_normalized_match": 0.9})

    result = run_sweep(train, documents_per_type=AT_SCALE, include_rank=True)
    assert result["total_runs"] == 9


# ==========================================================================
# Quantization thresholds
# ==========================================================================


def test_every_serving_format_has_a_threshold():
    """GGUF formats are absent by design: they are validated in llama.cpp, not
    by the serving gate, because the serving endpoint never loads one."""
    assert set(THRESHOLDS) == {"bf16", "fp8", "awq_int4"}


def test_an_unknown_format_is_refused_rather_than_defaulted():
    """A permissive default would let an unlisted format pass by omission."""
    with pytest.raises(ThresholdError, match="no quantization threshold"):
        threshold_for("int2")


def test_a_gguf_format_says_where_it_is_actually_validated():
    """Asking the serving gate about a GGUF is a category error, and the error
    should say so rather than reading as 'unsupported'."""
    with pytest.raises(ThresholdError, match="llama.cpp"):
        threshold_for("q5_k_m")


def test_the_allowance_loosens_as_the_bits_drop():
    drops = [threshold_for(f).max_exact_match_drop_pp for f in ("bf16", "fp8", "awq_int4")]
    assert drops == sorted(drops), "a lower-bit format should not be held to a tighter bar"


def test_the_unforgiving_field_class_gets_the_tighter_margin():
    """A wrong policy number is a wrong extraction; a slightly-off entity name is
    usually still matchable. One allowance would let the forgiving class lend its
    slack to the unforgiving one."""
    for fmt in ("fp8", "awq_int4"):
        threshold = threshold_for(fmt)
        assert threshold.max_exact_match_drop_pp < threshold.max_fuzzy_match_drop_pp


def test_schema_validity_is_a_floor_not_a_margin():
    """Structured decoding guarantees it, so anything below 100% means the
    guarantee is not working — which is an error, not a degraded result."""
    for fmt in THRESHOLDS:
        assert threshold_for(fmt).min_schema_validity == 1.0


def test_margins_are_absolute_percentage_points_not_relative():
    """v1 allowed a 2% RELATIVE drop, which is 1.9pp at F1 0.95 and 1.4pp at
    0.70 — so the rule got stricter as the model got worse. The same number
    should mean the same thing whatever the baseline."""
    allowance_pp = threshold_for("fp8").max_fuzzy_match_drop_pp / 100.0

    high = {**BF16, "field_normalized_match": 0.95}
    low = {**BF16, "field_normalized_match": 0.70}
    assert validate_quant({"bf16": high, "fp8": _metrics(0.95 - allowance_pp)}).all_passed
    assert validate_quant({"bf16": low, "fp8": _metrics(0.70 - allowance_pp)}).all_passed


def test_a_format_inside_its_allowance_passes():
    report = validate_quant({"bf16": BF16, "fp8": _metrics(0.897)})   # 0.3pp, allowance 0.5pp
    assert report.servable_formats == ["bf16", "fp8"]
    assert report.all_passed


def test_a_format_over_its_allowance_is_blocked():
    report = validate_quant({"bf16": BF16, "fp8": _metrics(0.88)})    # 2.0pp drop
    assert "fp8" in report.blocked_formats
    assert "field match dropped" in report.results[-1].describe()


def test_the_boundary_is_evaluated_exactly():
    allowance = threshold_for("fp8").max_fuzzy_match_drop_pp / 100.0
    at = BF16["field_normalized_match"] - allowance

    assert validate_quant({"bf16": BF16, "fp8": _metrics(at)}).all_passed
    assert not validate_quant({"bf16": BF16, "fp8": _metrics(at - 0.001)}).all_passed


def test_normalized_match_takes_the_fuzzy_allowance_and_exact_match_its_own():
    """Normalized match was held to the 0.5pp exact-match allowance, rejecting FP8
    at half the drop §13b permits, and max_fuzzy_match_drop_pp was read by
    nothing. Each class is held to its own margin."""
    reference = {**BF16, "field_exact_match": 0.90}

    fuzzy_only = {**_metrics(0.892), "field_exact_match": 0.90}       # 0.8pp normalized
    assert validate_quant({"bf16": reference, "fp8": fuzzy_only}).all_passed

    exact_drop = {**_metrics(0.90), "field_exact_match": 0.892}       # 0.8pp exact
    report = validate_quant({"bf16": reference, "fp8": exact_drop})
    assert "fp8" in report.blocked_formats
    assert any("exact match" in r for r in report.results[-1].reasons)


def test_row_recall_is_checked_on_its_own_margin():
    """A quantized model that keeps its field accuracy while dropping claim rows
    has lost exactly what a Loss Run is for, and an aggregate field score would
    not show it."""
    reference = {**BF16, "field_exact_match": 0.90, "list_field_recall": 0.90}
    candidate = {**_metrics(0.90), "field_exact_match": 0.90, "list_field_recall": 0.86}
    report = validate_quant({"bf16": reference, "fp8": candidate})

    assert "fp8" in report.blocked_formats
    assert any("row recall" in r for r in report.results[-1].reasons)


def test_entity_collapse_blocks_a_format():
    """Quantization collapsing two entities is exactly the failure canonical
    mapping exists to prevent."""
    reference = {**BF16, "confusable_misattribution_rate": 0.01}
    candidate = {**_metrics(0.90), "confusable_misattribution_rate": 0.05}
    report = validate_quant({"bf16": reference, "fp8": candidate})

    assert "fp8" in report.blocked_formats
    assert any("misattribution" in r for r in report.results[-1].reasons)


def test_calibration_drift_blocks_a_format_on_its_own():
    """A miscalibrated model routes the wrong documents to review, whatever its
    accuracy."""
    report = validate_quant({"bf16": BF16, "fp8": _metrics(0.899, ece=0.10)})
    assert "fp8" in report.blocked_formats
    assert any("ECE rose" in r for r in report.results[-1].reasons)


def test_a_format_that_was_not_measured_does_not_pass():
    report = validate_quant({"bf16": BF16, "awq_int4": {"field_normalized_match": 0.88}})
    assert "awq_int4" in report.blocked_formats
    assert any("not measured" in r for r in report.results[-1].reasons)


def test_validation_without_the_reference_is_refused():
    """Every threshold is a margin against bf16, so without it each format would
    be compared to nothing."""
    with pytest.raises(QuantValidationError, match="no bf16 metrics"):
        validate_quant({"fp8": _metrics(0.88)})


def test_the_reference_must_itself_emit_valid_json():
    """A reference that cannot produce valid JSON makes every comparison against
    it meaningless."""
    report = validate_quant({"bf16": _metrics(0.90, validity=0.90)})
    assert "bf16" in report.blocked_formats


def test_promotion_of_a_blocked_format_is_refused():
    report = validate_quant({"bf16": BF16, "awq_int4": _metrics(0.70)})
    with pytest.raises(QuantValidationError, match="did not pass"):
        assert_servable(report, ["awq_int4"])


def test_there_is_no_override_for_a_blocked_format():
    """Every other guarantee becomes advisory the moment one exists."""
    import inspect

    from postprocessing import validate_quant as module

    source = inspect.getsource(module)
    for forbidden in ("force", "override", "skip_validation", "ignore_threshold"):
        assert f"{forbidden}=" not in source and f"--{forbidden}" not in source


def test_the_result_is_recorded_for_the_manifest():
    entry = validate_quant({"bf16": BF16, "fp8": _metrics(0.897)}).manifest_entry()
    assert entry["reference_format"] == "bf16"
    assert entry["servable"] and "per_format" in entry


def test_the_default_serving_target_is_bf16_until_fp8_is_verified():
    """FP8 is the plan, not a proven capability. Phase 0 spike item 9 exists to
    confirm a decoder-only FP8 export loads and runs in vLLM; until it has, the
    first cycle serves the merged bf16 model — slower and certain, rather than
    fast and unproven (arch v2.1 §13a)."""
    assert DEFAULT_SERVING_FORMAT == "bf16"


def test_gguf_is_no_longer_a_per_cycle_deliverable():
    """It is llama.cpp's format and the endpoint runs vLLM, so producing one per
    cycle spent conversion and eval compute on an artifact nothing could deploy."""
    from postprocessing.quantize import DEFAULT_FORMATS, QuantizationError, plan_quantization

    assert "q5_k_m" not in DEFAULT_FORMATS
    with pytest.raises(QuantizationError, match="vLLM does not"):
        plan_quantization(version="v2", formats=["q5_k_m"])


def test_a_serving_plan_always_carries_the_bf16_reference():
    """Every §13b threshold is an absolute margin against bf16, so a run that
    produces only a quantized format has nothing to measure its drop from."""
    from postprocessing.quantize import QuantizationError, plan_quantization

    with pytest.raises(QuantizationError, match="omit the bf16 reference"):
        plan_quantization(version="v2", formats=["fp8"])


def test_fp8_is_refused_until_the_spike_verifies_it():
    """An unverified serving format is a deployment that fails at load — or
    worse, one that loads and reads badly."""
    from postprocessing.quantize import QuantizationError, plan_quantization, quantize

    plan = plan_quantization(version="v2", formats=["bf16", "fp8"])
    with pytest.raises(QuantizationError, match="not verified"):
        quantize(plan, dry_run=True)

    verified = plan_quantization(version="v2", formats=["bf16", "fp8"], fp8_verified=True)
    produced = quantize(verified, dry_run=True)
    assert set(produced) == {"bf16", "fp8"}


def test_bf16_is_never_re_exported():
    """It IS the merged model. A copy would be a second 16GB artifact identical
    to the first."""
    from postprocessing.quantize import plan_quantization, quantize

    plan = plan_quantization(version="v2", formats=["bf16"])
    assert quantize(plan, dry_run=True)["bf16"] == plan.merged_model
    assert plan.quantized_formats == []


def test_the_vision_path_is_never_quantized():
    """Compressing it produces a model that loads cleanly and reads pages badly
    — very hard to notice, because it still emits well-formed JSON (§13a)."""
    from postprocessing.quantize import NEVER_QUANTIZED

    joined = " ".join(NEVER_QUANTIZED)
    assert "visual" in joined and "merger" in joined and "lm_head" in joined


def test_a_gguf_export_still_refuses_an_unverified_mmproj():
    """Unchanged from v1, and it stays unchanged: exporting without one produces
    a model that loads and cannot see."""
    from postprocessing.quantize import QuantizationError, export_gguf, plan_gguf_export

    plan = plan_gguf_export(version="v2", formats=["q5_k_m"])
    with pytest.raises(QuantizationError, match="cannot see"):
        export_gguf(plan, dry_run=True)

    verified = plan_gguf_export(version="v2", formats=["q5_k_m"], mmproj_verified=True)
    assert export_gguf(verified, dry_run=True)
