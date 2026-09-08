"""SPEC_08 + SPEC_09 — metrics, the promotion gate, and confidence calibration.

Two guards here matter more than the rest: the gate must block on **any single**
regression with no override, and calibration must **raise** rather than pass raw
overconfident numbers through as if they were calibrated.
"""

from __future__ import annotations

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.apply_calibration import (
    apply_calibration,
    calibrated_output,
    load_calibration,
)
from calibration.fit_calibration import (
    CalibrationError,
    CalibrationParams,
    assert_held_out,
    fit_calibration,
)
from calibration.list_completeness import check_completeness, check_document, merge_review_flags
from calibration.logprob_confidence import (
    ConfidenceError,
    aggregate_span,
    field_confidences,
    overall_confidence,
    unusable_fields,
)
from evaluation.gating import GATING_METRICS, promotion_gate
from evaluation.metrics.confusable import score_alias_accuracy, score_misattribution
from evaluation.metrics.coverage_metrics import (
    expected_calibration_error,
    score_by_mode,
    score_lob,
)
from evaluation.metrics.field_accuracy import score_fields, score_list_field
from inference_core.span_map import FieldSpan

# --------------------------------------------------------------------------
# Field accuracy
# --------------------------------------------------------------------------

def test_normalized_match_beats_string_equality():
    """Insurance fields need it or correct extractions are under-counted."""
    report = score_fields(
        {"effective_date": "2026-04-01", "total_premium": 47250.0},
        {"effective_date": "04/01/2026", "total_premium": "$47,250.00"},
    )
    assert report.normalized_match == 1.0
    assert report.exact_match == 0.0        # neither is string-identical


def test_per_field_accuracy_shows_which_field_fails():
    report = score_fields(
        {"insured_name": "Rivera", "policy_number": "WC-1"},
        {"insured_name": "Rivera", "policy_number": "WC-2"},
    )
    by_field = report.by_field()
    assert by_field["insured_name"] == 1.0
    assert by_field["policy_number"] == 0.0


def test_list_recall_catches_a_dropped_row():
    """A missing row generates no tokens, so recall is the only signal that
    sees it (arch §5)."""
    expected = [{"claim_number": f"C{i}", "paid": i * 100} for i in range(1, 9)]
    got = expected[:6]                       # two claims silently dropped

    report = score_list_field(expected, got, "claims")
    assert report.recall == pytest.approx(6 / 8)
    assert report.precision == 1.0           # every row it DID produce is correct
    assert report.missed_rows == 2


def test_rows_are_matched_by_identity_not_position():
    """Dropping row 2 must not mis-align every row after it."""
    expected = [{"claim_number": f"C{i}"} for i in range(1, 6)]
    got = [row for row in expected if row["claim_number"] != "C2"]
    assert score_list_field(expected, got, "claims").matched_rows == 4


def test_a_missing_list_field_scores_zero_recall():
    from evaluation.metrics.field_accuracy import score_all_list_fields

    reports = score_all_list_fields({"claims": [{"claim_number": "C1"}]}, {})
    assert reports["claims"].recall == 0.0


# --------------------------------------------------------------------------
# Confusable misattribution — the gating metric
# --------------------------------------------------------------------------

def test_misattribution_detects_a_stolen_value():
    """The failure that is fluent, schema-valid, confidently wrong, and caught by
    nothing else in the pipeline."""
    expected = {
        "insured_name": "Rivera Fabrication LLC",
        "certificate_holder": "Meridian Property Group LLC",
    }
    got = {
        "insured_name": "Meridian Property Group LLC",   # the certificate holder
        "certificate_holder": "Meridian Property Group LLC",
    }
    report = score_misattribution(expected, got, "policy", source_id="policy_0001")
    assert report.cases
    case = report.cases[0]
    assert case.field_path == "insured_name"
    assert case.stolen_from == "certificate_holder"


def test_an_ordinary_wrong_value_is_not_misattribution():
    """Only a value that is right for ANOTHER field counts — a merely wrong
    value is ordinary field error with a different fix."""
    expected = {"insured_name": "Rivera Fabrication LLC", "producer": "Hanover"}
    got = {"insured_name": "Completely Unrelated Co", "producer": "Hanover"}
    assert not score_misattribution(expected, got, "policy").cases


def test_a_correct_extraction_has_no_misattribution():
    expected = {"insured_name": "Rivera", "certificate_holder": "Meridian"}
    assert not score_misattribution(expected, dict(expected), "policy").cases


# --------------------------------------------------------------------------
# Alias accuracy — reported, not gating
# --------------------------------------------------------------------------

def test_alias_accuracy_names_the_weak_variant():
    """Turns 'field accuracy is 0.87' into 'we are weak on Applicant'."""
    documents = []
    for _ in range(10):                       # strong on the dominant label
        documents.append((
            {"insured_name": "Rivera"}, {"insured_name": "Rivera"},
            {"insured_name": "Named Insured"},
        ))
    for i in range(4):                        # weak on the rare one
        documents.append((
            {"insured_name": "Rivera"}, {"insured_name": "Wrong" if i < 3 else "Rivera"},
            {"insured_name": "Applicant"},
        ))

    report = score_alias_accuracy(documents)
    assert report.accuracy("insured_name", "Named Insured") == 1.0
    assert report.accuracy("insured_name", "Applicant") == pytest.approx(0.25)

    weakest = report.weakest_variants()
    assert weakest[0][:2] == ("insured_name", "Applicant")
    assert report.spread("insured_name") == pytest.approx(0.75)


def test_alias_accuracy_is_not_a_gating_metric():
    """A rare variant has too little support for a stable threshold; gating on
    it would block good models on noise."""
    assert "alias_accuracy" not in GATING_METRICS
    assert "confusable_misattribution_rate" in GATING_METRICS


# --------------------------------------------------------------------------
# LoB, ECE, mode accuracy
# --------------------------------------------------------------------------

def test_lob_accuracy_is_reported_per_value():
    """A rare class must not hide inside a healthy aggregate (arch §0b)."""
    documents = [({"line_of_business": "workers_comp"}, {"line_of_business": "workers_comp"})] * 9
    documents.append(({"line_of_business": "umbrella"}, {"line_of_business": "property"}))

    report = score_lob(documents)
    assert report.overall == pytest.approx(0.9)
    assert report.accuracy_by_value()["workers_comp"] == 1.0
    assert report.accuracy_by_value()["umbrella"] == 0.0     # invisible in the aggregate


def test_unmeasured_lob_values_are_reported():
    """An unmeasured class is not a passing class."""
    report = score_lob([({"line_of_business": "workers_comp"}, {"line_of_business": "workers_comp"})])
    assert "umbrella" in report.unmeasured_values()


def test_ece_detects_overconfidence():
    """A model 95% confident and 50% correct is badly calibrated whatever its
    accuracy is."""
    overconfident = expected_calibration_error([0.95] * 10, [True] * 5 + [False] * 5)
    well_calibrated = expected_calibration_error([0.5] * 10, [True] * 5 + [False] * 5)
    assert overconfident > 0.4
    assert well_calibrated < 0.05


def test_mode_accuracy_carries_document_counts():
    """The ViT gate needs the counts: below its floor it declines to decide."""
    docs = [({"a": "x"}, {"a": "x"}, "image_only", False) for _ in range(3)]
    docs += [({"a": "x"}, {"a": "y"}, "ocr_plus_image", True) for _ in range(2)]

    report = score_by_mode(docs)
    assert report.accuracy_by_mode["image_only"] == 1.0
    assert report.documents_by_mode["image_only"] == 3
    assert report.scanned_documents == 2
    assert report.error_mix()                       # errors classified for the gate


# --------------------------------------------------------------------------
# The promotion gate
# --------------------------------------------------------------------------

def _metrics(**over) -> dict[str, float]:
    base = {name: 0.90 for name in GATING_METRICS}
    base["ece_confidence"] = 0.04                   # lower is better
    base["confusable_misattribution_rate"] = 0.02   # lower is better
    base.update(over)
    return base


def test_an_improved_candidate_passes():
    result = promotion_gate(_metrics(field_exact_match=0.92), _metrics())
    assert result.passed


def test_a_single_regression_blocks_promotion():
    """Any one metric. That is what 'every gating metric' means."""
    result = promotion_gate(_metrics(list_field_recall=0.85), _metrics())
    assert not result.passed
    assert "list_field_recall" in result.failed_gates


def test_lob_regression_alone_blocks_promotion():
    result = promotion_gate(_metrics(lob_detection_accuracy=0.80), _metrics())
    assert not result.passed


def test_confusable_regression_alone_blocks_promotion():
    """Rising misattribution is a regression even though the number goes UP."""
    result = promotion_gate(_metrics(confusable_misattribution_rate=0.09), _metrics())
    assert not result.passed
    assert "confusable_misattribution_rate" in result.failed_gates


def test_error_rate_metrics_are_scored_in_the_right_direction():
    """Treating ECE like accuracy would promote a model that got worse at
    exactly the thing hardest to notice."""
    improved = promotion_gate(_metrics(ece_confidence=0.02), _metrics())
    assert improved.passed


def test_an_unmeasured_metric_blocks_promotion():
    """A metric that was not measured has not passed."""
    candidate = _metrics()
    del candidate["image_only_accuracy"]
    result = promotion_gate(candidate, _metrics())
    assert not result.passed
    assert "image_only_accuracy" in result.failed_gates


def test_the_first_version_has_nothing_to_regress_against():
    result = promotion_gate(_metrics(), None)
    assert result.passed and result.is_first_version


def test_a_continued_foundation_needs_cross_type_evidence():
    """Continued training compounds drift, and a patch aimed at one type can
    quietly degrade the others (arch §12)."""
    result = promotion_gate(_metrics(), _metrics(), continued_from="foundation-v1")
    assert not result.passed
    assert "cross_type_regression_evidence" in result.failed_gates


def test_a_continued_foundation_passes_with_evidence():
    result = promotion_gate(
        _metrics(), _metrics(), continued_from="foundation-v1",
        cross_type_evidence={
            "acord": {"candidate": _metrics(), "current": _metrics()},
            "lossrun": {"candidate": _metrics(), "current": _metrics()},
        },
    )
    assert result.passed


def test_cross_type_regression_blocks_a_continued_foundation():
    result = promotion_gate(
        _metrics(), _metrics(), continued_from="foundation-v1",
        cross_type_evidence={"acord": {"candidate": _metrics(field_exact_match=0.70),
                                       "current": _metrics()}},
    )
    assert not result.passed
    assert any("acord" in gate for gate in result.failed_gates)


def test_the_gate_has_no_override_parameter():
    """A gate that can be waived is a suggestion, not a guarantee — and every
    other assurance in the pipeline rests on it."""
    import inspect

    signature = inspect.signature(promotion_gate)
    for forbidden in ("force", "override", "skip", "ignore_regressions"):
        assert forbidden not in signature.parameters


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------

def _span(path: str, value, logprobs: list[float], mapped: bool = True) -> FieldSpan:
    span = FieldSpan(field_path=path, value=value, char_start=0, char_end=1)
    span.token_logprobs = logprobs
    span.mapped = mapped
    if not mapped:
        span.reason = "no token span"
    return span


def test_min_aggregation_is_sensitive_to_the_weakest_token():
    """Nine confident tokens and one uncertain one is worth checking; mean would
    average that away."""
    logprobs = [-0.01] * 9 + [-2.5]
    assert aggregate_span(logprobs, "min") < 0.1
    assert aggregate_span(logprobs, "mean") > 0.8


def test_geomean_does_not_underflow_on_long_spans():
    """Multiplying probabilities directly would reach zero; log space does not."""
    assert aggregate_span([-3.0] * 200, "geomean") > 0.0


def test_unmapped_fields_get_no_confidence_and_are_surfaced():
    """A field with no confidence must not be indistinguishable from a confident
    one — that is backwards for a risk signal."""
    spans = {
        "carrier": _span("carrier", "Sentinel", [-0.01, -0.02]),
        "policy_number": _span("policy_number", "WC-1", [], mapped=False),
    }
    confidences = field_confidences(spans)
    assert confidences["carrier"].is_usable
    assert not confidences["policy_number"].is_usable
    assert unusable_fields(confidences) == [("policy_number", "no token span")]
    assert overall_confidence(confidences) > 0.9      # the unusable one is excluded


def test_empty_span_cannot_be_aggregated():
    with pytest.raises(ConfidenceError, match="empty logprob span"):
        aggregate_span([], "min")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_temperature_scaling_reduces_ece_on_an_overconfident_model():
    """The failure calibration exists for: a fine-tuned model whose confidence
    barely varies between right and wrong answers."""
    confidences = [0.95] * 60 + [0.92] * 40
    correctness = [True] * 45 + [False] * 15 + [True] * 20 + [False] * 20

    params = fit_calibration(confidences, correctness, doc_type="policy", model_version="v1")
    assert params.improved
    assert params.ece_after < params.ece_before
    assert params.temperature > 1.0             # the overconfident direction


def test_isotonic_also_reduces_ece():
    confidences = [0.9] * 50 + [0.6] * 50
    correctness = [True] * 30 + [False] * 20 + [True] * 40 + [False] * 10

    params = fit_calibration(
        confidences, correctness, doc_type="policy", model_version="v1", method="isotonic"
    )
    assert params.ece_after <= params.ece_before
    assert params.breakpoints


def test_isotonic_breakpoints_are_monotonic():
    """Confidence that falls as raw confidence rises would be uninterpretable."""
    params = fit_calibration(
        [i / 100 for i in range(100)], [i % 3 != 0 for i in range(100)],
        doc_type="policy", model_version="v1", method="isotonic",
    )
    values = [y for _x, y in params.breakpoints]
    assert values == sorted(values)


def test_fitting_on_training_data_is_refused():
    """It measures memorisation, looks excellent, and fails in production."""
    with pytest.raises(CalibrationError, match="BOTH"):
        assert_held_out({"policy_0001", "policy_0002"}, {"policy_0002", "policy_0003"})


def test_missing_calibration_raises_rather_than_passing_raw_through():
    """THE guard. Raw confidence from a fine-tuned model is systematically
    overconfident; returning it unmarked would under-route review."""
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    with pytest.raises(CalibrationError, match="no calibration exists"):
        load_calibration("v2", "policy", client)


def test_calibration_round_trips_through_blob():
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    params = CalibrationParams(method="temperature", doc_type="policy",
                               model_version="v2", temperature=1.8)
    client.write_json(paths.calibration_params("v2", "policy"), params.as_dict())

    loaded = load_calibration("v2", "policy", client)
    assert loaded.temperature == 1.8 and loaded.method == "temperature"


def test_calibration_flags_fields_below_the_review_threshold():
    spans = {
        "carrier": _span("carrier", "Sentinel", [-0.001]),
        "valuation_date": _span("valuation_date", "2026-03-31", [-3.0]),
    }
    params = CalibrationParams(method="temperature", doc_type="lossrun",
                               model_version="v1", temperature=1.0)

    result = apply_calibration(field_confidences(spans), params, review_threshold=0.70)
    assert not result.fields["carrier"].needs_review
    assert result.fields["valuation_date"].needs_review
    assert "valuation_date:low_confidence" in result.review_flags

    output = calibrated_output(result)
    assert set(output["fields"]["carrier"]) == {"value", "confidence"}


def test_a_field_without_confidence_is_routed_to_review():
    spans = {"policy_number": _span("policy_number", "WC-1", [], mapped=False)}
    params = CalibrationParams(method="temperature", doc_type="policy",
                               model_version="v1", temperature=1.0)
    result = apply_calibration(field_confidences(spans), params)
    assert result.fields["policy_number"].needs_review
    assert "policy_number:no_confidence" in result.review_flags


# --------------------------------------------------------------------------
# List completeness — what logprobs cannot see
# --------------------------------------------------------------------------

def test_a_stated_count_mismatch_flags_the_list():
    """6 of 8 claims: the 2 missing rows produce no tokens, so nothing else
    catches this."""
    signal = check_completeness("claims", extracted_rows=6, stated_count=8)
    assert signal.flagged
    assert signal.missing_rows == 2
    assert signal.confidence < 1.0


def test_a_detected_row_mismatch_flags_the_list_independently():
    """The second cross-check, from MinerU's table row counts."""
    signal = check_completeness("claims", extracted_rows=6, detected_rows=9)
    assert signal.flagged


def test_structure_count_tolerates_a_small_difference():
    """A continuation header or totals row can shift the detected count by one;
    the document's stated count is an assertion and is compared exactly."""
    assert not check_completeness("claims", extracted_rows=8, detected_rows=9).flagged
    assert check_completeness("claims", extracted_rows=8, stated_count=9).flagged


def test_no_cross_check_available_is_not_the_same_as_verified():
    signal = check_completeness("claims", extracted_rows=6)
    assert not signal.flagged
    assert signal.confidence == 0.5             # unverified, not confirmed
    assert "could not be verified" in signal.reasons[0]


def test_a_complete_list_is_not_flagged():
    assert not check_completeness("claims", 8, stated_count=8, detected_rows=8).flagged


def test_document_check_uses_ocr_row_counts_for_the_pages_read():
    """A page-routed long document is compared against the pages it actually
    read, not the whole file."""
    extraction = {"claims": [{"claim_number": f"C{i}"} for i in range(6)],
                  "total_claims_reported": 8}
    signals = check_document(
        extraction, "lossrun",
        ocr_meta={"table_row_counts": {"1": 5, "2": 4, "3": 99}},
        pages_used=[1, 2],
    )
    assert signals["claims"].flagged
    assert signals["claims"].detected_rows == 9      # page 3 excluded


def test_a_flagged_list_reaches_review_regardless_of_value_confidence():
    """The values present may be perfect while the list is incomplete."""
    signals = {"claims": check_completeness("claims", 6, stated_count=8)}
    flags = merge_review_flags(signals, existing_flags=[])
    assert "claims:row_count_mismatch" in flags


def test_a_hallucinated_list_is_not_maximally_confident():
    """The `or` chain read a stated count of 0 as absent and fell through to the
    extracted count itself, so the pure-hallucination case scored 1.0 — in a
    value that ships in the output contract."""
    from calibration.list_completeness import check_completeness

    signal = check_completeness("claims", 4, stated_count=0)
    assert signal.flagged
    assert signal.confidence == 0.0


def test_over_extraction_is_scored_like_under_extraction():
    """`min(1.0, extracted/reference)` reported a surplus as perfect."""
    from calibration.list_completeness import check_completeness

    short = check_completeness("claims", 4, stated_count=5)
    surplus = check_completeness("claims", 6, stated_count=5)
    assert short.flagged and surplus.flagged
    assert surplus.confidence < 1.0


def test_a_perfect_list_is_fully_confident():
    from calibration.list_completeness import check_completeness

    signal = check_completeness("claims", 4, stated_count=4)
    assert not signal.flagged and signal.confidence == 1.0


def test_a_coincidental_value_collision_is_not_misattribution():
    """The registry filter was computed and never applied, so unrelated
    duplicate values entered a gating metric and could block a promotion."""
    from evaluation.metrics.confusable import score_misattribution

    expected = {"insured_name": "Rivera Fabrication LLC", "producer": "Hanover Risk Partners"}
    got = {"insured_name": "Rivera Fabrication LLC", "producer": "Hanover Risk Partners"}
    assert score_misattribution(expected, got, "policy", source_id="policy_0001").rate == 0.0


def test_the_lob_balance_target_is_reachable():
    """Five values against a 0.20 floor sum to exactly 1.00, so any corpus that
    was not perfectly uniform failed — the flag carried no information."""
    from common.lob import compute_coverage

    coverage = compute_coverage(
        ["workers_comp"] * 21 + ["general_liability"] * 20 + ["commercial_auto"] * 20
        + ["property"] * 20 + ["umbrella"] * 19
    )
    assert coverage.is_balanced, coverage.warning()
