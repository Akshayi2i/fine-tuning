"""SPEC_10 + SPEC_11 + SPEC_12 — routing, page handling, the pipeline, and parity.

The parity test at the end is one of the four that guard silent failures: if the
testing harness ever stops calling the serving pipeline, its numbers describe a
system that is not the one in production, and no other test would notice.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.fit_calibration import CalibrationParams
from inference_core.model_runner import EchoBackend, load_model
from postprocessing.merge_adapter import plan_merge
from postprocessing.quantize import QuantizationError, plan_quantization, quantize
from serving.adapter_router import route
from serving.doc_type_classifier import (
    Classification,
    ClassificationError,
    StaticClassifier,
    parse_classification,
)
from serving.page_router import merge_page_extractions, plan_pages
from serving.pipeline import ExtractionRequest, PipelineError, extract
from testing.run_extraction import run_document, summarise

GOLDEN = {
    "insured_name": "Rivera Fabrication LLC",
    "policy_number": "WC-8842317-01",
    "line_of_business": ["workers_comp"],
    "effective_date": "2026-04-01",
}
RESPONSE = json.dumps(GOLDEN)
CALIBRATION = CalibrationParams(
    method="temperature", doc_type="policy", model_version="v1", temperature=1.0
)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def model(client):
    return load_model("base", client, backend_impl=EchoBackend(RESPONSE))


def _request(**over) -> ExtractionRequest:
    base = dict(
        source_id="policy_0001",
        image_paths=["processed/default/policy/policy_0001/page_1.png"],
        ocr_text="**Applicant** Rivera Fabrication LLC",
        known_doc_type="policy",
    )
    base.update(over)
    return ExtractionRequest(**base)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def test_classifier_response_is_parsed():
    result = parse_classification('{"doc_type":"lossrun","acord_form":null,"confidence":0.93}')
    assert result.doc_type == "lossrun" and result.confidence == 0.93


def test_fenced_json_is_recovered():
    """A parse failure would route a classifiable document to review for no
    reason."""
    result = parse_classification('```json\n{"doc_type":"policy","confidence":0.8}\n```')
    assert result.doc_type == "policy"


def test_unknown_doc_type_is_discarded():
    assert parse_classification('{"doc_type":"invoice","confidence":0.99}').doc_type is None


def test_acord_without_a_form_is_unusable_however_confident():
    """No form means no schema can be selected — confidence is irrelevant."""
    result = parse_classification('{"doc_type":"acord","acord_form":null,"confidence":0.99}')
    assert result.confidence == 0.0


def test_unparseable_response_raises():
    with pytest.raises(ClassificationError, match="not JSON"):
        parse_classification("I think this is a policy document.")


# --------------------------------------------------------------------------
# Routing — the low-confidence fallback
# --------------------------------------------------------------------------

def test_confident_classification_selects_the_adapter():
    result = route(
        Classification("lossrun", None, 0.95),
        adapter_map={"lossrun": "adapters/lossrun/v2"},
    )
    assert result.adapter == "adapters/lossrun/v2"
    assert not result.foundation_only
    assert not result.review_flags


def test_low_confidence_falls_back_to_foundation_only():
    """A wrong-adapter extraction is worse than a slightly generic one: the
    generic one is less sharp, the wrong one confidently answers the wrong
    question (arch §4a)."""
    result = route(
        Classification("lossrun", None, 0.40),
        confidence_threshold=0.70,
        adapter_map={"lossrun": "adapters/lossrun/v2"},
    )
    assert result.adapter is None
    assert result.foundation_only
    assert "routing:low_confidence" in result.review_flags
    assert result.needs_routing_review


def test_low_confidence_still_keeps_the_best_guess_schema():
    """A generic extraction validated against a plausible schema beats one
    validated against nothing."""
    result = route(Classification("lossrun", None, 0.40), confidence_threshold=0.70)
    assert result.schema_doc_type == "lossrun"


def test_unclassifiable_document_gets_a_fallback_and_a_flag():
    result = route(Classification(None, None, 0.0))
    assert result.foundation_only
    assert "routing:unclassified" in result.review_flags


def test_missing_adapter_serves_foundation_only_without_failing():
    """During the pilot the Foundation may be the only model trained."""
    result = route(Classification("policy", None, 0.99), adapter_map={})
    assert result.foundation_only
    assert "routing:no_adapter_available" in result.review_flags


def test_acord_form_reaches_the_schema_selection():
    result = route(Classification("acord", "25", 0.95), adapter_map={"acord": "a/25"})
    assert result.schema_acord_form == "25"


# --------------------------------------------------------------------------
# Page routing
# --------------------------------------------------------------------------

def test_short_documents_are_not_routed():
    """ACORD forms and most Loss Runs are a few pages; routing them would add
    latency and a failure mode for nothing."""
    plan = plan_pages({1: "text", 2: "text"}, page_threshold=5)
    assert not plan.routed and plan.pages == [1, 2]


def test_long_documents_select_the_relevant_pages():
    pages = {i: "boilerplate exclusions and definitions" for i in range(1, 21)}
    pages[3] = "COMMON POLICY DECLARATIONS named insured policy number policy period"
    pages[7] = "COVERAGE SCHEDULE limits of insurance premium"

    plan = plan_pages(pages, page_threshold=5)
    assert plan.routed
    assert 3 in plan.pages and 7 in plan.pages
    assert plan.declarations_page == 3
    assert len(plan.pages) < len(pages)


def test_the_first_page_is_always_read():
    """A poorly-OCR'd scan can score zero while still being the declarations
    page."""
    pages = {i: "" for i in range(1, 11)}
    pages[6] = "declarations named insured"
    assert 1 in plan_pages(pages, page_threshold=5).pages


def test_no_matching_pages_falls_back_to_reading_everything():
    """An empty selection means the heuristic failed, not that the document is
    empty."""
    plan = plan_pages({i: "unrecognisable layout" for i in range(1, 11)}, page_threshold=5)
    assert not plan.routed
    assert len(plan.pages) == 10


def test_declarations_page_wins_policy_level_conflicts():
    """arch §7 — a schedule page can repeat a policy number differently; the
    declarations page is the authority."""
    merged = merge_page_extractions(
        {3: {"policy_number": "WC-8842317-01"}, 7: {"policy_number": "WC-8842317"}},
        declarations_page=3,
    )
    assert merged.extraction["policy_number"] == "WC-8842317-01"
    assert merged.conflicts


def test_list_fields_concatenate_across_pages():
    """A claims table spanning three pages must produce all its rows —
    overwriting would silently drop two thirds of them."""
    merged = merge_page_extractions({
        1: {"claims": [{"claim_number": "C1"}]},
        2: {"claims": [{"claim_number": "C2"}, {"claim_number": "C3"}]},
    })
    assert len(merged.extraction["claims"]) == 3


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

def test_pipeline_produces_the_output_contract(model):
    result = extract(_request(), model, StaticClassifier("policy"), CALIBRATION)
    assert result.schema_valid
    assert result.doc_type == "policy"
    # A list under arch v2.1 §0b. The span mapper emits one span per value —
    # each has its own tokens and its own logprob — and the pipeline collapses
    # them into one field carrying the weakest value's confidence.
    assert result.line_of_business["value"] == ["workers_comp"]
    assert 0.0 <= result.line_of_business["confidence"] <= 1.0
    assert "insured_name" in result.fields
    assert set(result.fields["insured_name"]) == {"value", "confidence"}
    assert result.pages_used


def test_schema_invalid_output_is_rejected_not_returned(client):
    """Mirrors the Fideon SPEC_07 Stage 3 audit gate — an invalid response is an
    error, not a result."""
    bad = load_model("base", client, backend_impl=EchoBackend('{"insured_name":"X"}'))
    with pytest.raises(PipelineError, match="failed schema validation"):
        extract(_request(), bad, StaticClassifier("policy"), CALIBRATION)


def test_unparseable_generation_fails_loudly(client):
    bad = load_model("base", client, backend_impl=EchoBackend("here is the policy data"))
    with pytest.raises(PipelineError, match="parseable JSON"):
        extract(_request(), bad, StaticClassifier("policy"), CALIBRATION)


def test_low_confidence_routing_reaches_the_result(model):
    result = extract(
        _request(known_doc_type=None), model,
        StaticClassifier("policy", confidence=0.3), CALIBRATION,
    )
    assert result.route_info["foundation_only"]
    assert "routing:low_confidence" in result.review_flags


def test_image_only_mode_runs_without_ocr(model):
    result = extract(
        _request(ocr_text=None, modality_mode="image_only"),
        model, StaticClassifier("policy"), CALIBRATION,
    )
    assert result.mode == "image_only"
    assert result.schema_valid


def test_list_completeness_flags_reach_the_result(client):
    """A dropped claim row is invisible to per-field confidence."""
    response = json.dumps({
        "carrier": "Sentinel", "policy_number": "WC-1", "valuation_date": "2026-03-31",
        "line_of_business": ["workers_comp"], "total_claims_reported": 8,
        "claims": [{"claim_number": f"C{i}", "loss_date": "2024-01-01", "status": "open"}
                   for i in range(6)],
    })
    lossrun_model = load_model("base", client, backend_impl=EchoBackend(response))
    result = extract(
        _request(source_id="lossrun_0001", known_doc_type="lossrun"),
        lossrun_model, StaticClassifier("lossrun"),
        CalibrationParams(method="temperature", doc_type="lossrun",
                          model_version="v1", temperature=1.0),
    )
    assert "claims:row_count_mismatch" in result.review_flags
    assert result.list_fields["claims"]["flagged"]


# --------------------------------------------------------------------------
# test == prod parity
# --------------------------------------------------------------------------

def test_the_harness_calls_the_serving_pipeline_not_a_copy(model):
    """THE parity test. Two extraction paths would be two systems, and the
    harness would be describing the wrong one."""
    direct = extract(_request(), model, StaticClassifier("policy"), CALIBRATION)
    via_harness, _metrics = run_document(
        _request(), model, StaticClassifier("policy"), CALIBRATION
    )
    assert via_harness.as_dict() == direct.as_dict()


def test_the_harness_does_not_reimplement_inference():
    """Enforced by inspection, not by discipline."""
    import inspect

    from testing import run_extraction

    source = inspect.getsource(run_extraction)
    assert "from serving.pipeline import" in source
    for forbidden in ("map_field_spans", "build_messages", "generate("):
        assert forbidden not in source, f"the harness reimplements {forbidden}"


def test_confidence_is_emitted_without_ground_truth(model):
    """The production case: on an unlabeled document confidence is still
    available to route review (arch §17)."""
    _result, metrics = run_document(_request(), model, StaticClassifier("policy"), CALIBRATION)
    assert metrics["overall_confidence"] > 0
    assert "not supplied" in metrics["ground_truth"]
    assert "field_normalized_match_rate" not in metrics


def test_ground_truth_adds_accuracy_metrics(model):
    _result, metrics = run_document(
        _request(), model, StaticClassifier("policy"), CALIBRATION, golden=GOLDEN
    )
    assert metrics["field_normalized_match_rate"] == 1.0
    assert all(f["correct"] for f in metrics["fields"].values())


def test_summary_aggregates_across_a_batch(model):
    results = [
        run_document(_request(source_id=f"policy_{i:04d}"), model,
                     StaticClassifier("policy"), CALIBRATION, golden=GOLDEN)
        for i in range(3)
    ]
    summary = summarise(results, "v1")
    assert summary.documents == 3
    assert summary.schema_validity_rate == 1.0
    assert summary.metrics["mean_field_normalized_match"] == 1.0


# --------------------------------------------------------------------------
# Merge and quantize
# --------------------------------------------------------------------------

def test_merge_plans_target_the_staging_volume():
    """The merged model is ~16GB and quantization also runs on RunPod — pushing
    it to Azure and back is a 32GB round trip for nothing."""
    plan = plan_merge(base_model="Qwen/Qwen3-VL-8B-Instruct",
                      foundation_version="v2", out_version="v2")
    assert plan.output_dir.startswith("/runpod-volume/")
    assert plan.is_unified


def test_quantization_defaults_to_baseline_plus_one_serving_format():
    """Producing all six every cycle wastes eval compute on formats nobody
    deploys (arch §13a)."""
    plan = plan_quantization(version="v2")
    assert plan.formats == ["fp16", "q5_k_m"]


def test_quantization_refuses_an_unverified_multimodal_projector():
    """A GGUF without an mmproj loads and cannot see — it would fail silently on
    every image-only document."""
    plan = plan_quantization(version="v2", formats=["q4_k_m"])
    with pytest.raises(QuantizationError, match="mmproj"):
        quantize(plan, dry_run=True)


def test_quantization_proceeds_once_the_projector_is_verified():
    plan = plan_quantization(version="v2", formats=["fp16", "q4_k_m"], mmproj_verified=True)
    outputs = quantize(plan, dry_run=True)
    assert set(outputs) == {"fp16", "q4_k_m"}
    assert all("gguf" in path for path in outputs.values())


def test_unknown_quantization_format_is_refused():
    with pytest.raises(QuantizationError, match="unknown quantization format"):
        plan_quantization(version="v2", formats=["q3_k_s"])
