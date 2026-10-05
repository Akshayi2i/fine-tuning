"""IMPL-04 §1 and §6 — pre-annotation and the confidence-routed review queue.

Two guards carry this file. The **external pre-annotation refusal** is a
compliance boundary: it must refuse, not warn, because a warning is dismissed
once and the disclosure it warned about cannot be taken back. The **day-zero
routing refusal** is the same shape in a different place: routing by confidence
before a model has earned trust sends its own mistakes past a reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.list_completeness import ROW_MISMATCH_SUFFIX
from data_pipeline.labeling import active_learning, pre_annotate
from data_pipeline.labeling.active_learning import (
    SPOT_CHECK_THRESHOLD,
    ActiveLearningError,
    assert_routing_permitted,
    build_queue,
)
from data_pipeline.labeling.pre_annotate import (
    DEFAULT_BACKEND,
    Draft,
    ExternalBackendRefused,
    PreAnnotationError,
    assert_external_allowed,
    resolve_backend_model,
)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def no_external(monkeypatch):
    """Neither permission present — the default posture."""
    monkeypatch.setattr(pre_annotate, "env", lambda name, default=None: default)


def seed_labels(client: BlobClient, doc_type: str, count: int) -> None:
    for i in range(1, count + 1):
        client.write_json(paths.golden_label(doc_type, f"{doc_type}_{i:04d}"),
                          {"line_of_business": ["workers_comp"]})


# --------------------------------------------------------------------------
# The external-backend refusal
# --------------------------------------------------------------------------


def test_external_is_refused_without_the_flag(no_external):
    with pytest.raises(ExternalBackendRefused, match="--allow-external"):
        assert_external_allowed(allow_external=False)


def test_external_is_refused_without_the_env_permission(no_external):
    """The flag alone is one person's decision; the env var is the deployment's."""
    with pytest.raises(ExternalBackendRefused, match="ALLOW_EXTERNAL_PREANNOTATION"):
        assert_external_allowed(allow_external=True)


def test_external_is_refused_without_a_zero_retention_endpoint(monkeypatch):
    """A default consumer endpoint may retain prompts, which is the exact thing
    the guard exists to prevent."""
    monkeypatch.setattr(
        pre_annotate, "env",
        lambda name, default=None: "true" if name == "ALLOW_EXTERNAL_PREANNOTATION" else None,
    )
    with pytest.raises(ExternalBackendRefused, match="zero-retention"):
        assert_external_allowed(allow_external=True)


def test_external_is_allowed_only_with_both_permissions_and_an_endpoint(monkeypatch):
    monkeypatch.setattr(
        pre_annotate, "env",
        lambda name, default=None: {
            "ALLOW_EXTERNAL_PREANNOTATION": "true",
            "EXTERNAL_PREANNOTATION_ENDPOINT": "https://zero-retention.example/v1",
        }.get(name, default),
    )
    assert_external_allowed(allow_external=True)  # does not raise


def test_the_refusal_is_a_distinct_error_type():
    """A batch loop swallows one unreadable PDF; it must not swallow a
    compliance refusal, which applies to every document in the batch."""
    assert issubclass(ExternalBackendRefused, PreAnnotationError)


def test_a_refusal_stops_the_whole_batch(client, no_external):
    with pytest.raises(ExternalBackendRefused):
        pre_annotate.pre_annotate_batch(
            ["policy_0001", "policy_0002"], "policy", client,
            backend="external_frontier", allow_external=False,
        )


def test_the_self_hosted_backend_needs_no_permission():
    assert DEFAULT_BACKEND == "base_qwen3vl"
    assert resolve_backend_model("base_qwen3vl", client=None) == "base"  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Backend resolution
# --------------------------------------------------------------------------


def test_own_finetuned_refuses_before_a_foundation_is_promoted(client):
    """The day-zero case, which is why the self-hosted base is the default."""
    with pytest.raises(PreAnnotationError, match="day-zero"):
        resolve_backend_model("own_finetuned", client)


def test_an_unknown_backend_is_refused(client):
    with pytest.raises(PreAnnotationError, match="unknown backend"):
        pre_annotate.pre_annotate("policy_0001", "policy", client, backend="gpt")  # type: ignore[arg-type]


def test_an_undocumented_document_cannot_be_drafted(client):
    with pytest.raises(PreAnnotationError, match="not been OCR"):
        pre_annotate.pre_annotate("policy_0001", "policy", client)


# --------------------------------------------------------------------------
# The draft itself
# --------------------------------------------------------------------------


def test_a_draft_is_never_trusted():
    draft = Draft(source_id="p1", doc_type="policy", backend="base_qwen3vl", model_version="base")
    assert draft.trusted is False
    assert draft.as_dict()["trusted"] is False


def test_a_draft_is_not_written_where_a_golden_label_lives(client):
    """One careless glob away from entering a corpus unreviewed."""
    draft = Draft(source_id="p1", doc_type="policy", backend="base_qwen3vl", model_version="base")
    key = pre_annotate.write_draft(draft, client)

    assert key != paths.golden_label("policy", "p1")
    assert key.endswith("draft.json")
    assert not client.exists(paths.golden_label("policy", "p1"))


# --------------------------------------------------------------------------
# Confidence routing — the day-zero refusal
# --------------------------------------------------------------------------


def test_routing_is_refused_below_the_day_zero_threshold(client):
    seed_labels(client, "policy", 3)
    with pytest.raises(ActiveLearningError, match="day-zero"):
        assert_routing_permitted(client, ["policy"])


def test_routing_is_refused_when_any_single_type_is_thin(client):
    """Routing ACORD by confidence while ACORD has four labels is the mistake,
    even if Policy has three hundred."""
    seed_labels(client, "policy", 40)
    seed_labels(client, "acord", 4)
    with pytest.raises(ActiveLearningError, match="acord"):
        assert_routing_permitted(client, ["policy", "acord"])


def test_routing_is_permitted_once_every_type_clears_the_threshold(client):
    for doc_type in ("policy", "acord", "lossrun"):
        seed_labels(client, doc_type, 25)
    assert_routing_permitted(client, ["policy", "acord", "lossrun"])  # does not raise


# --------------------------------------------------------------------------
# Queue ordering
# --------------------------------------------------------------------------


@dataclass
class _Result:
    """The parts of an ExtractionResult the queue reads."""

    source_id: str
    doc_type: str = "policy"
    overall_confidence: float = 0.5
    fields: dict[str, Any] = field(default_factory=dict)
    review_flags: list[str] = field(default_factory=list)


def _fields(**scores: float) -> dict[str, Any]:
    return {name: {"value": "x", "confidence": score} for name, score in scores.items()}


def test_the_queue_is_ordered_by_ascending_confidence():
    """The reviewer's first hour goes where it changes the most."""
    queue = build_queue([
        _Result("a", overall_confidence=0.95, fields=_fields(insured_name=0.95)),
        _Result("b", overall_confidence=0.30, fields=_fields(insured_name=0.30)),
        _Result("c", overall_confidence=0.60, fields=_fields(insured_name=0.60)),
    ])
    assert [item.source_id for item in queue.items] == ["b", "c", "a"]


def test_row_completeness_outranks_every_confidence_score():
    """A missing row has no confidence to be low — its absence is the problem,
    and no per-value number can express it."""
    queue = build_queue([
        _Result("low", overall_confidence=0.20, fields=_fields(insured_name=0.20)),
        _Result("incomplete", overall_confidence=0.99, fields=_fields(insured_name=0.99),
                # The flag string the pipeline ACTUALLY emits, taken from the
                # module that emits it. This test used to hand-write
                # "list:claims_row_count_mismatch" — a string nothing in the
                # codebase produces — so it passed while the override it guards
                # was dead code for every real document.
                review_flags=[f"claims{ROW_MISMATCH_SUFFIX}"]),
    ])
    assert queue.items[0].source_id == "incomplete"
    assert queue.items[0].routing == "full_review"


def test_the_queue_reads_the_flags_list_completeness_actually_emits():
    """A seam test. Both sides of this contract were written independently and
    disagreed in silence: the emitter used '{field}:row_count_mismatch', the
    consumer matched the prefixes 'list:'/'rows:'/'completeness:'. Asserting the
    real emitted string is what keeps them from drifting apart again."""
    from calibration.list_completeness import check_completeness, merge_review_flags

    signal = check_completeness("claims", 6, stated_count=8)
    assert signal.flagged, "a 6-of-8 list should be flagged"
    emitted = merge_review_flags({"claims": signal}, [])
    assert emitted, "list_completeness emitted no flag for a flagged list"

    queue = build_queue([
        _Result("incomplete", overall_confidence=0.99,
                fields=_fields(insured_name=0.99), review_flags=emitted),
    ])
    assert queue.items[0].completeness_flags == sorted(emitted)
    assert queue.items[0].routing == "full_review"


def test_a_confident_document_gets_only_a_spot_check():
    queue = build_queue([
        _Result("a", overall_confidence=0.97, fields=_fields(insured_name=0.98, policy_number=0.96)),
    ])
    assert queue.items[0].routing == "spot_check"
    assert queue.spot_check_share == 1.0


def test_low_confidence_is_reported_per_field_not_per_document():
    """Flagging a whole 40-row Loss Run because one date was uncertain wastes
    the reviewer on 39 correct rows."""
    queue = build_queue([
        _Result("a", overall_confidence=0.80,
                fields=_fields(insured_name=0.99, effective_date=0.31, policy_number=0.97)),
    ])
    item = queue.items[0]
    assert item.routing == "field_review"
    assert [name for name, _score in item.low_confidence_fields] == ["effective_date"]


def test_a_field_with_no_confidence_is_not_a_confident_field():
    """It means the span could not be mapped — a reason to look, not to skip."""
    queue = build_queue([
        _Result("a", overall_confidence=0.9, fields={"insured_name": {"value": "x"}}),
    ])
    assert queue.items[0].low_confidence_fields == [("insured_name", 0.0)]


def test_the_spot_check_share_is_the_number_that_shows_cost_falling():
    queue = build_queue([
        _Result("a", overall_confidence=0.99, fields=_fields(x=0.99)),
        _Result("b", overall_confidence=0.99, fields=_fields(x=0.99)),
        _Result("c", overall_confidence=0.20, fields=_fields(x=0.20)),
        _Result("d", overall_confidence=0.20, fields=_fields(x=0.20)),
    ])
    assert queue.spot_check_share == 0.5
    assert queue.by_routing == {"field_review": 2, "spot_check": 2}


def test_the_spot_check_threshold_is_above_the_review_threshold():
    """A document only needs a light touch when it is well clear of the line
    that would have sent individual fields to review."""
    from common.constants import DEFAULT_REVIEW_CONFIDENCE_THRESHOLD

    assert SPOT_CHECK_THRESHOLD > DEFAULT_REVIEW_CONFIDENCE_THRESHOLD


def test_an_empty_queue_reports_no_saving_rather_than_dividing_by_zero():
    assert build_queue([]).spot_check_share == 0.0


def test_the_queue_is_written_where_reviewers_look(client):
    queue = build_queue([_Result("a", fields=_fields(x=0.9))], model_version="v2")
    key = active_learning.write_queue(queue, client)
    assert key.endswith("v2.json")
    assert client.read_json(key)["documents"] == 1


def test_the_documented_env_flag_is_actually_read():
    """It was in .env.example and referenced in no module — a guard that existed
    only in documentation. `test_repo_contracts` now catches the general case;
    this pins the specific one that bit."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    source = (root / "data_pipeline" / "labeling" / "pre_annotate.py").read_text(encoding="utf-8")
    template = (root / ".env.example").read_text(encoding="utf-8")
    assert "ALLOW_EXTERNAL_PREANNOTATION" in source
    assert "ALLOW_EXTERNAL_PREANNOTATION" in template
