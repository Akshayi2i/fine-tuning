"""SPEC_04 §4 — the review interface.

The load-bearing property is consistency between reviewers. Deciding which
canonical field a value belongs to *is* the decision the model has to learn, so
if three reviewers define ``insured_name`` three ways, the disagreement does not
average out — it trains in as the answer.
"""

from __future__ import annotations

import pytest

from data_pipeline.labeling.review_tool import (
    build_labeling_config,
    completed_to_golden,
    double_annotation_sample,
    review_fields_for,
    task_from_document,
)
from data_pipeline.labeling.review_tool.tasks import ReviewToolError

PAGES = ["processed/default/policy/policy_0001/page_1.png"]


# --------------------------------------------------------------------------
# The form is derived from the schema
# --------------------------------------------------------------------------


def test_every_schema_field_appears_on_the_form():
    """A hand-maintained form drifts the moment a field is added, and reviewers
    keep filling a form the corpus builder no longer validates against."""
    from common.schemas import resolved_schema

    fields = {f.path for f in review_fields_for("policy")}
    assert fields == set(resolved_schema("policy")["properties"])


def test_each_field_carries_the_same_gloss_the_model_gets():
    """Reviewer and model work from one definition, or the corpus teaches one
    thing while the prompt asks for another."""
    from common.schemas import resolved_schema

    schema = resolved_schema("policy")["properties"]
    for f in review_fields_for("policy"):
        assert f.gloss == schema[f.path].get("description", "")
        assert f.gloss, f"{f.path} has no definition for the reviewer"


def test_line_of_business_is_mandatory_on_every_type():
    """arch §0b — it cannot be left unset, because unset is indistinguishable
    from undetermined downstream."""
    for doc_type, form in (("policy", None), ("lossrun", None), ("acord", "25")):
        lob = next(f for f in review_fields_for(doc_type, form) if f.path == "line_of_business")
        assert lob.required


def test_confusables_are_shown_as_a_negative_instruction():
    """The boundary reviewers get wrong most often, and the one the
    misattribution metric scores the model on later."""
    assert "NEVER take from:" in build_labeling_config("policy")


def test_the_form_shows_expected_surface_labels_to_the_human():
    """The opposite of the master §1.4 anti-pattern: a reviewer reading the alias
    list is what makes the corpus teach the mapping. Only a *runtime* lookup is
    forbidden."""
    assert "Commonly labelled:" in build_labeling_config("policy")


def test_list_fields_name_their_row_fields():
    fields = {f.path: f for f in review_fields_for("lossrun")}
    claims = fields["claims"]
    assert claims.is_list
    assert {f.path for f in claims.row_fields} >= {"claims[].claim_number", "claims[].loss_date"}


def test_enum_fields_offer_an_explicit_null_choice():
    """Undetermined is a decision the reviewer makes, not an empty box."""
    assert "null (undetermined)" in build_labeling_config("policy")


def test_every_field_has_a_provenance_input():
    """One extra field per value, and it is what makes per-alias diagnosis and
    alias derivation possible at all."""
    config = build_labeling_config("policy")
    for path in ("insured_name", "policy_number"):
        assert f'name="{path}__seen_as"' in config


def test_the_form_shows_the_pages_and_the_ocr_side_by_side():
    config = build_labeling_config("policy")
    assert "$page_images" in config and "$ocr_text" in config


# --------------------------------------------------------------------------
# Tasks in
# --------------------------------------------------------------------------


def test_a_task_carries_the_draft_for_correction():
    task = task_from_document(
        "policy_0001", "policy", PAGES, "# POLICY", draft={"insured_name": "Rivera"}
    )
    assert task.as_dict()["data"]["draft"] == {"insured_name": "Rivera"}


def test_a_task_without_page_images_is_refused():
    """A reviewer cannot verify a value against a document they cannot see, and
    the OCR text is the thing being checked."""
    with pytest.raises(ReviewToolError, match="no page images"):
        task_from_document("policy_0001", "policy", [], "# POLICY")


def test_an_acord_task_needs_its_form_to_have_a_field_list():
    with pytest.raises(ReviewToolError, match="no form number"):
        task_from_document("acord_0001", "acord", PAGES, "# ACORD")


# --------------------------------------------------------------------------
# Double annotation
# --------------------------------------------------------------------------


def test_the_double_annotation_sample_is_stable_as_the_batch_grows():
    """An agreement score over a set that changes between batches is not
    comparable across them — the same reason the corpus split assigns by hash."""
    small = double_annotation_sample([f"policy_{i:04d}" for i in range(1, 41)])
    large = double_annotation_sample([f"policy_{i:04d}" for i in range(1, 201)])
    assert set(small) <= set(large)


def test_the_sample_is_never_empty_when_a_rate_was_asked_for():
    """An agreement score over zero documents is not a measurement."""
    assert double_annotation_sample(["policy_0001", "policy_0002"], rate=0.01)


def test_a_zero_rate_selects_nobody():
    assert double_annotation_sample(["policy_0001"], rate=0.0) == []


def test_an_impossible_rate_is_refused():
    with pytest.raises(ReviewToolError, match=r"\[0, 1\]"):
        double_annotation_sample(["policy_0001"], rate=1.5)


# --------------------------------------------------------------------------
# Annotations out
# --------------------------------------------------------------------------


def test_provenance_survives_the_round_trip():
    """Without it there is only an aggregate, and an aggregate cannot say which
    phrasing the model is failing on."""
    source_id, label, provenance = completed_to_golden({
        "data": {"source_id": "policy_0001"},
        "result": {
            "insured_name": "Rivera Fabrication LLC",
            "insured_name__seen_as": "Applicant",
            "policy_number": "WC-8842317-01",
            "policy_number__seen_as": "Policy No.",
        },
    })
    assert source_id == "policy_0001"
    assert label == {"insured_name": "Rivera Fabrication LLC", "policy_number": "WC-8842317-01"}
    assert provenance == {"insured_name": "Applicant", "policy_number": "Policy No."}


def test_an_explicit_null_is_a_decision_not_an_empty_box():
    _sid, label, _p = completed_to_golden({
        "data": {"source_id": "policy_0001"},
        "result": {"line_of_business": "null (undetermined)"},
    })
    assert label == {"line_of_business": None}


def test_provenance_for_an_unfilled_field_is_dropped():
    """It describes nothing, and would be rejected downstream as naming an
    absent field."""
    _sid, label, provenance = completed_to_golden({
        "data": {"source_id": "policy_0001"},
        "result": {"carrier__seen_as": "Insurer"},
    })
    assert "carrier" not in provenance and "carrier" not in label


def test_an_annotation_without_a_source_id_is_refused():
    with pytest.raises(ReviewToolError, match="no source_id"):
        completed_to_golden({"data": {}, "result": {"insured_name": "X"}})


def test_the_adapter_writes_nothing_itself():
    """``export_golden_labels`` is the single place that decides whether a label
    may enter the corpus. Two gates disagree."""
    import inspect

    from data_pipeline.labeling.review_tool import tasks

    source = inspect.getsource(tasks)
    assert "write_json" not in source and "golden_label(" not in source
