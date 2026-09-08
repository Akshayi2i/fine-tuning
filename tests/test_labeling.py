"""SPEC_04 — golden-label admission, the day-zero rule, and alias derivation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.labeling import derive_aliases as DA
from data_pipeline.labeling.export_golden_labels import (
    LabelValidationError,
    export_golden_label,
    inter_annotator_agreement,
    list_labeled_source_ids,
    review_requirement,
    validate_golden_label,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _policy_label(**over):
    label = json.loads((FIXTURES / "golden/policy_0001.golden.json").read_text(encoding="utf-8"))
    label.update(over)
    return label


def _fixture_documents():
    docs = []
    for golden in sorted((FIXTURES / "golden").glob("*.golden.json")):
        source_id = golden.name.replace(".golden.json", "")
        ocr = list((FIXTURES / "ocr").glob(f"{source_id}_page_*.md"))
        if ocr:
            docs.append((
                source_id,
                json.loads(golden.read_text(encoding="utf-8")),
                ocr[0].read_text(encoding="utf-8"),
            ))
    return docs


# --------------------------------------------------------------------------
# Label admission
# --------------------------------------------------------------------------

def test_valid_label_passes():
    validate_golden_label(_policy_label(), "policy")


def test_label_without_line_of_business_is_rejected():
    """Required in every label, every type, even when null (arch §0b)."""
    label = _policy_label()
    del label["line_of_business"]
    with pytest.raises(LabelValidationError, match="line_of_business is missing"):
        validate_golden_label(label, "policy")


def test_out_of_enum_line_of_business_is_rejected():
    with pytest.raises(LabelValidationError, match="invalid line_of_business"):
        validate_golden_label(_policy_label(line_of_business="marine_cargo"), "policy")


def test_null_line_of_business_is_accepted():
    """Null means the document does not determine it — a correct answer."""
    validate_golden_label(_policy_label(line_of_business=None), "policy")


def test_acord_label_without_a_form_is_rejected():
    """The form selects the schema, which is why classification is two-level."""
    label = json.loads((FIXTURES / "golden/acord_0001.golden.json").read_text(encoding="utf-8"))
    with pytest.raises(LabelValidationError, match="acord_form"):
        validate_golden_label(label, "acord")


def test_provenance_naming_a_confusable_is_rejected():
    """The highest-value annotation check: this mistake teaches the model to
    conflate distinct parties, and it would train perfectly happily."""
    with pytest.raises(LabelValidationError, match="registered CONFUSABLE"):
        validate_golden_label(
            _policy_label(), "policy",
            field_provenance={"insured_name": "Certificate Holder"},
        )


def test_provenance_with_a_real_alias_is_accepted():
    validate_golden_label(
        _policy_label(), "policy", field_provenance={"insured_name": "Applicant"}
    )


def test_provenance_naming_an_absent_field_is_rejected():
    with pytest.raises(LabelValidationError, match="not in the label"):
        validate_golden_label(
            _policy_label(), "policy", field_provenance={"nonexistent_field": "Whatever"}
        )


# --------------------------------------------------------------------------
# Day-zero rule (arch §4c)
# --------------------------------------------------------------------------

def test_day_zero_requires_full_review_until_the_threshold(client):
    assert review_requirement(client, "policy") == "full"
    assert review_requirement(client, "policy", minimum=0) == "confidence_routed"


def test_accepting_a_draft_unreviewed_is_refused_on_day_zero(client):
    """Drafts at this stage come from the untuned base model; trusting one puts
    its errors straight into the training target."""
    with pytest.raises(LabelValidationError, match="every draft still requires full human"):
        export_golden_label(
            _policy_label(), "policy_0001", "policy", client,
            reviewer_id="alice", accepted_without_review=True,
        )


def test_export_writes_label_and_provenance(client):
    export_golden_label(
        _policy_label(), "policy_0001", "policy", client,
        reviewer_id="alice", draft_backend="base_qwen3vl",
        field_provenance={"insured_name": "Applicant"},
    )
    assert client.exists(paths.golden_label("policy", "policy_0001"))

    meta = client.read_json(paths.label_metadata("policy", "policy_0001"))
    assert meta["reviewer_id"] == "alice"
    assert meta["draft_backend"] == "base_qwen3vl"
    assert meta["field_provenance"]["insured_name"] == "Applicant"
    assert meta["review_requirement"] == "full"


def test_labeled_source_ids_are_listed(client):
    for source_id in ("policy_0001", "policy_0002"):
        export_golden_label(_policy_label(), source_id, "policy", client, reviewer_id="alice")
    assert list_labeled_source_ids(client, "policy") == ["policy_0001", "policy_0002"]


def test_inter_annotator_agreement_measures_labeling_noise():
    """Sets a realistic ceiling on model scores — some residual error at plateau
    is human disagreement, not model failure (arch §7)."""
    a = _policy_label()
    b = _policy_label(insured_name="Someone Else Entirely")
    agreement, disagreements = inter_annotator_agreement(a, b)
    assert agreement < 1.0
    assert "insured_name" in disagreements

    perfect, none = inter_annotator_agreement(a, _policy_label())
    assert perfect == 1.0 and not none


def test_agreement_uses_normalized_comparison():
    """`Acme Mfg LLC` and `ACME MANUFACTURING LLC` are the same answer."""
    a = _policy_label(insured_name="Acme Mfg LLC")
    b = _policy_label(insured_name="ACME MANUFACTURING LLC")
    agreement, disagreements = inter_annotator_agreement(a, b)
    assert "insured_name" not in disagreements
    assert agreement == 1.0


# --------------------------------------------------------------------------
# Alias derivation (SPEC_04 §3) — the headline capability
# --------------------------------------------------------------------------

def test_one_canonical_field_derives_its_several_surface_labels():
    """The whole point: the golden JSON gives the value, the OCR gives the text,
    and the label is whatever introduces that value on the page."""
    report = DA.derive_aliases(_fixture_documents())
    labels = {label.casefold() for label in report.aliases["insured_name"]}
    assert {"applicant", "named insured", "insured"} <= labels


def test_normalized_matching_anchors_dates_and_currency():
    """The golden JSON is canonical (2026-03-31) while the page is not
    (03/31/2026). Without the normalized pass nearly every date would be
    unresolved."""
    report = DA.derive_aliases(_fixture_documents())
    assert "Valued As Of" in report.aliases["valuation_date"]
    assert any("Premium" in label for label in report.aliases["total_premium"])


def test_table_cells_resolve_to_their_column_header():
    """In a multi-column data row the cell to the LEFT is a different field's
    value, not this field's label."""
    report = DA.derive_aliases(_fixture_documents())
    assert "Claim Number" in report.aliases["claims[].claim_number"]
    assert "Total Incurred" in report.aliases["claims[].total_incurred"]


def test_list_rows_keep_their_leaf_identity():
    """Collapsing to the array name would make every cell in a Loss Run table an
    alias for `claims`, which is noise rather than evidence."""
    report = DA.derive_aliases(_fixture_documents())
    assert "claims" not in report.aliases
    assert any(field.startswith("claims[].") for field in report.aliases)


def test_confusables_are_derived_from_co_occurrence():
    """Evidence beats guessing: these are the distractors the real documents
    actually contain."""
    report = DA.derive_aliases(_fixture_documents())
    confusable = {label.casefold() for label in report.confusables["insured_name"]}
    assert "certificate holder" in confusable
    assert "producer" in confusable or "PRODUCER".casefold() in confusable


def test_provenance_is_backfilled_without_reannotation():
    report = DA.derive_aliases(_fixture_documents())
    assert report.provenance["policy_0001"]["insured_name"] == "Applicant"
    assert report.provenance["policy_0002"]["insured_name"] == "Named Insured"


def test_values_absent_from_their_document_are_reported():
    """The free QA pass. line_of_business is inferred, never printed on a page —
    so it *should* appear here, and that is the correct behaviour."""
    report = DA.derive_aliases(_fixture_documents())
    unresolved_fields = {u["field"] for u in report.unresolved}
    assert "line_of_business" in unresolved_fields


def test_low_entropy_values_are_skipped_not_guessed():
    """A value of "CA" or "3" matches everywhere and produces noise."""
    report = DA.derive_aliases(_fixture_documents())
    assert report.skipped_low_entropy
    assert all(len(s["value"]) < DA.MIN_VALUE_CHARS for s in report.skipped_low_entropy)


def test_markdown_emphasis_is_not_part_of_the_label():
    assert DA.clean_label("**Named Insured:**") == "Named Insured"
    assert DA.clean_label("| Policy No. |") == "Policy No"
    assert DA.clean_label("Applicant .........") == "Applicant"


def test_proposal_carries_evidence_and_never_overwrites(tmp_path):
    """High-quality candidate generation, not ground truth. A human confirms."""
    report = DA.derive_aliases(_fixture_documents())
    out = tmp_path / "policy.aliases.proposed.json"
    DA.write_proposal(report, str(out))

    proposed = json.loads(out.read_text(encoding="utf-8"))
    assert "DERIVED" in proposed["$comment"]
    assert "NEVER rendered into a prompt" in proposed["$comment"]

    entry = proposed["insured_name"]["aliases"][0]
    assert {"label", "documents", "confidence", "pattern", "examples"} <= set(entry)
    assert "$unresolved" in proposed        # the QA pass travels with the proposal


def test_a_broken_document_does_not_stop_the_batch():
    docs = _fixture_documents() + [("broken_0001", {"insured_name": object()}, "text")]
    report = DA.derive_aliases(docs)
    assert report.aliases["insured_name"], "good documents must still be processed"
