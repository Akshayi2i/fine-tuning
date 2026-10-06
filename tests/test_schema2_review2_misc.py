"""Second review: what judging the served JSON let through, what the detection
measurement fed its classifier, and what the field-type proposal would table.

Judging a common-model policy by its filled JSON is right for the client's
if/then rules, but the fill also satisfies the rule that an object holds
something: a carrier written as {} was served as nulls and judged valid. The
noisy detection measurement spent its corruption on pages the classifier never
reads, so it mostly measured clean text. And the field-type proposal guessed a
type by name for fields the common model declares a type for, where a row in
the reviewed table would override that declaration on every line.
"""

from __future__ import annotations

import copy
import json

import pytest

from common.canonical import values_view
from common.schemas import iter_validation_errors, iter_validation_errors_by_keyword
from serving.pipeline import PipelineError
from tests.test_schema2_serving import AUTO, _compact_auto, _serve_replay, _stated

# --------------------------------------------------------------------------
# Serving: an object written empty is not passed off as a row of nulls
# --------------------------------------------------------------------------


def _empty_declarations(answer):
    """The declarations window writes every required section with nothing in it."""
    if "carrier" in json.loads(answer):
        return json.dumps({"document": {}, "carrier": {}, "named_insured": {}, "policy": {}})
    return answer


def _empty_form_row(answer):
    parsed = json.loads(answer)
    if "forms_and_endorsements" in parsed:
        parsed["forms_and_endorsements"].append({})
    return json.dumps(parsed)


def test_a_carrier_and_an_insured_written_empty_fail_strict_validation():
    """The decoder lets a window write carrier: {} (it drops the client's
    minProperties), and the fill turned it into an envelope of nulls the rule
    accepts: a policy with no carrier, no insured and no policy number was
    served as valid. The answer before the fill is judged for emptiness."""
    with pytest.raises(PipelineError, match=r"carrier: \{\} should be non-empty"):
        _serve_replay(_compact_auto(), 3, strict=True, edit=_empty_declarations)

    result, _backend, _answers = _serve_replay(_compact_auto(), 3, edit=_empty_declarations)
    assert not result.schema_valid and "schema:invalid" in result.review_flags
    assert {"carrier: {} should be non-empty",
            "named_insured: {} should be non-empty"} <= set(result.validation_errors)
    # document and policy were written as {} too, and the system fields fill
    # them before emptiness is judged: they are never reported empty.
    assert not [e for e in result.validation_errors if e.startswith(("document:", "policy:"))]


def test_an_empty_row_the_model_wrote_fails_validation():
    """A [{}] row was served as a phantom row of nulls, judged valid."""
    with pytest.raises(PipelineError, match=r"forms_and_endorsements/1: \{\} should be non-empty"):
        _serve_replay(_compact_auto(), 3, strict=True, edit=_empty_form_row)

    result, _backend, _answers = _serve_replay(_compact_auto(), 3, edit=_empty_form_row)
    assert not result.schema_valid and "schema:invalid" in result.review_flags
    assert "forms_and_endorsements/1: {} should be non-empty" in result.validation_errors


def test_the_perfect_replay_and_the_flat_deductible_still_pass_strict_validation():
    """Emptiness is the only rule judged before the fill: the if/then rules the
    fill exists to satisfy (a flat deductible's amount) are still judged on the
    JSON served."""
    result, _backend, _answers = _serve_replay(_compact_auto(), 3, strict=True)
    assert result.schema_valid and result.validation_errors == []

    gold = copy.deepcopy(AUTO)
    deductible = next(c for c in gold["coverages"] if c.get("deductibles"))["deductibles"][0]
    page = deductible["amount"]["page_ref"][0]
    deductible["amount"] = {"raw": None, "parsed": None, "confidence": {"score": 1.0, "source": "deterministic"},
                            "page_ref": [], "flagged": False}
    deductible["peril"] = _stated("Collision", page)
    result, _backend, _answers = _serve_replay(gold, 6, strict=True)
    assert result.schema_valid and "schema:invalid" not in result.review_flags
    served = [d for c in result.extraction["coverages"] for d in c.get("deductibles") or []
              if values_view(d.get("peril")) == "Collision"]
    assert len(served) == 1 and values_view(served[0]["amount"]) is None


def test_the_homeowners_example_replays_under_strict_validation():
    from pathlib import Path

    example = (Path(__file__).resolve().parents[1] / "configs" / "canonical schema" / "common schema"
               / "examples" / "homeowners_minimal.json")
    gold = json.loads(example.read_text(encoding="utf-8"))
    result, _backend, _answers = _serve_replay(gold, gold["document"]["page_count"], "homeowners", strict=True)
    assert result.schema_valid and result.validation_errors == []


def test_a_section_no_window_wrote_still_fails():
    def without_declarations(answer):
        return json.dumps({k: v for k, v in json.loads(answer).items()
                           if k not in ("carrier", "named_insured", "policy")})

    with pytest.raises(PipelineError, match="a required section no window wrote"):
        _serve_replay(_compact_auto(), 3, strict=True, edit=without_declarations)


def test_errors_are_selected_by_the_rules_keyword():
    """Only minProperties errors, whatever else is wrong with the document."""
    gold = copy.deepcopy(AUTO)
    gold["carrier"] = {}
    gold["policy"]["policy_number"] = 12345  # a type error, not an empty object
    every = list(iter_validation_errors(gold, "policy", None, "personal_auto"))
    empty = list(iter_validation_errors_by_keyword(gold, "policy", None, "personal_auto",
                                                   keywords=("minProperties",)))
    assert empty == ["carrier: {} should be non-empty"]
    assert len(every) > len(empty)


# --------------------------------------------------------------------------
# The detection measurement corrupts what the classifier reads
# --------------------------------------------------------------------------

PAGE = ("DECLARATIONS PAGE Policy Number HO-123456 Named Insured John Smith Coverage A Dwelling "
        "$350,000 Deductible $1,000 Homeowners policy form HO-3. ")


def _classifier_text(request):
    from serving.doc_type_classifier import classifier_messages
    from serving.pipeline import _classifier_input

    content = classifier_messages(*_classifier_input(request))[1]["content"]
    return [part["text"] for part in content if part["type"] == "text"]


def test_a_noisy_case_on_a_long_policy_reads_noisy_text_and_the_training_rows_noise():
    """On a 12-page policy the budget, spread over every page, left the two
    pages the classifier reads clean in most documents; the classify rows it is
    trained on corrupt those two pages, seeded per document."""
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, classify_rows
    from evaluation.golden_eval import GoldenDocument
    from evaluation.lob_detection_eval import detection_cases

    pages = 12
    documents = [GoldenDocument(f"policy_{i:04d}", "policy", {}, [f"policy_{i:04d}/page_{p}.png"
                                                                  for p in range(1, pages + 1)],
                                page_texts={p: f"{PAGE * 3} page {p}" for p in range(1, pages + 1)},
                                lob="homeowners")
                 for i in range(12)]
    local = {key: f"/cache/{key}" for d in documents for key in d.image_keys}
    clean = detection_cases(documents, local)
    noisy = detection_cases(documents, local, mode="noisy_ocr_image")
    for doc, plain, corrupted in zip(documents, clean, noisy, strict=True):
        assert _classifier_text(corrupted.request) != _classifier_text(plain.request), doc.source_id
        # The pages it does not read are left as they were.
        assert {p: t for p, t in corrupted.request.page_texts.items() if p > 2} == {
            p: t for p, t in doc.page_texts.items() if p > 2}
        source = SourceDocument(source_id=doc.source_id, doc_type="policy", golden_label={},
                                ocr_pages=[doc.page_texts[p] for p in range(1, pages + 1)],
                                image_paths=list(doc.image_keys), lob="homeowners", tenant_id="default")
        [row] = classify_rows(source, "train", ("noisy_ocr_image",))
        trained = [part["text"] for part in row["messages"][1]["content"] if part["type"] == "text"]
        assert trained == _classifier_text(corrupted.request), doc.source_id


# --------------------------------------------------------------------------
# The field-type proposal leaves a declared type to the common model
# --------------------------------------------------------------------------


def _proposal(tmp_path):
    from scripts import propose_field_types as proposer

    out = tmp_path / "proposed.yaml"
    assert proposer.main(["--bundles", str(tmp_path / "no_bundles"), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8").splitlines()
    rows = {}
    for line in text[text.index("fields:") + 1:]:
        body = line.strip()
        if not body.startswith("#"):
            path, rest = body.split(":", 1)
            rows[path] = rest.split("#", 1)[0].strip()
    return text, rows


def test_a_field_the_common_model_types_gets_no_row(tmp_path):
    """billing.amount_due is money by the common model; by name it is free
    text, and a row saying so would move it off its calibrator on every line."""
    from calibration.features import common_model_field_types
    from scripts import propose_field_types as proposer

    declared = common_model_field_types()
    fields = proposer.schema_fields()
    text, rows = _proposal(tmp_path)
    assert not set(rows) & set(declared)
    for path in sorted(fields & set(declared)):
        assert f"  # {path}: {declared[path]}  (declared by the common model; not tabled)" in text, path
    proposal = proposer.propose("billing.amount_due", [])
    assert proposal["declared"] and proposal["type"] == "money" and not proposal["review"]


def test_a_field_the_common_model_does_not_type_is_proposed_as_before(tmp_path):
    from calibration.features import common_model_field_types, infer_field_type
    from scripts import propose_field_types as proposer

    declared = common_model_field_types()
    undeclared = proposer.schema_fields() - set(declared)
    _text, rows = _proposal(tmp_path)
    assert set(rows) == undeclared
    for path in undeclared:
        proposal = proposer.propose(path, [])
        assert not proposal["declared"] and proposal["by_name"] == infer_field_type(path)
        assert rows[path] == proposal["type"], path
