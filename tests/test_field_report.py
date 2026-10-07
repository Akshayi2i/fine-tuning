"""Every field of a document's schema, with what one answer holds for it (testing.comparison.field_report)."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import pytest

from common.canonical import with_all_keys
from common.schemas import _strip_prefixed, is_canonical, load_schema, resolved_schema, schema_selectors
from testing.comparison import compare, field_report, schema_fields

EXAMPLE = (Path(__file__).resolve().parents[1] / "configs" / "canonical schema" / "common schema"
           / "examples" / "homeowners_minimal.json")


@pytest.fixture
def gold() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _served(answer: dict) -> dict:
    """As serving returns an answer: every key of the line's schema."""
    return with_all_keys(answer, _strip_prefixed(load_schema("policy", None, "homeowners"), ("fideon:",)))


def _value(parsed, page=1) -> dict:
    return {"raw": None if parsed is None else str(parsed), "parsed": parsed,
            "confidence": {"score": 0.9, "source": "vlm"}, "page_ref": [page], "flagged": True}


def _cell(report: dict, field_name: str) -> dict:
    """The first table cell of a schema field."""
    return next(entry for entry in report["fields"].values() if entry.get("field") == field_name)


def _unlisted(report: dict, declared: list[str]) -> list[str]:
    """Declared fields the report holds no entry for, at the field or inside it (text_sections' keys)."""
    listed = {entry.get("field", path) for path, entry in report["fields"].items()}
    return [name for name in declared
            if not any(other == name or other.startswith((name + ".", name + "[")) for other in listed)]


def test_every_field_the_schema_declares_is_listed(gold):
    report = field_report(_served(gold), lob="homeowners")
    declared = schema_fields("policy", None, "homeowners")
    assert _unlisted(report, declared) == []
    assert report["summary"]["schema_fields"] == len(declared) and report["schema"] == "policy:homeowners"


def test_each_field_gets_its_result_against_gold(gold):
    answer = copy.deepcopy(gold)
    answer["carrier"]["name"] = _value("Other Mutual")          # wrong
    answer["carrier"]["phone"] = _value(None)                   # missed
    answer["producer"]["email"] = _value("agent@example.com")   # invented
    fields = field_report(_served(answer), gold, lob="homeowners")["fields"]

    assert fields["carrier.naic_code"]["result"] == "correct"
    assert fields["carrier.name"]["result"] == "wrong"
    assert fields["carrier.name"]["gold"] == "Example Mutual Insurance Company"
    assert (fields["carrier.phone"]["status"], fields["carrier.phone"]["result"]) == ("null", "missed")
    assert fields["producer.email"]["result"] == "invented"
    assert fields["carrier.fax"] == {"value": None, "status": "null", "gold": None, "result": "empty"}


def test_a_value_the_model_wrote_keeps_its_confidence_and_pages(gold):
    answer = copy.deepcopy(gold)
    answer["carrier"]["name"] = _value("Example Mutual Insurance Company", page=2)
    entry = field_report(_served(answer), gold, lob="homeowners")["fields"]["carrier.name"]
    assert (entry["confidence"], entry["page_ref"], entry["result"]) == (0.9, [2], "correct")


def test_a_table_with_no_rows_lists_its_columns(gold):
    report = field_report(_served(gold), gold, lob="homeowners")
    assert report["fields"]["scheduled_items[].description"] == {
        "value": None, "status": "no_rows", "gold": None, "result": "empty"}


def test_a_row_the_answer_does_not_have_is_missed(gold):
    answer = copy.deepcopy(gold)
    answer["coverages"] = answer["coverages"][:1]
    fields = field_report(_served(answer), gold, lob="homeowners")["fields"]
    name = fields["coverages[coverage_code=hocovc].coverage_name"]
    assert (name["status"], name["result"], name["gold"]) == ("not_in_output", "missed", "Coverage C - Personal Property")


def test_review_flags_are_attached_to_their_field(gold):
    flags = ["carrier.name:uncalibrated", "coverages[0].limits[0].amount:low_confidence", "window_failed:arrays:p3"]
    report = field_report(_served(gold), lob="homeowners", review_flags=flags)
    assert report["fields"]["carrier.name"]["flags"] == ["uncalibrated"]
    amount = report["fields"]["coverages[coverage_code=hocova].limits[#1].amount"]
    assert (amount["flags"], amount["value"]) == (["low_confidence"], 350000)
    assert report["document_flags"] == ["window_failed:arrays:p3"]


def test_without_gold_each_field_has_its_status_only(gold):
    report = field_report(_served(gold), lob="homeowners")
    assert report["graded_against_gold"] is False and "by_result" not in report["summary"]
    assert not any("result" in entry or "gold" in entry for entry in report["fields"].values())
    assert report["fields"]["carrier.name"]["status"] == "extracted"
    assert report["fields"]["carrier.fax"]["status"] == "null"


def test_system_fields_and_row_ids_are_listed_but_not_graded(gold):
    report = field_report(_served(gold), gold, lob="homeowners")
    assert report["fields"]["document.page_count"] == {"value": 3, "status": "system"}
    coverage_id = _cell(report, "coverages[].coverage_id")
    assert coverage_id["status"] == "structural" and "result" not in coverage_id


def test_the_counts_are_the_comparisons(gold):
    answer = copy.deepcopy(gold)
    answer["carrier"]["name"] = _value("Other Mutual")
    answer["coverages"] = answer["coverages"][:1]
    served = _served(answer)
    tally = field_report(served, gold, lob="homeowners")["summary"]["by_result"]
    adapter = compare(gold, {}, served, lob="homeowners").adapter
    assert [tally.get(name, 0) for name in ("correct", "wrong", "missed", "invented")] == [
        adapter.correct, adapter.wrong, adapter.missed, adapter.invented]


# --------------------------------------------------------------------------
# Every document type, not one line
# --------------------------------------------------------------------------


def _full_answer(selector: tuple) -> dict:
    """An answer with a value in every field the selector's schema declares."""
    schema = (_strip_prefixed(load_schema(*selector), ("fideon:",)) if is_canonical(*selector)
              else resolved_schema(*selector))
    defs, counter = schema.get("$defs") or {}, itertools.count(1)

    def resolve(sub):
        for _ in range(20):
            if not isinstance(sub, dict):
                return {}
            if "$ref" in sub:
                sub = defs.get(sub["$ref"].rsplit("/", 1)[-1]) if sub["$ref"].startswith("#/$defs/") else None
                continue
            branches = sub.get("anyOf") or sub.get("oneOf")
            if branches and "properties" not in sub and "items" not in sub:
                resolved = [resolve(b) for b in branches]
                sub = next((b for b in resolved if "properties" in b or "items" in b),
                           next((b for b in resolved if b.get("type") not in (None, "null")), {}))
                continue
            return sub
        return {}

    def envelope(sub):
        return {"raw", "parsed", "page_ref"} <= set(sub.get("properties") or {})

    def scalar(sub):
        n = next(counter)
        enum = [v for v in sub.get("enum") or [] if v is not None]
        kind = sub.get("type")
        kind = next((k for k in kind if k != "null"), "string") if isinstance(kind, list) else kind
        return enum[0] if enum else {"integer": n, "number": n + 0.5, "boolean": True}.get(kind, f"s{n}")

    def build(sub, depth=0):
        sub = resolve(sub)
        if envelope(sub) or depth > 30:
            return _value(f"v{next(counter)}")
        if sub.get("properties"):
            return {key: build(child, depth + 1) for key, child in sub["properties"].items()}
        if sub.get("type") == "array" or "items" in sub:
            rows = resolve(sub.get("items"))
            return [build(rows, depth + 1)] if rows.get("properties") or envelope(rows) else [scalar(rows)]
        return scalar(sub)

    return build(schema)


@pytest.mark.parametrize("selector", schema_selectors(), ids=lambda s: ":".join(p for p in s if p))
def test_every_document_type_lists_every_field_and_grades_a_perfect_answer_correct(selector):
    doc_type, acord_form, lob = selector
    answer = _full_answer(selector)
    report = field_report(answer, answer, doc_type=doc_type, acord_form=acord_form, lob=lob)

    assert _unlisted(report, schema_fields(*selector)) == []
    tally = report["summary"]["by_result"]
    assert not any(tally.get(name) for name in ("wrong", "missed", "invented")), tally
    adapter = compare(answer, {}, answer, doc_type=doc_type, acord_form=acord_form, lob=lob).adapter
    assert [tally.get(name, 0) for name in ("correct", "wrong", "missed", "invented")] == [
        adapter.correct, adapter.wrong, adapter.missed, adapter.invented]


def test_a_field_the_model_is_never_asked_for_is_listed_but_not_graded(gold):
    # The example's form carries the page_range the pipeline fills, and its
    # text_sections the full printed text another path reads.
    report = field_report(_served(gold), gold, lob="homeowners")
    page_range = _cell(report, "forms_and_endorsements[].page_range")
    assert page_range["status"] == "system" and "result" not in page_range
    text = [entry for path, entry in report["fields"].items() if path.startswith("text_sections.")]
    assert text and all(e["status"] == "system" and "result" not in e and "note" in e for e in text)
    assert "invented" not in report["summary"]["by_result"]
    assert not compare(gold, {}, _served(gold), lob="homeowners").adapter.invented


def test_a_document_no_schema_selects_still_lists_what_it_holds():
    answer = {"insured_name": "A. Smith", "policy_number": None}
    report = field_report(answer, {"insured_name": "A. Smith"}, doc_type="acord", acord_form=None)
    assert report["schema"] is None
    assert report["fields"]["insured_name"]["result"] == "correct"
    assert report["fields"]["policy_number"] == {"value": None, "status": "null", "gold": None, "result": "empty"}


def test_the_extraction_json_carries_every_field_after_the_summaries(tmp_path):
    from serving.pipeline import ExtractionResult
    from testing.run_extraction import write_outputs

    result = ExtractionResult(source_id="doc-1", doc_type="policy", model_version="v-test", mode="ocr_plus_image",
                              schema_valid=True, overall_confidence=0.9, extraction={"policy": {}})
    report = {"summary": {"schema_fields": 1}, "fields": {}}
    results_path, metrics_path = write_outputs(result, {"document": "doc-1", "all_fields": report}, root=tmp_path)
    written = json.loads(results_path.read_text(encoding="utf-8"))
    keys = list(written)
    assert written["all_fields"] == report and keys.index("all_fields") == keys.index("list_fields") + 1
    assert "all_fields" not in json.loads(metrics_path.read_text(encoding="utf-8"))
