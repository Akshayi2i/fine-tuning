"""Scoring a common-model line (SPEC_21): rows, links, codes, overflow, and no ids.

The gold and the answer below differ only where each test says. Ids are each
writer's own numbering, so an answer that numbers its rows differently from the
gold scores exactly as well; a wrong limit inside a coverage row now costs field
match; a link, a code and an overflow value each have their own number.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evaluation.run_eval import build_report

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schema2"
GOLD = json.loads((FIXTURES / "personal_auto_6page.json").read_text(encoding="utf-8"))


def _metrics(got, *, gold=GOLD, **meta):
    metadata = {"source_id": "pa", "doc_type": "policy", "lob": "personal_auto",
                "modality_mode": "ocr_plus_image", **meta}
    [full] = build_report("t", [(gold, got, metadata)]).full_set()
    return full.metrics


def _renumbered(doc):
    """The same answer, its rows numbered the other way round."""
    out = copy.deepcopy(doc)
    swap = {"veh_1": "veh_2", "veh_2": "veh_1"}
    for vehicle in out["vehicles"]:
        vehicle["unit_id"] = swap[vehicle["unit_id"]]
    for row in out["coverages"] + out["interested_parties"]:
        row["applies_to"] = [swap[v] for v in row["applies_to"]]
    out["vehicles"].reverse()
    return out


def test_the_gold_scores_perfectly_on_every_count():
    m = _metrics(copy.deepcopy(GOLD))
    assert m["field_normalized_match"] == 1.0
    assert m["reference_accuracy"] == 1.0 and m["coverage_code_accuracy"] == 1.0
    assert m["additional_fields_recall"] == 1.0 and m["additional_fields_precision"] == 1.0


def test_an_answer_numbering_its_rows_differently_scores_the_same():
    m = _metrics(_renumbered(GOLD))
    assert m["field_normalized_match"] == 1.0 and m["reference_accuracy"] == 1.0


def test_a_wrong_limit_inside_a_coverage_costs_field_match():
    got = copy.deepcopy(GOLD)
    got["coverages"][2]["limits"][0]["amount"] = {**got["coverages"][2]["limits"][0]["amount"],
                                                  "raw": "$25,000", "parsed": 25000}
    assert _metrics(got)["field_normalized_match"] < 1.0


def test_a_lost_link_costs_reference_accuracy_not_field_match():
    got = copy.deepcopy(GOLD)
    for coverage in got["coverages"]:
        coverage.pop("applies_to")
    m = _metrics(got)
    assert m["reference_accuracy"] < 1.0
    assert m["field_normalized_match"] == 1.0


def test_a_wrong_code_costs_coverage_code_accuracy():
    got = copy.deepcopy(GOLD)
    got["coverages"][1]["coverage_code"] = "X_COMPREHENSIVE"
    assert _metrics(got)["coverage_code_accuracy"] == 0.75


def test_overflow_is_scored_apart_from_the_core():
    got = copy.deepcopy(GOLD)
    got["additional_fields"] = []
    m = _metrics(got)
    assert m["additional_fields_recall"] == 0.0
    assert m["field_normalized_match"] == 1.0


def test_codes_and_links_are_never_hallucinations():
    text = " ".join(str(v.get("raw")) for v in _envelopes(GOLD))
    m = _metrics(copy.deepcopy(GOLD), ocr_text=text)
    assert m["hallucination_rate"] == 0.0


def test_the_weakest_line_is_reported_once_it_has_enough_values(monkeypatch):
    from evaluation import run_eval

    monkeypatch.setattr(run_eval, "WORST_LINE_MIN_VALUES", 1)
    assert _metrics(copy.deepcopy(GOLD))["worst_line_field_match"] == 1.0
    monkeypatch.setattr(run_eval, "WORST_LINE_MIN_VALUES", 10_000)
    assert "worst_line_field_match" not in _metrics(copy.deepcopy(GOLD))


def test_the_weakest_line_is_a_conditional_gate_at_the_pilot_floor():
    from evaluation.gating import CONDITIONAL_METRICS, GATING_METRICS, PILOT_FLOORS

    assert GATING_METRICS["worst_line_field_match"] == "higher_is_better"
    assert "worst_line_field_match" in CONDITIONAL_METRICS
    assert PILOT_FLOORS["worst_line_field_match"] == 0.85


def test_a_self_contained_line_is_scored_as_before():
    gold = {"policy": {"policy_number": {"raw": "GL-1", "parsed": "GL-1", "page_ref": [1]}}}
    metrics = build_report("t", [(gold, gold, {"source_id": "g", "doc_type": "policy", "lob": "gl",
                                               "modality_mode": "ocr_plus_image"})]).full_set()[0].metrics
    assert metrics["field_normalized_match"] == 1.0
    assert "reference_accuracy" not in metrics and "worst_line_field_match" not in metrics


def _envelopes(node):
    if isinstance(node, dict):
        if "raw" in node and "page_ref" in node:
            yield node
        for value in node.values():
            yield from _envelopes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _envelopes(value)


# --------------------------------------------------------------------------
# Calibration field types and the unseen-label slice
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path,kind", [
    ("coverages[0].limits[1].amount", "money"), ("policy.effective_date", "date"),
    ("vehicles[0].year", "number"), ("coverages[0].included", "enum"),
    ("lob_parts[0].premium", "money"),
])
def test_a_common_model_field_is_calibrated_as_its_value_type(path, kind):
    from calibration.features import infer_field_type

    assert infer_field_type(path, common_model=True) == kind


def test_an_overflow_value_is_typed_by_what_was_parsed():
    from calibration.features import infer_field_type

    assert infer_field_type("additional_fields[0].value", 12.0, common_model=True) == "number"
    assert infer_field_type("additional_fields[0].value", "Applied", common_model=True) == "free_text"


def test_labels_split_into_seen_and_unseen_by_what_the_page_prints():
    from types import SimpleNamespace

    from evaluation.metrics.unseen_labels import common_model_aliases, label_split

    aliases = common_model_aliases("personal_auto")
    assert "Effective Date" in aliases["policy.effective_date"]
    results = [SimpleNamespace(field_path="policy.effective_date", correct=True),
               SimpleNamespace(field_path="policy.expiration_date", correct=False)]
    text = "Policy Period From 04/01/2026  Expiry Date 10/01/2026"
    split = label_split(results, text, aliases, seen={"Policy Period From"})
    assert split == {"seen": [True], "unseen": [False]}


def test_the_unseen_slice_reaches_the_report_when_the_metadata_carries_it():
    text = "Policy Number PA-2026-0042 Policy Period From 04/01/2026"
    m = _metrics(copy.deepcopy(GOLD), ocr_text=text, seen_labels=["Policy Number"])
    assert m["seen_label_field_match"] == 1.0 and m["unseen_label_field_match"] == 1.0
