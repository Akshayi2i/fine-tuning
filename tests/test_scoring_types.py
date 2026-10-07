"""Scoring compares a field by the type its schema declares (accuracy plan, stage 0.1)."""

from __future__ import annotations

import pytest

from common.normalize import normalize_identifier, values_match
from evaluation.metrics.field_accuracy import score_fields, scoring_kind, values_agree


def test_two_numbers_are_compared_as_numbers_whatever_the_field_is_called():
    # premium.total and year_built are read as text by their names: 957 vs 957.0 was "wrong".
    assert values_match(957, 957.0, field_path="premium.total")
    assert values_match(1955, 1955.0, field_path="buildings[0].year_built")
    assert not values_match(957, 958.0, field_path="premium.total")
    assert not values_match(True, 1, field_path="flags.ok")       # a flag is not a number


def test_an_identifier_read_as_a_decimal_is_the_whole_number():
    assert normalize_identifier(1.0) == "1"
    assert values_match(1, 1.0, field_path="locations[0].location_number")
    assert not values_match(10, 1.0, field_path="locations[0].location_number")


@pytest.mark.parametrize(("path", "kind"), [
    ("premium.total", "currency"),
    ("coverages[coverage_code=hocova].limits[#1].amount", "currency"),
    ("buildings[0].year_built", "number"),
    ("policy.effective_date", "date"),
    ("carrier.name", None),                    # not a typed value: left to the name
])
def test_the_scoring_kind_is_the_declared_type(path, kind):
    assert scoring_kind(path) == kind


def test_a_printed_amount_matches_its_number_on_a_money_field():
    assert not values_match("$831.00", 831.0, field_path="premium.total")   # by name: text
    assert values_agree("$831.00", 831.0, "premium.total")                   # by type: money
    assert values_agree("Included*", "Included*", "coverages[0].premium")    # unparseable: as text


def test_a_correct_number_scores_correct_and_exact_never_exceeds_it():
    report = score_fields({"premium": {"total": 957}, "policy": {"term_months": 12}},
                          {"premium": {"total": 957.0}, "policy": {"term_months": "twelve"}})
    by_path = {r.field_path: r for r in report.results}
    assert by_path["premium.total"].correct
    assert all(r.correct or not r.exact for r in report.results)
    assert report.exact_match <= report.normalized_match
