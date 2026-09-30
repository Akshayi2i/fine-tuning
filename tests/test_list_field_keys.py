"""Row matching for the canonical policy tables (evaluation.metrics.field_accuracy).

Knowing only Loss Run identifiers, every policy table fell back to matching on
all its fields at once: one wrong character dropped the whole row, and list
recall read 0.0 on every smoke-run checkpoint.
"""

from __future__ import annotations

from evaluation.metrics.field_accuracy import (
    _infer_key_fields,
    score_all_list_fields,
    score_list_field,
)


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


FORMS = [
    {"form_number": "HO 00 03", "edition_date": "05/11", "form_title": "Homeowners 3", "premium": None},
    {"form_number": "HO 04 90", "edition_date": "05/11", "form_title": "Replacement Cost", "premium": 45},
]


def test_forms_match_on_form_number_even_with_a_wrong_title():
    got = [dict(FORMS[0], form_title="Homeowners Special"), dict(FORMS[1])]
    report = score_list_field(FORMS, got, "forms_and_endorsements")
    assert _infer_key_fields(FORMS) == ["form_number"]
    assert report.recall == 1.0                     # both forms found
    assert report.row_field_accuracy < 1.0          # the wrong title still costs


def test_form_numbers_match_across_spacing_and_separators():
    got = [dict(FORMS[0], form_number="HO-0003"), dict(FORMS[1], form_number="ho 0490")]
    assert score_list_field(FORMS, got, "forms").recall == 1.0


def test_a_dropped_row_is_still_a_missed_row():
    report = score_list_field(FORMS, [dict(FORMS[0])], "forms")
    assert report.recall == 0.5 and report.missed_rows == 1


def test_vehicles_match_on_vin():
    vehicles = [{"vehicle_number": 1, "year": 2019, "make": "Honda", "vin": "1HGCM82633A004352"},
                {"vehicle_number": 2, "year": 2021, "make": "Ford", "vin": "1FTFW1E50MFA00001"}]
    got = [dict(vehicles[1], make="FORD MOTOR"), dict(vehicles[0], year=2018)]
    assert _infer_key_fields(vehicles) == ["vin"]
    assert score_list_field(vehicles, got, "auto.vehicles").recall == 1.0


def test_a_mostly_empty_identifier_is_passed_over():
    """A VIN column left blank on most rows cannot identify them."""
    vehicles = [{"vehicle_number": 1, "vin": None, "make": "Honda"},
                {"vehicle_number": 2, "vin": None, "make": "Ford"},
                {"vehicle_number": 3, "vin": "1FTFW1E50MFA00001", "make": "Ford"}]
    assert _infer_key_fields(vehicles) == ["vehicle_number"]


def test_locations_key_on_location_and_building_together():
    rows = [{"location_number": 1, "building_number": 1, "address": "a"},
            {"location_number": 1, "building_number": 2, "address": "b"}]
    assert _infer_key_fields(rows) == ["location_number", "building_number"]
    assert score_list_field(rows, list(reversed(rows)), "locations").recall == 1.0


def test_loss_runs_still_key_on_claim_number():
    claims = [{"claim_number": "C-1", "loss_date": "01/01/2024", "paid_amount": 10}]
    assert _infer_key_fields(claims) == ["claim_number"]


def test_a_table_with_no_identifier_still_matches_strictly():
    rows = [{"premium_basis": "a", "rate": "1"}, {"premium_basis": "b", "exposure": "2"}]
    assert _infer_key_fields(rows) == ["exposure", "premium_basis", "rate"]   # every scalar


def test_sub_items_key_on_their_label():
    rows = [{"label": "Each Occurrence", "value": "$1,000,000"}, {"label": "Aggregate", "value": "$2M"}]
    assert _infer_key_fields(rows) == ["label"]


def test_canonical_envelopes_are_matched_on_their_values():
    """The scorer compares values (values_view), not raw text or page_ref."""
    expected = {"forms_and_endorsements": [
        {"form_number": _env("HO 00 03", 1), "form_title": _env("Homeowners 3", 1)}]}
    got = {"forms_and_endorsements": [
        {"form_number": _env("HO 00 03", 7), "form_title": _env("Homeowners", 7)}]}
    report = score_all_list_fields(expected, got)["forms_and_endorsements"]
    assert report.recall == 1.0 and report.row_field_accuracy < 1.0


def test_an_identifier_missing_from_the_first_row_is_still_found():
    """Labels leave unstated values out; a first mortgagee without a name made
    the whole table match on every field."""
    rows = [{"rank": 1, "clause_type": "ISAOA"},
            {"rank": 2, "name": "Ownerschoice Funding", "loan_number": None},
            {"rank": 3, "name": "First Bank"}]
    assert _infer_key_fields(rows) == ["name"]
