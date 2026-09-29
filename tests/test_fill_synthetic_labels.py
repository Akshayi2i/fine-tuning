"""Completing a synthetic label from its source's reviewed gold, without regeneration.

Only missing fields are added; a field the twin has is never changed. What is
added is made true of the twin, or left out: dates shifted, replaced identities
swapped only when known, amounts only where the template kept them.
"""

from __future__ import annotations

import csv
import json

import pytest

from data_pipeline.ingestion.fill_synthetic_labels import fill, fill_delivery, is_identifying, shift_date


def env(raw, parsed=None, page=1):
    return {"raw": raw, "parsed": raw if parsed is None else parsed,
            "confidence": {"score": 1.0, "source": "deterministic"}, "page_ref": [page], "flagged": False}


def _source():
    return {
        "carrier": {"company_name": env("Acme Mutual")},
        "named_insured": {"primary_name": env("John Real"), "mailing_address": {"line_1": env("12 Elm St")}},
        "producer": {"agency_name": env("Real Agency")},
        "policy": {"policy_number": env("HO-111"), "effective_date": env("09/09/2024"),
                   "expiration_date": env("09/09/2025")},
        "premium": {"total_policy_premium": env("$1,200.00", 1200.0)},
        "homeowners": {"section_i_property_coverages": {
            "coverage_a_dwelling_limit": env("$300,000", 300000.0),
            "coverage_b_other_structures_limit": env("$30,000", 30000.0)}},
        "forms_and_endorsements": [
            {"form_number": env("HO 00 03"), "edition_date": env("05 11")},
            {"form_number": env("HO 04 90"), "edition_date": env("0699")},
        ],
        "signature": {"signer_name": env("John Real")},
        "text_sections": [{"title": "Notice", "text": "real text"}],
    }


def _twin():
    return {
        "carrier": {"company_name": env("Acme Mutual")},
        "named_insured": {"primary_name": env("Jane Fake")},
        "policy": {"policy_number": env("HO-999"), "effective_date": env("06/16/2024")},
        "premium": {"total_policy_premium": env("$1,200.00", 1200.0)},
        "homeowners": {"section_i_property_coverages": {"coverage_a_dwelling_limit": env("$300,000", 300000.0)}},
        "forms_and_endorsements": [{"form_number": env("HO 00 03")}],
        "text_sections": [{"title": "Notice", "text": "fake text"}],
        "fideon:provenance": {"date_shift_days": -85},
        "fideon:absent": ["policy.expiration_date", "signature.signer_name", "homeowners.x"],
    }


def test_fields_the_twin_has_are_never_changed():
    merged, _ = fill(_source(), _twin())
    assert merged["named_insured"]["primary_name"]["raw"] == "Jane Fake"
    assert merged["policy"]["policy_number"]["raw"] == "HO-999"
    assert merged["text_sections"] == _twin()["text_sections"]
    assert merged["fideon:provenance"] == {"date_shift_days": -85}


def test_unchanged_values_are_added_as_they_are():
    merged, stats = fill(_source(), _twin())
    cov = merged["homeowners"]["section_i_property_coverages"]
    assert cov["coverage_b_other_structures_limit"]["raw"] == "$30,000"      # amounts all agreed
    assert merged["carrier"]["company_name"]["raw"] == "Acme Mutual"
    assert stats.added["homeowners"] == 1


def test_dates_are_shifted_by_the_recorded_offset_and_edition_dates_kept():
    merged, _ = fill(_source(), _twin())
    assert merged["policy"]["expiration_date"]["raw"] == "06/16/2025"         # 09/09/2025 - 85 days
    forms = merged["forms_and_endorsements"]
    assert forms[0]["edition_date"]["raw"] == "05 11"
    assert forms[1]["edition_date"]["raw"] == "0699"


def test_a_replaced_identity_is_swapped_when_known_and_left_out_when_not():
    merged, stats = fill(_source(), _twin())
    assert merged["signature"]["signer_name"]["raw"] == "Jane Fake"           # John Real -> Jane Fake is known
    assert "producer" not in merged                                            # the agency's new name is not
    assert "mailing_address" not in merged["named_insured"]
    assert stats.dropped["identifying: replacement unknown"] == 2


def test_amounts_are_not_added_where_the_template_changed_them():
    twin = _twin()
    twin["premium"]["total_policy_premium"] = env("$980.00", 980.0)
    merged, stats = fill(_source(), twin)
    assert "coverage_b_other_structures_limit" not in merged["homeowners"]["section_i_property_coverages"]
    assert stats.dropped["amount: this template's amounts differ"] == 1


def test_rows_are_matched_on_unchanged_values_and_filled():
    merged, stats = fill(_source(), _twin())
    forms = merged["forms_and_endorsements"]
    assert [f["form_number"]["raw"] for f in forms] == ["HO 00 03", "HO 04 90"]   # source order, no duplicate
    assert stats.rows_matched == 1 and stats.rows_added == 1


def test_no_row_is_added_while_the_twin_has_rows_that_did_not_match():
    twin = _twin()
    twin["forms_and_endorsements"].append({"form_number": env("HO 0490")})    # the generator read it differently
    merged, stats = fill(_source(), twin)
    assert len(merged["forms_and_endorsements"]) == 2                          # not three
    assert stats.dropped["table row: twin has rows that did not match"] == 1


def test_absent_paths_that_are_now_filled_leave_the_absent_list():
    merged, _ = fill(_source(), _twin())
    assert merged["fideon:absent"] == ["homeowners.x"]


@pytest.mark.parametrize("value,days,expected", [
    ("09/09/2024", 10, "09/19/2024"), ("9/9/24", 10, "9/19/24"), ("2024-01-31", 1, "2024-02-01"),
    ("DEC 4, 2025", -105, "AUG 21, 2025"), ("December 4, 2025", 1, "December 5, 2025"),
])
def test_dates_keep_their_layout(value, days, expected):
    assert shift_date(value, days) == expected


@pytest.mark.parametrize("path,identifying", [
    ("named_insured.primary_name", True), ("auto.drivers[0].driver_name", True),
    ("producer.agency_name", True), ("interested_parties[1].name", True),
    ("homeowners.discounts[0].discount_name", False), ("forms_and_endorsements[0].form_name", False),
    ("auto.vehicles[0].coverages[0].coverage_name", False), ("carrier.company_name", False),
    ("carrier.claims_phone", True), ("named_insured.mailing_address.postal_code", True),
])
def test_what_counts_as_a_replaced_identity(path, identifying):
    assert is_identifying(path, "x") is identifying


def test_the_delivery_is_completed_in_place_only_with_apply(tmp_path):
    delivery, reviewed = tmp_path / "delivery", tmp_path / "reviewed"
    (delivery / "Train/gold json").mkdir(parents=True)
    (reviewed / "Acme/homeowners").mkdir(parents=True)
    (reviewed / "Acme/homeowners/ho_1.json").write_text(json.dumps(_source()), encoding="utf-8")
    gold = delivery / "Train/gold json/homeowners__ho_1__synth_001.json"
    gold.write_text(json.dumps(_twin()), encoding="utf-8")
    with (delivery / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "lob", "source", "kind", "gold", "pages"])
        w.writeheader()
        w.writerow({"split": "Train", "lob": "homeowners", "source": "Acme/homeowners/ho_1.pdf",
                    "kind": "synthetic", "gold": "Train/gold json/homeowners__ho_1__synth_001.json", "pages": 3})
    dry = fill_delivery(delivery, reviewed, lines=frozenset({"homeowners"}))
    assert dry.filled == 1 and dry.fields_after > dry.fields_before
    assert json.loads(gold.read_text(encoding="utf-8")) == _twin()             # a dry run writes nothing
    fill_delivery(delivery, reviewed, lines=frozenset({"homeowners"}), apply=True)
    assert "coverage_b_other_structures_limit" in json.loads(gold.read_text(encoding="utf-8"))["homeowners"][
        "section_i_property_coverages"]
