"""The agreed label conventions, applied as bundles are prepared (data_pipeline.ingestion.label_rules)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from data_pipeline.ingestion.label_rules import PRINTED_NOWHERE, apply_label_rules, page_texts

#: Enough ordinary words that a page's text counts as a whole page.
FILLER = " ".join(f"word{i}" for i in range(60))
PAGES = {
    1: f"Mercury Casualty Company  Policy Number NYHP0000004000\n{FILLER}",
    2: "Policy Number NYHP0000004000\nNamed Insured Clemence Fairbank\nLocation: 2 TOWN RD MOUNT MARION NY\n"
       "Coverage A Dwelling $219,000  Protection Class 3  Paid By: Mortgagee  Escrow: Yes\n"
       f"Mortgagee M & T Bank\n{FILLER}",
    3: f"Policy Number NYHP0000004000\nSection II Medical Payments $219,000 aggregate\nPage 3 of 3\n{FILLER}",
    4: f"Policy Number NYHP0000004000\nWater Leak Detection: No\n{FILLER}",
}


def _v(raw, pages, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed,
            "confidence": {"score": 1.0, "source": "vlm"}, "page_ref": pages, "flagged": False}


def _gold(**extra):
    gold = {
        "policy": {"policy_number": _v("NYHP0000004000", [2])},
        "named_insured": {"primary_name": _v("Clemence Fairbank", [2])},
        "locations": [{"unit_id": "loc_1", "location_number": _v("2", [2], 2),
                       "address": {"street": _v("2 TOWN RD", [2])}}],
        "coverages": [{"coverage_id": "cov_1", "coverage_code": "HO_COV_A",
                       "limits": [{"limit_type": "per_occurrence", "amount": _v("$219,000", [2], 219000.0)}]}],
        "interested_parties": [{"role": "mortgagee", "name": _v("M & T Bank", [2]),
                                "is_payor": _v("Yes", [2], True)}],
        "additional_fields": [],
    }
    gold.update(extra)
    return gold


def _extra(label, raw, pages):
    return {"label": label, "value": _v(raw, pages), "section_hint": "policy", "page_ref": pages}


def _kinds(notes):
    return [kind for kind, _ in notes]


def test_a_document_level_value_lists_every_page_that_prints_it():
    gold = _gold()
    notes = apply_label_rules(gold, PAGES)
    assert gold["policy"]["policy_number"]["page_ref"] == [1, 2, 3, 4]
    assert ("page list", "policy.policy_number [2] -> [1, 2, 3, 4]") in notes


def test_a_table_value_keeps_to_the_pages_it_cites():
    gold = _gold()
    apply_label_rules(gold, PAGES)     # $219,000 is also printed on page 3, for another coverage
    assert gold["coverages"][0]["limits"][0]["amount"]["page_ref"] == [2]


def test_a_page_that_does_not_print_the_value_is_taken_out():
    gold = _gold(policy={"policy_number": _v("NYHP0000004000", [2, 3]), "effective_date": _v("06/04/2025", [2, 3])},
                 premium={"total": _v("$1,234.00", [3], 1234.0), "change": _v("$56.78", [3], 56.78)})
    pages = {**PAGES, 2: PAGES[2] + "\nEffective 06/04/2025", 3: PAGES[3] + "\nTotal $1,234.00  Change $56.78"}
    apply_label_rules(gold, pages)
    assert gold["policy"]["effective_date"]["page_ref"] == [2]


def test_a_name_keeps_a_page_it_cites_whose_text_does_not_hold_it():
    gold = _gold(carrier={"name": _v("Leatherstocking Cooperative Insurance Company", [1, 4])})
    apply_label_rules(gold, {**PAGES, 4: PAGES[4] + "\nLeatherstocking Cooperative Insurance Company"})
    assert gold["carrier"]["name"]["page_ref"] == [1, 4]               # page 1 prints it in its logo


def test_a_phrase_is_added_where_it_heads_a_line_not_where_it_falls_in_a_sentence():
    gold = _gold(document={"title": _v("Policy Declarations", [1])})
    pages = {**PAGES, 1: PAGES[1] + "\nPolicy Declarations",
             3: PAGES[3] + "\nThe limits are as shown in the Policy Declarations for this policy.",
             4: PAGES[4] + "\nPOLICY DECLARATIONS (continued)"}
    apply_label_rules(gold, pages)
    assert gold["document"]["title"]["page_ref"] == [1, 4]


def test_a_page_whose_text_holds_the_words_but_not_the_amounts_settles_nothing():
    # Amounts drawn in a font that records no characters: every caption is in the text, no figure is.
    page = PAGES[2].replace("$219,000", "").replace("Protection Class 3", "Protection Class")
    gold = _gold(premium={"total": _v("$1,234.00", [2], 1234.0)})
    notes = apply_label_rules(gold, {**PAGES, 2: page})
    assert PRINTED_NOWHERE not in _kinds(notes)


def test_a_page_whose_text_misses_most_values_settles_nothing():
    pages = {**PAGES, 2: f"Policy Number NYHP0000004000\n{FILLER}"}     # its table was an image
    gold = _gold(named_insured={"primary_name": _v("Clemence Fairbank", [2, 3])})
    notes = apply_label_rules(gold, pages)
    # Printed on no page now - but page 2 cannot settle that, so nothing is taken out.
    assert gold["named_insured"]["primary_name"]["page_ref"] == [2, 3]
    assert "location_number" in gold["locations"][0]                   # nor the number on page 2
    assert PRINTED_NOWHERE not in _kinds(notes)


def test_an_extra_field_repeating_a_field_folds_into_it():
    gold = _gold(additional_fields=[_extra("Policy Number", "NYHP0000004000", [3])])
    notes = apply_label_rules(gold, PAGES)
    assert gold["additional_fields"] == []
    assert ("extra field repeats a field", "additional_fields[0] 'Policy Number' -> policy.policy_number") in notes


def test_a_common_amount_is_not_taken_for_a_repeat():
    gold = _gold(premium={"total": _v("$0.00", [2], 0.0)},
                 additional_fields=[_extra("Prior Annual Fire Fee", "$0.00", [2])])
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nPrior Annual Fire Fee $0.00 Total $0.00"})
    assert [e["label"] for e in gold["additional_fields"]] == ["Prior Annual Fire Fee"]


def test_a_year_is_not_taken_for_a_repeat():
    gold = _gold(watercraft=[{"unit_id": "wc_1", "year": _v("2015", [2], 2015)}],
                 additional_fields=[_extra("Trailer Information Year", "2015", [2])])
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nYear 2015  Trailer Information Year 2015"})
    assert [e["label"] for e in gold["additional_fields"]] == ["Trailer Information Year"]


def test_an_extra_field_repeating_a_table_value_gives_its_row_no_page():
    gold = _gold(additional_fields=[_extra("Modifies coverage(s) at renewal", "Coverage A - Dwelling", [4])])
    gold["coverages"][0]["coverage_name"] = _v("Coverage A - Dwelling", [2])
    apply_label_rules(gold, {**PAGES, 4: PAGES[4] + "\nModifies coverage(s) at renewal: Coverage A - Dwelling"})
    assert gold["additional_fields"] == []
    assert gold["coverages"][0]["coverage_name"]["page_ref"] == [2]     # mentioned on page 4, listed on 2


def test_the_same_extra_field_twice_becomes_one_with_both_pages():
    gold = _gold(additional_fields=[_extra("Water Leak Detection", "No", [4]),
                                    _extra("Water Leak Detection", "No", [2])])
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nWater Leak Detection: No"})
    assert len(gold["additional_fields"]) == 1
    assert gold["additional_fields"][0]["value"]["page_ref"] == [2, 4]


def test_page_counters_and_prose_are_not_extra_fields():
    prose = "It is your responsibility to select and maintain adequate amounts of insurance on your dwelling"
    gold = _gold(additional_fields=[_extra("Page 3 of 3", "Page 3 of 3", [3]), _extra(prose, prose, [1]),
                                    _extra("Water Leak Detection", "No", [4])])
    notes = apply_label_rules(gold, PAGES)
    assert [e["label"] for e in gold["additional_fields"]] == ["Water Leak Detection"]
    assert _kinds(notes).count("extra field not a value") == 2


def test_a_long_label_printed_before_an_amount_is_a_value():
    label = "Your 12-month policy premium excluding billing fees and payment option discounts is"
    gold = _gold(additional_fields=[_extra(label, "$352.00", [2])])
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + f"\n{label} $352.00"})
    assert [e["label"] for e in gold["additional_fields"]] == [label]


def test_protection_class_and_a_paying_mortgagee_go_to_their_fields():
    gold = _gold(additional_fields=[_extra("Protection Class", "3", [2]), _extra("Paid By", "Mortgagee", [2])])
    notes = apply_label_rules(gold, PAGES)
    assert gold["additional_fields"] == []
    assert gold["locations"][0]["protection_class"]["raw"] == "3"
    assert _kinds(notes).count("extra field moved to its field") == 2      # is_payor was already set


def test_an_extra_field_with_figures_no_page_prints_is_listed_like_a_field():
    gold = _gold(additional_fields=[_extra("Water Leak Detection", "No", [4]),
                                    _extra("Terrorism Premium", "$1,234.00", [2]),
                                    _extra("Terrorism", "Rejected by insured", [2])])
    notes = apply_label_rules(gold, PAGES)
    assert len(gold["additional_fields"]) == 3                          # listed, not deleted
    assert (PRINTED_NOWHERE, "additional_fields[1].value") in notes
    assert (PRINTED_NOWHERE, "additional_fields[2].value") not in notes  # a phrase may be reworded


def test_an_amount_printed_in_another_format_is_printed():
    gold = _gold(additional_fields=[_extra("Fire Fee", "$1,234.00", [2])])
    notes = apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nFire Fee 1234.00"})
    assert [e["label"] for e in gold["additional_fields"]] == ["Fire Fee"]
    assert PRINTED_NOWHERE not in _kinds(notes)


def test_a_street_number_is_not_a_location_number():
    gold = _gold()
    notes = apply_label_rules(gold, PAGES)                 # "Location: 2 TOWN RD" is the address
    assert "location_number" not in gold["locations"][0]
    assert ("number not printed", "locations[0].location_number '2'") in notes


def _one_location(number="1"):
    return [{"unit_id": "loc_1", "location_number": _v(number, [2], int(number)),
             "address": {"street": _v("2 TOWN RD", [2])}}]


def test_a_location_number_printed_as_one_stays():
    gold = _gold(locations=_one_location())
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nProperty: 1 of 1"})
    assert gold["locations"][0]["location_number"]["raw"] == "1"


def test_a_number_in_a_numbered_column_is_printed():
    gold = _gold(locations=_one_location())       # a table's cells come out of the text layer as lines
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nLoc. #\nLocation Description and Address\n1\n2 TOWN RD"})
    assert gold["locations"][0]["location_number"]["raw"] == "1"


def test_a_number_printed_on_a_page_it_does_not_cite_is_printed():
    gold = _gold(locations=_one_location())
    apply_label_rules(gold, {**PAGES, 3: PAGES[3] + "\nCoverage Information for Location 1 of 1"})
    assert gold["locations"][0]["location_number"]["raw"] == "1"


def test_a_field_value_with_figures_no_page_prints_is_listed_not_changed():
    gold = _gold(policy={"policy_number": _v("ZZ-9999-1234", [2]), "effective_date": _v("06/04/2025", [2])})
    before = copy.deepcopy(gold["policy"])
    notes = apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nEffective 06/04/2025"})
    assert (PRINTED_NOWHERE, "policy.policy_number") in notes
    assert gold["policy"] == before


def test_one_missing_figure_on_a_page_of_few_settles_nothing():
    gold = _gold(policy={"policy_number": _v("ZZ-9999-1234", [2])})   # 2 of its 3 figures printed
    assert PRINTED_NOWHERE not in _kinds(apply_label_rules(gold, PAGES))


def test_the_ocr_text_of_a_scanned_page_settles_nothing():
    gold = _gold(policy={"policy_number": _v("ZZ-9999-1234", [2]), "effective_date": _v("06/04/2025", [2])},
                 locations=[{"unit_id": "loc_1", "location_number": _v("1", [2], 1)}])
    notes = apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nEffective 06/04/2025"}, scanned={2})
    assert PRINTED_NOWHERE not in _kinds(notes)                # OCR may have misread it
    assert "location_number" in gold["locations"][0]


def test_scanned_pages_are_those_whose_text_is_drawn_invisibly(tmp_path):
    import pymupdf

    from data_pipeline.ingestion.label_rules import scanned_pages

    pdf = tmp_path / "doc.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "an OCR reading laid over a scan", render_mode=3)
        doc.new_page().insert_text((72, 72), "printed text")
        doc.save(pdf)
    assert scanned_pages(pdf) == {1}
    assert scanned_pages(tmp_path / "missing.pdf") == set()


def test_a_name_is_never_listed_as_printed_nowhere():
    gold = _gold(carrier={"name": _v("Leatherstocking Cooperative Insurance Company", [1])})
    notes = apply_label_rules(gold, PAGES)                  # printed in the logo, not the text layer
    assert PRINTED_NOWHERE not in _kinds(notes)


def test_without_a_text_layer_only_the_rules_needing_no_text_apply():
    gold = _gold(additional_fields=[_extra("Page 3 of 3", "Page 3 of 3", [3])])
    notes = apply_label_rules(gold, {})
    assert _kinds(notes) == ["extra field not a value"]
    assert gold["policy"]["policy_number"]["page_ref"] == [2]
    assert "location_number" in gold["locations"][0]


def test_page_texts_reads_each_pages_text_layer(tmp_path):
    import pymupdf

    pdf = tmp_path / "doc.pdf"
    with pymupdf.open() as doc:
        for text in ("first page NYHP0000004000", "second page"):
            doc.new_page().insert_text((72, 72), text)
        doc.save(pdf)
    texts = page_texts(pdf)
    assert sorted(texts) == [1, 2] and "NYHP0000004000" in texts[1]
    assert page_texts(tmp_path / "missing.pdf") == {}


def test_the_corpus_build_leaves_out_windows_holding_a_value_printed_nowhere():
    from data_pipeline.dataset_builder.build_jsonl import _held_unprinted, _unprinted_values

    label = _gold(policy={"policy_number": _v("ZZ-9999-1234", [2])})
    unprinted = _unprinted_values(label, ["policy.policy_number"])
    assert unprinted == [("policy.policy_number", "policy", {2})]
    decl = SimpleNamespace(group="decl", pages=(1, 2, 3), single=True)
    arrays = SimpleNamespace(group="arrays", pages=(1, 2), single=False)
    assert _held_unprinted(decl, "homeowners", unprinted) == ["policy.policy_number"]
    assert _held_unprinted(arrays, "homeowners", unprinted) == []        # not its section

    label = _gold(additional_fields=[_extra("Terrorism Premium", "$1,234.00", [2])])
    unprinted = _unprinted_values(label, ["additional_fields[0].value"])
    assert unprinted == [("additional_fields[0].value", "additional_fields", {2})]
    assert _held_unprinted(arrays, "homeowners", unprinted) == ["additional_fields[0].value"]
    assert _held_unprinted(decl, "homeowners", unprinted) == []


def test_the_corpus_build_leaves_out_that_window_and_builds_the_others():
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, _policy_window_rows

    texts = [PAGES[number] for number in sorted(PAGES)]
    document = SourceDocument(source_id="s1", doc_type="policy", lob="homeowners", ocr_pages=texts,
                              golden_label=_gold(policy={"policy_number": _v("ZZ-9999-1234", [2])}),
                              image_paths=[f"p{number}.png" for number in sorted(PAGES)],
                              unprinted_values=["policy.policy_number"])
    details: list[str] = []
    rows = _policy_window_rows(document, "train", "ocr_plus_image", texts, details)
    assert any("left out" in d and "policy.policy_number" in d for d in details)
    assert rows and all("ZZ-9999-1234" not in row["messages"][-1]["content"] for row in rows)


def test_a_bundles_metadata_carries_the_values_printed_nowhere():
    from data_pipeline.ingestion.prepare_bundles import _gold_changed, _unprinted

    notes = [("renders identical", "x"), (PRINTED_NOWHERE, "policy.policy_number")]
    assert _unprinted(notes) == ["policy.policy_number"]
    assert not _gold_changed(notes)                       # the gold itself is as delivered
    assert _gold_changed([*notes, ("page list", "a [1] -> [1, 2]")])


def test_corrected_gold_runs_the_rules_on_the_text_layer(tmp_path):
    import pymupdf

    from data_pipeline.ingestion.label_rules import _envelopes
    from data_pipeline.ingestion.prepare_bundles import corrected_gold

    gold = json.loads(Path("configs/canonical schema/common schema/examples/homeowners_minimal.json")
                      .read_text(encoding="utf-8"))
    # Page 1 prints every value the label cites there but the policy number.
    printed = [str(e["raw"]) for path, e in _envelopes(gold)
               if e.get("raw") and 1 in (e.get("page_ref") or []) and path != "policy.policy_number"]
    pdf = tmp_path / "doc.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=2000)
        for line, text in enumerate([*printed, FILLER[:200]]):
            page.insert_text((36, 40 + 14 * line), text, fontsize=9)
        doc.save(pdf)
    _fixed, notes, problem = corrected_gold(gold, "homeowners", carrier="example", text_pdf=pdf, recodes={})
    assert problem is None
    assert (PRINTED_NOWHERE, "policy.policy_number") in notes
