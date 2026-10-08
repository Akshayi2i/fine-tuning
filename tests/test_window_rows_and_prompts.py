"""What a window is taught and told (accuracy plan, stage 2): a coverage's form
numbers only where its pages print them, and the window prompt's rules.

A training row teaches exactly what its target shows against what its prompt
says. A form number in the target of a window that does not show it is a value
to invent; a prompt that calls form numbers unprinted, or never says which
values other windows read, contradicts the targets it is trained with.
"""

from __future__ import annotations

from common.prompts import render_system_prompt
from data_pipeline.dataset_builder.policy_windows import PolicyWindowPlan, printed_pages_in, window_target


def _v(raw, pages, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": pages}


LABEL = {"coverages": [{"coverage_id": "cov_1", "coverage_code": "HO_COV_A",
                        "coverage_name": _v("Coverage A - Dwelling", [2, 3]),
                        "form_refs": ["HO 04 90", "HO 00 03"],
                        "limits": [{"limit_type": "per_occurrence", "amount": _v("$219,000", [2], 219000.0)}]}]}
TEXTS = {1: "Declarations", 2: "Coverage A - Dwelling $219,000  Forms: HO 04 90",
         3: "Coverage A - Dwelling (continued)"}


def _refs(page, printed):
    plan = PolicyWindowPlan(group="arrays", window_index=page - 2, pages=(page,), single=False)
    return window_target(LABEL, "homeowners", plan, printed_pages=printed)["coverages"][0].get("form_refs")


def test_a_form_number_is_taught_on_the_window_whose_pages_print_it():
    printed = printed_pages_in(TEXTS)
    assert _refs(2, printed) == ["HO 04 90", "HO 00 03"]
    # Page 3 shows the coverage, not "HO 04 90". "HO 00 03" the text shows on no
    # page - OCR may have misread it - so it stays with its row.
    assert _refs(3, printed) == ["HO 00 03"]


def test_without_the_documents_text_form_numbers_ride_with_their_row():
    assert _refs(3, None) == ["HO 04 90", "HO 00 03"]


def test_printed_pages_reads_page_texts_in_order_or_by_number():
    in_order = printed_pages_in(["Declarations", "Forms: HO 04 90"])
    assert in_order("HO 04 90") == {2} and in_order("HO 99 99") == set()
    assert printed_pages_in({7: "Forms: HO 04 90"})("HO 04 90") == {7}


def test_a_split_run_shares_a_page_between_its_windows():
    from data_pipeline.dataset_builder.expand_tasks import _split_overlapping, plan_policy_windows

    assert _split_overlapping(list(range(1, 13)), 6, 1) == [[1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [9, 10, 11, 12]]
    assert _split_overlapping([1, 2, 3], 6, 1) == [[1, 2, 3]]                 # fits: no split, no overlap
    assert _split_overlapping([1, 2, 3], 1, 1) == [[1], [2], [3]]             # no room to share a page
    pages = [1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17]
    # Only the run that has to be split overlaps; the leading pages and whole runs are as before.
    assert plan_policy_windows(pages, pages_per_window=4, leading=[1, 2, 3], overlap=1) == [
        [1, 2, 3], [10, 11, 12, 13], [13, 14, 15], [15, 16, 17]]
    assert plan_policy_windows(pages, pages_per_window=4, leading=[1, 2, 3]) == [
        [1, 2, 3], [10, 11, 12, 13], [14, 15, 16, 17]]


def _prompt(group, lob="homeowners"):
    return render_system_prompt("policy", "ocr_plus_image", None, lob, group)


def test_a_window_with_no_required_keys_reads_as_a_sentence():
    for group in ("arrays", "lineblk"):
        text = _prompt(group)
        assert "Emit  always" not in text and "Emit a key only when the document states something" in text
    assert "Emit `document`, `carrier`, `named_insured` and `policy` always" in _prompt("decl")


def test_the_window_that_decides_extra_fields_is_told_what_other_windows_read():
    arrays = _prompt("arrays")
    assert ("read by other requests and are not shown here: `document`, `carrier`, `producer`, "
            "`named_insured`, `policy`, `lob_parts`, `premium`, `billing` and `forms_and_endorsements`") in arrays
    # Only where the additional fields are decided; and never in a whole-schema prompt.
    assert "read by other requests" not in _prompt("decl")
    assert "read by other requests" not in _prompt(None)


def test_form_numbers_are_described_as_printed_and_the_rule_admits_them():
    arrays = _prompt("arrays")
    assert "Form numbers printed with this coverage" in arrays
    assert "unless its description says it is printed" in arrays
    assert "The value the label introduces, as printed." in arrays          # not "Insured value."


def test_the_common_model_prompt_states_the_conventions():
    text = _prompt("arrays")
    assert "Write it once, in its own field, and list in its `page_ref` every page that prints it." in text
    assert "`page_ref` is the only reference to the source." in text and "source references" not in text
    assert "list only the rows printed on the pages you were given" in text
    assert "A value that has a field of its own goes in that field only" in text
    assert "whether a coverage is included, are values only where the document prints them" in text


def test_a_line_outside_the_common_model_keeps_its_rules():
    text = render_system_prompt("lossrun", "ocr_plus_image")
    assert "notes, or source references. Values only." in text
    assert "Write it once" not in text and "list only the rows printed" not in text
