"""Hagerty labels written as personal auto, used as classic auto (common.label_mapping).

Without the mapping every classic-auto line-block window trained on an empty
target: the vehicles, drivers and coverages sat under `auto`, which the
classic-auto schema does not read.
"""

from __future__ import annotations

from common.label_mapping import map_label


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


HAGERTY = {
    "policy": {"policy_number": _env("HG-1")},
    "auto": {
        "vehicles": [{"vin": _env("1G1YY0000"), "body_type": _env("Convertible"),
                      "stated_amount": _env(42000), "garaging_territory": _env("12")}],
        "drivers": [{"name": _env("Jane Rivera"), "driver_number": _env(1)}],
        "liability_coverages": {"bodily_injury_per_person_limit": _env(100000),
                                "uninsured_motorist_bodily_injury_limit": _env(100000),
                                "personal_injury_protection_limit": _env(50000)},
        "no_fault_benefits": {"death_benefit": _env(2000)},
    },
}


def test_the_auto_block_moves_into_classic_auto_with_its_fields_renamed():
    mapped = map_label(HAGERTY, ["classic_auto"])
    ca = mapped["classic_auto"]
    vehicle = ca["vehicles"][0]
    assert vehicle["body_style"]["raw"] == "Convertible" and vehicle["agreed_value"]["parsed"] == 42000
    assert ca["operators"][0]["name"]["raw"] == "Jane Rivera"
    assert ca["liability_coverages"]["uninsured_motorist_limit"]["parsed"] == 100000
    assert "auto" not in mapped and mapped["policy"] == HAGERTY["policy"]


def test_fields_the_classic_auto_schema_lacks_are_dropped_not_guessed():
    ca = map_label(HAGERTY, ["classic_auto"])["classic_auto"]
    assert "garaging_territory" not in ca["vehicles"][0]
    assert "personal_injury_protection_limit" not in ca["liability_coverages"]
    assert "no_fault_benefits" not in ca


def test_the_stored_label_is_never_changed():
    map_label(HAGERTY, ["classic_auto"])
    assert "auto" in HAGERTY and "classic_auto" not in HAGERTY


def test_other_lines_and_labels_already_in_the_target_format_are_untouched():
    assert map_label(HAGERTY, ["personal_auto"]) is HAGERTY
    native = {"classic_auto": {"vehicles": [{"vin": _env("X")}]}, "auto": {"vehicles": []}}
    assert map_label(native, ["classic_auto"]) is native


def test_the_classic_auto_line_block_window_now_has_a_target():
    from data_pipeline.dataset_builder.policy_windows import plan_windows, window_target

    targets = [window_target(HAGERTY, ["classic_auto"], plan)
               for plan in plan_windows(["classic_auto"], [1]) if plan.group == "lineblk"]
    assert targets and targets[0]["classic_auto"]["vehicles"][0]["vin"]["raw"] == "1G1YY0000"


def test_a_gold_label_is_scored_in_the_classic_auto_format():
    from evaluation.run_eval import build_report

    answer = {"classic_auto": {"vehicles": [{"vin": _env("1G1YY0000")}]},
              "policy": {"policy_number": _env("HG-1")}}
    gold = {"policy": HAGERTY["policy"], "auto": {"vehicles": [{"vin": _env("1G1YY0000")}]}}
    meta = {"source_id": "p1", "doc_type": "policy", "lob": ["classic_auto"],
            "modality_mode": "ocr_plus_image"}
    [full] = build_report("t", [(gold, answer, meta)]).full_set()
    assert full.metrics["list_field_recall"] == 1.0
