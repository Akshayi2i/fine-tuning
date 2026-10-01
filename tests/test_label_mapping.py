"""Labels written for one line's schema, used under another's (common.label_mapping).

No mapping is configured today, so the engine is tested against one given here:
a line whose labels keep their vehicles under another line's block.
"""

from __future__ import annotations

import pytest

from common import label_mapping
from common.label_mapping import map_label


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


MAPPING = {"motorcycle": {"from": "auto", "to": "motorcycle", "sections": {
    "vehicles": {"to": "units", "rename": {"body_type": "body_style_unknown_to_schema"}},
    "drivers": {"to": "operators"},
}}}

LABEL = {
    "policy": {"policy_number": _env("MC-1")},
    "auto": {"vehicles": [{"vin": _env("1HD1KEF18KB668131"), "year": _env(2019),
                           "garaging_territory": _env("12")}],
             "drivers": [{"name": _env("Jane Rivera")}]},
}


@pytest.fixture(autouse=True)
def mapping(monkeypatch):
    monkeypatch.setattr(label_mapping, "_mappings", lambda: MAPPING)


def test_the_source_block_moves_into_the_line_block():
    mapped = map_label(LABEL, ["motorcycle"])
    assert mapped["motorcycle"]["units"][0]["vin"]["raw"] == "1HD1KEF18KB668131"
    assert mapped["motorcycle"]["operators"][0]["name"]["raw"] == "Jane Rivera"
    assert "auto" not in mapped and mapped["policy"] == LABEL["policy"]


def test_fields_the_target_schema_lacks_are_dropped_not_guessed():
    unit = map_label(LABEL, ["motorcycle"])["motorcycle"]["units"][0]
    assert "garaging_territory" not in unit


def test_the_stored_label_is_never_changed():
    map_label(LABEL, ["motorcycle"])
    assert "auto" in LABEL and "motorcycle" not in LABEL


def test_other_lines_and_labels_already_in_the_target_format_are_untouched():
    assert map_label(LABEL, ["personal_auto"]) is LABEL
    native = {"motorcycle": {"units": [{"vin": _env("X")}]}, "auto": {"vehicles": []}}
    assert map_label(native, ["motorcycle"]) is native


def test_no_mapping_is_configured():
    """Classic auto was the one mapped line; it is personal auto now."""
    from pathlib import Path

    import yaml

    assert not yaml.safe_load(Path(label_mapping.CONFIG).read_text(encoding="utf-8"))
