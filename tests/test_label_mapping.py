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


#: The line-block machinery is the self-contained files' (the common-model lines
#: have no line block), so a self-contained line is the target: commercial auto,
#: whose labels here keep their vehicles under a foreign "fleet" block.
MAPPING = {"commercial_auto": {"from": "fleet", "to": "auto", "sections": {
    "units": {"to": "vehicles", "rename": {"body_type": "body_style_unknown_to_schema"}},
    "operators": {"to": "drivers"},
}}}

LABEL = {
    "policy": {"policy_number": _env("CA-1")},
    "fleet": {"units": [{"vin": _env("1HD1KEF18KB668131"), "year": _env(2019),
                         "engine_displacement": _env("1868")}],
              "operators": [{"name": _env("Jane Rivera")}]},
}


@pytest.fixture(autouse=True)
def mapping(monkeypatch):
    monkeypatch.setattr(label_mapping, "_mappings", lambda: MAPPING)


def test_the_source_block_moves_into_the_line_block():
    mapped = map_label(LABEL, ["commercial_auto"])
    assert mapped["auto"]["vehicles"][0]["vin"]["raw"] == "1HD1KEF18KB668131"
    assert mapped["auto"]["drivers"][0]["name"]["raw"] == "Jane Rivera"
    assert "fleet" not in mapped and mapped["policy"] == LABEL["policy"]


def test_fields_the_target_schema_lacks_are_dropped_not_guessed():
    vehicle = map_label(LABEL, ["commercial_auto"])["auto"]["vehicles"][0]
    assert "engine_displacement" not in vehicle and "year" in vehicle


def test_the_stored_label_is_never_changed():
    map_label(LABEL, ["commercial_auto"])
    assert "fleet" in LABEL and "auto" not in LABEL


def test_other_lines_and_labels_already_in_the_target_format_are_untouched():
    assert map_label(LABEL, ["gl"]) is LABEL
    native = {"auto": {"vehicles": [{"vin": _env("X")}]}, "fleet": {"units": []}}
    assert map_label(native, ["commercial_auto"]) is native


def test_no_mapping_is_configured():
    """Classic auto was the one mapped line; it is personal auto now."""
    from pathlib import Path

    import yaml

    assert not yaml.safe_load(Path(label_mapping.CONFIG).read_text(encoding="utf-8"))
