"""The SPEC_21 line overlays load as one self-contained schema each (the bundle).

An overlay is a thin file whose blocks ``$ref`` the common model in another file.
Every reader in this repo - the validator, the section slicer, the prompt, the
every-key fill - expects one self-contained schema, so the registry bundles the
overlay at load time. These tests pin what the bundle is and what it refuses.
"""

import json
import logging
import re

import pytest

from common import schemas as S

OVERLAY_LINES = (
    "homeowners", "personal_auto", "dwelling_fire", "ocean_marine",
    "motorcycle", "recreational_vehicle", "personal_umbrella",
)
EXAMPLE_GOLD = S.CANONICAL_ROOT / "common schema" / "examples" / "homeowners_minimal.json"


def _overlay(lob: str) -> dict:
    return json.loads((S.CANONICAL_DIR / f"{lob}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("lob", OVERLAY_LINES)
def test_an_overlay_is_registered_as_a_common_model_line(lob):
    assert S.is_common_model("policy", lob=lob)


def test_the_self_contained_lines_are_not_common_model_lines():
    # General liability's family-mates still use self-contained files, as do
    # the lines of the families not yet moved.
    for lob in ("auto", "wc", "umbrella", "property", "cyber"):
        assert not S.is_common_model("policy", lob=lob), lob


def test_general_liability_and_the_fallback_compose_the_common_model():
    """gl.json is a SPEC_21 overlay since its 3.0.0, and _fallback.json - the
    schema of a policy of no known line - composes common model 1.1.0 too."""
    assert S.is_common_model("policy", lob="gl")
    assert S.is_common_model("policy")  # the fallback
    assert S.schema_version("policy", lob="gl") == "3.0.0+common.1.1.0"
    assert S.schema_version("policy") == "1.1.0+common.1.1.0"


@pytest.mark.parametrize("lob", OVERLAY_LINES)
def test_a_bundle_has_no_reference_outside_its_own_definitions(lob):
    text = json.dumps(S.load_schema("policy", lob=lob))
    refs = re.findall(r'"\$ref": "([^"]+)"', text)
    assert refs and all(ref.startswith("#/$defs/") for ref in refs)
    definitions = S.load_schema("policy", lob=lob)["$defs"]
    assert {ref.split("/")[2] for ref in refs} <= set(definitions)


@pytest.mark.parametrize("lob", OVERLAY_LINES)
def test_a_bundle_version_names_the_overlay_and_the_common_model(lob):
    assert S.schema_version("policy", lob=lob) == "1.0.0+common.1.1.0"


def test_the_spec21_example_gold_validates_against_its_bundle():
    gold = json.loads(EXAMPLE_GOLD.read_text(encoding="utf-8"))
    assert list(S.iter_validation_errors(gold, "policy", lob="homeowners")) == []


def test_a_gold_in_the_old_shape_is_refused_with_readable_errors():
    envelope = {"raw": "X", "parsed": "X", "confidence": {"score": 1.0, "source": "vlm"},
                "page_ref": [1], "flagged": False}
    old = {"carrier": {"company_name": envelope}, "named_insured": {}, "policy": {},
           "homeowners": {"dwelling": {}}}
    errors = list(S.iter_validation_errors(old, "policy", lob="homeowners"))
    assert any("homeowners" in e for e in errors)
    assert any("company_name" in e for e in errors)


def test_a_bundle_carries_each_coverage_codes_meaning():
    names = S.load_schema("policy", lob="homeowners")["fideon:coverage_code_names"]
    assert list(names) == _overlay("homeowners")["fideon:coverage_codes"]
    assert names["HO_COV_A"] and names["X_MED_PAY"]


def test_a_reference_into_another_file_is_refused():
    overlay = _overlay("homeowners")
    overlay["properties"]["carrier"] = {"$ref": "../common/other_model.json#/$defs/carrier"}
    with pytest.raises(S.SchemaError, match="outside the common model"):
        S._bundle(overlay, S.CANONICAL_DIR / "homeowners.json")


def test_a_pointer_outside_the_definitions_is_refused():
    overlay = _overlay("homeowners")
    overlay["properties"]["carrier"] = {"$ref": "../common/common_model.json#/properties/x"}
    with pytest.raises(S.SchemaError, match="outside the common model"):
        S._bundle(overlay, S.CANONICAL_DIR / "homeowners.json")


def test_an_overlay_written_against_another_common_model_version_is_refused():
    overlay = _overlay("homeowners")
    overlay[S.COMMON_MODEL_VERSION_KEY] = "9.9.9"
    with pytest.raises(S.SchemaError, match="9.9.9"):
        S._bundle(overlay, S.CANONICAL_DIR / "homeowners.json")


def test_a_code_file_listing_other_codes_is_refused(tmp_path):
    overlay = _overlay("personal_umbrella")
    (tmp_path / overlay["fideon:coverage_codes_file"]).write_text(
        "codes:\n  - {code: PU_SOMETHING_ELSE, name: Something}\n", encoding="utf-8"
    )
    with pytest.raises(S.SchemaError, match="different coverage codes"):
        S._bundle(overlay, tmp_path / "personal_umbrella.json")


def test_a_code_without_a_meaning_is_refused(tmp_path):
    overlay = _overlay("personal_umbrella")
    codes = "".join(f"  - {{code: {c}}}\n" for c in overlay["fideon:coverage_codes"])
    (tmp_path / overlay["fideon:coverage_codes_file"]).write_text(f"codes:\n{codes}", encoding="utf-8")
    with pytest.raises(S.SchemaError, match="no name"):
        S._bundle(overlay, tmp_path / "personal_umbrella.json")


def test_a_family_only_partly_on_the_common_model_still_loads(caplog):
    """A family moves line by line: general liability is on the common model
    while auto, workers' comp and umbrella are not yet. Every schema still loads,
    so the other families' work goes on, and the load says which lines lag."""
    with caplog.at_level(logging.WARNING, logger=S.__name__):
        S._check_overlay_families(["homeowners", "personal_auto"])
    assert "'personal_lines' is only partly on the common model" in caplog.text
    assert "dwelling_fire" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=S.__name__):
        S._sources.cache_clear()
        sources = S._sources()
    assert "'casualty_fleet' is only partly on the common model" in caplog.text
    assert sources["policy:gl"].common_model
    assert not any(sources[f"policy:{lob}"].common_model for lob in ("auto", "wc", "umbrella"))


def test_the_partly_migrated_families_name_the_lines_still_behind():
    assert S.partly_migrated_families() == {"casualty_fleet": ("auto", "umbrella", "wc")}


def test_a_scope_whose_corpus_holds_both_shapes_of_one_family_is_refused():
    """One adapter cannot train on two output shapes: general liability on the
    common model and auto on a self-contained file, in one corpus."""
    from common.scopes import ScopeError, assert_one_output_shape, get_scope

    with pytest.raises(ScopeError, match="'casualty_fleet' on two output shapes"):
        assert_one_output_shape(get_scope("unified"), ["gl", "auto"])
    with pytest.raises(ScopeError, match="two output shapes"):
        assert_one_output_shape(get_scope("policy"), ["homeowners", "gl", "wc"])
    # Without the corpus's lines, covering both kinds of line is enough.
    with pytest.raises(ScopeError, match="two output shapes"):
        assert_one_output_shape(get_scope("unified"))


def test_a_scope_whose_corpus_holds_one_shape_of_each_family_trains():
    from common.scopes import assert_one_output_shape, get_scope

    assert_one_output_shape(get_scope("unified"), ["homeowners", "gl"])
    assert_one_output_shape(get_scope("unified"), ["auto", "wc", "umbrella"])
    assert_one_output_shape(get_scope("personal_lines"))
    # A line-scoped scope trains only its own lines, whatever else the corpus holds.
    assert_one_output_shape(get_scope("personal_lines"), ["homeowners", "gl", "auto"])
    # A scope that trains no policy has no line to train two ways.
    assert_one_output_shape(get_scope("lossrun"), ["gl", "auto"])


def test_an_overlay_in_no_family_is_refused():
    with pytest.raises(S.SchemaError, match="no layout family"):
        S._check_overlay_families(["not_a_line"])


def test_commercial_auto_reads_the_clients_auto_file():
    assert S.schema_key("policy", lob="commercial_auto") == "policy:auto"


def test_classic_auto_is_not_a_line_of_its_own():
    assert "policy:classic_auto" not in S._sources()
    assert S.schema_key("policy", lob="classic_auto") == "policy:personal_auto"


@pytest.mark.parametrize("lob", [None, "gl", "wc", "auto"])
def test_no_prompt_asks_for_the_full_text_tier(lob):
    from common.prompts import schema_json_for_prompt

    assert "text_sections" not in schema_json_for_prompt("policy", lob=lob)


def test_the_prompt_hash_covers_the_common_model_and_the_code_files():
    from common.prompts import prompt_input_files

    files = prompt_input_files()
    assert S.COMMON_MODEL in files
    for lob in OVERLAY_LINES:
        assert S.CANONICAL_DIR / f"{lob}.coverage_codes.yaml" in files
