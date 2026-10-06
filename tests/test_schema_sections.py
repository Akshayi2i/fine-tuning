"""The section map: which canonical sections one extraction call asks for.

These guard a failure that is silent by construction. A section no window asks
for produces null fields in an extraction that still validates against the
client's schema — there is no exception, no warning, and the only evidence is a
policy that came back emptier than it should have.
"""

from __future__ import annotations

import json

import pytest

from common import schemas
from common.schema_sections import (
    SectionMapError,
    array_key,
    assert_sections_cover_the_schema,
    group_names,
    groups_for,
    pages_for,
    sections_for,
)


def _registered_lobs() -> list[str | None]:
    """Every policy schema, the fallback included (as ``None``)."""
    out: list[str | None] = []
    for key in schemas._sources():
        if key == "policy":
            out.append(None)
        elif key.startswith("policy:"):
            out.append(key.split(":", 1)[1])
    return out


@pytest.mark.parametrize("lob", _registered_lobs())
def test_every_section_is_asked_for_by_exactly_one_group(lob):
    """The check that keeps the map honest when the client ships a new schema.

    A section in no group is a field no window requests: it comes back null and
    the extraction still validates. A section in two groups is two windows both
    claiming to own it, and a merge conflict that should not exist.
    """
    assert_sections_cover_the_schema(lob)


@pytest.mark.parametrize("lob", _registered_lobs())
def test_a_slice_is_a_subset_of_the_whole_schema(lob):
    from common.schema_sections import _excluded, _profile

    # Annotations and (on the common-model lines) the full-text tier are asked
    # for by no group, by design.
    whole = {name for name in schemas.load_schema("policy", None, lob).get("properties") or {}
             if not _excluded(name, _profile(lob))}
    seen: set[str] = set()
    for group in groups_for(lob):
        sections = set(sections_for(group, lob))
        assert sections <= whole, f"{group} names sections {lob} does not have"
        assert not (sections & seen), f"{group} overlaps an earlier group"
        seen |= sections
    assert seen == whole, f"unclaimed on {lob}: {sorted(whole - seen)}"


def test_a_group_with_nothing_to_ask_for_is_not_a_window():
    """A window whose schema has no properties asks the model for `{}` — a
    wasted call whose output cannot be told apart from a page holding nothing.

    document_type_detail is carried by a few self-contained lines only, and the
    fallback schema has no line-specific block at all.
    """
    assert "dtd" in group_names()
    assert "dtd" in groups_for("flood")
    assert "dtd" not in groups_for("gl")
    # The fallback has no line block; its remainder window reads only the coverage
    # and overflow lists the delivered files added to every self-contained schema.
    assert sections_for("lineblk", None) == ("coverages", "text_sections", "additional_fields")


def test_the_remainder_group_claims_a_section_the_map_never_names():
    """`lineblk` is defined as the remainder, not as a list.

    commercial auto's line block is called `auto`, general liability's
    `general_liability`, and the delivered files added sections this file has
    never heard of. Naming them would mean the first unlisted one was silently
    dropped — which is the failure the whole map exists to prevent.
    """
    added = ("coverages", "text_sections", "additional_fields")
    assert sections_for("lineblk", "commercial_auto") == ("auto", *added)
    assert sections_for("lineblk", "gl") == ("general_liability", *added)


def test_a_slice_narrows_required_rather_than_keeping_it_whole():
    """Left whole, structured decoding would force an `arrays` window to emit
    carrier, named_insured and policy as `{}`. Two windows would then both own
    carrier, and on the training side every schedule target would teach the
    model to emit an empty one."""
    whole = schemas.load_schema("policy", None, "gl")["required"]
    assert set(whole) == {"carrier", "named_insured", "policy"}

    decl = schemas.load_schema("policy", None, "gl", "decl")
    assert set(decl["required"]) == set(whole), "all three are declarations sections"

    arrays = schemas.load_schema("policy", None, "gl", "arrays")
    assert "required" not in arrays, "no required section survives into arrays"
    assert set(arrays["properties"]) == set(sections_for("arrays", "gl"))


def test_a_sliced_key_resolves_its_file_on_the_base():
    """Two slices of a line are two views of one file at one version.

    Everything that addresses the FILE — where it lives, which version it
    declares, whether it is canonical — has to strip the slice first, or it
    KeyErrors on a key no source table contains.
    """
    assert schemas.schema_key("policy", None, "gl", "arrays") == "policy:gl#arrays"
    assert schemas.base_key("policy:gl#arrays") == "policy:gl"
    assert schemas.slice_of("policy:gl#arrays") == "arrays"
    assert schemas.slice_of("policy:gl") is None

    for group in groups_for("gl"):
        assert schemas.schema_version("policy", None, "gl", group) == "1.5.0"
        assert schemas.is_canonical("policy", None, "gl")


def test_every_slice_still_carries_the_fieldvalue_definition():
    """Each leaf still `$ref`s the envelope. A slice whose refs do not resolve
    is not a schema."""
    for group in groups_for("gl"):
        sliced = schemas.resolved_schema("policy", None, "gl", group)
        assert "FieldValue" in json.dumps(sliced.get("$defs", {}))


def test_no_slice_leaks_a_fideon_key():
    """The strip has to survive slicing — the aliases sit inside the sections."""
    for lob in ("gl", "flood"):
        for group in groups_for(lob):
            assert "fideon:" not in schemas.schema_text("policy", None, lob, group)


# --------------------------------------------------------------------------
# Page rules
# --------------------------------------------------------------------------


def test_the_declarations_group_follows_the_declarations_page():
    """Scanned intake routinely opens with a fax cover sheet, a billing notice
    or a broker letter, which puts the real declarations on page 4 or 5.

    Pinned to pages 1-3, nothing would ask for the policy-level fields on
    exactly the documents most likely to be messy, and every one of them would
    come back null.
    """
    routed = [1, 2, 3, 5, 40, 41]
    assert pages_for("decl", routed) == [1, 2, 3]
    assert pages_for("decl", routed, declarations_page=5) == [1, 2, 3, 5]
    # A detected page outside the routed set is not invented into it.
    assert pages_for("decl", routed, declarations_page=99) == [1, 2, 3]


def test_the_array_groups_read_every_routed_page():
    """A location schedule can be printed anywhere in a 200-page policy, and
    guessing where costs a whole table."""
    routed = [1, 2, 3, 12, 140, 141, 180]
    assert pages_for("arrays", routed) == routed
    assert pages_for("lineblk", routed) == routed


def test_the_declarations_group_is_never_given_an_empty_page_set():
    """A routed set with no leading page still has to be asked for the
    policy-level fields somewhere."""
    assert pages_for("decl", [40, 41, 42]) == [40]


def test_an_unknown_group_is_refused_by_name():
    with pytest.raises(SectionMapError, match="no section group"):
        sections_for("schedules_probably", "homeowners")


# --------------------------------------------------------------------------
# What slicing buys
# --------------------------------------------------------------------------


def test_slicing_leaves_room_for_more_pages_than_the_whole_schema():
    """The reason the cross product is affordable at all.

    Every group is asked over every page that could carry it, which is only
    cheap because a sliced prompt is small enough to take more pages per call.
    """
    from common.config import sequence_for_task, vision_for_task
    from common.prompts import render_system_prompt
    from data_pipeline.dataset_builder.cap_check import (
        CHARS_PER_TOKEN,
        TEMPLATE_OVERHEAD_TOKENS,
    )

    budget = sequence_for_task("extract", "policy")
    per_page = vision_for_task("extract")["max_pixels"] // 1024 + 700

    def capacity(lob: str, group: str | None) -> int:
        prompt = len(
            render_system_prompt("policy", "ocr_plus_image", None, lob, group)
        ) / CHARS_PER_TOKEN
        room = (
            budget["max_seq_len"]
            - budget["max_output_tokens"]
            - TEMPLATE_OVERHEAD_TOKENS
            - prompt
        )
        return max(1, int(room // per_page))

    for lob in ("gl", "flood"):
        whole = capacity(lob, None)
        for group in groups_for(lob):
            assert capacity(lob, group) > whole, (
                f"{lob}/{group} is no smaller than the whole schema, so slicing bought nothing"
            )


def test_every_shared_array_has_an_identifying_key():
    """An array is asked over several page windows, so a table spanning a
    boundary comes back twice. De-duplication is load-bearing, not tidying, and
    it needs a key per array."""
    for section in sections_for("arrays", "gl"):
        assert array_key(section), f"{section} has no identifying key to de-duplicate on"


# --------------------------------------------------------------------------
# The common-model lines (configs/schema_sections.yaml `common_model`)
# --------------------------------------------------------------------------

COMMON_MODEL_LINES = (
    "homeowners", "personal_auto", "dwelling_fire", "ocean_marine",
    "motorcycle", "recreational_vehicle", "personal_umbrella",
)


@pytest.mark.parametrize("lob", COMMON_MODEL_LINES)
def test_a_common_model_line_is_read_in_three_groups(lob):
    assert groups_for(lob) == ("decl", "arrays", "lineblk")
    # The remainder is the forms list: no line block, and never an annotation
    # or the full-text tier.
    assert sections_for("lineblk", lob) == ("forms_and_endorsements",)
    for group in groups_for(lob):
        for section in sections_for(group, lob):
            assert not section.startswith("fideon:") and section != "text_sections"


@pytest.mark.parametrize("lob", COMMON_MODEL_LINES)
def test_every_table_that_refers_to_a_unit_is_read_with_the_units(lob):
    from common.schema_sections import references

    arrays = set(sections_for("arrays", lob))
    for table in ("coverages", "deductibles", "interested_parties", "rating_modifiers"):
        if table in schemas.load_schema("policy", None, lob)["properties"]:
            assert table in arrays
    assert set(references(lob)["applies_to"]) & set(schemas.load_schema("policy", None, lob)["properties"]) \
        <= arrays


@pytest.mark.parametrize("lob", COMMON_MODEL_LINES)
def test_no_window_is_shown_a_reference_it_cannot_resolve(lob):
    """A reference field stays in a window's schema only if a table it can name
    is read in that window."""
    from common.schema_sections import references

    for group in groups_for(lob):
        present = set(sections_for(group, lob))
        text = schemas.schema_text("policy", None, lob, group)
        for field, tables in references(lob).items():
            if f'"{field}"' in text:
                assert present & set(tables), f"{lob}/{group} shows {field} but reads none of {tables}"


@pytest.mark.parametrize("lob", COMMON_MODEL_LINES)
def test_every_common_model_slice_compiles_and_is_one_xgrammar_supports(lob):
    from jsonschema import Draft202012Validator

    from tests.test_vllm_patches import _xgrammar_unsupported

    for group in groups_for(lob):
        sliced = schemas.resolved_schema("policy", None, lob, group)
        Draft202012Validator.check_schema(sliced)
        assert not _xgrammar_unsupported(sliced), f"{lob}/{group}"
        assert "fideon:" not in json.dumps(sliced)


def test_each_table_holds_its_own_id_prefix():
    defs = schemas.resolved_schema("policy", None, "personal_auto", "arrays")["$defs"]
    assert defs["Vehicle"]["properties"]["unit_id"]["pattern"] == "^veh_[0-9]+$"
    assert defs["Driver"]["properties"]["unit_id"]["pattern"] == "^drv_[0-9]+$"
    assert defs["Location"]["properties"]["unit_id"]["pattern"] == "^loc_[0-9]+$"


@pytest.mark.parametrize("lob", COMMON_MODEL_LINES)
def test_a_schedule_window_still_holds_several_pages(lob):
    from data_pipeline.dataset_builder.expand_tasks import pages_per_extraction_call

    assert pages_per_extraction_call("policy", None, lob, "arrays") >= 2


def test_a_common_model_group_keeps_its_task_and_page_rule():
    from common.schema_sections import reads_declarations, task_for

    assert task_for("decl") == "policy_declarations" and reads_declarations("decl")
    assert task_for("arrays") == "policy_schedule" and not reads_declarations("arrays")
    assert group_names("homeowners") == ("decl", "arrays", "lineblk")
    assert group_names("gl") == group_names()


def test_the_common_model_keys_are_their_own():
    assert array_key("interested_parties", "homeowners") == ("role", "name")
    assert array_key("interested_parties", "gl") == ("name", "party_type")
