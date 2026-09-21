"""SPEC_01 acceptance criteria for the schema and prompt contracts.

These guard failures that are otherwise silent — a schema that stops requiring
``line_of_business``, a field whose gloss goes missing, an alias string leaking
into a prompt. None of them would surface as an exception at training time; they
would surface as a worse model, weeks later, with no obvious cause.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common import aliases, prompts, schemas
from common.constants import ACORD_FORMS, ACTIVE_DOC_TYPES, MODALITY_MODES, canonical_key, canonical_model
from common.lob import lob_values

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_STEMS = {
    ("lossrun", None): "lossrun",
    ("policy", None): "policy_doc",
    ("acord", "25"): "acord25",
    ("acord", "125"): "acord125",
    ("acord", "140"): "acord140",
}
ALL_KEYS = list(EXAMPLE_STEMS)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_schema_loads_and_refs_resolve(doc_type, acord_form):
    resolved = schemas.resolved_schema(doc_type, acord_form)
    assert resolved["type"] == "object"
    assert json.dumps(resolved), "resolved schema must be JSON-serialisable for prompt injection"
    assert "$ref" not in json.dumps(resolved), "every $ref must be inlined — the model cannot follow a pointer"


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_every_field_has_a_description(doc_type, acord_form):
    """Descriptions are prompt text (arch §0c), not documentation.

    A field without one gives the model no semantic anchor, which is exactly what
    the canonical-mapping design depends on.
    """
    schemas.assert_all_fields_described(doc_type, acord_form)


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_line_of_business_is_a_required_list_of_enum_values(doc_type, acord_form):
    """The VLM is the fallback LoB detector when L1/L2 miss (arch §0b), and it is
    a LIST under v2.1: a certificate or package policy routinely covers several
    lines, and the v1 scalar forced the annotator to pick one and discard the
    rest — which taught the model to do the same."""
    resolved = schemas.resolved_schema(doc_type, acord_form)
    lob = resolved["properties"]["line_of_business"]
    assert "line_of_business" in resolved["required"]
    assert lob["type"] == "array", "a document can cover several lines"
    assert set(lob["items"]["enum"]) == set(lob_values())
    assert lob.get("uniqueItems") is True, "the same line twice is a labelling error"


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_an_empty_lob_list_is_valid_and_means_undetermined(doc_type, acord_form):
    """Undetermined is a correct answer, not a missing one. Under v1 that was
    `null`; under a list it is `[]`, and it must stay representable or the model
    learns to guess a line rather than decline."""
    path = ROOT / "schemas" / "examples" / f"{EXAMPLE_STEMS[(doc_type, acord_form)]}.example.json"
    inst = json.loads(path.read_text(encoding="utf-8"))
    inst["line_of_business"] = []
    schemas.validate(inst, doc_type, acord_form)


def test_lines_outside_the_enum_go_to_their_own_field():
    """Inland marine, cyber, crime and EPLI are real lines this enum does not
    support. Recorded rather than dropped, because the count of what lands here
    is the evidence for whether the enum should grow — and it is a list, because
    a document can name several (v2.1 correction)."""
    inst = json.loads((ROOT / "schemas/examples/lossrun.example.json").read_text(encoding="utf-8"))
    inst["line_of_business_other"] = ["inland_marine", "cyber"]
    schemas.validate(inst, "lossrun")


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_schema_validates_its_example(doc_type, acord_form):
    path = ROOT / "schemas" / "examples" / f"{EXAMPLE_STEMS[(doc_type, acord_form)]}.example.json"
    schemas.validate(json.loads(path.read_text(encoding="utf-8")), doc_type, acord_form)


def test_schema_rejects_out_of_enum_lob():
    inst = json.loads((ROOT / "schemas/examples/lossrun.example.json").read_text(encoding="utf-8"))
    inst["line_of_business"] = ["marine_cargo"]
    assert not schemas.is_valid(inst, "lossrun")


def test_schema_rejects_the_v1_scalar_shape():
    """A corpus row still carrying the v1 scalar must fail loudly here rather
    than reaching training, where it would be one silently malformed target."""
    inst = json.loads((ROOT / "schemas/examples/lossrun.example.json").read_text(encoding="utf-8"))
    inst["line_of_business"] = "workers_comp"
    assert not schemas.is_valid(inst, "lossrun")


def test_schema_rejects_a_duplicated_line():
    inst = json.loads((ROOT / "schemas/examples/lossrun.example.json").read_text(encoding="utf-8"))
    inst["line_of_business"] = ["property", "property"]
    assert not schemas.is_valid(inst, "lossrun")


def test_schema_rejects_unexpected_field():
    inst = json.loads((ROOT / "schemas/examples/policy_doc.example.json").read_text(encoding="utf-8"))
    inst["surface_label_leaked_in"] = "Applicant"
    assert not schemas.is_valid(inst, "policy")


def test_shared_fields_do_not_drift_between_schemas():
    """The whole reason ``common_fields.json`` exists.

    Defining ``insured_name`` per-schema lets two definitions of the same
    canonical field diverge, and nothing catches it — each schema still validates
    on its own. This is the test that would.
    """
    per_schema = {
        key: dict(schemas.iter_described_fields(schemas.resolved_schema(*key)))
        for key in ALL_KEYS
    }
    shared = set.intersection(*(set(d) for d in per_schema.values()))
    assert {"insured_name", "line_of_business"} <= shared

    for field in shared:
        descriptions = {json.dumps(d[field], sort_keys=True) for d in per_schema.values()}
        assert len(descriptions) == 1, f"{field!r} has drifted between schemas: {descriptions}"


def test_acord_requires_a_form_to_select_a_schema():
    """Two-level classification (arch §4b) — the form picks the schema."""
    with pytest.raises(schemas.SchemaError, match="acord_form"):
        schemas.load_schema("acord")
    with pytest.raises(schemas.SchemaError, match="unknown ACORD form"):
        schemas.load_schema("acord", "999")


# --------------------------------------------------------------------------
# Prompts — the parity-critical path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_noisy_ocr_renders_identically_to_ocr_plus_image(doc_type, acord_form):
    """The noise lives in the data, not the instruction (arch §6).

    A distinct prompt for noisy OCR would teach the model that bad OCR is
    announced — and at inference nothing announces it.
    """
    a = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    b = prompts.render_system_prompt(doc_type, "noisy_ocr_image", acord_form)
    assert a == b


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_image_only_prompt_declares_the_absence(doc_type, acord_form):
    """Explicit declaration, not a silently missing block (arch §6).

    Asserts the property rather than one sentence: the image-only prompt must
    say that no OCR is given, **and** must not carry the OCR-precedence
    instructions. A prompt that merely dropped the OCR block would leave the
    model with no statement about what it is reading from.
    """
    rendered = prompts.render_system_prompt(doc_type, "image_only", acord_form)
    ocr_mode = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)

    assert "no OCR text" in rendered
    assert "primary source" not in rendered, "image-only carries OCR-precedence rules"
    assert "primary source" in ocr_mode
    assert rendered != ocr_mode


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_prompt_carries_descriptions_but_no_alias_strings(doc_type, acord_form):
    """The gloss goes in; the alias registry never does (master §1.4).

    An alias list in the prompt gives the model a lexical prior that makes
    confusable errors worse, and would make every newly observed alias a retrain
    trigger.
    """
    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    resolved = schemas.resolved_schema(doc_type, acord_form)
    some_description = resolved["properties"]["line_of_business"]["description"]
    assert some_description[:40] in rendered, "field descriptions must reach the model"

    registry_doc_type = "acord" if doc_type == "acord" else doc_type
    for field, entry in aliases.load_registry(registry_doc_type).items():
        for alias in entry.aliases:
            # A one-word alias may legitimately collide with ordinary prose
            # ("Insured", "Company"); multi-word aliases are unambiguous.
            if " " in alias:
                assert alias not in rendered, f"alias {alias!r} leaked into the prompt for {field}"


@pytest.mark.parametrize("modality_mode", MODALITY_MODES)
def test_prompt_rendering_is_deterministic(modality_mode):
    """Prompt parity is meaningless if rendering is not reproducible."""
    a = prompts.render_system_prompt("policy", modality_mode)
    b = prompts.render_system_prompt("policy", modality_mode)
    assert a == b
    assert prompts.prompt_fingerprint("policy", modality_mode) == prompts.prompt_fingerprint("policy", modality_mode)


def test_unknown_modality_mode_raises():
    with pytest.raises(prompts.PromptError, match="unknown modality_mode"):
        prompts.render_system_prompt("policy", "ocr_only")


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

def test_doc_type_to_canonical_mapping():
    """master §1.2 — the two vocabularies resolve in exactly one place."""
    assert canonical_key("lossrun") == "loss_run"
    assert canonical_model("lossrun") == "LossRunDocument"
    assert canonical_key("policy") == "policy_check"
    assert canonical_key("acord") == "acord_mapping"


def test_deferred_doc_type_is_refused():
    """Quote is designed for but explicitly not implemented (master §1)."""
    with pytest.raises(KeyError, match="deferred"):
        canonical_key("quote")


def test_active_types_and_forms_are_what_the_specs_say():
    assert set(ACTIVE_DOC_TYPES) == {"acord", "policy", "lossrun"}
    assert {"25", "125", "140"} == ACORD_FORMS


# --------------------------------------------------------------------------
# Prompt structure — invariants, not prose
# --------------------------------------------------------------------------

PROMPT_SECTIONS = ("# Output", "# Resolving fields", "# Distinct roles",
                   "# Value formats", "# Absence", "# Repeating tables")


@pytest.mark.parametrize("modality_mode", ["ocr_plus_image", "image_only"])
def test_every_rule_section_survives_in_both_modes(modality_mode):
    """Only the Evidence block is modality-specific. A rule that exists in one
    mode and not the other trains two different behaviours for one task."""
    rendered = prompts.render_system_prompt("policy", modality_mode)
    for section in PROMPT_SECTIONS:
        assert section in rendered, f"{section} missing from the {modality_mode} prompt"


def test_the_two_modes_differ_only_in_the_evidence_block():
    """Guards the `{% if %}` branch: editing one arm and not the other is how
    the modes silently diverge on a rule that has nothing to do with OCR."""
    ocr = prompts.render_system_prompt("policy", "ocr_plus_image")
    image = prompts.render_system_prompt("policy", "image_only")

    assert ocr.split("# Output", 1)[1] == image.split("# Output", 1)[1]
    assert ocr.split("# Evidence", 1)[0] == image.split("# Evidence", 1)[0]


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_the_schema_is_the_last_thing_in_the_prompt(doc_type, acord_form):
    """Instructions before data. The schema is the largest block by far, and
    rules placed after it are read as commentary on it."""
    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    assert rendered.index("Schema:") > rendered.index("# Repeating tables")


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_the_prompt_stays_inside_its_token_budget(doc_type, acord_form):
    """This prompt is the conditioning input on **every** training row and every
    request, so its length is a recurring cost rather than a one-off. The
    ceiling is deliberately loose; it exists so that doubling the prompt is a
    decision someone makes rather than one that accumulates."""
    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    approx_tokens = len(rendered) / 4
    assert approx_tokens < 4000, (
        f"{doc_type} prompt is ~{approx_tokens:.0f} tokens. Rules and schema descriptions both "
        "cost tokens on every example — trim before raising this."
    )


# --------------------------------------------------------------------------
# Per-document-type rules (SPEC_12 §2)
# --------------------------------------------------------------------------

DOC_TYPE_BLOCK = "# This document type"


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_every_type_carries_its_own_rules_block(doc_type, acord_form):
    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    assert DOC_TYPE_BLOCK in rendered


def test_every_active_doc_type_has_a_rules_file():
    """A missing partial raises TemplateNotFound at render time rather than
    rendering an empty block — but only if someone renders that type. This
    fails at the moment a type is added instead."""
    from common.constants import ACTIVE_DOC_TYPES

    for doc_type in ACTIVE_DOC_TYPES:
        assert (prompts.PROMPT_DIR / "doc_types" / f"{doc_type}.jinja").exists(), (
            f"{doc_type} has no per-type rules block"
        )


def test_the_type_blocks_are_actually_different():
    """Three identical blocks would be three copies of the shared rules with
    extra tokens and no per-type guidance."""
    blocks = {
        doc_type: prompts.render_system_prompt(doc_type, "ocr_plus_image",
                                               "25" if doc_type == "acord" else None)
                        .split(DOC_TYPE_BLOCK, 1)[1].split("Schema:", 1)[0]
        for doc_type in ("policy", "lossrun", "acord")
    }
    assert len(set(blocks.values())) == 3


def test_each_acord_form_gets_its_own_paragraph():
    """One adapter, but a distinct schema per form (arch §4b) — so the prose
    describing the form must move with the schema, not with the type."""
    rendered = {
        form: prompts.render_system_prompt("acord", "ocr_plus_image", form)
        for form in ("25", "125", "140")
    }
    bodies = {f: t.split(DOC_TYPE_BLOCK, 1)[1].split("Schema:", 1)[0] for f, t in rendered.items()}
    assert len(set(bodies.values())) == 3


def test_the_form_paragraph_travels_with_the_form_schema():
    """The same `acord_form` selects both, so a prompt cannot describe one form
    while carrying another's schema."""
    for form in ("25", "125", "140"):
        rendered = prompts.render_system_prompt("acord", "ocr_plus_image", form)
        schema = rendered.split("Schema:", 1)[1]
        assert f"acord{form}" in schema or f"acord_{form}" in schema or form in schema[:400]


@pytest.mark.parametrize("modality_mode", ["ocr_plus_image", "image_only"])
def test_the_type_block_is_identical_across_modalities(modality_mode):
    """Type structure does not change with the evidence available."""
    reference = prompts.render_system_prompt("lossrun", "ocr_plus_image")
    rendered = prompts.render_system_prompt("lossrun", modality_mode)
    assert (
        rendered.split(DOC_TYPE_BLOCK, 1)[1] == reference.split(DOC_TYPE_BLOCK, 1)[1]
    )


def test_the_stated_count_rule_survives_in_the_lossrun_block():
    """SPEC_09's row-completeness check compares the model's row count against a
    count the *document* states. A model that computes that count from its own
    rows makes the cross-check compare a number to itself, and the one signal
    that can see a dropped row goes silent."""
    rendered = prompts.render_system_prompt("lossrun", "ocr_plus_image")
    block = rendered.split(DOC_TYPE_BLOCK, 1)[1]
    assert "never computed from the rows you produced" in block


TYPE_BLOCK_SECTIONS = ("## What you are given", "## Where the values are",
                       "## How to work through it", "## Rules for this document type",
                       "## Output for this document type")


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_each_type_block_carries_the_full_set_of_sections(doc_type, acord_form):
    """Input shape, value locations, an ordered procedure, the binding rules and
    the output shape — every type answers all five, or the types are not
    comparable to each other."""
    block = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form) \
                   .split(DOC_TYPE_BLOCK, 1)[1]
    for section in TYPE_BLOCK_SECTIONS:
        assert section in block, f"{doc_type} {acord_form or ''} is missing {section}"


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_the_output_shape_is_derived_from_the_schema(doc_type, acord_form):
    """A hand-written key list in a prompt file drifts the first time a field is
    added, and the drift is silent: the model is shown one shape and validated
    against another."""
    from common.schemas import resolved_schema

    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    shape = prompts.output_shape_for_prompt(doc_type, acord_form)

    assert shape in rendered
    for key in resolved_schema(doc_type, acord_form).get("properties", {}):
        assert key in shape, f"{key} is in the schema but missing from the shown output shape"


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_the_output_shape_names_every_row_key_of_every_array(doc_type, acord_form):
    from common.schemas import resolved_schema

    shape = prompts.output_shape_for_prompt(doc_type, acord_form)
    for name, node in resolved_schema(doc_type, acord_form).get("properties", {}).items():
        for row_key in (node.get("items") or {}).get("properties", {}):
            assert row_key in shape, f"{name}[].{row_key} is missing from the shown output shape"


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_the_output_shape_is_an_outline_not_a_second_schema(doc_type, acord_form):
    """Shape only — no types, no descriptions, no JSON syntax. The schema
    follows immediately after and is the authority on both; a second copy here
    would double the token cost to say the same thing twice.

    Asserted structurally rather than by keyword, because JSON Schema keywords
    collide with real field names — `description` is a genuine key on a claim row.
    """
    shape = prompts.output_shape_for_prompt(doc_type, acord_form)
    schema = prompts.schema_json_for_prompt(doc_type, acord_form)

    assert '"' not in shape, "the shape is a bare key outline, not JSON"
    assert ":" not in shape.replace(": [", ""), "only array keys carry a colon"
    assert len(shape) < len(schema) / 3, "the outline has grown into a second schema"


# --------------------------------------------------------------------------
# Schema selection by line of business (arch v2.1 §0b)
# --------------------------------------------------------------------------


def test_a_policy_falls_back_to_the_generic_schema_until_an_lob_file_exists():
    """Adding a per-LOB schema later is a single _SCHEMA_FILES entry and no
    call-site change — which is the whole point of resolving the fallback here
    rather than at each caller."""
    from common.schemas import schema_key

    assert schema_key("policy") == "policy"
    assert schema_key("policy", None, "workers_comp") == "policy"
    assert schema_key("policy", None, ["workers_comp"]) == "policy"


def test_an_lob_selects_a_schema_only_when_one_is_registered():
    """Simulates the state after per-LOB schemas land, without shipping one."""
    from common import schemas

    registered = dict(schemas._SCHEMA_FILES)
    registered["policy:workers_comp"] = "policy_doc.schema.json"
    original, schemas._SCHEMA_FILES = schemas._SCHEMA_FILES, registered
    try:
        assert schemas.schema_key("policy", None, "workers_comp") == "policy:workers_comp"
        assert schemas.schema_key("policy", None, "commercial_auto") == "policy"
    finally:
        schemas._SCHEMA_FILES = original


def test_a_package_policy_uses_the_generic_schema_rather_than_one_of_its_lines():
    """A policy covering GL, Property and Auto is one document with a section per
    line. Picking one line's schema would validate the whole document against a
    third of itself."""
    from common import schemas

    registered = dict(schemas._SCHEMA_FILES)
    registered["policy:workers_comp"] = "policy_doc.schema.json"
    original, schemas._SCHEMA_FILES = schemas._SCHEMA_FILES, registered
    try:
        assert schemas.schema_key(
            "policy", None, ["workers_comp", "general_liability"]
        ) == "policy"
    finally:
        schemas._SCHEMA_FILES = original


def test_a_list_valued_lob_does_not_break_the_schema_cache():
    """line_of_business is a list, and the loaders are cached — caching on the
    arguments would raise 'unhashable type: list' on every multi-LOB call."""
    from common.schemas import load_schema, resolved_schema, validator_for

    assert load_schema("policy", None, ["workers_comp", "property"])["type"] == "object"
    assert validator_for("policy", None, ["workers_comp"]) is validator_for("policy")
    assert resolved_schema("policy", None, ["workers_comp"])["properties"]
