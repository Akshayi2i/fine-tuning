"""Labeling configuration, generated from the schema (SPEC_04 §4).

**Derived, never hand-written.** A hand-maintained reviewer form drifts from the
schema the moment a field is added, and the drift is silent in the worst
direction: reviewers keep filling a form that no longer matches what the corpus
builder validates against, so the labels fail validation in bulk long after the
review time was spent.

Everything a reviewer needs comes from one place:

* the field list and its types — from the resolved schema;
* the **semantic gloss** — the schema ``description``, the same text the model
  gets in its prompt, so reviewer and model are working from one definition;
* the **surface labels** to expect — from the alias registry, shown to the human
  and never to the model (see the package docstring);
* the **confusables** — the labels that must *not* fill this field, which is the
  boundary a reviewer gets wrong most often.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from xml.sax.saxutils import escape

from common import aliases as alias_registry
from common.schemas import resolved_schema

log = logging.getLogger(__name__)

#: Mandatory on every document type (arch §0b). A reviewer must set it to a
#: valid enum value or explicitly to null; it cannot be left unset, because an
#: unset LoB is indistinguishable from an undetermined one downstream.
MANDATORY_FIELDS = ("line_of_business",)


@dataclass
class ReviewField:
    """One field as the reviewer sees it."""

    path: str
    types: list[str]
    gloss: str
    required: bool = False
    enum: list[Any] | None = None
    aliases: tuple[str, ...] = ()
    confusables: tuple[str, ...] = ()
    row_fields: list[ReviewField] = field(default_factory=list)

    @property
    def is_list(self) -> bool:
        return "array" in self.types

    @property
    def nullable(self) -> bool:
        return "null" in self.types

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "types": self.types,
            "gloss": self.gloss,
            "required": self.required,
            "enum": self.enum,
            "expect_labels": list(self.aliases),
            "never_from": list(self.confusables),
            "rows": [f.as_dict() for f in self.row_fields],
        }


def _types(node: dict[str, Any]) -> list[str]:
    declared = node.get("type")
    if isinstance(declared, list):
        return list(declared)
    return [declared] if declared else []


def review_fields_for(doc_type: str, acord_form: str | None = None) -> list[ReviewField]:
    """Every field a reviewer fills for this document type, in schema order."""
    schema = resolved_schema(doc_type, acord_form)
    required = set(schema.get("required", []))
    registry = alias_registry.load_registry(doc_type)

    fields: list[ReviewField] = []
    for name, node in schema.get("properties", {}).items():
        entry = registry.get(name)
        gloss = node.get("description", "")
        if not gloss:
            # Loud, because a field with no gloss is a field three reviewers will
            # define three ways — and the model has no definition for it either.
            log.warning(
                "field %r in %s has no description, so the reviewer form shows no definition for "
                "it and the prompt carries none either (SPEC_01 requires one on every field).",
                name, doc_type,
            )

        row_fields: list[ReviewField] = []
        items = (node.get("items") or {}) if "array" in _types(node) else {}
        row_required = set(items.get("required", []))
        for row_name, row_node in items.get("properties", {}).items():
            row_entry = registry.get(f"{name}[].{row_name}")
            row_fields.append(ReviewField(
                path=f"{name}[].{row_name}",
                types=_types(row_node),
                gloss=row_node.get("description", ""),
                required=row_name in row_required,
                enum=row_node.get("enum"),
                aliases=tuple(row_entry.aliases) if row_entry else (),
                confusables=tuple(row_entry.confusables) if row_entry else (),
            ))

        # An array-of-enum (line_of_business under v2.1 §0b) declares its values
        # on `items.enum`, not `enum`. Read both, or the control silently
        # degrades to a free-text box and the reviewer can type anything.
        node_enum = node.get("enum") or (node.get("items") or {}).get("enum")

        fields.append(ReviewField(
            path=name,
            types=_types(node),
            gloss=gloss,
            required=name in required or name in MANDATORY_FIELDS,
            enum=node_enum,
            aliases=tuple(entry.aliases) if entry else (),
            confusables=tuple(entry.confusables) if entry else (),
            row_fields=row_fields,
        ))
    return fields


def _hint(f: ReviewField) -> str:
    """The one-line rulebook shown beside the input."""
    parts = [f.gloss] if f.gloss else []
    if f.aliases:
        parts.append(f"Commonly labelled: {', '.join(f.aliases)}.")
    if f.confusables:
        # The boundary reviewers get wrong most often, and the one the
        # confusable-misattribution metric scores the model on later.
        parts.append(f"NEVER take from: {', '.join(f.confusables)}.")
    if f.nullable:
        parts.append("Leave empty only if the document does not state it.")
    return " ".join(parts)


def build_labeling_config(doc_type: str, acord_form: str | None = None) -> str:
    """A Label Studio labeling config for one document type.

    The layout is the one the spec requires: page images and OCR text beside the
    draft, with each field's gloss attached to the input that fills it rather
    than buried in a separate rulebook nobody opens mid-review.
    """
    fields = review_fields_for(doc_type, acord_form)
    label = doc_type if not acord_form else f"{doc_type} {acord_form}"

    lines = [
        "<View>",
        f'  <Header value="{escape(label)} — correct every field against the document"/>',
        '  <View style="display: flex;">',
        '    <View style="flex: 55%; padding-right: 1em;">',
        '      <Image name="pages" valueList="$page_images" zoom="true" zoomControl="true"/>',
        '      <Text name="ocr" value="$ocr_text" granularity="word"/>',
        "    </View>",
        '    <View style="flex: 45%;">',
    ]

    for f in fields:
        hint = escape(_hint(f))
        # A list of ENUM values is a multi-select, not a row editor. Two
        # different list shapes live in these schemas — `claims` is a list of
        # objects and needs one row per claim, `line_of_business` is a list of
        # enum values and needs checkboxes — and treating them alike gave the
        # reviewer a free-text box where a controlled vocabulary belongs.
        if f.is_list and not f.enum:
            row_names = ", ".join(r.path.split("[].")[-1] for r in f.row_fields)
            lines.append(f'      <Header value="{escape(f.path)} (one entry per row)"/>')
            lines.append(f'      <Text name="{f.path}_hint" value="{hint} Row fields: {escape(row_names)}."/>')
            lines.append(f'      <TextArea name="{f.path}" toName="pages" rows="6" editable="true"/>')
            continue

        lines.append(f'      <Header value="{escape(f.path)}{" *" if f.required else ""}"/>')
        lines.append(f'      <Text name="{f.path}_hint" value="{hint}"/>')
        if f.enum:
            # A list-valued enum gets MULTIPLE choice. Forcing single choice on
            # line_of_business is what made the v1 annotator pick one line off a
            # package policy and discard the rest — the label shape and the
            # control have to agree, or the tool quietly caps what can be
            # recorded (arch v2.1 §0b).
            multiple = f.is_list
            choice = "multiple" if multiple else "single"
            # Required is dropped for a multi-select: selecting nothing IS the
            # answer for a document that determines no line, and a required
            # control makes that undetermined-but-correct label unrecordable.
            required = "false" if multiple else str(f.required).lower()
            lines.append(
                f'      <Choices name="{f.path}" toName="pages" choice="{choice}" '
                f'required="{required}">'
            )
            for value in f.enum:
                shown = "null (undetermined)" if value is None else str(value)
                lines.append(f'        <Choice value="{escape(shown)}"/>')
            lines.append("      </Choices>")
            if multiple:
                lines.append(
                    f'      <Text name="{f.path}_none" '
                    f'value="select none if the document determines no value"/>'
                )
        else:
            lines.append(
                f'      <TextArea name="{f.path}" toName="pages" rows="1" editable="true" '
                f'required="{str(f.required).lower()}"/>'
            )
        # The observed surface label. One extra field per value, and it is what
        # makes per-alias diagnosis and alias derivation possible later — without
        # it there is only an aggregate, and an aggregate cannot say which
        # phrasing the model is failing on.
        lines.append(
            f'      <TextArea name="{f.path}__seen_as" toName="pages" rows="1" editable="true" '
            f'placeholder="label printed on the document"/>'
        )

    lines += ["    </View>", "  </View>", "</View>"]
    return "\n".join(lines)
