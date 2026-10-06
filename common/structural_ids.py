"""Structural ids on a common-model (SPEC_21) document: what they name, and their numbering.

A common-model document links its rows by id: a coverage ``applies_to`` a
vehicle, a building has a ``location_ref``, an entry belongs to a ``part``. The
ids themselves are never printed - they are numbers the writer assigns in
printed order (``veh_1``, ``veh_2``) - so the same vehicle is ``veh_3`` in a gold
label, ``veh_1`` in the one window that shows it, and whatever the merge makes
of it in the served answer.

So ids are always renumbered, never compared:

* a training target is renumbered **per window**, so a window is taught the ids
  it can see, counted from 1 in its own rows, and never a reference to a row it
  is not shown;
* the served answer is renumbered **once more after the merge**, so its ids are
  unique and in printed order across the whole document;
* scoring compares references by the **units' own keys** (VIN, location number)
  instead, so the numbering a model chose is never right or wrong.

One function does the numbering for both sides (:func:`renumber_structural_ids`):
two implementations would teach one numbering and serve another. The tables,
id fields, prefixes, reference fields and unit keys are the section map's
(``configs/schema_sections.yaml``, ``common_model``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from common.canonical import values_view
from common.normalize import normalize_text, normalize_value


@dataclass
class IdReport:
    """What renumbering found."""

    #: References that named no row the document holds, as ``path -> id``. In a
    #: window target they are references to rows other windows read, left out on
    #: purpose; in a served answer each is a link the model could not make.
    dangling: list[str] = field(default_factory=list)
    #: Old id -> new id, per table.
    renumbered: dict[str, dict[str, str]] = field(default_factory=dict)


def renumber_structural_ids(
    doc: dict[str, Any],
    lob: str | list[str] | None,
    *,
    extra_index: dict[str, str] | None = None,
    assign: tuple[str, ...] = (),
    in_place: bool = False,
) -> tuple[dict[str, Any], IdReport]:
    """``doc`` with every table's ids numbered from 1 in row order, and every
    reference rewritten to them. A copy; ``doc`` is not changed.

    Row order is printed order: a target keeps the label's order and the merge
    keeps page order. A table's id field is written only where the row has one
    or the field is in ``assign`` (``coverage_id``, which the pipeline assigns).

    ``extra_index`` maps an id that is not a row's own - a window's id for a row
    the merge joined into another - to the id of the row it now is; it is
    consulted before a reference is called dangling.

    A reference that names no row is dropped (and reported): a list keeps the
    ids that resolve and is removed when none do, a single reference is removed.

    ``in_place`` changes ``doc`` itself: the serving merge keeps each value's
    token spans by object identity, which a copy would lose.
    """
    from common.schema_sections import references, structural_ids

    ids = structural_ids(lob)
    out = doc if in_place else copy.deepcopy(doc)
    report = IdReport()
    by_table: dict[str, dict[str, str]] = {}
    for table, spec in ids.items():
        rows = out.get(table)
        if not isinstance(rows, list):
            continue
        mapping: dict[str, str] = {}
        number = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            id_field, prefix = spec["field"], spec["prefix"]
            if id_field not in row and id_field not in assign:
                continue
            number += 1
            new = f"{prefix}_{number}"
            old = row.get(id_field)
            if isinstance(old, str) and old:
                mapping.setdefault(old, new)
            row[id_field] = new
        if mapping:
            by_table[table] = mapping
    report.renumbered = by_table

    refs = references(lob)
    extra = dict(extra_index or {})

    def resolve(value: Any, tables: tuple[str, ...]) -> str | None:
        if not isinstance(value, str):
            return None
        mapped = extra.get(value)
        old = value if mapped is None else mapped
        for table in tables:
            if old in by_table.get(table, {}):
                return by_table[table][old]
        # An id extra_index resolved to one already in the new numbering - and
        # only such an id. A reference that names none of the document's rows is
        # dangling even when it reads like a new id: a window holding only the
        # label's second vehicle numbers it veh_1, and the label's veh_1 is
        # still the first vehicle, which that window does not hold.
        if mapped is not None:
            for table in tables:
                if mapped in by_table.get(table, {}).values():
                    return mapped
        return None

    def walk(node: Any, path: str) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            return
        for name in [n for n in node if n in refs]:
            value = node[name]
            where = f"{path}.{name}" if path else name
            if isinstance(value, list):
                kept = []
                for item in value:
                    new = resolve(item, refs[name])
                    if new is None:
                        report.dangling.append(f"{where}={item}")
                    elif new not in kept:
                        kept.append(new)
                if kept:
                    node[name] = kept
                else:
                    del node[name]
            else:
                new = resolve(value, refs[name])
                if new is None:
                    report.dangling.append(f"{where}={value}")
                    del node[name]
                else:
                    node[name] = new
        for key, value in node.items():
            if key not in refs:
                walk(value, f"{path}.{key}" if path else key)

    walk(out, "")
    return out, report


def unit_key(table: str, row: Any, lob: str | list[str] | None) -> tuple | None:
    """What identifies a risk unit across windows: the first of the table's
    unit keys the row states, normalised (``configs/schema_sections.yaml``).

    ``None`` for a row stating none of them - a fragment the merge cannot place
    by itself.
    """
    return _unit_key(table, row, lob, {})


def _unit_key(table: str, row: Any, lob: str | list[str] | None, names: dict[str, str]) -> tuple | None:
    """:func:`unit_key`, with a part that is a reference written as the name of
    the row it refers to, where ``names`` holds one (:func:`_unit_names`). A
    name is already normalised, and is taken as it is. With no names this is
    :func:`unit_key` exactly - the merge compares ids it has rewritten itself."""
    from common.schema_sections import references, unit_keys

    if not isinstance(row, dict):
        return None
    refs = references(lob) if names else {}
    for key in unit_keys(lob).get(table, ()):
        fields = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        parts = tuple(
            names[row[name]] if name in refs and isinstance(row.get(name), str) and row[name] in names
            else _normalised(row.get(name), f"{table}[].{name}")
            for name in fields)
        if all(part not in (None, "") for part in parts):
            return (table, fields, parts)
    return None


def same_unit(table: str, a: Any, b: Any, lob: str | list[str] | None) -> bool | None:
    """Whether two rows of a unit table are one unit: decided by the first of
    the table's unit keys that BOTH rows state (a row with only its VIN and a
    fragment with only its vehicle number compare on neither, and are not
    known to be one; a row with both meets either). ``None`` when no key is
    stated by both."""
    from common.schema_sections import unit_keys

    if not (isinstance(a, dict) and isinstance(b, dict)):
        return None
    for key in unit_keys(lob).get(table, ()):
        names = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        left = tuple(_normalised(a.get(name), f"{table}[].{name}") for name in names)
        right = tuple(_normalised(b.get(name), f"{table}[].{name}") for name in names)
        if all(v not in (None, "") for v in left + right):
            return left == right
    return None


def resolve_references(doc: Any, lob: str | list[str] | None) -> Any:
    """``doc`` with every reference replaced by the units' own keys.

    ``applies_to: ["veh_2"]`` becomes ``["vehicles:vin=1hgcm82633a004352"]``:
    the same in a gold label and in an answer however each numbered its rows.
    A unit key that itself holds a reference is written with the name of the
    row it refers to: a building is ``buildings:location_ref=locations:
    location_number=2,building_number=1``, not its writer's ``loc_2``, so a
    link to a building does not change with how either side numbered its
    locations. A reference to no row, or to a row with no unit key, stays as it
    was. A copy; ``doc`` is not changed.
    """
    from common.schema_sections import references

    refs = references(lob)
    names = _unit_names(doc, lob) if isinstance(doc, dict) else {}

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for key, value in node.items():
            if key in refs:
                if isinstance(value, list):
                    out[key] = sorted(names.get(v, v) if isinstance(v, str) else v for v in value)
                else:
                    out[key] = names.get(value, value) if isinstance(value, str) else value
            else:
                out[key] = walk(value)
        return out

    return walk(doc)


def _unit_names(doc: dict[str, Any], lob: str | list[str] | None) -> dict[str, str]:
    """Each unit row's id -> its name: its table and the first unit key it
    states (``vehicles:vin=1hgcm82633a004352``).

    A key part that is a reference is named by the row it refers to, so rows
    are named again until no name changes: a building takes its location's
    name whatever order the tables come in, and a longer chain of such keys
    resolves the same way. The rounds are bounded by the number of tables, so
    ids that refer to each other in a circle cannot loop.
    """
    from common.schema_sections import structural_ids

    ids = structural_ids(lob)
    names: dict[str, str] = {}
    for _ in range(len(ids) + 1):
        before = dict(names)
        for table, spec in ids.items():
            for row in doc.get(table) or []:
                if not isinstance(row, dict) or not isinstance(row.get(spec["field"]), str):
                    continue
                key = _unit_key(table, row, lob, names)
                if key is not None:
                    _, fields, values = key
                    names[row[spec["field"]]] = (
                        f"{table}:" + ",".join(f"{f}={v}" for f, v in zip(fields, values, strict=True)))
        if names == before:
            break
    return names


def _normalised(value: Any, path: str) -> Any:
    value = values_view(value)
    if value in (None, "", []):
        return None
    if isinstance(value, dict):
        leaves = sorted(normalize_text(v) or "" for v in value.values() if v not in (None, ""))
        return "|".join(leaves) or None
    normalised = normalize_value(value, field_path=path)
    if normalised is None:
        normalised = normalize_text(value)
    return normalised


def comparable_view(doc: Any, lob: str | list[str] | None) -> Any:
    """``doc`` as two readings of one policy can be compared: references written
    as the units' own keys, and every structural id left out.

    The numbering is the writer's choice - a gold label's veh_3 is an answer's
    veh_1 - so an id is never right or wrong; what a reference NAMES is. A
    copy; ``doc`` is not changed.
    """
    from common.schema_sections import structural_ids

    id_fields = {spec["field"] for spec in structural_ids(lob).values()}
    if not id_fields:
        return doc
    resolved = resolve_references(doc, lob)
    parts = resolved.get("lob_parts") if isinstance(resolved, dict) else None
    single_part = not (isinstance(parts, list) and len(parts) > 1)

    def strip(node: Any) -> Any:
        if isinstance(node, list):
            return [strip(item) for item in node]
        if not isinstance(node, dict):
            return node
        return {k: strip(v) for k, v in node.items()
                if k not in id_fields and not (single_part and k == "part")}

    return strip(resolved)


#: Fields holding a row's printed name: what identifies a row that has one.
_PRINTED_NAMES = ("coverage_name", "name")


def reference_pairs(doc: Any, lob: str | list[str] | None) -> list[tuple[Any, ...]]:
    """Every link in ``doc`` as ``(table, row identity, field, unit key)``.

    The row is identified by its table's keys other than its references, the
    unit by its own keys - so a link counts as the same in two documents
    whatever either numbered. A coverage is known by its printed name, and by
    its code only when no name is printed: a code is chosen from a list and
    scored on its own (``coverage_code_accuracy``), and one wrong code must not
    also lose the links the row got right.
    """
    from common.schema_sections import array_key, references

    refs = references(lob)
    if not refs or not isinstance(doc, dict):
        return []
    link_fields = set(refs)
    parts = doc.get("lob_parts")
    if not (isinstance(parts, list) and len(parts) > 1):
        # One part: `part` is omitted by its own definition, so it links nothing.
        refs = {k: v for k, v in refs.items() if k != "part"}
    resolved = resolve_references(doc, lob)
    pairs: list[tuple[Any, ...]] = []
    for table, rows in resolved.items():
        if not isinstance(rows, list):
            continue
        keys = [k for k in (*array_key(table, lob), *_PRINTED_NAMES) if k not in link_fields]
        for row in rows:
            if not isinstance(row, dict):
                continue
            named = any(_normalised(row.get(k), f"{table}[].{k}") is not None for k in _PRINTED_NAMES)
            # A bare code (coverage_code) is left out - as None, so every row
            # of the table keeps one shape - when the row prints a name.
            identity = tuple(
                None if named and k.endswith("_code") and not isinstance(row.get(k), dict)
                else _normalised(row.get(k), f"{table}[].{k}")
                for k in dict.fromkeys(keys))
            for field_name in refs:
                value = row.get(field_name)
                for target in value if isinstance(value, list) else [value] if value else []:
                    pairs.append((table, identity, field_name, target))
    return pairs


def comparable_for(doc: Any, doc_type: str, acord_form: str | None, lob: Any) -> Any:
    """:func:`comparable_view` for a common-model policy; ``doc`` itself otherwise."""
    from common.schemas import SchemaError, is_common_model

    try:
        common_model = doc_type == "policy" and is_common_model(doc_type, acord_form, lob)
    except SchemaError:
        common_model = False
    return comparable_view(doc, lob) if common_model else doc
