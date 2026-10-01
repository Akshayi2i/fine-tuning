"""Merge a policy's section windows into one canonical document (arch v2.1 §7b).

A long policy is read as a cross product of section groups and page windows
(``configs/schema_sections.yaml``): ``decl`` over the declarations pages,
``arrays`` and ``lineblk`` over every routed page in several windows, ``dtd`` where
the line carries it. Each window returns its slice of the model form. This module
turns those partial answers into one document — and carries each value's token
spans with it, so confidence and ``flagged`` land on the value they describe.

**Deterministic, not model-mediated.** The same reason as
:mod:`serving.lossrun_merge`: asking the model to reconcile its own windows puts a
second sampling step between the extraction and the answer.

Rules, in order:

1. **Windows are merged in group order, then page order.** ``decl`` leads, so the
   declarations value wins any conflict by construction — the same rule
   ``policy.jinja`` states to the model.
2. **A value already present is kept.** A later window's differing value is a
   recorded conflict, never a silent overwrite.
3. **Arrays concatenate**, then rows are de-duplicated. A table crossing a window
   boundary legitimately comes back twice, so this is load-bearing, not tidying.
   Top-level arrays use the identifying key the section map declares
   (``array_key``), compared on normalised values; a row with no key value at all
   is kept rather than guessed into another. Nested arrays with no declared key
   collapse only rows whose values are identical.
4. **Duplicates keep the more complete row**, take any field only the other row
   has, and union the page references of fields both carry.

Spans travel by object identity rather than by path. A row's path changes when
rows are concatenated and de-duplicated; the envelope object does not.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from common.canonical import is_field_value, values_view
from common.normalize import normalize_text, normalize_value

log = logging.getLogger(__name__)


@dataclass
class PolicyWindow:
    """One window's answer: which group, over which pages, and what came back.

    ``extraction`` is the model form (sparse ``raw``/``parsed``/``page_ref``
    envelopes), dates already normalised. ``spans`` is keyed by values-view path,
    as :func:`common.canonical.collapse_spans` produces it.
    """

    group: str
    pages: list[int]
    extraction: dict[str, Any]
    spans: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None


@dataclass
class MergedPolicy:
    """The merged model-form document, its spans, and what merging did."""

    extraction: dict[str, Any] = field(default_factory=dict)
    spans: dict[str, Any] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    duplicates_collapsed: int = 0
    unkeyed_rows: int = 0

    @property
    def review_flags(self) -> list[str]:
        """A conflict is two windows reading one field differently. The merge
        kept one mechanically; a person should see which."""
        return [f"{path}:merge_conflict" for path in sorted({c.split(":", 1)[0] for c in self.conflicts})]


# --------------------------------------------------------------------------
# Spans by identity
# --------------------------------------------------------------------------


def _index_spans(node: Any, path: str, spans: Mapping[str, Any], out: dict[int, Any]) -> None:
    """Map each envelope object to the span the window recorded at its path."""
    if is_field_value(node):
        span = spans.get(path)
        if span is not None:
            out[id(node)] = span
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _index_spans(value, f"{path}.{key}" if path else key, spans, out)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _index_spans(item, f"{path}[{index}]", spans, out)


def _collect_spans(node: Any, path: str, by_id: Mapping[int, Any], out: dict[str, Any]) -> None:
    """Re-key spans to the merged document's values-view paths."""
    if is_field_value(node):
        span = by_id.get(id(node))
        if span is not None:
            out[path] = span
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _collect_spans(value, f"{path}.{key}" if path else key, by_id, out)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _collect_spans(item, f"{path}[{index}]", by_id, out)


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------


def _filled(node: Any) -> int:
    """How many stated values a node holds — "the more complete row"."""
    if is_field_value(node):
        return int(node.get("raw") is not None or node.get("parsed") is not None)
    if isinstance(node, dict):
        return sum(_filled(v) for v in node.values())
    if isinstance(node, list):
        return sum(_filled(v) for v in node)
    return int(node is not None)


def _union_pages(kept: dict[str, Any], other: dict[str, Any]) -> None:
    kept["page_ref"] = sorted({*kept.get("page_ref", []), *other.get("page_ref", [])})


def _same_value(a: dict[str, Any], b: dict[str, Any], path: str) -> bool:
    va, vb = values_view(a), values_view(b)
    if va == vb:
        return True
    return normalize_value(va, field_path=path) == normalize_value(vb, field_path=path)


def _merge_into(kept: Any, other: Any, path: str, report: MergedPolicy, *, top: bool) -> Any:
    """Merge ``other`` into ``kept``, keeping ``kept`` wherever both say something."""
    # An explicit null is an absence, not a value: the other window may state it.
    if kept is None:
        return other
    if other is None:
        return kept
    if is_field_value(kept) and is_field_value(other):
        if not _same_value(kept, other, path):
            report.conflicts.append(
                f"{path}: kept {values_view(kept)!r}, a later window read {values_view(other)!r}"
            )
        else:
            _union_pages(kept, other)
        return kept
    if isinstance(kept, dict) and isinstance(other, dict):
        for key, value in other.items():
            child = f"{path}.{key}" if path else key
            if key in kept:
                kept[key] = _merge_into(kept[key], value, child, report, top=False)
            else:
                kept[key] = value
        return kept
    if isinstance(kept, list) and isinstance(other, list):
        section = path if top else None
        return _dedupe(kept + other, path, section, report)
    # A shape disagreement — one window wrote an object, another a value. The
    # earlier window's answer stands; the other is recorded, never dropped silently.
    report.conflicts.append(f"{path}: windows disagree on the shape of this field")
    return kept


def _row_key(row: Any, key_fields: tuple[str, ...], path: str) -> tuple | None:
    """A row's identity on normalised values, or ``None`` when it has none."""
    if not isinstance(row, dict) or not key_fields:
        return None
    parts = []
    for name in key_fields:
        value = values_view(row.get(name))
        if isinstance(value, dict):
            leaves = sorted(
                normalize_text(v) or "" for v in _leaf_values(value) if v is not None
            )
            parts.append(tuple(leaves) or None)
        else:
            normalised = normalize_value(value, field_path=f"{path}[].{name}")
            # A value the typed normaliser cannot read is still a value. A form
            # edition such as `05/11` is not a parseable date, and dropping it
            # to None would make `HO 00 03 05/11` and `HO 00 03 10/00` — two
            # different forms — one row.
            if normalised is None and value is not None:
                normalised = normalize_text(value)
            parts.append(normalised)
    return tuple(parts) if any(p for p in parts) else None


def _leaf_values(node: Any) -> list[Any]:
    if isinstance(node, dict):
        return [v for child in node.values() for v in _leaf_values(child)]
    if isinstance(node, list):
        return [v for child in node for v in _leaf_values(child)]
    return [node]


def _dedupe(rows: list[Any], path: str, section: str | None, report: MergedPolicy) -> list[Any]:
    from common.schema_sections import array_key

    # A list of envelopes is a set of values, not a table: one entry per value.
    if rows and all(is_field_value(r) for r in rows):
        seen: dict[Any, dict[str, Any]] = {}
        out = []
        for row in rows:
            marker = normalize_value(values_view(row), field_path=path)
            if marker in seen:
                _union_pages(seen[marker], row)
                report.duplicates_collapsed += 1
                continue
            seen[marker] = row
            out.append(row)
        return out

    key_fields = array_key(section) if section else ()
    # No declared key (the tables inside a line block: vehicles, units,
    # coverages): the identifier scoring matches rows on - VIN, unit number,
    # coverage name. Matching only identical rows left a vehicle read across two
    # windows as two half-rows. Rows sharing an identifier join only when no
    # field they both state disagrees; otherwise they are different rows.
    inferred = () if key_fields else _identifiers(rows)
    out: list[Any] = []
    # Every row an identity names, not only the first: two real rows can share
    # an inferred identifier (two "Liability" coverages with different limits),
    # and a later window's re-read of the SECOND must join it, not be appended
    # again as a third.
    positions: dict[Any, list[int]] = {}
    for row in rows:
        if key_fields:
            identity = _row_key(row, key_fields, path)
        elif inferred:
            identity = _row_key(row, inferred, path)
        else:
            # Nothing identifies these rows: only an identical row is the same row.
            identity = ("exact", repr(values_view(row)))
        if identity is None:
            out.append(row)
            continue
        candidates = positions.setdefault(identity, [])
        position = next(
            (i for i in candidates if not (inferred and _disagree(out[i], row, path))), None
        )
        if position is None:
            candidates.append(len(out))
            out.append(row)
            continue
        kept, other = out[position], row
        if _filled(other) > _filled(kept):
            kept, other = other, kept
        out[position] = _merge_into(kept, other, f"{path}[{position}]", report, top=False)
        report.duplicates_collapsed += 1
    return out


#: Identifiers strong enough to join two windows' rows on. Scoring may match on
#: weaker ones (``description``, ``state``, ``label``, ``rank``) because a wrong
#: match there costs one comparison; a wrong JOIN merges two real rows into one
#: and loses the second, so the merge only trusts a value that names one row.
JOIN_IDENTIFIERS: frozenset[str] = frozenset({
    "claim_number", "policy_number", "form_number", "vin", "vin_or_hull_id",
    "hull_identification_number", "serial_number", "loan_number", "license_number",
    "docket_number", "vehicle_number", "driver_number", "unit_number", "motor_number",
    "item_number", "installment_number", "location_number", "building_number",
    "structure_number", "residence_number", "object_number", "agreement_number",
    "coverage_code", "coverage_name", "endorsement_name", "discount_name", "name",
    "individual_name", "entity_name",
})


def _identifiers(rows: list[Any]) -> tuple[str, ...]:
    """The strong identifier these rows carry, as scoring infers it; ``()``
    when none is filled in most rows - those rows join only when identical."""
    from evaluation.metrics.field_accuracy import _infer_key_fields

    if not rows or not all(isinstance(r, dict) for r in rows):
        return ()
    keys = _infer_key_fields(values_view(rows))
    return tuple(k for k in keys if k in JOIN_IDENTIFIERS)


def _disagree(a: dict[str, Any], b: dict[str, Any], path: str) -> bool:
    """Whether two rows state a different value for any field both carry."""
    for key, value in a.items():
        other = b.get(key)
        if is_field_value(value) and is_field_value(other):
            stated = (values_view(value), values_view(other))
            if None not in stated and not _same_value(value, other, f"{path}[].{key}"):
                return True
    return False


def merge_policy_windows(windows: list[PolicyWindow]) -> MergedPolicy:
    """One canonical model-form document from a policy's windows.

    ``windows`` must arrive in group order, then page order —
    :func:`common.schema_sections.group_names` order, which puts ``decl`` first.
    """
    from common.schema_sections import group_names

    rank = {name: index for index, name in enumerate(group_names())}
    ordered = sorted(
        windows, key=lambda w: (rank.get(w.group, len(rank)), min(w.pages or [0]))
    )

    by_id: dict[int, Any] = {}
    report = MergedPolicy()
    for window in ordered:
        _index_spans(window.extraction, "", window.spans, by_id)
        for key, value in window.extraction.items():
            if key in report.extraction:
                report.extraction[key] = _merge_into(
                    report.extraction[key], value, key, report, top=True
                )
            else:
                report.extraction[key] = (
                    _dedupe(value, key, key, report) if isinstance(value, list) else value
                )

    _collect_spans(report.extraction, "", by_id, report.spans)
    # Counted once, on the result: arrays are re-deduplicated as each window
    # merges in, so counting inside _dedupe would count one row several times.
    from common.schema_sections import array_key

    report.unkeyed_rows = sum(
        1
        for section, rows in report.extraction.items()
        if isinstance(rows, list) and array_key(section)
        for row in rows
        if _row_key(row, array_key(section), section) is None
    )
    if report.conflicts:
        log.info("merged %d policy windows with %d conflict(s)", len(windows), len(report.conflicts))
    return report
