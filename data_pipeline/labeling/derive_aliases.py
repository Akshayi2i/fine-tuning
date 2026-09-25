"""Derive the alias registry from labeled documents (SPEC_04 §3, arch §0c).

The registry is **built from evidence, not hand-written**. Given documents you
have already labeled canonically, the surface label for each field is recoverable
by alignment: the golden JSON supplies the **value**, the OCR supplies the
**text**, and the label is whatever introduces that value on the page.

Three outputs from one pass:

* the **alias registry** — observed surface forms per canonical field;
* **``field_provenance`` backfilled** onto documents labeled before provenance
  capture existed, so per-alias evaluation works without re-annotating anything;
* a **QA list** of golden values that appear nowhere in their document, which
  means either the label is wrong or OCR failed that page — neither visible any
  other way.

Everything it produces is a **proposal with evidence**, never a silent overwrite.
A human confirms once.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from common.canonical import printed_view
from common.normalize import normalize_entity_name, normalize_text, normalize_value

log = logging.getLogger(__name__)

Pattern = Literal[
    "inline_kv", "bold_adjacent", "table_row_label",
    "table_column_header", "stacked_form", "dotted_leader",
]

#: Layout patterns in priority order. An inline ``Label: value`` is unambiguous;
#: a stacked form layout is a weaker signal because adjacency is doing the work.
PATTERN_PRIORITY: dict[Pattern, int] = {
    "inline_kv": 100,
    "table_column_header": 95,   # in a data table this is the label, not the cell to the left
    "bold_adjacent": 90,
    "table_row_label": 85,       # only correct for 2-column "| Label | value |" tables
    "stacked_form": 60,
    "dotted_leader": 55,
}

#: Values shorter than this match everywhere and produce noise, not evidence.
MIN_VALUE_CHARS = 4

#: How far back to look for a label introducing a value.
LOOKBACK_CHARS = 120

_LABEL_CLEAN = re.compile(r"^[\s*#>|\-]+|[\s*:|.]+$")
_DOTTED = re.compile(r"\.{3,}")
_EMPHASIS = re.compile(r"\*+")

#: A candidate line that already contains its own "Label: value" pair is not a
#: label for the NEXT line's value — it is a different field entirely. Without
#: this, stacked-form matching walks up and grabs the neighbouring field's text.
_LOOKS_LIKE_A_FILLED_FIELD = re.compile(r"[:：]\s*\S")


@dataclass
class AliasEvidence:
    """One observed ``(canonical field, surface label)`` pairing."""

    field: str
    surface_label: str
    source_id: str
    pattern: Pattern
    value: str

    @property
    def confidence(self) -> float:
        return PATTERN_PRIORITY[self.pattern] / 100.0


@dataclass
class DerivationReport:
    """What one derivation pass found, with the evidence behind it."""

    aliases: dict[str, dict[str, list[AliasEvidence]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    confusables: dict[str, dict[str, set[str]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    unresolved: list[dict[str, str]] = field(default_factory=list)
    skipped_low_entropy: list[dict[str, str]] = field(default_factory=list)
    ambiguous: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_registry(self, *, min_documents: int = 1) -> dict[str, Any]:
        """Render as an alias-registry document, with evidence attached."""
        out: dict[str, Any] = {
            "$comment": (
                "DERIVED from labeled documents by data_pipeline.labeling.derive_aliases. "
                "Every entry carries evidence: document count, matched layout pattern, and "
                "example source_ids. Review before promoting to the live registry. "
                "NEVER rendered into a prompt and NEVER consulted at inference (arch 0c)."
            )
        }
        for canonical_field in sorted(set(self.aliases) | set(self.confusables)):
            entries = []
            for label, evidence in sorted(
                self.aliases.get(canonical_field, {}).items(),
                key=lambda kv: (-len({e.source_id for e in kv[1]}), kv[0]),
            ):
                documents = sorted({e.source_id for e in evidence})
                if len(documents) < min_documents:
                    continue
                entries.append({
                    "label": label,
                    "documents": len(documents),
                    "confidence": round(max(e.confidence for e in evidence), 2),
                    "pattern": max(evidence, key=lambda e: e.confidence).pattern,
                    "examples": documents[:3],
                })
            confusable_entries = [
                {"label": label, "documents": len(sources), "basis": "co_occurring_type_compatible"}
                for label, sources in sorted(
                    self.confusables.get(canonical_field, {}).items(),
                    key=lambda kv: (-len(kv[1]), kv[0]),
                )
            ]
            if entries or confusable_entries:
                out[canonical_field] = {"aliases": entries, "confusables": confusable_entries}

        out["$unresolved"] = self.unresolved
        out["$skipped_low_entropy"] = self.skipped_low_entropy
        out["$ambiguous"] = self.ambiguous
        return out

    def summary(self) -> str:
        alias_count = sum(len(v) for v in self.aliases.values())
        return (
            f"{len(self.aliases)} fields, {alias_count} distinct surface labels, "
            f"{len(self.unresolved)} unresolved, {len(self.ambiguous)} ambiguous, "
            f"{len(self.skipped_low_entropy)} skipped as low-entropy"
        )


# --------------------------------------------------------------------------
# Label extraction
# --------------------------------------------------------------------------

def clean_label(raw: str) -> str:
    """Normalise a candidate label so casing and punctuation converge."""
    text = _DOTTED.sub(" ", raw)
    text = _EMPHASIS.sub("", text)          # markdown emphasis is not part of the label
    text = _LABEL_CLEAN.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :*|-").strip()


def _find_value_positions(text: str, value: Any, field_path: str) -> list[int]:
    """Character offsets where a golden value appears in the OCR text.

    Matches literally first, then through ``common.normalize`` — the golden JSON
    is canonical (``2026-04-01``, ``12400.0``) while the page is not
    (``04/01/2026``, ``$12,400.00``), so roughly every date and currency field
    needs the normalized pass to anchor at all.
    """
    if value is None or isinstance(value, bool):
        return []
    literal = str(value)
    positions = [m.start() for m in re.finditer(re.escape(literal), text)]
    if positions:
        return positions

    normalized_target = normalize_value(value, field_path)
    if normalized_target is None:
        return []

    # Scan candidate substrings, bounded by line - labels and values sit together
    # on a line in these documents. Both table cells AND whitespace-delimited
    # tokens are tried: a date in a table is its own cell, but the same date in
    # "**Valued As Of:** 03/31/2026" is a bare token on a longer line, and
    # missing that case would leave nearly every date and currency field
    # unresolved.
    found: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        seen_here: set[int] = set()
        # Each cell is searched at its OWN offset within the line. `line.find`
        # returned the first occurrence anywhere, so a value repeated across
        # cells of one row — `| 5,000.00 | 5,000.00 |` under a Paid / Incurred
        # header — always resolved to the leftmost column, and the
        # duplicate-position guard then discarded the correct one entirely. The
        # derived alias became the neighbouring field's label: exactly the field
        # conflation this module exists to prevent.
        cell_start = 0
        for cell in re.split(r"[|\t]", line):
            for candidate in _candidate_substrings(cell):
                if normalize_value(candidate, field_path) != normalized_target:
                    continue
                within = cell.find(candidate)
                if within < 0:
                    continue
                index = cell_start + within
                if index not in seen_here:
                    seen_here.add(index)
                    found.append(offset + index)
            cell_start += len(cell) + 1        # +1 for the delimiter split consumed
        offset += len(line)
    return found


def _candidate_substrings(cell: str) -> list[str]:
    """Substrings of a cell worth testing against a normalized golden value.

    The whole cell first (a table cell IS the value), then whitespace-delimited
    tokens and adjacent pairs - enough to catch ``03/31/2026`` inside a longer
    line and ``$47,250.00`` after a label, without degenerating into a scan of
    every possible slice.
    """
    stripped = cell.strip()
    if not stripped:
        return []
    out = [stripped] if len(stripped) >= 2 else []
    tokens = stripped.split()
    for i, token in enumerate(tokens):
        cleaned = token.strip(".,;:*|")
        if len(cleaned) >= 2:
            out.append(cleaned)
        if i + 1 < len(tokens):                      # e.g. "$47,250.00 USD"
            pair = f"{token} {tokens[i + 1]}".strip(".,;:*|")
            if len(pair) >= 2:
                out.append(pair)
    return out


def _extract_label(text: str, value_start: int) -> tuple[str, Pattern] | None:
    """Read backward from a value for the label introducing it."""
    line_start = text.rfind("\n", 0, value_start) + 1
    before_on_line = text[line_start:value_start]
    window_start = max(0, value_start - LOOKBACK_CHARS)

    # Table row: | Label | value |. Only valid for a two-column table - in a
    # wider data row the cell to the left is another field's value.
    if before_on_line.count("|") >= 1 and text[line_start:line_start + 1] == "|":
        line_end = text.find("\n", value_start)
        full_line = text[line_start:line_end if line_end != -1 else len(text)]
        columns = [c for c in full_line.strip().strip("|").split("|")]
        if len(columns) == 2:
            cells = [c.strip() for c in before_on_line.split("|") if c.strip()]
            if cells:
                label = clean_label(cells[-1])
                if label:
                    return label, "table_row_label"

    # Inline key-value: **Label:** value   /   Label: value
    inline = re.search(r"([^|\n]{2,60}?)\s*[:：]\s*$", before_on_line)
    if inline:
        label = clean_label(inline.group(1))
        if label:
            return label, "inline_kv"

    # Dotted leader: Label ......... value
    if _DOTTED.search(before_on_line):
        label = clean_label(_DOTTED.split(before_on_line)[0])
        if label:
            return label, "dotted_leader"

    # Bold label, value adjacent on the same line: **Label** value
    bold = re.search(r"\*\*([^*\n]{2,60})\*\*\s*$", before_on_line)
    if bold:
        label = clean_label(bold.group(1))
        if label:
            return label, "bold_adjacent"

    # Stacked form: a bare label on its own line, value on the next. The
    # preceding line must not already carry its own value, or we would attribute
    # the neighbouring field's label to this one.
    if not before_on_line.strip():
        preceding = text[window_start:line_start].rstrip("\n")
        previous_line = preceding.rsplit("\n", 1)[-1] if preceding else ""
        if previous_line.strip() and not _LOOKS_LIKE_A_FILLED_FIELD.search(previous_line):
            label = clean_label(previous_line)
            if label and len(label) <= 60:
                return label, "stacked_form"

    return None


def _table_column_header(text: str, value_start: int) -> tuple[str, Pattern] | None:
    """The header cell above a value's column in a markdown table."""
    line_start = text.rfind("\n", 0, value_start) + 1
    line_end = text.find("\n", value_start)
    line = text[line_start:line_end if line_end != -1 else len(text)]
    if not line.strip().startswith("|"):
        return None

    column = line[: value_start - line_start].count("|") - 1
    if column < 0:
        return None

    # Walk up to the header row (the one above the |---| separator).
    lines = text[:line_start].splitlines()
    for index in range(len(lines) - 1, -1, -1):
        candidate = lines[index].strip()
        if not candidate.startswith("|"):
            break
        cells = [c.strip() for c in candidate.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells if c):        # separator
            if index > 0:
                header_cells = [c.strip() for c in lines[index - 1].strip().strip("|").split("|")]
                if 0 <= column < len(header_cells):
                    label = clean_label(header_cells[column])
                    if label:
                        return label, "table_column_header"
            break
    return None


def _iter_leaf_fields(label: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten a golden label to ``(field_path, value)`` leaves.

    A list of OBJECTS yields indexed leaves — ``claims[0].claim_number`` — because
    each row's surface label is its own column header.

    A list of SCALARS yields one leaf per value at the **unindexed** path. They
    share one canonical field and one surface label, so indexing them would split
    a single field's alias evidence across ``line_of_business[0]``,
    ``[1]`` … and align none of it. Previously these were dropped entirely, so a
    set-valued field was invisible to alias derivation and to the unresolved
    report the moment it stopped being a scalar (arch v2.1 §0b).
    """
    out: list[tuple[str, Any]] = []
    for key, value in label.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.extend(_iter_leaf_fields(value, f"{path}."))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    out.extend(_iter_leaf_fields(item, f"{path}[{index}]."))
                elif item is not None:
                    out.append((path, item))
        else:
            out.append((path, value))
    return out


def canonical_field_of(field_path: str) -> str:
    """The canonical field an alias belongs to, with row indices collapsed.

    ``claims[0].claim_number`` -> ``claims[].claim_number``. Collapsing to the
    array name instead would make every cell in a Loss Run table an "alias" for
    ``claims``, which is noise: the surface label of a table cell is its COLUMN
    HEADER, and the header belongs to the leaf field, not to the array.
    """
    return re.sub(r"\[\d+\]", "[]", field_path)


def _value_shape(value: Any) -> str:
    """Coarse type, for judging whether two values are confusable.

    Two organisation names on one page are confusable; an org name and a date are
    not, however close their labels look.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    text = str(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return "date"
    if re.fullmatch(r"[A-Z0-9][A-Z0-9\-/ ]{4,}", text):
        return "identifier"
    return "name"


# --------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------

def derive_from_document(
    golden: dict[str, Any],
    ocr_text: str,
    source_id: str,
    report: DerivationReport,
) -> None:
    """Align one document's golden label against its OCR text."""
    provenance: dict[str, str] = {}
    located: dict[str, tuple[str, Any]] = {}   # field -> (surface_label, value)

    # A canonical label nests each value in an envelope; the value that can be
    # found in the OCR text is the one printed on the page (`raw`), not its
    # normalised form. Collapsed first, the envelope's own keys never become
    # "fields" with aliases of their own. A flat label passes through unchanged.
    for field_path, value in _iter_leaf_fields(printed_view(golden)):
        if value is None:
            continue
        canonical = canonical_field_of(field_path)

        if len(str(value)) < MIN_VALUE_CHARS:
            report.skipped_low_entropy.append({
                "source_id": source_id, "field": field_path, "value": str(value),
                "reason": f"value shorter than {MIN_VALUE_CHARS} chars matches everywhere",
            })
            continue

        positions = _find_value_positions(ocr_text, value, field_path)
        if not positions:
            report.unresolved.append({
                "source_id": source_id, "field": field_path, "value": str(value)[:60],
                "reason": "value_not_found_in_ocr",
            })
            continue

        candidates: list[tuple[str, Pattern]] = []
        for position in positions:
            # A column header, where one exists, always beats the preceding cell:
            # in a multi-column data row the cell to the left is a DIFFERENT
            # field's value, not this field's label.
            header = _table_column_header(ocr_text, position)
            if header:
                candidates.append(header)
                continue
            if found := _extract_label(ocr_text, position):
                candidates.append(found)

        if not candidates:
            report.unresolved.append({
                "source_id": source_id, "field": field_path, "value": str(value)[:60],
                "reason": "value_found_but_no_label_precedes_it",
            })
            continue

        best_label, best_pattern = max(candidates, key=lambda c: PATTERN_PRIORITY[c[1]])
        rivals = {
            normalize_text(label)
            for label, pattern in candidates
            if PATTERN_PRIORITY[pattern] == PATTERN_PRIORITY[best_pattern]
        }
        if len(rivals) > 1:
            report.ambiguous.append({
                "source_id": source_id, "field": field_path,
                "candidates": sorted({c[0] for c in candidates}),
                "reason": "value appears under more than one equally strong label",
            })
            continue

        report.aliases[canonical][best_label].append(
            AliasEvidence(canonical, best_label, source_id, best_pattern, str(value)[:60])
        )
        provenance.setdefault(canonical, best_label)
        located[canonical] = (best_label, value)

    # Confusables: two rules, not one.
    #   1. sibling canonical fields whose values are type-compatible — the case a
    #      real policy schema hits most, since certificate_holder and producer are
    #      themselves canonical fields;
    #   2. (handled by callers) unmapped labels of the same shape.
    for field_a, (_label_a, value_a) in located.items():
        for field_b, (label_b, value_b) in located.items():
            if field_a == field_b:
                continue
            if (
                _value_shape(value_a) == _value_shape(value_b) == "name"
                and normalize_entity_name(value_a) != normalize_entity_name(value_b)
            ):
                report.confusables[field_a][label_b].add(source_id)

    if provenance:
        report.provenance[source_id] = provenance


def derive_aliases(
    documents: list[tuple[str, dict[str, Any], str]],
) -> DerivationReport:
    """Derive a registry from ``(source_id, golden_label, ocr_text)`` triples."""
    report = DerivationReport()
    for source_id, golden, ocr_text in documents:
        try:
            derive_from_document(golden, ocr_text, source_id, report)
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop a corpus
            log.warning("alias derivation failed for %s: %s", source_id, exc)
            report.unresolved.append({
                "source_id": source_id, "field": "*", "value": "",
                "reason": f"derivation_error: {exc}",
            })
    log.info("alias derivation: %s", report.summary())
    return report


def write_proposal(report: DerivationReport, path: str, *, min_documents: int = 1) -> str:
    """Write ``*.aliases.proposed.json``. Never touches the live registry."""
    registry = report.to_registry(min_documents=min_documents)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path
