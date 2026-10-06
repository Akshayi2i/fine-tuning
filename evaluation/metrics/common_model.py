"""Scoring a common-model (SPEC_21) policy: what is new about its shape.

Three things the self-contained scoring does not see, and one it must not see:

* **Values inside rows.** A limit lives inside its coverage, a building's year
  inside its row. Field match skipped every path inside a table, which was
  harmless while limits were named slots and is blind now; here the rows are
  paired by identity and their values scored with the rest.
* **Links.** Which vehicle a coverage applies to - compared by the unit's own
  keys, never its id (``common.structural_ids.reference_pairs``).
* **Codes.** Whether a coverage got the code that says what it covers.
* **Overflow.** ``additional_fields`` scored on its own, keyed by label and
  value, so the core fields' score is not diluted by the long tail.

And never scored as a value: a structural id. The numbering is the writer's
choice; the documents reaching here have been through ``comparable_view``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from common.canonical import values_view

ADDITIONAL_FIELDS = "additional_fields"


@dataclass
class CommonModelTally:
    """The counts behind this module's metrics, summed over documents."""

    overflow_gold: int = 0
    overflow_written: int = 0
    overflow_matched: int = 0
    links_gold: int = 0
    links_found: int = 0
    codes_scored: int = 0
    codes_right: int = 0

    def add(self, expected: dict[str, Any], got: dict[str, Any], lob: Any) -> None:
        gold, wrote = _overflow(expected), _overflow(got)
        self.overflow_gold += sum(gold.values())
        self.overflow_written += sum(wrote.values())
        self.overflow_matched += sum((gold & wrote).values())

        from common.structural_ids import reference_pairs

        links, found = Counter(reference_pairs(expected, lob)), Counter(reference_pairs(got, lob))
        self.links_gold += sum(links.values())
        self.links_found += sum((links & found).values())

        right, scored = _codes(expected, got)
        self.codes_right += right
        self.codes_scored += scored

    def metrics(self) -> dict[str, float | None]:
        def ratio(a: int, b: int) -> float | None:
            return round(a / b, 4) if b else None

        return {
            "additional_fields_recall": ratio(self.overflow_matched, self.overflow_gold),
            "additional_fields_precision": ratio(self.overflow_matched, self.overflow_written),
            "reference_accuracy": ratio(self.links_found, self.links_gold),
            "coverage_code_accuracy": ratio(self.codes_right, self.codes_scored),
        }


def core_for_scoring(expected: dict[str, Any], got: dict[str, Any], lob: Any) -> tuple[Any, Any]:
    """``(expected, got)`` as field match scores them: values, overflow left
    out (scored on its own), each table's rows paired by identity at one index -
    a row the answer left out after the answer's rows, so its values count as
    misses - and links left out (scored by :class:`CommonModelTally`)."""
    from common.schema_sections import references
    from evaluation.metrics.field_accuracy import aligned_for_scoring

    def core(doc: Any) -> Any:
        return values_view({k: v for k, v in (doc or {}).items() if k != ADDITIONAL_FIELDS})

    aligned_expected, aligned_got = aligned_for_scoring(core(expected), core(got))
    links = set(references(lob))
    return _without(aligned_expected, links), _without(aligned_got, links)


def without_overflow(doc: Any) -> Any:
    return {k: v for k, v in doc.items() if k != ADDITIONAL_FIELDS} if isinstance(doc, dict) else doc


def _without(node: Any, names: set[str]) -> Any:
    if isinstance(node, dict):
        return {k: _without(v, names) for k, v in node.items() if k not in names}
    if isinstance(node, list):
        return [_without(v, names) for v in node]
    return node


def _overflow(doc: Any) -> Counter:
    from common.normalize import normalize_text

    entries = (doc or {}).get(ADDITIONAL_FIELDS) if isinstance(doc, dict) else None
    out: Counter = Counter()
    for entry in entries or []:
        if isinstance(entry, dict):
            value = values_view(entry.get("value"))
            out[(normalize_text(entry.get("label")), normalize_text(value))] += 1
    return out


def _codes(expected: dict[str, Any], got: dict[str, Any]) -> tuple[int, int]:
    """Coverages paired by what they apply to and their printed name; of the
    pairs, how many carry the same code."""
    from common.normalize import normalize_text

    def by_identity(doc: Any) -> dict[tuple, list[str]]:
        out: dict[tuple, list[str]] = {}
        for row in (doc or {}).get("coverages") or [] if isinstance(doc, dict) else []:
            if not isinstance(row, dict):
                continue
            name = normalize_text(values_view(row.get("coverage_name")))
            if not name:
                continue
            key = (tuple(sorted(row.get("applies_to") or [])), name)
            out.setdefault(key, []).append(str(row.get("coverage_code") or ""))
        return out

    gold, answer = by_identity(expected), by_identity(got)
    right = scored = 0
    for key, codes in gold.items():
        for code, other in zip(codes, answer.get(key, []), strict=False):
            scored += 1
            right += code == other
    return right, scored
