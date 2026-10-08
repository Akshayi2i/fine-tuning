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
    #: Coverages the label holds: the denominator of a code found at all.
    codes_gold: int = 0

    def add(self, expected: dict[str, Any], got: dict[str, Any], lob: Any) -> None:
        gold, wrote = _overflow(expected), _overflow(got)
        self.overflow_gold += sum(gold.values())
        self.overflow_written += sum(wrote.values())
        self.overflow_matched += sum((gold & wrote).values())

        found, gold_links = _links(expected, got, lob)
        self.links_gold += gold_links
        self.links_found += found

        right, scored, gold_codes = _codes(expected, got)
        self.codes_right += right
        self.codes_scored += scored
        self.codes_gold += gold_codes

    def metrics(self) -> dict[str, float | None]:
        def ratio(a: int, b: int) -> float | None:
            return round(a / b, 4) if b else None

        return {
            "additional_fields_recall": ratio(self.overflow_matched, self.overflow_gold),
            "additional_fields_precision": ratio(self.overflow_matched, self.overflow_written),
            "reference_accuracy": ratio(self.links_found, self.links_gold),
            "coverage_code_accuracy": ratio(self.codes_right, self.codes_scored),
            # Of the label's coverages, the share the answer holds with the right
            # code: a coverage it missed counts too, where the accuracy above
            # only judges the coverages it paired.
            "coverage_code_recall": ratio(self.codes_right, self.codes_gold),
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
    """Coverages paired by what the page prints - what they apply to and their
    name, then the name alone - never by the code being measured; of the
    pairs, how many carry the same code.

    The name alone pairs a coverage whose link was lost between windows (its
    units and its coverages on different pages): keyed on the link as well,
    no coverage of such a policy paired and the metric was never measured.
    Where several share a name, the one sharing the most values is taken
    (``field_accuracy.pair_coverages_by_name``).
    """
    from evaluation.metrics.field_accuracy import pair_coverages_by_name

    def coverages(doc: Any) -> list[dict[str, Any]]:
        rows = (doc or {}).get("coverages") if isinstance(doc, dict) else None
        return [row for row in rows or [] if isinstance(row, dict)]

    answer, label = coverages(got), coverages(expected)
    right = scored = 0
    for row, mate in zip(answer, pair_coverages_by_name(label, answer), strict=True):
        if mate is None:
            continue
        scored += 1
        right += str(values_view(mate.get("coverage_code")) or "") == str(
            values_view(row.get("coverage_code")) or "")
    return right, scored, len(label)


def _links(expected: dict[str, Any], got: dict[str, Any], lob: Any) -> tuple[int, int]:
    """``(links found, links the label holds)``, judged row by row.

    Rows are paired as field match pairs them (``aligned_for_scoring``); a label
    row's link is found when its answer row links the same unit. The same unit
    is decided by ``common.structural_ids.same_unit`` - the first unit key BOTH
    rows state - so a location the label names by its address and the answer by
    its number and address is one location. Compared as exact key strings, every
    link to it was lost, and with it every link of a row whose name differed.

    A link to a unit with no key at all (a building with no number) can only
    match the same text. A single-part policy's ``part`` links nothing.
    """
    from common.canonical import values_view
    from common.schema_sections import references
    from evaluation.metrics.field_accuracy import aligned_for_scoring

    refs = dict(references(lob))
    parts = (expected or {}).get("lob_parts") if isinstance(expected, dict) else None
    if not (isinstance(parts, list) and len(parts) > 1):
        refs.pop("part", None)
    if not refs:
        return 0, 0
    label, answer = values_view(expected or {}), values_view(got or {})
    units = (_units(label, lob), _units(answer, lob))
    docs = (label, answer)
    aligned_label, aligned_answer = aligned_for_scoring(label, answer)
    found = total = 0
    for table, rows in (aligned_label or {}).items():
        if not isinstance(rows, list):
            continue
        mates = (aligned_answer or {}).get(table) or []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            mate = mates[index] if index < len(mates) and isinstance(mates[index], dict) else {}
            for field_name in refs:
                wanted = _targets(row.get(field_name))
                offered = _targets(mate.get(field_name))
                total += len(wanted)
                for target in wanted:
                    match = next((t for t in offered if _same_target(target, t, units, lob, docs)), None)
                    if match is not None:
                        offered.remove(match)
                        found += 1
    return found, total


def _targets(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value] if value not in (None, "") else []
    return [str(v) for v in values if v not in (None, "")]


def _units(doc: dict[str, Any], lob: Any) -> dict[str, tuple[str, dict[str, Any]]]:
    """Each unit row by the name a link to it carries (``locations:location_number=2``)."""
    from common.schema_sections import structural_ids
    from common.structural_ids import _unit_key

    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for table in structural_ids(lob):
        for row in doc.get(table) or []:
            key = _unit_key(table, row, lob, {})
            if key is not None:
                out[f"{table}:" + ",".join(f"{f}={v}" for f, v in zip(key[1], key[2], strict=True))] = (table, row)
    return out


def _same_target(wanted: str, offered: str, units: tuple[dict, dict], lob: Any,
                 docs: tuple[dict, dict] = ({}, {})) -> bool:
    """Whether a label's link and an answer's name one unit - or one premises: a
    location, and the only building at it. The labels name such a premises by
    its location in 385 documents and by its building in 176, seed by seed, and
    the client's own example names the building; either names the same thing."""
    if wanted == offered:
        return True
    from common.structural_ids import same_unit

    left, right = units[0].get(wanted), units[1].get(offered)
    if left and right and left[0] == right[0]:
        return bool(same_unit(left[0], left[1], right[1], lob))
    a, b = _premises(wanted, units[0], docs[0], lob), _premises(offered, units[1], docs[1], lob)
    if a is None or b is None or a[0] == b[0]:
        return False
    if a[1] == b[1]:
        return True
    here, there = units[0].get(a[1]), units[1].get(b[1])
    return bool(here and there and same_unit("locations", here[1], there[1], lob))


def _premises(target: str, units: dict, doc: dict, lob: Any) -> tuple[str, str] | None:
    """``(kind, the location's name)`` for a link to a location, or to a building
    that is the only one at its location; else None. A building with no key
    keeps its id as its link, and is the one named when it is the document's only
    building."""
    from common.schema_sections import structural_ids

    unit = units.get(target)
    if unit and unit[0] == "locations":
        return "locations", target
    buildings = [row for row in doc.get("buildings") or [] if isinstance(row, dict)]
    prefix = (structural_ids(lob).get("buildings") or {}).get("prefix")
    if unit and unit[0] == "buildings":
        row = unit[1]
    elif not unit and prefix and target.startswith(f"{prefix}_") and len(buildings) == 1:
        row = buildings[0]
    else:
        return None
    home = row.get("location_ref")
    if not isinstance(home, str) or sum(b.get("location_ref") == home for b in buildings) != 1:
        return None
    return "buildings", home
