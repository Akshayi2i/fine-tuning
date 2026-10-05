"""Loss Run claims table F1 (Fideon SPEC_09 handoff item 8).

A claim is found when an extracted claim has the expected claim's number and
its total incurred within a cent. A wrong total on the right claim is a miss
and an extra: the row is there, and the figure a reader would sum is wrong.
Micro-averaged over every claim of every Loss Run, so a long report weighs by
its claims rather than counting once.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from common.normalize import normalize_currency, normalize_identifier

#: Within a cent: the same figure, rounded the same way.
TOTAL_TOLERANCE = 0.01


@dataclass
class TableScore:
    matched: int = 0
    extracted: int = 0
    expected: int = 0

    @property
    def precision(self) -> float | None:
        return self.matched / self.extracted if self.extracted else None

    @property
    def recall(self) -> float | None:
        return self.matched / self.expected if self.expected else None

    @property
    def f1(self) -> float | None:
        total = self.extracted + self.expected
        return 2 * self.matched / total if total else None


def _claims(document: Any) -> list[dict[str, Any]]:
    rows = document.get("claims") if isinstance(document, dict) else None
    return [r for r in rows or [] if isinstance(r, dict)
            and str(r.get("row_type") or "claim").casefold() not in ("subtotal", "total")]


def _same_total(a: Any, b: Any) -> bool:
    x, y = normalize_currency(a), normalize_currency(b)
    if x is None or y is None:
        return x is None and y is None
    return abs(float(x) - float(y)) <= TOTAL_TOLERANCE + 1e-9


def score_claims(expected: dict[str, Any], got: dict[str, Any]) -> TableScore:
    """One Loss Run's claims against its gold."""
    wanted = _claims(expected)
    found = _claims(got)
    open_by_number: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in wanted:
        number = normalize_identifier(claim.get("claim_number"))
        if number:
            open_by_number[number].append(claim)
    matched = 0
    for claim in found:
        candidates = open_by_number.get(normalize_identifier(claim.get("claim_number")) or "", [])
        hit = next((c for c in candidates if _same_total(c.get("total_incurred"),
                                                         claim.get("total_incurred"))), None)
        if hit is not None:
            candidates.remove(hit)
            matched += 1
    return TableScore(matched=matched, extracted=len(found), expected=len(wanted))


def table_f1(pairs: Iterable[tuple[dict[str, Any], dict[str, Any]]]) -> TableScore:
    """Micro-averaged over the ``(expected, got)`` pairs of Loss Runs."""
    total = TableScore()
    for expected, got in pairs:
        score = score_claims(expected, got or {})
        total.matched += score.matched
        total.extracted += score.extracted
        total.expected += score.expected
    return total


def lossrun_pairs(scored: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]):
    return [(expected, got) for expected, got, metadata in scored if metadata.get("doc_type") == "lossrun"]
