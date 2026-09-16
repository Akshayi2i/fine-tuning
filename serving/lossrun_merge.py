"""Deterministic row merge across Loss Run windows (arch v2.1 §7b).

``lossrun_rows`` runs on overlapping page windows, so the same claim is extracted
more than once by design — the overlap exists so a row split across a page break
is seen whole by at least one window. Merging is therefore not an optimisation;
it is the step that turns several partial answers into one document.

**Deterministic, not model-mediated.** Asking the model to reconcile its own
windows would make the result depend on generation order and put a second
sampling step between the extraction and the answer. The rules below are
mechanical and auditable, which is what a claims list needs to be.

Dedupe keys, in order of trust:

1. ``claim_number`` after normalisation. A carrier's own identifier for the
   claim, and the only key that is reliably unique.
2. ``(date of loss, claimant, total incurred)`` when the claim number is absent
   — ordinary on older Loss Runs, and on reports that redact it.

Rows that survive neither key are kept rather than dropped. A duplicate inflates
a total, which reconciliation catches; a dropped row understates a loss history,
which nothing catches.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from common.normalize import normalize_currency, normalize_identifier, normalize_text

log = logging.getLogger(__name__)

#: Fields the fallback key is built from, when there is no claim number.
FALLBACK_KEY_FIELDS = ("loss_date", "claimant", "total_incurred")


@dataclass
class MergeReport:
    """The merged claims list, and what merging did."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    total_rows: list[dict[str, Any]] = field(default_factory=list)

    duplicates_collapsed: int = 0
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    unkeyed_rows: int = 0
    continued_rows_joined: int = 0

    @property
    def flagged(self) -> bool:
        """Whether merging alone is reason for a human to look.

        A conflict means two windows read the same claim differently, and the
        merge picked one. That choice is mechanical and recorded, but it is still
        a choice about a value nobody verified.
        """
        return bool(self.conflicts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": len(self.rows),
            "duplicates_collapsed": self.duplicates_collapsed,
            "unkeyed_rows": self.unkeyed_rows,
            "continued_rows_joined": self.continued_rows_joined,
            "conflicts": self.conflicts[:25],
            "flagged": self.flagged,
        }


def _claim_key(row: dict[str, Any]) -> tuple[str, ...] | None:
    """The row's identity, or ``None`` when it has no usable key."""
    number = normalize_identifier(row.get("claim_number"))
    if number:
        return ("claim", number)

    parts = [
        normalize_text(row.get("loss_date")) or "",
        normalize_text(row.get("claimant")) or "",
        str(normalize_currency(row.get("total_incurred")) or ""),
    ]
    return ("fallback", *parts) if any(parts) else None


def _completeness(row: dict[str, Any]) -> int:
    """How many fields the row actually filled.

    Used to pick a winner between two readings of one claim: the more complete
    reading is preferred, because a window that saw the whole row has strictly
    more evidence than one that saw it cut off at a page break.
    """
    return sum(1 for value in row.values() if value not in (None, "", [], {}))


def merge_windows(
    window_outputs: Sequence[Sequence[dict[str, Any]]],
    *,
    totals_output: dict[str, Any] | None = None,
) -> MergeReport:
    """Merge overlapping window outputs into one claims list.

    Args:
        window_outputs: each window's rows, in window order. Rows carry
            ``row_type`` (``claim`` | ``subtotal`` | ``total``) from §7b.
        totals_output: the ``lossrun_totals`` task's output, if it ran.
    """
    report = MergeReport()
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}

    for window_index, rows in enumerate(window_outputs):
        for row in rows:
            if not isinstance(row, dict):
                continue

            row_type = str(row.get("row_type") or "claim").casefold()
            if row_type in ("subtotal", "total"):
                # Totals are deduped on their own identity — a subtotal repeated
                # across an overlap is one subtotal, and counting it twice would
                # double the figure reconciliation checks against.
                if not any(_same_total(row, seen) for seen in report.total_rows):
                    report.total_rows.append(dict(row))
                continue

            key = _claim_key(row)
            if key is None:
                # No key at all. Kept: a duplicate inflates a total, which
                # reconciliation catches; a dropped row understates a loss
                # history, which nothing catches.
                report.unkeyed_rows += 1
                report.rows.append(dict(row))
                continue

            existing = by_key.get(key)
            if existing is None:
                by_key[key] = dict(row)
                continue

            report.duplicates_collapsed += 1
            merged, conflicts = _reconcile_row(existing, row)
            by_key[key] = merged
            for field_name, (kept, discarded) in conflicts.items():
                report.conflicts.append({
                    "claim_number": row.get("claim_number"),
                    "field": field_name,
                    "kept": kept,
                    "discarded": discarded,
                    "window": window_index,
                    "reason": "two windows read the same claim differently",
                })

    report.rows.extend(by_key.values())
    # Stable order so two runs over the same document produce the same list —
    # a claims list whose order depends on dict iteration is not comparable
    # against its own previous extraction.
    report.rows.sort(key=lambda r: (
        str(normalize_identifier(r.get("claim_number")) or ""),
        str(normalize_text(r.get("loss_date")) or ""),
    ))

    if totals_output:
        report.total_rows.append({"row_type": "total", **totals_output})

    if report.duplicates_collapsed:
        log.info(
            "merged %d duplicate row(s) across windows into %d claim(s); %d conflict(s)",
            report.duplicates_collapsed, len(report.rows), len(report.conflicts),
        )
    return report


def _same_total(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        str(a.get("row_type", "")).casefold() == str(b.get("row_type", "")).casefold()
        and normalize_text(a.get("policy_period")) == normalize_text(b.get("policy_period"))
        and normalize_currency(a.get("total_incurred")) == normalize_currency(b.get("total_incurred"))
    )


def _reconcile_row(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    """Combine two readings of one claim, recording where they disagreed.

    A field present in one and absent in the other is filled in — that is the
    overlap doing its job, joining a row that a page break cut in half. A field
    present in both with different values is a **conflict**: the more complete
    row's value wins, and the discarded one is recorded rather than lost.
    """
    merged = dict(existing)
    conflicts: dict[str, tuple[Any, Any]] = {}
    prefer_incoming = _completeness(incoming) > _completeness(existing)

    for name, value in incoming.items():
        current = merged.get(name)
        if current in (None, "", [], {}):
            merged[name] = value
            continue
        if value in (None, "", [], {}) or current == value:
            continue

        winner, loser = (value, current) if prefer_incoming else (current, value)
        merged[name] = winner
        conflicts[name] = (winner, loser)
    return merged, conflicts


def merge_and_reconcile(
    window_outputs: Sequence[Sequence[dict[str, Any]]],
    *,
    totals_output: dict[str, Any] | None = None,
) -> tuple[MergeReport, Any]:
    """Merge the windows, then reconcile the result against printed totals.

    The two steps belong together on the serving path: reconciliation is what
    turns a merged list into a list somebody can trust, and a merge whose result
    is never reconciled has no completeness signal at all — a missed row produces
    no tokens, so per-field confidence is structurally blind to it (§5.5).
    """
    from calibration.reconciliation import reconcile

    merged = merge_windows(window_outputs, totals_output=totals_output)
    report = reconcile(
        merged.rows,
        [r for r in merged.total_rows if str(r.get("row_type", "")).casefold() == "subtotal"],
        totals_output,
    )
    return merged, report
