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
2. ``(date of loss, claimant, description, total incurred)`` when the claim
   number is absent — ordinary on older Loss Runs, and on reports that redact
   it. The Loss Run schema carries no claimant, so the description is what tells
   two incident-only claims of one day apart.

Only rows of DIFFERENT windows are joined: the overlap is what repeats a claim,
so two rows with one key in one window are two claims (a claim number printed
on several rows, two $0 incidents on one date). Rows keep document order -
the schema's "in document order" - which is as reproducible as any sort.

Rows that survive neither key are kept rather than dropped. A duplicate inflates
a total, which reconciliation catches; a dropped row understates a loss history,
which nothing catches.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from common.normalize import normalize_currency, normalize_identifier, normalize_text

log = logging.getLogger(__name__)

#: Fields the fallback key is built from, when there is no claim number.
FALLBACK_KEY_FIELDS = ("loss_date", "claimant", "description", "total_incurred")


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
        normalize_text(row.get("description")) or "",
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
    # Every claim in document order; per key, the claims holding it and the
    # windows each has already absorbed a row from.
    merged_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, ...], list[tuple[int, set[int]]]] = {}

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
                merged_rows.append(dict(row))
                continue

            # The first claim with this key not yet read in THIS window: one
            # window's two rows with a key are two claims, not one read twice.
            slot = next(((i, seen) for i, seen in by_key.get(key, []) if window_index not in seen), None)
            if slot is None:
                merged_rows.append(dict(row))
                by_key.setdefault(key, []).append((len(merged_rows) - 1, {window_index}))
                continue

            index, seen = slot
            seen.add(window_index)
            report.duplicates_collapsed += 1
            merged, conflicts = _reconcile_row(merged_rows[index], row)
            merged_rows[index] = merged
            for field_name, (kept, discarded) in conflicts.items():
                report.conflicts.append({
                    "claim_number": row.get("claim_number"),
                    "field": field_name,
                    "kept": kept,
                    "discarded": discarded,
                    "window": window_index,
                    "reason": "two windows read the same claim differently",
                })

    # Document order: first seen, window by window, row by row. Reproducible -
    # the same windows give the same list - and what the schema asks for.
    report.rows = merged_rows

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
        if _cut_reading(current, value):
            # One reading is the start of the other: the window that saw the row
            # cut at a page break read a prefix. The longer one is the row; not
            # a disagreement.
            merged[name] = max(current, value, key=lambda v: len(str(v).strip()))
            continue

        winner, loser = (value, current) if prefer_incoming else (current, value)
        merged[name] = winner
        conflicts[name] = (winner, loser)
    return merged, conflicts


def _cut_reading(a: Any, b: Any) -> bool:
    """Whether one text is the other cut short (a row split at a page break)."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    short, long = sorted((a.strip(), b.strip()), key=len)
    return bool(short) and short != long and long.startswith(short)


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


_EMPTY = (None, "", [], {})


def merge_extracted_windows(
    windows: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], MergeReport, Any]:
    """Whole-schema extractions of a Loss Run's page windows, as one document.

    ``windows`` are ``(extraction, spans)`` in page order, spans keyed by
    values-view path (``claims[3].paid``). Returns the merged extraction, its
    spans re-keyed to it, the merge report and the reconciliation report.

    A header field keeps the first window's value that read one: the header is
    printed at the top, and a later window without it reads null. The claims of
    every window are merged by :func:`merge_and_reconcile`. Each merged value
    keeps the span of the window row it came from, so its confidence describes
    the tokens that produced it.
    """
    extraction: dict[str, Any] = {}
    spans: dict[str, Any] = {}
    for window, window_spans in windows:
        for key, value in window.items():
            if key == "claims":
                continue
            if extraction.get(key) in _EMPTY and value not in _EMPTY:
                extraction[key] = value
                spans.update({path: span for path, span in window_spans.items()
                              if path == key or path.startswith((f"{key}.", f"{key}["))})
            else:
                extraction.setdefault(key, value)

    claims = [list(window.get("claims") or []) for window, _ in windows]
    merged, reconciliation = merge_and_reconcile(claims)
    extraction["claims"] = merged.rows
    # Each window's rows by key, built once: looking them up per merged value
    # re-keyed every row of every window for every field.
    rows_by_key: list[dict[Any, list[tuple[int, dict[str, Any]]]]] = []
    for rows in claims:
        index: dict[Any, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for position, candidate in enumerate(rows):
            if isinstance(candidate, dict):
                index[_claim_key(candidate)].append((position, candidate))
        rows_by_key.append(index)
    for row_index, row in enumerate(merged.rows):
        key = _claim_key(row)
        for name, value in row.items():
            span = _source_span(windows, rows_by_key, key, row, name, value)
            if span is not None:
                spans[f"claims[{row_index}].{name}"] = span
    return extraction, spans, merged, reconciliation


def _source_span(windows, rows_by_key, key, row, name, value):
    """The span of the first window row that is this claim and holds this value."""
    for (_window, window_spans), index in zip(windows, rows_by_key, strict=True):
        for position, candidate in index.get(key, ()):
            if candidate.get(name) != value or (key is None and candidate != row):
                continue
            return window_spans.get(f"claims[{position}].{name}")
    return None


def reconcile_extraction(extraction: dict[str, Any]) -> Any:
    """The reconciliation report of a Loss Run read in one call."""
    from calibration.reconciliation import reconcile

    rows = [r for r in extraction.get("claims") or [] if isinstance(r, dict)]
    claim_rows = [r for r in rows if str(r.get("row_type") or "claim").casefold() not in ("subtotal", "total")]
    subtotals = [r for r in rows if str(r.get("row_type") or "").casefold() == "subtotal"]
    return reconcile(claim_rows, subtotals, None)
