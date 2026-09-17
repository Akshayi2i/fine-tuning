"""Loss Run totals reconciliation, per policy period (arch v2.1 §5.5).

**A missed row has no tokens, so it has no low logprob.** Per-field confidence is
structurally blind to omission: the model produced eleven claims confidently, and
nothing about those eleven says a twelfth existed. Reconciliation against the
document's own printed totals is the only signal that sees it.

**Two corrections v2.1 makes to v2.0.** Printed totals sit at the *end* of the
report and *per policy period* — not on pages 1–2 where v2.0 looked for them. And
a Loss Run routinely spans several policy periods, so a single grand-total check
passes whenever two periods' errors happen to cancel. Reconciling per period
catches that; reconciling only the grand total is how a missing claim in 2024
hides behind an invented one in 2025.

Totals are collected from **two** sources, because neither alone is reliable:
total and subtotal rows captured inside row windows (tagged ``row_type``), and
the dedicated ``lossrun_totals`` task on the last pages. A document that prints
no totals at all is **unverifiable, not verified** — its claims list routes to
review rather than passing silently.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from common.normalize import normalize_currency

log = logging.getLogger(__name__)

#: Absolute tolerance when comparing a sum against a printed total. Printed
#: totals round, and a rounding difference is not the omission this check exists
#: to catch. Half a cent per contributing row would be tighter but needs the row
#: count, and this is already inside the noise of any real discrepancy.
MONEY_TOLERANCE = 0.51

#: The money columns reconciled. A Loss Run that balances on incurred but not on
#: paid has a real problem, so each is checked separately rather than summed.
#: Spelled exactly as the Loss Run claim schema spells them: a column the rows
#: never carry is skipped silently, which is how ``reserve`` (the schema says
#: ``reserved``) left reserves unreconciled.
RECONCILED_COLUMNS = ("total_incurred", "paid", "reserved")


@dataclass
class PeriodReconciliation:
    """One policy period's sums against its printed totals."""

    period: str
    extracted_rows: int = 0
    extracted: dict[str, float] = field(default_factory=dict)
    printed: dict[str, float] = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)

    @property
    def verifiable(self) -> bool:
        """Whether the document printed anything to check against."""
        return bool(self.printed)

    @property
    def reconciled(self) -> bool:
        return self.verifiable and not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "extracted_rows": self.extracted_rows,
            "status": (
                "reconciled" if self.reconciled
                else "unverifiable" if not self.verifiable
                else "mismatch"
            ),
            "extracted": {k: round(v, 2) for k, v in sorted(self.extracted.items())},
            "printed": {k: round(v, 2) for k, v in sorted(self.printed.items())},
            "mismatches": self.mismatches,
        }


@dataclass
class ReconciliationReport:
    """A whole Loss Run's completeness evidence."""

    by_period: list[PeriodReconciliation] = field(default_factory=list)
    grand_total_printed: dict[str, float] = field(default_factory=dict)
    grand_total_mismatches: list[str] = field(default_factory=list)

    @property
    def verifiable(self) -> bool:
        return bool(self.grand_total_printed) or any(p.verifiable for p in self.by_period)

    @property
    def reconciled(self) -> bool:
        """Every verifiable period balances, and so does the grand total."""
        return (
            self.verifiable
            and not self.grand_total_mismatches
            and all(p.reconciled for p in self.by_period if p.verifiable)
        )

    @property
    def flagged(self) -> bool:
        """Whether this document's claims list needs a human.

        Unverifiable counts as flagged. A document that printed no totals has not
        demonstrated completeness — it has merely not been caught.
        """
        return not self.reconciled

    @property
    def status(self) -> str:
        if self.reconciled:
            return "reconciled"
        return "unverifiable" if not self.verifiable else "mismatch"

    def reasons(self) -> list[str]:
        if self.reconciled:
            return []
        if not self.verifiable:
            return [
                "the document prints no totals, so row completeness cannot be verified. "
                "A missed claim produces no tokens and no low confidence — this check is "
                "the only thing that would have seen it (arch v2.1 §5.5)."
            ]
        out = list(self.grand_total_mismatches)
        for period in self.by_period:
            out.extend(f"{period.period}: {m}" for m in period.mismatches)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "flagged": self.flagged,
            "by_policy_period": [p.as_dict() for p in self.by_period],
            "grand_total_printed": {
                k: round(v, 2) for k, v in sorted(self.grand_total_printed.items())
            },
            "grand_total_mismatches": self.grand_total_mismatches,
            "reasons": self.reasons(),
        }


def _period_of(row: dict[str, Any]) -> str:
    return str(row.get("policy_period") or row.get("period") or "unspecified")


def _sum_column(rows: Sequence[dict[str, Any]], column: str) -> float | None:
    values = [
        normalize_currency(row.get(column))
        for row in rows if isinstance(row, dict) and row.get(column) is not None
    ]
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def reconcile(
    claim_rows: Sequence[dict[str, Any]],
    total_rows: Sequence[dict[str, Any]] = (),
    grand_totals: dict[str, Any] | None = None,
) -> ReconciliationReport:
    """Reconcile extracted claims against printed totals, per period then overall.

    Args:
        claim_rows: rows tagged ``row_type: claim`` — the extraction itself.
        total_rows: rows tagged ``subtotal`` or ``total``, captured inside row
            windows. Each carries the policy period it belongs to.
        grand_totals: the ``lossrun_totals`` task's output from the last pages.
    """
    report = ReconciliationReport()

    claims_by_period: dict[str, list[dict[str, Any]]] = {}
    for row in claim_rows:
        if isinstance(row, dict):
            claims_by_period.setdefault(_period_of(row), []).append(row)

    printed_by_period: dict[str, dict[str, float]] = {}
    for row in total_rows:
        if not isinstance(row, dict):
            continue
        # A row tagged `total` rather than `subtotal` is the document's own grand
        # total appearing inside a window. Folded into the grand-total check, not
        # treated as a period — otherwise it would be compared against one
        # period's claims and always mismatch.
        if str(row.get("row_type", "")).casefold() == "total":
            for column in RECONCILED_COLUMNS:
                amount = normalize_currency(row.get(column))
                if amount is not None:
                    report.grand_total_printed.setdefault(column, amount)
            continue
        bucket = printed_by_period.setdefault(_period_of(row), {})
        for column in RECONCILED_COLUMNS:
            amount = normalize_currency(row.get(column))
            if amount is not None:
                bucket[column] = amount

    for column, amount in (grand_totals or {}).items():
        value = normalize_currency(amount)
        if value is not None and column in RECONCILED_COLUMNS:
            report.grand_total_printed[column] = value

    # A single-period Loss Run commonly prints the period once in the header and
    # not on every row, so the claims land under "unspecified" while the subtotal
    # names the period — and every such document would flag. When there is
    # exactly ONE printed period and no claim names a period, they are that
    # period's claims.
    #
    # Deliberately not done for several periods: unattributed claims across two
    # periods cannot be assigned without guessing, and a guess here would produce
    # a reconciliation that means nothing.
    if (
        len(printed_by_period) == 1
        and set(claims_by_period) == {"unspecified"}
    ):
        only_period = next(iter(printed_by_period))
        claims_by_period[only_period] = claims_by_period.pop("unspecified")
        log.debug(
            "attributed %d unattributed claim(s) to the document's single policy period %r",
            len(claims_by_period[only_period]), only_period,
        )
    elif len(printed_by_period) > 1 and "unspecified" in claims_by_period:
        log.warning(
            "%d claim row(s) name no policy period while the document prints %d of them. "
            "They cannot be attributed without guessing, so their period reconciles as its "
            "own bucket and will not balance.",
            len(claims_by_period["unspecified"]), len(printed_by_period),
        )

    for period in sorted(set(claims_by_period) | set(printed_by_period)):
        rows = claims_by_period.get(period, [])
        printed = printed_by_period.get(period, {})
        entry = PeriodReconciliation(
            period=period, extracted_rows=len(rows), printed=dict(printed)
        )
        for column, stated in sorted(printed.items()):
            extracted = _sum_column(rows, column)
            if extracted is None:
                entry.mismatches.append(
                    f"a {column} subtotal of {stated:,.2f} is printed but no extracted claim "
                    "row carries that column at all"
                )
                continue
            entry.extracted[column] = extracted
            if abs(extracted - stated) > MONEY_TOLERANCE:
                entry.mismatches.append(
                    f"{column}: extracted rows sum to {extracted:,.2f} against a printed "
                    f"{stated:,.2f} (off by {extracted - stated:+,.2f})"
                )
        report.by_period.append(entry)

    for column, stated in sorted(report.grand_total_printed.items()):
        extracted = _sum_column(list(claim_rows), column)
        if extracted is None:
            report.grand_total_mismatches.append(
                f"a grand total of {stated:,.2f} is printed for {column} but no extracted "
                "claim row carries that column"
            )
            continue
        if abs(extracted - stated) > MONEY_TOLERANCE:
            report.grand_total_mismatches.append(
                f"{column}: all extracted rows sum to {extracted:,.2f} against a printed grand "
                f"total of {stated:,.2f} (off by {extracted - stated:+,.2f})"
            )

    if report.flagged:
        log.warning(
            "loss run claims list flagged (%s): %s", report.status, "; ".join(report.reasons())
        )
    return report


def reconciliation_rate(reports: Sequence[ReconciliationReport]) -> float:
    """Share of VERIFIABLE documents that reconciled — the gating metric.

    Unverifiable documents are excluded from the denominator rather than counted
    as failures: a document that prints no totals says nothing about whether the
    model missed a row, and letting those drag the metric down would make it a
    measure of the corpus rather than of the model.

    The §0d floor is stated the other way round — "missed-row documents flagged
    by totals reconciliation ≥ 80%" — and that is a different question, measured
    against known-missing rows in evaluation.
    """
    verifiable = [r for r in reports if r.verifiable]
    if not verifiable:
        return 0.0
    return sum(1 for r in verifiable if r.reconciled) / len(verifiable)
