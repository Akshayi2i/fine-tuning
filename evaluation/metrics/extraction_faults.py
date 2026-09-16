"""False nulls, hallucinations and page-selection recall (arch v2.1 §15.2).

Three gating metrics that share a shape — each counts a specific way an
extraction can be wrong that **field accuracy cannot see** — and that is why they
are separate metrics rather than folded into one number.

* **False null.** The value is on the page and the model emitted ``null``. An
  aggregate field score counts this as one miss among many, but it is the failure
  mode that quietly halves a form's usefulness while looking like ordinary
  imperfection. It also produces no low-confidence signal: a null has no tokens
  to be uncertain about, so §5 confidence is blind to it and only this metric
  sees it.

* **Hallucination.** A value was emitted that appears nowhere in the document. A
  fluent, plausible, invented policy number scores as one wrong field — but a
  wrong value is worse than an absent one, because an absent one gets reviewed
  and a confident wrong one gets used.

* **Page-selection recall.** For routed policies: a field-bearing page the
  selector did not choose. The extraction that follows cannot be right about a
  page it never saw, so this caps everything downstream of it — and it fails
  silently, because the extraction looks complete against the pages it did get.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from common.normalize import normalize_text, values_match


@dataclass
class FaultReport:
    """One fault type's rate, with the instances that produced it."""

    name: str
    hits: int = 0
    opportunities: int = 0
    instances: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.hits / self.opportunities if self.opportunities else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.name,
            "rate": round(self.rate, 4),
            "hits": self.hits,
            "opportunities": self.opportunities,
            # Capped: the rate is the metric, the instances are for diagnosis,
            # and a report carrying every instance of a bad run is unreadable.
            "instances": self.instances[:25],
        }


def _is_empty(value: Any) -> bool:
    if isinstance(value, (list, dict)):
        return not value
    if isinstance(value, str):
        return not value.strip()
    return value is None


def score_false_nulls(
    documents: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    source_ids: Sequence[str] | None = None,
) -> FaultReport:
    """Fields the golden label has a value for, which the model emitted as null.

    The denominator is **opportunities** — fields that actually had a value —
    not all fields. Dividing by every field would let a schema with many
    legitimately-empty fields dilute the rate toward zero.
    """
    from evaluation.metrics.field_accuracy import flatten_scalars

    report = FaultReport("false_null_rate")
    for index, (expected, got) in enumerate(documents):
        source_id = source_ids[index] if source_ids and index < len(source_ids) else str(index)
        expected_flat = flatten_scalars(expected)
        got_flat = flatten_scalars(got)

        for path, truth in expected_flat.items():
            if _is_empty(truth):
                continue
            report.opportunities += 1
            if _is_empty(got_flat.get(path)):
                report.hits += 1
                report.instances.append({
                    "source_id": source_id, "field": path, "expected": str(truth)[:80],
                })
    return report


def score_hallucinations(
    documents: Sequence[tuple[dict[str, Any], dict[str, Any], str]],
) -> FaultReport:
    """Values the model emitted that appear nowhere in the document text.

    Each tuple is ``(expected, got, document_text)`` — the OCR text of the pages
    that were actually sent. Checking against the golden label alone would call
    every wrong value a hallucination, including an honest misread of something
    printed on the page, and those have different remedies: a misread wants
    better perception, an invention wants better grounding.

    A field is only checked when the model emitted something. Correctly-absent
    values cannot be hallucinated, and checking them would make the denominator
    meaningless.
    """
    from evaluation.metrics.field_accuracy import flatten_scalars

    report = FaultReport("hallucination_rate")
    for expected, got, text in documents:
        haystack = normalize_text(text) or ""
        expected_flat = flatten_scalars(expected)

        for path, value in flatten_scalars(got).items():
            if _is_empty(value) or isinstance(value, bool):
                continue
            report.opportunities += 1

            # Right answers are never hallucinations, whatever the text lookup
            # says — normalisation can legitimately make a correct value
            # unfindable as a substring (a date reformatted to ISO, a currency
            # figure stripped of its symbol).
            if values_match(expected_flat.get(path), value, field_path=path):
                continue

            needle = normalize_text(value)
            if needle and needle not in haystack:
                report.hits += 1
                report.instances.append({
                    "field": path, "emitted": str(value)[:80],
                    "reason": "value does not appear in the text of the pages that were sent",
                })
    return report


@dataclass
class PageSelectionReport:
    """Whether page selection found the pages carrying the fields (arch §7b)."""

    selected_relevant: int = 0
    relevant: int = 0
    selected_total: int = 0
    missed: list[dict[str, Any]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        """The gating number. Recall, not precision: an extra page costs tokens,
        a missing page costs the fields on it, and only one of those is
        recoverable downstream."""
        return self.selected_relevant / self.relevant if self.relevant else 1.0

    @property
    def precision(self) -> float:
        return self.selected_relevant / self.selected_total if self.selected_total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": "page_selection_recall",
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "relevant_pages": self.relevant,
            "selected_pages": self.selected_total,
            "missed": self.missed[:25],
        }


def score_page_selection(
    documents: Sequence[tuple[str, Sequence[int], Sequence[int]]],
) -> PageSelectionReport:
    """Score routed page selection against field provenance.

    Each tuple is ``(source_id, provenance_pages, selected_pages)``. The
    provenance pages come free from labelling (§7c): they are where the golden
    values were found.

    A document with no field-bearing pages contributes nothing rather than a
    perfect score — it is not evidence the selector works.
    """
    report = PageSelectionReport()
    for source_id, provenance, selected in documents:
        relevant = {int(p) for p in provenance}
        chosen = {int(p) for p in selected}
        if not relevant:
            continue

        report.relevant += len(relevant)
        report.selected_total += len(chosen)
        report.selected_relevant += len(relevant & chosen)

        for page in sorted(relevant - chosen):
            report.missed.append({
                "source_id": source_id, "page": page,
                "reason": "carries a golden value but was not selected; the extraction "
                          "cannot be right about a page it never saw",
            })
    return report
