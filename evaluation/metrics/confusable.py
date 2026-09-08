"""Canonical-mapping metrics: confusable misattribution and alias accuracy (arch §0c).

Two metrics, deliberately treated differently.

**Confusable misattribution — GATING.** How often a confusable entity's value is
returned as the canonical field: the certificate holder emitted as
``insured_name``. It is isolated from ordinary field accuracy because it is
*systematic rather than random*. A random wrong value is noise; this means the
model has collapsed two distinct entities, it will keep doing so on every similar
document, and the output is fluent and schema-valid enough to pass every
structural check. Nothing else in the pipeline catches it.

**Alias accuracy — REPORTED, not gating.** Field accuracy sliced by the observed
surface label, joined through the ``field_provenance`` captured at labeling time.
It turns "field accuracy is 0.87" into "0.94 on *Named Insured*, 0.61 on
*Applicant*", which names the documents to go collect. It is not a gate because
a rare alias has too little support for a stable threshold — gating on it would
block good models on noise.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from common import aliases as alias_registry
from common.normalize import values_match


@dataclass
class MisattributionCase:
    """One field that took a confusable entity's value."""

    source_id: str
    field_path: str
    expected: Any
    got: Any
    stolen_from: str          # the canonical field whose value was returned
    confusable_label: str | None = None


@dataclass
class MisattributionReport:
    """Confusable misattribution across the eval set."""

    cases: list[MisattributionCase] = field(default_factory=list)
    evaluated_fields: int = 0

    @property
    def rate(self) -> float:
        return len(self.cases) / self.evaluated_fields if self.evaluated_fields else 0.0

    @property
    def by_field(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for case in self.cases:
            counts[case.field_path] += 1
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        if not self.cases:
            return f"no misattribution across {self.evaluated_fields} fields"
        pairs = ", ".join(f"{c.field_path} <- {c.stolen_from}" for c in self.cases[:4])
        return f"{self.rate:.1%} misattribution ({len(self.cases)} cases): {pairs}"


def score_misattribution(
    expected: dict[str, Any],
    got: dict[str, Any],
    doc_type: str,
    *,
    source_id: str = "",
) -> MisattributionReport:
    """Detect fields that returned a *different* field's correct value.

    The check is deliberately specific: the value must be wrong for this field
    **and** right for another one. A merely wrong value is ordinary field error;
    this is the model conflating two parties, which is a different failure with a
    different fix.
    """
    report = MisattributionReport()

    for field_path, expected_value in sorted(expected.items()):
        if isinstance(expected_value, (list, dict)):
            continue
        got_value = got.get(field_path)
        report.evaluated_fields += 1

        if got_value is None or values_match(expected_value, got_value, field_path=field_path):
            continue

        for other_field, other_expected in expected.items():
            if other_field == field_path or isinstance(other_expected, (list, dict)):
                continue
            if other_expected is None:
                continue
            if values_match(other_expected, got_value, field_path=other_field):
                # The filter this comment always described, now actually applied.
                # Counting every coincidental collision put unrelated duplicate
                # values into a GATING metric: two cities that happen to match
                # blocked a promotion, and the rate stopped measuring the
                # entity-collapse failure it is named for.
                confusables = alias_registry.confusables_for(doc_type, field_path)
                label = alias_registry.aliases_for(doc_type, other_field)
                registered = bool(confusables) and bool(label) and any(c in label for c in confusables)
                if confusables and label and not registered:
                    continue
                report.cases.append(
                    MisattributionCase(
                        source_id=source_id,
                        field_path=field_path,
                        expected=expected_value,
                        got=got_value,
                        stolen_from=other_field,
                        confusable_label=next(
                            (c for c in confusables if c in label), label[0] if label else None
                        ),
                    )
                )
                break
    return report


def aggregate_misattribution(reports: list[MisattributionReport]) -> MisattributionReport:
    """Combine per-document reports into a corpus-level rate."""
    combined = MisattributionReport()
    for report in reports:
        combined.cases.extend(report.cases)
        combined.evaluated_fields += report.evaluated_fields
    return combined


# --------------------------------------------------------------------------
# Alias accuracy
# --------------------------------------------------------------------------

@dataclass
class AliasAccuracyReport:
    """Field accuracy sliced by the surface label the document actually used."""

    #: field -> surface label -> (correct, total)
    counts: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)

    def accuracy(self, field_path: str, surface_label: str) -> float:
        correct, total = self.counts.get(field_path, {}).get(surface_label, (0, 0))
        return correct / total if total else 0.0

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            field_path: {
                label: round(correct / total, 4) if total else 0.0
                for label, (correct, total) in sorted(labels.items())
            }
            for field_path, labels in sorted(self.counts.items())
        }

    def weakest_variants(self, min_support: int = 2) -> list[tuple[str, str, float, int]]:
        """Variants scoring worst, with enough support to be worth acting on.

        This is the output that names which documents to go collect: a field at
        0.94 on its dominant label and 0.61 on a rare one is a coverage problem,
        not a model problem.
        """
        rows = [
            (field_path, label, correct / total, total)
            for field_path, labels in self.counts.items()
            for label, (correct, total) in labels.items()
            if total >= min_support
        ]
        return sorted(rows, key=lambda r: r[2])

    def spread(self, field_path: str, min_support: int = 2) -> float | None:
        """Gap between a field's best and worst surface variant.

        A large spread means the model memorised label strings rather than
        learning the semantic mapping — the central claim of arch §0c.
        """
        scores = [
            correct / total
            for _label, (correct, total) in self.counts.get(field_path, {}).items()
            if total >= min_support
        ]
        return max(scores) - min(scores) if len(scores) >= 2 else None


def score_alias_accuracy(
    documents: list[tuple[dict[str, Any], dict[str, Any], dict[str, str]]],
) -> AliasAccuracyReport:
    """Score accuracy per ``(canonical field, observed surface label)``.

    Args:
        documents: ``(expected, got, field_provenance)`` triples. Provenance is
            what makes the slicing possible; without it there is only an
            aggregate, and an aggregate cannot say *which* phrasing is failing.
    """
    report = AliasAccuracyReport()

    for expected, got, provenance in documents:
        for field_path, surface_label in provenance.items():
            if field_path not in expected:
                continue
            correct = values_match(expected[field_path], got.get(field_path), field_path=field_path)
            bucket = report.counts.setdefault(field_path, {})
            hits, total = bucket.get(surface_label, (0, 0))
            bucket[surface_label] = (hits + int(correct), total + 1)

    return report
