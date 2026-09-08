"""LoB accuracy, schema validity, calibration error, and mode accuracy (arch §15).

Four metrics that share a shape — each aggregates per-document outcomes into one
number the promotion gate reads — but each answers a different question:

* **LoB accuracy**, reported **per value**, never folded into overall field
  accuracy: a class that is rare in the corpus must not hide inside a healthy
  aggregate (arch §0b).
* **Schema validity**, which mirrors the Fideon SPEC_07 Stage 3 audit gate.
* **ECE**, which asks whether the confidence numbers are trustworthy — not
  whether the extractions are.
* **Mode accuracy**, split by ``image_only`` and ``scanned``, which feeds the ViT
  escalation gate. It also classifies errors, because that gate keys on the
  perception-vs-reasoning distinction rather than on the accuracy number.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from common.lob import lob_values
from common.normalize import values_match
from common.schemas import is_valid

# --------------------------------------------------------------------------
# Line of Business (arch §0b)
# --------------------------------------------------------------------------

@dataclass
class LobReport:
    """LoB detection accuracy, overall and per value."""

    per_value: dict[str, tuple[int, int]] = field(default_factory=dict)
    null_correct: int = 0
    null_total: int = 0

    @property
    def overall(self) -> float:
        correct = sum(c for c, _ in self.per_value.values()) + self.null_correct
        total = sum(t for _, t in self.per_value.values()) + self.null_total
        return correct / total if total else 0.0

    def accuracy_by_value(self) -> dict[str, float]:
        return {
            value: round(correct / total, 4) if total else 0.0
            for value, (correct, total) in sorted(self.per_value.items())
        }

    def unmeasured_values(self) -> list[str]:
        """LoB values with no eval documents at all.

        An unmeasured class is not a passing class — it is one whose accuracy
        nobody knows, and the aggregate will not say so.
        """
        return sorted(v for v in lob_values() if self.per_value.get(v, (0, 0))[1] == 0)


def score_lob(documents: list[tuple[dict[str, Any], dict[str, Any]]]) -> LobReport:
    """Score ``line_of_business`` per value across the eval set."""
    report = LobReport()
    for expected, got in documents:
        truth = expected.get("line_of_business")
        prediction = got.get("line_of_business")
        correct = truth == prediction

        if truth is None:
            report.null_total += 1
            report.null_correct += int(correct)
        else:
            hits, total = report.per_value.get(truth, (0, 0))
            report.per_value[truth] = (hits + int(correct), total + 1)
    return report


# --------------------------------------------------------------------------
# Schema validity
# --------------------------------------------------------------------------

@dataclass
class SchemaValidityReport:
    valid: int = 0
    total: int = 0
    invalid_source_ids: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.valid / self.total if self.total else 0.0


def score_schema_validity(
    outputs: list[tuple[str, dict[str, Any], str, str | None]],
) -> SchemaValidityReport:
    """Validate each output against its document type's schema.

    Mirrors the audit gate that runs on every production call, so a validity
    regression is caught before promotion rather than in production.
    """
    report = SchemaValidityReport()
    for source_id, output, doc_type, acord_form in outputs:
        report.total += 1
        if is_valid(output, doc_type, acord_form):
            report.valid += 1
        else:
            report.invalid_source_ids.append(source_id)
    return report


# --------------------------------------------------------------------------
# Expected Calibration Error
# --------------------------------------------------------------------------

def expected_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    *,
    bins: int = 10,
) -> float:
    """ECE — the gap between stated confidence and observed accuracy.

    Answers whether the confidence numbers can be trusted, which is a different
    question from whether the extractions are right. A model can be 95% accurate
    and badly calibrated, and the review routing built on its confidence would
    then send the wrong documents to humans.
    """
    if len(confidences) != len(correctness):
        raise ValueError(
            f"{len(confidences)} confidences but {len(correctness)} outcomes — they must align"
        )
    if not confidences:
        return 0.0

    total = len(confidences)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (c, ok) for c, ok in zip(confidences, correctness, strict=True)
            if (low < c <= high) or (index == 0 and c == 0.0)
        ]
        if not members:
            continue
        mean_confidence = sum(c for c, _ in members) / len(members)
        accuracy = sum(ok for _, ok in members) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return round(error, 4)


def reliability_bins(
    confidences: list[float], correctness: list[bool], *, bins: int = 10
) -> list[dict[str, float]]:
    """Per-bin confidence vs accuracy — the shape behind the ECE number.

    Useful because the direction matters: over-confidence routes too little to
    review, under-confidence routes too much, and one ECE value cannot say which.
    """
    out: list[dict[str, float]] = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (c, ok) for c, ok in zip(confidences, correctness, strict=True) if low < c <= high
        ]
        if not members:
            continue
        out.append({
            "bin_low": low,
            "bin_high": high,
            "count": len(members),
            "mean_confidence": round(sum(c for c, _ in members) / len(members), 4),
            "accuracy": round(sum(ok for _, ok in members) / len(members), 4),
        })
    return out


# --------------------------------------------------------------------------
# Mode accuracy and error classification (feeds the ViT gate)
# --------------------------------------------------------------------------

@dataclass
class ModeReport:
    """Accuracy per input regime, with document counts.

    Counts travel with the accuracy because the ViT gate needs them: below its
    document floor the gate returns ``insufficient_data`` rather than deciding
    (arch §16c).
    """

    accuracy_by_mode: dict[str, float] = field(default_factory=dict)
    documents_by_mode: dict[str, int] = field(default_factory=dict)
    scanned_accuracy: float = 0.0
    scanned_documents: int = 0
    error_classes: dict[str, int] = field(default_factory=dict)

    def error_mix(self) -> dict[str, float]:
        total = sum(self.error_classes.values())
        return (
            {k: round(v / total, 3) for k, v in sorted(self.error_classes.items())}
            if total else {}
        )


def score_by_mode(
    documents: list[tuple[dict[str, Any], dict[str, Any], str, bool]],
) -> ModeReport:
    """Score accuracy split by modality mode and by scanned-vs-digital.

    Args:
        documents: ``(expected, got, modality_mode, is_scanned)`` tuples.
    """
    from training.vit_gate import classify_error

    by_mode: dict[str, list[float]] = defaultdict(list)
    scanned: list[float] = []
    error_counter: Counter = Counter()

    for expected, got, mode, is_scanned in documents:
        outcomes: list[bool] = []
        for field_path, expected_value in expected.items():
            if isinstance(expected_value, (list, dict)):
                continue
            got_value = got.get(field_path)
            correct = values_match(expected_value, got_value, field_path=field_path)
            outcomes.append(correct)
            if not correct:
                error_counter[classify_error(expected_value, got_value, all_expected=expected)] += 1

        accuracy = sum(outcomes) / len(outcomes) if outcomes else 0.0
        by_mode[mode].append(accuracy)
        if is_scanned:
            scanned.append(accuracy)

    report = ModeReport(
        accuracy_by_mode={
            mode: round(sum(scores) / len(scores), 4) for mode, scores in sorted(by_mode.items())
        },
        documents_by_mode={mode: len(scores) for mode, scores in sorted(by_mode.items())},
        scanned_accuracy=round(sum(scanned) / len(scanned), 4) if scanned else 0.0,
        scanned_documents=len(scanned),
        error_classes=dict(error_counter),
    )
    return report
