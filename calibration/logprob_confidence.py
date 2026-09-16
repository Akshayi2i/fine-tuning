"""Raw per-field logprob aggregation (arch v2.1 §5.1).

**Superseded as a confidence source.** What this module produces is now one
*feature* among nine, not the confidence itself — see :mod:`calibration.features`
and :mod:`calibration.feature_calibrator`.

v1 used the minimum token probability in a field's span as its confidence, on the
reasoning that the weakest link is what makes a field worth reviewing. Good
instinct, broken statistic: **the minimum of n draws falls as n grows**, so a long
correct value scores lower than a short wrong one. ``ABC-1234567-01`` is nine
tokens and ``2026`` is one — under min-aggregation the policy number looks less
trustworthy than the year, on every document, in the same direction every time.

That is not fixable by choosing a different aggregation. Mean washes out the
weak link the minimum was there to catch; geometric mean has the same length
dependence in gentler form. The fix is to hand the calibrator the minimum **and**
the mean **and** the first token **and** the length, plus the five non-logprob
features in §5.2, and let it learn what the combination means.

The functions here remain because those aggregates are still computed — as
inputs. Nothing downstream should treat a value from this module as a confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from inference_core.span_map import FieldSpan

Aggregation = Literal["min", "mean", "geomean"]


class ConfidenceError(RuntimeError):
    """Raised when confidence cannot be derived from a generation."""


@dataclass
class FieldConfidence:
    """One field's raw (uncalibrated) confidence."""

    field_path: str
    value: Any
    confidence: float
    token_count: int
    aggregation: Aggregation
    mapped: bool = True
    reason: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.mapped


def _probabilities(logprobs: list[float]) -> list[float]:
    return [math.exp(lp) for lp in logprobs]


def aggregate_span(logprobs: list[float], method: Aggregation = "min") -> float:
    """Collapse a span's token probabilities to one confidence."""
    if not logprobs:
        raise ConfidenceError("cannot aggregate an empty logprob span")
    probabilities = _probabilities(logprobs)

    if method == "min":
        return min(probabilities)
    if method == "mean":
        return sum(probabilities) / len(probabilities)
    if method == "geomean":
        # Computed in log space: a long span of small probabilities underflows
        # to zero if multiplied directly.
        return math.exp(sum(logprobs) / len(logprobs))
    raise ConfidenceError(f"unknown aggregation {method!r}; expected min, mean or geomean")


def field_confidences(
    spans: dict[str, FieldSpan],
    *,
    aggregation: Aggregation = "min",
) -> dict[str, FieldConfidence]:
    """Raw confidence per field, including the ones that could not be mapped.

    Unmapped fields are **included and flagged**, never dropped. A field with no
    confidence must not be indistinguishable from a confident one — that is
    exactly backwards for a signal whose job is flagging risk (SPEC_07).
    """
    out: dict[str, FieldConfidence] = {}
    for path, span in spans.items():
        if not span.mapped or not span.token_logprobs:
            out[path] = FieldConfidence(
                field_path=path, value=span.value, confidence=0.0,
                token_count=0, aggregation=aggregation, mapped=False,
                reason=span.reason or "no token span",
            )
            continue
        out[path] = FieldConfidence(
            field_path=path,
            value=span.value,
            confidence=round(aggregate_span(span.token_logprobs, aggregation), 6),
            token_count=len(span.token_logprobs),
            aggregation=aggregation,
        )
    return out


def overall_confidence(
    confidences: dict[str, FieldConfidence],
    *,
    method: Aggregation = "mean",
) -> float:
    """A single document-level number.

    Mean by default rather than min: one uncertain optional field should not make
    an otherwise-solid extraction look worthless. Per-field confidence is what
    drives review routing; this is for reporting and sorting.
    """
    usable = [c.confidence for c in confidences.values() if c.is_usable]
    if not usable:
        return 0.0
    if method == "min":
        return round(min(usable), 6)
    if method == "geomean":
        return round(math.exp(sum(math.log(max(c, 1e-12)) for c in usable) / len(usable)), 6)
    return round(sum(usable) / len(usable), 6)


def unusable_fields(confidences: dict[str, FieldConfidence]) -> list[tuple[str, str]]:
    """Fields with no confidence, and why — surfaced, not swallowed."""
    return [(c.field_path, c.reason or "unknown") for c in confidences.values() if not c.is_usable]
