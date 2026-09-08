"""Raw per-field confidence from token logprobs (arch §5).

The model already produces a probability for every token it generates, so
confidence costs nothing extra at training or inference time. This module turns
those into one number per field; :mod:`calibration.fit_calibration` then corrects
them, because raw logprobs from a fine-tuned model are systematically
overconfident.

**Default aggregation is the minimum token probability in the span.** It is the
most sensitive to the weakest link, which is what you want when the purpose is
flagging risky fields — a policy number that is nine confident tokens and one
uncertain one is a policy number worth checking. Mean would average that away.
The choice is configurable and empirical, and whichever is used is recorded.
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
