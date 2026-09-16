"""Risk-controlled review thresholds (arch v2.1 §5.4).

v1 used one number for every field: ``review_threshold: 0.70``, marked "tunable,
a starting point". It was never tuned, and it could not be — nothing measured
what error rate it actually bought. A threshold that is not tied to an outcome is
a guess wearing a decimal point.

**What replaces it.** For each field type, choose the *lowest* threshold such
that the error rate among fields at or above it meets the §0d target — measured
with the **upper bound** of a 95% binomial interval, not the point estimate.

Lowest, because a higher threshold sends more fields to review than the guarantee
requires, and review is the expensive resource this whole confidence system
exists to ration.

Upper bound, because the point estimate on a small sample is optimistic by
construction: two errors in forty auto-accepted fields is a 5% point estimate and
an 18% upper bound, and the promise being made is about the future, not about
those forty.

**Fitted on the threshold half, never the calibration half** (§8.2). The
calibrator has already seen the errors in its own half and pulled its curve
toward them, so a threshold chosen there prices risk the model has already been
shown. That is the difference between a guarantee and a hope.

The output is a sentence rather than a number: *"at most 1% error among
auto-accepted money fields, with 95% confidence"*.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evaluation.bootstrap import binomial_upper_bound

log = logging.getLogger(__name__)

#: Target error rate among auto-accepted fields, per field type. Identifiers,
#: money and dates are unforgiving — a wrong policy number is a wrong extraction,
#: full stop — so they get the tightest target. Free text is never auto-accepted
#: at all, which is why it is absent rather than set to 1.0.
DEFAULT_ERROR_TARGETS: dict[str, float] = {
    "identifier": 0.01,
    "money": 0.01,
    "date": 0.01,
    "enum": 0.02,
    "entity": 0.05,
    "address": 0.05,
}

#: Below this many fields at or above a candidate threshold, the interval is too
#: wide to support any promise. The field type then routes everything to review.
MIN_ACCEPTED_SAMPLE = 30

#: What a target actually costs in evidence, with ZERO observed errors:
#:
#:      0 errors in 100 -> 3.7% upper bound
#:      0 errors in 200 -> 1.9%
#:      0 errors in 300 -> 1.3%
#:      0 errors in 400 -> 0.95%
#:
#: So a 1% guarantee needs roughly **400 clean auto-accepted fields of that type**
#: — a perfect run on 200 cannot buy it, and no threshold choice changes that.
#: The identifier/money/date targets below are therefore not reachable at pilot
#: volume, and those types route everything to review until the corpus grows.
#: That is the system working: the alternative is a promise the evidence does not
#: support. Revisit the targets at pilot exit alongside the §0d production
#: values, not before.

#: Candidate thresholds, scanned low to high. 0.01 granularity: finer than the
#: calibrator's resolution on a few hundred instances, and the scan is cheap.
_GRID = [i / 100 for i in range(50, 100)]


class ThresholdError(RuntimeError):
    """Raised when a threshold cannot be chosen safely."""


@dataclass
class FieldTypeThreshold:
    """One field type's threshold, and the guarantee it buys."""

    field_type: str
    threshold: float | None
    target_error_rate: float
    accepted: int = 0
    errors: int = 0
    reviewed: int = 0
    achieved_upper_bound: float = 1.0
    reason: str | None = None

    @property
    def enforced(self) -> bool:
        return self.threshold is not None

    @property
    def auto_accept_share(self) -> float:
        total = self.accepted + self.reviewed
        return self.accepted / total if total else 0.0

    def guarantee(self) -> str:
        """The promise, as a sentence somebody can be held to."""
        if not self.enforced:
            return (
                f"{self.field_type}: every field routed to review ({self.reason})"
            )
        return (
            f"{self.field_type}: at most {self.achieved_upper_bound:.1%} error among "
            f"auto-accepted fields (95% confidence), accepting {self.auto_accept_share:.0%} "
            f"of them at threshold {self.threshold:.2f}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_type": self.field_type,
            "threshold": self.threshold,
            "enforced": self.enforced,
            "target_error_rate": self.target_error_rate,
            "achieved_upper_bound": round(self.achieved_upper_bound, 4),
            "accepted": self.accepted,
            "errors": self.errors,
            "reviewed": self.reviewed,
            "auto_accept_share": round(self.auto_accept_share, 4),
            "guarantee": self.guarantee(),
            "reason": self.reason,
        }


def choose_threshold(
    field_type: str,
    scored: Sequence[tuple[float, bool]],
    *,
    target_error_rate: float | None = None,
    min_accepted: int = MIN_ACCEPTED_SAMPLE,
) -> FieldTypeThreshold:
    """The lowest threshold whose auto-accepted error rate meets the target.

    ``scored`` is ``(calibrated_confidence, was_correct)`` per field, from the
    **threshold half** of validation.
    """
    target = (
        target_error_rate if target_error_rate is not None
        else DEFAULT_ERROR_TARGETS.get(field_type)
    )
    if target is None:
        return FieldTypeThreshold(
            field_type, None, 1.0,
            reviewed=len(scored),
            reason="no error target defined for this field type; it is never auto-accepted",
        )

    if not scored:
        return FieldTypeThreshold(
            field_type, None, target,
            reason="no scored fields on the threshold half, so no guarantee can be made",
        )

    best: FieldTypeThreshold | None = None
    for candidate in _GRID:
        accepted = [correct for confidence, correct in scored if confidence >= candidate]
        if len(accepted) < min_accepted:
            # Not enough evidence at this threshold to promise anything. Scanning
            # on rather than stopping: a higher threshold accepts fewer fields,
            # so this only gets worse — but the loop is cheap and stopping early
            # on a non-monotonic sample would be a subtle bug.
            continue
        errors = sum(1 for correct in accepted if not correct)
        bound = binomial_upper_bound(errors, len(accepted))
        if bound <= target:
            best = FieldTypeThreshold(
                field_type=field_type,
                threshold=candidate,
                target_error_rate=target,
                accepted=len(accepted),
                errors=errors,
                reviewed=len(scored) - len(accepted),
                achieved_upper_bound=bound,
            )
            break   # lowest threshold that meets the target — review is the
                    # expensive resource, so accept as much as the promise allows

    if best is None:
        return FieldTypeThreshold(
            field_type, None, target,
            reviewed=len(scored),
            reason=(
                f"no threshold on [{_GRID[0]:.2f}, {_GRID[-1]:.2f}] achieves a {target:.0%} "
                f"error bound with at least {min_accepted} accepted fields. Everything of this "
                "type routes to review — which is the honest outcome, not a failure of the scan."
            ),
        )
    log.info("%s", best.guarantee())
    return best


@dataclass
class ThresholdSet:
    """Every field type's threshold for one release and serving format."""

    release_id: str
    serving_format: str
    thresholds: dict[str, FieldTypeThreshold] = field(default_factory=dict)

    def for_type(self, field_type: str) -> FieldTypeThreshold | None:
        return self.thresholds.get(field_type)

    def needs_review(self, field_type: str, confidence: float | None) -> bool:
        """Whether this field goes to a human.

        A field with no confidence (no calibrator, or an unmapped span) is
        reviewed. A field type with no enforced threshold is reviewed. Absence is
        never treated as acceptance.
        """
        if confidence is None:
            return True
        threshold = self.thresholds.get(field_type)
        if threshold is None or not threshold.enforced:
            return True
        return confidence < (threshold.threshold or 1.0)

    @property
    def unenforced_types(self) -> list[str]:
        return sorted(t for t, v in self.thresholds.items() if not v.enforced)

    def guarantees(self) -> list[str]:
        return [v.guarantee() for _, v in sorted(self.thresholds.items())]

    def as_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "serving_format": self.serving_format,
            "fitted_on": "validation threshold half (arch v2.1 §8.2)",
            "unenforced_types": self.unenforced_types,
            "guarantees": self.guarantees(),
            "thresholds": {t: v.as_dict() for t, v in sorted(self.thresholds.items())},
        }


def fit_thresholds(
    scored_by_type: dict[str, Sequence[tuple[float, bool]]],
    *,
    release_id: str,
    serving_format: str,
    targets: dict[str, float] | None = None,
) -> ThresholdSet:
    """Choose every field type's threshold from the validation threshold half."""
    table = targets or DEFAULT_ERROR_TARGETS
    result = ThresholdSet(release_id=release_id, serving_format=serving_format)
    for field_type, scored in sorted(scored_by_type.items()):
        result.thresholds[field_type] = choose_threshold(
            field_type, scored, target_error_rate=table.get(field_type)
        )
    if result.unenforced_types:
        log.warning(
            "release %s/%s has no enforced threshold for %s — every field of those types is "
            "reviewed. That is the honest outcome when the evidence cannot support a promise.",
            release_id, serving_format, result.unenforced_types,
        )
    return result


def auto_accept_error_rate(
    scored_by_type: dict[str, Sequence[tuple[float, bool]]],
    thresholds: ThresholdSet,
) -> float:
    """The error rate among fields the thresholds would auto-accept.

    The gating metric (arch v2.1 §15.2) — and the one the business actually
    experiences, because it is the rate of wrong values that reached a user
    without a human looking at them.
    """
    accepted = 0
    errors = 0
    for field_type, scored in scored_by_type.items():
        for confidence, correct in scored:
            if thresholds.needs_review(field_type, confidence):
                continue
            accepted += 1
            errors += int(not correct)
    return errors / accepted if accepted else 0.0
