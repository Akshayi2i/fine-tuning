"""The promotion gate (arch §13, §15).

A candidate is promoted only if it matches or exceeds the current production
version on **every** gating metric. **There is no override path, and adding one
would be a design error, not a convenience** — a gate that can be waived stops
being a guarantee and becomes a suggestion, and every other assurance in the
pipeline rests on it.

Two rules beyond the metric comparison:

* **Higher is better for most metrics, lower for two of them.** ECE and
  confusable misattribution are error rates; treating them like accuracy would
  promote a model that got worse at exactly the things hardest to notice.
* **A ``--continue-from`` Foundation must show cross-type regression evidence.**
  Continued training compounds drift, and a patch aimed at one document type can
  quietly degrade the others (arch §12). The gate refuses to promote one without
  proof it did not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)

Direction = Literal["higher_is_better", "lower_is_better"]

#: Every gating metric, and which direction counts as improvement.
#: `alias_accuracy` is deliberately absent: it is reported, not gated, because a
#: rare surface variant has too little support for a stable threshold and gating
#: on it would block good models on noise (arch §0c).
GATING_METRICS: dict[str, Direction] = {
    "field_exact_match": "higher_is_better",
    "field_normalized_match": "higher_is_better",
    "field_f1_list_fields": "higher_is_better",
    "list_field_recall": "higher_is_better",
    "schema_validity_rate": "higher_is_better",
    "ocr_arbitration_accuracy": "higher_is_better",
    "image_only_accuracy": "higher_is_better",
    "scanned_accuracy": "higher_is_better",
    "doc_type_classifier_accuracy": "higher_is_better",
    "lob_detection_accuracy": "higher_is_better",
    "ece_confidence": "lower_is_better",
    "confusable_misattribution_rate": "lower_is_better",
}

#: Movement smaller than this is noise, not regression. Without it a model would
#: be blocked by float jitter on an unchanged metric.
DEFAULT_TOLERANCE = 0.001


@dataclass
class MetricDelta:
    """One metric's movement between the current version and the candidate."""

    name: str
    current: float | None
    candidate: float | None
    direction: Direction
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def delta(self) -> float | None:
        if self.current is None or self.candidate is None:
            return None
        return round(self.candidate - self.current, 6)

    @property
    def regressed(self) -> bool:
        if self.current is None or self.candidate is None:
            return False
        if self.direction == "higher_is_better":
            return self.candidate < self.current - self.tolerance
        return self.candidate > self.current + self.tolerance

    @property
    def unmeasured(self) -> bool:
        """The candidate has no value for this metric.

        Independent of whether a baseline exists. Requiring ``current is not
        None`` made ``require_all_measured`` a no-op for the first version —
        ``promotion_gate({}, None)`` passed with zero metrics measured, and v1 is
        the version that defines the baseline everything after it is gated
        against. It was also a no-op for any metric the baseline happened to
        lack, which is how an unmeasured metric becomes permanently unmeasured.
        """
        return self.candidate is None

    def describe(self) -> str:
        if self.candidate is None:
            return f"{self.name}: NOT MEASURED (current {self.current})"
        if self.current is None:
            return f"{self.name}: {self.candidate:.4f} (no baseline)"
        arrow = "↓" if (self.delta or 0) < 0 else "↑"
        return f"{self.name}: {self.current:.4f} {arrow} {self.candidate:.4f} ({self.delta:+.4f})"


@dataclass
class GateResult:
    """The gate's decision, and everything behind it."""

    passed: bool
    deltas: list[MetricDelta] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    is_first_version: bool = False

    def report(self) -> str:
        lines = [f"promotion gate: {'PASS' if self.passed else 'BLOCKED'}"]
        if self.is_first_version:
            lines.append("  (no previous version — nothing to regress against)")
        lines.extend(f"  {d.describe()}" for d in self.deltas)
        lines.extend(f"  BLOCKED: {reason}" for reason in self.blocking_reasons)
        return "\n".join(lines)


def promotion_gate(
    candidate_metrics: dict[str, Any],
    current_metrics: dict[str, Any] | None,
    *,
    continued_from: str | None = None,
    cross_type_evidence: dict[str, Any] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    require_all_measured: bool = True,
) -> GateResult:
    """Decide whether a candidate may be promoted.

    Args:
        candidate_metrics: the candidate's scores against the frozen golden eval set.
        current_metrics: the production version's scores, or ``None`` for the first.
        continued_from: set when the Foundation continued from a checkpoint rather
            than retraining from base. Demands cross-type evidence (arch §12).
        cross_type_evidence: per-doc-type metrics proving the other types did not
            regress.
        require_all_measured: block when a gating metric is missing. A metric that
            was not measured has not passed — treating absence as success is how a
            regression ships.

    Returns:
        A :class:`GateResult`. **There is no parameter that forces a pass.**
    """
    result = GateResult(passed=True, is_first_version=current_metrics is None)
    baseline = current_metrics or {}

    for name, direction in GATING_METRICS.items():
        delta = MetricDelta(
            name=name,
            current=baseline.get(name),
            candidate=candidate_metrics.get(name),
            direction=direction,
            tolerance=tolerance,
        )
        result.deltas.append(delta)

        if delta.regressed:
            result.failed_gates.append(name)
            result.blocking_reasons.append(
                f"{name} regressed: {delta.current:.4f} -> {delta.candidate:.4f} "
                f"({delta.delta:+.4f})"
            )
        elif require_all_measured and delta.unmeasured:
            result.failed_gates.append(name)
            result.blocking_reasons.append(
                f"{name} was not measured for the candidate. A metric that was not measured "
                "has not passed — treating its absence as success is how a regression ships."
            )

    # A continued Foundation must prove it did not degrade the OTHER types.
    if continued_from:
        if not cross_type_evidence:
            result.blocking_reasons.append(
                f"this Foundation continued from {continued_from} rather than retraining from "
                "base, so promotion requires cross-type regression evidence: continued training "
                "compounds drift, and a patch aimed at one document type can quietly degrade the "
                "others (arch §12)."
            )
            result.failed_gates.append("cross_type_regression_evidence")
        else:
            for doc_type, metrics in sorted(cross_type_evidence.items()):
                candidate_side = metrics.get("candidate") or {}
                baseline_side = metrics.get("current") or {}
                if not candidate_side or not baseline_side:
                    # An empty entry is the absence of evidence, not evidence of
                    # no regression. Passing it made the whole cross-type
                    # requirement satisfiable with `{"acord": {}}`.
                    result.failed_gates.append(f"cross_type:{doc_type}")
                    result.blocking_reasons.append(
                        f"cross-type evidence for {doc_type} is empty "
                        f"(candidate: {len(candidate_side)} metric(s), "
                        f"baseline: {len(baseline_side)}). A continued Foundation must show it "
                        "did not degrade the other document types; an empty entry shows nothing."
                    )
                    continue

                sub = promotion_gate(
                    candidate_side, baseline_side,
                    tolerance=tolerance, require_all_measured=False,
                )
                if not sub.passed:
                    result.failed_gates.append(f"cross_type:{doc_type}")
                    result.blocking_reasons.append(
                        f"the continued Foundation regressed {doc_type}: {'; '.join(sub.blocking_reasons)}"
                    )

    result.passed = not result.blocking_reasons
    log.info("%s", result.report())
    return result


def apply_to_manifest(result: GateResult, manifest: Any, *, gated_against: str | None = None) -> Any:
    """Record the gate decision on the candidate's run manifest (SPEC_02).

    The decision travels with the artifact, so "was this promoted, and against
    what" is answerable later without reconstructing it.
    """
    manifest.promotion.beat_previous_on_all_gates = result.passed
    manifest.promotion.failed_gates = list(result.failed_gates)
    if gated_against:
        manifest.promotion.gated_against = gated_against
    manifest.status = "evaluated"
    return manifest
