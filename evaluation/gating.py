"""The promotion gate (arch v2.1 §15.5).

**Why v1's gate was replaced.** It required the candidate to match or beat the
current release on *every* one of twelve metrics, within a flat tolerance of
0.001, with no override — on the reasoning that a gate which can be waived stops
being a guarantee.

That was right about the risk and wrong about the remedy. At pilot volume the
eval set holds three or four documents per type, so classification accuracy can
only take the values 0, 0.25, 0.5, 0.75, 1.0: the tolerance was two hundred and
fifty times finer than the measurement could resolve. With twelve metrics each
needing not to move down, a genuinely-equal model passed all twelve with
probability near 0.5¹² — about 0.02%. And it does not improve at scale: at n=50
the standard error of a proportion is ~0.07, still seventy times the tolerance.

A gate that cannot be passed and cannot be waived is not a strict gate. It is a
rule everyone agrees to ignore, on the afternoon it first blocks something
obviously fine.

**Three conditions replace it**, all of which must hold:

1. **Absolute floors.** Every gating metric meets its §0d floor. This applies to
   the first release too, which has no predecessor — v1's gate passed a first
   version on *zero* measured metrics.
2. **Non-inferiority.** For each metric, the lower bound of the 95% CI on
   (candidate − current), from a paired bootstrap over documents, sits above −δ.
   Per-metric δ, because a 1pp drop in identifier accuracy and a 1pp drop in
   free-text similarity are not the same event.
3. **Improvement.** At least one primary metric improves with its CI lower bound
   above zero — or the release fixes a documented defect. Without this a model
   could pass forever on non-inferiority alone.

**And an override exists**, requiring a named person and a written reason, both
recorded in the gate decision and the release bundle. A waiver nobody can
attribute later is the thing actually worth preventing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from evaluation.bootstrap import ConfidenceInterval, paired_bootstrap

log = logging.getLogger(__name__)

Direction = Literal["higher_is_better", "lower_is_better"]

#: Every gating metric, and which direction counts as improvement.
#: `alias_accuracy` is deliberately absent: it is reported, not gated, because a
#: rare surface variant has too little support for a stable threshold and gating
#: on it would block good models on noise (arch §0c). So is
#: `held_out_carrier_match` — reported, because a two-carrier slice is too small
#: to gate on and its whole purpose is to be harder than the rest.
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
    "false_null_rate": "lower_is_better",
}

#: Gating under arch v2.1 §15.2, but **not enforced yet** — nothing produces
#: them. Each names the phase that turns it on.
#:
#: This distinction is not a softening of the gate; it is the same mistake being
#: avoided twice. `require_all_measured` blocks on any unmeasured metric, and
#: rightly so — absence is not success. But listing a metric no producer emits
#: makes the gate permanently unpassable, which is precisely the defect this
#: rewrite exists to fix. A metric moves up into GATING_METRICS in the same
#: commit as the code that computes it, and `test_module_seams` checks that what
#: the gate demands is what the scorer emits.
PENDING_GATING_METRICS: dict[str, str] = {
    "list_field_precision": "row alignment (§15.1 Hungarian matching)",
    "page_selection_recall": "serving page_select task (§7b)",
    "lossrun_totals_reconciliation_rate": "totals reconciliation (§5.5)",
    "hallucination_rate": "OCR text carried through eval metadata (§15.2)",
    "auto_accept_error_rate": "calibrated review thresholds (§5.4)",
}

#: Absolute floors (arch v2.1 §0d, pilot-exit column). The single source of
#: targets: §16.3's pilot criteria reference the same table. Production values
#: are set by the product owner at pilot exit, which is why these are named
#: pilot floors rather than hardcoded forever.
PILOT_FLOORS: dict[str, float] = {
    "field_exact_match": 0.85,
    "field_normalized_match": 0.85,
    "list_field_recall": 0.75,
    "schema_validity_rate": 0.98,
    "doc_type_classifier_accuracy": 0.95,
    "lob_detection_accuracy": 0.85,
    "confusable_misattribution_rate": 0.05,
}

#: Non-inferiority margins in absolute units (arch v2.1 §15.5). Per metric,
#: because a 1pp drop on identifiers is a different event from 1.5pp on names:
#: a wrong policy number is a wrong extraction, a slightly-off entity name is
#: usually still matchable.
NON_INFERIORITY_DELTA: dict[str, float] = {
    "field_exact_match": 0.010,
    "field_normalized_match": 0.010,
    "field_f1_list_fields": 0.015,
    "list_field_recall": 0.010,
    "schema_validity_rate": 0.005,
    "ocr_arbitration_accuracy": 0.015,
    "image_only_accuracy": 0.015,
    "scanned_accuracy": 0.015,
    "doc_type_classifier_accuracy": 0.005,
    "lob_detection_accuracy": 0.015,
    "ece_confidence": 0.010,
    "confusable_misattribution_rate": 0.010,
    "false_null_rate": 0.010,
}
DEFAULT_DELTA = 0.015

#: At least one of these must improve, or the release must fix a documented
#: defect. They are the metrics the business actually experiences.
PRIMARY_METRICS: tuple[str, ...] = ("field_normalized_match", "list_field_recall")


class GateError(RuntimeError):
    """Raised when a gate decision cannot be made."""


@dataclass
class MetricVerdict:
    """One metric's outcome: its floor, and its movement against the baseline."""

    name: str
    direction: Direction
    candidate: float | None
    current: float | None = None
    floor: float | None = None
    interval: ConfidenceInterval | None = None
    delta: float = DEFAULT_DELTA

    @property
    def unmeasured(self) -> bool:
        return self.candidate is None

    @property
    def meets_floor(self) -> bool:
        """Whether the absolute floor is met. Applies with or without a baseline."""
        if self.candidate is None:
            return False
        if self.floor is None:
            return True
        return (
            self.candidate >= self.floor if self.direction == "higher_is_better"
            else self.candidate <= self.floor
        )

    @property
    def non_inferior(self) -> bool:
        """Whether the candidate is not meaningfully worse than the baseline.

        With no baseline this is vacuously true — the floor is what gates a first
        release. With a baseline but no per-document scores, it falls back to a
        point comparison against δ, which is weaker than a bootstrap and is
        recorded as such in ``basis``.
        """
        if self.current is None or self.candidate is None:
            return True
        if self.interval is not None:
            return self.interval.non_inferior(self.delta)
        signed = (
            self.candidate - self.current if self.direction == "higher_is_better"
            else self.current - self.candidate
        )
        return signed > -abs(self.delta)

    @property
    def improved(self) -> bool:
        """Improvement distinguishable from noise. Requires an interval."""
        return bool(self.interval and self.interval.improved)

    @property
    def basis(self) -> str:
        if self.current is None:
            return "floor_only"
        return "paired_bootstrap" if self.interval is not None else "point_estimate"

    def describe(self) -> str:
        if self.candidate is None:
            return f"{self.name}: NOT MEASURED"
        parts = [f"{self.name}: {self.candidate:.4f}"]
        if self.floor is not None:
            parts.append(f"floor {self.floor:.4f} {'✓' if self.meets_floor else '✗'}")
        if self.interval is not None:
            parts.append(self.interval.describe().split(": ", 1)[1])
        elif self.current is not None:
            parts.append(f"was {self.current:.4f} ({self.candidate - self.current:+.4f})")
        return " | ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "candidate": self.candidate,
            "current": self.current,
            "floor": self.floor,
            "delta": self.delta,
            "meets_floor": self.meets_floor,
            "non_inferior": self.non_inferior,
            "improved": self.improved,
            "basis": self.basis,
            "interval": self.interval.as_dict() if self.interval else None,
        }


@dataclass
class GateOverrideRecord:
    """A named, written waiver of specific gates (arch v2.1 §15.5)."""

    approver: str
    reason: str
    waived_gates: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.approver.strip() or self.approver.strip().lower() in {
            "system", "ci", "automated", "n/a", "none", "-",
        }:
            raise GateError(
                "a gate override must name the person accountable for it, not a system "
                "identity. A waiver nobody can attribute later is the thing worth preventing."
            )
        if len(self.reason.strip()) < 20:
            raise GateError(
                "a gate override needs a written reason, not a word. The reason is what a "
                "later reviewer reads when asking why this shipped."
            )
        if not self.waived_gates:
            raise GateError(
                "an override must name the gates it waives. A blanket override is the "
                "unaccountable waiver in a different costume."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "approver": self.approver,
            "reason": self.reason,
            "waived_gates": sorted(self.waived_gates),
        }


@dataclass
class GateResult:
    """The gate's decision, and everything behind it."""

    passed: bool
    verdicts: list[MetricVerdict] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    is_first_version: bool = False
    override: GateOverrideRecord | None = None

    #: Gates that failed but were waived. Kept separate from `failed_gates` so
    #: "passed with a waiver" is never indistinguishable from "passed".
    waived: list[str] = field(default_factory=list)

    @property
    def improved_metrics(self) -> list[str]:
        return [v.name for v in self.verdicts if v.improved]

    def report(self) -> str:
        verdict = "PASS" if self.passed else "BLOCKED"
        if self.passed and self.waived:
            verdict = f"PASS (with {len(self.waived)} waived gate(s))"
        lines = [f"promotion gate: {verdict}"]
        if self.is_first_version:
            lines.append("  first release — gated on absolute floors alone (arch v2.1 §15.5)")
        lines.extend(f"  {v.describe()}" for v in self.verdicts if not v.unmeasured)
        lines.extend(f"  UNMEASURED: {v.name}" for v in self.verdicts if v.unmeasured)
        lines.extend(f"  BLOCKED: {reason}" for reason in self.blocking_reasons)
        if self.override:
            lines.append(
                f"  OVERRIDE by {self.override.approver}: {self.override.reason} "
                f"(waived {sorted(self.override.waived_gates)})"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "is_first_version": self.is_first_version,
            "failed_gates": sorted(self.failed_gates),
            "waived_gates": sorted(self.waived),
            "blocking_reasons": list(self.blocking_reasons),
            "improved_metrics": self.improved_metrics,
            "verdicts": [v.as_dict() for v in self.verdicts],
            "override": self.override.as_dict() if self.override else None,
        }


def promotion_gate(
    candidate_metrics: dict[str, Any],
    current_metrics: dict[str, Any] | None,
    *,
    per_document: dict[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    floors: dict[str, float] | None = None,
    continued_from: str | None = None,
    cross_type_evidence: dict[str, Any] | None = None,
    require_all_measured: bool = True,
    fixes_defect: str | None = None,
    override: GateOverrideRecord | None = None,
) -> GateResult:
    """Decide whether a candidate release may be promoted.

    Args:
        candidate_metrics: aggregate scores over the frozen golden eval set.
        current_metrics: the promoted release's scores, or ``None`` for the first.
        per_document: ``{metric: (current_scores, candidate_scores)}``, aligned by
            index. Supplied, each metric is judged by a paired bootstrap rather
            than by comparing two point estimates — which is the whole reason
            this gate can be passed at all.
        floors: absolute floors, defaulting to the §0d pilot-exit table.
        fixes_defect: satisfies the improvement requirement when a release exists
            to fix something rather than to score higher.
        override: a named, written waiver. Validated — an override that cannot be
            attributed is refused.
    """
    result = GateResult(passed=True, is_first_version=current_metrics is None)
    baseline = current_metrics or {}
    floor_table = PILOT_FLOORS if floors is None else floors
    samples = per_document or {}

    if override is not None:
        override.validate()
        result.override = override

    for name, direction in GATING_METRICS.items():
        verdict = MetricVerdict(
            name=name,
            direction=direction,
            candidate=candidate_metrics.get(name),
            current=baseline.get(name),
            floor=floor_table.get(name),
            delta=NON_INFERIORITY_DELTA.get(name, DEFAULT_DELTA),
        )
        if name in samples and verdict.current is not None:
            current_scores, candidate_scores = samples[name]
            # Sign-normalise for lower-is-better metrics so "improvement" means
            # the same direction for every interval, and the CI can be read the
            # same way regardless of which metric it belongs to.
            if direction == "lower_is_better":
                current_scores = [-x for x in current_scores]
                candidate_scores = [-x for x in candidate_scores]
            verdict.interval = paired_bootstrap(name, current_scores, candidate_scores)
        result.verdicts.append(verdict)

        if verdict.unmeasured:
            if require_all_measured:
                _block(result, name,
                       f"{name} was not measured. A metric that was not measured has not "
                       "passed — treating its absence as success is how a regression ships.")
            continue
        if not verdict.meets_floor:
            _block(result, name,
                   f"{name} is {verdict.candidate:.4f} against a floor of {verdict.floor:.4f} "
                   "(arch v2.1 §0d). Floors apply to the first release too.")
        if not verdict.non_inferior:
            bound = verdict.interval.lower if verdict.interval else None
            detail = (
                f"95% CI lower bound {bound:+.4f} is below -{verdict.delta:.3f}"
                if bound is not None else
                f"dropped by more than {verdict.delta:.3f} (point estimate; no per-document "
                "scores were supplied, so this is weaker evidence than a bootstrap)"
            )
            _block(result, name, f"{name} is not non-inferior: {detail}")

    _require_improvement(result, fixes_defect)
    _require_cross_type_evidence(result, continued_from, cross_type_evidence, floor_table)

    # An override lifts only the gates it names, and only ones that actually
    # failed. Waiving a gate that passed would make the record say something
    # untrue about what the approver decided.
    if result.override:
        waived = sorted(set(result.override.waived_gates) & set(result.failed_gates))
        result.waived = waived
        result.failed_gates = [g for g in result.failed_gates if g not in waived]
        result.blocking_reasons = [
            r for r in result.blocking_reasons
            if not any(r.startswith(f"{g} ") or r.startswith(f"{g}:") for g in waived)
        ]
        unused = sorted(set(result.override.waived_gates) - set(waived))
        if unused:
            log.warning(
                "override names %s, which did not fail. Recorded as written, but the waiver "
                "covers nothing there.", unused,
            )

    result.passed = not result.blocking_reasons
    log.info("%s", result.report())
    return result


def _block(result: GateResult, gate: str, reason: str) -> None:
    if gate not in result.failed_gates:
        result.failed_gates.append(gate)
    result.blocking_reasons.append(reason)


def _require_improvement(result: GateResult, fixes_defect: str | None) -> None:
    """At least one primary metric must improve, or a defect must be named.

    Without this a model could pass forever on non-inferiority alone — every
    release no worse than the last, none better, and the cycle producing nothing.
    The first release is exempt: there is nothing to improve on.
    """
    if result.is_first_version or fixes_defect:
        return
    improved = [v.name for v in result.verdicts if v.name in PRIMARY_METRICS and v.improved]
    if improved:
        return
    measurable = [
        v for v in result.verdicts if v.name in PRIMARY_METRICS and v.interval is not None
    ]
    if not measurable:
        # No per-document scores means no interval, so "improved" is unknowable
        # rather than false. Blocking here would make the improvement rule fire
        # on missing evidence rather than on a missing improvement.
        log.warning(
            "no per-document scores for %s, so the improvement requirement cannot be "
            "evaluated and is not enforced. Supply per_document scores to enforce it.",
            list(PRIMARY_METRICS),
        )
        return
    _block(
        result, "improvement",
        f"no primary metric improved with its CI lower bound above zero {list(PRIMARY_METRICS)}. "
        "A release must be better at something, or name the documented defect it fixes "
        "(arch v2.1 §15.5).",
    )


def _require_cross_type_evidence(
    result: GateResult,
    continued_from: str | None,
    cross_type_evidence: dict[str, Any] | None,
    floors: dict[str, float],
) -> None:
    """A continued run must prove it did not degrade the OTHER document types.

    Continued training compounds drift, and a patch aimed at one type can quietly
    degrade the others (arch §12).
    """
    if not continued_from:
        return
    if not cross_type_evidence:
        _block(
            result, "cross_type_regression_evidence",
            f"this run continued from {continued_from} rather than retraining from base, so "
            "promotion requires cross-type regression evidence: continued training compounds "
            "drift, and a patch aimed at one document type can quietly degrade the others "
            "(arch §12).",
        )
        return

    for doc_type, metrics in sorted(cross_type_evidence.items()):
        candidate_side = metrics.get("candidate") or {}
        baseline_side = metrics.get("current") or {}
        if not candidate_side or not baseline_side:
            # An empty entry is the absence of evidence, not evidence of no
            # regression. Passing it made the whole requirement satisfiable
            # with `{"acord": {}}`.
            _block(
                result, f"cross_type:{doc_type}",
                f"cross-type evidence for {doc_type} is empty (candidate: "
                f"{len(candidate_side)} metric(s), baseline: {len(baseline_side)}). A "
                "continued run must show it did not degrade the other document types; an "
                "empty entry shows nothing.",
            )
            continue

        sub = promotion_gate(
            candidate_side, baseline_side,
            floors=floors,
            require_all_measured=False,
            # The per-type slice is small and its job is regression detection,
            # not proving the type got better.
            fixes_defect=f"cross-type regression check for {doc_type}",
        )
        if not sub.passed:
            _block(
                result, f"cross_type:{doc_type}",
                f"the continued run regressed {doc_type}: {'; '.join(sub.blocking_reasons)}",
            )


def apply_to_manifest(result: GateResult, manifest: Any, *, gated_against: str | None = None) -> Any:
    """Record the gate decision on the candidate's run manifest (SPEC_02).

    The decision travels with the artifact, so "was this promoted, and against
    what" is answerable later without reconstructing it.
    """
    manifest.promotion.beat_previous_on_all_gates = result.passed and not result.waived
    manifest.promotion.failed_gates = list(result.failed_gates) + [
        f"{g} (WAIVED)" for g in result.waived
    ]
    if gated_against:
        manifest.promotion.gated_against = gated_against
    manifest.status = "evaluated"
    return manifest
