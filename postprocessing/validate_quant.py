"""Quantized-format validation against the fp16 reference (IMPL-10 §4).

Every quantized format **intended for serving** passes this before promotion.
There is no override flag, for the same reason the promotion gate has none: an
override exists to be used on the afternoon someone is in a hurry, which is
exactly the afternoon it should not be.

The scoring itself is not reimplemented here. Each format runs through the
IMPL-12 extraction routine against the frozen golden eval set — the same code
path production uses — and this module compares the resulting metrics to fp16's.
A second scoring implementation would mean the numbers this gate reads describe a
system that is not the one being served.

**The drop is measured relative, the ECE and validity floors absolutely.** A 2%
relative accuracy loss is the meaningful quantity because it means the same thing
at F1 0.95 and 0.70; an ECE increase of 0.02 does not — a calibration error is an
absolute probability error either way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from postprocessing.quant_thresholds import (
    DEFAULT_SERVING_FORMAT,
    REFERENCE_FORMAT,
    is_reference,
    threshold_for,
)

log = logging.getLogger(__name__)

#: Floating-point slack on every threshold comparison. A format landing exactly
#: on its allowance must pass: `0.90 * (1 - 0.02)` recomputes to a drop of
#: 0.020000000000000018, so a bare `>` blocked a format that was precisely at
#: spec. The tolerance is far below any difference that could matter — the
#: thresholds are stated to three decimal places at most.
EPSILON = 1e-9

#: Metrics a format must report to be judged at all. A format missing one has not
#: been measured on it, and unmeasured is not passing — the same rule the
#: promotion gate applies.
REQUIRED_METRICS = ("field_normalized_match", "ece_confidence", "schema_validity_rate")


class QuantValidationError(RuntimeError):
    """Raised when a format cannot be validated."""


@dataclass
class FormatResult:
    """One format, judged against fp16."""

    fmt: str
    passed: bool = False
    field_f1: float | None = None
    field_f1_drop: float | None = None
    ece: float | None = None
    ece_increase: float | None = None
    json_validity: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def servable(self) -> bool:
        return self.passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.fmt,
            "passed": self.passed,
            "field_normalized_match": self.field_f1,
            "field_f1_drop_vs_fp16": (
                None if self.field_f1_drop is None else round(self.field_f1_drop, 5)
            ),
            "ece_confidence": self.ece,
            "ece_increase_vs_fp16": (
                None if self.ece_increase is None else round(self.ece_increase, 5)
            ),
            "schema_validity_rate": self.json_validity,
            "reasons": self.reasons,
        }

    def describe(self) -> str:
        if self.passed:
            return f"{self.fmt}: PASS (F1 drop {self.field_f1_drop:+.2%})" if self.field_f1_drop is not None \
                else f"{self.fmt}: PASS (reference)"
        return f"{self.fmt}: BLOCKED — " + "; ".join(self.reasons)


@dataclass
class ValidationReport:
    """Every format judged, and what may be served."""

    results: list[FormatResult] = field(default_factory=list)
    reference_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def servable_formats(self) -> list[str]:
        return [r.fmt for r in self.results if r.passed]

    @property
    def blocked_formats(self) -> list[str]:
        return [r.fmt for r in self.results if not r.passed]

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and not self.blocked_formats

    def manifest_entry(self) -> dict[str, Any]:
        """``quant_threshold_results`` for the RunManifest (IMPL-10 §4)."""
        return {
            "reference_format": REFERENCE_FORMAT,
            "reference_metrics": self.reference_metrics,
            "servable": self.servable_formats,
            "blocked": self.blocked_formats,
            "per_format": [r.as_dict() for r in self.results],
        }

    def report(self) -> str:
        lines = [f"quantization validation against {REFERENCE_FORMAT}:"]
        lines += [f"  {r.describe()}" for r in self.results]
        if self.blocked_formats:
            lines.append(
                f"  {len(self.blocked_formats)} format(s) may not be served. Lower-bit "
                "quantization degrades exactly what was fine-tuned in — JSON structure, precise "
                "values, calibration — so a blocked format is a measurement, not a tuning knob."
            )
        return "\n".join(lines)


def validate_format(fmt: str, metrics: dict[str, Any], reference: dict[str, Any]) -> FormatResult:
    """Judge one format's metrics against the bf16 reference.

    Margins are **absolute percentage points** (arch v2.1 §13b). v1 used a
    relative drop, which is a different amount of damage at every accuracy level:
    2% of 0.95 is 1.9pp and 2% of 0.70 is 1.4pp, so the rule got stricter as the
    model got worse. The same number should mean the same thing whatever the
    baseline.
    """
    threshold = threshold_for(fmt)
    result = FormatResult(fmt=fmt.strip().lower())

    missing = [m for m in REQUIRED_METRICS if not isinstance(metrics.get(m), (int, float))]
    if missing:
        result.reasons.append(
            f"not measured on {missing}. A format that was not measured has not passed — "
            "treating absence as success is how an unvalidated format reaches serving."
        )
        return result

    result.field_f1 = float(metrics["field_normalized_match"])
    result.ece = float(metrics["ece_confidence"])
    result.json_validity = float(metrics["schema_validity_rate"])

    if is_reference(fmt):
        # bf16 defines the baseline. It is still held to its own validity floor,
        # because a reference that cannot emit valid JSON makes every comparison
        # against it meaningless.
        result.passed = result.json_validity >= threshold.min_schema_validity - EPSILON
        if not result.passed:
            result.reasons.append(
                f"schema validity {result.json_validity:.3f} < "
                f"{threshold.min_schema_validity:.3f} — the reference itself is not usable, so "
                "no format can be validated against it"
            )
        return result

    ref_f1 = reference.get("field_normalized_match")
    ref_ece = reference.get("ece_confidence")
    if not isinstance(ref_f1, (int, float)) or not isinstance(ref_ece, (int, float)):
        raise QuantValidationError(
            f"cannot judge {fmt}: the {REFERENCE_FORMAT} reference has no "
            "field_normalized_match/ece_confidence. Every threshold is a margin against it, so "
            "without the reference there is nothing to measure a margin from."
        )

    # Absolute, in percentage points.
    result.field_f1_drop = float(ref_f1) - result.field_f1
    result.ece_increase = result.ece - float(ref_ece)

    # Normalized match is the forgiving comparison — the §13b "Match — names,
    # addresses" row — so it takes the fuzzy allowance. Holding it to the
    # exact-match allowance rejected formats at half the drop §13b permits, while
    # max_fuzzy_match_drop_pp was read by nothing. The unforgiving class is held
    # to its own margin through field_exact_match below.
    allowance = threshold.max_fuzzy_match_drop_pp / 100.0
    if result.field_f1_drop > allowance + EPSILON:
        result.reasons.append(
            f"normalized field match dropped {result.field_f1_drop * 100:.2f}pp against a "
            f"{threshold.max_fuzzy_match_drop_pp:.1f}pp allowance"
        )

    # Each field class carries its own margin, because a wrong policy number is a
    # wrong extraction while a slightly-off entity name is usually still
    # matchable. Folding them into one allowance would let the unforgiving class
    # absorb the forgiving one's slack.
    for metric_name, allowed_pp, label in (
        ("field_exact_match", threshold.max_exact_match_drop_pp, "exact match"),
        ("list_field_recall", threshold.max_row_recall_drop_pp, "row recall"),
    ):
        candidate, baseline = metrics.get(metric_name), reference.get(metric_name)
        if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float)):
            drop = float(baseline) - float(candidate)
            if drop > allowed_pp / 100.0 + EPSILON:
                result.reasons.append(
                    f"{label} dropped {drop * 100:.2f}pp against a {allowed_pp:.1f}pp allowance"
                )

    rise = _increase(metrics, reference, "confusable_misattribution_rate")
    if rise is not None and rise > threshold.max_misattribution_increase_pp / 100.0 + EPSILON:
        result.reasons.append(
            f"confusable misattribution rose {rise * 100:.2f}pp against a "
            f"{threshold.max_misattribution_increase_pp:.1f}pp allowance — quantization "
            "collapsing two entities is exactly the failure canonical mapping exists to prevent"
        )

    if result.ece_increase > threshold.max_ece_increase + EPSILON:
        result.reasons.append(
            f"ECE rose {result.ece_increase:+.4f} against a {threshold.max_ece_increase:.4f} "
            "allowance — a miscalibrated model routes the wrong documents to review"
        )
    if result.json_validity < threshold.min_schema_validity - EPSILON:
        result.reasons.append(
            f"schema validity {result.json_validity:.3f} < "
            f"{threshold.min_schema_validity:.3f} — structural discipline is one of the first "
            "things quantization costs, and structured decoding is supposed to guarantee it"
        )

    result.passed = not result.reasons
    return result


def _increase(
    metrics: dict[str, Any], reference: dict[str, Any], name: str
) -> float | None:
    """How much a lower-is-better metric rose, or ``None`` when unmeasured."""
    candidate, baseline = metrics.get(name), reference.get(name)
    if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float)):
        return float(candidate) - float(baseline)
    return None


def validate_quant(
    metrics_by_format: dict[str, dict[str, Any]],
    *,
    serving_formats: list[str] | None = None,
) -> ValidationReport:
    """Judge every produced format, and refuse the ones over threshold.

    Args:
        metrics_by_format: format -> metrics from the IMPL-12 extraction routine
            over the frozen golden eval set. Must include ``bf16``.
        serving_formats: the formats actually intended for serving. A format
            produced but not served still gets a result, because knowing which
            formats *would* pass is what makes the trade-off a decision.

    Returns:
        A :class:`ValidationReport`. **There is no parameter that forces a pass.**
    """
    if REFERENCE_FORMAT not in metrics_by_format:
        raise QuantValidationError(
            f"no {REFERENCE_FORMAT} metrics supplied. Every threshold is a degradation relative "
            f"to {REFERENCE_FORMAT}, so validating without it would be comparing each format "
            "to nothing."
        )

    reference = metrics_by_format[REFERENCE_FORMAT]
    report = ValidationReport(reference_metrics=dict(reference))

    # Canonical order, so two runs of the same formats report identically.
    for fmt in sorted(metrics_by_format, key=lambda f: (not is_reference(f), f)):
        report.results.append(validate_format(fmt, metrics_by_format[fmt], reference))

    intended = [f.strip().lower() for f in (serving_formats or [DEFAULT_SERVING_FORMAT])]
    blocked_and_intended = [f for f in intended if f in report.blocked_formats]
    if blocked_and_intended:
        log.error(
            "%s were intended for serving and did not pass: %s",
            blocked_and_intended, report.report(),
        )

    log.info("%s", report.report())
    return report


def assert_servable(report: ValidationReport, serving_formats: list[str]) -> None:
    """Refuse to promote a format that did not pass.

    Called by ``package`` between quantize and push (IMPL-13 §4). No override
    flag, deliberately — every other guarantee in the pipeline becomes advisory
    the moment one exists.
    """
    wanted = [f.strip().lower() for f in serving_formats]

    # A format that appears in neither servable nor blocked was never judged at
    # all — `validate_quant` only produces a result for formats it was given
    # metrics for. Checking blocked_formats alone let an unscored GGUF sail
    # through to push with zero measurements, which is precisely the
    # "unmeasured is not passing" rule this module claims to enforce.
    judged = {result.fmt for result in report.results}
    unmeasured = [f for f in wanted if f not in judged]
    if unmeasured:
        raise QuantValidationError(
            f"{unmeasured} are intended for serving but were never measured — no metrics were "
            f"supplied for them, so `validate_quant` produced no verdict. A format nobody scored "
            "has not passed; score it against the frozen golden eval set (IMPL-12) and re-run, or "
            "drop it from --formats. There is no override (IMPL-10)."
        )

    blocked = [f for f in wanted if f in report.blocked_formats]
    if blocked:
        raise QuantValidationError(
            f"{blocked} did not pass quantization validation and will not be promoted:\n"
            + report.report()
            + "\nServe a format that passed, or re-quantize — there is no override (IMPL-10)."
        )
