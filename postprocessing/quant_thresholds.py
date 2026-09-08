"""Acceptable degradation per quantized format (SPEC_10 §3, arch §13b).

"Re-validate before promotion" needs numbers attached, or it degrades into a
judgement call taken under release pressure. These are those numbers.

Kept as **data in one module**, not as constants scattered across the code that
reads them, because the table is explicitly provisional: these are pilot-cycle
targets to be revised once absolute F1 values are known. A 2% relative drop means
something very different at F1 0.95 than at 0.70, and revising it must be one
edit rather than a search.

The asymmetry across formats is deliberate. Lower-bit quantization degrades
exactly the behaviours that were fine-tuned in — strict JSON structure, precise
dates and policy numbers, and confidence calibration — so the JSON-validity floor
loosens far more slowly than the accuracy allowance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The format every other one is measured against. Not itself validated: it is
#: the reference, so a threshold on it would be a threshold against itself.
REFERENCE_FORMAT = "fp16"


@dataclass(frozen=True)
class Threshold:
    """What one format is allowed to lose relative to fp16."""

    fmt: str
    max_field_f1_drop: float       # relative, e.g. 0.02 == 2% of the fp16 score
    max_ece_increase: float        # absolute
    min_json_validity: float       # absolute floor

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.fmt,
            "max_field_f1_drop": self.max_field_f1_drop,
            "max_ece_increase": self.max_ece_increase,
            "min_json_validity": self.min_json_validity,
        }


THRESHOLDS: dict[str, Threshold] = {
    "fp16":   Threshold("fp16",   0.000, 0.000, 1.000),
    "bf16":   Threshold("bf16",   0.005, 0.005, 1.000),
    "q8_0":   Threshold("q8_0",   0.010, 0.010, 1.000),
    "q6_k":   Threshold("q6_k",   0.015, 0.015, 0.995),
    "q5_k_m": Threshold("q5_k_m", 0.020, 0.020, 0.995),
    "q4_k_m": Threshold("q4_k_m", 0.040, 0.030, 0.990),
}

#: The default serving target. Q4_K_M exists for VRAM-constrained deployments and
#: is used only when it passes — the trade is a per-deployment decision, which is
#: why every format stays available rather than one being hardcoded.
DEFAULT_SERVING_FORMAT = "q5_k_m"


class ThresholdError(KeyError):
    """Raised for a format with no defined threshold."""


def threshold_for(fmt: str) -> Threshold:
    """The threshold for one format.

    Raises rather than defaulting: an unknown format with a permissive fallback
    would pass validation by not being listed, which is the opposite of what a
    threshold table is for.
    """
    try:
        return THRESHOLDS[fmt.strip().lower()]
    except KeyError:
        raise ThresholdError(
            f"no quantization threshold defined for {fmt!r}; known formats are "
            f"{sorted(THRESHOLDS)}. A format with no threshold cannot be validated, and "
            "validating it against a default would let an unlisted format pass by omission."
        ) from None


def is_reference(fmt: str) -> bool:
    return fmt.strip().lower() == REFERENCE_FORMAT


def as_table() -> list[dict[str, Any]]:
    """The whole table, for a manifest or a report."""
    return [t.as_dict() for t in THRESHOLDS.values()]
