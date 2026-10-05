"""Acceptable degradation per serving format (arch v2.1 §13b, IMPL-10 §3).

"Re-validate before promotion" needs numbers attached, or it degrades into a
judgement call taken under release pressure. These are those numbers.

**Absolute margins in percentage points, not relative drops.** v1 allowed a 2%
*relative* field-F1 drop, which is a different amount of damage at every accuracy
level: 2% of 0.95 is 1.9pp, 2% of 0.70 is 1.4pp — so the rule got stricter as the
model got worse, which is backwards. v2.1 §13b states them absolutely, and the
same number means the same thing whatever the baseline.

**Per field class, not one number.** A 1pp drop on identifiers, money and dates
is a different event from 1pp on names and addresses. A wrong policy number is a
wrong extraction; a slightly-off entity name is usually still matchable. Folding
them into one allowance would let the unforgiving class absorb the forgiving
one's slack.

**The reference is bf16**, because that is what the merged model *is* and what
every quantized format is produced from. v1 measured against fp16, a GGUF format
— so the reference itself was an artifact the serving endpoint could not load.

Kept as data in one module, not scattered across the code that reads it: the
table is explicitly provisional, to be revised once absolute accuracy is known,
and revising it must be one edit rather than a search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The format every other one is measured against. Not itself validated: it is
#: the reference, so a threshold on it would be a threshold against itself.
REFERENCE_FORMAT = "bf16"


@dataclass(frozen=True)
class Threshold:
    """What one serving format may lose against bf16, in absolute terms.

    Every field is a **percentage-point** allowance except ``max_ece_increase``,
    which is an absolute ECE difference, and ``min_schema_validity``, which is a
    floor rather than a margin — structured decoding guarantees it, so anything
    below 100% means the guarantee is not working.
    """

    fmt: str
    max_exact_match_drop_pp: float       # identifiers, money, dates
    max_fuzzy_match_drop_pp: float       # names, addresses
    max_row_recall_drop_pp: float
    max_misattribution_increase_pp: float
    max_ece_increase: float
    min_schema_validity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.fmt,
            "max_exact_match_drop_pp": self.max_exact_match_drop_pp,
            "max_fuzzy_match_drop_pp": self.max_fuzzy_match_drop_pp,
            "max_row_recall_drop_pp": self.max_row_recall_drop_pp,
            "max_misattribution_increase_pp": self.max_misattribution_increase_pp,
            "max_ece_increase": self.max_ece_increase,
            "min_schema_validity": self.min_schema_validity,
        }


#: arch v2.1 §13b, in percentage points against the bf16 reference.
THRESHOLDS: dict[str, Threshold] = {
    "bf16": Threshold("bf16", 0.0, 0.0, 0.0, 0.0, 0.00, 1.0),
    "fp8": Threshold(
        "fp8",
        max_exact_match_drop_pp=0.5,
        max_fuzzy_match_drop_pp=1.0,
        max_row_recall_drop_pp=0.5,
        max_misattribution_increase_pp=0.5,
        max_ece_increase=0.01,
        min_schema_validity=1.0,
    ),
    "awq_int4": Threshold(
        "awq_int4",
        max_exact_match_drop_pp=1.0,
        max_fuzzy_match_drop_pp=2.0,
        max_row_recall_drop_pp=1.0,
        max_misattribution_increase_pp=1.0,
        max_ece_increase=0.02,
        min_schema_validity=1.0,
    ),
}

#: A GGUF edge build is validated IN llama.cpp, not by the serving gate — the
#: serving endpoint never loads one. When it is validated, it uses the AWQ INT4
#: column, which is the closest comparable compression level.
GGUF_THRESHOLD_COLUMN = "awq_int4"

#: What is actually served. bf16 until Phase 0 spike item 9 verifies FP8 loads
#: and runs in vLLM; FP8 from the cycle after that.
DEFAULT_SERVING_FORMAT = "bf16"


class ThresholdError(KeyError):
    """Raised for a format with no defined threshold."""


def threshold_for(fmt: str) -> Threshold:
    """The threshold for one serving format.

    Raises rather than defaulting: an unknown format with a permissive fallback
    would pass validation by not being listed, which is the opposite of what a
    threshold table is for.
    """
    name = fmt.strip().lower()
    try:
        return THRESHOLDS[name]
    except KeyError:
        from postprocessing.quantize import GGUF_FORMATS

        if name in GGUF_FORMATS:
            raise ThresholdError(
                f"{fmt!r} is a GGUF format, validated in llama.cpp against the "
                f"{GGUF_THRESHOLD_COLUMN!r} column — not by the serving gate, because the "
                "serving endpoint never loads one (arch v2.1 §13a)."
            ) from None
        raise ThresholdError(
            f"no quantization threshold defined for {fmt!r}; known serving formats are "
            f"{sorted(THRESHOLDS)}. A format with no threshold cannot be validated, and "
            "validating it against a default would let an unlisted format pass by omission."
        ) from None


def is_reference(fmt: str) -> bool:
    return fmt.strip().lower() == REFERENCE_FORMAT


def as_table() -> list[dict[str, Any]]:
    """The whole table, for a manifest or a report."""
    return [t.as_dict() for t in THRESHOLDS.values()]
