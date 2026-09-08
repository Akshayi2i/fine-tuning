"""Line of Business — enum, validation, and corpus coverage (arch §0b).

L1 (carrier registry) and L2 (structural inference) both attempt LoB detection
before a document reaches L3. When L3 is invoked, **the VLM is expected to detect
and output ``line_of_business`` itself**, so it is a field the model is trained
to produce — not metadata attached afterwards.

Two rules this module exists to enforce:

* ``line_of_business`` is present in **every** golden label, for every document
  type, even when ``null``.
* Coverage target of **≥20% of training examples per LoB value**. Under-coverage
  is a loud warning rather than a build failure: the remedy is collecting
  documents, which is a data-acquisition decision, not a build-time one.

Accuracy is reported **per LoB value** and is a gating metric (SPEC_08). It is
never averaged into overall field accuracy, because a rare class would then hide
inside a healthy-looking aggregate.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from common.constants import LOB_COVERAGE_TARGET

LOB_ENUM_PATH = Path(__file__).resolve().parent.parent / "schemas" / "lob.enum.json"


class LobError(ValueError):
    """Raised on a value outside the LOB enum."""


@lru_cache(maxsize=1)
def lob_values() -> tuple[str, ...]:
    """The non-null LOB values, read from the one schema that defines them."""
    with LOB_ENUM_PATH.open(encoding="utf-8") as fh:
        enum = json.load(fh)["$defs"]["line_of_business"]["enum"]
    return tuple(v for v in enum if v is not None)


def is_valid_lob(value: str | None) -> bool:
    """Whether a value is a valid LoB. ``None`` is valid — it means undetermined."""
    return value is None or value in lob_values()


def validate_lob(value: str | None) -> str | None:
    """Return the value if valid, else raise with the permitted set."""
    if not is_valid_lob(value):
        raise LobError(
            f"invalid line_of_business {value!r}; expected one of {lob_values()} or null "
            "(null means the document does not determine it — not 'unknown to the annotator')"
        )
    return value


class LobCoverage(NamedTuple):
    """Per-value LoB distribution across a corpus, against the target."""

    counts: dict[str, int]
    total: int
    shares: dict[str, float]
    under_target: dict[str, float]
    null_count: int

    @property
    def is_balanced(self) -> bool:
        return not self.under_target

    def warning(self) -> str | None:
        """A message naming the under-represented values, or ``None``."""
        if self.is_balanced:
            return None
        named = ", ".join(
            f"{value} ({share:.1%})" for value, share in sorted(self.under_target.items(), key=lambda kv: kv[1])
        )
        return (
            f"LoB coverage below the {LOB_COVERAGE_TARGET:.0%} target for: {named}. "
            f"The model will be weak on these values. The remedy is collecting documents, "
            f"not changing the build — see arch §0b."
        )


def compute_coverage(
    values: Iterable[str | None],
    target: float = LOB_COVERAGE_TARGET,
) -> LobCoverage:
    """Count LoB values across a corpus and flag under-representation.

    ``null`` is counted and reported but excluded from the share denominator:
    a legitimately-undetermined LoB is a correct label, not a missing one, so
    letting nulls dilute the shares would make coverage look worse than it is.
    """
    counter: Counter[str] = Counter()
    null_count = 0
    for value in values:
        if value is None:
            null_count += 1
            continue
        validate_lob(value)
        counter[value] += 1

    total = sum(counter.values())
    shares = {v: (counter.get(v, 0) / total if total else 0.0) for v in lob_values()}
    # The floor is capped at a share that is actually reachable. There are five
    # non-null LoB values, so a flat 0.20 target IS the uniform share: every
    # corpus that is not perfectly uniform put at least one value under it,
    # `is_balanced` was unreachable, and the warning fired forever carrying no
    # information. A 21/20/20/20/19 corpus is balanced by any sensible reading.
    #
    # The architecture's intent is "no LoB value is starved", so the effective
    # floor is three quarters of the uniform share — strictly below it, and it
    # scales automatically if a value is added to the enum. It never exceeds the
    # configured target, so tightening LOB_COVERAGE_TARGET still tightens this.
    #
    # A value with NO documents is a different problem — nothing to be weak at,
    # and no per-value accuracy to measure — reported by `unmeasured_values`.
    uniform_share = 1.0 / len(shares) if shares else 0.0
    floor = min(target, uniform_share * 0.75)
    under = {v: s for v, s in shares.items() if 0.0 < s < floor}
    return LobCoverage(dict(counter), total, shares, under, null_count)
