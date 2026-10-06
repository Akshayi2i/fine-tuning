"""Auto-accept error: of the values a user got WITHOUT a review flag, the share
that were wrong (arch v2.1 §15.2).

This is the number the review thresholds promise. The calibrate stage measures
it on the validation half the thresholds were chosen on, where it passes almost
by construction; here it is measured on the frozen golden set, through serving,
with the release's calibrators and thresholds - the documents the promise has
never seen.

The source of truth is the golden label. A value counts once, where it was
delivered:

* **Accepted** - its envelope says ``flagged: false``. A list of envelopes (a
  set field such as ``line_of_business``) is one value, accepted only when no
  element is flagged.
* **Wrong** - it does not match the label's value at the same place
  (``values_match``, the comparison every accuracy metric uses), or the label
  holds nothing there: an invented value accepted unreviewed is the worst case.
* **Table rows** are matched to the label's rows by their identifiers (VIN, form
  number, claim number ... the same matching as list recall), never by
  position: a reordered row is not wrong, and a row the label does not hold is
  invented in every value. A common-model document's rows pair as its field
  match pairs them (``field_accuracy._pair_rows``): a coverage with a wrong code
  or a lost link is still the coverage it read.

Not counted: values the model left out. An omission reaches nobody as an
accepted value; recall and ``false_null_rate`` measure it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AutoAcceptTally:
    """Accepted values and wrong accepted values, summed over documents."""

    flags_seen: int = 0
    accepted: int = 0
    wrong: int = 0

    def add(self, other: AutoAcceptTally) -> None:
        self.flags_seen += other.flags_seen
        self.accepted += other.accepted
        self.wrong += other.wrong

    @property
    def rate(self) -> float | None:
        """``None`` when no value carried a flag (not measured, e.g. a flat
        extraction); 0.0 when flags were seen and nothing was accepted - no
        unreviewed value can be a wrong one."""
        if not self.flags_seen:
            return None
        return self.wrong / self.accepted if self.accepted else 0.0


def score_auto_accept(expected: Any, got: Any, *, common_model: bool = False) -> AutoAcceptTally:
    """Tally one document: ``expected`` the golden label (either form),
    ``got`` the served extraction with its envelopes. ``common_model``: the
    document is on a common-model line, and its rows pair as field match pairs
    them."""
    from common.canonical import values_view

    tally = AutoAcceptTally()
    _walk(values_view(expected), got, "", tally, common_model)
    return tally


def _walk(expected: Any, got: Any, path: str, tally: AutoAcceptTally, common_model: bool) -> None:
    from common.canonical import is_field_value

    if is_field_value(got):
        _count(expected, got, [got], path, tally)
    elif isinstance(got, list) and got and all(is_field_value(item) for item in got):
        _count(expected, got, got, path, tally)
    elif isinstance(got, dict):
        sub = expected if isinstance(expected, dict) else {}
        for key, value in got.items():
            _walk(sub.get(key), value, f"{path}.{key}" if path else key, tally, common_model)
    elif isinstance(got, list):
        for index, (row, match) in enumerate(_match_rows(expected, got, common_model=common_model)):
            _walk(match, row, f"{path}[{index}]", tally, common_model)


def _count(expected: Any, node: Any, envelopes: list[dict[str, Any]], path: str,
           tally: AutoAcceptTally) -> None:
    from common.canonical import values_view
    from common.normalize import values_match
    from evaluation.metrics.field_accuracy import _is_empty

    flags = [e.get("flagged") for e in envelopes]
    if any(not isinstance(f, bool) for f in flags):
        return  # no flag, no decision to measure
    value = values_view(node)
    if _is_empty(value):
        return  # nothing was delivered
    tally.flags_seen += 1
    if any(flags):
        return
    tally.accepted += 1
    if _is_empty(expected) or not values_match(expected, value, field_path=path):
        tally.wrong += 1


def _match_rows(expected: Any, got_rows: list[Any], *, common_model: bool = False) -> list[tuple[Any, Any]]:
    """Each served row with the label row it reads (``None`` when the label
    holds no such row), matched on identifiers as list recall matches them -
    on a common-model line as its field match pairs them."""
    from common.canonical import values_view
    from evaluation.metrics.field_accuracy import _infer_key_fields, _pair_rows, _row_key

    expected_rows = expected if isinstance(expected, list) else []
    if common_model:
        mates, _unpaired = _pair_rows(expected_rows, got_rows)
        return list(zip(got_rows, mates, strict=True))
    keys = _infer_key_fields(expected_rows) if expected_rows else []
    pool: dict[tuple, list[Any]] = {}
    for row in expected_rows:
        pool.setdefault(_row_key(row, keys), []).append(row)
    pairs = []
    for row in got_rows:
        candidates = pool.get(_row_key(values_view(row), keys)) if expected_rows else None
        pairs.append((row, candidates.pop(0) if candidates else None))
    return pairs
