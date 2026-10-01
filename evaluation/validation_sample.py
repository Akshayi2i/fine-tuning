"""The fixed validation sample: what in-training checks and checkpoint selection read.

The full validation split is ~3,300 rows on the delivered corpus. Measured on the
smoke run, one pass over it costs about an hour of evaluation loss and ~1.5 h of
generation per checkpoint, and at an evaluation every 50 steps the run spent
50-60 h validating - longer than training. Neither use needs every row: the
in-training checks only show a trend, and checkpoint selection only ranks a few
candidates against each other, on the same rows.

The sample is:

- **stratified** by line of business and reading mode, allocated in proportion
  with at least one row per group, so no line or regime drops out;
- **deterministic**: chosen by a stable hash of each row's identity, so the
  training checks and checkpoint selection - in different processes, on rows in
  different formats - pick the same rows, and a rerun picks them again;
- the whole split when it is no larger than the sample.

What ships is still gated on the full frozen test set; and a near-tie between
the top two candidates is broken on the full validation split (checkpoint_eval).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

#: Rows in the sample, unless configured otherwise.
SAMPLE_ROWS = 400


def _stratum(row: dict[str, Any]) -> tuple[str, str]:
    lob = row.get("lob")
    lob = ",".join(sorted(map(str, lob))) if isinstance(lob, list) else str(lob or "unknown")
    return lob, str(row.get("modality_mode") or "unknown")


def _identity(row: dict[str, Any]) -> str:
    """What makes a validation row itself, independent of its file position."""
    key = [row.get("source_id"), row.get("sections"), row.get("modality_mode"),
           row.get("window_pages"), row.get("doc_type")]
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def validation_sample(rows: Sequence[dict[str, Any]], size: int = SAMPLE_ROWS) -> list[dict[str, Any]]:
    """The sample of ``rows``, in their original order."""
    rows = list(rows)
    if size <= 0 or len(rows) <= size:
        return rows

    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(_stratum(row), []).append(index)

    # Proportional, at least one per group, largest remainders take what is left.
    exact = {g: len(members) * size / len(rows) for g, members in groups.items()}
    quota = {g: max(1, int(share)) for g, share in exact.items()}
    spare = size - sum(quota.values())
    for group in sorted(exact, key=lambda g: (-(exact[g] - int(exact[g])), g)):
        if spare <= 0:
            break
        if quota[group] < len(groups[group]):
            quota[group] += 1
            spare -= 1

    chosen: list[int] = []
    for group, members in groups.items():
        ranked = sorted(members, key=lambda i: _identity(rows[i]))
        chosen.extend(ranked[: min(quota[group], len(members))])
    return [rows[i] for i in sorted(chosen)]
