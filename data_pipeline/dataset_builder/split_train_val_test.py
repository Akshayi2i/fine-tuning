"""Train/val/test splitting — at ``source_id`` level, **before** modality expansion.

This is the single most consequential ordering rule in the corpus pipeline.

Each source document expands into three JSONL rows (``ocr_plus_image``,
``noisy_ocr_image``, ``image_only``). Splitting *after* that expansion would put
the same document in both ``train`` and ``test`` wearing different input modes —
the model would be evaluated on a document it had already memorised, and every
eval number would be inflated without reflecting any real generalisation.

So: assign documents to splits first, then expand each into its three variants
inside whichever split it landed in.

Split ratios scale with how much data actually exists (arch §8). At pilot volume
a strict 70/20/10 leaves two or three test documents per type, too few to trust
any single metric — which is why pilot numbers are directional.

**On assignment stability.** Documents are assigned by hash *threshold* rather
than by slice position, so at a fixed ratio the corpus can grow without
reassigning anything. Two things do move documents, and both are deliberate:

* **crossing a volume band**, because the ratios themselves change (70/18/12 →
  75/15/15 → 80/10/10). Scaling ratios and perfect stability are in tension, and
  the architecture chooses scaling;
* **the empty-split repair**, which fires only at very small N.

Neither is a comparability problem, because cross-version comparability comes
from the **frozen golden eval set** — versioned separately and held constant
across corpus versions (arch §8) — not from the corpus test split.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from common.constants import SplitRatio, split_ratio_for

log = logging.getLogger(__name__)

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")


class SplitError(RuntimeError):
    """Raised when a split assignment would leak or is otherwise unusable."""


@dataclass
class SplitAssignment:
    """A deterministic ``source_id -> split`` mapping, and how it was made."""

    assignment: dict[str, Split] = field(default_factory=dict)
    seed: int = 42
    ratios_by_doc_type: dict[str, dict[str, float]] = field(default_factory=dict)
    counts_by_doc_type: dict[str, dict[str, int]] = field(default_factory=dict)

    def split_of(self, source_id: str) -> Split:
        try:
            return self.assignment[source_id]
        except KeyError:
            raise SplitError(
                f"{source_id} has no split assignment. Every document must be assigned before "
                "expansion, or its variants could land in different splits."
            ) from None

    def source_ids_in(self, split: Split) -> list[str]:
        return sorted(sid for sid, s in self.assignment.items() if s == split)

    def as_dict(self) -> dict[str, Any]:
        return {
            "assignment": dict(sorted(self.assignment.items())),
            "seed": self.seed,
            "ratios_by_doc_type": self.ratios_by_doc_type,
            "counts_by_doc_type": self.counts_by_doc_type,
        }


def _stable_hash(source_id: str, seed: int) -> float:
    """A deterministic value in [0, 1) for a document.

    Hash-based rather than shuffle-based so that adding documents later does not
    reshuffle the ones already assigned: a document that was in ``test`` stays in
    ``test``, which is what keeps eval numbers comparable across corpus versions.
    """
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_splits(
    source_ids_by_doc_type: dict[str, list[str]],
    *,
    seed: int = 42,
    ratios: SplitRatio | None = None,
) -> SplitAssignment:
    """Assign each document to a split, per document type.

    Splitting per type rather than globally keeps every type represented in every
    split — a global split on an unbalanced corpus can leave a type with no test
    documents at all, and its accuracy then silently unmeasured.
    """
    result = SplitAssignment(seed=seed)

    for doc_type, source_ids in sorted(source_ids_by_doc_type.items()):
        unique = sorted(set(source_ids))
        if not unique:
            continue

        ratio = ratios or split_ratio_for(len(unique))
        result.ratios_by_doc_type[doc_type] = {
            "train": ratio.train, "val": ratio.val, "test": ratio.test,
        }

        # Assign by hash THRESHOLD, not by slice position. A document's hash is
        # fixed and the thresholds are fixed, so adding documents later never
        # moves an existing one — where slicing would shift every boundary and
        # migrate documents near it between splits.
        #
        # The cost is that counts are approximate rather than exact. That is the
        # right trade: which split a document lands in must be stable, whereas a
        # corpus being 71% train instead of 70% changes nothing.
        train_edge = ratio.train
        val_edge = ratio.train + ratio.val

        buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for source_id in sorted(unique):
            position = _stable_hash(source_id, seed)
            split = "train" if position < train_edge else "val" if position < val_edge else "test"
            buckets[split].append(source_id)

        # Repair only when a split would be empty: an empty test split cannot be
        # evaluated at all, and an empty val split disables early stopping. This
        # fires only at very small N, and it is the one case where assignment is
        # not stable under growth.
        if len(unique) >= 3:
            for starved in ("test", "val"):
                if buckets[starved]:
                    continue
                donor = max(buckets, key=lambda s: len(buckets[s]))
                if len(buckets[donor]) < 2:
                    continue
                moved = sorted(buckets[donor], key=lambda sid: _stable_hash(sid, seed))[-1]
                buckets[donor].remove(moved)
                buckets[starved].append(moved)
                log.warning(
                    "%s: %r moved to %s so the split is not empty — an empty test split cannot "
                    "be evaluated. This is the one case where assignment is not stable under "
                    "corpus growth.", doc_type, moved, starved,
                )

        for split, members in buckets.items():
            for source_id in members:
                result.assignment[source_id] = split  # type: ignore[assignment]
        result.counts_by_doc_type[doc_type] = {s: len(m) for s, m in buckets.items()}

        if len(unique) < 10:
            log.warning(
                "%s has only %d documents (%s). At this volume eval metrics are directional, "
                "not conclusive — the batch's job is proving the pipeline works (arch §8).",
                doc_type, len(unique), result.counts_by_doc_type[doc_type],
            )

    return result


def assert_no_leakage(assignment: SplitAssignment, rows: list[dict[str, Any]]) -> None:
    """Assert no ``source_id`` appears in more than one split.

    Run over the **expanded** rows, because that is where the failure would
    actually show up: the assignment itself is a dict and cannot contain a
    duplicate, but expansion is what could place variants inconsistently.
    """
    splits_seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_id, split = row.get("source_id"), row.get("split")
        if not source_id or not split:
            raise SplitError(f"row is missing source_id or split: {row.get('source_id')!r}")
        splits_seen[source_id].add(split)

    leaked = {sid: sorted(s) for sid, s in splits_seen.items() if len(s) > 1}
    if leaked:
        raise SplitError(
            f"LEAKAGE: {len(leaked)} source_id(s) appear in more than one split — "
            f"{dict(list(leaked.items())[:5])}. The model would be evaluated on documents it "
            "trained on, inflating every metric. Split BEFORE modality expansion (arch §8)."
        )

    for source_id, seen in splits_seen.items():
        expected = assignment.assignment.get(source_id)
        if expected and seen != {expected}:
            raise SplitError(
                f"{source_id} was assigned to {expected!r} but its rows are in {sorted(seen)}"
            )


def assert_single_tenant(rows: list[dict[str, Any]]) -> None:
    """Assert a corpus file carries exactly one tenant.

    The one live tenancy rule: corpus composition **is** training data, so a file
    mixing tenants means a model trained on Broker A's data contains Broker B's
    (arch §8b).
    """
    tenants = {row.get("tenant_id") for row in rows}
    if len(tenants) > 1:
        raise SplitError(
            f"corpus rows span multiple tenants: {sorted(t for t in tenants if t)}. "
            "Cross-tenant mixing is prohibited — a model trained on one broker's data must "
            "never contain another's (arch §8b)."
        )
