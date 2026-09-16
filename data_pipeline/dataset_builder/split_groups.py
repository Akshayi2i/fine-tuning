"""Train/val/test splitting at **group** level, before any expansion (arch v2.1 §8.2).

This is the single most consequential ordering rule in the corpus pipeline, and
v2.1 moves the unit of splitting from the document to the family.

**Why not source documents.** v1 split at ``source_id``, which stopped a
document's own modality variants spanning the split. It did nothing about
families: the same carrier's template, the same account renewed yearly, one
agency's four hundred near-identical certificates. Put one member in ``train``
and another in ``test`` and the eval number measures template memorisation. The
failure is invisible — nothing crashes, the metrics simply come back better than
the model deserves.

So: assign **groups** to splits first, then expand every document, task, window
and modality mode of a group inside whichever split it landed in.

Three things this adds beyond the group unit:

* **A held-out-carrier slice.** At least two carriers per doc type go entirely to
  test once six exist, because "unseen template" is a different question from
  "unseen document" and only a held-out carrier answers it.
* **Validation halves.** Validation is split by group into a calibration half
  (fits calibrators, §5.3) and a threshold half (sets review thresholds, §5.4).
  Fitting both on the same data makes every threshold optimistic — the
  calibrator has already seen the errors the threshold is meant to price.
* **Synthetic documents are train-only.** A synthetic ACORD has perfect labels
  because it was generated from them; scoring against it measures the generator.

**On assignment stability.** Groups are assigned by hash *threshold* rather than
slice position, so at a fixed ratio the corpus can grow without reassigning
anything. Two things do move groups, and both are deliberate: crossing a volume
band, because the ratios themselves change; and the empty-split repair, which
fires only at very small N. Neither is a comparability problem — cross-version
comparability comes from the frozen golden eval set (§15.4), not from the corpus
test split.
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

ValHalf = Literal["calibration", "threshold"]

#: Carriers to hold out entirely per doc type, once the corpus has enough of them
#: that removing two still leaves a usable training set.
HELD_OUT_CARRIERS_PER_TYPE = 2
MIN_CARRIERS_FOR_HELD_OUT = 6


class SplitError(RuntimeError):
    """Raised when a split assignment would leak or is otherwise unusable."""


@dataclass
class GroupSplitAssignment:
    """A deterministic ``group_id -> split`` mapping, and how it was made."""

    assignment: dict[str, Split] = field(default_factory=dict)

    #: Which half of validation a group is in. Empty for non-validation groups.
    val_half: dict[str, ValHalf] = field(default_factory=dict)

    #: Carriers placed entirely in test, per doc type.
    held_out_carriers: dict[str, list[str]] = field(default_factory=dict)

    seed: int = 42
    ratios_by_doc_type: dict[str, dict[str, float]] = field(default_factory=dict)
    counts_by_doc_type: dict[str, dict[str, int]] = field(default_factory=dict)

    def split_of(self, group_id: str) -> Split:
        try:
            return self.assignment[group_id]
        except KeyError:
            raise SplitError(
                f"group {group_id} has no split assignment. Every group must be assigned before "
                "expansion, or its documents could land in different splits."
            ) from None

    def half_of(self, group_id: str) -> ValHalf | None:
        return self.val_half.get(group_id)

    def groups_in(self, split: Split) -> list[str]:
        return sorted(g for g, s in self.assignment.items() if s == split)

    def as_dict(self) -> dict[str, Any]:
        return {
            "assignment": dict(sorted(self.assignment.items())),
            "val_half": dict(sorted(self.val_half.items())),
            "held_out_carriers": {k: sorted(v) for k, v in sorted(self.held_out_carriers.items())},
            "seed": self.seed,
            "ratios_by_doc_type": self.ratios_by_doc_type,
            "counts_by_doc_type": self.counts_by_doc_type,
        }


@dataclass
class GroupRecord:
    """One family, as the splitter sees it."""

    group_id: str
    doc_type: str
    source_ids: list[str]
    carrier: str | None = None

    #: Generated documents never leave train: they have perfect labels because
    #: they were produced from them, so scoring against one measures the
    #: generator, not the model (arch v2.1 §4d).
    synthetic: bool = False

    @property
    def size(self) -> int:
        return len(self.source_ids)


def _stable_hash(group_id: str, seed: int, salt: str = "") -> float:
    """A deterministic value in [0, 1) for a group.

    Hash-based rather than shuffle-based so adding documents later never
    reshuffles the groups already assigned.
    """
    digest = hashlib.sha256(f"{seed}:{salt}:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _pick_held_out_carriers(groups: list[GroupRecord], seed: int) -> list[str]:
    """Choose carriers to place entirely in test.

    Chosen by hash so the choice is reproducible, but **weighted away from the
    largest carriers**: holding out the carrier that contributes 40% of the
    training data answers the generalisation question by destroying the corpus.
    Held-out carriers are drawn from the smaller half.
    """
    by_carrier: dict[str, int] = defaultdict(int)
    for group in groups:
        if group.carrier and not group.synthetic:
            by_carrier[group.carrier] += group.size
    if len(by_carrier) < MIN_CARRIERS_FOR_HELD_OUT:
        return []

    ordered = sorted(by_carrier.items(), key=lambda kv: (kv[1], kv[0]))
    smaller_half = [c for c, _ in ordered[: max(2, len(ordered) // 2)]]
    return sorted(smaller_half, key=lambda c: _stable_hash(c, seed, "carrier"))[
        :HELD_OUT_CARRIERS_PER_TYPE
    ]


def assign_group_splits(
    groups_by_doc_type: dict[str, list[GroupRecord]],
    *,
    seed: int = 42,
    ratios: SplitRatio | None = None,
    hold_out_carriers: bool = True,
) -> GroupSplitAssignment:
    """Assign each group to a split, per document type.

    Splitting per type rather than globally keeps every type represented in every
    split — a global split on an unbalanced corpus can leave a type with no test
    documents at all, and its accuracy then silently unmeasured.
    """
    result = GroupSplitAssignment(seed=seed)

    for doc_type, records in sorted(groups_by_doc_type.items()):
        unique = sorted({r.group_id: r for r in records}.values(), key=lambda r: r.group_id)
        if not unique:
            continue

        # Ratios are chosen on DOCUMENT count, not group count: the volume bands
        # in §8.2 describe how much data exists, and one group of forty
        # certificates is still forty documents of training signal.
        document_count = sum(r.size for r in unique)
        ratio = ratios or split_ratio_for(document_count)
        result.ratios_by_doc_type[doc_type] = {
            "train": ratio.train, "val": ratio.val, "test": ratio.test,
        }

        held_out = _pick_held_out_carriers(unique, seed) if hold_out_carriers else []
        if held_out:
            result.held_out_carriers[doc_type] = held_out

        train_edge = ratio.train
        val_edge = ratio.train + ratio.val
        buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}

        for record in unique:
            if record.synthetic:
                buckets["train"].append(record.group_id)
                continue
            if record.carrier and record.carrier in held_out:
                buckets["test"].append(record.group_id)
                continue
            position = _stable_hash(record.group_id, seed)
            split = "train" if position < train_edge else "val" if position < val_edge else "test"
            buckets[split].append(record.group_id)

        _repair_empty_splits(buckets, unique, doc_type, seed)

        for split, members in buckets.items():
            for group_id in members:
                result.assignment[group_id] = split  # type: ignore[assignment]

        _assign_validation_halves(result, buckets["val"], seed)

        result.counts_by_doc_type[doc_type] = {s: len(m) for s, m in buckets.items()}

        if document_count < 10:
            log.warning(
                "%s has only %d documents in %d group(s) (%s). At this volume eval metrics are "
                "directional, not conclusive (arch §8.2).",
                doc_type, document_count, len(unique), result.counts_by_doc_type[doc_type],
            )

    return result


def _repair_empty_splits(
    buckets: dict[str, list[str]], groups: list[GroupRecord], doc_type: str, seed: int
) -> None:
    """Move one group into a starved split rather than ship an unusable corpus.

    An empty test split cannot be evaluated at all and an empty val split
    disables both early stopping and calibration. Fires only at very small N, and
    it is the one case where assignment is not stable under corpus growth.

    Synthetic groups are never moved out of train — that rule outranks this
    repair, because a test split made of generated documents measures the
    generator and reports it as model accuracy.
    """
    if len(groups) < 3:
        return
    movable = {g.group_id for g in groups if not g.synthetic}
    for starved in ("test", "val"):
        if buckets[starved]:
            continue
        donor = max(buckets, key=lambda s: len([g for g in buckets[s] if g in movable]))
        candidates = [g for g in buckets[donor] if g in movable]
        if len(candidates) < 2:
            continue
        moved = sorted(candidates, key=lambda g: _stable_hash(g, seed))[-1]
        buckets[donor].remove(moved)
        buckets[starved].append(moved)
        log.warning(
            "%s: group %r moved to %s so the split is not empty — an empty test split cannot be "
            "evaluated. This is the one case where assignment is not stable under growth.",
            doc_type, moved, starved,
        )


def _assign_validation_halves(
    result: GroupSplitAssignment, val_groups: list[str], seed: int
) -> None:
    """Split validation by group into a calibration half and a threshold half.

    Fitting the calibrator and setting the review threshold on the same documents
    makes the threshold optimistic: the calibrator has already seen the errors the
    threshold is meant to price, so the measured error rate among auto-accepted
    fields is lower than production will deliver (arch v2.1 §5.3-5.4).

    Split by GROUP, for the same reason the outer split is: two renewals of one
    account across the halves is the same leak at a smaller scale.
    """
    for group_id in sorted(val_groups):
        half: ValHalf = (
            "calibration" if _stable_hash(group_id, seed, "valhalf") < 0.5 else "threshold"
        )
        result.val_half[group_id] = half


def assert_no_leakage(assignment: GroupSplitAssignment, rows: list[dict[str, Any]]) -> None:
    """Assert no group appears in more than one split.

    Run over the **expanded** rows, because that is where the failure would
    actually show up: the assignment itself is a dict and cannot contain a
    duplicate, but expansion into tasks, windows and modes is what could place
    them inconsistently.
    """
    splits_seen: dict[str, set[str]] = defaultdict(set)
    sources_seen: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        group_id, split = row.get("group_id"), row.get("split")
        source_id = row.get("source_id")
        if not group_id or not split:
            raise SplitError(
                f"row is missing group_id or split: source_id={source_id!r}. Every row carries "
                "its group, because the group is the unit that must not span the split."
            )
        splits_seen[group_id].add(split)
        if source_id:
            sources_seen[source_id].add(split)

    leaked = {g: sorted(s) for g, s in splits_seen.items() if len(s) > 1}
    if leaked:
        raise SplitError(
            f"LEAKAGE: {len(leaked)} group(s) appear in more than one split — "
            f"{dict(list(leaked.items())[:5])}. A family spanning the split means the model is "
            "evaluated on a template it trained on, which inflates every metric without "
            "reflecting generalisation (arch v2.1 §8.2)."
        )

    document_leaks = {s: sorted(v) for s, v in sources_seen.items() if len(v) > 1}
    if document_leaks:
        raise SplitError(
            f"LEAKAGE: {len(document_leaks)} source_id(s) span splits — "
            f"{dict(list(document_leaks.items())[:5])}. Split before task and modality "
            "expansion, not after."
        )

    for group_id, seen in splits_seen.items():
        expected = assignment.assignment.get(group_id)
        if expected and seen != {expected}:
            raise SplitError(
                f"group {group_id} was assigned to {expected!r} but its rows are in {sorted(seen)}"
            )


def assert_synthetic_is_train_only(rows: list[dict[str, Any]]) -> None:
    """Synthetic documents never appear in validation or test (arch v2.1 §4d).

    Their labels are perfect because they were generated from them, so a metric
    scored against one reports how faithfully the generator rendered its own
    input — and reports it as model accuracy.
    """
    escaped = sorted({
        str(row.get("source_id"))
        for row in rows
        if row.get("synthetic") and row.get("split") != "train" and row.get("source_id")
    })
    if escaped:
        raise SplitError(
            f"{len(escaped)} synthetic document(s) reached a non-train split: {escaped[:5]}. "
            "Scoring against generated labels measures the generator, not the model."
        )


def assert_single_tenant(rows: list[dict[str, Any]]) -> None:
    """Assert a corpus file carries exactly one tenant.

    Corpus composition **is** training data, so a file mixing tenants means a
    model trained on Broker A's data contains Broker B's (arch §8b).
    """
    tenants = {row.get("tenant_id") for row in rows}
    if len(tenants) > 1:
        raise SplitError(
            f"corpus rows span multiple tenants: {sorted(t for t in tenants if t)}. "
            "Cross-tenant mixing is prohibited — a model trained on one broker's data must "
            "never contain another's (arch §8b)."
        )
