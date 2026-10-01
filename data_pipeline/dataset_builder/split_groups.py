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

**The ratio** comes from the document type's volume (``common.constants.
SPLIT_RATIOS_BY_VOLUME``: 70/18/12 below 200 documents, 75/15/10 below 1000,
80/10/10 from 1000). It is per TYPE because the gate's metrics are per type: the
band answers "are 10% of this type's documents enough to measure it".

**Placement is per line of business** (a policy's ``lob``). Every group gets a
stable hash position; below the train edge it trains, below the val edge it is
validation, above it is test — the same rule in every line, so each line lands
near the type's ratio. Two per-line rules on top:

* a line with fewer than :data:`MIN_DOCS_TO_MEASURE_LINE` documents trains on
  everything it has. One or two documents cannot measure a line, and holding
  them out would leave the model with no example of it at all;
* a line with enough groups is repaired to have at least one val and one test
  group, so no measured line is absent from evaluation by the luck of the hash.

**On assignment stability.** Groups are assigned by hash *threshold*, not slice
position, so at a fixed ratio the corpus grows without reassigning anything.
Crossing a volume band changes the edges, and :func:`assert_bands_only_grow_train`
guarantees both edges only ever move UP (train edge 0.70 -> 0.75 -> 0.80, val
edge 0.88 -> 0.90 -> 0.90). So a band crossing only moves groups toward train —
test -> val, val -> train — and never moves a document a model trained on into
val or test. Two small-N rules can still move a group the other way, and both
say so in the log: the empty-split repair, and a line crossing
:data:`MIN_DOCS_TO_MEASURE_LINE`. Cross-version comparability comes from the
frozen golden eval set (§15.4), which no reassignment touches.
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

#: Below this many documents a line of business is not measured: all of it trains.
MIN_DOCS_TO_MEASURE_LINE = 5

#: A line needs this many groups before the repair forces a val and a test group:
#: with fewer, the repair would hand one line's whole evaluation to one family.
MIN_GROUPS_TO_REPAIR = 3

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

    #: Groups per split per line of business, per doc type — what the ratio
    #: actually produced for each line.
    counts_by_line: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    #: Lines too small to measure, trained on whole.
    train_only_lines: dict[str, list[str]] = field(default_factory=dict)

    seed: int = 42
    ratios_by_doc_type: dict[str, dict[str, float]] = field(default_factory=dict)
    counts_by_doc_type: dict[str, dict[str, int]] = field(default_factory=dict)

    #: The split came with the data (``assign_delivered_splits``) rather than
    #: being drawn here. Synthetic documents may then sit in val and test: the
    #: delivery split by source document, so a test document's layout is one the
    #: model never trained on, and the gate reports real documents separately.
    delivered: bool = False

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
            "counts_by_line": self.counts_by_line,
            "train_only_lines": {k: sorted(v) for k, v in sorted(self.train_only_lines.items())},
            "delivered": self.delivered,
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

    #: The line of business the family belongs to (:func:`line_of`). ``None`` for
    #: types without one (ACORD, Loss Runs), which are then one line per type.
    line: str | None = None

    @property
    def size(self) -> int:
        return len(self.source_ids)


def line_of(lob: str | list[str] | None) -> str | None:
    """A document's line as a stratum key. A package policy's lines, sorted and
    joined, are one stratum of their own: they are read against the fallback
    schema, not either line's."""
    if lob is None or lob == [] or lob == "":
        return None
    if isinstance(lob, str):
        return lob.strip().lower()
    return "+".join(sorted(str(x).strip().lower() for x in lob))


def assert_bands_only_grow_train() -> None:
    """Both split edges must be non-decreasing as volume grows.

    That is what makes a band crossing move groups only toward train. An edge
    that fell — a later band with a smaller train share, say — would move groups
    a previous version trained on into val or test, and score it on them.
    """
    from common.constants import SPLIT_RATIOS_BY_VOLUME

    edges = [(r.train, r.train + r.val) for _t, r in SPLIT_RATIOS_BY_VOLUME]
    for (t0, v0), (t1, v1) in zip(edges, edges[1:], strict=False):
        if t1 < t0 - 1e-9 or v1 < v0 - 1e-9:
            raise ValueError(
                f"split bands {edges} move an edge DOWN between bands; a band crossing would "
                "move trained documents into val or test."
            )


assert_bands_only_grow_train()


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
    with_test: bool = True,
) -> GroupSplitAssignment:
    """Assign each group to a split, per document type.

    Splitting per type rather than globally keeps every type represented in every
    split — a global split on an unbalanced corpus can leave a type with no test
    documents at all, and its accuracy then silently unmeasured.

    ``with_test=False`` once the golden eval set is frozen: the frozen set IS the
    test set, so new documents split into train and val only, in the ratio's own
    train:val proportion, and no carrier is held out (the frozen set already
    holds the carriers the first split held out).
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
        if not with_test:
            share = ratio.train + ratio.val
            ratio = SplitRatio(ratio.train / share, ratio.val / share, 0.0)
        result.ratios_by_doc_type[doc_type] = {
            "train": ratio.train, "val": ratio.val, "test": ratio.test,
        }

        held_out = (
            _pick_held_out_carriers(unique, seed) if hold_out_carriers and with_test else []
        )
        if held_out:
            result.held_out_carriers[doc_type] = held_out

        train_edge = ratio.train
        val_edge = ratio.train + ratio.val
        buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}

        by_line: dict[str | None, list[GroupRecord]] = defaultdict(list)
        for record in unique:
            by_line[record.line].append(record)

        for line, members in sorted(by_line.items(), key=lambda kv: kv[0] or ""):
            line_buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
            line_documents = sum(r.size for r in members)
            measured = line_documents >= MIN_DOCS_TO_MEASURE_LINE
            for record in members:
                if record.synthetic:
                    line_buckets["train"].append(record.group_id)
                    continue
                if record.carrier and record.carrier in held_out:
                    line_buckets["test"].append(record.group_id)
                    continue
                if not measured and line is not None:
                    line_buckets["train"].append(record.group_id)
                    continue
                position = _stable_hash(record.group_id, seed)
                split = "train" if position < train_edge else "val" if position < val_edge else "test"
                line_buckets[split].append(record.group_id)

            if measured or line is None:
                _repair_empty_splits(
                    line_buckets, members, f"{doc_type}/{line or '-'}", seed,
                    starved=("test", "val") if with_test else ("val",),
                )
            elif not line_buckets["test"]:
                result.train_only_lines.setdefault(doc_type, []).append(line)
                log.info(
                    "%s/%s has %d document(s), fewer than %d: all of it trains, and the line is "
                    "not measured until it has more", doc_type, line, line_documents,
                    MIN_DOCS_TO_MEASURE_LINE,
                )
            else:
                # A held-out carrier's documents stay held out even in a small
                # line: training on them would make the held-out slice a
                # carrier the model has seen. So this line is NOT train-only,
                # and recording it as one said the opposite of what happened.
                log.warning(
                    "%s/%s has %d document(s), fewer than %d, and %d group(s) belong to a "
                    "held-out carrier: those are tested, not trained. %s",
                    doc_type, line, line_documents, MIN_DOCS_TO_MEASURE_LINE,
                    len(line_buckets["test"]),
                    "The line has NO training documents - the model never sees it."
                    if not line_buckets["train"] else
                    f"{len(line_buckets['train'])} group(s) train.",
                )
            if line is not None:
                result.counts_by_line.setdefault(doc_type, {})[line] = {
                    k: len(v) for k, v in line_buckets.items()
                }
            for split_name, ids in line_buckets.items():
                buckets[split_name].extend(ids)

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


#: The split names a delivery may carry, in a document's metadata ``split``.
DELIVERED_SPLITS = ("train", "val", "test")


def assign_delivered_splits(
    groups_by_doc_type: dict[str, list[GroupRecord]],
    split_of_group: dict[str, str],
    *,
    seed: int = 42,
    with_test: bool = True,
) -> GroupSplitAssignment:
    """Use the split the data arrived with, instead of drawing one.

    For a delivery that was split upstream — a synthetic set whose generator put
    each source document and all its twins in one split. The families here are
    the delivery's own (one per source document), so the split is taken as
    given; what this still enforces is that it is usable: no family in two
    splits, a non-empty val, and a non-empty test unless the eval set is frozen.
    Validation is halved by group exactly as for a drawn split.
    """
    result = GroupSplitAssignment(seed=seed, delivered=True)
    for doc_type, records in sorted(groups_by_doc_type.items()):
        unique = sorted({r.group_id: r for r in records}.values(), key=lambda r: r.group_id)
        if not unique:
            continue
        buckets: dict[str, list[str]] = {name: [] for name in DELIVERED_SPLITS}
        documents = dict.fromkeys(DELIVERED_SPLITS, 0)
        by_line: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(DELIVERED_SPLITS, 0))
        for record in unique:
            split = split_of_group.get(record.group_id)
            if split not in DELIVERED_SPLITS:
                raise SplitError(f"group {record.group_id} has no delivered split (got {split!r})")
            if split == "test" and not with_test:
                raise SplitError(
                    f"group {record.group_id} is delivered as test, but the eval set is frozen: "
                    "a frozen yardstick takes no new documents. Deliver new documents as train or val."
                )
            buckets[split].append(record.group_id)
            documents[split] += record.size
            if record.line is not None:
                by_line[record.line][split] += 1
        if not buckets["val"]:
            raise SplitError(f"{doc_type}: the delivered split has no val documents")
        if with_test and not buckets["test"]:
            raise SplitError(f"{doc_type}: the delivered split has no test documents")
        for split, members in buckets.items():
            for group_id in members:
                result.assignment[group_id] = split  # type: ignore[assignment]
        _assign_validation_halves(result, buckets["val"], seed)
        total = sum(documents.values())
        result.ratios_by_doc_type[doc_type] = {s: round(n / total, 4) for s, n in documents.items()}
        result.counts_by_doc_type[doc_type] = {s: len(m) for s, m in buckets.items()}
        if by_line:
            result.counts_by_line[doc_type] = {line: dict(c) for line, c in sorted(by_line.items())}
    return result


def _repair_empty_splits(
    buckets: dict[str, list[str]], groups: list[GroupRecord], doc_type: str, seed: int,
    starved: tuple[str, ...] = ("test", "val"),
) -> None:
    """Move one group into a starved split rather than ship an unusable corpus.

    An empty test split cannot be evaluated at all and an empty val split
    disables both early stopping and calibration. Fires only at very small N, and
    it is the one case where assignment is not stable under corpus growth.

    Synthetic groups are never moved out of train — that rule outranks this
    repair, because a test split made of generated documents measures the
    generator and reports it as model accuracy.
    """
    if len(groups) < MIN_GROUPS_TO_REPAIR:
        return
    movable = {g.group_id for g in groups if not g.synthetic}
    for split in starved:
        if buckets[split]:
            continue
        donor = max(buckets, key=lambda s: len([g for g in buckets[s] if g in movable]))
        candidates = [g for g in buckets[donor] if g in movable]
        if len(candidates) < 2:
            continue
        moved = sorted(candidates, key=lambda g: _stable_hash(g, seed))[-1]
        buckets[donor].remove(moved)
        buckets[split].append(moved)
        log.warning(
            "%s: group %r moved to %s so the split is not empty — an empty test split cannot be "
            "evaluated. This is the one case where assignment is not stable under growth.",
            doc_type, moved, split,
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
