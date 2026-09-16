"""Compile documents into chat-format JSONL training rows (SPEC_05, arch §6, §7).

Split first, expand second, inside whichever split the document was already
assigned to:

* **train** — one row per epoch, each in that epoch's sampled modality regime
  (arch v2.1 §6.1), written as ``train/epoch_1..4.jsonl``;
* **val / test** — three rows, one per regime, so image-only and noisy-OCR
  accuracy are measured on the full eval population.

The rows are built through :func:`inference_core.input_builder.build_training_row`,
which is the same function the serving path uses to assemble a request. That is
what makes the training prompt and the inference prompt provably identical rather
than identical by convention (arch §7).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from common.constants import MODALITY_MODES
from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages
from data_pipeline.dataset_builder.sample_modes import EPOCH_FILES, ModeAssignment, sample_modes
from data_pipeline.dataset_builder.split_groups import (
    GroupSplitAssignment,
    assert_no_leakage,
    assert_single_tenant,
    assert_synthetic_is_train_only,
)
from inference_core.input_builder import build_training_row

log = logging.getLogger(__name__)


class CorpusBuildError(RuntimeError):
    """Raised when a corpus cannot be assembled correctly."""


@dataclass
class SourceDocument:
    """One labeled document, ready to expand into training rows."""

    source_id: str
    doc_type: str
    golden_label: dict[str, Any]
    #: One markdown string per page, in page order — never one joined blob.
    #: Joining is what destroyed the image/text pairing and split every table
    #: crossing a page boundary (a blank line ends a Markdown table).
    ocr_pages: list[str]
    image_paths: list[str]
    acord_form: str | None = None
    tenant_id: str | None = None
    field_provenance: dict[str, str] = field(default_factory=dict)
    is_scanned: bool = False

    #: The family this document belongs to (arch v2.1 §8.2). Splitting happens at
    #: this level, not at source_id: the same carrier's template and the same
    #: account's renewals must not span the split, or eval measures template
    #: memorisation. Defaults to the document's own id, which makes a document
    #: with no detected family its own group rather than silently ungrouped.
    group_id: str | None = None
    carrier: str | None = None

    #: Generated rather than collected (arch v2.1 §4d). Train-only: its labels
    #: are perfect because they were generated from them, so scoring against one
    #: measures the generator.
    synthetic: bool = False

    @property
    def family(self) -> str:
        return self.group_id or self.source_id


@dataclass
class BuildResult:
    """The compiled corpus, plus what went into it."""

    rows_by_split: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    modality_counts: Counter = field(default_factory=Counter)
    corruption_details: dict[str, list[str]] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def all_rows(self) -> list[dict[str, Any]]:
        return [row for rows in self.rows_by_split.values() for row in rows]

    def summary(self) -> str:
        counts = {split: len(rows) for split, rows in sorted(self.rows_by_split.items())}
        return f"{len(self.all_rows)} rows {counts}, modality {dict(self.modality_counts)}"


def expand_document(
    document: SourceDocument,
    split: str,
    *,
    seed: int = 42,
    modes: tuple[str, ...] = MODALITY_MODES,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand one document into one row per mode in ``modes``.

    Every row lands in the **same** split, because the document was assigned
    before expansion. Returns ``(rows, corruption_details)``.

    Under arch v2.1 §6.1 the caller passes ONE mode per epoch, drawn by
    :mod:`data_pipeline.dataset_builder.sample_modes`. Passing all three — the v1
    behaviour — produces a 33/33/33 mix rather than the 50/20/30 the architecture
    specifies, and shows every document to the model three times per epoch, so a
    "3 epoch" run is nine passes. Val and test still take all three, so
    image-only accuracy is measured on the full eval population.
    """
    golden_json = json.dumps(document.golden_label, ensure_ascii=False, sort_keys=False)
    rows: list[dict[str, Any]] = []
    details: list[str] = []

    for mode in modes:
        if mode == "image_only":
            ocr_pages = None
        elif mode == "noisy_ocr_image":
            # One budget for the whole document, spent across its pages. Calling
            # the single-page corrupter per page would multiply the noise by the
            # page count and teach "OCR is always garbage" instead of arbitration.
            ocr_pages, page_details = corrupt_ocr_pages(
                document.ocr_pages, document.source_id, seed=seed
            )
            details.extend(page_details)
        else:
            ocr_pages = list(document.ocr_pages)

        row = build_training_row(
            document.doc_type,
            document.source_id,
            document.image_paths,
            ocr_pages,
            mode,
            golden_json,
            acord_form=document.acord_form,
            tenant_id=document.tenant_id,
            split=split,
            # Recorded, not enforced: de-identification is blocked (SPEC_05 §1).
            deidentified=False,
        )
        rows.append(row)

    return rows, details


def build_corpus(
    documents: list[SourceDocument],
    assignment: GroupSplitAssignment,
    *,
    seed: int = 42,
    mode_assignment: ModeAssignment | None = None,
) -> BuildResult:
    """Compile every document into its split's rows, then assert no leakage.

    ``mode_assignment`` carries the per-document, per-epoch modality draw
    (arch v2.1 §6.1). Without one, it is drawn here over the **train** documents
    with the same seed. Train always gets one row per epoch; there is no path
    that expands a train document into all three regimes.
    """
    result = BuildResult(rows_by_split={"train": [], "val": [], "test": []})
    if mode_assignment is None:
        mode_assignment = sample_modes(train_source_ids(documents, assignment), seed=seed)

    for document in documents:
        try:
            split = assignment.split_of(document.family)
        except Exception as exc:
            result.skipped.append((document.source_id, str(exc)))
            continue

        # Train draws ONE mode per epoch (§6.1); val and test keep all three so
        # image-only and noisy-OCR accuracy are measured on the full eval
        # population rather than a sample of it.
        if split == "train":
            epoch_modes = tuple(
                mode_assignment.mode_for(document.source_id, e)
                for e in range(1, mode_assignment.epochs + 1)
            )
        else:
            epoch_modes = MODALITY_MODES

        try:
            rows, details = expand_document(document, split, seed=seed, modes=epoch_modes)
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop a corpus
            result.skipped.append((document.source_id, f"expansion failed: {exc}"))
            log.warning("skipping %s: %s", document.source_id, exc)
            continue

        # Train rows are stamped with the epoch they belong to, so the corpus can
        # be written as epoch_1..4.jsonl and a run reproduced from the files
        # alone rather than from a sampler behaving identically at training time.
        if split == "train":
            for epoch, row in enumerate(rows, start=1):
                row["epoch"] = epoch

        # Stamped here rather than inside expand_document: the family is a
        # property of the corpus build, not of one document's expansion, and the
        # leakage assertion reads it off every row.
        for row in rows:
            row["group_id"] = document.family
            row["synthetic"] = document.synthetic
            if assignment.half_of(document.family):
                row["val_half"] = assignment.half_of(document.family)

        result.rows_by_split[split].extend(rows)
        for row in rows:
            result.modality_counts[row["modality_mode"]] += 1
        if details:
            result.corruption_details[document.source_id] = details

    # The checks that make the ordering rule real rather than documented.
    assert_no_leakage(assignment, result.all_rows)
    assert_synthetic_is_train_only(result.all_rows)
    assert_single_tenant(result.all_rows)

    log.info("corpus built: %s", result.summary())
    return result


def train_source_ids(
    documents: list[SourceDocument], assignment: GroupSplitAssignment
) -> list[str]:
    """The documents that will be sampled into epochs.

    Drawing modes for val and test documents too would not change their rows —
    they take all three regimes — but it would put their draws into the realised
    mix that the mix check reads, measuring something other than what trains.
    """
    found = []
    for document in documents:
        try:
            if assignment.split_of(document.family) == "train":
                found.append(document.source_id)
        except Exception:  # noqa: BLE001 - unassigned documents are reported by build_corpus
            continue
    return found


def train_rows_by_epoch(
    result: BuildResult, epochs: int = EPOCH_FILES
) -> dict[int, list[dict[str, Any]]]:
    """Split the train rows into one list per epoch file.

    Every epoch file must hold every train document exactly once. A document
    missing from an epoch, or present twice, means the run trains on something
    other than what the manifest's epoch count says, so that is refused here
    rather than discovered in a loss curve.
    """
    by_epoch: dict[int, list[dict[str, Any]]] = {e: [] for e in range(1, epochs + 1)}
    for row in result.rows_by_split.get("train", []):
        epoch = row.get("epoch")
        if epoch not in by_epoch:
            raise CorpusBuildError(
                f"train row for {row.get('source_id')!r} has epoch {epoch!r}; expected 1-{epochs}. "
                "Every train row belongs to exactly one epoch file (arch v2.1 §6.1)."
            )
        by_epoch[epoch].append(row)

    expected = sorted({row["source_id"] for rows in by_epoch.values() for row in rows})
    for epoch, rows in by_epoch.items():
        ids = sorted(row["source_id"] for row in rows)
        if ids != expected:
            raise CorpusBuildError(
                f"epoch {epoch} does not hold every train document exactly once "
                f"({len(ids)} rows for {len(expected)} documents)."
            )
    return by_epoch


def write_jsonl(rows: list[dict[str, Any]]) -> str:
    """Serialise rows to JSONL — one complete JSON object per line.

    Sorted by ``(source_id, modality_mode)`` so a rebuild with the same seed is
    byte-identical. A corpus that changes between builds cannot be compared
    across model versions.
    """
    ordered = sorted(rows, key=lambda r: (r["source_id"], r["modality_mode"]))
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
