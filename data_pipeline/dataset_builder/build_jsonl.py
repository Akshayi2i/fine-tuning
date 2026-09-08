"""Compile documents into chat-format JSONL training rows (SPEC_05, arch §6, §7).

Split first, expand second. Each source document becomes **three** rows — one per
modality regime — inside whichever split the document was already assigned to.

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

from common.constants import MODALITY_MIX, MODALITY_MODES
from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages
from data_pipeline.dataset_builder.split_train_val_test import (
    SplitAssignment,
    assert_no_leakage,
    assert_single_tenant,
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
    """Expand one document into its modality variants.

    All three rows land in the **same** split, because the document was assigned
    before expansion. Returns ``(rows, corruption_details)``.
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
    assignment: SplitAssignment,
    *,
    seed: int = 42,
) -> BuildResult:
    """Compile every document into its split's rows, then assert no leakage."""
    result = BuildResult(rows_by_split={"train": [], "val": [], "test": []})

    for document in documents:
        try:
            split = assignment.split_of(document.source_id)
        except Exception as exc:
            result.skipped.append((document.source_id, str(exc)))
            continue

        try:
            rows, details = expand_document(document, split, seed=seed)
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop a corpus
            result.skipped.append((document.source_id, f"expansion failed: {exc}"))
            log.warning("skipping %s: %s", document.source_id, exc)
            continue

        result.rows_by_split[split].extend(rows)
        for row in rows:
            result.modality_counts[row["modality_mode"]] += 1
        if details:
            result.corruption_details[document.source_id] = details

    # The checks that make the ordering rule real rather than documented.
    assert_no_leakage(assignment, result.all_rows)
    assert_single_tenant(result.all_rows)

    log.info("corpus built: %s", result.summary())
    return result


def assert_modality_mix(result: BuildResult, *, tolerance: float = 0.02) -> None:
    """Assert the realised modality mix matches the target (arch §6).

    Checked on the **train split**, because that is the only split
    :func:`sample_to_target_mix` samples — val and test deliberately keep all
    three variants of every document so image-only and noisy-OCR accuracy are
    measured on the full eval population rather than a sample. Asserting the
    corpus-wide mix against 50/20/30 would therefore fail on a correct corpus.

    A silently wrong mix changes what the model learns about arbitration and
    about the no-OCR pathway, and produces no other symptom — no error, no
    metric movement, nothing in a loss curve. This is the only thing that
    notices.
    """
    total = sum(result.modality_counts.values())
    if not total:
        raise CorpusBuildError("corpus is empty")

    missing = set(MODALITY_MODES) - set(result.modality_counts)
    if missing:
        raise CorpusBuildError(
            f"corpus contains no {sorted(missing)} rows. Each regime teaches something specific: "
            "image_only satisfies the no-OCR production pathway, noisy_ocr_image teaches "
            "image-over-OCR arbitration (arch §6)."
        )

    train = result.rows_by_split.get("train") or []
    if not train:
        return  # nothing sampled yet; the shape check above is all that applies

    # One row is worth 1/n of the mix, so below n = 1/tolerance the target is
    # arithmetically unreachable and the check would fail on a correct corpus.
    # At 2% that is 50 rows — well under a real corpus and well over a fixture
    # set. Reported rather than silently skipped: an unenforced check that looks
    # enforced is how the previous version of this function passed a 96/2/2 mix.
    minimum = int(round(1 / tolerance)) if tolerance else 0
    if len(train) < minimum:
        log.warning(
            "train split has %d rows, so the %.0f%% modality tolerance is unreachable "
            "(one row is %.1f%% of the mix) — the ratio check is not enforced at this size. "
            "It applies from %d rows.",
            len(train), tolerance * 100, 100 / len(train), minimum,
        )
        return

    counts = Counter(row["modality_mode"] for row in train)
    realised = {mode: counts.get(mode, 0) / len(train) for mode in MODALITY_MODES}
    drifted = {
        mode: (share, MODALITY_MIX[mode])
        for mode, share in realised.items()
        if abs(share - MODALITY_MIX[mode]) > tolerance
    }
    if drifted:
        detail = "; ".join(
            f"{mode}: {got:.1%} against a {want:.0%} target"
            for mode, (got, want) in sorted(drifted.items())
        )
        raise CorpusBuildError(
            f"the train split's modality mix is off target by more than {tolerance:.0%} — {detail}. "
            f"({len(train)} rows.) The 50/20/30 split is what teaches the model to arbitrate "
            "between OCR and image and to work without OCR at all; a corpus that drifts from it "
            "trains a different behaviour and reports nothing."
        )


def sample_to_target_mix(
    result: BuildResult,
    *,
    seed: int = 42,
    target: dict[str, float] | None = None,
) -> BuildResult:
    """Down-sample rows so the realised mix approaches the arch §6 target.

    Expansion produces an even 1/3 split; the target is 50/20/30. Sampling is
    per-split and seeded, and **train is the only split sampled** — val and test
    keep all three variants for every document, so image-only and noisy-OCR
    accuracy can be measured on the full eval population rather than a sample.
    """
    import random

    target = target or MODALITY_MIX
    sampled = BuildResult(
        rows_by_split={s: list(rows) for s, rows in result.rows_by_split.items()},
        modality_counts=Counter(),
        corruption_details=dict(result.corruption_details),
        skipped=list(result.skipped),
    )

    train_rows = sampled.rows_by_split.get("train", [])
    if train_rows:
        by_mode: dict[str, list[dict[str, Any]]] = {}
        for row in train_rows:
            by_mode.setdefault(row["modality_mode"], []).append(row)

        # Size the corpus by whichever regime is most constrained by its target.
        total = min(
            int(len(rows) / target[mode]) for mode, rows in by_mode.items() if target.get(mode)
        )
        rng = random.Random(seed)
        kept: list[dict[str, Any]] = []
        for mode, rows in sorted(by_mode.items()):
            wanted = max(1, int(round(total * target.get(mode, 0))))
            ordered = sorted(rows, key=lambda r: r["source_id"])
            kept.extend(ordered if wanted >= len(ordered) else rng.sample(ordered, wanted))
        sampled.rows_by_split["train"] = sorted(kept, key=lambda r: (r["source_id"], r["modality_mode"]))

    for rows in sampled.rows_by_split.values():
        for row in rows:
            sampled.modality_counts[row["modality_mode"]] += 1
    return sampled


def write_jsonl(rows: list[dict[str, Any]]) -> str:
    """Serialise rows to JSONL — one complete JSON object per line.

    Sorted by ``(source_id, modality_mode)`` so a rebuild with the same seed is
    byte-identical. A corpus that changes between builds cannot be compared
    across model versions.
    """
    ordered = sorted(rows, key=lambda r: (r["source_id"], r["modality_mode"]))
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
