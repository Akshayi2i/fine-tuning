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

from common.canonical import CanonicalLabelError, has_envelopes, training_target
from common.constants import MODALITY_MODES, TRAINER_SPECIAL_TAGS
from common.schemas import is_canonical
from data_pipeline.dataset_builder.cap_check import CapReport, estimate_row, evaluate
from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages
from data_pipeline.dataset_builder.policy_windows import (
    TargetReport,
    plan_windows,
    routed_pages,
    unread_values,
    window_target,
)
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
    #: A policy's line of business, from its label metadata. Selects the
    #: canonical schema the prompt carries and the target is written in; absent,
    #: the policy uses the client's canonical fallback.
    lob: str | list[str] | None = None
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

    #: The split the document arrived with (metadata ``split``), when the
    #: delivery was split upstream; and the source document it was made from
    #: (metadata ``template_id``), the family that split was drawn by.
    delivered_split: str | None = None
    template_id: str | None = None

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
    #: Gold values no window's target could carry, per document: printed only on
    #: pages no window of their group reads, or with no page recorded in a group
    #: read over several windows. The same values are unreachable at serving, so
    #: this is where a page rule that misses real content becomes visible.
    window_notes: dict[str, list[str]] = field(default_factory=dict)
    #: Documents set aside, by reason: ``budget``, ``trainer_tag``, ``expansion``.
    #: Every document that does not reach the corpus is counted here, so a stage
    #: summary can say how many were lost and why rather than a log line alone.
    set_aside: Counter = field(default_factory=Counter)
    #: Every row's estimated size against its task budget, and every document
    #: set aside because one of its rows did not fit (``cap_check``).
    cap_report: CapReport = field(default_factory=CapReport)

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
    """Expand one document into rows, per mode in ``modes``.

    One row per mode for a flat document type. A canonical policy is read as
    windows - a section group over a set of pages (``policy_windows``) - so it
    becomes one row per window per mode, each prompting for its slice of the
    schema and targeting its slice of the label. Every row carries
    ``mode_index``, its position in ``modes``, which ``build_corpus`` stamps an
    epoch from.

    Every row lands in the **same** split, because the document was assigned
    before expansion. Returns ``(rows, details)``; details prefixed ``window``
    are the gold values no window could carry.

    Under arch v2.1 §6.1 the caller passes ONE mode per epoch, drawn by
    :mod:`data_pipeline.dataset_builder.sample_modes`. Passing all three — the v1
    behaviour — produces a 33/33/33 mix rather than the 50/20/30 the architecture
    specifies, and shows every document to the model three times per epoch, so a
    "3 epoch" run is nine passes. Val and test still take all three, so
    image-only accuracy is measured on the full eval population.
    """
    windowed = document.doc_type == "policy" and is_canonical(
        document.doc_type, document.acord_form, document.lob
    )
    if windowed and not has_envelopes(document.golden_label):
        # Checked before slicing: a flat label has none of the canonical
        # sections, so every window's target would come out `{}` and the corpus
        # would teach "this policy states nothing" with no error anywhere.
        raise CanonicalLabelError(
            f"{document.source_id} is a policy with a flat, pre-canonical label. Convert it to "
            "the canonical schema before it enters the corpus."
        )

    golden_json = ""
    if not windowed:
        # The target, not the label: every date is written MM/DD/YYYY.
        target = training_target(
            document.golden_label, document.doc_type, document.acord_form, document.lob
        )
        golden_json = json.dumps(target, ensure_ascii=False, sort_keys=False)
    rows: list[dict[str, Any]] = []
    details: list[str] = []

    for mode_index, mode in enumerate(modes):
        if mode == "image_only":
            ocr_pages = None
        elif mode == "noisy_ocr_image" and windowed:
            # Corrupted per WINDOW below, each under its own budget: a window is
            # what one row shows the model. One budget for a 200-page policy put
            # two corruptions in one or two of its windows, and every other
            # "noisy" row was byte-identical to its clean one.
            ocr_pages = list(document.ocr_pages)
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

        if windowed:
            for row in _policy_window_rows(
                document, split, mode, ocr_pages, details, seed=seed
            ):
                row["mode_index"] = mode_index
                rows.append(row)
            continue

        row = build_training_row(
            document.doc_type,
            document.source_id,
            document.image_paths,
            ocr_pages,
            mode,
            golden_json,
            acord_form=document.acord_form,
            lob=document.lob,
            tenant_id=document.tenant_id,
            split=split,
            # Recorded, not enforced: de-identification is blocked (SPEC_05 §1).
            deidentified=False,
        )
        row["mode_index"] = mode_index
        rows.append(row)

    return rows, details


def _policy_window_rows(
    document: SourceDocument,
    split: str,
    mode: str,
    ocr_pages: list[str] | None,
    details: list[str],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """One row per window, planned exactly as serving plans them.

    The routed pages come from the OCR text THIS row carries - clean, corrupted
    or none - because that is what serving routes on for the same input.
    """
    page_count = len(document.image_paths)
    routed, declarations_page = routed_pages(ocr_pages, page_count)
    plans = plan_windows(document.lob, routed, declarations_page)

    report = TargetReport()
    rows: list[dict[str, Any]] = []
    for plan in plans:
        target = window_target(document.golden_label, document.lob, plan, report)
        indices = [page - 1 for page in plan.pages]
        window_ocr = None if ocr_pages is None else [ocr_pages[i] for i in indices]
        if window_ocr is not None and mode == "noisy_ocr_image":
            # The document-level budget, applied to the pages this row carries.
            # Seeded per window, so a rebuild is byte-identical.
            window_ocr, window_details = corrupt_ocr_pages(
                window_ocr, f"{document.source_id}#{plan.group}:{plan.window_index}", seed=seed
            )
            details.extend(
                f"window {plan.group}:{plan.window_index} {d}" for d in window_details
            )
        row = build_training_row(
            document.doc_type,
            document.source_id,
            [document.image_paths[i] for i in indices],
            window_ocr,
            mode,
            json.dumps(target, ensure_ascii=False, sort_keys=False),
            acord_form=document.acord_form,
            lob=document.lob,
            sections=plan.group,
            tenant_id=document.tenant_id,
            split=split,
            deidentified=False,
            # The document's own page numbers, never window-relative: the
            # target's page_ref is the document's, and so are the markers.
            page_numbers=list(plan.pages),
            total_pages=page_count,
        )
        row["task"] = plan.task
        row["window_index"] = plan.window_index
        row["window_pages"] = list(plan.pages)
        rows.append(row)

    notes = [f"window {mode}: unread {path}" for path in unread_values(
        document.golden_label, document.lob, plans
    )]
    notes += [f"window {mode}: unplaced {path}" for path in report.unplaced]
    if notes:
        log.warning(
            "%s (%s): %d gold value(s) no window can carry - printed on pages no window of "
            "their group reads, or with no page recorded. Serving cannot reach them either.",
            document.source_id, mode, len(notes),
        )
        details.extend(notes)
    return rows


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
            result.set_aside["expansion"] += 1
            log.warning("skipping %s: %s", document.source_id, exc)
            continue

        # Reject, never truncate. A row over its task budget would be clipped on
        # the pod, and a clipped target trains the model to stop early. The
        # whole document is set aside, not the one row: a document missing from
        # some epochs and not others is a run whose epoch count describes
        # something else, which train_rows_by_epoch refuses.
        refusal = _refusal(document.source_id, rows, result.cap_report)
        if refusal:
            category, reason = refusal
            result.skipped.append((document.source_id, reason))
            result.set_aside[category] += 1
            log.warning("setting aside %s: %s", document.source_id, reason)
            continue

        # Train rows are stamped with the epoch they belong to, so the corpus can
        # be written as epoch_1..4.jsonl and a run reproduced from the files
        # alone rather than from a sampler behaving identically at training time.
        for row in rows:
            mode_index = row.pop("mode_index")
            if split == "train":
                row["epoch"] = mode_index + 1

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
        corruption = [d for d in details if not d.startswith("window ")]
        windows = [d for d in details if d.startswith("window ")]
        if corruption:
            result.corruption_details[document.source_id] = corruption
        if windows:
            result.window_notes[document.source_id] = windows

    # The checks that make the ordering rule real rather than documented.
    assert_no_leakage(assignment, result.all_rows)
    if not assignment.delivered:
        # A delivered split placed synthetic twins of held-out sources in val and
        # test on purpose (assign_delivered_splits); the gate reports real ones apart.
        assert_synthetic_is_train_only(result.all_rows)
    assert_single_tenant(result.all_rows)

    log.info("corpus built: %s", result.summary())
    return result


def _refusal(
    source_id: str, rows: list[dict[str, Any]], report: CapReport
) -> tuple[str, str] | None:
    """``(category, why)`` this document cannot enter the corpus, or ``None``.

    Records the budget verdict on ``report``. Each row's estimate is computed
    once and reused for the verdict and the record.

    Two reasons, both judged over EVERY row before anything is recorded, because
    the document is the unit that is kept or lost:

    * **Text the trainer would parse as a tag.** ms-swift reads ``<image>``,
      ``<video>`` and the rest as special tags wherever they appear, so OCR or a
      target containing one would be rewritten and train on a prompt serving
      never sends. Checked here rather than at staging, so one bad page is set
      aside and reported at build instead of aborting a paid training run.
    * **A row over its task budget** — reject, never truncate.
    """
    for row in rows:
        for text in _texts(row):
            tag = next((t for t in TRAINER_SPECIAL_TAGS if t in text), None)
            if tag:
                return "trainer_tag", (
                    f"its text contains {tag!r}, which the trainer parses as a special tag — "
                    "the prompt it would train on is not the one serving sends"
                )

    estimates = [_estimate(row) for row in rows]
    verdicts = [evaluate(estimate) for estimate in estimates]
    failed = [(e, v) for e, v in zip(estimates, verdicts, strict=True) if not v[0]]
    if failed:
        estimate, (_fits, cap, reason) = failed[0]
        report.reject(source_id, estimate.task, estimate, cap, reason=reason)
        report.documents_rejected += 1
        return "budget", f"{len(failed)} of {len(rows)} row(s) exceed their task budget: {reason}"
    for estimate in estimates:
        report.record(estimate.task, estimate)
    report.documents_accepted += 1
    return None


def _texts(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for message in row["messages"]:
        content = message["content"]
        if isinstance(content, str):
            out.append(content)
        else:
            out += [b.get("text", "") for b in content if b.get("type") == "text"]
    return out


def _estimate(row: dict[str, Any]):
    """One built row's estimated cost against its task budget."""
    messages = row["messages"]
    user = messages[1]["content"] if len(messages) > 1 else []
    blocks = user if isinstance(user, list) else []
    return estimate_row(
        # A flat row is the whole-document `extract` task; a window names its own.
        task=row.get("task") or "extract",
        system_prompt=messages[0]["content"],
        ocr_pages=[b.get("text", "") for b in blocks if b.get("type") == "text"],
        page_count=sum(1 for b in blocks if b.get("type") == "image"),
        target_json=messages[-1]["content"] if messages[-1]["role"] == "assistant" else "",
        doc_type=row.get("doc_type"),
    )


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

    Every epoch file must hold every train document, and no row twice. A
    document missing from an epoch, or a row present twice, means the run trains
    on something other than what the manifest's epoch count says, so that is
    refused here rather than discovered in a loss curve.

    A canonical policy is several windows, so it is several rows in every epoch.
    Its windows can differ between epochs - each epoch draws the document in its
    own input mode, and an image-only row routes every page where an OCR row
    routes by keyword - so what must hold across epochs is the set of DOCUMENTS,
    not the set of rows.
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
        keys = [_row_identity(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise CorpusBuildError(f"epoch {epoch} holds a row twice.")
        ids = sorted({row["source_id"] for row in rows})
        if ids != expected:
            raise CorpusBuildError(
                f"epoch {epoch} does not hold every train document "
                f"({len(ids)} documents for {len(expected)} expected)."
            )
    return by_epoch


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, int]:
    """One row: its document, input mode, schema slice and window."""
    return (
        row["source_id"], row["modality_mode"],
        row.get("sections") or "", int(row.get("window_index") or 0),
    )


def write_jsonl(rows: list[dict[str, Any]]) -> str:
    """Serialise rows to JSONL — one complete JSON object per line.

    Sorted by ``(source_id, modality_mode, sections, window_index)`` so a rebuild
    with the same seed is byte-identical. A corpus that changes between builds
    cannot be compared across model versions.
    """
    ordered = sorted(rows, key=_row_identity)
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
