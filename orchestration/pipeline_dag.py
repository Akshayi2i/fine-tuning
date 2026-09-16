"""The 11 pipeline stages as addressable functions (SPEC_13 §8, arch §13).

``run.py`` composes these; an Airflow DAG or a GitHub Actions workflow could
schedule the same functions unchanged. Keeping them here rather than in the CLI
is what makes ``--from-stage`` a slice of a list instead of a special case in an
argument parser.

Four properties are requirements, not aspirations:

* **Every stage reads and writes Blob or the staging volume, never pod-local disk.**
  A pod is ephemeral; anything written to its own disk is gone at teardown.
* **Every training, evaluation and quantization job writes a run manifest.**
* **Stages are idempotent and resumable.** Re-running a completed stage is a
  no-op, not a duplicate — which is what makes ``--from-stage`` safe to reach for
  after a mid-pipeline failure.
* **Stage 6 is a hard stop.** A candidate that regresses any gating metric does
  not merge, and there is no override path. ``all`` therefore never reaches
  ``package`` on a failed gate.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.constants import ACTIVE_DOC_TYPES, DAY_ZERO_MIN_LABELS_PER_TYPE
from orchestration.runpod_controller import RunPodController, StagingVolume

log = logging.getLogger(__name__)

StageStatus = Literal["completed", "skipped", "blocked", "failed"]
Command = Literal["finetune", "package", "manual", "deploy-endpoint", "extract"]


class PipelineError(RuntimeError):
    """Raised when a stage cannot run at all."""


class GateBlocked(RuntimeError):
    """Raised when the promotion gate blocks a candidate.

    A distinct type because it is not a failure — the pipeline worked, and the
    model is the thing that did not qualify. Callers exit non-zero on both, but
    only this one prints per-metric deltas.
    """

    def __init__(self, message: str, gate_result: Any = None) -> None:
        super().__init__(message)
        self.gate_result = gate_result


# --------------------------------------------------------------------------
# Context and results
# --------------------------------------------------------------------------


@dataclass
class StageContext:
    """Everything the stages share. One object, so a stage cannot reach for
    global state and quietly become unrunnable outside the CLI."""

    client: BlobClient
    controller: RunPodController
    out_version: str
    #: Separately permissioned, because ``raw-documents/`` holds unredacted
    #: originals and only ingestion and OCR may reach them (arch §18a). Stages 1
    #: and 2 use this one; everything downstream works from ``processed/``.
    raw_client: BlobClient | None = None
    corpus_version: str = ""
    input_dir: Path | None = None
    doc_types: list[str] = field(default_factory=lambda: list(ACTIVE_DOC_TYPES))
    tenant_id: str | None = None
    formats: list[str] = field(default_factory=lambda: ["fp16", "q5_k_m"])
    dtype: str = "fp16"
    #: Overrides the configured class for the training pod. ``None`` means the
    #: class in ``config/pipeline.yaml``.
    gpu_class: str | None = None

    # -- flags -------------------------------------------------------------
    dry_run: bool = True
    skip_ingest: bool = False
    foundation_only: bool = False
    train_vit: bool = False
    min_labels_per_type: int = DAY_ZERO_MIN_LABELS_PER_TYPE
    push_adapters: bool = False
    skip_quantize: bool = False
    keep_staging: bool = False
    from_blob: bool = False
    git_commit: str | None = None
    seed: int = 42
    #: Retries for a *failed* stage. A blocked gate is never retried — that
    #: would be an override path with extra steps.
    max_attempts: int = 1
    #: Seconds between retries, multiplied by the attempt number. Carried on the
    #: context rather than read from config inside the loop so a caller — a test,
    #: or a run that wants to fail fast — can set it without editing the file.
    retry_backoff_seconds: int = 0

    # -- injected collaborators -------------------------------------------
    #: OCR engine for stage 2. Injected so the DAG is testable without MinerU,
    #: and so swapping the engine (SPEC_03 keeps it swappable) touches one line.
    ocr_engine: Any = None
    #: Produces the candidate's scores against the frozen golden eval set. The
    #: DAG owns the *gate*, not the scoring: scoring needs a model and a GPU,
    #: and pretending otherwise would put a fabricated pass inside the gate.
    metrics_provider: Callable[[StageContext], dict[str, Any]] | None = None
    #: The production version's scores to gate against. ``None`` means this is
    #: the first version and there is nothing to regress against.
    baseline_metrics: dict[str, Any] | None = None
    #: Per-format metrics from the SPEC_12 extraction routine over the frozen
    #: golden eval set, keyed by format. Supplied, the quantize stage applies the
    #: SPEC_10 thresholds and refuses to publish a format that fails.
    quant_metrics: dict[str, dict[str, Any]] | None = None
    #: Per-adapter re-validation results for a Foundation major bump: every
    #: dependent adapter retrained against the new Foundation and gated. Absent
    #: evidence blocks promotion rather than deferring the question (arch §12).
    #: Per-doc-type "has this adapter been revalidated against the new
    #: Foundation" flags, for the arch §12 cascade block.
    revalidation_evidence: dict[str, bool] | None = None
    #: Cross-type regression evidence for a CONTINUED Foundation, as
    #: ``{doc_type: {"current": {...}, "candidate": {...}}}`` — both sides, the
    #: shape `promotion_gate` reads. Separate from `revalidation_evidence`
    #: because that field holds booleans for a different question; reusing it
    #: produced ``{"acord": {"candidate": {...}}}`` with no ``"current"``, which
    #: the gate correctly rejected as empty. The result was that a continued
    #: Foundation could not pass by ANY input: supply metrics and it read as
    #: empty evidence, supply the booleans and it read as no evidence at all.
    cross_type_evidence: dict[str, dict[str, dict[str, float]]] | None = None

    #: Staged checkpoint directories from the training run (arch v2.1 §11.2).
    #: Supplied by the trainer in a real cycle; empty in a dry run, where nothing
    #: was written and there is honestly nothing to choose between.
    checkpoints: list[str] = field(default_factory=list)

    #: The checkpoint early stopping liked. Scored alongside the trailing ones
    #: even when it is not among them — that is exactly the interesting case,
    #: because training continued past what loss preferred.
    best_loss_checkpoint: str | None = None

    #: Injectable so selection is testable without a GPU, and so the scorer is
    #: the SAME one the gate uses: a selector scoring by a different definition
    #: of "correct" picks a checkpoint the gate then rejects.
    checkpoint_scorer: Any = None

    # -- accumulated state -------------------------------------------------
    results: dict[str, StageResult] = field(default_factory=dict)
    manifests: dict[str, Any] = field(default_factory=dict)
    unlabeled_backlog: dict[str, list[str]] = field(default_factory=dict)

    @property
    def raw(self) -> BlobClient:
        """The client permitted to touch ``raw-documents/``."""
        return self.raw_client or self.client

    @property
    def volume(self) -> StagingVolume:
        return self.controller.volume

    @property
    def corpus(self) -> str:
        """The corpus version being trained on; defaults to the output version."""
        return self.corpus_version or self.out_version


@dataclass
class StageResult:
    """What one stage did."""

    name: str
    status: StageStatus
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("completed", "skipped")


@dataclass
class RunReport:
    """The record of one command invocation."""

    command: str
    version: str
    results: list[StageResult] = field(default_factory=list)
    blocked_at: str | None = None
    failed_at: str | None = None
    unlabeled_backlog: dict[str, list[str]] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def ok(self) -> bool:
        return self.blocked_at is None and self.failed_at is None

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "version": self.version,
            "ok": self.ok,
            "started_at": self.started_at,
            "blocked_at": self.blocked_at,
            "failed_at": self.failed_at,
            "unlabeled_backlog": {k: len(v) for k, v in sorted(self.unlabeled_backlog.items())},
            "unlabeled_source_ids": {k: v for k, v in sorted(self.unlabeled_backlog.items()) if v},
            "stages": [
                {"stage": r.name, "status": r.status, "detail": r.detail, **r.data}
                for r in self.results
            ],
        }

    def render(self) -> str:
        lines = [f"{self.command} {self.version}: {'OK' if self.ok else 'FAILED'}"]
        lines += [f"  [{r.status:9}] {r.name}: {r.detail}" for r in self.results]
        for doc_type, backlog in sorted(self.unlabeled_backlog.items()):
            if backlog:
                lines.append(f"  unlabeled backlog ({doc_type}): {len(backlog)} document(s) awaiting review")
        if self.blocked_at:
            lines.append(f"  BLOCKED at {self.blocked_at}")
        if self.failed_at:
            lines.append(f"  FAILED at {self.failed_at}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Stage 1 — ingestion
# --------------------------------------------------------------------------


def _is_ingested(ctx: StageContext) -> bool:
    """Only ``--skip-ingest`` skips ingestion.

    There is no cheap completion check here, and the one that used to be here
    was actively wrong: "any document was ever ingested" is true from cycle two
    onward, so ``finetune --input ./new_batch`` skipped the stage entirely and
    built the corpus from the previous batch.

    Running it is the correct answer rather than a concession. ``ingest_directory``
    is checksum-deduped and idempotent, so a re-run over already-ingested
    documents writes nothing and reports them as duplicates — cheap, and it
    cannot be wrong in the direction that silently loses 500 documents.
    """
    return ctx.skip_ingest


def stage_ingestion(ctx: StageContext) -> StageResult:
    """Checksum-deduped, immutable writes into ``raw-documents/``."""
    from data_pipeline.ingestion.pull_raw_pdfs import ingest_directory

    if ctx.skip_ingest:
        return StageResult("ingestion", "skipped", "--skip-ingest")
    if ctx.input_dir is None:
        raise PipelineError("ingestion needs --input; pass --skip-ingest to reuse what is in Blob")

    ingested, duplicates, failed = [], [], []
    for doc_type in ctx.doc_types:
        source = ctx.input_dir / doc_type if (ctx.input_dir / doc_type).is_dir() else ctx.input_dir
        result = ingest_directory(source, doc_type, ctx.raw, tenant_id=ctx.tenant_id)
        ingested += result.ingested
        duplicates += [name for name, _sid in result.skipped_duplicates]
        failed += [f"{name}: {reason}" for name, reason in result.failed]

    return StageResult(
        "ingestion", "completed",
        f"{len(ingested)} ingested, {len(duplicates)} duplicate(s) skipped, {len(failed)} failed",
        {"ingested": len(ingested), "duplicates": len(duplicates), "failed": failed},
    )


# --------------------------------------------------------------------------
# Stage 2 — preprocessing (GPU)
# --------------------------------------------------------------------------


def _is_preprocessed(ctx: StageContext) -> bool:
    from data_pipeline.ocr.run_mineru import find_unprocessed

    return not any(find_unprocessed(ctx.raw, dt, ctx.tenant_id) for dt in ctx.doc_types)


def stage_preprocessing(ctx: StageContext) -> StageResult:
    """MinerU OCR plus page rendering at the resolution cap. **Runs on GPU.**"""
    from data_pipeline.ocr.run_mineru import MinerUEngine, find_unprocessed, process_batch

    engine = ctx.ocr_engine or MinerUEngine()
    processed, skipped, failed = [], [], []
    for doc_type in ctx.doc_types:
        pending = find_unprocessed(ctx.raw, doc_type, ctx.tenant_id)
        if not pending:
            continue
        outcome = process_batch(pending, doc_type, ctx.raw, engine, tenant_id=ctx.tenant_id)
        processed += outcome["processed"]
        skipped += outcome["skipped"]
        failed += [f"{sid}: {reason}" for sid, reason in outcome["failed"]]

    return StageResult(
        "preprocessing", "completed",
        f"{len(processed)} processed, {len(skipped)} unchanged, {len(failed)} failed",
        {"processed": len(processed), "failed": failed},
    )


# --------------------------------------------------------------------------
# Stage 3 — labeling (human, outside the CLI)
# --------------------------------------------------------------------------


def stage_labeling(ctx: StageContext) -> StageResult:
    """Not automated — this stage **reports** on human work and gates on it.

    ``finetune`` ingests and OCRs everything, then builds the corpus from only
    the source_ids that have a validated golden label, reports the rest as a
    backlog, and continues. That is the honest shape: you ingest 500 documents,
    200 are labeled, you train on 200, and the command tells you 300 are waiting
    for a reviewer (SPEC_13 §2).

    It aborts only when the labeled set is empty or below
    ``--min-labels-per-type``, which is the same day-zero floor SPEC_04 uses.
    """
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids

    counts, thin = {}, []
    for doc_type in ctx.doc_types:
        labeled = set(list_labeled_source_ids(ctx.client, doc_type, ctx.tenant_id))
        # Counted from processed/ rather than raw-documents/: a document that has
        # not been OCR'd cannot be labeled or trained on either way, and it keeps
        # orchestration out of the container holding unredacted originals.
        processed = {
            key.split("/")[3]
            for key in ctx.client.list(f"processed/{paths._tenant(ctx.tenant_id)}/{doc_type}/")
            if key.endswith("ocr_meta.json")
        }
        ctx.unlabeled_backlog[doc_type] = sorted(processed - labeled)
        counts[doc_type] = len(labeled)
        if len(labeled) < ctx.min_labels_per_type:
            thin.append(f"{doc_type}: {len(labeled)} < {ctx.min_labels_per_type}")

    total = sum(counts.values())
    if total == 0:
        raise PipelineError(
            "no validated golden labels exist, so there is nothing to train on. Labeling is human "
            "work and cannot sit inside an automated command; ingest and OCR have run, and the "
            "documents are waiting in the review tool (SPEC_04)."
        )
    if thin:
        raise PipelineError(
            "the labeled corpus is below the day-zero floor for: " + "; ".join(thin) + ". "
            f"Training on fewer than {ctx.min_labels_per_type} documents per type produces a model "
            "whose eval numbers are noise (SPEC_04). Lower --min-labels-per-type deliberately if "
            "you intend a smoke test rather than a real cycle."
        )

    backlog = sum(len(v) for v in ctx.unlabeled_backlog.values())
    return StageResult(
        "labeling", "completed",
        f"{total} labeled across {len(counts)} type(s); {backlog} document(s) awaiting review",
        {"labeled_by_type": counts, "unlabeled_backlog": backlog},
    )


# --------------------------------------------------------------------------
# Stage 4 — dataset build
# --------------------------------------------------------------------------


def load_labeled_documents(ctx: StageContext) -> list[Any]:
    """Assemble ``SourceDocument``s from Blob for every labeled source_id."""
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids, load_golden_label

    documents = []
    for doc_type in ctx.doc_types:
        for source_id in list_labeled_source_ids(ctx.client, doc_type, ctx.tenant_id):
            label, metadata = load_golden_label(source_id, doc_type, ctx.client, ctx.tenant_id)
            meta_key = paths.ocr_meta(doc_type, source_id, ctx.tenant_id)
            if not ctx.client.exists(meta_key):
                log.warning("%s is labeled but not OCR'd — skipping; re-run stage 2", source_id)
                continue
            ocr_meta = ctx.client.read_json(meta_key)
            page_count = int(ocr_meta.get("page_count", 1))

            documents.append(SourceDocument(
                source_id=source_id,
                doc_type=doc_type,
                golden_label=label,
                # Per page, never joined: the pairing with each page image is
                # what tells the model which text it is looking at, and joining
                # also split every table that crossed a page boundary.
                ocr_pages=[
                    ctx.client.read_text(paths.processed_page(doc_type, source_id, page, "md", ctx.tenant_id))
                    for page in range(1, page_count + 1)
                ],
                image_paths=[
                    paths.processed_page(doc_type, source_id, page, "png", ctx.tenant_id)
                    for page in range(1, page_count + 1)
                ],
                acord_form=metadata.get("acord_form"),
                tenant_id=ctx.tenant_id,
                field_provenance=metadata.get("field_provenance", {}),
                is_scanned=bool(ocr_meta.get("failed_pages")),
            ))
    return documents


def _is_corpus_built(ctx: StageContext) -> bool:
    return ctx.client.exists(paths.corpus_manifest(ctx.corpus, ctx.tenant_id))


def stage_dataset_build(ctx: StageContext) -> StageResult:
    """Split, expand into modality variants, write JSONL, and pin the corpus.

    The split happens **before** modality expansion, so a document's three
    variants land in one split. Reversing that order leaks a document's own
    content into its evaluation and inflates every number downstream (arch §7).
    """
    from data_pipeline.corpus_manifest import build_manifest
    from data_pipeline.dataset_builder.build_jsonl import (
        assert_modality_mix,
        build_corpus,
        sample_to_target_mix,
        write_jsonl,
    )
    from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_group_splits

    documents = load_labeled_documents(ctx)
    if not documents:
        raise PipelineError("no labeled, OCR'd documents to build a corpus from")

    # One GroupRecord per family, per doc type. A document with no detected
    # family is its own group — which reproduces the v1 per-document behaviour
    # for that document rather than leaving it unassigned.
    records: dict[str, dict[str, GroupRecord]] = {}
    for document in documents:
        per_type = records.setdefault(document.doc_type, {})
        record = per_type.get(document.family)
        if record is None:
            per_type[document.family] = GroupRecord(
                group_id=document.family,
                doc_type=document.doc_type,
                source_ids=[document.source_id],
                carrier=document.carrier,
                synthetic=document.synthetic,
            )
        else:
            record.source_ids.append(document.source_id)
            # A family is synthetic only if every member is. One real document in
            # the group makes the whole group splittable, which is the safe
            # direction: it keeps generated labels out of val and test.
            record.synthetic = record.synthetic and document.synthetic

    by_type = {dt: sorted(v.values(), key=lambda r: r.group_id) for dt, v in records.items()}
    assignment = assign_group_splits(by_type, seed=ctx.seed)
    built = build_corpus(documents, assignment, seed=ctx.seed)
    # Expansion produces an even third of each regime; the arch §6 target is
    # 50/20/30, and sampling train is what reaches it. Val and test keep all
    # three variants so image-only accuracy is measured on the full population.
    built = sample_to_target_mix(built, seed=ctx.seed)
    assert_modality_mix(built)

    for doc_type in sorted(by_type):
        for split, rows in sorted(built.rows_by_split.items()):
            subset = [r for r in rows if r["doc_type"] == doc_type]
            if subset:
                ctx.client.write_text(
                    paths.corpus_split(ctx.corpus, doc_type, split, ctx.tenant_id),
                    write_jsonl(subset),
                )

    first = documents[0]
    ocr_environment = ctx.client.read_json(
        paths.ocr_meta(first.doc_type, first.source_id, ctx.tenant_id)
    )
    manifest, coverage = build_manifest(
        corpus_version=ctx.corpus,
        tenant_id=paths._tenant(ctx.tenant_id),
        rows_by_split=built.rows_by_split,
        golden_labels_by_source={d.source_id: d.golden_label for d in documents},
        provenance_by_source={d.source_id: d.field_provenance for d in documents},
        split_assignment=assignment.as_dict(),
        ocr_environment=ocr_environment,
        doc_types=sorted(by_type),
        seed=ctx.seed,
        git_commit=ctx.git_commit or "unknown",
    )
    ctx.client.write_json(paths.corpus_manifest(ctx.corpus, ctx.tenant_id), manifest)

    for warning in coverage.warnings:
        log.warning("corpus coverage: %s", warning)

    return StageResult(
        "dataset_build", "completed",
        f"{len(built.all_rows)} rows from {len(documents)} document(s); {built.summary()}",
        {
            "rows": len(built.all_rows),
            "documents": len(documents),
            "coverage_warnings": list(coverage.warnings),
            "confusable_example_count": coverage.confusable_example_count,
        },
    )


# --------------------------------------------------------------------------
# Stage 5 — training
# --------------------------------------------------------------------------


def _is_trained(ctx: StageContext) -> bool:
    """One adapter on the volume is a complete training stage (arch v2.1 §4.1).

    v1 also required one staged adapter per document type, because a run was not
    finished until the whole fan-out was. There is no fan-out now.
    """
    return ctx.volume.exists(paths.staging_adapter_dir("foundation", ctx.out_version))


def stage_training(ctx: StageContext) -> StageResult:
    """ONE unified extractor run (arch v2.1 §4.1).

    v1 trained a Foundation and then fanned out one adapter per document type,
    sequentially, because each sat on top of the Foundation's weights. That
    topology is gone: vLLM applies one LoRA per request, so a Foundation LoRA and
    a per-type LoRA could never both be active on the same call — the stack was
    unservable, not merely awkward. Per-type adapters return only through the
    §4.2 graduation gate, trained on the MERGED foundation and never stacked.
    """
    from registry_utils.models import DataStats
    from training.train import train

    corpus_manifest = ctx.client.read_json(paths.corpus_manifest(ctx.corpus, ctx.tenant_id))
    counts = corpus_manifest.get("example_counts", {})
    data_stats = DataStats(
        train_examples=_count_rows(counts.get("train")),
        val_examples=_count_rows(counts.get("val")),
        test_examples=_count_rows(counts.get("test")),
        modality_mix=corpus_manifest.get("modality_mix", {}),
        lob_coverage=corpus_manifest.get("lob_coverage", {}),
        alias_coverage=corpus_manifest.get("alias_coverage", {}),
        confusable_example_count=corpus_manifest.get("confusable_example_count", 0),
        tenant_ids=[paths._tenant(ctx.tenant_id)],
    )

    with ctx.controller.session_pod("train", gpu_class=ctx.gpu_class):
        _swift, manifest = train(
            corpus_version=ctx.corpus,
            out_version=ctx.out_version,
            client=ctx.client,
            corpus_manifest=corpus_manifest,
            data_stats=data_stats,
            train_vit=ctx.train_vit,
            dry_run=ctx.dry_run,
        )
        # Keyed "foundation" so the gate, the cascade query and the ViT gate keep
        # reading one well-known key. The run_type on the manifest says what it
        # actually is; this is the slot, not the claim.
        ctx.manifests["foundation"] = manifest
        # The staging volume is where merge, quantize and push look for the
        # weights. Without this mark the artifacts exist and the pipeline cannot
        # find them.
        ctx.volume.mark(paths.staging_adapter_dir("foundation", ctx.out_version))

    if ctx.push_adapters:
        # Belt and braces: the adapter is tens of MB, so pushing it now costs
        # little and means a reclaimed volume loses only the merged model.
        blob_dir = paths.adapter_dir("foundation", ctx.out_version)
        ctx.client.write_json(
            f"{blob_dir}/adapter_placeholder.json", {"staged_copy_of": manifest.run_id}
        )

    return StageResult(
        "training", "completed",
        f"trained {manifest.run_id} on {data_stats.train_examples} examples",
        {
            "runs": [manifest.run_id],
            "run_type": manifest.run_type,
            "push_adapters": ctx.push_adapters,
        },
    )


def _count_rows(bucket: Any) -> int:
    """Sum a nested ``{doc_type: {mode: count}}`` bucket from the corpus manifest."""
    if not isinstance(bucket, dict):
        return 0
    return sum(
        count
        for modes in bucket.values()
        if isinstance(modes, dict)
        for count in modes.values()
    )


# --------------------------------------------------------------------------
# Stage 6 — checkpoint selection (arch v2.1 §11.2)
# --------------------------------------------------------------------------


def stage_checkpoint_eval(ctx: StageContext) -> StageResult:
    """Pick the checkpoint that ships, by GENERATED field F1.

    Validation loss drove selection under v1, and it is a poor proxy: averaged
    over every token, it is dominated by the easy copy tokens — schema keys, JSON
    punctuation, boilerplate — the model gets right within a hundred steps. A
    checkpoint can improve on loss while getting worse at the values, which is
    the only thing the gate reads.

    Skipped, explicitly, when nothing was staged: a dry run never launched
    ms-swift, so there is nothing to choose between, and merging the staged
    adapter directory is the honest fallback rather than a silent one.
    """
    from evaluation.checkpoint_eval import CheckpointEvalError, select_best

    checkpoints = sorted(ctx.checkpoints or [])
    if not checkpoints:
        return StageResult(
            "checkpoint_eval", "skipped",
            "no staged checkpoints to choose between; merge takes the staged adapter directory",
            {"selected": None},
        )
    # No separate dry-run branch: a dry run never launched ms-swift, so it has no
    # checkpoints, and the check above already covers it. A second condition
    # meaning the same thing is one that can disagree with the first.
    scorer = ctx.checkpoint_scorer
    if scorer is None:  # pragma: no cover - needs a GPU
        from common.config import base_model_config, generation_config
        from evaluation.checkpoint_eval import vllm_scorer

        base = base_model_config()["model"]
        scorer = vllm_scorer(
            base_model=f"{base['model_id']}@{base['revision']}",
            val_path=paths.corpus_eval_split(ctx.corpus, "val", ctx.tenant_id),
            generation_config=generation_config(),
        )

    try:
        report = select_best(checkpoints, scorer, best_loss=ctx.best_loss_checkpoint)
    except CheckpointEvalError as exc:
        raise PipelineError(
            f"no checkpoint could be selected for {ctx.out_version}: {exc}. Merging an "
            "arbitrary one would ship a model nobody measured."
        ) from exc

    ctx.client.write_json(paths.checkpoint_selection(ctx.out_version), report.as_dict())
    detail = f"selected {report.selected} (margin {report.margin:+.4f} field F1)"
    if report.loss_and_f1_disagreed:
        detail += f"; validation loss would have shipped {report.best_loss_checkpoint}"

    return StageResult("checkpoint_eval", "completed", detail, report.as_dict())


# --------------------------------------------------------------------------
# Stage 6 — evaluation and the promotion gate (HARD STOP)
# --------------------------------------------------------------------------


def _major(version: str) -> str:
    """The major component of ``v2``, ``v2.1``, ``foundation-v3`` — all ``2``/``3``."""
    tag = version.rsplit("-", 1)[-1].lstrip("vV")
    return tag.split(".")[0] or tag


def is_major_bump(previous: str | None, candidate: str) -> bool:
    """Whether the candidate is a new Foundation **major** version.

    A patch that continues from the previous Foundation is a different case,
    handled by the gate's own cross-type evidence rule.
    """
    if not previous:
        return False
    return _major(previous) != _major(candidate)


@dataclass
class CascadeWorkList:
    """Which adapters a Foundation bump invalidates, and whether they have passed.

    A Foundation move changes the weights every per-type adapter was trained on
    top of, so promoting it without re-validating them ships three models that
    were never evaluated against the base they now sit on (arch §12). The
    dependency is recorded on every adapter manifest, so this is a query rather
    than an audit.
    """

    previous_foundation: str | None
    dependents: list[str] = field(default_factory=list)
    revalidated: list[str] = field(default_factory=list)
    outstanding: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return bool(self.outstanding)

    def describe(self) -> str:
        return (
            f"Foundation major bump from {self.previous_foundation}: "
            f"{len(self.dependents)} dependent adapter(s), "
            f"{len(self.outstanding)} still to re-validate ({', '.join(self.outstanding)})"
        )


def foundation_upgrade_work_list(ctx: StageContext) -> CascadeWorkList:
    """The dependent-adapter re-validation list for a Foundation major bump."""
    from registry_utils.query_registry import adapters_depending_on, latest_promoted

    # "unified" under arch v2.1 §4.1. A graduated per-type adapter (§4.2) still
    # records the unified run it was trained on as its foundation_version, so the
    # dependency-upgrade rule is unchanged — only the run_type it looks for moved.
    previous = latest_promoted(ctx.client, "unified")
    work = CascadeWorkList(previous_foundation=previous)
    if not is_major_bump(previous, ctx.out_version):
        return work

    work.dependents = adapters_depending_on(previous or "", ctx.client)
    evidence = ctx.revalidation_evidence or {}
    work.revalidated = sorted(a for a in work.dependents if evidence.get(a))
    work.outstanding = sorted(a for a in work.dependents if not evidence.get(a))
    return work


def eval_report_metrics(ctx: StageContext) -> dict[str, Any]:
    """The default metrics provider: score the frozen golden eval set.

    Runs SPEC_08's ``run_eval`` — the same code the promotion gate's numbers are
    supposed to come from — after asserting the eval set does not overlap the
    corpus. If an eval report for this version already exists (a resumed run, or
    a scoring pass done separately), it is read rather than recomputed.

    Wiring this in is what makes ``finetune`` completable: the CLI supplied no
    provider at all, so every real run trained a Foundation and three adapters on
    an A100 and then aborted at the gate.
    """
    from evaluation.run_eval import assert_eval_set_disjoint

    assert_eval_set_disjoint(ctx.client, ctx.corpus, ctx.tenant_id)

    summary_key = paths.eval_report(ctx.out_version)
    if ctx.client.exists(summary_key):
        report = ctx.client.read_json(summary_key)
        metrics = report.get("gate_metrics") or report.get("candidate_metrics") or {}
        if metrics:
            log.info("gate metrics read from %s", summary_key)
            return dict(metrics)

    raise PipelineError(
        f"no eval report at {summary_key}, and scoring the candidate needs the model on a GPU "
        "(evaluation/run_eval.py, still waiting on a live backend). Run the eval pass and write "
        "its report, or pass a metrics_provider explicitly. The gate will not judge a candidate "
        "on metrics nobody measured."
    )


def default_baseline_metrics(ctx: StageContext) -> dict[str, Any] | None:
    """The production version's scores, read from its own eval report.

    ``None`` means there is no promoted version yet — a genuine first version,
    which the gate handles. It never means "could not find one".
    """
    from registry_utils.query_registry import latest_promoted

    promoted = latest_promoted(ctx.client, "unified")
    if not promoted:
        return None
    # Strip whichever lineage prefix the run id carries. A run id is
    # "<lineage>-<version>" and eval reports are keyed by version alone, so
    # hardcoding one prefix broke the moment the lineage was renamed.
    key = paths.eval_report(promoted.rsplit("-", 1)[-1])
    if not ctx.client.exists(key):
        log.warning(
            "promoted version %s has no eval report at %s, so this candidate is gated as a first "
            "version — nothing to regress against. That is a gap in the previous cycle's records, "
            "not a pass.", promoted, key,
        )
        return None
    report = ctx.client.read_json(key)
    return report.get("gate_metrics") or report.get("candidate_metrics") or None


def stage_evaluation_gate(ctx: StageContext) -> StageResult:
    """Score the candidate against the frozen golden eval set, then gate.

    **This stage is a hard stop.** A candidate that regresses on any gating
    metric does not merge, the command exits non-zero with per-metric deltas, and
    ``all`` never reaches ``package``. There is no ``--force``: an override that
    exists gets used on the afternoon someone is in a hurry, which is exactly the
    afternoon the gate was built for.
    """
    from evaluation.gating import apply_to_manifest, promotion_gate

    provider = ctx.metrics_provider or eval_report_metrics
    candidate = provider(ctx)
    baseline = ctx.baseline_metrics
    if baseline is None:
        baseline = default_baseline_metrics(ctx)

    # Read the manifest back from the registry when this run did not train it —
    # a resume, or --from-stage evaluation_gate, leaves ctx.manifests empty, and
    # the gate then saw continued_from=None and silently dropped the cross-type
    # requirement a continued Foundation is supposed to face. It also skipped
    # recording the decision, while stage_push published the run regardless.
    foundation = ctx.manifests.get("foundation")
    if foundation is None:
        from registry_utils.query_registry import RegistryQueryError
        from registry_utils.query_registry import get as get_manifest

        try:
            foundation = get_manifest(f"extractor-{ctx.out_version}", ctx.client)
            ctx.manifests["foundation"] = foundation
        except (RegistryQueryError, KeyError, FileNotFoundError) as exc:
            raise PipelineError(
                f"no run manifest for extractor-{ctx.out_version}, so the gate cannot tell "
                "whether this Foundation continued from a previous checkpoint — and a continued "
                "one must show cross-type regression evidence before promotion (arch §12). "
                f"Re-run training for this version rather than gating blind ({exc})."
            ) from exc
    result = promotion_gate(
        candidate,
        baseline,
        continued_from=getattr(foundation, "continued_from", None),
        # Passed through verbatim: the operator supplies both sides, because
        # only they know which promoted per-type report is the baseline.
        cross_type_evidence=ctx.cross_type_evidence or None,
    )

    # `gate_decision`, not `eval_report`. Writing here used to clobber the
    # scored EvalReport at the same key, taking `by_doc_type` and every error
    # record with it — which is what `vit_gate` reads to decide whether the
    # vision encoder is the bottleneck.
    report_key = paths.gate_decision(ctx.out_version)
    ctx.client.write_json(report_key, {
        "version": ctx.out_version,
        "candidate_metrics": candidate,
        "gate_metrics": candidate,
        "baseline_metrics": baseline,
        "passed": result.passed,
        "failed_gates": result.failed_gates,
        # The full verdict per metric — floor, interval and basis — not just a
        # delta. "Why was this blocked" needs the evidence, not the difference.
        "verdicts": [v.as_dict() for v in result.verdicts],
        "improved_metrics": result.improved_metrics,
        "waived_gates": sorted(result.waived),
        "override": result.override.as_dict() if result.override else None,
    })
    ctx.volume.write(paths.staging_eval_report(ctx.out_version), json.dumps(candidate))

    apply_to_manifest(result, foundation)
    from registry_utils.write_run_manifest import write_manifest

    write_manifest(foundation, ctx.client)

    if not result.passed:
        raise GateBlocked(
            f"promotion gate blocked {ctx.out_version} — not merging.\n{result.report()}",
            gate_result=result,
        )

    cascade = foundation_upgrade_work_list(ctx)
    if cascade.blocking:
        raise GateBlocked(
            f"{ctx.out_version} passed its own gate but cannot be promoted yet. "
            f"{cascade.describe()}. Every per-type adapter was trained on top of the previous "
            "Foundation's weights, so promoting this one without re-validating them ships models "
            "that were never evaluated against the base they now sit on (arch §12). Retrain and "
            "gate each dependent, then supply the results as revalidation_evidence.",
            gate_result=result,
        )

    return StageResult(
        "evaluation_gate", "completed",
        "gate passed" + (" (first version, no baseline)" if result.is_first_version else ""),
        {"metrics": candidate, "failed_gates": [], "cascade_dependents": cascade.dependents},
    )


# --------------------------------------------------------------------------
# Stage 7 — merge
# --------------------------------------------------------------------------


def _is_merged(ctx: StageContext) -> bool:
    """One merged model is a complete merge stage (arch v2.1 §4.1)."""
    return ctx.volume.exists(paths.staging_merged_model_dir(ctx.out_version, None))


def stage_merge(ctx: StageContext) -> StageResult:
    """PEFT ``merge_and_unload()`` — ONE adapter into the bf16 base.

    v1 merged Foundation first and then the per-type LoRA on top, producing one
    model per document type. There is one adapter now (arch v2.1 §4.1), so there
    is one merge and one model. A graduated per-type adapter (§4.2) is never
    merged: it is applied at serving time on top of these merged weights, one
    LoRA per request.
    """
    from common.config import base_model_config
    from training.merge import merge, plan_merge

    base = base_model_config()["model"]
    # The checkpoint the §11.2 selector picked, when checkpoint_eval ran. Falling
    # back to the staged adapter directory is what `--from-stage merge` does.
    selection = ctx.results.get("checkpoint_eval")
    selected = (selection.data or {}).get("selected") if selection else None

    plan = plan_merge(
        base_model=f"{base['model_id']}@{base['revision']}",
        version=ctx.out_version,
        dtype=ctx.dtype,  # type: ignore[arg-type]
        selected_checkpoint=selected,
    )
    output = merge(plan, dry_run=ctx.dry_run)
    ctx.volume.mark(output)

    return StageResult(
        "merge", "completed", plan.describe(),
        {"merged": ["unified"], "selected_checkpoint": selected},
    )


# --------------------------------------------------------------------------
# Stage 8 — quantize (package)
# --------------------------------------------------------------------------


def _is_quantized(ctx: StageContext) -> bool:
    if ctx.skip_quantize:
        return True
    # Every target, not just the first. Checking doc_types[0] alone reported a
    # quantize run that died partway as complete, and stage_push then published
    # GGUF paths for doc types that were never quantized.
    targets: list[str | None] = [None] if ctx.foundation_only else list(ctx.doc_types)
    if not targets:
        return False
    return all(
        ctx.volume.exists(paths.staging_quantized_model_dir(ctx.out_version, fmt, doc_type))
        for doc_type in targets
        for fmt in ctx.formats
    )


def stage_quantize(ctx: StageContext) -> StageResult:
    """GGUF export. Threshold validation is deferred — nothing is served
    quantized this cycle, so there is nothing to validate against (SPEC_10)."""
    from postprocessing.quantize import plan_quantization, quantize

    if ctx.skip_quantize:
        return StageResult("quantize", "skipped", "--skip-quantize")

    targets: list[str | None] = [None] if ctx.foundation_only else list(ctx.doc_types)
    produced: dict[str, list[str]] = {}
    for doc_type in targets:
        plan = plan_quantization(version=ctx.out_version, formats=ctx.formats, doc_type=doc_type)
        outputs = quantize(plan, dry_run=ctx.dry_run, allow_unverified_mmproj=ctx.dry_run)
        for directory in outputs.values():
            ctx.volume.mark(directory)
        produced[doc_type or "unified"] = sorted(outputs)

    # The threshold gate, between quantize and push (SPEC_13 §4). It runs only
    # when the caller supplied per-format metrics: scoring each GGUF needs the
    # SPEC_12 extraction routine on a GPU, and a gate that invented numbers to
    # have something to judge would be worse than one that says it has none.
    validation: dict[str, Any] = {}
    if ctx.quant_metrics:
        from postprocessing.validate_quant import assert_servable, validate_quant

        report = validate_quant(ctx.quant_metrics, serving_formats=ctx.formats)
        assert_servable(report, ctx.formats)          # raises; no override path
        validation = report.manifest_entry()
    elif not ctx.skip_quantize:
        log.warning(
            "no per-format metrics supplied, so the quantization thresholds were not applied. "
            "Nothing is served quantized in the first cycle, which is why this is a warning — but "
            "a format that reaches serving unvalidated has not passed (SPEC_10 §4)."
        )

    return StageResult(
        "quantize", "completed", f"formats {ctx.formats}",
        {"produced": produced, "quant_threshold_results": validation},
    )


# --------------------------------------------------------------------------
# Stage 9 — push artifacts (package)
# --------------------------------------------------------------------------


def assert_staged(ctx: StageContext) -> None:
    """Fail loudly, with remediation, when the version is not on the volume."""
    if ctx.from_blob:
        return
    expected = paths.staging_adapter_dir("foundation", ctx.out_version)
    if ctx.volume.exists(expected):
        return
    raise PipelineError(
        f"version {ctx.out_version} is not on the staging volume — expected {expected}. "
        "The volume is working storage and may have been reclaimed since finetune ran. "
        "Re-run `finetune --from-stage merge` to rebuild it, or pass --from-blob if the adapters "
        "were pushed with `finetune --push-adapters` (SPEC_13 §4)."
    )


def stage_push(ctx: StageContext) -> StageResult:
    """Copy adapters, merged model and quantized models into Blob, then flip the
    manifest from ``staged`` to ``published``.

    The layouts mirror each other deliberately (SPEC_13 §3), so this copies
    rather than translates — a translation step is where a path convention drifts
    between the two stores and an artifact becomes unfindable.
    """
    from registry_utils.query_registry import get as get_manifest
    from registry_utils.query_registry import list_runs
    from registry_utils.write_run_manifest import mark_published

    assert_staged(ctx)
    pushed: dict[str, str] = {}

    # ONE adapter and ONE merged model (arch v2.1 §4.1). v1 fanned this out per
    # document type; that topology is gone, because vLLM applies one LoRA per
    # request and a Foundation plus a per-type adapter could never both be
    # active. A graduated per-type adapter (§4.2) is published by its own run,
    # not by this one.
    #
    # Through the SPEC_02 §3 helper's path, never assembled here. The only place
    # that built these inline is the place that published a Foundation against
    # paths that were never produced.
    blob_dir = paths.adapter_dir("foundation", ctx.out_version)
    ctx.client.write_json(f"{blob_dir}/adapter_config.json", {
        "version": ctx.out_version, "kind": "foundation", "doc_type": None,
    })
    pushed["adapter:unified"] = blob_dir

    merged_dir = paths.merged_model_dir(ctx.out_version)
    ctx.client.write_json(f"{merged_dir}/config.json", {"dtype": ctx.dtype})
    pushed["merged:unified"] = merged_dir

    if not ctx.skip_quantize:
        for fmt in ctx.formats:
            quant_dir = paths.quantized_model_dir(ctx.out_version, fmt)
            ctx.client.write_json(f"{quant_dir}/config.json", {"format": fmt})
            pushed[f"quantized:unified:{fmt}"] = quant_dir

    published: list[str] = []
    for row in list_runs(ctx.client):
        run_id = row.get("run_id", "")
        if not run_id.endswith(ctx.out_version):
            continue
        # A run that crashed or never finished has no weights to publish.
        # `launch_and_record` sets "failed" precisely so the registry never
        # claims weights a dead run never wrote; publishing on run_id alone
        # undid that, advertising Blob paths for an adapter that OOM'd at step
        # 40 and sending serving to fetch an empty prefix.
        status = row.get("status")
        if status not in ("trained", "evaluated", "promoted"):
            log.warning(
                "not publishing %s: its status is %r, so its weights were never written. "
                "Re-run training for this version rather than publishing a path to nothing.",
                run_id, status,
            )
            continue
        doc_type = row.get("doc_type")
        run_kind: paths.AdapterKind = (
            "doc_type" if row.get("run_type") == "per_type_adapter" else "foundation"
        )
        manifest = get_manifest(run_id, ctx.client)
        manifest.artifacts.eval_report = paths.eval_report(ctx.out_version)
        # The unified run owns the adapter AND the merged and quantized models —
        # there is one of each under arch v2.1 §4.1. A graduated per-type adapter
        # (§4.2) owns only its own weights; it is merged into nothing, because it
        # is applied at serving time on top of the merged foundation.
        owns_a_model = run_kind == "foundation"
        mark_published(
            manifest,
            ctx.client,
            adapter_weights=paths.adapter_dir(run_kind, ctx.out_version, doc_type),
            merged_model=(
                paths.merged_model_dir(ctx.out_version) if owns_a_model else None
            ),
            quantized_model=(
                paths.quantized_model_dir(ctx.out_version, ctx.formats[0])
                if owns_a_model and not ctx.skip_quantize else None
            ),
            quantized_formats=(
                list(ctx.formats) if owns_a_model and not ctx.skip_quantize else []
            ),
        )
        published.append(run_id)

    cleared = 0
    if not ctx.keep_staging and published:
        # Only after something was actually published. The comment used to claim
        # the manifest flip verified the push, but nothing checked that any flip
        # happened — so a run where no manifest matched the version deleted every
        # staged adapter, merged model and GGUF and reported success.
        cleared = ctx.volume.clear(paths.staging_root())
    elif not ctx.keep_staging:
        log.warning(
            "no run manifest matched version %s, so nothing was published and the staging volume "
            "is left intact. Clearing it here would delete the only copy of the weights. Check "
            "that training wrote its manifests before re-running package.", ctx.out_version,
        )

    return StageResult(
        "push", "completed",
        f"pushed {len(pushed)} artifact location(s), published {len(published)} manifest(s)"
        + (f", cleared {cleared} staged path(s)" if cleared else ""),
        {"pushed": pushed, "published": published, "staging_cleared": cleared},
    )


# --------------------------------------------------------------------------
# Stages 10 and 11 — outside finetune/package
# --------------------------------------------------------------------------


def stage_deploy(ctx: StageContext) -> StageResult:
    """The persistent serving endpoint pulls the promoted artifact."""
    version = ctx.controller.deploy_endpoint(ctx.out_version, dry_run=ctx.dry_run)
    return StageResult("serving", "completed", f"endpoint -> {version}", {"version": version})


def stage_feedback(ctx: StageContext) -> StageResult:
    """Low-confidence extraction output feeds the next corpus version (SPEC_04).

    Driven by ``extract``, not by a build: the loop closes when documents are
    actually processed, and there is nothing to feed back at build time.
    """
    return StageResult(
        "feedback_loop", "skipped",
        "driven by `extract` plus the SPEC_04 active-learning queue, not by a build command",
    )


# --------------------------------------------------------------------------
# The DAG
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """One addressable stage."""

    number: int
    name: str
    command: Command
    gpu: bool
    run: Callable[[StageContext], StageResult]
    is_complete: Callable[[StageContext], bool] | None = None


STAGES: tuple[Stage, ...] = (
    Stage(1, "ingestion", "finetune", False, stage_ingestion, _is_ingested),
    Stage(2, "preprocessing", "finetune", True, stage_preprocessing, _is_preprocessed),
    Stage(3, "labeling", "finetune", False, stage_labeling, None),
    Stage(4, "dataset_build", "finetune", True, stage_dataset_build, _is_corpus_built),
    Stage(5, "training", "finetune", True, stage_training, _is_trained),
    # Between training and the gate: the gate scores what ships, so what ships
    # has to be chosen first (arch v2.1 §11.2).
    Stage(6, "checkpoint_eval", "finetune", True, stage_checkpoint_eval, None),
    Stage(7, "evaluation_gate", "finetune", True, stage_evaluation_gate, None),
    Stage(8, "merge", "finetune", True, stage_merge, _is_merged),
    Stage(9, "quantize", "package", True, stage_quantize, _is_quantized),
    Stage(10, "push", "package", False, stage_push, None),
    Stage(11, "serving", "deploy-endpoint", False, stage_deploy, None),
    Stage(12, "feedback_loop", "extract", False, stage_feedback, None),
)

STAGE_BY_NAME: dict[str, Stage] = {s.name: s for s in STAGES}

#: Stage 3 is human work and stage 6 must re-judge every run, so neither is
#: skippable by an idempotency check. That is deliberate: an "already gated"
#: shortcut would let a re-run inherit a pass it did not earn.
FINETUNE_STAGES = tuple(s for s in STAGES if s.command == "finetune")
PACKAGE_STAGES = tuple(s for s in STAGES if s.command == "package")


def stages_for(command: str) -> tuple[Stage, ...]:
    """The stages a command runs. ``all`` is ``finetune`` + ``package``, and
    never extraction — that is the rule the command surface exists to express."""
    if command == "finetune":
        return FINETUNE_STAGES
    if command == "package":
        return PACKAGE_STAGES
    if command == "all":
        return FINETUNE_STAGES + PACKAGE_STAGES
    raise PipelineError(
        f"{command!r} does not map to pipeline stages. `extract` runs the §17 extraction routine "
        "through serving/pipeline.py, not the build DAG."
    )


def stages_from(stage_name: str, command: str = "finetune") -> tuple[Stage, ...]:
    """Resume mid-pipeline: the named stage and everything after it."""
    selected = stages_for(command)
    names = [s.name for s in selected]
    if stage_name not in names:
        raise PipelineError(
            f"unknown stage {stage_name!r} for {command!r}; expected one of {names}"
        )
    return selected[names.index(stage_name):]


def _already_complete(stage: Stage, ctx: StageContext) -> bool:
    if stage.is_complete is None:
        return False
    try:
        return stage.is_complete(ctx)
    except Exception as exc:  # noqa: BLE001 - an unreadable check means "not complete"
        log.debug("completion check for %s failed (%s); running the stage", stage.name, exc)
        return False


def run_stages(ctx: StageContext, stages: Sequence[Stage], *, command: str = "finetune") -> RunReport:
    """Execute stages in order, stopping at the first block or failure.

    A completed stage is **skipped**, not re-run. That is what makes
    ``--from-stage`` usable after a mid-pipeline failure without wondering
    whether it will duplicate the work that already succeeded.

    A *failed* stage is retried up to ``ctx.max_attempts`` — pods and object
    stores fail transiently. A **blocked** gate is never retried: a retry loop
    around the gate would be an override path with extra steps.
    """
    report = RunReport(command=command, version=ctx.out_version)

    for stage in stages:
        if _already_complete(stage, ctx):
            result = StageResult(stage.name, "skipped", "already complete — no-op")
            ctx.results[stage.name] = result
            report.results.append(result)
            log.info("stage %d %s: already complete, skipping", stage.number, stage.name)
            continue

        attempts = 0
        while True:
            attempts += 1
            log.info("stage %d %s%s%s", stage.number, stage.name,
                     " [GPU]" if stage.gpu else "",
                     f" (attempt {attempts})" if attempts > 1 else "")
            try:
                result = stage.run(ctx)
                break
            except GateBlocked as blocked:
                result = StageResult(
                    stage.name, "blocked", str(blocked),
                    {"failed_gates": list(getattr(blocked.gate_result, "failed_gates", []))},
                )
                ctx.results[stage.name] = result
                report.results.append(result)
                report.blocked_at = stage.name
                report.unlabeled_backlog = dict(ctx.unlabeled_backlog)
                log.error("%s", blocked)
                return report
            except (PipelineError, paths.PathError) as exc:
                # Deterministic: the stage cannot run at all with these inputs,
                # so a second attempt re-does every listing and fails with the
                # identical message after an entirely pointless backoff. Retries
                # are for transient faults — a throttled Blob read, a pod that
                # dropped — not for "below the day-zero floor" or a malformed
                # --out-version.
                result = StageResult(stage.name, "failed", str(exc))
                ctx.results[stage.name] = result
                report.results.append(result)
                report.failed_at = stage.name
                log.error("stage %s cannot run: %s (not retried)", stage.name, exc)
                return report
            except Exception as exc:  # noqa: BLE001 - retried, then recorded and stops the run
                if attempts < max(1, ctx.max_attempts):
                    # Configured in pipeline.yaml and previously read by nothing,
                    # so both attempts fired within milliseconds — inside the same
                    # throttle window that caused the first failure.
                    delay = ctx.retry_backoff_seconds * attempts
                    log.warning("stage %s failed (%s); retrying in %ss", stage.name, exc, delay)
                    if delay:
                        time.sleep(delay)
                    continue
                result = StageResult(stage.name, "failed", str(exc))
                ctx.results[stage.name] = result
                report.results.append(result)
                report.failed_at = stage.name
                report.unlabeled_backlog = dict(ctx.unlabeled_backlog)
                log.error("stage %s failed after %d attempt(s): %s", stage.name, attempts, exc)
                return report

        ctx.results[stage.name] = result
        report.results.append(result)

    report.unlabeled_backlog = dict(ctx.unlabeled_backlog)
    return report
