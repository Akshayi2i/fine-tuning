"""The unified extractor LoRA (arch v2.1 §4.1, §9, §10).

A **thin wrapper**: it assembles an ms-swift configuration and launches it. The
training loop is ms-swift's own ``Seq2SeqTrainer``, built on the Hugging Face
Transformers ``Trainer`` — this file deliberately contains no optimizer step, no
backward pass and no collator (arch v2.1 §10).

*(TRL is installed as an ms-swift dependency for its RLHF trainers. It is not the
SFT loop used here — the v1 documentation said it was, and that was wrong.)*

**One run, not four.** v1 trained a Foundation LoRA and then a per-type LoRA for
each document type, stacked on top. That topology is unservable: vLLM applies one
LoRA per request, so the two could never both be active. It also did not fit the
data — at 25-30 documents per type a rank-16 adapter memorises its own training
set. So: one adapter, every document type, every task, conditioned by the prompt.

What this file owns is the policy the trainer cannot enforce for itself:

* **ViT and mergers frozen.** ``freeze_vit`` and ``freeze_aligner``, and the
  merger is not a LoRA target. Arbitration is learned in the decoder, where image
  tokens and OCR text tokens attend to each other; the mergers never see OCR text.
* **One max_length across a mixed-task corpus**, taken from the largest task cap,
  because ms-swift takes a single value and the corpus interleaves a 4k classify
  example with a 32k policy extraction.
* **A manifest written before launch**, so a pod that dies at step 40 leaves a
  record rather than nothing.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.config import (
    base_model_config,
    sequence_for_task,
    training_config,
    validate_all,
)
from common.run_ids import is_valid_run_id
from common.scopes import Scope, default_scope, get_scope
from common.tasks import Task
from registry_utils.models import (
    Artifacts,
    DataStats,
    Dependencies,
    RunManifest,
    TrainingConfig,
)
from registry_utils.write_run_manifest import capture_git_commit, is_dirty_worktree, write_manifest
from training.base_precision import manifest_descriptor, swift_quantization_args, technique
from training.callbacks.early_stopping import swift_early_stopping_args
from training.corpus_view import materialize

log = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when a run cannot be configured or launched safely."""


@dataclass
class SwiftConfig:
    """The ms-swift invocation. Rendered, inspectable, and recorded.

    Kept as data rather than built inline so a run's exact configuration can be
    logged and diffed — which is what makes "why did this regress" answerable.
    """

    args: dict[str, Any]
    #: Environment for the launch. ms-swift's Qwen-VL processor reads its resize
    #: budget from here, not from a CLI flag.
    env: dict[str, str] = field(default_factory=dict)

    #: Rendered as ``--flag true`` / ``--flag false`` rather than as a bare
    #: presence flag. ms-swift parses these with HfArgumentParser, which accepts
    #: an explicit value — and dropping a False was silently disabling every
    #: option whose correct value is False: ``freeze_vit=False`` (so the flag
    #: never reached the trainer while the manifest recorded that it had),
    #: ``bnb_4bit_use_double_quant``, ``load_best_model_at_end``.
    BOOL_FLAGS = ("true", "false")

    def to_cli(self) -> list[str]:
        argv = ["swift", "sft"]
        for key, value in sorted(self.args.items()):
            if value is None:
                continue
            flag = f"--{key}"
            if isinstance(value, bool):
                argv.extend([flag, "true" if value else "false"])
            elif isinstance(value, (list, tuple)):
                # One argv element per item. Joining them into a single
                # space-separated string gave HfArgumentParser's nargs="+" a
                # one-element list containing every value as one token, so six
                # corpus paths became one nonexistent filename and the LoRA
                # target modules matched nothing at all.
                argv.append(flag)
                argv.extend(str(v) for v in value)
            else:
                argv.extend([flag, str(value)])
        return argv


def corpus_max_length(scope: Scope | None = None) -> int:
    """The single ``max_length`` for a mixed-task corpus.

    ms-swift takes one value and the corpus interleaves every task, so this is
    the **largest** per-task cap (arch v2.1 §7a). Taking anything smaller would
    truncate the longest task rather than reject it, and a clipped assistant span
    trains the model to stop early.

    It is the cap that decides whether a run fits on one 80GB card, which is why
    ``use_logits_to_keep`` and padding-free batching are not optional here.

    **Scoped to what the run actually carries.** Taking the maximum over every
    task and every document type gives 32768 — the policy extraction override —
    to a lossrun-only run whose largest cap is 20480, and that cap is what
    decides the card. A scope narrows both dimensions: its own tasks, and only
    the per-doc-type overrides for types it trains.
    """
    from common.config import shared_sequence_config

    declared = shared_sequence_config().get("tasks", {})
    tasks = tuple(scope.tasks) if scope else tuple(Task)
    doc_types = set(scope.doc_types) if scope else None

    caps: list[int] = []
    for task in tasks:
        name = str(task)
        caps.append(int(sequence_for_task(name)["max_seq_len"]))
        # Per-doc-type overrides are separate caps, not variations on one.
        # Iterating only the bare tasks gave 24576 while the policy extraction
        # override is 32768, so every routed policy would have been TRUNCATED —
        # which is the exact failure the §7a caps exist to prevent.
        for doc_type in (declared.get(name, {}).get("by_doc_type") or {}):
            if doc_types is not None and doc_type not in doc_types:
                continue
            caps.append(int(sequence_for_task(name, doc_type)["max_seq_len"]))
    return max(caps)


def build_training_config(
    *,
    corpus_paths: list[str],
    output_dir: str,
    val_paths: list[str] | None = None,
    train_vit: bool = False,
    deepspeed: str | None = None,
    resume_from: str | None = None,
    config_name: str | None = None,
    scope: Scope | None = None,
) -> tuple[SwiftConfig, TrainingConfig]:
    """Assemble the ms-swift arguments and the manifest's record of them.

    Both come from the same source so the manifest describes what actually ran,
    rather than what a YAML file happened to say afterwards.

    ``corpus_paths`` are the materialized epoch files, in order. The run reads the
    first ``num_train_epochs`` of them **once each** — see the epoch comment
    below for why ms-swift is told one epoch.
    """
    base = base_model_config()
    scope = scope or default_scope()
    cfg = training_config(config_name or scope.training_config)
    lora, opt, batch = cfg["lora"], cfg["optimization"], cfg["batch"]
    evaluation, memory = cfg["evaluation"], cfg.get("memory", {})

    epochs = int(opt["num_train_epochs"])
    if len(corpus_paths) < epochs:
        raise TrainingError(
            f"the config asks for {epochs} epochs but only {len(corpus_paths)} epoch file(s) were "
            "given. Each epoch is its own file with its own modality draw (arch v2.1 §6.1); "
            "reusing a file would show every document in the same regime twice."
        )

    target_modules = list(lora["target_modules"])
    if lora.get("include_vision_projector", False):
        # Off by default under v2.1 §9a. Reachable only through the vision
        # ablation, and a vision-module LoRA is merged into the base rather than
        # hot-swapped — tower/connector LoRA in vLLM is experimental.
        target_modules.append("merger")

    max_length = corpus_max_length(scope)

    args: dict[str, Any] = {
        "model_type": "qwen3-vl-8b-instruct",
        "model_id_or_path": base["model"]["model_id"],
        "model_revision": base["model"]["revision"],
        # The first N epoch files, read once each. Every file holds every train
        # document, in that epoch's modality draw, so the concatenation has the
        # CONTENT of an N-epoch run: each document N times, in N regimes. Not its
        # ORDER — the Trainer shuffles the concatenated rows and groups them by
        # length, so a document's passes can land next to each other, and an
        # early stop can come after some documents have been seen more often
        # than others. Over a run that averages out; within one it does not.
        "dataset": list(corpus_paths[:epochs]),
        "output_dir": output_dir,
        "train_type": "lora",
        "lora_rank": lora["rank"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "lora_target_modules": target_modules,
        "use_rslora": bool(lora.get("use_rslora", False)),
        # bf16 frozen base by default; 4-bit NF4 only when the config asks for it
        # (arch §9). One helper shared with any future trainer so they cannot
        # disagree about how the base is held.
        **swift_quantization_args(base),
        "attn_impl": base["attention"]["attn_implementation"],
        "learning_rate": opt["learning_rate"],
        "lr_scheduler_type": opt["lr_scheduler_type"],
        # Steps, not a ratio: at pilot volume a 0.03 ratio over a handful of
        # steps rounds to zero warmup, and the first optimizer step then lands at
        # full learning rate on a freshly-initialised adapter.
        "warmup_steps": opt["warmup_steps"],
        # One pass over the concatenated epoch files, never N. Telling ms-swift
        # N here as well loops the N files N times — nine passes for a "3 epoch"
        # run, the inflation per-epoch sampling exists to remove. The logical
        # epoch count is recorded on the manifest below.
        "num_train_epochs": 1,
        "optim": opt["optim"],
        "weight_decay": opt["weight_decay"],
        "max_grad_norm": opt["max_grad_norm"],
        "per_device_train_batch_size": batch["per_device_train_batch_size"],
        "gradient_accumulation_steps": batch["gradient_accumulation_steps"],
        "gradient_checkpointing": batch["gradient_checkpointing"],
        "bf16": batch["bf16"],
        "max_length": max_length,
        # Explicit, never left to ms-swift's default. Every row was checked
        # against its task budget at corpus build (cap_check), so a row over
        # max_length here means that estimate was wrong — and the answer is to
        # drop it, not to cut it: "left" would remove the system prompt, "right"
        # the end of the target, and a clipped target trains the model to stop
        # early. What was dropped shows in the ms-swift log.
        "truncation_strategy": "delete",
        # ms-swift tokenizes multimodal rows lazily, and a row that fails to
        # encode is by default REPLACED with a randomly chosen other row: the
        # document is lost, another is trained twice, and nothing records it.
        # Strict turns that into a hard error. With every row checked against a
        # pessimistic budget at corpus build it should never fire; if it does,
        # the estimate is wrong and a run that says so beats one that hides it.
        "strict": True,
        # The three settings that make the largest task cap affordable (§9.3).
        # Without use_logits_to_keep the LM head produces a 151k-vocabulary
        # distribution at every position of a 32k sequence, which dominates
        # activation memory on its own.
        "use_logits_to_keep": bool(memory.get("use_logits_to_keep", True)),
        "padding_free": bool(memory.get("padding_free", True)),
        "length_grouped_sampling": bool(memory.get("length_grouped_sampling", True)),
        # Frozen, both of them (arch v2.1 §9.4). `train_vit` is the §3 escalation
        # and remains LoRA-on-ViT, never a full fine-tune.
        "freeze_vit": not train_vit,
        "freeze_aligner": True,
        "eval_strategy": evaluation["eval_strategy"],
        "eval_steps": evaluation["eval_steps"],
        "save_steps": evaluation["save_steps"],
        "save_total_limit": evaluation["save_total_limit"],
        # Unpacked FIRST so the explicit keys above win. Unpacking it last
        # silently overrode this config's metric_for_best_model and
        # load_best_model_at_end with the helper's own defaults.
        **swift_early_stopping_args(
            int(evaluation.get("early_stopping_patience", 3)),
            metric_for_best_model=evaluation["metric_for_best_model"],
            greater_is_better=bool(evaluation.get("greater_is_better", False)),
            load_best_model_at_end=bool(evaluation["load_best_model_at_end"]),
        ),
        "logging_steps": cfg["logging"]["logging_steps"],
        "seed": cfg["seed"],
    }

    distributed = cfg.get("distributed", {})
    if deepspeed:
        args["deepspeed"] = f"configs/deepspeed/{deepspeed}.json"
    if int(distributed.get("sequence_parallel_size", 1)) > 1:
        # Reached before any ZeRO config when a task cap does not fit (§9.3).
        args["sequence_parallel_size"] = distributed["sequence_parallel_size"]
    if val_paths:
        args["val_dataset"] = val_paths
    if resume_from:
        args["resume_from_checkpoint"] = resume_from

    recorded = TrainingConfig(
        # Derived from the config that actually ran, never defaulted. These three
        # carried QLoRA/NF4/paged-8bit defaults on the model, so a bf16 run that
        # did not pass them recorded a technique it never used — in the one
        # record the whole reproducibility story rests on.
        technique=technique(base),
        base_quantization=manifest_descriptor(base),
        lora_rank=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        learning_rate=opt["learning_rate"],
        lr_scheduler=opt["lr_scheduler_type"],
        warmup_ratio=0.0,
        epochs=opt["num_train_epochs"],
        optimizer=opt["optim"],
        adam_beta1=opt["adam_beta1"],
        adam_beta2=opt["adam_beta2"],
        adam_epsilon=opt["adam_epsilon"],
        weight_decay=opt["weight_decay"],
        max_grad_norm=opt["max_grad_norm"],
        per_device_batch_size=batch["per_device_train_batch_size"],
        gradient_accumulation_steps=batch["gradient_accumulation_steps"],
        effective_batch_size=batch["effective_batch_size"],
        gradient_checkpointing=batch["gradient_checkpointing"],
        mixed_precision="bf16" if batch["bf16"] else "fp32",
        target_modules=target_modules,
        vit_trainable=train_vit,
        vit_method="lora" if train_vit else "frozen",
        resolution_cap_px=base["vision"]["max_image_long_side_px"],
        max_seq_len=max_length,
        seed=cfg["seed"],
    )
    return SwiftConfig(args, env=_pixel_budget(scope)), recorded


def _pixel_budget(scope: Scope) -> dict[str, str]:
    """The image resize budget ms-swift's Qwen-VL processor reads from the env.

    The same per-task ``max_pixels`` the corpus was sized against (cap_check) and
    serving uses. Left to the processor's default, training would resize pages to
    a budget nobody chose — today the rendered pages happen to sit inside it, so
    nothing differs, but the day a render or a vision budget changes, training
    and serving would see different pixels with nothing to say so.
    """
    from common.config import vision_for_task
    from common.tasks import FULL_RESOLUTION_TASKS

    tasks = [task for task in scope.tasks if task in FULL_RESOLUTION_TASKS] or [Task.EXTRACT]
    budgets = [vision_for_task(str(task)) for task in tasks]
    return {
        "MAX_PIXELS": str(max(int(b["max_pixels"]) for b in budgets)),
        "MIN_PIXELS": str(min(int(b["min_pixels"]) for b in budgets)),
    }


def build_manifest(
    *,
    run_id: str,
    corpus_version: str,
    corpus_manifest: dict[str, Any],
    training_cfg: TrainingConfig,
    data_stats: DataStats,
    staging_path: str,
    continued_from: str | None = None,
    scope: Scope | None = None,
) -> RunManifest:
    """Build the run manifest. Written to Blob even while weights are staged."""
    base = base_model_config()["model"]
    if is_dirty_worktree():
        log.warning(
            "the working tree has uncommitted changes, so code_git_commit does not fully "
            "describe what ran — this run is not exactly reproducible."
        )

    scope = scope or default_scope()
    return RunManifest(
        run_id=run_id,
        # A unified-scope run keeps writing "unified", so extractor-v1 and
        # extractor-v2 remain the same kind of thing in the registry. Anything
        # narrower is "scoped" and says what it covers.
        run_type="unified" if scope.is_unified else "scoped",
        scope=None if scope.is_unified else scope.name,
        doc_types=[] if scope.is_unified else list(scope.doc_types),
        continued_from=continued_from,
        dependencies=Dependencies(
            base_model=f"{base['model_id']}@{base['revision']}",
            corpus_version=corpus_version,
            code_git_commit=capture_git_commit(),
            mineru_version=corpus_manifest.get("mineru_version"),
            ocr_device=corpus_manifest.get("ocr_device"),
            schema_version=str(corpus_manifest.get("schema_versions", {})),
            prompt_template_version=corpus_manifest.get("prompt_template_version"),
        ),
        training_config=training_cfg,
        data_stats=data_stats,
        artifacts=Artifacts(status="staged", staging_path=staging_path),
        status="training",
    )


def train(
    *,
    corpus_version: str,
    out_version: str,
    client: BlobClient,
    corpus_manifest: dict[str, Any],
    data_stats: DataStats,
    train_vit: bool = False,
    deepspeed: str | None = None,
    continue_from: str | None = None,
    dry_run: bool = False,
    tenant_id: str | None = None,
    scope: Scope | None = None,
) -> tuple[SwiftConfig, RunManifest]:
    """Configure and launch one training run.

    Args:
        continue_from: **a filesystem checkpoint path**, not a registry run-id.
            It goes straight to ms-swift's ``resume_from_checkpoint``, which reads
            a directory — passing a run-id like ``extractor-v3`` produced a run
            that silently trained from base while its manifest recorded a lineage
            that never happened. ``assert_checkpoint_path`` refuses the run-id
            shape rather than letting it through.
        dry_run: assemble and record everything without launching.
        tenant_id: whose corpus to read. Omitting it reads the default tenant's
            files, which for any other tenant do not exist — or worse, do.
        scope: what this run covers (``common.scopes``). Defaults to the unified
            scope, which trains every document type and writes exactly the run
            id, staging path and manifest shape it always did.
    """
    validate_all(require_pinned_revision=not dry_run)

    # Train and val are kept apart. Passing both to `--dataset` made ms-swift
    # treat the validation split as training data and then carve its own eval
    # split out of the union, so the selected checkpoint was chosen on documents
    # the model had memorised — and the promotion gate read that number.
    scope = scope or default_scope()
    # The corpus is built once for every type; a narrower scope reads a filtered
    # VIEW of it rather than a corpus of its own, so the split and the group
    # assignment are shared and the two runs stay comparable.
    view = materialize(scope, corpus_version, client, tenant_id=tenant_id)
    corpus_paths = list(view.epoch_files)
    val_paths = [view.val_path]
    staging = paths.scoped_staging_adapter_dir(scope.name, out_version)

    if not scope.is_unified:
        # The caller counts the whole corpus. A scoped run trains on its view,
        # and its manifest has to say so: a lossrun run recording every policy
        # row as its training data describes a run that did not happen.
        data_stats = data_stats.model_copy(update={
            "train_examples": view.train_rows,
            "val_examples": view.val_rows,
            "test_examples": view.test_rows,
        })

    # Every check that costs nothing runs before staging, which downloads every
    # page image the run reads. A run that is going to be refused should be
    # refused before that, not after thousands of images reach a paid pod.
    if continue_from:
        assert_checkpoint_path(continue_from)
    val_rows = view.val_rows if not scope.is_unified else _count_rows(client, view.val_path)
    if not dry_run and not val_rows:
        # The config asks for evaluation, early stopping and
        # load_best_model_at_end, and checkpoint selection generates on
        # validation: with no rows the run fails after the GPU is paid for, or
        # trains with nothing to choose its checkpoint by.
        raise TrainingError(
            f"scope {scope.name!r} has no validation rows in corpus {corpus_version}. "
            "Checkpoint selection and early stopping both read validation, so the run "
            "cannot choose what to ship. Rebuild the corpus with enough documents of "
            f"{list(scope.doc_types)} to fill a validation split."
        )

    # Built against the Blob keys first: this is where the epoch count and every
    # other configuration error surfaces, still before any download.
    swift, recorded = build_training_config(
        corpus_paths=corpus_paths,
        val_paths=val_paths,
        output_dir=staging,
        train_vit=train_vit,
        deepspeed=deepspeed,
        resume_from=continue_from,
        scope=scope,
    )

    if not dry_run:
        # ms-swift reads LOCAL files in ITS row format. The view is Blob keys in
        # the corpus's own shape — handed over as-is, the trainer finds nothing,
        # and a row that did load would fail on its mixed-type content column.
        # A dry run launches nothing, so it records the Blob keys instead.
        from training.stage_data import stage_training_data

        epochs = len(swift.args["dataset"])
        staged = stage_training_data(
            view.epoch_files[:epochs], view.val_path, client,
            paths.staging_train_data_dir(scope.name, out_version),
            images_root=paths.staging_train_images_dir(corpus_version, tenant_id),
        )
        swift.args["dataset"] = staged.epoch_files
        swift.args["val_dataset"] = [staged.val_path]

    # Only the epochs this run uses. Four files are always materialized because
    # the §11a sweep tests up to four passes (§6.1), and a sweep that regenerates
    # its own data is not comparing what it thinks it is.
    manifest = build_manifest(
        run_id=scope.run_id(out_version),
        corpus_version=corpus_version,
        corpus_manifest=corpus_manifest,
        training_cfg=recorded,
        data_stats=data_stats,
        staging_path=staging,
        continued_from=continue_from,
        scope=scope,
    )
    write_manifest(manifest, client)

    if dry_run:
        log.info("dry run — configuration recorded, nothing launched")
        return swift, manifest

    manifest = launch_and_record(swift, manifest, client)
    return swift, manifest


def _count_rows(client: BlobClient, key: str) -> int:
    """Rows in a Blob JSONL file, without staging it. Zero when it is absent."""
    if not client.exists(key):
        return 0
    return sum(1 for line in client.read_text(key).splitlines() if line.strip())


def assert_checkpoint_path(continue_from: str) -> None:
    """Refuse a registry run-id where a checkpoint directory is required.

    ms-swift's ``resume_from_checkpoint`` reads a directory. Handed a run-id it
    finds nothing, trains from base, and the manifest records a lineage that
    never happened — a run that claims to continue v2 while being v1 again.

    The shape comes from :mod:`common.run_ids`, so a scoped id (``policy-v2``) is
    refused for the same reason ``extractor-v2`` is. The three-alternative regex
    this replaced listed the lineages v1 happened to mint, and would have waved
    every scoped id straight through.
    """
    if is_valid_run_id(continue_from.strip().rstrip("/").split("/")[-1]) and "/" not in continue_from:
        raise TrainingError(
            f"--continue-from got {continue_from!r}, which is a registry run-id, not a "
            "checkpoint path. ms-swift resumes from a DIRECTORY; given a run-id it silently "
            "trains from base while the manifest records a lineage that never happened. "
            "Pass the staged checkpoint directory instead."
        )


def count_examples(bucket: Any) -> int:
    """Sum a ``{doc_type: {mode: count}}`` bucket from the corpus manifest.

    The manifest nests counts by type and modality. ``int()`` on that dict is a
    TypeError, which is how the CLI used to fail before training started.
    """
    if not isinstance(bucket, dict):
        return 0
    return sum(
        count
        for modes in bucket.values()
        if isinstance(modes, dict)
        for count in modes.values()
    )


def launch_and_record(
    config: SwiftConfig,
    manifest: RunManifest,
    client: BlobClient,
) -> RunManifest:
    """Run training, then record what actually happened.

    The manifest is written before this is called — the run_id has to be reserved
    and the configuration captured even for a run that dies. But "written" is not
    "trained": the status stays ``training`` until ms-swift returns, and becomes
    ``failed`` if it does not, so the registry never claims weights that a
    crashed run never wrote.
    """
    try:
        launch(config)
    except Exception as exc:
        manifest.status = "failed"
        write_manifest(manifest, client)
        log.error("training failed (%s); manifest %s marked failed", exc, manifest.run_id)
        raise
    manifest.status = "trained"
    write_manifest(manifest, client)
    return manifest


def launch(config: SwiftConfig) -> None:
    """Launch ms-swift. The training loop lives there, not here (arch v2.1 §10)."""
    import shutil
    import subprocess

    if shutil.which("swift") is None:
        raise TrainingError(
            "the `swift` CLI is not on PATH. ms-swift is the Layer-3 entrypoint (arch v2.1 §10); "
            'install the [train] extra on the pod: pip install -e ".[train]"\n'
            "If ms-swift turns out to lack a Qwen3-VL capability this needs, the documented "
            "fallback is a framework migration to TRL SFTTrainer — including re-implementing "
            "the template, collator and masking against the §10.2 parity tests. A contingency, "
            "not a layer swap."
        )
    import os

    argv = config.to_cli()
    log.info("launching: %s (env %s)", " ".join(argv), config.env)
    subprocess.run(argv, check=True, env={**os.environ, **config.env})


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Train the unified extractor LoRA")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out-version", required=True)
    parser.add_argument("--deepspeed", default=None, choices=[None, "zero2", "zero3"])
    parser.add_argument("--train-vit", action="store_true", help="the §3 escalation; LoRA-on-ViT")
    parser.add_argument("--continue-from", default=None, help="a checkpoint DIRECTORY, not a run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tenant", default=None, help="whose corpus to read; defaults from env")
    parser.add_argument("--scope", default=None,
                        help="what this run covers (configs/scopes.yaml); defaults to unified")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = BlobClient()
    corpus_manifest = client.read_json(paths.corpus_manifest(args.corpus, args.tenant))
    counts = corpus_manifest.get("example_counts", {})
    train(
        corpus_version=args.corpus,
        out_version=args.out_version,
        client=client,
        corpus_manifest=corpus_manifest,
        data_stats=DataStats(
            train_examples=count_examples(counts.get("train")),
            val_examples=count_examples(counts.get("val")),
            test_examples=count_examples(counts.get("test")),
        ),
        train_vit=args.train_vit,
        deepspeed=args.deepspeed,
        continue_from=args.continue_from,
        dry_run=args.dry_run,
        tenant_id=args.tenant,
        scope=get_scope(args.scope) if args.scope else None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
