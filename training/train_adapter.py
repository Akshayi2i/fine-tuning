"""Per-type adapter training (SPEC_06, arch §4, §12).

A small LoRA (rank 16) trained on one document type's slice, on top of the
current promoted Foundation. It learns only the final schema-mapping
specialisation for that type; everything shared — terminology, table reading,
OCR-versus-image arbitration, canonical field mapping — is the Foundation's job.

**Always fresh from the Foundation. Never continued from its own last
checkpoint** (arch §12). Per-type adapters are small and cheap, so incremental
training buys nothing, while starting fresh from a fixed, well-evaluated
Foundation stops small errors stacking across cycles. It also gives every adapter
version a dependency on exactly one Foundation version, which is what makes "why
did this document type regress" answerable instead of archaeological.

**ACORD is one adapter, not three** (arch §4b). With limited data per form, one
adapter learning shared "ACORD-ness" generalises better than three data-starved
ones. The per-form *schemas* still exist, because the classifier has to select
the right one regardless.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.config import base_model_config, training_config, validate_all
from common.constants import ACTIVE_DOC_TYPES
from registry_utils.models import Artifacts, DataStats, Dependencies, RunManifest, TrainingConfig
from registry_utils.query_registry import latest_promoted
from registry_utils.write_run_manifest import capture_git_commit, write_manifest
from training.callbacks.early_stopping import swift_early_stopping_args
from training.train_foundation import SwiftConfig, TrainingError, launch_and_record

log = logging.getLogger(__name__)


def resolve_foundation(client: BlobClient, explicit: str | None = None) -> str:
    """The Foundation this adapter will be built on.

    Resolved from the registry at launch rather than pinned in config, so an
    adapter can never be built against a stale Foundation because someone forgot
    to update a YAML file.
    """
    if explicit:
        return explicit
    promoted = latest_promoted(client, "foundation")
    if not promoted:
        raise TrainingError(
            "no promoted Foundation exists, so there is nothing to train an adapter on top of. "
            "Train and promote a Foundation first — a per-type adapter is a specialisation of "
            "one, not a standalone model (arch §4)."
        )
    return promoted


def _foundation_checkpoint(client: BlobClient, run_id: str, tag: str) -> str:
    """The Foundation's weights, from wherever its manifest says they are.

    A *staged* Foundation lives on the volume; a *published* one lives in Blob.
    The registry already records which, so this reads it rather than assuming.
    """
    from registry_utils.query_registry import RegistryQueryError, get

    try:
        manifest = get(run_id, client)
    except (RegistryQueryError, KeyError, FileNotFoundError) as exc:
        # No fallback. Constructing a staging path here meant an explicitly
        # requested --foundation with no manifest silently trained against a
        # directory nobody had checked, re-opening the bug the caller's comment
        # claims to have closed. The registry is what knows where weights live.
        raise TrainingError(
            f"no run manifest for {run_id}, so where its weights live is unknown — staged on the "
            "volume or published in Blob are different paths, and guessing produces an adapter "
            f"trained against something nobody verified. Train and register it first ({exc})."
        ) from exc

    if manifest.artifacts.status == "staged":
        return manifest.artifacts.staging_path or paths.staging_adapter_dir("foundation", tag)
    return manifest.artifacts.adapter_weights or paths.adapter_dir("foundation", tag)


def build_adapter_config(
    doc_type: str,
    *,
    corpus_version: str,
    foundation_adapter_path: str,
    output_dir: str,
    deepspeed: str = "zero2",
) -> tuple[SwiftConfig, TrainingConfig]:
    """Assemble the ms-swift arguments for a per-type adapter."""
    base = base_model_config()
    cfg = training_config(f"{doc_type}_adapter")
    lora, opt, batch, evaluation = cfg["lora"], cfg["optimization"], cfg["batch"], cfg["evaluation"]

    target_modules = list(lora["target_modules"])
    if lora.get("include_vision_projector", True):
        target_modules.append("merger")

    args: dict[str, Any] = {
        "model_type": "qwen3-vl-8b-instruct",
        "model_id_or_path": base["model"]["model_id"],
        "model_revision": base["model"]["revision"],
        # The Foundation is loaded and FROZEN; the new LoRA trains on top of it.
        #
        # `adapters` — not `resume_from_checkpoint`. Resume means "continue THIS
        # run": ms-swift would restore optimizer state and the completed
        # global_step, so a fresh 3-epoch adapter run resumes at the end of the
        # Foundation's schedule and trains zero steps. It would also try to load
        # rank-64 Foundation weights into this run's rank-16 LoRA config.
        "adapters": [foundation_adapter_path],
        "train_type": "lora",
        # Train only. Passing val here makes ms-swift treat the validation split
        # as training data and carve its own eval split out of the union, so
        # `metric_for_best_model` selects on memorised documents — and the
        # promotion gate reads that number. Fixed in train_foundation and missed
        # here on the first pass.
        "dataset": [paths.corpus_split(corpus_version, doc_type, "train")],
        "val_dataset": [paths.corpus_split(corpus_version, doc_type, "val")],
        "output_dir": output_dir,
        "sft_type": "lora",
        "lora_rank": lora["rank"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "lora_target_modules": target_modules,
        "quantization_bit": 4,
        "bnb_4bit_quant_type": base["quantization"]["bnb_4bit_quant_type"],
        "bnb_4bit_use_double_quant": base["quantization"]["bnb_4bit_use_double_quant"],
        "bnb_4bit_compute_dtype": base["quantization"]["bnb_4bit_compute_dtype"],
        "attn_impl": base["attention"]["attn_implementation"],
        "learning_rate": opt["learning_rate"],
        "lr_scheduler_type": opt["lr_scheduler_type"],
        "warmup_ratio": opt["warmup_ratio"],
        "num_train_epochs": opt["num_train_epochs"],
        "optim": opt["optim"],
        "weight_decay": opt["weight_decay"],
        "max_grad_norm": opt["max_grad_norm"],
        "per_device_train_batch_size": batch["per_device_train_batch_size"],
        "gradient_accumulation_steps": batch["gradient_accumulation_steps"],
        "gradient_checkpointing": batch["gradient_checkpointing"],
        "bf16": batch["bf16"],
        "max_length": base["sequence"]["max_seq_len"],
        "eval_strategy": evaluation["eval_strategy"],
        "eval_steps": evaluation["eval_steps"],
        "save_steps": evaluation["save_steps"],
        # The helper is unpacked FIRST so the explicit keys below win. Unpacking
        # it last silently overrode this config's metric_for_best_model and
        # load_best_model_at_end with the helper's own defaults.
        **swift_early_stopping_args(int(evaluation.get("early_stopping_patience", 2))),
        "metric_for_best_model": evaluation["metric_for_best_model"],
        # Both were declared in YAML and dropped here, so the best checkpoint was
        # identified and then thrown away — a 4-epoch adapter that peaked at step
        # 150 and overfit by step 400 was staged at step 400 — and every
        # checkpoint was retained on a finite volume.
        "save_total_limit": evaluation["save_total_limit"],
        "load_best_model_at_end": evaluation["load_best_model_at_end"],
        "logging_steps": cfg["logging"]["logging_steps"],
        "seed": cfg["seed"],
        "deepspeed": f"configs/deepspeed/{deepspeed}.json",
        "freeze_vit": True,   # never escalated at the per-type layer
    }

    recorded = TrainingConfig(
        lora_rank=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        learning_rate=opt["learning_rate"],
        lr_scheduler=opt["lr_scheduler_type"],
        warmup_ratio=opt["warmup_ratio"],
        epochs=opt["num_train_epochs"],
        optimizer=opt["optim"],
        weight_decay=opt["weight_decay"],
        max_grad_norm=opt["max_grad_norm"],
        per_device_batch_size=batch["per_device_train_batch_size"],
        gradient_accumulation_steps=batch["gradient_accumulation_steps"],
        effective_batch_size=batch["effective_batch_size"],
        gradient_checkpointing=batch["gradient_checkpointing"],
        mixed_precision="bf16" if batch["bf16"] else "fp32",
        target_modules=target_modules,
        vit_trainable=False,
        vit_method="frozen",
        resolution_cap_px=base["vision"]["max_image_long_side_px"],
        max_seq_len=base["sequence"]["max_seq_len"],
        seed=cfg["seed"],
    )
    return SwiftConfig(args), recorded


def train_adapter(
    doc_type: str,
    *,
    corpus_version: str,
    out_version: str,
    client: BlobClient,
    corpus_manifest: dict[str, Any],
    data_stats: DataStats,
    foundation_version: str | None = None,
    deepspeed: str = "zero2",
    dry_run: bool = False,
) -> tuple[SwiftConfig, RunManifest]:
    """Configure and launch a per-type adapter run."""
    if doc_type not in ACTIVE_DOC_TYPES:
        raise TrainingError(f"unknown doc_type {doc_type!r}; expected one of {list(ACTIVE_DOC_TYPES)}")

    validate_all(require_pinned_revision=not dry_run)
    foundation = resolve_foundation(client, foundation_version)
    foundation_tag = foundation.replace("foundation-", "")
    # Where that Foundation's weights actually are. Building the path from the
    # staging volume unconditionally pointed at a reclaimable working directory
    # for a *promoted* Foundation whose artifacts are in Blob — so on a fresh
    # pod the checkpoint was missing and the adapter trained from base while its
    # manifest claimed the Foundation as a hard dependency.
    foundation_path = _foundation_checkpoint(client, foundation, foundation_tag)

    staging = paths.staging_adapter_dir("doc_type", out_version, doc_type)
    swift, recorded = build_adapter_config(
        doc_type,
        corpus_version=corpus_version,
        foundation_adapter_path=foundation_path,
        output_dir=staging,
        deepspeed=deepspeed,
    )

    base = base_model_config()["model"]
    manifest = RunManifest(
        run_id=f"{doc_type}-adapter-{out_version}",
        run_type="per_type_adapter",
        doc_type=doc_type,
        dependencies=Dependencies(
            base_model=f"{base['model_id']}@{base['revision']}",
            corpus_version=corpus_version,
            code_git_commit=capture_git_commit(),
            # Recorded as a hard dependency: this is what turns the arch §12
            # Foundation-upgrade rule into a query rather than a manual audit.
            foundation_version=foundation,
            mineru_version=corpus_manifest.get("mineru_version"),
            ocr_device=corpus_manifest.get("ocr_device"),
            prompt_template_version=corpus_manifest.get("prompt_template_version"),
        ),
        training_config=recorded,
        data_stats=data_stats,
        artifacts=Artifacts(status="staged", staging_path=staging),
    )
    # `training`, not `trained` — see launch_and_record. The status is what a
    # later `package` reads to decide whether weights exist to merge.
    manifest.status = "training"
    write_manifest(manifest, client)

    log.info("training %s adapter on %s (fresh, never continued from a previous adapter)",
             doc_type, foundation)

    if dry_run:
        return swift, manifest
    launch_and_record(swift, manifest, client)
    return swift, manifest


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Train a per-document-type LoRA adapter")
    parser.add_argument("--doc-type", required=True, choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out-version", required=True)
    parser.add_argument("--foundation", default=None,
                        help="default: the currently promoted Foundation")
    parser.add_argument("--acord-form", default=None,
                        help="reserved for the future per-form split; one shared ACORD adapter "
                             "is trained today (arch §4b)")
    parser.add_argument("--deepspeed", choices=["zero2", "zero3"], default="zero2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.acord_form:
        log.warning(
            "--acord-form is reserved: one shared ACORD adapter covers forms 25/125/140 until a "
            "form has 1000+ examples AND eval shows the shared adapter underperforming (arch §4b)."
        )

    client = BlobClient()
    corpus_manifest = client.read_json(paths.corpus_manifest(args.corpus))
    train_adapter(
        args.doc_type,
        corpus_version=args.corpus,
        out_version=args.out_version,
        client=client,
        corpus_manifest=corpus_manifest,
        data_stats=DataStats(train_examples=0, val_examples=0, test_examples=0),
        foundation_version=args.foundation,
        deepspeed=args.deepspeed,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
