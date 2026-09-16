"""Foundation LoRA training (SPEC_06, arch §4, §9, §10, §12).

A **thin wrapper**: it assembles an ms-swift configuration and launches it. The
training loop is TRL's ``SFTTrainer``, which ms-swift constructs — this file
deliberately contains no optimizer step, no backward pass, and no collator
(arch §10).

What it *does* own is the policy the trainer cannot enforce for itself:

* **Trains on the mixed corpus** — all document types, all three modality
  regimes. That mix is what makes the Foundation learn shared behaviour:
  insurance terminology, table and checkbox reading, OCR-versus-image
  arbitration, JSON discipline, and the canonical field mapping.
* **ViT frozen**, and when escalated it is a LoRA on the ViT, never a full
  fine-tune (arch §3).
* **Major expansion retrains from the HF base**, because continuing on top of a
  previous LoRA compounds drift across cycles (arch §12).
* **Writes a run manifest to Blob** even though the weights stay staged, so a
  reclaimed volume never means a training run that left no trace.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.config import base_model_config, training_config, validate_all
from common.constants import ACTIVE_DOC_TYPES
from registry_utils.models import (
    Artifacts,
    DataStats,
    Dependencies,
    RunManifest,
    TrainingConfig,
)
from registry_utils.write_run_manifest import capture_git_commit, is_dirty_worktree, write_manifest
from training.base_precision import (
    manifest_descriptor,
    swift_quantization_args,
    technique,
)
from training.callbacks.early_stopping import swift_early_stopping_args

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

    #: Rendered as ``--flag true`` / ``--flag false`` rather than as a bare
    #: presence flag. ms-swift parses these with HfArgumentParser, which accepts
    #: an explicit value — and dropping a False was silently disabling every
    #: option whose correct value is False: ``freeze_vit=False`` (so ``--train-vit``
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


def build_swift_config(
    *,
    corpus_paths: list[str],
    output_dir: str,
    val_paths: list[str] | None = None,
    train_vit: bool = False,
    deepspeed: str = "zero2",
    resume_from: str | None = None,
) -> tuple[SwiftConfig, TrainingConfig]:
    """Assemble the ms-swift arguments and the manifest's record of them.

    Both come from the same source so the manifest describes what actually ran,
    rather than what a YAML file happened to say afterwards.
    """
    base = base_model_config()
    cfg = training_config("foundation")
    lora, opt, batch, evaluation = cfg["lora"], cfg["optimization"], cfg["batch"], cfg["evaluation"]

    model_id = base["model"]["model_id"]
    revision = base["model"]["revision"]

    target_modules = list(lora["target_modules"])
    if lora.get("include_vision_projector", True):
        # The projector is where image evidence fuses with language — the locus
        # of OCR-versus-image arbitration, and the highest-leverage target after
        # the decoder itself (arch §9a).
        target_modules.append("merger")

    args: dict[str, Any] = {
        "model_type": "qwen3-vl-8b-instruct",
        "model_id_or_path": model_id,
        "model_revision": revision,
        "dataset": corpus_paths,
        "output_dir": output_dir,
        "sft_type": "lora",
        "lora_rank": lora["rank"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "lora_target_modules": target_modules,
        # bf16 frozen base by default; 4-bit NF4 only when the config asks for it
        # (arch §9). One helper for both trainers so they cannot disagree.
        **swift_quantization_args(base),
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
        "save_total_limit": evaluation["save_total_limit"],
        # These come from the helper above, which now receives them from the
        # YAML. Setting them here as well and letting the helper's unpack
        # override them is the ordering bug that was fixed in train_adapter and
        # missed here — the same fixed-one-of-two mistake as the val-leak.
        # Configured in YAML and previously read by nothing, so a run trained
        # every step regardless of a field-F1 plateau. `swift_early_stopping_args`
        # is the single definition of these; calling it keeps the callback's
        # contract and the launched run in agreement.
        **swift_early_stopping_args(
            int(evaluation.get("early_stopping_patience", 2)),
            metric_for_best_model=evaluation["metric_for_best_model"],
            greater_is_better=bool(evaluation.get("greater_is_better", True)),
            load_best_model_at_end=bool(evaluation["load_best_model_at_end"]),
        ),
        "logging_steps": cfg["logging"]["logging_steps"],
        "seed": cfg["seed"],
        "deepspeed": f"configs/deepspeed/{deepspeed}.json",
        # Frozen by default. When escalated it is LoRA-on-ViT: full fine-tuning
        # risks the pretrained document/OCR capability the image-only pathway
        # depends on (arch §3).
        "freeze_vit": not train_vit,
    }
    if val_paths:
        args["val_dataset"] = val_paths
    if resume_from:
        args["resume_from_checkpoint"] = resume_from

    recorded = TrainingConfig(
        # Derived from the config that actually ran, never defaulted. These
        # three used to carry "QLoRA"/NF4/paged-8bit defaults on the model, so a
        # bf16 run that did not pass them recorded a technique it never used —
        # in the one record the whole reproducibility story rests on.
        technique=technique(base),
        base_quantization=manifest_descriptor(base),
        lora_rank=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        learning_rate=opt["learning_rate"],
        lr_scheduler=opt["lr_scheduler_type"],
        warmup_ratio=opt["warmup_ratio"],
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
        max_seq_len=base["sequence"]["max_seq_len"],
        seed=cfg["seed"],
    )
    return SwiftConfig(args), recorded


def build_manifest(
    *,
    run_id: str,
    corpus_version: str,
    corpus_manifest: dict[str, Any],
    training_cfg: TrainingConfig,
    data_stats: DataStats,
    staging_path: str,
    continued_from: str | None = None,
) -> RunManifest:
    """Build the run manifest. Written to Blob even while weights are staged."""
    base = base_model_config()["model"]
    if is_dirty_worktree():
        log.warning(
            "the working tree has uncommitted changes, so code_git_commit does not fully "
            "describe what ran — this run is not exactly reproducible."
        )

    return RunManifest(
        run_id=run_id,
        run_type="foundation",
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
    )


def train_foundation(
    *,
    corpus_version: str,
    out_version: str,
    client: BlobClient,
    corpus_manifest: dict[str, Any],
    data_stats: DataStats,
    train_vit: bool = False,
    deepspeed: str = "zero2",
    continue_from: str | None = None,
    dry_run: bool = False,
) -> tuple[SwiftConfig, RunManifest]:
    """Configure and launch a Foundation run.

    Args:
        continue_from: **a filesystem checkpoint path**, not a registry run-id.
            It goes straight to ms-swift's `resume_from_checkpoint`, which reads
            a directory — passing a run-id like "foundation-v3" produced a run
            that silently trained from base while its manifest recorded a
            lineage that never happened. `assert_checkpoint_path` refuses the
            run-id shape rather than letting it through.
            Continue on top of an existing Foundation rather than
            retraining from the HF base. Permitted for a **minor patch** only,
            and it marks the manifest so the promotion gate demands cross-type
            regression evidence before promoting (arch §12).
        dry_run: assemble and record everything without launching.
    """
    validate_all(require_pinned_revision=not dry_run)

    # Train and val are kept apart. Passing both to `--dataset` made ms-swift
    # treat the validation split as training data and then carve its own eval
    # split out of the union, so `metric_for_best_model` selected on documents
    # the model had memorised — and the promotion gate read that number.
    corpus_paths = [
        paths.corpus_split(corpus_version, doc_type, "train") for doc_type in ACTIVE_DOC_TYPES
    ]
    val_paths = [
        paths.corpus_split(corpus_version, doc_type, "val") for doc_type in ACTIVE_DOC_TYPES
    ]
    staging = paths.staging_adapter_dir("foundation", out_version)

    if continue_from:
        assert_checkpoint_path(continue_from)

    swift, recorded = build_swift_config(
        corpus_paths=corpus_paths,
        val_paths=val_paths,
        output_dir=staging,
        train_vit=train_vit,
        deepspeed=deepspeed,
        resume_from=continue_from,
    )

    if train_vit:
        log.warning(
            "ViT training is enabled. This must be a LoRA on the vision encoder, never a full "
            "fine-tune, and it should only follow a gate decision of 'fire' (arch §3)."
        )
    if continue_from:
        log.warning(
            "continuing from %s rather than retraining from base. Permitted for a minor patch, "
            "but promotion will require cross-type regression evidence — continued training "
            "compounds drift across cycles (arch §12).", continue_from,
        )

    manifest = build_manifest(
        run_id=f"foundation-{out_version}",
        corpus_version=corpus_version,
        corpus_manifest=corpus_manifest,
        training_cfg=recorded,
        data_stats=data_stats,
        staging_path=staging,
        continued_from=continue_from,
    )
    # `training`, not `trained`: nothing has run yet. The manifest is written
    # first only to reserve the run_id and capture the config.
    manifest.status = "training"
    write_manifest(manifest, client)

    if dry_run:
        log.info("dry run — configuration assembled and manifest written, nothing launched")
        return swift, manifest

    launch_and_record(swift, manifest, client)
    return swift, manifest


#: What a registry run-id looks like: `foundation-v3`, `acord-v2`. A value of
#: this shape passed as a checkpoint path is a mistake every time.
#: The run-ids this codebase actually generates: dotted versions
#: (`foundation-v2.1`, which paths.py explicitly accepts) and the
#: `{doc_type}-adapter-v{n}` form train_adapter emits. Requiring an
#: undotted single-lineage id let every real adapter id and every point
#: release past the guard, into ms-swift as resume_from_checkpoint, where
#: it found no directory, trained from base, and recorded a lineage that
#: never happened.
RUN_ID_SHAPE = re.compile(
    r"^(foundation|policy|lossrun|acord)(-adapter)?-v\d+(\.\d+)*$", re.IGNORECASE
)


def assert_checkpoint_path(continue_from: str) -> None:
    """Refuse a registry run-id where a checkpoint directory is required.

    `continue_from` reaches ms-swift as `resume_from_checkpoint`, which wants a
    directory on disk. A run-id looks close enough to be passed by mistake, and
    the failure is silent in the worst way: ms-swift finds no checkpoint, trains
    from base, and the manifest records `continued_from` — a lineage that did
    not happen, which is then read as evidence when deciding what regression
    testing a promotion needs.
    """
    if RUN_ID_SHAPE.match(continue_from.strip()):
        raise TrainingError(
            f"--continue-from takes a checkpoint DIRECTORY, not the registry run-id "
            f"{continue_from!r}. Resolve the run-id to its staging or published path first "
            "(artifact_registry.resolve_model_version), then pass that. A run-id here trains "
            "from base while the manifest records a lineage that never happened."
        )
    if "://" in continue_from:
        raise TrainingError(
            f"--continue-from takes a local checkpoint directory, not the URI {continue_from!r}. "
            "ms-swift reads it from disk; pull the checkpoint to the staging volume first."
        )


def launch_and_record(
    config: SwiftConfig,
    manifest: RunManifest,
    client: BlobClient,
) -> RunManifest:
    """Run training, then record what actually happened.

    The manifest is written before this is called — the run_id has to be
    reserved and the configuration captured even for a run that dies. But
    "written" is not "trained": the status stays ``training`` until ms-swift
    returns, and becomes ``failed`` if it does not, so the registry never claims
    weights that a crashed run never wrote.
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
    """Launch ms-swift. The training loop lives there, not here (arch §10)."""
    import shutil
    import subprocess

    if shutil.which("swift") is None:
        raise TrainingError(
            "the `swift` CLI is not on PATH. ms-swift is the Layer-3 entrypoint (arch §10); "
            'install the [train] extra on the pod: pip install -e ".[train]"\n'
            "If ms-swift turns out to lack a Qwen3-VL capability this needs, the documented "
            "fallback is invoking TRL SFTTrainer directly — a contingency, not a parallel option."
        )
    argv = config.to_cli()
    log.info("launching: %s", " ".join(argv))
    subprocess.run(argv, check=True)


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Train the Foundation LoRA")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out-version", required=True)
    parser.add_argument("--deepspeed", choices=["zero2", "zero3"], default="zero2")
    parser.add_argument("--train-vit", action="store_true",
                        help="escalate to a ViT LoRA — only after a gate decision of 'fire'")
    parser.add_argument("--continue-from", default=None,
                        help="minor patch only; promotion then requires cross-type regression evidence")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = BlobClient()
    corpus_manifest = client.read_json(paths.corpus_manifest(args.corpus))

    train_foundation(
        corpus_version=args.corpus,
        out_version=args.out_version,
        client=client,
        corpus_manifest=corpus_manifest,
        data_stats=DataStats(
            train_examples=corpus_manifest.get("total_rows", 0),
            val_examples=0, test_examples=0,
            lob_coverage=corpus_manifest.get("lob_coverage", {}),
            alias_coverage=corpus_manifest.get("alias_coverage", {}),
            confusable_example_count=corpus_manifest.get("confusable_example_count", 0),
            deidentified=corpus_manifest.get("deidentified", False),
        ),
        train_vit=args.train_vit,
        deepspeed=args.deepspeed,
        continue_from=args.continue_from,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
