"""The operator command surface (SPEC_13 §1).

Three commands plus one umbrella::

    python -m orchestration.run finetune --input ./intake --out-version v2 --gpu a100-80
    python -m orchestration.run package  --version v2 --formats fp16 q5_k_m
    python -m orchestration.run extract  --model base --input testing/test_data/
    python -m orchestration.run all      --input ./intake --out-version v2

``all`` is ``finetune`` then ``package``. **It never runs ``extract``** —
extraction is run against a chosen model version whenever you want, including
against models trained weeks earlier and against the untuned base, so folding it
into the build would conflate "produce a model" with "use a model".

This module is argument parsing and nothing else. Every decision about stage
order, idempotency and the gate lives in ``pipeline_dag``, so the CLI cannot
develop its own opinion about them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from artifact_registry.blob_client import BlobClient, for_ingestion
from common.constants import ACTIVE_DOC_TYPES, DAY_ZERO_MIN_LABELS_PER_TYPE
from orchestration import pipeline_dag
from orchestration.pipeline_dag import RunReport, StageContext, run_stages, stages_for, stages_from
from orchestration.runpod_controller import RunPodController
from orchestration.settings import backoff_seconds, defaults, retry_policy

log = logging.getLogger(__name__)


def build_context(args: argparse.Namespace, *, client: BlobClient | None = None,
                  controller: RunPodController | None = None, **over: object) -> StageContext:
    """Turn parsed arguments into the one object the stages share."""
    from registry_utils.write_run_manifest import capture_git_commit

    commit = getattr(args, "commit", None) or capture_git_commit()
    controller = controller or RunPodController(git_commit=commit)

    kwargs: dict[str, object] = {
        "client": client or BlobClient(),
        # Only ingestion and OCR may reach raw-documents/; every other stage
        # works from processed/ (arch §18a).
        "raw_client": None if client else for_ingestion(),
        "controller": controller,
        "out_version": getattr(args, "out_version", None) or getattr(args, "version", ""),
        "corpus_version": getattr(args, "corpus_version", "") or "",
        "input_dir": Path(args.input) if getattr(args, "input", None) else None,
        "doc_types": list(getattr(args, "doc_types", None) or ACTIVE_DOC_TYPES),
        "tenant_id": getattr(args, "tenant", None),
        "formats": list(getattr(args, "formats", None) or ["fp16", "q5_k_m"]),
        "dtype": getattr(args, "dtype", "fp16"),
        "gpu_class": getattr(args, "gpu", None),
        "dry_run": getattr(args, "dry_run", False),
        "skip_ingest": getattr(args, "skip_ingest", False),
        "foundation_only": getattr(args, "foundation_only", False),
        "train_vit": getattr(args, "train_vit", False),
        "min_labels_per_type": getattr(args, "min_labels_per_type", DAY_ZERO_MIN_LABELS_PER_TYPE),
        "push_adapters": getattr(args, "push_adapters", False),
        "skip_quantize": getattr(args, "skip_quantize", False),
        "keep_staging": getattr(args, "keep_staging", False),
        "from_blob": getattr(args, "from_blob", False),
        "git_commit": commit,
        "max_attempts": int(retry_policy()["max_attempts"]),
        "retry_backoff_seconds": backoff_seconds(),
    }
    kwargs.update(over)
    return StageContext(**kwargs)  # type: ignore[arg-type]


def _selected_stages(command: str, from_stage: str | None) -> Sequence[pipeline_dag.Stage]:
    if not from_stage:
        return stages_for(command)
    # `--from-stage` on `all` still spans both commands: resuming at `merge`
    # should carry on into packaging, not stop at the end of finetune.
    return stages_from(from_stage, command)


def run_command(command: str, ctx: StageContext, *, from_stage: str | None = None) -> RunReport:
    """Run one build command. ``all`` is the two build commands back to back."""
    return run_stages(ctx, _selected_stages(command, from_stage), command=command)


# --------------------------------------------------------------------------
# extract — a wrapper, never a second implementation
# --------------------------------------------------------------------------


def run_extract(args: argparse.Namespace) -> int:
    """Delegate to ``testing/run_extraction.py``, which wraps the serving pipeline.

    Two layers of wrapping with no logic in either is the point: extraction has
    exactly one implementation (``serving/pipeline.py``), so the numbers this
    command produces describe the system that actually runs in production.
    """
    from testing import run_extraction

    argv = ["--model", args.model, "--input", str(args.input), "--mode", args.mode]
    if args.ground_truth:
        argv += ["--ground-truth", str(args.ground_truth)]
    if args.quant_format:
        argv += ["--format", args.quant_format]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.tenant:
        argv += ["--tenant", args.tenant]
    return run_extraction.main(argv)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _add_finetune_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, help="directory of PDFs to ingest")
    parser.add_argument("--doc-types", nargs="+", dest="doc_types", choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--corpus-version", dest="corpus_version", default="")
    parser.add_argument("--out-version", dest="out_version", required=True)
    parser.add_argument("--gpu", default=None, help="GPU class override for the training pod")
    parser.add_argument("--commit", default=None, help="repo commit the pod clones; defaults to HEAD")
    parser.add_argument("--from-stage", dest="from_stage", default=None,
                        help="resume at this stage; earlier stages are not re-run")
    parser.add_argument("--skip-ingest", dest="skip_ingest", action="store_true")
    parser.add_argument("--foundation-only", dest="foundation_only", action="store_true",
                        help="train the Foundation and skip per-type adapters")
    parser.add_argument("--train-vit", dest="train_vit", action="store_true",
                        help="LoRA on the vision encoder; never a full fine-tune (arch §3)")
    parser.add_argument("--min-labels-per-type", dest="min_labels_per_type", type=int,
                        default=defaults().get("min_labels_per_type", DAY_ZERO_MIN_LABELS_PER_TYPE))
    parser.add_argument("--push-adapters", dest="push_adapters", action="store_true",
                        help="also push adapters to Blob immediately, leaving only the merged model staged")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="assemble and record everything without launching GPU work")


def _add_package_flags(parser: argparse.ArgumentParser, *, version_required: bool = True) -> None:
    if version_required:
        parser.add_argument("--version", required=True)
    parser.add_argument("--formats", nargs="+", default=list(defaults().get("formats", ["fp16", "q5_k_m"])))
    parser.add_argument("--skip-quantize", dest="skip_quantize", action="store_true")
    parser.add_argument("--keep-staging", dest="keep_staging", action="store_true")
    parser.add_argument("--from-blob", dest="from_blob", action="store_true",
                        help="artifacts were pushed with --push-adapters; read them from Blob")
    parser.add_argument("--dtype", default=defaults().get("dtype", "fp16"), choices=["fp16", "bf16"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestration.run",
        description="Fine-tuning lifecycle: finetune -> package -> extract. `all` = finetune + package.",
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_finetune_flags(sub.add_parser("finetune", help="stages 1-7: ingest through merge; artifacts staged"))

    package = sub.add_parser("package", help="stages 8-9: quantize and push to Azure Blob")
    _add_package_flags(package)
    parser_all = sub.add_parser("all", help="finetune then package. Never runs extract.")
    _add_finetune_flags(parser_all)
    _add_package_flags(parser_all, version_required=False)

    extract = sub.add_parser("extract", help="extract with a chosen model version")
    extract.add_argument("--model", required=True,
                         help="base | v1 | v2 ... — 'base' is the untuned model with no adapter")
    extract.add_argument("--input", required=True, type=Path)
    extract.add_argument("--mode", default="ocr_plus_image", choices=["ocr_plus_image", "image_only"])
    extract.add_argument("--ground-truth", dest="ground_truth", type=Path, default=None)
    extract.add_argument("--format", dest="quant_format", default=None)
    extract.add_argument("--limit", type=int, default=None)
    extract.add_argument("--tenant", default=None)

    deploy = sub.add_parser("deploy-endpoint", help="point the serving endpoint at a promoted version")
    deploy.add_argument("--model", required=True)
    deploy.add_argument("--dry-run", dest="dry_run", action="store_true")

    rollback = sub.add_parser("rollback-endpoint", help="restore the previously promoted version")
    rollback.add_argument("--dry-run", dest="dry_run", action="store_true")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    if args.command == "extract":
        return run_extract(args)

    if args.command in ("deploy-endpoint", "rollback-endpoint"):
        controller = RunPodController()
        if args.command == "deploy-endpoint":
            controller.deploy_endpoint(args.model, dry_run=args.dry_run)
        else:
            controller.rollback_endpoint(dry_run=args.dry_run)
        return 0

    if args.command == "all" and not getattr(args, "version", None):
        args.version = args.out_version

    ctx = build_context(args)
    report = run_command(args.command, ctx, from_stage=getattr(args, "from_stage", None))

    print(report.render())
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
