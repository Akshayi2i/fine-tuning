"""A small end-to-end run on the pod, before the real one: does the whole path work?

    python -m orchestration.smoke_run                      # every step
    python -m orchestration.smoke_run --steps check,finetune

It takes a handful of source documents from the uploaded batch — each with all
of its synthetic twins, so the delivered split stays intact — and runs them
through import, OCR, the post-OCR check and ``finetune``, under their own
tenant (``smoke``) and version (``v0``). Nothing it writes touches the real
data: corpus, labels and OCR output are tenant-scoped, and the model versions
start at ``v1``. It never freezes an eval set or packages a release: the frozen
set is not tenant-scoped, and a smoke yardstick would become the real one.

What it proves: MinerU runs on the GPU, the labels survive import, the corpus
builds with the delivered split, ms-swift trains, a checkpoint is chosen and
merged. What it measures: how many rows exceed the sequence caps (the corpus
build reports rejections) and how long a step takes. Its scores mean nothing.

Steps: ``select`` (pick the documents), ``import``, ``ocr`` (in the OCR
environment), ``check`` (post-OCR values), ``preflight`` (everything finetune
depends on, the corpus built in memory), ``finetune``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

STEPS = ("select", "import", "ocr", "check", "preflight", "finetune")
DEFAULT_TENANT = "smoke"
DEFAULT_VERSION = "v0"
WORKSPACE = Path(os.environ.get("FIDEON_WORKSPACE") or "/workspace")  # an empty .env value means unset
OCR_PYTHON = WORKSPACE / "venv-ocr" / "bin" / "python"


class SmokeError(RuntimeError):
    """Raised when the smoke run cannot be set up."""


def select_sources(bundles: Path, *, train_sources: int = 4, val_sources: int = 1,
                   test_sources: int = 1) -> list[Path]:
    """Folders of a few whole source families, spread across lines, split kept.

    A source is taken with every twin it has, so no family is cut in two, and
    train sources are drawn from as many different lines as there are.
    """
    families: dict[tuple[str, str], list[Path]] = defaultdict(list)
    line_of: dict[tuple[str, str], str] = {}
    for meta_path in sorted(bundles.glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        split = str(meta.get("split") or "").lower()
        if split not in ("train", "val", "test"):
            raise SmokeError(f"{meta_path.parent.name} has no delivered split; the smoke run needs one")
        key = (split, str(meta.get("template_id") or meta_path.parent.name))
        families[key].append(meta_path.parent)
        line_of[key] = str(meta.get("lob"))
    chosen: list[Path] = []
    for split, count in (("train", train_sources), ("val", val_sources), ("test", test_sources)):
        keys = sorted(k for k in families if k[0] == split)
        if len(keys) < count:
            raise SmokeError(f"the batch has {len(keys)} {split} source(s); the smoke run needs {count}")
        picked: list[tuple[str, str]] = []
        seen_lines: set[str] = set()
        for key in keys:                                    # one per line first
            if len(picked) < count and line_of[key] not in seen_lines:
                picked.append(key)
                seen_lines.add(line_of[key])
        for key in keys:                                    # then fill up
            if len(picked) < count and key not in picked:
                picked.append(key)
        for key in picked:
            chosen += families[key]
    return chosen


def assert_fresh(client, tenant: str, version: str, steps: list[str]) -> None:
    """Refuse a smoke run that would land on an earlier one's leftovers.

    Labels, OCR output and the corpus are kept per tenant, and the run registry
    per version. A second smoke run under the same names imported its documents
    beside the first run's, then found the corpus "already built" and the
    training "already complete", re-merged the OLD adapter and reported success
    without training on the new batch at all.

    Checked only when the run starts from the beginning (``import``): resuming a
    failed run with ``--steps check,finetune`` is the same run, and is allowed.
    """
    from artifact_registry import paths
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids

    if "import" not in steps:
        return
    held = list_labeled_source_ids(client, "policy", tenant)
    problems = []
    if held:
        problems.append(f"tenant {tenant!r} already holds {len(held)} imported document(s)")
    if client.exists(paths.corpus_manifest(version, tenant)):
        problems.append(f"corpus {version} already exists for tenant {tenant!r}")
    if "finetune" in steps:
        from common.scopes import load_scopes
        from registry_utils.query_registry import RegistryQueryError
        from registry_utils.query_registry import get as get_manifest

        run_id = load_scopes()["personal_lines"].run_id(version)
        try:
            status = get_manifest(run_id, client).status
        except (RegistryQueryError, KeyError, FileNotFoundError):
            status = None
        if status is not None:
            problems.append(f"run {run_id} already exists (status {status})")
    if problems:
        raise SmokeError(
            "; ".join(problems) + ". This smoke run would reuse that state instead of testing the "
            "new batch. Start it under names nothing has used: --tenant smoke2 --version v0.1 "
            "(any unused pair). To carry on a smoke run that stopped part-way, leave out "
            "select and import: --steps check,preflight,finetune."
        )


def stage_subset(folders: list[Path], out: Path) -> int:
    """Link (or copy) the chosen folders into ``out``, replacing an earlier subset."""
    if out.exists():
        shutil.rmtree(out)
    for folder in folders:
        dest = out / folder.name
        dest.mkdir(parents=True)
        for item in folder.iterdir():
            try:
                os.link(item, dest / item.name)
            except OSError:
                shutil.copy2(item, dest / item.name)
    return len(folders)


def commands(*, batch_dir: Path, subset_dir: Path, tenant: str, version: str,
             check_out: Path) -> dict[str, list[str]]:
    """The command each step runs, with the interpreter of its environment."""
    py = sys.executable
    return {
        "import": [py, "-m", "data_pipeline.ingestion.import_labeled_pdfs", "--input", str(subset_dir),
                   "--doc-type", "policy", "--tenant", tenant],
        "ocr": [str(OCR_PYTHON), "-m", "data_pipeline.ocr.run_mineru", "--doc-type", "policy",
                "--all-unprocessed", "--tenant", tenant],
        "check": [py, "-m", "data_pipeline.ocr_check", "--doc-type", "policy", "--tenant", tenant,
                  "--out", str(check_out)],
        # --min-labels-per-type 1: the 25-document floor is for a real run. A
        # smoke subset of sources WITHOUT synthetic twins is 6 documents, and
        # both steps refused it; it only ever passed on the twins' numbers.
        "preflight": [py, "-m", "orchestration.preflight", "--scope", "personal_lines", "--tenant", tenant,
                      "--corpus-version", version, "--out-version", version, "--ocr-check", str(check_out),
                      "--out", str(check_out / "preflight.json"), "--min-labels-per-type", "1"],
        "finetune": [py, "-m", "orchestration.run", "finetune", "--scope", "personal_lines", "--skip-ingest",
                     "--corpus-version", version, "--out-version", version, "--tenant", tenant,
                     "--min-labels-per-type", "1"],
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI over the functions above
    parser = argparse.ArgumentParser(description="A small end-to-end run on the pod")
    parser.add_argument("--batch", default="personal-v1", help="the uploaded batch under intake/")
    parser.add_argument("--tenant", default=DEFAULT_TENANT)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--train-sources", type=int, default=4)
    parser.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated, from {STEPS}")
    args = parser.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown step(s) {unknown}; choose from {STEPS}")
    if args.tenant in ("", "default"):
        parser.error("the smoke run needs its own tenant, never the real data's")
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("orchestration.smoke_run", argv, hint="smoke"):
        return 0

    from artifact_registry.blob_client import BlobClient

    try:
        assert_fresh(BlobClient(), args.tenant, args.version, steps)
    except SmokeError as exc:
        print(f"smoke: {exc}", flush=True)
        return 1

    batch_dir = WORKSPACE / "intake" / args.batch
    subset_dir = WORKSPACE / "intake" / f"{args.tenant}-{args.batch}"
    check_out = WORKSPACE / f"ocr_check_{args.tenant}"
    if "select" in steps:
        if not batch_dir.is_dir():
            from artifact_registry.blob_client import for_ingestion
            from data_pipeline.ingestion.pull_intake import pull_intake

            print(f"pulling {args.batch} to {batch_dir} ...", flush=True)
            report = pull_intake(for_ingestion(), args.batch, batch_dir.parent)
            print(report.describe(), flush=True)
            if report.failed:
                return 1
        folders = select_sources(batch_dir, train_sources=args.train_sources)
        print(f"smoke subset: {stage_subset(folders, subset_dir)} document(s) in {subset_dir}", flush=True)
    env = {**os.environ, "FIDEON_DETACHED": "1"}          # already in tmux: steps run in place
    for step, command in commands(batch_dir=batch_dir, subset_dir=subset_dir, tenant=args.tenant,
                                  version=args.version, check_out=check_out).items():
        if step not in steps:
            continue
        print(f"\n=== smoke: {step} ===\n{' '.join(command)}", flush=True)
        code = subprocess.run(command, env=env).returncode
        if code != 0:
            print(f"smoke: {step} failed (exit {code}); fix it and resume with --steps "
                  f"{','.join(steps[steps.index(step):])}", flush=True)
            return code
    print("\nsmoke run complete. Read the corpus build's rejected-row count and the post-OCR report "
          f"({check_out}/report.md) before the real run.", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
