"""Turn a synthetic-data delivery into import bundles: one folder per document.

The generator (fideon_synth ``scripts/build_dataset.py``) delivers

    <input>/Train|Val|Test/pdfs/<name>.pdf
    <input>/Train|Val|Test/gold json/<name>.json
    <input>/manifest.csv      split, lob, carrier, source, sample, kind, pdf, gold, ok, ...

and the importer and the audit read one folder per document:

    <out>/<name>/document.pdf, golden.json, metadata.json

``metadata.json`` is written from the manifest row, nothing guessed:

    lob          the line, which selects the canonical schema
    synthetic    true for a generated twin, false for the source itself
    template_id  the source document it was made from - the family the delivery
                 was split by, so a source and its twins stay on one side
    split        the delivery's own train/val/test, which the corpus build then
                 uses as given (split_groups.assign_delivered_splits)

Rows outside the scope's lines (flood, for personal_lines), rows the generator
flagged (``ok`` false) and rows whose files are missing are skipped and counted.

    python -m data_pipeline.ingestion.prepare_bundles --input "data/training data" --out data/bundles

Files are hard-linked by default: instant, and no second copy of 16 GB on the
same drive. Where a link is impossible (another drive) they are copied.

Documents the audit rejected are listed in an exclusion file (``--exclude``,
default ``data/bundle_exclusions.csv``: ``document,reason``), so a re-run keeps
them out - and removes their folders if an earlier run made them - instead of
the fix living only in a folder someone deleted by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

DEFAULT_OUT = Path("data") / "bundles"
DEFAULT_EXCLUSIONS = Path("data") / "bundle_exclusions.csv"
#: ``source_prefix,lob,reason``: a line the generator's folder got wrong, e.g. a
#: carrier's classic-car policies filed under personal_auto.
DEFAULT_LOB_OVERRIDES = Path("data") / "bundle_lob_overrides.csv"
REQUIRED_COLUMNS = ("split", "lob", "source", "kind", "pdf", "gold", "ok")


class BundleError(RuntimeError):
    """Raised when a delivery cannot be read at all."""


@dataclass
class PrepareReport:
    out: Path
    written: Counter = field(default_factory=Counter)          # (split, kind) -> documents
    skipped: Counter = field(default_factory=Counter)          # reason -> rows
    skipped_lines: Counter = field(default_factory=Counter)    # lob -> rows outside the scope
    relined: Counter = field(default_factory=Counter)          # "old -> new" -> rows given another line
    linked: int = 0
    copied: int = 0

    @property
    def documents(self) -> int:
        return sum(self.written.values())

    def describe(self) -> str:
        lines = [f"{self.documents} document folder(s) in {self.out}"]
        for split in ("train", "val", "test"):
            parts = {kind: n for (s, kind), n in self.written.items() if s == split}
            if parts:
                lines.append(f"  {split:5} " + ", ".join(f"{n} {kind}" for kind, n in sorted(parts.items())))
        for reason, n in sorted(self.skipped.items()):
            lines.append(f"  skipped {n}: {reason}")
        if self.skipped_lines:
            lines.append("  outside the scope: " + ", ".join(f"{k} {v}" for k, v in sorted(self.skipped_lines.items())))
        if self.relined:
            lines.append("  line overridden: " + ", ".join(f"{k} ({v})" for k, v in sorted(self.relined.items())))
        lines.append(f"  files linked {self.linked}, copied {self.copied}")
        return "\n".join(lines)


def _place(source: Path, dest: Path, mode: str, report: PrepareReport) -> None:
    """Put ``source`` at ``dest``: a hard link where possible, else a copy (or a move)."""
    if dest.exists():
        if dest.stat().st_size == source.stat().st_size:
            return
        dest.unlink()
    if mode == "move":
        shutil.move(str(source), str(dest))
        report.copied += 1
        return
    if mode == "link":
        try:
            os.link(source, dest)
            report.linked += 1
            return
        except OSError:
            pass                                    # another drive, or no link support
    shutil.copy2(source, dest)
    report.copied += 1


def prepare_bundles(
    delivery: Path,
    out: Path = DEFAULT_OUT,
    *,
    lines: frozenset[str] | None = None,
    mode: str = "link",
    exclusions: dict[str, str] | None = None,
    lob_overrides: list[tuple[str, str]] | None = None,
) -> PrepareReport:
    """Write ``out/<name>/{document.pdf, golden.json, metadata.json}`` for each usable row."""
    manifest = delivery / "manifest.csv"
    if not manifest.is_file():
        raise BundleError(f"no manifest.csv in {delivery}")
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing_columns = [c for c in REQUIRED_COLUMNS if rows and c not in rows[0]]
    if not rows or missing_columns:
        raise BundleError(f"{manifest} is empty or lacks columns {missing_columns}")
    if mode not in ("link", "copy", "move"):
        raise BundleError(f"mode {mode!r} is not link, copy or move")

    report = PrepareReport(out=out)
    seen: dict[str, str] = {}
    for row in rows:
        lob = row["lob"].strip()
        for prefix, new_lob in lob_overrides or ():
            if row["source"].strip().startswith(prefix):
                report.relined[f"{lob} -> {new_lob}"] += 1
                lob = new_lob
                break
        from common.lob import merge_line

        # Classic auto is read as personal auto (common.lob.MERGED_LINES).
        lob = str(merge_line(lob))
        if lines is not None and lob not in lines:
            report.skipped_lines[lob] += 1
            continue
        if row["ok"].strip().lower() != "true":
            report.skipped["flagged by the generator (ok is false)"] += 1
            continue
        pdf, gold = delivery / row["pdf"], delivery / row["gold"]
        if not pdf.is_file() or not gold.is_file():
            report.skipped["files listed in the manifest are missing"] += 1
            continue
        name = Path(row["pdf"]).stem
        if exclusions and name in exclusions:
            report.skipped[f"excluded: {exclusions[name]}"] += 1
            if (out / name).is_dir():
                shutil.rmtree(out / name)          # made by an earlier run: take it out
            continue
        if name in seen:
            raise BundleError(f"two manifest rows name the document {name!r} ({seen[name]}, {row['pdf']})")
        seen[name] = row["pdf"]
        split = row["split"].strip().split("/")[-1].lower()
        if split not in ("train", "val", "test"):
            raise BundleError(f"{name}: split {row['split']!r} is not Train, Val or Test")
        kind = row["kind"].strip().lower()

        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        _place(pdf, folder / "document.pdf", mode, report)
        _place(gold, folder / "golden.json", mode, report)
        metadata = {
            "lob": lob,
            "synthetic": kind == "synthetic",
            "template_id": str(PurePosixPath(row["source"].strip()).with_suffix("")),
            "split": split,
            "carrier": row.get("carrier") or None,
            "source_system": "fideon_synth",
            "sample": int(row["sample"]) if (row.get("sample") or "").strip().isdigit() else None,
        }
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        report.written[(split, "synthetic" if metadata["synthetic"] else "real")] += 1
    return report


def read_exclusions(path: Path) -> dict[str, str]:
    """``{document: reason}`` from a ``document,reason`` CSV; empty when there is none."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and not {"document", "reason"} <= set(rows[0]):
        raise BundleError(f"{path} needs the columns document,reason")
    return {r["document"].strip(): (r["reason"] or "").strip() for r in rows if r["document"].strip()}


def read_lob_overrides(path: Path) -> list[tuple[str, str]]:
    """``[(source_prefix, lob)]`` from a ``source_prefix,lob,reason`` CSV; empty when there is none."""
    from common.scopes import known_lines

    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and not {"source_prefix", "lob"} <= set(rows[0]):
        raise BundleError(f"{path} needs the columns source_prefix,lob,reason")
    out = []
    for r in rows:
        lob = r["lob"].strip()
        if lob not in known_lines():
            raise BundleError(f"{path}: {lob!r} is not a line with a canonical schema")
        out.append((r["source_prefix"].strip(), lob))
    return out


def main(argv: list[str] | None = None) -> int:
    from common.scopes import get_scope

    parser = argparse.ArgumentParser(description="Turn a synthetic-data delivery into import bundles")
    parser.add_argument("--input", required=True, type=Path, help="the delivery: Train/Val/Test + manifest.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scope", default="personal_lines",
                        help="keep only this scope's lines ('none' keeps every line)")
    parser.add_argument("--mode", choices=("link", "copy", "move"), default="link",
                        help="link (default: instant, no second copy on the same drive), copy, or move")
    parser.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUSIONS,
                        help="document,reason CSV of documents to leave out")
    parser.add_argument("--lob-overrides", type=Path, default=DEFAULT_LOB_OVERRIDES,
                        help="source_prefix,lob,reason CSV of lines to correct")
    args = parser.parse_args(argv)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("data_pipeline.ingestion.prepare_bundles", argv):
        return 0

    lines = None if args.scope == "none" else get_scope(args.scope).lines or None
    try:
        report = prepare_bundles(args.input, args.out, lines=lines, mode=args.mode,
                                 exclusions=read_exclusions(args.exclude),
                                 lob_overrides=read_lob_overrides(args.lob_overrides))
    except BundleError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(report.describe())
    print(f"\nNext: python -m data_pipeline.audit --input \"{args.out}\"")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
