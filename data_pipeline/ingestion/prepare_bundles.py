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

**A SPEC_21 delivery** (``split_manifest.csv``: split, lob, carrier, seed, twin,
digital, scanned, gold) delivers each twin twice - a digital PDF and a scanned
rendering of it - with one gold. Each render becomes a document of its own
(``render_mode`` digital or scanned, ``sample`` the twin's number, the seed as
the family), both reading one gold. The gold the bundles hold is corrected
where the delivered one cannot be trained on as it is (:func:`corrected_gold`),
each change listed in ``<out>/corrections.csv``; the delivery is never changed.
A gold still outside its line's schema after that is left out, with the reason.
``--dry-run`` reports all of it and writes nothing.
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

#: A SPEC_21 delivery's manifest: one row per twin, both renders and the gold.
TWIN_MANIFEST = "split_manifest.csv"
TWIN_COLUMNS = ("split", "lob", "carrier", "seed", "twin", "digital", "scanned", "gold")

#: Line names a delivery misspells, in its folders and its manifest's lob column.
DELIVERY_SPELLINGS = {"motorcyle": "motorcycle"}

#: Coverage codes in no list of their line, and the listed code each one is
#: (the overlays' changelogs; see the file).
RECODES = Path(__file__).resolve().parents[2] / "configs" / "coverage_code_recodes.yaml"


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
    #: Gold corrections by kind, and one row per change: (document, kind, detail).
    corrections: Counter = field(default_factory=Counter)
    correction_rows: list = field(default_factory=list)
    dry_run: bool = False

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
        for kind, n in sorted(self.corrections.items()):
            lines.append(f"  gold corrected ({kind}): {n}")
        lines.append(f"  files linked {self.linked}, copied {self.copied}")
        if self.dry_run:
            lines.append("  dry run: nothing was written")
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
    if (delivery / TWIN_MANIFEST).is_file():
        return prepare_twin_bundles(delivery, out, lines=lines, mode=mode, exclusions=exclusions)
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
            # The twin's render mode (SPEC_21 §7.8: native, scanned_from_digital,
            # ...): the per-seed twin cap counts each mode apart.
            "render_mode": render_mode(row, gold),
        }
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        report.written[(split, "synthetic" if metadata["synthetic"] else "real")] += 1
    return report


def delivery_line(lob: str) -> str:
    """A delivery's line name as this repository reads it: its misspellings put
    right, and classic auto read as personal auto (common.lob.MERGED_LINES)."""
    from common.lob import merge_line

    name = lob.strip().lower()
    return str(merge_line(DELIVERY_SPELLINGS.get(name, name)))


def prepare_twin_bundles(
    delivery: Path,
    out: Path = DEFAULT_OUT,
    *,
    lines: frozenset[str] | None = None,
    mode: str = "link",
    exclusions: dict[str, str] | None = None,
    dry_run: bool = False,
) -> PrepareReport:
    """Bundles from a SPEC_21 delivery: two documents per twin, one corrected gold."""
    import re

    manifest = delivery / TWIN_MANIFEST
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing_columns = [c for c in TWIN_COLUMNS if rows and c not in rows[0]]
    if not rows or missing_columns:
        raise BundleError(f"{manifest} is empty or lacks columns {missing_columns}")
    if mode not in ("link", "copy", "move"):
        raise BundleError(f"mode {mode!r} is not link, copy or move")

    recodes = read_recodes()
    report = PrepareReport(out=out, dry_run=dry_run)
    seen: dict[str, str] = {}
    for row in rows:
        lob = delivery_line(row["lob"])
        if lines is not None and lob not in lines:
            report.skipped_lines[lob] += 2
            continue
        gold_path = delivery / row["gold"]
        renders = (("digital", delivery / row["digital"]), ("scanned", delivery / row["scanned"]))
        if not gold_path.is_file() or not all(pdf.is_file() for _, pdf in renders):
            report.skipped["files listed in the manifest are missing"] += 2
            continue
        split = row["split"].strip().lower()
        if split not in ("train", "val", "test"):
            raise BundleError(f"{row['twin']}: split {row['split']!r} is not train, val or test")

        gold, notes, problem = corrected_gold(
            json.loads(gold_path.read_text(encoding="utf-8")), lob,
            carrier=row["carrier"], text_pdf=delivery / row["digital"], recodes=recodes)
        if problem:
            report.skipped[f"gold outside its line's schema after correction: {problem}"] += 2
            report.correction_rows.append((row["twin"], "left out", problem))
            continue
        for kind, detail in notes:
            report.corrections[kind] += 1
            report.correction_rows.append((row["twin"], kind, detail))

        number = re.search(r"(\d+)$", row["twin"].strip())
        for render, pdf in renders:
            name = pdf.stem
            if exclusions and name in exclusions:
                report.skipped[f"excluded: {exclusions[name]}"] += 1
                if not dry_run and (out / name).is_dir():
                    shutil.rmtree(out / name)
                continue
            if name in seen:
                raise BundleError(f"two manifest rows name the document {name!r} ({seen[name]}, {pdf})")
            seen[name] = str(pdf)
            metadata = {
                "lob": lob,
                "synthetic": True,
                # The seed: every twin of it, in both renders, is one family.
                "template_id": f"{lob}/{row['seed'].strip()}",
                "split": split,
                "carrier": row["carrier"].strip() or None,
                "source_system": "fideon_synth",
                "sample": int(number.group(1)) if number else None,
                "render_mode": render,
            }
            report.written[(split, "synthetic")] += 1
            if dry_run:
                continue
            folder = out / name
            folder.mkdir(parents=True, exist_ok=True)
            _place(pdf, folder / "document.pdf", mode, report)
            if notes:
                (folder / "golden.json").write_text(json.dumps(gold, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
            else:
                _place(gold_path, folder / "golden.json", mode, report)
            (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not dry_run and report.correction_rows:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "corrections.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["twin", "correction", "detail"])
            writer.writerows(report.correction_rows)
    return report


def read_recodes(path: Path = RECODES) -> dict[str, dict[str, str]]:
    """``{line: {code in no list: listed code}}``; empty when there is no table."""
    import yaml

    if not path.is_file():
        return {}
    return {str(line): {str(k): str(v) for k, v in (codes or {}).items()}
            for line, codes in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items()}


def _amount(value: object) -> float | None:
    parsed = value.get("parsed") if isinstance(value, dict) else value
    try:
        return round(float(parsed), 2) if parsed is not None else None
    except (TypeError, ValueError):
        return None


def corrected_gold(
    gold: dict, lob: str, *, carrier: str, text_pdf: Path, recodes: dict[str, dict[str, str]],
) -> tuple[dict, list[tuple[str, str]], str | None]:
    """``(gold, [(kind, detail)], problem)``: the gold a bundle holds.

    For a SPEC_21 common-model line, three corrections, each only where the
    delivered gold cannot be trained on as it is:

    * **coverage code** in no list of the line, re-coded to the listed code the
      overlays' changelogs give for it (``configs/coverage_code_recodes.yaml``):
      the decoder only writes listed codes, so a target with another one is an
      answer the served model can never produce;
    * **premium written twice**: a ``premium.items`` entry for a coverage whose
      own ``premium`` holds the same amount - or, for a code on several rows
      (one per vehicle), their sum - is dropped, so a coverage's premium has one
      home (``coverages[].premium``), the schema review's proposal. An item whose
      coverage row has no premium, or that names no coverage row, is the only
      place that premium is written, and stays;
    * **carrier missing**, where the delivery's carrier is printed in the
      document: ``carrier.name`` is filled with it as printed, citing the pages
      that print it (All State's policies print only the brand, "Allstate").

    ``problem`` is the first schema error left after them; such a gold is not
    bundled.
    """
    import copy

    from common.schemas import is_common_model, iter_validation_errors, load_schema

    notes: list[tuple[str, str]] = []
    if is_common_model("policy", None, lob):
        gold = copy.deepcopy(gold)
        listed = set(load_schema("policy", None, lob).get("fideon:coverage_codes") or [])
        table = recodes.get(lob, {})
        premium = gold.get("premium") if isinstance(gold.get("premium"), dict) else {}
        for where, rows in (("coverages", gold.get("coverages") or []),
                            ("premium.items", premium.get("items") or [])):
            for index, row in enumerate(rows):
                code = row.get("coverage_code") if isinstance(row, dict) else None
                if isinstance(code, str) and code not in listed and code in table:
                    row["coverage_code"] = table[code]
                    notes.append(("coverage code", f"{where}[{index}] {code} -> {table[code]}"))

        # Each coverage row's premium, and per code the sum of its rows' premiums:
        # an item holding either is the same money written a second time.
        by_code: dict[object, list[float]] = {}
        for c in gold.get("coverages") or []:
            if isinstance(c, dict) and _amount(c.get("premium")) is not None:
                by_code.setdefault(c.get("coverage_code"), []).append(_amount(c.get("premium")))
        own = {(code, amount) for code, amounts in by_code.items() for amount in amounts}
        own |= {(code, round(sum(amounts), 2)) for code, amounts in by_code.items() if len(amounts) > 1}
        if premium.get("items"):
            kept = []
            for item in premium["items"]:
                if (isinstance(item, dict) and item.get("unit_type") == "coverage"
                        and (item.get("coverage_code"), _amount(item.get("amount"))) in own):
                    notes.append(("premium written twice",
                                  f"dropped premium.items {item.get('coverage_code')} {_amount(item.get('amount'))}"))
                    continue
                kept.append(item)
            premium["items"] = kept

        if not gold.get("carrier") and carrier.strip():
            printed = printed_carrier(text_pdf, carrier)
            if printed:
                raw, pages = printed
                gold["carrier"] = {"name": {"raw": raw, "parsed": raw,
                                            "confidence": {"score": 1.0, "source": "deterministic"},
                                            "page_ref": pages, "flagged": False}}
                notes.append(("carrier missing", f"carrier.name = {raw!r} as printed on page(s) {pages}"))

    errors = list(iter_validation_errors(gold, "policy", None, lob))
    return gold, notes, (errors[0][:200] if errors else None)


def printed_carrier(pdf: Path, carrier: str) -> tuple[str, list[int]] | None:
    """The delivery's carrier as the document prints it, and the pages it is on.

    ``carrier`` is the manifest's name for it (``all_state``); it matches
    "Allstate" or "All State" in the visible text. None when no page prints it.
    """
    import re

    import pymupdf

    tokens = [t for t in re.split(r"[\s_]+", carrier.strip()) if t]
    if not tokens:
        return None
    pattern = re.compile(r"\b" + r"[\s-]*".join(map(re.escape, tokens)) + r"\b", re.IGNORECASE)
    raw, pages = None, []
    with pymupdf.open(pdf) as doc:
        for number, page in enumerate(doc, start=1):
            found = pattern.search(page.get_text())
            if found:
                raw = raw or found.group(0)
                pages.append(number)
    return (raw, pages) if raw else None


def render_mode(row: dict[str, str], gold: Path) -> str | None:
    """A twin's render mode: the manifest's ``mode`` column (SPEC_21 manifest.csv),
    else its gold's ``fideon:provenance.mode``; None when neither says."""
    value = (row.get("mode") or row.get("render_mode") or "").strip()
    if value:
        return value.lower()
    try:
        provenance = json.loads(gold.read_text(encoding="utf-8")).get("fideon:provenance") or {}
    except (OSError, ValueError, AttributeError):
        return None
    mode = provenance.get("mode") if isinstance(provenance, dict) else None
    return str(mode).strip().lower() if isinstance(mode, str) and mode.strip() else None


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
    from common.lob import merge_line

    for r in rows:
        # Classic auto is read as personal auto (common.lob.MERGED_LINES): an
        # override file written before the merge still names it.
        lob = str(merge_line(r["lob"].strip()))
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
    parser.add_argument("--dry-run", action="store_true",
                        help="SPEC_21 delivery: report the bundles and gold corrections, write nothing")
    args = parser.parse_args(argv)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("data_pipeline.ingestion.prepare_bundles", argv):
        return 0

    lines = None if args.scope == "none" else get_scope(args.scope).lines or None
    try:
        if (args.input / TWIN_MANIFEST).is_file():
            report = prepare_twin_bundles(args.input, args.out, lines=lines, mode=args.mode,
                                          exclusions=read_exclusions(args.exclude), dry_run=args.dry_run)
        else:
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
