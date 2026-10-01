"""How much does reading a policy in windows lose, before any model is involved?

    python scripts/diagnose_windowing.py                     # local bundles, every page routed
    python scripts/diagnose_windowing.py --blob --tenant T   # on the pod: real MinerU routing

The oracle merge. Each document's GOLD label is sliced into its windows exactly
as the dataset build slices it (``policy_windows.window_target``), the slices
are merged by the serving merge (``serving.policy_merge``), and the result is
compared with the full gold label. A perfect model reproduces exactly those
slices, so whatever this loses, the model cannot win back: it is the ceiling
windowing puts on accuracy.

The gold is compared as training and scoring use it: narrowed to what the
line's schema can hold (``common.canonical.schema_label``). Values outside the
schema are counted separately - no window can be asked for them and no model
can write them, so they are a labelling-format gap, not a windowing loss.

Every gold value the merged result does not reproduce gets a reason:

* ``no_page_ref``   - no page recorded, in a group read over several windows,
  so no window may be asked for it (left out of training);
* ``unread_page``   - printed only on pages no window of its group reads (the
  page router skipped them; only measurable with OCR, i.e. ``--blob``);
* ``row_unmatched`` - its table row came back split or keyless and was not
  matched to the gold row by its identifier;
* ``merge``         - anything else the merge dropped or overwrote.

Writes ``summary.txt``, ``documents.csv`` and ``lost_values.csv`` (the last
quotes gold values - personal data, keep it local) under ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REASONS = ("no_page_ref", "unread_page", "row_unmatched", "merge")


@dataclass
class DocumentResult:
    source_id: str
    lob: str
    synthetic: bool
    pages: int
    windows: int = 0
    gold_values: int = 0
    outside_schema: int = 0
    recovered: int = 0
    lost: Counter = field(default_factory=Counter)
    gold_rows: int = 0
    merged_rows: int = 0
    conflicts: int = 0
    lost_values: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def recall(self) -> float | None:
        return self.recovered / self.gold_values if self.gold_values else None


def oracle(source_id: str, label: dict, lob: Any, page_count: int,
           ocr_pages: list[str] | None, synthetic: bool = False) -> DocumentResult:
    """Slice ``label`` into its windows, merge the slices, compare with ``label``."""
    from common.canonical import schema_label, values_view, without_system_fields
    from common.label_mapping import map_label
    from data_pipeline.dataset_builder.policy_windows import (
        TargetReport,
        plan_windows,
        routed_pages,
        unread_values,
        window_target,
    )
    from serving.policy_merge import PolicyWindow, merge_policy_windows

    line = ", ".join(lob) if isinstance(lob, list) else str(lob)
    result = DocumentResult(source_id, line, synthetic, page_count)
    routed, declarations_page = routed_pages(ocr_pages, page_count)
    plans = plan_windows(lob, routed, declarations_page)
    report = TargetReport()
    windows = [PolicyWindow(group=p.group, pages=list(p.pages),
                            extraction=window_target(label, lob, p, report)) for p in plans]
    merged = merge_policy_windows(windows)
    result.windows, result.conflicts = len(plans), len(merged.conflicts)

    mapped = map_label(label, lob)
    scored = without_system_fields(schema_label(label, "policy", None, lob))
    result.outside_schema = _stated(without_system_fields(mapped)) - _stated(scored)
    gold = values_view(scored)
    got = values_view(without_system_fields(merged.extraction))
    reasons = {
        "no_page_ref": {_bare(p.split(":", 1)[1]) for p in report.unplaced},
        "unread_page": {_bare(p) for p in unread_values(mapped, lob, plans)},
    }

    def lose(path: str, value: Any, unmatched_row: bool) -> None:
        bare = _bare(path)
        if bare in reasons["no_page_ref"] or any(bare.startswith(r + ".") for r in reasons["no_page_ref"]):
            reason = "no_page_ref"
        elif bare in reasons["unread_page"]:
            reason = "unread_page"
        elif unmatched_row:
            reason = "row_unmatched"
        else:
            reason = "merge"
        result.lost[reason] += 1
        result.lost_values.append((path, str(value)[:80], reason))

    def compare(expected: Any, actual: Any, path: str, unmatched_row: bool = False) -> None:
        from common.normalize import values_match
        from evaluation.metrics.field_accuracy import _infer_key_fields, _is_empty, _row_key

        if isinstance(expected, dict):
            sub = actual if isinstance(actual, dict) else {}
            for key, value in expected.items():
                compare(value, sub.get(key), f"{path}.{key}" if path else key, unmatched_row)
        elif isinstance(expected, list) and expected and all(isinstance(r, dict) for r in expected):
            rows = actual if isinstance(actual, list) else []
            result.gold_rows += len(expected)
            result.merged_rows += len(rows)
            keys = _infer_key_fields(expected)
            pool: dict[tuple, list[Any]] = defaultdict(list)
            for row in rows:
                pool[_row_key(row, keys)].append(row)
            for index, row in enumerate(expected):
                match = pool.get(_row_key(row, keys))
                compare(row, match.pop(0) if match else None, f"{path}[{index}]",
                        unmatched_row or not match)
        elif not _is_empty(expected):
            result.gold_values += 1
            if values_match(expected, actual, field_path=path):
                result.recovered += 1
            else:
                lose(path, expected, unmatched_row)

    compare(gold, got, "")
    return result


def _stated(node: Any) -> int:
    """Stated values in a label: envelopes with a raw or parsed value."""
    from common.canonical import is_field_value

    if is_field_value(node):
        return int(node.get("raw") is not None or node.get("parsed") is not None)
    if isinstance(node, dict):
        return sum(_stated(v) for v in node.values())
    if isinstance(node, list):
        return sum(_stated(v) for v in node)
    return 0


def _bare(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

def from_bundles(root: Path, limit: int | None):
    """Local bundles. No OCR here, so every page is routed and ``unread_page``
    cannot occur; page counts come from the PDF."""
    import fitz  # pymupdf

    for bundle in sorted(p for p in root.iterdir() if (p / "golden.json").exists())[:limit]:
        meta = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        pdf = bundle / "document.pdf"
        if not pdf.exists():
            continue
        with fitz.open(pdf) as doc:
            pages = doc.page_count
        yield (bundle.name, json.loads((bundle / "golden.json").read_text(encoding="utf-8")),
               meta.get("lob"), pages, None, bool(meta.get("synthetic")))


def from_blob(tenant: str | None, limit: int | None):
    """The pod's store: the imported labels and the MinerU text of every page,
    so the page router runs on what it runs on in training and serving."""
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids, load_golden_label

    client = BlobClient()
    for source_id in list_labeled_source_ids(client, "policy", tenant)[:limit]:
        label, meta = load_golden_label(source_id, "policy", client, tenant)
        meta_key = paths.ocr_meta("policy", source_id, tenant)
        if not client.exists(meta_key):
            print(f"  skip {source_id}: not OCR'd")
            continue
        pages = int(client.read_json(meta_key).get("page_count") or 0)
        texts = [client.read_text(paths.processed_page("policy", source_id, p, "md", tenant))
                 for p in range(1, pages + 1)]
        yield source_id, label, meta.get("lob"), pages, texts, bool(meta.get("synthetic"))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def summarise(results: list[DocumentResult]) -> str:
    def block(title: str, docs: list[DocumentResult]) -> list[str]:
        gold = sum(d.gold_values for d in docs)
        recovered = sum(d.recovered for d in docs)
        lost = sum((d.lost for d in docs), Counter())
        rows, merged_rows = sum(d.gold_rows for d in docs), sum(d.merged_rows for d in docs)
        line = (f"{title:<34} docs {len(docs):>4}  windows/doc {sum(d.windows for d in docs) / len(docs):5.1f}"
                f"  value recall {recovered / gold if gold else 0:7.2%}  ({gold - recovered} of {gold} lost)")
        detail = "    lost: " + (", ".join(f"{r} {lost[r]}" for r in REASONS if lost[r]) or "nothing")
        detail += f" | rows gold {rows} merged {merged_rows}"
        detail += f" | conflicts {sum(d.conflicts for d in docs)}"
        outside = sum(d.outside_schema for d in docs)
        share = outside / (gold + outside) if gold + outside else 0
        detail += (f"\n    outside schema (not trained, not scored): {outside} of "
                   f"{gold + outside} stated values ({share:.1%})")
        return [line, detail]

    out = ["Oracle merge: gold sliced into windows, merged, compared with gold.",
           "Recall below 100% is accuracy windowing loses before any model is involved.", ""]
    out += block("ALL", results)
    originals = [d for d in results if not d.synthetic]
    if originals and len(originals) < len(results):
        out += block("originals only", originals)
    out.append("")
    by_lob: dict[str, list[DocumentResult]] = defaultdict(list)
    for d in results:
        by_lob[d.lob].append(d)
    for lob in sorted(by_lob):
        out += block(lob, by_lob[lob])
    worst = sorted((d for d in results if d.recall is not None), key=lambda d: d.recall)[:10]
    out += ["", "Lowest-recall documents:"]
    out += [f"  {d.recall:7.2%}  {d.source_id}  ({d.pages} pages, {d.windows} windows) "
            f"{dict(d.lost)}" for d in worst]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundles", type=Path, default=Path("data/bundles"))
    parser.add_argument("--blob", action="store_true", help="read labels and OCR from the pod's store")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("testing/results/windowing"))
    args = parser.parse_args(argv)

    source = from_blob(args.tenant, args.limit) if args.blob else from_bundles(args.bundles, args.limit)
    results, failed = [], []
    for source_id, label, lob, pages, texts, synthetic in source:
        try:
            results.append(oracle(source_id, label, lob, pages, texts, synthetic))
        except Exception as exc:  # noqa: BLE001 - one bad document must not hide the rest
            failed.append((source_id, f"{type(exc).__name__}: {exc}"))
    if not results:
        print("no documents scored", failed[:5])
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    text = summarise(results)
    if failed:
        text += f"\n\n{len(failed)} document(s) failed:\n" + "\n".join(f"  {s}: {e}" for s, e in failed[:20])
    (args.out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    with (args.out / "documents.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_id", "lob", "synthetic", "pages", "windows", "gold_values",
                         "recovered", "recall", "outside_schema", *REASONS, "gold_rows", "merged_rows", "conflicts"])
        for d in results:
            writer.writerow([d.source_id, d.lob, d.synthetic, d.pages, d.windows, d.gold_values,
                             d.recovered, f"{d.recall:.4f}" if d.recall is not None else "",
                             d.outside_schema,
                             *(d.lost[r] for r in REASONS), d.gold_rows, d.merged_rows, d.conflicts])
    with (args.out / "lost_values.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_id", "lob", "path", "gold_value", "reason"])
        for d in results:
            for path, value, reason in d.lost_values:
                writer.writerow([d.source_id, d.lob, path, value, reason])
    print(text)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
