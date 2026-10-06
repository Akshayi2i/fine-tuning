"""Do the training documents print the labels carriers use? (common-model lines)

Aliases never go in the prompt (master §1.4): the variety of labels a field is
printed under has to come from the documents the model trains on. This counts,
per common-model line, how many training documents print each recorded label -
the overlay's, the common model's per field, each coverage code's - and lists
the labels no training document prints. A long gap list is a reason to add
documents (or twins) before training, not a reason to put the list in the prompt.

The labels seen here are also what evaluation calls "seen"
(evaluation.metrics.unseen_labels).

    python scripts/check_alias_coverage.py --bundles data/bundles_v2
    python scripts/check_alias_coverage.py --bundles data/bundles_v2 --split train --out reports/alias_coverage_schema2.json

Reads each bundle's ``metadata.json`` (lob, split) and its document's text
layer; a document with no text layer (a scan) is counted and skipped, since
only OCR would read it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics.unseen_labels import common_model_aliases, normalise, printed  # noqa: E402


def read_texts(bundles: Path, split: str) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Normalised text per line, of the bundles in ``split``; and how many were scans."""
    import pymupdf

    from common.lob import merge_line
    from common.schemas import is_common_model

    texts: dict[str, list[str]] = defaultdict(list)
    scans: dict[str, int] = defaultdict(int)
    for metadata_path in sorted(bundles.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        lob = str(merge_line(metadata.get("lob") or ""))
        if str(metadata.get("split", "")).lower() != split or not lob:
            continue
        if not is_common_model("policy", None, lob):
            continue
        pdfs = sorted(metadata_path.parent.glob("*.pdf"))
        if not pdfs:
            continue
        with pymupdf.open(pdfs[0]) as document:
            text = " ".join(page.get_text() for page in document)
        if not text.strip():
            scans[lob] += 1
            continue
        texts[lob].append(normalise(text))
    return texts, scans


def coverage(texts: list[str], aliases: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    return {
        path: {label: sum(1 for text in texts if printed(label, text)) for label in labels}
        for path, labels in aliases.items() if labels
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundles", type=Path, required=True, help="root of the bundles to read")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "alias_coverage_schema2.json")
    args = parser.parse_args(argv)

    texts, scans = read_texts(args.bundles, args.split)
    report: dict[str, object] = {"split": args.split, "lines": {}}
    for lob in sorted(texts):
        counts = coverage(texts[lob], common_model_aliases(lob))
        gaps = sorted({label for labels in counts.values() for label, n in labels.items() if n == 0})
        report["lines"][lob] = {
            "documents": len(texts[lob]),
            "scans_without_text": scans.get(lob, 0),
            "labels": sum(len(labels) for labels in counts.values()),
            "labels_never_printed": len(gaps),
            "seen_labels": sorted({label for labels in counts.values() for label, n in labels.items() if n}),
            "gaps": gaps,
            "by_field": counts,
        }
        print(f"{lob:22s} docs {len(texts[lob]):4d}  labels {report['lines'][lob]['labels']:4d}  "
              f"never printed {len(gaps):4d}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
