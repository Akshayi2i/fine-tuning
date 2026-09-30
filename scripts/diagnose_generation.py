"""Show what vLLM actually returns for validation rows, and why a row fails to parse.

    python scripts/diagnose_generation.py --checkpoint <dir> [--source policy_0034] [--rows 8]
    python scripts/diagnose_generation.py --checkpoint <dir> --against-base

``--against-base`` scores the untouched base model and the checkpoint on every
validation row, the way checkpoint selection scores, and prints them side by
side: whether fine-tuning beat the model it started from.

Checkpoint selection logs only the parse error ("Unterminated string ..."), which
cannot tell a model that loops from a decoder that stops mid-value. This runs a
few validation rows through the same engine and records, per row and per mode
(schema-constrained, as scoring runs, and unconstrained), how generation ended,
how long the answer is, and its first and last characters. Needs the GPU to
itself: stop any running pipeline first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _describe(result, schema_used: bool) -> dict:
    if isinstance(result, BaseException):
        return {"constrained": schema_used, "error": f"{type(result).__name__}: {result}"}
    try:
        json.loads(result.text)
        parse = "ok"
    except ValueError as exc:
        parse = str(exc)
    return {
        "constrained": schema_used,
        "finish_reason": result.finish_reason,
        "tokens": len(result.tokens),
        "chars": len(result.text),
        "parse": parse,
        "head": result.text[:160],
        "tail": result.text[-240:],
        "last_tokens": result.tokens[-6:],
    }


def main(argv: list[str] | None = None) -> int:
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient
    from common.schemas import resolved_schema, with_page_bounds
    from evaluation.validation_generation import read_rows, split_prompt
    from inference_core.input_builder import page_total
    from inference_core.model_runner import generate_batch, load_model, release_model
    from training.stage_data import localize_rows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None, help="adapter directory to apply")
    parser.add_argument("--corpus", default="v0")
    parser.add_argument("--tenant", default="smoke")
    parser.add_argument("--scope", default="personal_lines")
    parser.add_argument("--source", default=None, help="only rows of this source_id")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--out", default=None, help="report path (default under /workspace/logs)")
    parser.add_argument("--against-base", dest="against_base", action="store_true",
                        help="score base and checkpoint on every validation row")
    parser.add_argument("--rescore", default=None, metavar="REPORT",
                        help="score a saved --against-base report again, without the GPU")
    parser.add_argument("--show-lists", dest="show_lists", default=None, metavar="REPORT",
                        help="print expected vs written table rows from a saved report")
    parser.add_argument("--errors", default=None, metavar="REPORT",
                        help="count every error type per model from a saved report")
    args = parser.parse_args(argv)
    if args.errors:
        breakdown = error_breakdown(args.errors)
        labels = list(breakdown)
        print(f"{'':<40}" + "".join(f"{label:>14}" for label in labels))
        keys = sorted({k for b in breakdown.values() for k in b["stats"]},
                      key=lambda k: ("table" in k, "unusable" in k or "lost" in k, k))
        for key in keys:
            print(f"{key:<40}" + "".join(f"{breakdown[x]['stats'].get(key, 0):>14}" for x in labels))
        print("errors by type:")
        for kind in ERROR_TYPES:
            print(f"  {kind:<38}" + "".join(f"{breakdown[x]['errors'][kind]:>14}" for x in labels))
        for label in labels:
            print(f"most affected fields ({label}):")
            for kind in ERROR_TYPES:
                top_fields = ", ".join(f"{f} ({n})" for f, n in breakdown[label]["top_fields"][kind])
                if top_fields:
                    print(f"  {kind}: {top_fields}")
        return 0
    if args.show_lists:
        for line in list_details(args.show_lists):
            print(line)
        return 0
    if args.rescore:
        rescored = rescore(args.rescore)
        for name in COMPARED:
            values = [rescored.get(label, {}).get(name) for label in ("base", "checkpoint")]
            print(f"{name:<32}" + "".join(f"{v:>12.3f}" if v is not None else f"{'-':>12}"
                                          for v in values))
        return 0
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --rescore is given")

    # On the pod, run detached in tmux: loading vLLM takes minutes.
    from orchestration.detach import detach_script_if_needed

    if detach_script_if_needed(__file__, argv, "diagnose"):
        return 0

    client = BlobClient()
    rows = read_rows(client.read_text(
        paths.corpus_scope_eval_split(args.corpus, "val", args.scope, args.tenant)))
    if args.source:
        rows = [r for r in rows if r.get("source_id") == args.source]
    if not args.against_base:
        rows = rows[: args.rows]
    if not rows:
        print("no validation rows matched", file=sys.stderr)
        return 1
    rows = localize_rows(rows, client, paths.staging_train_images_dir(args.corpus, args.tenant))
    if args.against_base:
        return _against_base(rows, client, args)
    args.out = args.out or "/workspace/logs/diagnose_generation.json"

    prompts = [split_prompt(row)[0] for row in rows]
    schemas = [with_page_bounds(
                   resolved_schema(r["doc_type"], r.get("acord_form"), r.get("lob"), r.get("sections")),
                   page_total(prompt))
               for r, prompt in zip(rows, prompts, strict=True)]
    model = load_model("base", client)
    report = []
    try:
        constrained = generate_batch(model, list(zip(prompts, schemas, strict=True)),
                                     adapter=args.checkpoint)
        free = generate_batch(model, [(p, None) for p in prompts], adapter=args.checkpoint)
        for row, c, f in zip(rows, constrained, free, strict=True):
            report.append({
                "source_id": row.get("source_id"),
                "sections": row.get("sections"),
                "lob": row.get("lob"),
                "modality_mode": row.get("modality_mode"),
                "with_schema": _describe(c, True),
                "without_schema": _describe(f, False),
            })
    finally:
        release_model(model)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for entry in report:
        for mode in ("with_schema", "without_schema"):
            d = entry[mode]
            print(f"{entry['source_id']} {entry['sections'] or '-':<24} {mode:<15} "
                  f"finish={d.get('finish_reason')} tokens={d.get('tokens')} "
                  f"parse={'ok' if d.get('parse') == 'ok' else 'FAIL'}")
    print(f"full report: {args.out}")
    return 0


def _saved(generation) -> dict:
    """One generation as the report keeps it: the row's scoring metadata, the
    label, the answer. Page images are kept as a count, which is all scoring reads."""
    row = generation.row
    images = sum(1 for m in row.get("messages", []) if isinstance(m.get("content"), list)
                 for part in m["content"] if isinstance(part, dict) and part.get("type") == "image")
    slim = {k: row.get(k) for k in ("source_id", "doc_type", "acord_form", "lob", "sections",
                                    "modality_mode", "is_scanned")}
    slim["messages"] = [{"role": "user", "content": [{"type": "image"}] * images}]
    return {"row": slim, "golden": generation.golden, "extraction": generation.extraction,
            "error": generation.error, "failure_kind": generation.failure_kind}


def rescore(report_path: str) -> dict[str, dict]:
    """Score a saved comparison again with the current scorer. No GPU."""
    from evaluation.validation_generation import ValidationGeneration, score_generations

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rescored = {}
    for label, entry in report.items():
        generations = [ValidationGeneration(**saved) for saved in entry.get("generations", [])]
        if generations:
            metrics = score_generations(generations, model_version=label)
            rescored[label] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    return rescored


def list_details(report_path: str, limit: int = 6) -> list[str]:
    """For each table in a saved report: what the label holds and what each model wrote.

    One line per (model, row, table): the line of business, the identifier the
    scorer matched on, and the first identifiers on each side. No GPU.
    """
    from common.canonical import values_view
    from evaluation.metrics.field_accuracy import _at, _infer_key_fields, find_list_fields

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    lines: list[str] = []
    for label, entry in report.items():
        shown = 0
        for saved in entry.get("generations", []):
            expected = values_view(saved.get("golden") or {})
            got = values_view(saved.get("extraction") or {})
            for path, rows in find_list_fields(expected).items():
                keys = _infer_key_fields(rows)
                got_rows = _at(got, path) or []
                def ident(r, keys=keys):
                    return tuple(r.get(k) for k in keys) if isinstance(r, dict) else r

                lines.append(
                    f"{label:<10} {saved['row'].get('source_id')} lob={saved['row'].get('lob')} "
                    f"{path} key={keys}: expected {len(rows)} {[ident(r) for r in rows[:3]]} "
                    f"| wrote {len(got_rows)} {[ident(r) for r in got_rows[:3]]}")
                shown += 1
                if shown >= limit:
                    break
            if shown >= limit:
                break
    return lines


#: Error types, in the order they are reported.
ERROR_TYPES = ("left empty", "wrong value", "value from another field", "misread (near miss)",
               "invented (not in the label)")


def error_breakdown(report_path: str, top: int = 3) -> dict[str, dict]:
    """Every wrong field in a saved report, by type, per model. No GPU.

    Scored as checkpoint selection scores (``score_fields``), classified by the
    gate's own ``classify_error``, with its catch-all split in two: a value that
    belongs to another field of the same document, and a plain wrong value.
    Answers that were not usable JSON are counted apart, since every field in
    them is lost to the same cause.
    """
    from collections import Counter

    from common.canonical import values_view
    from evaluation.metrics.field_accuracy import (
        _at,
        _infer_key_fields,
        _row_key,
        find_list_fields,
        flatten_scalars,
        score_fields,
    )
    from training.vit_gate import classify_error

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for label, entry in report.items():
        counts: Counter = Counter()
        fields: dict[str, Counter] = {t: Counter() for t in ERROR_TYPES}
        stats = Counter()
        for saved in entry.get("generations", []):
            stats["rows"] += 1
            golden, got = saved.get("golden") or {}, saved.get("extraction") or {}
            if saved.get("error"):
                stats["unusable answers"] += 1
                stats["fields lost in unusable answers"] += len(flatten_scalars(golden))
                continue
            accuracy = score_fields(golden, got)
            stats["fields scored"] += accuracy.total
            stats["fields correct"] += sum(r.correct for r in accuracy.results)
            others = {str(v).strip() for v in flatten_scalars(golden).values() if v is not None}
            for r in accuracy.failures():
                kind = classify_error(r.expected, r.got, all_expected=golden)
                if kind == "omission":
                    name = "left empty"
                elif kind == "unknown":
                    name = "invented (not in the label)"
                elif kind == "perception":
                    name = "misread (near miss)"
                elif str(r.got).strip() in others:
                    name = "value from another field"
                else:
                    name = "wrong value"
                counts[name] += 1
                fields[name][r.field_path.split("[")[0]] += 1
            expected_v, got_v = values_view(golden), values_view(got)
            for path, rows in find_list_fields(expected_v).items():
                keys = _infer_key_fields(rows)
                got_rows = [r for r in (_at(got_v, path) or []) if isinstance(r, dict)]
                expected_keys = Counter(_row_key(r, keys) for r in rows)
                got_keys = Counter(_row_key(r, keys) for r in got_rows)
                found = sum((expected_keys & got_keys).values())
                stats["table rows expected"] += len(rows)
                stats["table rows found"] += found
                stats["table rows missed"] += len(rows) - found
                stats["table rows written with no/unknown ID"] += len(got_rows) - found
        out[label] = {"stats": dict(stats), "errors": {t: counts[t] for t in ERROR_TYPES},
                      "top_fields": {t: fields[t].most_common(top) for t in ERROR_TYPES}}
    return out


#: What the comparison prints, in order; the rest is in the report file.
COMPARED = ("field_normalized_match", "schema_validity_rate", "false_null_rate",
            "image_only_accuracy", "ocr_arbitration_accuracy", "list_field_recall",
            "confusable_misattribution_rate")


def _against_base(rows: list[dict], client, args) -> int:
    """Base model and checkpoint on the same rows, scored by the gate's own report."""
    from evaluation.validation_generation import generate_validation, score_generations
    from inference_core.model_runner import load_model, release_model

    model = load_model("base", client)
    results: dict[str, dict] = {}
    try:
        for label, adapter in (("base", None), ("checkpoint", args.checkpoint)):
            generations = generate_validation(rows, model, adapter=adapter)
            metrics = score_generations(generations, model_version=label)
            results[label] = {
                "adapter": adapter,
                "rows": len(generations),
                "unusable_json": sum(g.failure_kind == "output" for g in generations),
                "setup_failures": sum(bool(g.error) and g.failure_kind != "output"
                                      for g in generations),
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                # Every answer with its label, so a scoring question can be
                # answered again (--rescore) without the GPU.
                "generations": [_saved(g) for g in generations],
            }
    finally:
        release_model(model)

    out = Path(args.out or "/workspace/logs/base_vs_checkpoint.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    base, tuned = results["base"], results["checkpoint"]
    print()
    print(f"{'metric':<32}{'base':>10}{'checkpoint':>12}{'change':>10}")
    for name in COMPARED:
        b, t = base["metrics"].get(name), tuned["metrics"].get(name)
        if b is None or t is None:
            print(f"{name:<32}{'-':>10}{'-':>12}")
            continue
        print(f"{name:<32}{b:>10.3f}{t:>12.3f}{t - b:>+10.3f}")
    for key in ("unusable_json", "setup_failures"):
        print(f"{key:<32}{base[key]:>10}{tuned[key]:>12}")
    print(f"rows scored: {base['rows']}  (false_null_rate and confusable: lower is better)")
    print(f"full report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
