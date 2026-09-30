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
    parser.add_argument("--checkpoint", required=True, help="adapter directory to apply")
    parser.add_argument("--corpus", default="v0")
    parser.add_argument("--tenant", default="smoke")
    parser.add_argument("--scope", default="personal_lines")
    parser.add_argument("--source", default=None, help="only rows of this source_id")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--out", default=None, help="report path (default under /workspace/logs)")
    parser.add_argument("--against-base", dest="against_base", action="store_true",
                        help="score base and checkpoint on every validation row")
    args = parser.parse_args(argv)

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
