"""The extraction/testing harness (SPEC_12, arch §17).

Runs any model version over real PDFs and produces JSON, per-field confidence,
and quality metrics — organised by model version so results are never ambiguous
about their origin.

**It is a thin wrapper over** :func:`serving.pipeline.extract`. Steps 2–7 of the
routine (classify, route, generate, confidence, validate) are the serving
pipeline's, called rather than reproduced. Reimplementing them here would give
two extraction paths and make every number this harness produces a description of
a system that is not the one in production.

``--model`` accepts ``base`` as a first-class value: the untuned model with no
adapter. That is the same path used by the pilot's zero-shot baseline (SPEC_15)
and day-zero pre-annotation (SPEC_04), so all three share one implementation.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from calibration.fit_calibration import CalibrationParams
from serving.pipeline import ExtractionRequest, ExtractionResult, extract

log = logging.getLogger(__name__)

TESTING_ROOT = Path(__file__).resolve().parent


class HarnessError(RuntimeError):
    """Raised when the harness cannot run a document."""


@dataclass
class RunSummary:
    """Batch aggregates — the file compared across model versions.

    This is the testing-time mirror of the SPEC_08 promotion gate: it answers
    "did v3 beat v2?" using the same metrics the gate reads.
    """

    model_version: str
    documents: int = 0
    schema_valid: int = 0
    mean_confidence: float = 0.0
    mean_latency_ms: float = 0.0
    by_doc_type: dict[str, int] = field(default_factory=dict)
    by_mode: dict[str, int] = field(default_factory=dict)
    needing_review: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def schema_validity_rate(self) -> float:
        return self.schema_valid / self.documents if self.documents else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "documents": self.documents,
            "schema_validity_rate": round(self.schema_validity_rate, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "documents_needing_review": self.needing_review,
            "by_doc_type": self.by_doc_type,
            "by_mode": self.by_mode,
            "failed": self.failed,
            **self.metrics,
        }


def results_dir(model_version: str, root: Path = TESTING_ROOT) -> Path:
    """``results/{version}/`` — the version is visible in the path itself."""
    return root / "results" / model_version


def metrics_dir(model_version: str, root: Path = TESTING_ROOT) -> Path:
    return root / "metrics" / model_version


def score_against_ground_truth(
    result: ExtractionResult, golden: dict[str, Any]
) -> dict[str, Any]:
    """Per-document metrics, using the SPEC_08 modules.

    The same metric code the promotion gate uses, so "correct" means the same
    thing here as it does there.
    """
    from evaluation.metrics.field_accuracy import score_all_list_fields, score_fields

    accuracy = score_fields(golden, result.extraction)
    list_reports = score_all_list_fields(golden, result.extraction)

    return {
        "field_exact_match_rate": round(accuracy.exact_match, 4),
        "field_normalized_match_rate": round(accuracy.normalized_match, 4),
        "field_accuracy_by_field": accuracy.by_field(),
        "list_field_f1": {n: round(r.f1, 4) for n, r in list_reports.items()},
        "list_field_recall": {n: round(r.recall, 4) for n, r in list_reports.items()},
        "missed_rows": {n: r.missed_rows for n, r in list_reports.items()},
        "failures": [
            {"field": f.field_path, "expected": f.expected, "got": f.got}
            for f in accuracy.failures()
        ],
    }


def run_document(
    request: ExtractionRequest,
    model: Any,
    classifier: Any,
    calibration: CalibrationParams,
    *,
    golden: dict[str, Any] | None = None,
    **pipeline_kwargs: Any,
) -> tuple[ExtractionResult, dict[str, Any]]:
    """Extract one document through the serving pipeline, then score it.

    ``golden`` is optional. **Confidence is emitted either way** — that is the
    payoff of the calibration work: on a document nobody has labeled, confidence
    is still available to route low-confidence fields to review, which is exactly
    the production case (arch §17).
    """
    result = extract(request, model, classifier, calibration, **pipeline_kwargs)

    metrics: dict[str, Any] = {
        "document": request.source_id,
        "model_version": result.model_version,
        "doc_type": result.doc_type,
        "mode": result.mode,
        "schema_valid": result.schema_valid,
        "overall_confidence": result.overall_confidence,
        "pages_used": result.pages_used,
        "review_flags": result.review_flags,
        "fields": {
            path: {**value, "correct": None}
            for path, value in result.fields.items()
        },
    }

    if golden is not None:
        scored = score_against_ground_truth(result, golden)
        metrics.update(scored)
        for path, value in metrics["fields"].items():
            expected = golden.get(path)
            from common.normalize import values_match

            value["correct"] = values_match(expected, result.extraction.get(path), field_path=path)
    else:
        metrics["ground_truth"] = "not supplied — confidence reported, accuracy not measurable"

    return result, metrics


def write_outputs(
    result: ExtractionResult,
    metrics: dict[str, Any],
    *,
    root: Path = TESTING_ROOT,
) -> tuple[Path, Path]:
    """Write ``results/{version}/{doc}.json`` and its metrics file."""
    version = result.model_version
    results_path = results_dir(version, root) / f"{result.source_id}.json"
    metrics_path = metrics_dir(version, root) / f"{result.source_id}.metrics.json"

    for path, payload in ((results_path, result.as_dict()), (metrics_path, metrics)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return results_path, metrics_path


def append_registry(
    result: ExtractionResult,
    results_path: Path,
    metrics_path: Path,
    *,
    root: Path = TESTING_ROOT,
    quant_format: str | None = None,
) -> None:
    """Record provenance so an output JSON is never ambiguous about its origin.

    ``model_version`` here is the same tag that resolves to a ``run_manifest``,
    so a result traces back to the corpus version and git commit that produced
    the model that generated it (arch §17).
    """
    registry_path = root / "extraction_registry.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists() else {"extractions": []}
    )
    registry["extractions"].append({
        "document": result.source_id,
        "doc_type": result.doc_type,
        "model_version": result.model_version,
        "quant_format": quant_format,
        "mode": result.mode,
        "result_path": str(results_path.relative_to(root)),
        "metrics_path": str(metrics_path.relative_to(root)),
        "overall_confidence": result.overall_confidence,
        "schema_valid": result.schema_valid,
        "review_flags": result.review_flags,
        "extracted_at": datetime.now(UTC).isoformat(),
    })
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def summarise(results: list[tuple[ExtractionResult, dict[str, Any]]], model_version: str) -> RunSummary:
    """Aggregate a batch into the cross-version comparison file."""
    summary = RunSummary(model_version=model_version, documents=len(results))
    if not results:
        return summary

    confidences, latencies = [], []
    for result, _metrics in results:
        summary.schema_valid += int(result.schema_valid)
        summary.by_doc_type[result.doc_type] = summary.by_doc_type.get(result.doc_type, 0) + 1
        summary.by_mode[result.mode] = summary.by_mode.get(result.mode, 0) + 1
        summary.needing_review += int(result.needs_review)
        confidences.append(result.overall_confidence)
        if result.latency_ms is not None:
            latencies.append(result.latency_ms)

    summary.mean_confidence = sum(confidences) / len(confidences)
    summary.mean_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    scored = [m for _r, m in results if "field_normalized_match_rate" in m]
    if scored:
        summary.metrics["mean_field_normalized_match"] = round(
            sum(m["field_normalized_match_rate"] for m in scored) / len(scored), 4
        )
        recalls = [
            value
            for m in scored
            for value in (m.get("list_field_recall") or {}).values()
        ]
        if recalls:
            summary.metrics["mean_list_field_recall"] = round(sum(recalls) / len(recalls), 4)
    return summary


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Extract with a chosen model version")
    parser.add_argument("--model", required=True,
                        help="base | v1 | v2 ... — 'base' is the untuned model, no adapter")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mode", choices=["ocr_plus_image", "image_only"], default="ocr_plus_image")
    parser.add_argument("--ground-truth", type=Path, default=None,
                        help="optional; confidence is emitted either way")
    parser.add_argument("--format", dest="quant_format", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tenant", default=None)
    # Parsed for its validation and --help, then discarded: the CLI cannot run
    # until a real model backend exists, and accepting bad arguments silently
    # would be worse than the explicit exit below.
    parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(
        "This CLI needs a live model backend, which is the one thing still gated on the Phase 0 "
        "dependency spike (vLLM multi-LoRA for Qwen3-VL, and MinerU on GPU for the OCR step). "
        "The loop itself is three calls: load_model(args.model), OCR each PDF via SPEC_03, then "
        "run_document() per document. Everything it calls is complete and tested — and it must "
        "keep calling serving.pipeline.extract rather than reimplementing it, or these numbers "
        "stop describing the system that runs in production."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
