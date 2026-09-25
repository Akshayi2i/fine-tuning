"""Generate over the validation split once, and read it three ways (arch v2.1 §8.2, §11.2, §5.3).

Checkpoint selection, calibration and the promotion gate all need the same
thing: the model's own generations over validation documents, scored against
their golden labels. Three separate generators would mean three definitions of
"correct", and a selector that scores by one definition picks a checkpoint the
gate, scoring by another, rejects. So there is one generator here and three
readers:

* :func:`score_generations` — the gate's metrics, through
  :func:`evaluation.run_eval.build_report`. Checkpoint selection reads its
  ``field_normalized_match``.
* :func:`calibration_samples` — ``(features, correct)`` per field, split into the
  calibration and threshold halves the dataset build stamped on each row.
* :func:`vllm_scorer` in :mod:`evaluation.checkpoint_eval` — the first reader
  applied per checkpoint, with the checkpoint as a decoder LoRA.

The prompt is the row's own ``messages`` with the assistant turn removed, so the
model sees exactly what training rendered for that document (arch §7).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ValidationGeneration:
    """One validation row, generated and parsed."""

    row: dict[str, Any]
    golden: dict[str, Any]
    extraction: dict[str, Any] | None = None
    logprobs_by_path: dict[str, list[float]] = field(default_factory=dict)
    error: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """What :func:`evaluation.run_eval.subset_of` reads."""
        images = sum(
            1
            for message in self.row.get("messages", [])
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "image"
        )
        return {
            "source_id": self.row.get("source_id", ""),
            "doc_type": self.row.get("doc_type", "unknown"),
            # Both select the schema the output is validated against. Without the
            # form every ACORD row scored as unselectable; without the line every
            # policy was judged against the canonical fallback.
            "acord_form": self.row.get("acord_form"),
            "lob": self.row.get("lob"),
            "modality_mode": self.row.get("modality_mode", "ocr_plus_image"),
            # Not on the row. Unknown is recorded as not scanned, which only
            # affects which eval subset a document is also counted in.
            "is_scanned": bool(self.row.get("is_scanned", False)),
            "page_count": images or 1,
        }

    @property
    def page_text(self) -> str | None:
        """The OCR text the prompt carried, for the OCR-agreement feature."""
        texts = [
            part.get("text", "")
            for message in self.row.get("messages", [])
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(texts) or None


def read_rows(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def split_prompt(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The prompt messages and the golden label from one corpus row."""
    messages = list(row.get("messages") or [])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            f"row {row.get('source_id')!r} has no trailing assistant turn, so it carries no "
            "golden label to score against"
        )
    content = messages[-1].get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return messages[:-1], json.loads(content)


def generate_validation(
    rows: Iterable[dict[str, Any]],
    model: Any,
    *,
    adapter: str | None = None,
    constrain: bool | None = None,
) -> list[ValidationGeneration]:
    """Generate every row once. A row that fails is recorded, never dropped.

    ``constrain`` defaults to what serving does. Calibration must read the
    distribution serving will produce; a calibrator fitted on unconstrained
    generations describes a model nobody serves.
    """
    from common.canonical import collapse_spans, with_output_dates
    from common.schemas import resolved_schema
    from inference_core.model_runner import generate
    from inference_core.span_map import map_field_spans

    if constrain is None:
        constrain = bool(getattr(model.config, "structured_outputs", False))

    out: list[ValidationGeneration] = []
    for row in rows:
        messages, golden = split_prompt(row)
        entry = ValidationGeneration(row=row, golden=golden)
        schema = (
            resolved_schema(row["doc_type"], row.get("acord_form"), row.get("lob"))
            if constrain else None
        )
        try:
            result = generate(model, messages, adapter=adapter, json_schema=schema)
            extraction = json.loads(result.text)
            if not isinstance(extraction, dict):
                raise TypeError(f"expected a JSON object, got {type(extraction).__name__}")
            # Keyed and formatted exactly as serving does it (serving.pipeline),
            # or the calibrators are fitted on paths and values serving never
            # looks up.
            spans = collapse_spans(
                map_field_spans(result.text, result.tokens, result.token_logprobs)
            )
            extraction = with_output_dates(extraction)
        except Exception as exc:  # noqa: BLE001 - one bad row must not lose the rest
            # Scored as an empty extraction, not skipped: a model that cannot
            # produce JSON for a document has got every field on it wrong, and
            # leaving it out would score the model on the documents it managed.
            entry.error = f"{type(exc).__name__}: {exc}"
            entry.extraction = {}
            log.warning("validation generation failed for %s: %s", row.get("source_id"), exc)
        else:
            entry.extraction = extraction
            entry.logprobs_by_path = {
                path: span.token_logprobs for path, span in spans.items() if span.mapped
            }
        out.append(entry)
    return out


def score_generations(
    generations: Sequence[ValidationGeneration], *, model_version: str = "validation"
) -> dict[str, Any]:
    """The gate's metrics over these generations — the one definition of correct."""
    from evaluation.run_eval import build_report

    report = build_report(
        model_version,
        [(g.golden, g.extraction or {}, g.metadata) for g in generations],
    )
    return dict(report.gate_metrics())


def calibration_samples(
    generations: Sequence[ValidationGeneration],
) -> dict[str, list[tuple[Any, bool]]]:
    """``(features, correct)`` per field, by validation half (arch v2.1 §8.2).

    Rows with no ``val_half`` are left out rather than guessed into a half:
    fitting a calibrator and choosing its threshold on the same rows prices risk
    the calibrator has already been pulled toward.
    """
    from calibration.features import build_document_features
    from common.normalize import values_match
    from evaluation.metrics.field_accuracy import flatten_scalars

    halves: dict[str, list[tuple[Any, bool]]] = {"calibration": [], "threshold": []}
    unassigned = 0
    for generation in generations:
        half = generation.row.get("val_half")
        if half not in halves:
            unassigned += 1
            continue
        if generation.error:
            # No generation means no features. The failure is already counted in
            # the scored metrics; it has no token evidence to calibrate on.
            continue
        expected = flatten_scalars(generation.golden)
        for features in build_document_features(
            extraction=generation.extraction or {},
            spans=generation.logprobs_by_path,
            page_text=generation.page_text,
        ):
            correct = values_match(
                expected.get(features.field_path), features.value, field_path=features.field_path
            )
            halves[half].append((features, bool(correct)))

    if unassigned:
        log.warning(
            "%d validation row(s) carry no val_half and were left out of calibration. "
            "Rebuild the corpus: the dataset build stamps every val row with its half.",
            unassigned,
        )
    return halves
