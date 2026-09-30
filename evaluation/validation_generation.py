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
    #: ``"setup"`` when nothing usable came back (the row could not be prepared,
    #: the request failed); ``"output"`` when the model answered with something
    #: that is not a JSON object - a wrong answer, scored as one.
    failure_kind: str | None = None

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
            # A policy row is one window: its output is judged against the slice
            # it was asked for, not the whole schema it is a part of.
            "sections": self.row.get("sections"),
            "modality_mode": self.row.get("modality_mode", "ocr_plus_image"),
            # Not on the row. Unknown is recorded as not scanned, which only
            # affects which eval subset a document is also counted in.
            "is_scanned": bool(self.row.get("is_scanned", False)),
            "page_count": images or 1,
        }

    @property
    def page_text(self) -> str | None:
        """The OCR text the prompt carried, for the OCR-agreement feature."""
        from calibration.features import shown_ocr_text

        if self.row.get("modality_mode") == "image_only":
            return None
        return shown_ocr_text([
            part.get("text", "")
            for message in self.row.get("messages", [])
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "text"
        ])


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


#: Rows per ``generate_batch`` call. vLLM schedules a call's requests together
#: (continuous batching); one row at a time left the GPU waiting on each answer,
#: ~50 s a row - hours per checkpoint on the smoke set, days on the full corpus.
#: Bounded so a call's opened page images stay a modest amount of memory.
BATCH_ROWS = 32


def generate_validation(
    rows: Iterable[dict[str, Any]],
    model: Any,
    *,
    adapter: str | None = None,
    constrain: bool | None = None,
    batch_rows: int = BATCH_ROWS,
) -> list[ValidationGeneration]:
    """Generate every row once. A row that fails is recorded, never dropped.

    ``constrain`` defaults to what serving does. Calibration must read the
    distribution serving will produce; a calibrator fitted on unconstrained
    generations describes a model nobody serves.
    """
    from common.schemas import resolved_schema, with_page_bounds
    from inference_core.input_builder import page_total
    from inference_core.model_runner import generate, generate_batch

    if constrain is None:
        constrain = bool(getattr(model.config, "structured_outputs", False))

    out: list[ValidationGeneration] = []
    pending: list[tuple[ValidationGeneration, list[dict[str, Any]], dict[str, Any] | None]] = []
    for row in rows:
        entry = ValidationGeneration(row=row, golden={})
        out.append(entry)
        try:
            # Inside the per-row guard: a row with no assistant turn, or an ACORD
            # row with no form (no schema to select), used to raise out of the
            # loop and abort every other row's scoring with it.
            messages, entry.golden = split_prompt(row)
            # Page references bounded to the document's pages (with_page_bounds).
            schema = (
                with_page_bounds(
                    resolved_schema(
                        row["doc_type"], row.get("acord_form"), row.get("lob"), row.get("sections")
                    ),
                    page_total(messages),
                )
                if constrain else None
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not lose the rest
            _record_failure(entry, exc, "setup")
            continue
        pending.append((entry, messages, schema))

    size = max(1, batch_rows)
    for first in range(0, len(pending), size):
        chunk = pending[first:first + size]
        results = generate_batch(model, [(m, s) for _e, m, s in chunk], adapter=adapter)
        if len(chunk) > 1 and all(isinstance(r, Exception) for r in results):
            # A whole call fails together (one unreadable page image refuses the
            # batch): retry its rows alone, so the bad one is the only loss.
            results = []
            for _entry, messages, schema in chunk:
                try:
                    results.append(generate(model, messages, adapter=adapter, json_schema=schema))
                except Exception as exc:  # noqa: BLE001
                    results.append(exc)
        for (entry, _messages, _schema), result in zip(chunk, results, strict=True):
            _finish(entry, result)
        log.info("validation generation: %d/%d row(s)%s", min(first + size, len(pending)),
                 len(pending), f" ({adapter})" if adapter else "")
    return out


def _record_failure(entry: ValidationGeneration, exc: BaseException, kind: str,
                    note: str = "", evidence: str = "") -> None:
    # Scored as an empty extraction, not skipped: a model that cannot produce
    # JSON for a document has got every field on it wrong, and leaving it out
    # would score the model on the documents it managed.
    entry.error = f"{note}{type(exc).__name__}: {exc}{evidence}"
    entry.failure_kind = kind
    entry.extraction = {}
    log.warning("validation generation failed for %s: %s", entry.row.get("source_id"), entry.error)


def _finish(entry: ValidationGeneration, result: Any) -> None:
    """Parse one generation into ``entry``, or record why it failed."""
    from common.canonical import collapse_spans, with_output_dates
    from inference_core.span_map import map_field_spans

    if isinstance(result, BaseException):
        _record_failure(entry, result, "setup")
        return
    try:
        extraction = json.loads(result.text)
        if not isinstance(extraction, dict):
            raise TypeError(f"expected a JSON object, got {type(extraction).__name__}")
        # Keyed and formatted exactly as serving does it (serving.pipeline), or
        # the calibrators are fitted on paths and values serving never looks up.
        spans = collapse_spans(map_field_spans(result.text, result.tokens, result.token_logprobs))
        extraction = with_output_dates(extraction)
    except Exception as exc:  # noqa: BLE001 - one bad row must not lose the rest
        truncated = bool(getattr(result, "truncated", lambda: False)())
        text = getattr(result, "text", "") or ""
        # How generation ended and what it ended on: the parse error alone cannot
        # tell a model looping to the limit from a decode that stopped mid-value.
        evidence = (f" [finish={getattr(result, 'finish_reason', None)}, "
                    f"{len(getattr(result, 'tokens', []) or [])} tokens, {len(text)} chars, "
                    f"ends {text[-60:]!r}]")
        _record_failure(entry, exc, "output",
                        "stopped at the token limit before the JSON closed - " if truncated else "",
                        evidence)
        return
    entry.extraction = extraction
    entry.logprobs_by_path = {
        path: span.token_logprobs for path, span in spans.items() if span.mapped
    }


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


#: Above this share of rows with nothing usable back, a pass over validation is
#: not a measurement. Scoring the failures as empty answers turned a broken
#: setup — unreadable images, a model that would not load — into a checkpoint
#: "chosen" at 0.0 and a calibrator fitted on nothing, with no error anywhere.
MAX_GENERATION_FAILURE_RATE = 0.10

#: Rows the model answered with unusable JSON are wrong answers, not a broken
#: pass, and are scored as such: a checkpoint that loops on 14% of documents has
#: to lose to one that does not, not be dropped from the comparison. Counting
#: them against the 10% above refused every checkpoint of the smoke run. Only
#: when most rows are unusable is there too little left to rank on.
MAX_UNUSABLE_OUTPUT_RATE = 0.50


class ValidationGenerationError(RuntimeError):
    """Raised when too many validation rows failed to generate to trust a score."""


def assert_generations_usable(
    generations: list[ValidationGeneration], *, what: str,
    max_failure_rate: float = MAX_GENERATION_FAILURE_RATE,
) -> None:
    """Refuse a pass that measured nothing.

    More than ``max_failure_rate`` of rows with nothing back is a broken setup.
    Rows the model answered with unusable JSON are wrong answers and stay in the
    score, until they are most of the pass (:data:`MAX_UNUSABLE_OUTPUT_RATE`).
    """
    if not generations:
        raise ValidationGenerationError(f"{what}: no validation rows were generated")
    failed = [g for g in generations if g.error and g.failure_kind != "output"]
    unusable = [g for g in generations if g.error and g.failure_kind == "output"]
    if unusable:
        log.info("%s: %d of %d row(s) answered with unusable JSON, scored as wrong answers",
                 what, len(unusable), len(generations))
    if len(unusable) / len(generations) > MAX_UNUSABLE_OUTPUT_RATE:
        sample = sorted({g.error for g in unusable})[:3]
        raise ValidationGenerationError(
            f"{what}: {len(unusable)} of {len(generations)} validation rows came back as "
            f"unusable JSON (e.g. {sample}). With most answers unreadable there is too little "
            "left to rank or calibrate on."
        )
    if len(failed) / len(generations) > max_failure_rate:
        sample = sorted({g.error for g in failed})[:3]
        raise ValidationGenerationError(
            f"{what}: {len(failed)} of {len(generations)} validation rows failed to generate "
            f"(e.g. {sample}). That is a broken pass, not a score — scoring the failures as "
            "empty answers would pick or fit on nothing."
        )
