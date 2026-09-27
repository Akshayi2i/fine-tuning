"""The request pipeline — the canonical extraction path (SPEC_11).

Every extraction goes through here: production requests and the testing harness
alike. **The harness calls this function; it does not reimplement it.** That is
the whole basis of "test == prod" — two code paths would be two systems, and the
tests would describe the wrong one.

    OCR (if provided)
      -> classify document type (+ ACORD form)
      -> route adapter / prompt / schema
      -> page-route if the document is long
      -> generate with logprobs
      -> per-field confidence, calibrated
      -> list-completeness cross-check
      -> validate against the schema
      -> JSON + confidence + review flags

The schema validation at the end mirrors the Fideon SPEC_07 Stage 3 audit gate: a
schema-invalid response is an **error**, not a returned result.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from calibration.apply_calibration import CalibratedField, CalibratedResult, apply_calibration
from calibration.features import shown_ocr_text
from calibration.fit_calibration import CalibrationParams
from calibration.list_completeness import (
    check_document,
    is_row_completeness_flag,
    merge_review_flags,
)
from calibration.logprob_confidence import field_confidences
from common.canonical import collapse_spans, envelope, values_view, with_output_dates
from common.constants import DEFAULT_LONG_DOC_PAGE_THRESHOLD, DEFAULT_REVIEW_CONFIDENCE_THRESHOLD
from common.schemas import is_canonical, is_valid, iter_validation_errors, resolved_schema
from inference_core.input_builder import EMPTY_PAGE_TEXT, build_messages
from inference_core.model_runner import Generation, LoadedModel, generate, generate_batch
from inference_core.span_map import SpanMapError, map_field_spans
from serving.adapter_router import Route, RoutingError, route
from serving.doc_type_classifier import Classifier
from serving.page_router import plan_pages

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when a request cannot produce a valid extraction."""


@dataclass
class ExtractionRequest:
    """One document to extract."""

    source_id: str
    image_paths: list[str]
    ocr_text: str | None = None
    page_texts: dict[int, str] = field(default_factory=dict)
    ocr_meta: dict[str, Any] = field(default_factory=dict)
    modality_mode: str = "ocr_plus_image"
    #: Skip classification when the caller already knows the type.
    known_doc_type: str | None = None
    known_acord_form: str | None = None
    #: A policy's line of business, when the caller (L1/L2) knows it. Selects the
    #: line's canonical schema; absent — or naming several lines — the policy is
    #: extracted into the client's canonical fallback. Either way the output is
    #: canonical JSON.
    known_lob: str | list[str] | None = None


@dataclass
class ExtractionResult:
    """The output contract (master §9)."""

    source_id: str
    doc_type: str
    model_version: str
    mode: str
    schema_valid: bool
    overall_confidence: float
    extraction: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    list_fields: dict[str, Any] = field(default_factory=dict)
    line_of_business: dict[str, Any] | None = None
    pages_used: list[int] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)
    route_info: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    validation_errors: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.review_flags)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "doc_type": self.doc_type,
            "model_version": self.model_version,
            "mode": self.mode,
            "schema_valid": self.schema_valid,
            "overall_confidence": self.overall_confidence,
            "line_of_business": self.line_of_business,
            "fields": self.fields,
            "list_fields": self.list_fields,
            "pages_used": self.pages_used,
            "review_flags": sorted(self.review_flags),
            "route": self.route_info,
        }


def _resolve_route(request: ExtractionRequest, classifier: Classifier, *,
                   confidence_threshold: float, adapter_map: dict[str, Any],
                   fallback_doc_type: str | None = None) -> Route:
    if request.known_doc_type:
        from serving.doc_type_classifier import StaticClassifier

        classifier = StaticClassifier(request.known_doc_type, request.known_acord_form)
    classification = classifier.classify(request.image_paths, request.ocr_text)
    try:
        return route(
            classification,
            confidence_threshold=confidence_threshold,
            adapter_map=adapter_map,
            fallback_doc_type=fallback_doc_type,
        )
    except RoutingError as exc:
        # Before generation, deliberately: a document nobody could route costs
        # nothing to refuse and a full extraction to answer wrongly.
        raise PipelineError(str(exc)) from exc


def _image_for(request: ExtractionRequest, page: int) -> str:
    """The image for one page, by the `page_N.` convention the OCR stage writes.

    Otherwise by position: page ``N`` is the ``N``-th image (the endpoint checks
    that ``page_texts`` numbers exactly 1..n against the images). The fallback
    used to index into ``page_texts``, which an image_only request does not
    have — so every page of it resolved to image 1 — and clamped an
    out-of-range page to the last image instead of saying it did not exist.
    """
    for path in request.image_paths:
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        if name.startswith(f"page_{page}."):
            return path
    if not 1 <= page <= len(request.image_paths):
        raise PipelineError(
            f"{request.source_id}: page {page} requested but only "
            f"{len(request.image_paths)} page image(s) were sent"
        )
    return request.image_paths[page - 1]


#: The blank-page placeholder, defined where training's messages are built too.
_EMPTY_PAGE = EMPTY_PAGE_TEXT


def _all_page_texts(request: ExtractionRequest) -> list[str]:
    """One markdown string per image, in page order.

    ``page_texts`` is the real source. ``ocr_text`` is accepted for a
    single-page request only: a joined blob cannot be split back into pages, and
    guessing where the seams were is exactly the failure this change removes.
    """
    if request.page_texts:
        return [request.page_texts[page] or _EMPTY_PAGE for page in sorted(request.page_texts)]
    if len(request.image_paths) == 1:
        return [request.ocr_text or ""]
    raise PipelineError(
        f"{request.source_id} supplies {len(request.image_paths)} page images but only a single "
        "joined ocr_text. Each image needs its own page text so the model can tell which text "
        "belongs to which page; populate page_texts from the per-page markdown the OCR stage "
        "wrote rather than joining it."
    )


def _build_request(
    model: LoadedModel,
    route_: Route,
    image_paths: list[str],
    ocr_pages: list[str] | None,
    modality_mode: str,
    *,
    page_numbers: list[int] | None = None,
    total_pages: int | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """The messages and the decoding schema for one generation.

    ``sections`` names the schema slice a policy window asks for; the prompt and
    the structured-decoding schema are both that slice, from the same selectors.
    """
    built = build_messages(
        route_.schema_doc_type,
        image_paths,
        ocr_pages,
        modality_mode,
        acord_form=route_.schema_acord_form,
        lob=lob,
        sections=sections,
        page_numbers=page_numbers,
        total_pages=total_pages,
    )
    # Constrained to the routed type's schema when serving is configured for it
    # (arch v2.1 §13). Without this the schema-validity floor in §13b measured an
    # unconstrained model, so one malformed bf16 output made every quantized
    # format unvalidatable.
    schema = (
        resolved_schema(route_.schema_doc_type, route_.schema_acord_form, lob, sections)
        if getattr(model.config, "structured_outputs", False) else None
    )
    return built.messages, schema


def _parse_generation(
    result: Generation, route_: Route, *, refuse_truncated: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(extraction, spans)`` from one generation, or :class:`PipelineError`.

    The extraction comes back with every date in ``MM/DD/YYYY`` and, for a
    canonical document, still in the model's sparse ``raw``/``parsed``/
    ``page_ref`` form; the spans are keyed by values-view path.

    ``refuse_truncated`` treats output cut off at the token limit as a failure
    even when what was cut still parses. A window can afford that: it is retried
    over fewer pages. Parsing a truncated answer as complete would drop every
    row after the cut with nothing to say so.
    """
    if refuse_truncated and result.truncated():
        raise PipelineError(
            f"the {route_.schema_doc_type} generation hit the token limit, so its output is "
            "incomplete"
        )
    # Parse and span-map together: both fail for the same underlying reason — the
    # model did not return JSON — and reporting that as two different errors from
    # two layers would obscure a single cause.
    try:
        extraction = json.loads(result.text)
        if not isinstance(extraction, dict):
            # A bare array or scalar parses cleanly and maps cleanly, then dies
            # with an AttributeError several layers down. The contract is one
            # JSON object, so this is the same generation failure as unparseable
            # output and is reported as one.
            raise TypeError(f"expected a JSON object, got {type(extraction).__name__}")
        spans = map_field_spans(result.text, result.tokens, result.token_logprobs)
    except (ValueError, TypeError, SpanMapError) as exc:
        raise PipelineError(
            f"the model did not return parseable JSON for {route_.schema_doc_type}: "
            f"{result.text[:160]!r}. The prompt asks for JSON only with no fences, so this is a "
            f"generation failure rather than a formatting quirk to work around ({exc})."
        ) from exc
    # Dates are reformatted AFTER the span map: the spans describe the tokens the
    # model generated, and a reformatted date is the same value in other
    # characters, not a different value with different evidence.
    return with_output_dates(extraction), collapse_spans(spans)


def _generate_once(
    model: LoadedModel,
    route_: Route,
    image_paths: list[str],
    ocr_pages: list[str] | None,
    modality_mode: str,
    *,
    page_numbers: list[int] | None = None,
    total_pages: int | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float | None]:
    """One scoped generation. Returns ``(extraction, spans, latency_ms)``."""
    messages, schema = _build_request(
        model, route_, image_paths, ocr_pages, modality_mode,
        page_numbers=page_numbers, total_pages=total_pages, lob=lob, sections=sections,
    )
    result = generate(model, messages, adapter=route_.adapter, json_schema=schema)
    extraction, spans = _parse_generation(result, route_)
    return extraction, spans, result.latency_ms


#: The review flag a window that could not be read leaves on the document. The
#: values its pages carried are missing, and nothing else would say so.
WINDOW_FAILED_FLAG = "window_failed"


def _extract_policy_windows(
    request: ExtractionRequest,
    model: LoadedModel,
    route_: Route,
    lob: str | list[str] | None,
    *,
    page_threshold: int,
) -> tuple[dict[str, Any], dict[str, Any], float | None, list[int], list[str]]:
    """Read a canonical policy window by window, then merge.

    Returns ``(extraction, spans, latency_ms, pages_used, flags)`` — the merged
    model-form document and its spans, in the shape the single-call path
    returns, so confidence, completeness and the envelope run on it unchanged.

    **Concurrent.** Every window of a round goes to the model in one batch; vLLM
    schedules them together, so a 17-window policy costs roughly the time of its
    slowest window rather than the sum of all of them. ``latency_ms`` is the
    wall time actually spent.

    **One bad window does not lose the document.** A window whose output is not
    parseable JSON, or was cut off at the token limit — the usual cause on a
    dense schedule page — is split in half and retried in the next round, since
    fewer pages means less to write. Retrying it unchanged would be pointless:
    decoding is greedy, so it would write the same thing again. A window that
    still fails at a single page is dropped and flagged ``window_failed``, so the
    document comes back with everything else and a person is told which pages
    it could not read. Only when no window at all succeeds is the request an
    error.
    """
    from data_pipeline.dataset_builder.policy_windows import plan_windows, routed_pages
    from serving.policy_merge import PolicyWindow, merge_policy_windows

    image_only = request.modality_mode == "image_only"
    page_count = len(request.page_texts) or len(request.image_paths)
    texts: dict[int, str] = {}
    if not image_only:
        texts = dict(request.page_texts) or dict(enumerate(_all_page_texts(request), start=1))

    # Routed on the same text, by the same rule, as the training rows were:
    # every page when there is no OCR text to score.
    routed, declarations_page = routed_pages(
        texts or None, page_count, page_threshold=page_threshold
    )
    pending: list[tuple[str, list[int]]] = [
        (plan.group, list(plan.pages)) for plan in plan_windows(lob, routed, declarations_page)
    ]

    windows: list[PolicyWindow] = []
    failed: list[str] = []
    first_cause: str | None = None
    wall_ms = 0.0
    rounds = 0
    while pending:
        rounds += 1
        requests = [
            _build_request(
                model, route_,
                [_image_for(request, page) for page in pages],
                None if image_only else [texts.get(page) or _EMPTY_PAGE for page in pages],
                request.modality_mode,
                page_numbers=pages, total_pages=page_count,
                lob=lob, sections=group,
            )
            for group, pages in pending
        ]
        started = time.perf_counter()
        results = generate_batch(model, requests, adapter=route_.adapter)
        wall_ms += (time.perf_counter() - started) * 1000

        retry: list[tuple[str, list[int]]] = []
        for (group, pages), result in zip(pending, results, strict=True):
            try:
                if isinstance(result, Exception):
                    raise PipelineError(str(result))
                extraction, spans = _parse_generation(result, route_, refuse_truncated=True)
            except PipelineError as exc:
                first_cause = first_cause or str(exc)
                if len(pages) > 1:
                    half = len(pages) // 2
                    retry += [(group, pages[:half]), (group, pages[half:])]
                    log.warning(
                        "%s: %s window over pages %s failed (%s); retrying as two smaller "
                        "windows", request.source_id, group, pages, exc,
                    )
                else:
                    failed.append(f"{group}:p{pages[0]}")
                    log.warning(
                        "%s: %s window over page %s failed even alone (%s); dropped and "
                        "flagged for review", request.source_id, group, pages[0], exc,
                    )
                continue
            windows.append(PolicyWindow(group, pages, extraction, spans, result.latency_ms))
        pending = retry

    if not windows:
        raise PipelineError(
            f"no window of {request.source_id} could be read ({len(failed)} failed: "
            f"{', '.join(failed[:6])}). The first failure: {first_cause}"
        )

    merged = merge_policy_windows(windows)
    flags = merged.review_flags + [f"{WINDOW_FAILED_FLAG}:{entry}" for entry in failed]
    log.info(
        "%s read in %d window(s) over %d routed page(s) in %d round(s); %d duplicate row(s) "
        "collapsed, %d conflict(s), %d window(s) unreadable", request.source_id, len(windows),
        len(routed), rounds, merged.duplicates_collapsed, len(merged.conflicts), len(failed),
    )
    return merged.extraction, merged.spans, round(wall_ms, 1), sorted(routed), flags


def _calibration_for(
    calibration: CalibrationParams | Mapping[str, CalibrationParams],
    doc_type: str | None,
) -> CalibrationParams:
    """Pick this document's calibration once its type is known."""
    if not isinstance(calibration, Mapping):
        return calibration
    if doc_type and doc_type in calibration:
        return calibration[doc_type]
    raise PipelineError(
        f"no calibration parameters for doc_type {doc_type!r}; the endpoint holds calibration for "
        f"{sorted(calibration)}. Confidence would otherwise be served raw, which is the one thing "
        "apply_calibration exists to prevent (SPEC_09)."
    )


def _feature_calibrated(
    *,
    extraction: dict[str, Any],
    spans: dict[str, Any],
    calibrators: Any,
    thresholds: Any,
    page_text: str | None,
) -> CalibratedResult:
    """Per-field-type feature calibration (arch v2.1 §5.1-5.4).

    Replaces the v1 path, which collapsed each field's span to its MINIMUM token
    probability and compared it to one hardcoded threshold. The minimum of n
    draws falls as n grows, so a long correct value scored below a short wrong
    one — ``ABC-1234567-01`` is nine tokens and ``2026`` is one, and the policy
    number looked less trustworthy on every document.

    Two absences are routing instructions here, never defaults:

    * **No calibrator for this field type** — the type had too little data to fit
      one, so every field of it goes to review rather than carrying a number
      nobody measured.
    * **No threshold for this field type** — no threshold on the grid achieved
      the §0d error target with enough accepted fields, so no promise can be
      made and review is the honest outcome.
    """
    from calibration.features import build_document_features

    result = CalibratedResult()
    usable: list[float] = []

    # The span mapper returns FieldSpan objects; the feature builder takes raw
    # logprob lists. Unmapped spans are dropped here rather than passed as empty
    # ones — an empty list is indistinguishable from a null, and a field the
    # mapper could not locate is a different problem from a field the model
    # deliberately left empty.
    logprobs_by_path = {
        path: span.token_logprobs
        for path, span in spans.items()
        if getattr(span, "mapped", False)
    }
    for features in build_document_features(
        extraction=extraction, spans=logprobs_by_path, page_text=page_text
    ):
        confidence = calibrators.predict(features) if calibrators else None
        needs_review = (
            thresholds.needs_review(features.field_type, confidence)
            if thresholds else True
        )

        if confidence is None:
            reason = (
                "no span located in the generation" if not features.is_usable
                else f"no enforced calibrator for {features.field_type} fields"
            )
            result.fields[features.field_path] = CalibratedField(
                field_path=features.field_path, value=features.value,
                raw_confidence=0.0, confidence=0.0, needs_review=True, reason=reason,
            )
            result.review_flags.append(f"{features.field_path}:no_confidence")
            continue

        result.fields[features.field_path] = CalibratedField(
            field_path=features.field_path, value=features.value,
            # The raw minimum is kept as a diagnostic, not as the confidence:
            # comparing it against the calibrated number is how a miscalibration
            # is spotted at all.
            raw_confidence=round(math.exp(features.min_logprob), 4) if features.token_count else 0.0,
            confidence=confidence, needs_review=needs_review,
            reason=(
                f"below the {features.field_type} review threshold" if needs_review else None
            ),
        )
        if needs_review:
            result.review_flags.append(f"{features.field_path}:low_confidence")
        usable.append(confidence)

    result.overall_confidence = round(sum(usable) / len(usable), 4) if usable else 0.0
    return result


def _lob_output(fields: dict[str, Any]) -> dict[str, Any] | None:
    """Collapse the per-value LoB spans into one ``{value, confidence}``.

    ``line_of_business`` is a list under arch v2.1 §0b, so the span mapper emits
    ``line_of_business[0]``, ``[1]`` … — correctly, because each emitted value has
    its own tokens and therefore its own logprob. The API returns one field, so
    they are collapsed here.

    Confidence is the **minimum** across the values, matching the aggregation used
    for a multi-token span: the weakest line is what makes the set worth
    reviewing, and averaging would let one confident line hide an invented one.

    An empty list is a real answer — the document determines no line — and is
    returned with full confidence rather than as a missing field, so it is not
    confused with a field the model failed to emit.
    """
    exact = fields.get("line_of_business")
    if exact is not None and isinstance(exact.value, list):
        return exact.as_output()

    elements = [
        (path, f) for path, f in fields.items()
        if path.startswith("line_of_business[")
    ]
    if not elements:
        # Distinguish "emitted as empty" from "never emitted". Only the former
        # is an answer.
        return {"value": [], "confidence": 1.0} if exact is not None else None

    elements.sort(key=lambda kv: kv[0])
    return {
        "value": [f.value for _, f in elements],
        "confidence": round(min(f.confidence for _, f in elements), 4),
    }


def extract(
    request: ExtractionRequest,
    model: LoadedModel,
    classifier: Classifier,
    calibration: CalibrationParams | Mapping[str, CalibrationParams],
    *,
    adapter_map: dict[str, Any] | None = None,
    #: The release bundle's fitted calibrators and thresholds for the serving
    #: format in use (arch v2.1 §5.3). Per FORMAT, always: quantization moves the
    #: logprob distribution, so a bf16 calibrator reports confidence for a
    #: distribution FP8 does not produce.
    calibrators: Any = None,
    thresholds: Any = None,
    classifier_threshold: float = 0.70,
    review_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
    page_threshold: int = DEFAULT_LONG_DOC_PAGE_THRESHOLD,
    strict_schema: bool = True,
    plan: Any = None,
    fallback_doc_type: str | None = None,
    long_doc_types: tuple[str, ...] = ("policy",),
    release_runtimes: Mapping[str, Any] | None = None,
) -> ExtractionResult:
    """Run one document through the full pipeline.

    Args:
        strict_schema: raise on a schema-invalid response rather than returning
            it. Mirrors the audit gate — a response that fails validation is an
            error, not a result.
        plan: the endpoint's :class:`~serving.release_router.ServingPlan`. Given
            one, a document type no promoted release covers is refused HERE,
            before any generation — extracting it would run a model that never
            trained on the type against a schema it has never seen.
        long_doc_types: which types get page routing. The page signals are policy
            vocabulary (declarations, schedule, endorsement), so routing a Loss
            Run through them spends a pass to select every page anyway.
    """
    route_ = _resolve_route(
        request, classifier,
        confidence_threshold=classifier_threshold,
        adapter_map=adapter_map or {},
        fallback_doc_type=fallback_doc_type,
    )

    release = None
    if plan is not None:
        from serving.release_router import UnservedDocType

        try:
            release = plan.release_for(route_.doc_type, request.known_lob)
        except UnservedDocType as exc:
            raise PipelineError(str(exc)) from exc

    # Serve THROUGH the release the plan chose: its adapter and its calibrators.
    # The choice used to be a yes/no check and was then thrown away, so a
    # personal-lines release "answered" a homeowners policy that the unified
    # model then read, with the unified model's confidence.
    runtime = (release_runtimes or {}).get(release.release_id) if release is not None else None
    if runtime is not None:
        if runtime.adapter is not None:
            route_ = replace(route_, adapter=runtime.adapter)
        if runtime.calibrators is not None:
            calibrators, thresholds = runtime.calibrators, runtime.thresholds

    # Calibration is per document type, and which type this is only becomes
    # known once the classifier has run — so it is selected here, not by the
    # caller. The endpoint used to resolve it from the caller-supplied
    # `doc_type`, which is absent on every classification-driven request; the
    # lookup fell through to `.get("")` and the request was refused before
    # extraction ever ran.
    # The v1 transform is needed only when no fitted calibrator set applies.
    calibration = _calibration_for(calibration, route_.doc_type) if calibrators is None else None

    # The line selects the policy's canonical schema. It is the caller's to
    # supply; with none, `schema_key` selects the client's canonical fallback,
    # so a policy's output is canonical JSON either way.
    lob = request.known_lob
    canonical = is_canonical(route_.schema_doc_type, route_.schema_acord_form, lob)

    # --- page routing, for long documents only -----------------------------
    routes_pages = route_.doc_type in long_doc_types
    page_plan = (
        plan_pages(request.page_texts, page_threshold=page_threshold)
        if request.page_texts and routes_pages else None
    )
    latency: float | None = None
    merge_flags: list[str] = []

    if canonical and route_.schema_doc_type == "policy":
        # Every canonical policy is read as windows — section group x page
        # window — planned by the SAME function the corpus build expanded its
        # training rows with, so a served window is a shape a training row had.
        # Always, whatever the length: a threshold computed from prompt length
        # would move between corpus builds and re-shape documents silently.
        extraction, all_spans, latency, pages_used, merge_flags = _extract_policy_windows(
            request, model, route_, lob, page_threshold=page_threshold,
        )
    elif page_plan and page_plan.routed:
        # The selected pages go in **one** call, not one call per page. Sending
        # them separately asked the model to produce a whole-document JSON from a
        # single page — a shape it never trained on — and stopped it from seeing
        # that a table on page 9 continues on page 14. Interleaving makes a
        # routed request a subsequence of the full document rather than a
        # different structure.
        pages_used = page_plan.pages
        page_images = [_image_for(request, page) for page in pages_used]
        page_ocr = (
            None if request.modality_mode == "image_only"
            else [request.page_texts.get(page) or _EMPTY_PAGE for page in pages_used]
        )
        extraction, all_spans, latency = _generate_once(
            model, route_, page_images, page_ocr, request.modality_mode,
            page_numbers=pages_used, total_pages=len(request.page_texts) or len(pages_used),
            lob=lob,
        )
    else:
        # Derived from the images actually sent, not from page_texts: an
        # image_only document has no page_texts, and defaulting to [1] made
        # check_document sum table_row_counts for page 1 alone — flagging every
        # multi-page image_only extraction as row-incomplete.
        pages_used = (
            sorted(request.page_texts) if request.page_texts
            else list(range(1, len(request.image_paths) + 1))
        )
        page_ocr = None if request.modality_mode == "image_only" else _all_page_texts(request)
        # Images by page, in the order the texts are in. Sent as the caller
        # listed them, `page_10.png` listed before `page_2.png` put page 10's
        # image beside page 2's text.
        extraction, all_spans, latency = _generate_once(
            model, route_, [_image_for(request, page) for page in pages_used],
            page_ocr, request.modality_mode,
            page_numbers=pages_used if request.page_texts else None,
            lob=lob,
        )

    # Calibration, completeness and the per-field output all read VALUES. For a
    # canonical document that is each envelope's `parsed`; a flat document is
    # its own values view. The spans were re-keyed to the same paths.
    values = values_view(extraction)

    # --- confidence --------------------------------------------------------
    # One span map for one generation, on both paths: a routed request is now a
    # single interleaved call rather than one call per page, so the spans
    # describe exactly the document that was returned. The previous per-page
    # loop left `all_spans` holding only the last page's map while `extraction`
    # was the merged document, so fields merged from earlier pages carried no
    # confidence and a field the declarations page won still reported the
    # rejected page's value.
    # The release bundle's calibrator set when the endpoint holds one, the v1
    # transform otherwise. Not a silent fallback: a release whose calibrate stage
    # never ran has no per-field-type curves and no measured thresholds, and
    # serving it through the v1 path is a knowing downgrade rather than an
    # equivalent.
    if calibrators is not None:
        calibrated: CalibratedResult = _feature_calibrated(
            # The model form, envelopes and all: flattened it gives the same
            # values, and it still carries each value's printed form.
            extraction=extraction, spans=all_spans,
            calibrators=calibrators, thresholds=thresholds,
            # The same definition validation fits the calibrators with: the OCR
            # text of the pages the model was shown, none for image_only.
            page_text=None if request.modality_mode == "image_only" else shown_ocr_text(
                [request.page_texts.get(page) for page in pages_used]
                if request.page_texts else [request.ocr_text]
            ),
        )
    else:
        log.warning(
            "no calibrator set supplied for %s, so confidence falls back to the v1 raw "
            "aggregate against a fixed %.2f threshold. That number is length-biased and tied "
            "to no measured error rate (arch v2.1 §5.1) — load the release bundle's "
            "calibrators to get the guarantee.", request.source_id, review_threshold,
        )
        raw = field_confidences(all_spans)
        calibrated = apply_calibration(raw, calibration, review_threshold=review_threshold)

    # --- list completeness: the signal logprobs cannot see ------------------
    completeness = check_document(
        values, route_.schema_doc_type,
        ocr_meta=request.ocr_meta, pages_used=pages_used,
    )
    flags = merge_review_flags(
        completeness, calibrated.review_flags + route_.review_flags + merge_flags
    )

    # --- the client's canonical envelope -----------------------------------
    # confidence and flagged are filled here, from the calibrated fields, never
    # by the model. A list whose row count disagrees with the document is
    # flagged on every value in it: the rows present may each be confident, and
    # the list as a whole still needs a person.
    if canonical:
        incomplete = tuple(
            flag.split(":", 1)[0] for flag in flags if is_row_completeness_flag(flag)
        )
        scores = {
            path: (
                f.confidence,
                f.needs_review or any(
                    path == name or path.startswith((f"{name}[", f"{name}."))
                    for name in incomplete
                ),
            )
            for path, f in calibrated.fields.items()
        }
        output = envelope(extraction, scores)
    else:
        output = extraction

    # --- schema validation, mirroring the audit gate ------------------------
    # Against the client's FULL schema for a canonical document, envelope and
    # all — the model form it was generated in is ours, the contract is theirs.
    schema_valid = is_valid(output, route_.schema_doc_type, route_.schema_acord_form, lob)
    validation_errors: list[str] = []
    if not schema_valid:
        validation_errors = list(
            iter_validation_errors(output, route_.schema_doc_type, route_.schema_acord_form, lob)
        )
        if strict_schema:
            raise PipelineError(
                f"the extraction for {request.source_id} failed schema validation and will not be "
                f"returned (Fideon SPEC_07 Stage 3 mirrors this on every production call): "
                + "; ".join(validation_errors[:4])
            )
        flags.append("schema:invalid")

    result = ExtractionResult(
        source_id=request.source_id,
        doc_type=route_.schema_doc_type,
        model_version=model.tag,
        mode=request.modality_mode,
        schema_valid=schema_valid,
        overall_confidence=calibrated.overall_confidence,
        extraction=output,
        fields={
            path: f.as_output()
            for path, f in sorted(calibrated.fields.items())
            if path != "line_of_business" and "[" not in path
        },
        list_fields={
            name: {"rows": values.get(name, []), **signal.as_output()}
            for name, signal in sorted(completeness.items())
        },
        line_of_business=_lob_output(calibrated.fields),
        pages_used=pages_used,
        review_flags=sorted(set(flags)),
        route_info={
            "adapter": route_.adapter,
            "foundation_only": route_.foundation_only,
            "classifier_confidence": (
                route_.classification.confidence if route_.classification else None
            ),
            "classifier_method": (
                route_.classification.method if route_.classification else None
            ),
        },
        latency_ms=latency,
        validation_errors=validation_errors,
    )

    log.info(
        "extracted %s as %s (confidence %.3f, %d review flag(s), adapter=%s)",
        request.source_id, result.doc_type, result.overall_confidence,
        len(result.review_flags), route_.adapter or "foundation-only",
    )
    return result
