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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from calibration.apply_calibration import CalibratedField, CalibratedResult, apply_calibration
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
from inference_core.input_builder import build_messages
from inference_core.model_runner import LoadedModel, generate
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

    Falls back to positional order when a caller supplies paths that do not
    follow it — silently pairing every page with page 1's image, as the previous
    code did, is worse than an approximate match.
    """
    for path in request.image_paths:
        if f"page_{page}." in path:
            return path
    index = sorted(request.page_texts).index(page) if page in request.page_texts else 0
    return request.image_paths[min(index, len(request.image_paths) - 1)]


#: Stands in for a page whose OCR produced nothing. An empty string would make
#: `build_messages` refuse the whole request — and ALWAYS_INCLUDE_FIRST_PAGE
#: makes a poorly-scanned first page the single most likely page to be selected,
#: so one blank page aborted exactly the documents page routing exists for. The
#: marker says the page is there and unreadable, which is true and is what the
#: image is for.
_EMPTY_PAGE = "(no OCR text recovered for this page — read it from the image)"


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
) -> tuple[dict[str, Any], dict[str, Any], float | None]:
    """One scoped generation. Returns ``(extraction, spans, latency_ms)``.

    The extraction comes back with every date in ``MM/DD/YYYY`` and, for a
    canonical document, still in the model's sparse ``raw``/``parsed``/
    ``page_ref`` form; the spans are keyed by values-view path. The caller adds
    confidence and wraps it in the client's envelope.
    """
    built = build_messages(
        route_.schema_doc_type,
        image_paths,
        ocr_pages,
        modality_mode,
        acord_form=route_.schema_acord_form,
        lob=lob,
        page_numbers=page_numbers,
        total_pages=total_pages,
    )
    # Constrained to the routed type's schema when serving is configured for it
    # (arch v2.1 §13). Without this the schema-validity floor in §13b measured an
    # unconstrained model, so one malformed bf16 output made every quantized
    # format unvalidatable.
    schema = (
        resolved_schema(route_.schema_doc_type, route_.schema_acord_form, lob)
        if getattr(model.config, "structured_outputs", False) else None
    )
    result = generate(model, built.messages, adapter=route_.adapter, json_schema=schema)

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
    return with_output_dates(extraction), collapse_spans(spans), result.latency_ms


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

    if plan is not None:
        from serving.release_router import UnservedDocType

        try:
            plan.release_for(route_.doc_type)
        except UnservedDocType as exc:
            raise PipelineError(str(exc)) from exc

    # Calibration is per document type, and which type this is only becomes
    # known once the classifier has run — so it is selected here, not by the
    # caller. The endpoint used to resolve it from the caller-supplied
    # `doc_type`, which is absent on every classification-driven request; the
    # lookup fell through to `.get("")` and the request was refused before
    # extraction ever ran.
    calibration = _calibration_for(calibration, route_.doc_type)

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

    if page_plan and page_plan.routed:
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
        extraction, all_spans, latency = _generate_once(
            model, route_, request.image_paths, page_ocr, request.modality_mode,
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
            extraction=values, spans=all_spans,
            calibrators=calibrators, thresholds=thresholds,
            page_text=request.ocr_text,
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
    flags = merge_review_flags(completeness, calibrated.review_flags + route_.review_flags)

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
