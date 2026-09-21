"""The RunPod Serverless handler (SPEC_11 §1, arch §14).

A persistent endpoint, unlike the ephemeral training pods: it is updated to a
promoted version and rolled back, not launched and destroyed.

Everything here is a thin shell around :func:`serving.pipeline.extract` and the
SPEC_07 inference core, so a served response is byte-for-byte what eval and the
testing harness produce for the same input. Generation logic in this file would
be a second implementation, and the numbers in the registry would then describe
the wrong one.

Three guards live here rather than deeper down, because this is the boundary
where a request enters the system:

**The MinerU pin is asserted at startup, not per request.** The model learned how
MinerU formats its output; serving OCR from a different version — or a different
*device*, if the Phase 0 spike shows the output differs — is the distribution
shift arch §8a exists to catch. A mismatch fails the cold start rather than
degrading quietly across every request afterwards.

**Missing calibration raises.** Returning raw logprob confidence when no
calibration exists for the served version would hand downstream review routing
numbers that look like probabilities and are not.

**Nothing plaintext reaches the logs, and no raw body is persisted.** Insurance
documents carry names, TINs, addresses and financials; a handler that logs its
request for debugging has exported all of it (master §8).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry.blob_client import BlobClient
from common.config import serving_config
from common.constants import ACTIVE_DOC_TYPES
from serving.pipeline import ExtractionRequest, ExtractionResult, extract
from serving.release_router import build_serving_plan

log = logging.getLogger(__name__)


class ServingError(RuntimeError):
    """Raised when a request cannot be served."""


class ColdStartError(ServingError):
    """Raised when the endpoint cannot come up safely.

    Distinct from a per-request failure: a cold start that fails must not fall
    back to serving anyway, because everything it checks is a whole-endpoint
    property — the wrong OCR version is wrong for every request, not this one.
    """


#: Keys never echoed back or logged. The document itself and its OCR text are
#: the payload, so they are exactly what must not appear in a log line.
SENSITIVE_KEYS = frozenset({
    "ocr_text", "page_texts", "images", "image_bytes", "pdf", "pdf_bytes", "extraction",
})


def safe_log_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """What may be logged about a request: shape, never content.

    Sizes and counts are enough to debug a malformed request. The values are
    what the compliance posture is about.
    """
    summary: dict[str, Any] = {}
    for key, value in sorted(payload.items()):
        if key in SENSITIVE_KEYS:
            if isinstance(value, (list, dict)):
                summary[key] = f"<{len(value)} item(s), not logged>"
            elif isinstance(value, (bytes, str)):
                summary[key] = f"<{len(value)} bytes, not logged>"
            else:
                summary[key] = "<not logged>"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        else:
            summary[key] = f"<{type(value).__name__}>"
    return summary


@dataclass
class EndpointState:
    """What a warm endpoint holds. Built once, on cold start."""

    model_version: str
    model: Any = None
    classifier: Any = None
    calibration: Any = None
    adapter_map: dict[str, Any] = field(default_factory=dict)
    corpus_manifest: dict[str, Any] = field(default_factory=dict)

    #: Which promoted release answers for each document type (arch v2.1 §12.3).
    #: More than one can be promoted at a time — a policy release alongside an
    #: older unified one — so this is what decides, and what refuses a type
    #: nothing covers.
    plan: Any = None
    ready: bool = False

    def release_for(self, doc_type: str) -> Any:
        """The release serving this type. Raises ``UnservedDocType``."""
        if self.plan is None:
            return None
        return self.plan.release_for(doc_type)


def assert_ocr_pin(corpus_manifest: dict[str, Any]) -> None:
    """Fail the cold start when serving OCR differs from the corpus pin.

    Checked at startup rather than per request: it is a property of the
    deployment, and a per-request check would turn one configuration mistake
    into a per-request cost while still not fixing it.
    """
    from data_pipeline.ocr.mineru_version import assert_version_matches

    if not corpus_manifest.get("mineru_version"):
        raise ColdStartError(
            "the corpus manifest records no mineru_version, so the serving OCR version cannot be "
            "checked against what the model trained on. That pin is the arch §8a guarantee; "
            "serving without it means an OCR upgrade becomes an unattributable accuracy drop."
        )
    # The manifest is passed whole: assert_version_matches reads both
    # mineru_version and ocr_device from it, and splitting them here would mean
    # two callers with two ideas of what the pin covers.
    assert_version_matches(corpus_manifest)


def calibration_for(state: EndpointState, doc_type: str | None) -> Any:
    """The calibration to serve this document with, or ``None``."""
    if not isinstance(state.calibration, dict):
        return state.calibration
    return state.calibration.get(doc_type or "")


def assert_calibration_present(calibration: Any, model_version: str, doc_type: str) -> None:
    """Refuse to serve without calibration parameters.

    Raw logprob confidence is systematically overconfident, and the review
    routing built on it would send the wrong documents to humans. A mapping is
    accepted whole — which entry applies is decided after classification, inside
    `extract` — but an empty one is no calibration at all.
    """
    if isinstance(calibration, Mapping):
        if not calibration:
            raise ServingError(
                f"no calibration parameters loaded for {model_version}. Fit them against the "
                "frozen golden eval set and push them before serving (SPEC_09)."
            )
        return
    if calibration is None:
        raise ServingError(
            f"no calibration parameters for {model_version}/{doc_type}. Raw logprob confidence is "
            "systematically overconfident, so serving it would route the wrong documents to "
            "review (SPEC_09)."
        )


def _adapter_exists(adapter_path: str, client: BlobClient) -> bool:
    """Whether an adapter directory actually holds weights.

    A staging path lives on the pod's volume and a published one in Blob, so
    both are checked: the marker file PEFT always writes is what distinguishes a
    real adapter from a prefix nobody ever wrote to.
    """
    marker = f"{adapter_path.rstrip('/')}/adapter_config.json"
    if adapter_path.startswith("/"):
        return Path(marker).exists()
    try:
        return bool(client.exists(marker))
    except Exception:  # noqa: BLE001 - an unreachable store is not a present adapter
        log.warning("could not confirm %s exists; treating it as absent", adapter_path)
        return False


def release_pins() -> dict[str, str]:
    """``{doc_type: release_id}`` overrides from the serving config."""
    from common.config import serving_config

    routing = serving_config().get("routing") or {}
    return {str(k): str(v) for k, v in (routing.get("release_pins") or {}).items() if v}


def build_adapter_map(model_version: str, client: BlobClient) -> dict[str, str]:
    """doc_type -> adapter path, resolved from the registry.

    ``resolved["adapters"]`` was read here and ``ResolvedModel`` defines no such
    key, so the map was always empty: every document served Foundation-only and
    carried a `routing:no_adapter_available` flag. Resolving each type on its own
    is what the registry is for, and a type with no promoted adapter is simply
    absent from the map — which the router already treats as Foundation-only.
    """
    from registry_utils.query_registry import RegistryQueryError, resolve_model_version

    adapter_map: dict[str, str] = {}
    for doc_type in ACTIVE_DOC_TYPES:
        try:
            resolved = resolve_model_version(model_version, client, doc_type=doc_type)
        except (RegistryQueryError, KeyError, FileNotFoundError):
            continue
        adapter = resolved.get("type_adapter")
        # `type_adapter` is constructed unconditionally by resolve_model_version
        # whenever a doc_type is passed — it is a path, not evidence that
        # anything was trained. Trusting it mapped all three types on a
        # --foundation-only build, so vLLM was handed a LoRARequest for a
        # directory that does not exist and EVERY request failed, instead of
        # serving Foundation-only with a routing flag as documented above.
        if adapter and _adapter_exists(adapter, client):
            adapter_map[doc_type] = adapter

    if not adapter_map:
        log.warning(
            "no per-type adapters resolved for %s — every document will serve Foundation-only "
            "and carry a routing review flag. That is correct during the pilot, when the "
            "Foundation may be the only model trained, and a defect at any other time.",
            model_version,
        )
    return adapter_map


def load_calibrations(model_version: str, client: BlobClient) -> dict[str, Any]:
    """doc_type -> calibration parameters for the served version.

    Left unset, the endpoint came up reporting "warm" and then rejected **every**
    request at :func:`assert_calibration_present` — an endpoint that cannot serve
    a single document while its own health check says it is ready.
    """
    from calibration.apply_calibration import load_calibration

    loaded: dict[str, Any] = {}
    for doc_type in ACTIVE_DOC_TYPES:
        try:
            loaded[doc_type] = load_calibration(model_version, doc_type, client)
        except Exception as exc:  # noqa: BLE001 - a missing fit is reported per request
            log.warning("no calibration for %s/%s: %s", model_version, doc_type, exc)
    return loaded


def cold_start(
    model_version: str,
    client: BlobClient,
    *,
    corpus_version: str | None = None,
    tenant_id: str | None = None,
    model: Any = None,
    classifier: Any = None,
) -> EndpointState:
    """Pull the promoted artifact and validate the deployment before serving.

    Order matters: the pin check runs **before** the model is loaded, so a
    misconfigured deployment fails in seconds rather than after pulling 16 GB.
    """
    from artifact_registry import paths
    from registry_utils.query_registry import resolve_model_version

    # Resolved for its side effect: an unknown version must fail the cold start
    # here rather than on the first request. The per-type adapters are resolved
    # separately by build_adapter_map below.
    resolve_model_version(model_version, client)

    manifest: dict[str, Any] = {}
    if corpus_version:
        manifest_key = paths.corpus_manifest(corpus_version, tenant_id)
        if not client.exists(manifest_key):
            raise ColdStartError(
                f"corpus manifest for {corpus_version} is missing, so the OCR pin cannot be "
                "verified. Serving a model whose training-time OCR version is unknown is the "
                "distribution-shift risk in arch §8a with no way to detect it."
            )
        manifest = client.read_json(manifest_key)
        assert_ocr_pin(manifest)

    config = serving_config()
    if model is None:
        raise ColdStartError(
            f"no model was supplied and the vLLM backend is not wired yet. Resolve it with "
            f"inference_core.model_runner.load_model({model_version!r}, client) once the Phase 0 "
            "spike confirms vLLM multi-LoRA hot-swap for Qwen3-VL; if it does not hold, SPEC_11 "
            "falls back to serving merged per-type models and only this function changes."
        )

    state = EndpointState(
        model_version=model_version,
        model=model,
        classifier=classifier,
        adapter_map=build_adapter_map(model_version, client),
        calibration=load_calibrations(model_version, client),
        plan=build_serving_plan(
            client, tenant_id=tenant_id, pins=release_pins(),
        ),
        corpus_manifest=manifest,
        ready=True,
    )
    log.info(
        "endpoint warm: version=%s adapters=%s logprobs=%s",
        model_version, sorted(state.adapter_map), config.get("logprobs", True),
    )
    return state


def build_request(payload: dict[str, Any]) -> ExtractionRequest:
    """Validate a request body into an :class:`ExtractionRequest`."""
    source_id = payload.get("source_id")
    images = payload.get("image_paths") or []
    if not source_id:
        raise ServingError("request is missing source_id, so its result could not be attributed")
    if not images:
        raise ServingError(
            "request supplies no page images. Both production modes need the image: image_only "
            "has nothing else, and ocr_plus_image arbitrates between the two (arch §6)."
        )

    mode = payload.get("modality_mode", "ocr_plus_image")
    ocr_text = payload.get("ocr_text")
    if mode == "image_only" and ocr_text:
        raise ServingError(
            "image_only was requested with OCR text attached. Serving OCR under the image-only "
            "prompt would mean the model is told no text exists while text is present, which is "
            "not a state it was trained on."
        )

    return ExtractionRequest(
        source_id=str(source_id),
        image_paths=[str(p) for p in images],
        ocr_text=ocr_text,
        page_texts={int(k): v for k, v in (payload.get("page_texts") or {}).items()},
        ocr_meta=payload.get("ocr_meta") or {},
        modality_mode=mode,
        known_doc_type=payload.get("doc_type"),
        known_acord_form=payload.get("acord_form"),
    )


def serving_thresholds() -> dict[str, Any]:
    """The tuneable thresholds, from configs/inference/vllm_serving.yaml.

    Each key was present in that file and read by no code, so an operator could
    edit it, redeploy, and see no change. Only keys `extract` accepts are
    returned, so an unknown one in the YAML is ignored rather than crashing a
    cold start.
    """
    config = serving_config()
    routing = config.get("routing") or {}
    confidence = config.get("confidence") or {}
    long_documents = config.get("long_documents") or {}

    tuning: dict[str, Any] = {}
    for key, value in (
        ("classifier_threshold", routing.get("classifier_confidence_threshold")),
        ("review_threshold", confidence.get("review_threshold")),
        ("page_threshold", long_documents.get("page_threshold")),
    ):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            tuning[key] = value
    return tuning


def handler(event: dict[str, Any], state: EndpointState) -> dict[str, Any]:
    """The RunPod Serverless entrypoint. One request in, the output contract out.

    Errors are returned as a structured payload rather than raised, because a
    serverless handler that raises returns an opaque platform error and the
    caller learns nothing about which of its inputs was wrong.
    """
    payload = event.get("input") or {}

    try:
        # Inside the try. `safe_log_payload` iterates the payload, so an `input`
        # that is not a JSON object — a bare string, a list, a number — raised
        # AttributeError straight out of the handler, and RunPod returned an
        # opaque platform error saying nothing about which input was wrong. That
        # is the exact failure this handler exists to prevent.
        log.info("request %s", safe_log_payload(payload))

        if not state.ready:
            return {"error": "endpoint is not warm", "retryable": True}

        request = build_request(payload)
        # The whole calibration map is handed to `extract`, which resolves the
        # right one once the classifier has said what the document is.
        # Resolving here from `request.known_doc_type` meant `.get("")` for
        # every request that did not name its own doc_type — which is every
        # classification-driven request, the endpoint's entire purpose — and
        # `assert_calibration_present` then refused it before extraction ran.
        assert_calibration_present(
            state.calibration, state.model_version, request.known_doc_type or "any"
        )
        # The thresholds come from configs/inference/vllm_serving.yaml. They
        # were declared there and read by nothing, so raising review_threshold
        # and redeploying changed no behaviour at all — the hardcoded defaults
        # inside `extract` won every time, silently.
        tuning = serving_thresholds()
        result: ExtractionResult = extract(
            request,
            state.model,
            state.classifier,
            state.calibration,
            adapter_map=state.adapter_map,
            **tuning,
        )
    except ServingError as exc:
        # The message names what was wrong with the request; it never echoes the
        # request, which is what keeps document content out of the logs.
        log.warning("request rejected: %s", exc)
        return {"error": str(exc), "retryable": False}
    except Exception as exc:  # noqa: BLE001 - a handler must never raise
        # Non-JSON output and schema-invalid output are the two most likely real
        # failures and both arrive as PipelineError, which is not a ServingError.
        # Letting them escape returned an opaque platform error carrying nothing
        # about which input was wrong — the exact thing this handler exists to
        # prevent. The exception type is named; the request never is.
        log.error("extraction failed (%s): %s", type(exc).__name__, exc)
        return {"error": f"{type(exc).__name__}: {exc}", "retryable": False}

    # `as_dict` is the master §9 output contract. Nothing raw is persisted here:
    # the caller owns the document, and the endpoint keeps no copy.
    return {"output": result.as_dict()}
