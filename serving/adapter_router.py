"""Routing a classified document to its adapter, prompt, and schema (arch §4a).

Three things are selected together and must agree: which per-type LoRA to apply,
which prompt to render, and which schema to validate against. They are chosen
here, once, from one classification result — selecting them independently is how
a document gets extracted with one type's adapter against another type's schema.

**The low-confidence fallback is the point of this module.** When the classifier
is not confident, the router does **not** guess. It falls back to Foundation-only
extraction with the document's best-guess schema and flags the document for human
routing review, because a wrong-adapter extraction is worse than a slightly
generic one: the generic one is merely less sharp, while the wrong one is
confidently answering the wrong question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from common.constants import DEFAULT_RESOLUTION_CAP_PX  # noqa: F401  (kept for config parity)
from serving.doc_type_classifier import Classification

log = logging.getLogger(__name__)


class RoutingError(RuntimeError):
    """Raised when a document cannot be routed at all."""


@dataclass
class Route:
    """Everything downstream needs, decided in one place."""

    doc_type: str
    acord_form: str | None

    #: ``None`` means the MERGED model with no adapter — which under arch v2.1
    #: §4.1 is the normal case, not a fallback. One unified adapter is merged
    #: into the base, and a per-type LoRA exists only for a type that cleared the
    #: §4.2 graduation gate. vLLM applies one LoRA per request, so there is never
    #: a stack: it is the merged model, or the merged model plus one adapter.
    adapter: str | None
    schema_doc_type: str
    schema_acord_form: str | None

    #: Kept under its v1 name so callers and manifests do not move in the same
    #: change as the topology. It now means "no graduated adapter for this type",
    #: which is the expected state for every type until one graduates.
    foundation_only: bool = False

    review_flags: list[str] = field(default_factory=list)
    classification: Classification | None = None

    #: The §4a low-confidence path: extract for BOTH candidate types, validate
    #: each against its own SPEC_00 schema, and send both plus the classification
    #: scores to human routing review. Empty on the normal path.
    #:
    #: v1 fell back to one guessed schema with a flag. That is worse than it
    #: sounds: a wrong-schema extraction is structurally valid and confidently
    #: wrong, so it passes the audit gate and reaches a user looking correct.
    candidate_routes: list[Route] = field(default_factory=list)

    @property
    def needs_routing_review(self) -> bool:
        return any(flag.startswith("routing:") for flag in self.review_flags)

    @property
    def is_ambiguous(self) -> bool:
        return bool(self.candidate_routes)


def route(
    classification: Classification,
    *,
    confidence_threshold: float = 0.70,
    adapter_map: dict[str, str | None] | None = None,
    fallback_doc_type: str = "policy",
) -> Route:
    """Select adapter, prompt, and schema from a classification.

    Args:
        confidence_threshold: below this, fall back to Foundation-only.
        adapter_map: doc_type -> adapter path or name. A missing entry also
            routes Foundation-only, which is correct during the pilot when no
            per-type adapter has been trained yet.
        fallback_doc_type: schema used when the type could not be determined.
            Something has to be validated against; the flag records that it was
            a fallback rather than a decision.
    """
    adapter_map = adapter_map or {}

    # Unusable classification — no type at all.
    if not classification.is_usable:
        log.warning(
            "document could not be classified (%r, confidence %.2f) — Foundation-only "
            "extraction and human routing review",
            classification.doc_type, classification.confidence,
        )
        return Route(
            doc_type=fallback_doc_type,
            acord_form=None,
            adapter=None,
            schema_doc_type=fallback_doc_type,
            schema_acord_form=None,
            foundation_only=True,
            review_flags=["routing:unclassified"],
            classification=classification,
        )

    # `is_usable` guarantees a doc_type; binding it once here is what lets the
    # rest of the function rely on that instead of re-asserting it three times.
    doc_type: str = classification.doc_type or fallback_doc_type

    # ACORD is two-level: the form selects the schema, and `schema_key` refuses
    # the pair (acord, None). Routing it anyway meant render_system_prompt raised
    # SchemaError and the whole extraction crashed — the documented guard ("no
    # schema can be selected") zeroed the confidence but left the doc_type in
    # place, so both branches below still produced the refused pair.
    if doc_type == "acord" and not classification.acord_form:
        log.warning(
            "ACORD form number missing, so no schema can be selected — falling back to the %r "
            "schema with Foundation-only extraction and a routing review flag rather than "
            "failing the request (arch §4b).", fallback_doc_type,
        )
        return Route(
            doc_type=fallback_doc_type,
            acord_form=None,
            adapter=None,
            schema_doc_type=fallback_doc_type,
            schema_acord_form=None,
            foundation_only=True,
            review_flags=["routing:acord_form_unknown"],
            classification=classification,
        )

    # Classified, but not confidently enough to commit to one schema.
    if classification.confidence < confidence_threshold:
        candidates = classification.top_two()
        log.warning(
            "classifier confidence %.2f below threshold %.2f for %r — extracting for %s and "
            "sending both to human routing review (arch v2.1 §4a)",
            classification.confidence, confidence_threshold, classification.doc_type, candidates,
        )
        primary = Route(
            doc_type=doc_type,
            acord_form=classification.acord_form,
            adapter=adapter_map.get(doc_type),
            schema_doc_type=doc_type,
            schema_acord_form=classification.acord_form,
            foundation_only=adapter_map.get(doc_type) is None,
            review_flags=["routing:low_confidence"],
            classification=classification,
        )
        # Each candidate is extracted against ITS OWN schema. v1 picked one and
        # flagged it, which produces a structurally valid, confidently wrong
        # extraction that passes the audit gate and reaches a user looking
        # correct. Two answers plus the scores is what a router can review.
        primary.candidate_routes = [
            Route(
                doc_type=candidate,
                acord_form=classification.acord_form if candidate == "acord" else None,
                adapter=adapter_map.get(candidate),
                schema_doc_type=candidate,
                schema_acord_form=classification.acord_form if candidate == "acord" else None,
                foundation_only=adapter_map.get(candidate) is None,
                review_flags=["routing:candidate"],
                classification=classification,
            )
            for candidate in candidates
            # An ACORD candidate with no form number has no schema to validate
            # against, so it cannot be one of the two answers offered.
            if not (candidate == "acord" and not classification.acord_form)
        ]
        if not primary.candidate_routes:
            primary.review_flags.append("routing:no_extractable_candidate")
            log.warning(
                "neither candidate could be extracted (%s) — routing to human review without "
                "an extraction rather than inventing a schema (arch v2.1 §4a).", candidates,
            )
        return primary

    adapter = adapter_map.get(doc_type)

    flags: list[str] = []
    if adapter is None:
        # The NORMAL case under arch v2.1 §4.1, not a degraded one. The merged
        # unified model serves every type; a per-type adapter exists only after
        # §4.2 graduation. Logged at debug, and deliberately NOT flagged for
        # review — flagging the expected path would put every document in the
        # queue and teach everyone to ignore the flag.
        log.debug("no graduated adapter for %r — serving the merged unified model", doc_type)

    return Route(
        doc_type=doc_type,
        acord_form=classification.acord_form,
        adapter=adapter,
        schema_doc_type=doc_type,
        schema_acord_form=classification.acord_form,
        foundation_only=adapter is None,
        review_flags=flags,
        classification=classification,
    )


def adapter_map_from_config(serving_config: dict[str, Any]) -> dict[str, str | None]:
    """Read the adapter map from serving config.

    Paths resolve through the registry rather than being hardcoded, so a promoted
    version change does not require a config edit.
    """
    return dict(serving_config.get("routing", {}).get("adapter_map", {}) or {})
