"""Document-type classification (arch §4a, §4b, §4c).

**Load-bearing, not preprocessing.** A wrong answer loads the wrong adapter *and*
the wrong prompt *and* the wrong schema, and the extraction fails regardless of
how good the model is. A classifier at 92% caps the whole system at 92%, which is
why it carries its own accuracy target and its own gating metric.

Two-level by necessity (arch §4b): type, then ACORD form number. One shared ACORD
adapter covers forms 25/125/140, but each form has a distinct schema, so the form
still has to be identified.

Default implementation is **zero-shot via the model already loaded** (arch §4a
Option C) — no extra model to train, serve, version, or keep in sync. The
interface is swappable for a dedicated vision classifier, which is the right
escalation if classification becomes a *measured* bottleneck, because it works in
both OCR and image-only modes.

On day zero this runs against the **base** model (arch §4c). That path is
explicitly temporary and retires when Foundation v1.0 is promoted.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from common.constants import ACORD_FORMS, ACTIVE_DOC_TYPES
from common.prompts import render_classifier_prompt

log = logging.getLogger(__name__)


class ClassificationError(RuntimeError):
    """Raised when the classifier returns something unusable."""


@dataclass
class Classification:
    """A classification result, with the confidence that drives routing."""

    doc_type: str | None
    acord_form: str | None
    confidence: float
    raw_response: str = ""
    method: str = "zero_shot"

    @property
    def is_usable(self) -> bool:
        return self.doc_type in ACTIVE_DOC_TYPES

    def meets(self, threshold: float) -> bool:
        return self.is_usable and self.confidence >= threshold


class Classifier(Protocol):
    """The swappable classifier interface."""

    def classify(self, image_paths: list[str], ocr_text: str | None) -> Classification: ...


def parse_classification(response: str) -> Classification:
    """Parse the model's JSON answer, tolerating the usual wrapping.

    Models occasionally wrap JSON in fences despite instruction. Recovering from
    that is worth it here: a parse failure routes an otherwise-classifiable
    document to human review for no reason.
    """
    text = response.strip()
    if fenced := re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        text = fenced.group(1)
    elif braced := re.search(r"\{.*\}", text, re.S):
        text = braced.group(0)

    try:
        payload = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ClassificationError(
            f"the classifier response was not JSON: {response[:120]!r}"
        ) from exc

    doc_type = (payload.get("doc_type") or "").strip().lower() or None
    acord_form = payload.get("acord_form")
    acord_form = str(acord_form).strip() if acord_form not in (None, "", "null") else None

    if doc_type and doc_type not in ACTIVE_DOC_TYPES:
        log.warning("classifier returned unknown doc_type %r", doc_type)
        doc_type = None
    if acord_form and acord_form not in ACORD_FORMS:
        log.warning("classifier returned unknown ACORD form %r", acord_form)
        acord_form = None

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    # A classification with no form for ACORD cannot select a schema, so it is
    # not usable however confident the model claims to be.
    if doc_type == "acord" and acord_form is None:
        log.warning("ACORD classified without a form number — no schema can be selected")
        confidence = min(confidence, 0.0)

    return Classification(
        doc_type=doc_type,
        acord_form=acord_form,
        confidence=max(0.0, min(1.0, confidence)),
        raw_response=response,
    )


class ZeroShotClassifier:
    """Classify by prompting the already-loaded model (arch §4a Option C)."""

    def __init__(self, model: Any, generate_fn: Any) -> None:
        self.model = model
        self.generate = generate_fn

    def classify(self, image_paths: list[str], ocr_text: str | None) -> Classification:
        content: list[dict[str, Any]] = [{"type": "image", "image": p} for p in image_paths[:2]]
        if ocr_text:
            # First page only: the type is determined by the header and layout,
            # and sending a 50-page policy to answer a one-word question is waste.
            content.append({"type": "text", "text": ocr_text[:4000]})

        messages = [
            {"role": "system", "content": render_classifier_prompt()},
            {"role": "user", "content": content},
        ]
        result = self.generate(self.model, messages, want_logprobs=False)
        classification = parse_classification(result.text)
        classification.method = "zero_shot_base" if getattr(self.model, "is_base", False) else "zero_shot"
        return classification


class StaticClassifier:
    """A fixed answer — for tests, and for callers that already know the type."""

    def __init__(self, doc_type: str, acord_form: str | None = None, confidence: float = 1.0) -> None:
        self.result = Classification(doc_type, acord_form, confidence, method="static")

    def classify(self, image_paths: list[str], ocr_text: str | None) -> Classification:
        return self.result
