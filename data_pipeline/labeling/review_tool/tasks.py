"""Task export and annotation import (SPEC_04 §4).

The round trip: a document becomes a review task, a reviewer corrects it, and the
completed annotation becomes a golden label plus its ``field_provenance``.

Two properties are the point of doing this in code rather than by hand:

* **The provenance survives the round trip.** The reviewer records the surface
  label they actually saw, and it has to arrive intact at
  ``export_golden_labels`` — that is what makes per-alias evaluation and alias
  derivation possible at all. A round trip that drops it leaves only an
  aggregate, and an aggregate cannot say which phrasing the model fails on.
* **Nothing is written that has not been validated.** Import produces a label and
  hands it to ``export_golden_labels``, which is the single place that decides
  whether a label may enter the corpus. Writing golden JSON here would be a
  second gate, and two gates disagree.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: Suffix the review form uses for "which label did you see on the page?".
PROVENANCE_SUFFIX = "__seen_as"

#: Default share of documents sent to a second, independent reviewer. Measures
#: how much ambiguity the labeling process itself carries — useful context when
#: eval plateaus below 100%, since part of that ceiling is human disagreement
#: rather than model error.
DEFAULT_DOUBLE_ANNOTATION_RATE = 0.15


class ReviewToolError(RuntimeError):
    """Raised when a task or an annotation cannot be adapted."""


@dataclass
class ReviewTask:
    """One document, ready for a reviewer."""

    source_id: str
    doc_type: str
    page_images: list[str]
    ocr_text: str
    draft: dict[str, Any] = field(default_factory=dict)
    acord_form: str | None = None
    double_annotate: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Label Studio task shape: everything under ``data``."""
        return {
            "data": {
                "source_id": self.source_id,
                "doc_type": self.doc_type,
                "acord_form": self.acord_form,
                "page_images": self.page_images,
                "ocr_text": self.ocr_text,
                # Pre-filled from the draft so a reviewer corrects rather than
                # types. A corrected draft is faster and more consistent than a
                # blank form (arch §7) — but it is never accepted unreviewed.
                "draft": self.draft,
                "double_annotate": self.double_annotate,
            }
        }


def task_from_document(
    source_id: str,
    doc_type: str,
    page_images: list[str],
    ocr_text: str,
    *,
    draft: dict[str, Any] | None = None,
    acord_form: str | None = None,
    double_annotate: bool = False,
) -> ReviewTask:
    """Build one review task."""
    if not page_images:
        raise ReviewToolError(
            f"{source_id} has no page images. A reviewer cannot verify a value against a document "
            "they cannot see, and the OCR text alone is the thing being checked."
        )
    if doc_type == "acord" and not acord_form:
        raise ReviewToolError(
            f"{source_id} is an ACORD with no form number, so no schema — and therefore no field "
            "list — can be selected for the review form (arch §4b)."
        )
    return ReviewTask(
        source_id=source_id,
        doc_type=doc_type,
        page_images=list(page_images),
        ocr_text=ocr_text,
        draft=dict(draft or {}),
        acord_form=acord_form,
        double_annotate=double_annotate,
    )


def double_annotation_sample(
    source_ids: list[str], *, rate: float = DEFAULT_DOUBLE_ANNOTATION_RATE, seed: int = 42
) -> list[str]:
    """Which documents get a second independent reviewer.

    Chosen by a stable hash of the ``source_id`` rather than by shuffling, so
    adding documents later never moves an existing one in or out of the sample —
    the same reason the corpus split assigns by hash threshold. An
    agreement score computed over a set that changes between batches is not
    comparable across them.
    """
    if not 0.0 <= rate <= 1.0:
        raise ReviewToolError(f"double-annotation rate must be in [0, 1], got {rate}")

    selected = []
    for source_id in sorted(set(source_ids)):
        digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
        position = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if position < rate:
            selected.append(source_id)

    if source_ids and rate > 0 and not selected:
        # At small batch sizes the sample can come up empty by chance, and an
        # agreement score over zero documents is not a measurement.
        selected = [sorted(set(source_ids))[0]]
        log.info("double-annotation sample was empty at rate %.2f; taking one document", rate)
    return selected


def completed_to_golden(
    annotation: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Adapt a completed annotation into ``(source_id, label, field_provenance)``.

    Deliberately does **not** write anything. ``export_golden_labels`` is the one
    place that decides whether a label may enter the corpus — it checks the
    schema, the LoB enum, the ACORD form, and that no provenance names a
    registered confusable. A second writer here would be a second gate.
    """
    data = annotation.get("data") or {}
    source_id = data.get("source_id")
    if not source_id:
        raise ReviewToolError(
            "the completed annotation carries no source_id, so its label cannot be attributed "
            "to a document"
        )

    values = annotation.get("result") or annotation.get("values") or {}
    if not isinstance(values, dict):
        raise ReviewToolError(
            f"{source_id}: expected the reviewer's values as a mapping of field -> value, "
            f"got {type(values).__name__}"
        )

    label: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for key, value in values.items():
        if key.endswith(PROVENANCE_SUFFIX):
            surface = (value or "").strip() if isinstance(value, str) else ""
            if surface:
                provenance[key[: -len(PROVENANCE_SUFFIX)]] = surface
            continue
        # "null (undetermined)" is what the enum control shows for an explicit
        # null. It is a decision the reviewer made, not an empty box.
        if isinstance(value, str) and value.strip().lower().startswith("null"):
            label[key] = None
        else:
            label[key] = value

    # Provenance for a field nobody filled describes nothing, and would be
    # rejected downstream as naming an absent field.
    orphaned = [f for f in provenance if f not in label]
    for name in orphaned:
        log.warning("%s: provenance recorded for %r, which the reviewer left unfilled", source_id, name)
        provenance.pop(name)

    return str(source_id), label, provenance
