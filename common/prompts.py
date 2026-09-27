"""Prompt rendering — the single renderer used by every stage.

Corpus build (SPEC_05), evaluation (SPEC_08), serving (SPEC_11) and testing
(SPEC_12) all render prompts through :func:`render_system_prompt`. There is
deliberately no second path.

Why that matters: the system prompt is the conditioning input on every training
row *and* every production request. If the two ever diverge — by a character —
the fine-tuned model degrades at serving time, and **nothing in the training
metrics shows it**, because training never sees the serving prompt (arch §7).
``test_prompt_parity`` asserts the equality this module exists to provide.

The rendered prompt carries each field's ``description``: a semantic gloss with
exclusions, which is how the model maps an unseen surface label onto a canonical
key (arch §0c). It carries **no alias list** — see the template header for why.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from common.constants import MODALITY_MODES, PROMPT_MODE_ALIASES
from common.normalize import OUTPUT_DATE_LABEL
from common.schemas import is_canonical, required_fields, resolved_schema, schema_version

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

#: Bumped whenever the template's rendered output changes in any way.
#: Recorded in the corpus manifest; a change forces a corpus rebuild and a new
#: training cycle, exactly like a schema change (arch §7).
PROMPT_TEMPLATE_VERSION = "6.0.0"

_SYSTEM_TEMPLATE = "system_prompt_template.jinja"
_CLASSIFIER_TEMPLATE = "doc_type_classifier_prompt.jinja"

#: How the document type is named to the model in prose.
_DOC_TYPE_LABELS: dict[str, str] = {
    "lossrun": "Loss Run",
    "policy": "Policy",
    "acord": "ACORD",
}


class PromptError(RuntimeError):
    """Raised on an unknown modality mode or a template that fails to render."""


@lru_cache(maxsize=1)
def _env() -> Environment:
    # StrictUndefined: a typo'd variable must explode at render time, not
    # silently produce an empty string in a prompt we then train on.
    return Environment(
        loader=FileSystemLoader(str(PROMPT_DIR)),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=False,
    )


def effective_prompt_mode(modality_mode: str) -> str:
    """Map a modality mode onto the mode the *prompt* uses.

    ``noisy_ocr_image`` renders identically to ``ocr_plus_image``: the noise
    lives in the data, not the instruction (arch §6). Teaching the model that a
    special prompt accompanies bad OCR would defeat the point — at inference you
    do not know the OCR is bad.
    """
    if modality_mode not in MODALITY_MODES:
        raise PromptError(f"unknown modality_mode {modality_mode!r}; expected one of {MODALITY_MODES}")
    return PROMPT_MODE_ALIASES.get(modality_mode, modality_mode)


def doc_type_label(doc_type: str, acord_form: str | None = None) -> str:
    """Human-facing name for the document type, e.g. ``ACORD 25``."""
    label = _DOC_TYPE_LABELS.get(doc_type.lower())
    if label is None:
        raise PromptError(f"no label for doc_type {doc_type!r}")
    if doc_type.lower() == "acord" and acord_form:
        return f"{label} {acord_form}"
    return label


def schema_json_for_prompt(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """The schema as it is injected into the prompt.

    ``$ref``-resolved where the refs point at other files the model was never
    shown, left alone where they point inside the schema itself — the decision
    lives in :func:`common.schemas.resolved_schema`, which owns it for validation
    and rendering alike.

    Serialised with sorted keys and a fixed separator so rendering is
    deterministic; an unstable key order would break prompt parity for no reason.
    """
    schema = resolved_schema(doc_type, acord_form, lob, sections)
    stripped = _strip_authoring_keys(schema)
    return json.dumps(stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _strip_authoring_keys(node: Any) -> Any:
    """Drop keys meant for humans maintaining the schema, not for the model.

    ``$comment`` and ``$schema``/``$id`` are authoring metadata. Leaving them in
    spends tokens on every training row and every request to say nothing the
    model can act on.
    """
    if isinstance(node, list):
        return [_strip_authoring_keys(v) for v in node]
    if isinstance(node, dict):
        return {
            k: _strip_authoring_keys(v)
            for k, v in node.items()
            if k not in ("$comment", "$schema", "$id", "version")
        }
    return node


def _is_array(node: dict[str, Any]) -> bool:
    declared = node.get("type")
    return declared == "array" or (isinstance(declared, list) and "array" in declared)


def output_shape_for_prompt(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """A compact skeleton of the JSON this document type must return.

    **Derived from the schema, never hand-written.** A hand-kept copy of the key
    set in a prompt file drifts the first time a field is added, and the drift is
    silent: the model is shown one shape and validated against another. Rendering
    it from :func:`resolved_schema` means the two cannot disagree.

    Takes ``lob`` for the same reason :func:`schema_json_for_prompt` does, and
    must be called with the same value: one renders the outline and the other the
    detail, into the same prompt. Given different lines they would describe
    different field sets a few lines apart.

    Deliberately shape-only — no descriptions, no types. The schema follows
    immediately after and is the authority on both; this exists so the model sees
    the outline before the detail.
    """
    schema = resolved_schema(doc_type, acord_form, lob, sections)
    properties: dict[str, Any] = schema.get("properties", {})

    scalars = [name for name, node in properties.items() if not _is_array(node)]
    lines = [f'  {", ".join(scalars)}'] if scalars else []

    for name, node in properties.items():
        if not _is_array(node):
            continue
        row = list((node.get("items") or {}).get("properties", {}))
        lines.append(f'  {name}: [ {{ {", ".join(row)} }}, ... ]' if row else f"  {name}: [ ... ]")
    return "\n".join(lines)


def render_system_prompt(
    doc_type: str,
    modality_mode: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """Render the system prompt. **The** entrypoint — never bypass it.

    Args:
        doc_type: ``acord`` | ``policy`` | ``lossrun``.
        modality_mode: one of :data:`common.constants.MODALITY_MODES`.
        acord_form: required when ``doc_type == "acord"``, selects the schema.
        lob: a policy's line of business. Selects the per-LOB canonical schema
            when exactly one line is named and a schema is registered for it,
            per :func:`common.schemas.schema_key`. A homeowners policy and a
            personal auto policy share a header and little else, so showing the
            model the generic schema for both would train it on a field set that
            matches neither document.

    Returns:
        The system message, identical for the same inputs in every context.
    """
    mode = effective_prompt_mode(modality_mode)
    template = _env().get_template(_SYSTEM_TEMPLATE)
    return template.render(
        # `doc_type` selects the per-type rules block, `acord_form` the form
        # paragraph inside it, and `lob` the canonical schema — the same values
        # that select the schema, so a prompt can never describe one form or
        # line while carrying another's schema.
        doc_type=doc_type.lower(),
        acord_form=acord_form,
        # The client's canonical FieldValue shape (sparse envelopes) or the flat
        # one (every key, null for absence). Decided by the schema registry, so
        # the prompt can never describe one shape while carrying the other's
        # schema.
        canonical=is_canonical(doc_type, acord_form, lob),
        required_keys=_prose_list(required_fields(doc_type, acord_form, lob, sections)),
        output_shape=output_shape_for_prompt(doc_type, acord_form, lob, sections),
        doc_type_label=doc_type_label(doc_type, acord_form),
        modality_mode=mode,
        # Rendered from the constant the post-process formats with, not written
        # into the template. A literal here is a second source of truth for the
        # output date format, and the failure is silent in both directions: the
        # model is asked for one format and its answer rewritten into another,
        # which reads as the model getting dates wrong.
        date_format=OUTPUT_DATE_LABEL,
        schema_json=schema_json_for_prompt(doc_type, acord_form, lob, sections),
    ).strip()


def _prose_list(names: list[str]) -> str:
    """``carrier``, ``named_insured`` and ``policy`` — for a sentence, not a list."""
    quoted = [f"`{n}`" for n in names]
    if len(quoted) <= 1:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def render_classifier_prompt() -> str:
    """Render the zero-shot document-type classifier prompt (arch §4a, §4b)."""
    return _env().get_template(_CLASSIFIER_TEMPLATE).render().strip()


def prompt_fingerprint(
    doc_type: str,
    modality_mode: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """A short hash of the rendered prompt.

    Cheap way for a caller to assert the prompt it is about to send matches the
    one the corpus was built with, without diffing the whole string. Takes the
    same selectors as the render it hashes — a fingerprint blind to ``lob`` would
    report two different prompts as the same one, which is precisely the drift it
    exists to catch.
    """
    import hashlib

    rendered = render_system_prompt(doc_type, modality_mode, acord_form, lob, sections)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


def prompt_versions(
    doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None
) -> dict[str, str]:
    """Versions to record in the corpus manifest (arch §7).

    A change to either forces a corpus rebuild and a new training cycle.
    """
    return {
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "schema_version": schema_version(doc_type, acord_form, lob),
    }


def prompt_input_files() -> list[Path]:
    """Every file that shapes what the model is shown, in a fixed order.

    The templates (the per-doc-type ones in ``prompts/doc_types/`` included),
    the schemas the prompt embeds — ours and the client's canonical files — and
    the section map that slices a policy into windows. The release hash used to
    cover the top-level templates alone, so an edit to ``policy.jinja``, a
    canonical schema or a section group changed every served prompt and left
    the hash — the one thing meant to notice — exactly as it was.
    """
    from common.config import CONFIG_DIR
    from common.schemas import CANONICAL_DIR, SCHEMA_DIR

    return (
        sorted(PROMPT_DIR.rglob("*.jinja"))
        + sorted(SCHEMA_DIR.glob("*.json"))
        + sorted(CANONICAL_DIR.glob("*.json"))
        + [CONFIG_DIR / "schema_sections.yaml"]
    )


def prompt_hash() -> str:
    """SHA-256 over :func:`prompt_input_files`, each keyed by its repo path."""
    import hashlib

    root = PROMPT_DIR.parent
    digest = hashlib.sha256()
    for path in prompt_input_files():
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
