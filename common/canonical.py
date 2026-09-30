"""The client's canonical ``FieldValue`` JSON, and the two shapes around it.

Every policy is extracted into the client's canonical schema
(``configs/canonical schema/LOB Schema``), where each leaf is an envelope::

    {"raw": "04/01/2026", "parsed": "04/01/2026",
     "confidence": {"score": 0.97, "source": "vlm"}, "page_ref": [1], "flagged": false}

Three shapes of one document meet here, and this module is the only place that
converts between them:

**Canonical** — the client's contract, what the endpoint returns and what a
golden label is written in. Validated against the client's file byte for byte.

**Model form** — what the model writes: the same tree, each leaf narrowed to
``raw``/``parsed``/``page_ref``, and **sparse** — a field the document does not
state is omitted rather than written as an envelope of nulls. Sparse is not a
style choice: homeowners declares 532 leaves, and an envelope for each is some
twenty thousand output tokens before the first value. ``confidence`` and
``flagged`` are never the model's to write; the pipeline fills them from the
calibrated logprobs and the review thresholds.

**Values view** — each envelope collapsed to its ``parsed`` value (``raw`` when
nothing was parsed). Calibration features, row completeness and field scoring
all read values, and reading them off this view means none of them had to learn
what an envelope is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from common.normalize import format_output_date, infer_field_kind

#: The keys of the client's ``FieldValue`` envelope.
FIELD_VALUE_KEYS: frozenset[str] = frozenset({"raw", "parsed", "confidence", "page_ref", "flagged"})

#: The subset the model writes (``common.schemas.MODEL_FIELD_VALUE``).
MODEL_VALUE_KEYS: tuple[str, ...] = ("raw", "parsed", "page_ref")

#: Written into ``confidence.source`` for every value this model produced. One of
#: the four the client's enum permits.
CONFIDENCE_SOURCE = "vlm"


class CanonicalLabelError(ValueError):
    """Raised when a label that should be canonical is not."""


def is_field_value(node: Any) -> bool:
    """Whether ``node`` is one leaf's envelope, in either the full or model form."""
    return (
        isinstance(node, dict)
        and "raw" in node
        and "parsed" in node
        and set(node) <= FIELD_VALUE_KEYS
    )


def _is_date(path: str) -> bool:
    return infer_field_kind(path) == "date"


def _output_value(path: str, value: Any) -> Any:
    """``parsed`` as it leaves the pipeline: a date in ``MM/DD/YYYY``.

    A date that cannot be read is kept as written, never replaced with ``None``:
    losing a value because the formatter could not parse it would turn a
    formatting question into a false null.
    """
    if value is None or not _is_date(path):
        return value
    return format_output_date(value) or value


def _join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


# --------------------------------------------------------------------------
# Canonical -> model form (training targets)
# --------------------------------------------------------------------------


def to_model_target(label: dict[str, Any], *, required: Iterable[str] = ()) -> dict[str, Any]:
    """The training target for a canonical golden label.

    Narrows every envelope to ``raw``/``parsed``/``page_ref``, writes dates in
    ``MM/DD/YYYY``, and drops what the document does not state — the exact shape
    the prompt asks the model to produce, so the target and the instruction agree.

    ``required`` are the schema's top-level required objects. They are always
    emitted, as ``{}`` when empty, because the canonical schema requires them and
    a target that omitted them would teach an output that fails validation.

    Raises :class:`CanonicalLabelError` for a label with a bare value where an
    envelope belongs — a flat, pre-canonical label. Training on one under a
    canonical prompt would teach the model the wrong shape, silently.
    """
    if not isinstance(label, dict):
        raise CanonicalLabelError(f"a canonical label is a JSON object, not {type(label).__name__}")
    target = _slim(label, "") or {}
    for key in required:
        target.setdefault(key, {})
    return target


def _slim(node: Any, path: str) -> Any:
    if is_field_value(node):
        if node.get("raw") is None and node.get("parsed") is None:
            return None
        return {
            "raw": node.get("raw"),
            "parsed": _output_value(path, node.get("parsed")),
            "page_ref": [int(p) for p in (node.get("page_ref") or [])],
        }
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            slim = _slim(value, _join(path, key))
            if slim not in (None, {}, []):
                out[key] = slim
        return out
    if isinstance(node, list):
        rows = [_slim(item, f"{path}[{index}]") for index, item in enumerate(node)]
        return [row for row in rows if row not in (None, {}, [])]
    if node is None:
        return None
    raise CanonicalLabelError(
        f"{path or '<root>'} holds a bare value ({node!r}) where the canonical schema puts a "
        "FieldValue envelope. This is a flat, pre-canonical label; convert it to the canonical "
        "schema before it enters the corpus."
    )


def in_schema_order(node: Any, schema: dict[str, Any]) -> Any:
    """``node`` with every object's keys in the order its schema declares them.

    Structured decoding (xgrammar) builds each object's grammar from its
    properties IN ORDER: a key may be skipped, never written after one declared
    later. Labels keep whatever order they were written in - some template
    families put ``form_title`` before ``form_number`` - and the targets kept it,
    so the model learned to open a form with its title, after which the grammar
    no longer allowed the number. On the smoke run every form came back without
    one. Training on schema order makes what the model learned writable.

    Keys the schema does not declare keep their place after the declared ones.
    Values are untouched; only key order changes.
    """
    defs = schema.get("$defs") or {}

    def resolve(sub: Any) -> dict[str, Any]:
        while isinstance(sub, dict) and "$ref" in sub:
            ref = sub["$ref"]
            sub = defs.get(ref.rsplit("/", 1)[-1]) if ref.startswith("#/$defs/") else None
        return sub if isinstance(sub, dict) else {}

    def walk(value: Any, sub: Any) -> Any:
        sub = resolve(sub)
        if isinstance(value, dict):
            props = sub.get("properties") or {}
            ordered = {key: walk(value[key], props[key]) for key in props if key in value}
            for key, item in value.items():
                if key not in ordered:
                    ordered[key] = walk(item, None)
            return ordered
        if isinstance(value, list):
            return [walk(item, sub.get("items")) for item in value]
        return value

    return walk(node, schema)


def training_target(
    label: dict[str, Any],
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
) -> dict[str, Any]:
    """What the model is trained to write for one golden label.

    Chosen by the same selectors that choose the prompt's schema, so the target
    and the prompt describe one shape: the sparse model form for a canonical
    policy, the label itself for a flat type — dates in ``MM/DD/YYYY`` either way.
    Golden labels keep whatever date format they were written in; only the target
    is converted.
    """
    from common.schemas import is_canonical, required_fields, resolved_schema

    # In the order the decoding grammar writes keys (in_schema_order).
    schema = resolved_schema(doc_type, acord_form, lob)
    if is_canonical(doc_type, acord_form, lob):
        target = to_model_target(label, required=required_fields(doc_type, acord_form, lob))
        return in_schema_order(target, schema)
    return in_schema_order(with_output_dates(label), schema)


# --------------------------------------------------------------------------
# Model form -> canonical (serving)
# --------------------------------------------------------------------------


def with_output_dates(node: Any, path: str = "") -> Any:
    """Every date in ``MM/DD/YYYY``, whichever shape the document is in.

    In a canonical document dates live in an envelope's ``parsed`` (``raw`` stays
    as printed); in a flat one — ACORD, Loss Run — they are the values
    themselves. Applied to training targets and to every generation, so the
    format is a guarantee rather than something the prompt merely asks for.
    """
    if is_field_value(node):
        return {**node, "parsed": _output_value(path, node.get("parsed"))}
    if isinstance(node, dict):
        return {k: with_output_dates(v, _join(path, k)) for k, v in node.items()}
    if isinstance(node, list):
        return [with_output_dates(v, f"{path}[{i}]") for i, v in enumerate(node)]
    return _output_value(path, node)


def envelope(
    generated: dict[str, Any],
    confidence: Mapping[str, tuple[float, bool]],
) -> dict[str, Any]:
    """The model's output as the client's canonical JSON.

    ``confidence`` maps a values-view path (``policy.effective_date``,
    ``locations[0].address.city``) to ``(calibrated score, needs review)``. A
    leaf with no entry gets score 0 and is flagged: a field with no measured
    confidence is one a person has to look at, never one that ships as certain.

    A list of envelopes is one field in the values view (it is scored as a set),
    so each of its elements carries that field's score.
    """
    return _envelope(generated, "", confidence)


def _envelope(node: Any, path: str, confidence: Mapping[str, tuple[float, bool]]) -> Any:
    if is_field_value(node):
        score, flagged = confidence.get(path, (0.0, True))
        return {
            "raw": node.get("raw"),
            "parsed": node.get("parsed"),
            "confidence": {"score": round(float(score), 4), "source": CONFIDENCE_SOURCE},
            "page_ref": [int(p) for p in (node.get("page_ref") or [])],
            "flagged": bool(flagged),
        }
    if isinstance(node, dict):
        return {k: _envelope(v, _join(path, k), confidence) for k, v in node.items()}
    if isinstance(node, list):
        if node and all(is_field_value(item) for item in node):
            return [_envelope(item, path, confidence) for item in node]
        return [_envelope(item, f"{path}[{i}]", confidence) for i, item in enumerate(node)]
    return node


# --------------------------------------------------------------------------
# Values view (calibration, completeness, scoring)
# --------------------------------------------------------------------------


def values_view(node: Any) -> Any:
    """Every envelope collapsed to its value: ``parsed``, else ``raw``.

    A no-op on a document with no envelopes, so a caller can apply it without
    first asking which kind of document it holds.
    """
    if is_field_value(node):
        parsed = node.get("parsed")
        return parsed if parsed is not None else node.get("raw")
    if isinstance(node, dict):
        return {k: values_view(v) for k, v in node.items()}
    if isinstance(node, list):
        return [values_view(v) for v in node]
    return node


def printed_view(node: Any) -> Any:
    """Every envelope collapsed to what the PAGE shows: ``raw``, else ``parsed``.

    For aligning a label against OCR text, where the printed form is the one
    that can be found — ``April 1, 2026``, not ``04/01/2026``. A no-op on a flat
    document.
    """
    if is_field_value(node):
        raw = node.get("raw")
        return raw if raw is not None else node.get("parsed")
    if isinstance(node, dict):
        return {k: printed_view(v) for k, v in node.items()}
    if isinstance(node, list):
        return [printed_view(v) for v in node]
    return node


def has_envelopes(node: Any) -> bool:
    """Whether a document is in the canonical shape at all."""
    if is_field_value(node):
        return True
    if isinstance(node, dict):
        return any(has_envelopes(v) for v in node.values())
    if isinstance(node, list):
        return any(has_envelopes(v) for v in node)
    return False


def leaf_values(node: Any, path: str = "") -> dict[str, Any]:
    """``path -> value`` for every field, envelopes collapsed to their value.

    Rows are indexed (``locations[0].address.city``); a list of scalars is one
    field, kept whole.
    """
    if is_field_value(node):
        return {path: values_view(node)}
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(leaf_values(value, _join(path, key)))
        return out
    if isinstance(node, list):
        if all(is_field_value(v) or not isinstance(v, dict | list) for v in node):
            return {path: values_view(node)}
        out = {}
        for index, value in enumerate(node):
            out.update(leaf_values(value, f"{path}[{index}]"))
        return out
    return {path: node}


def field_paths(node: Any, path: str = "") -> set[str]:
    """Every field path in a document: ``named_insured.primary_name``, and for a
    flat document its plain keys. An envelope is one field; its keys are not."""
    if is_field_value(node) or not isinstance(node, dict | list):
        return {path} if path else set()
    if isinstance(node, list):
        return {p for i, v in enumerate(node) for p in field_paths(v, f"{path}[{i}]")}
    out: set[str] = set()
    for key, value in node.items():
        child = _join(path, key)
        out.add(child)
        out |= field_paths(value, child)
    return out


def collapse_spans(spans: Mapping[str, Any]) -> dict[str, Any]:
    """Re-key token spans from envelope paths to values-view paths.

    The span mapper addresses every scalar it finds, so a canonical leaf arrives
    as ``policy.effective_date.raw``, ``….parsed`` and ``….page_ref[0]``. The
    value that is calibrated is ``parsed``, so its span becomes the field's; when
    nothing was parsed, ``raw``'s stands in, matching :func:`values_view`.
    ``page_ref`` tokens say where a value was found, not what it is, and are
    dropped.

    Spans with no envelope suffix are returned untouched, so a flat document
    passes through unchanged.
    """
    if not any(path.endswith(".parsed") for path in spans):
        return dict(spans)

    out: dict[str, Any] = {}
    for path, span in spans.items():
        if not path.endswith(".parsed"):
            continue
        field = path[: -len(".parsed")]
        raw = spans.get(f"{field}.raw")
        if getattr(span, "value", None) is None and raw is not None:
            span = raw
        out[field] = span
    return out
