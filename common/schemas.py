"""Schema registry — load, resolve, and validate against the canonical schemas.

The schemas in ``schemas/`` are the Fideon SPEC_00 canonical models serialised to
JSON Schema (master §1.1). They are the training target *and* the inference
output contract, so this module is the single place they get loaded.

Two things here are load-bearing rather than incidental:

``$ref`` resolution
    ``common_fields.json`` and ``lob.enum.json`` hold definitions shared across
    document types. Defining a field per-schema instead would let two
    definitions of the same canonical field drift apart, and no test would catch
    it because each schema still validates on its own (SPEC_01).

Field descriptions
    Every field carries a ``description`` — a semantic gloss with exclusions.
    These are **prompt text**: they are injected into the system prompt at
    corpus-build and inference time, so they are part of what the model trains
    on (arch §0c). :func:`assert_all_fields_described` enforces their presence.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from common.constants import ACORD_FORMS, ACTIVE_DOC_TYPES

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"

#: Shared definition files, loaded into the registry so ``$ref`` can reach them.
_SHARED = ("common_fields.json", "lob.enum.json")

#: doc_type (+ acord_form) -> schema filename.
_SCHEMA_FILES: dict[str, str] = {
    "lossrun": "lossrun.schema.json",
    "policy": "policy_doc.schema.json",
    "acord:25": "acord25.schema.json",
    "acord:125": "acord125.schema.json",
    "acord:140": "acord140.schema.json",
}


class SchemaError(RuntimeError):
    """Raised when a schema is missing, unresolvable, or structurally invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SchemaError(f"schema file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _registry() -> Registry:
    """A ``referencing`` registry holding every schema, keyed by filename.

    Refs are written relative (``common_fields.json#/$defs/insured_name``), so
    registering under the bare filename is what makes them resolve.
    """
    registry = Registry()
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        contents = _load_json(path)
        resource = Resource.from_contents(contents, default_specification=DRAFT202012)
        registry = resource @ registry  # register under its own $id
        registry = registry.with_resource(path.name, resource)  # and under filename
    return registry


def schema_key(
    doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None
) -> str:
    """Build the registry key for a document type, form and line of business.

    ACORD is two-level (arch §4b): one shared adapter, but a distinct schema per
    form, because field sets genuinely differ between 25, 125 and 140.

    **Policies are two-level too, by line of business.** A Workers' Comp policy
    and a Commercial Auto policy share a header and almost nothing else, so each
    LOB may register its own canonical schema (``policy:workers_comp``).

    Two rules, both here rather than at the call sites that would otherwise each
    decide for themselves:

    * **Fall back when no per-LOB schema is registered.** Adding one later is a
      single ``_SCHEMA_FILES`` entry and no call-site change.
    * **An LOB selects a schema only when exactly one is named.** A package
      policy covering GL, Property and Auto is one document with a section per
      line, so it uses the generic policy schema — picking one of its lines would
      validate the whole document against a third of itself.
    """
    doc_type = doc_type.lower()
    if doc_type == "acord":
        if acord_form is None:
            raise SchemaError(
                "acord requires an acord_form (25 | 125 | 140) to select a schema — "
                "the classifier is two-level for exactly this reason (arch §4b)"
            )
        form = str(acord_form).strip()
        if form not in ACORD_FORMS:
            raise SchemaError(f"unknown ACORD form {form!r}; known forms: {sorted(ACORD_FORMS)}")
        return f"acord:{form}"
    if doc_type not in ACTIVE_DOC_TYPES:
        raise SchemaError(f"unknown doc_type {doc_type!r}; active types: {ACTIVE_DOC_TYPES}")

    lines = [lob] if isinstance(lob, str) else list(lob or [])
    if len(lines) == 1:
        scoped = f"{doc_type}:{str(lines[0]).strip().lower()}"
        if scoped in _SCHEMA_FILES:
            return scoped
    return doc_type


def load_schema(
    doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None
) -> dict[str, Any]:
    """Return the raw (unresolved) schema for a document type."""
    return _schema_for_key(schema_key(doc_type, acord_form, lob))


def validator_for(
    doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None
) -> Draft202012Validator:
    """A ref-resolving validator for this document type."""
    return _validator_for_key(schema_key(doc_type, acord_form, lob))


# Cached on the resolved KEY rather than on the arguments: `lob` arrives as a
# list (line_of_business is a list under §0b) and a list is unhashable, so
# caching the public signature would raise on every multi-LOB call. The key is
# also the right cache granularity — two argument sets that resolve to one schema
# should share one parsed copy.
@cache
def _schema_for_key(key: str) -> dict[str, Any]:
    return _load_json(SCHEMA_DIR / _SCHEMA_FILES[key])


@cache
def _validator_for_key(key: str) -> Draft202012Validator:
    return Draft202012Validator(_schema_for_key(key), registry=_registry())


def validate(
    instance: Any, doc_type: str, acord_form: str | None = None,
    lob: str | list[str] | None = None,
) -> None:
    """Validate an extraction or golden label. Raises ``ValidationError``.

    Used by SPEC_04 before a golden label is admitted to the corpus, and by
    SPEC_11 before an inference response is returned — mirroring the Fideon
    SPEC_07 Stage 3 audit gate (arch §0a).
    """
    validator_for(doc_type, acord_form, lob).validate(instance)


def is_valid(
    instance: Any, doc_type: str, acord_form: str | None = None,
    lob: str | list[str] | None = None,
) -> bool:
    """Non-raising form of :func:`validate`."""
    return validator_for(doc_type, acord_form, lob).is_valid(instance)


def iter_validation_errors(
    instance: Any, doc_type: str, acord_form: str | None = None,
    lob: str | list[str] | None = None,
) -> Iterator[str]:
    """Human-readable validation errors, best-match first."""
    for err in sorted(validator_for(doc_type, acord_form, lob).iter_errors(instance), key=str):
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        yield f"{where}: {err.message}"


def required_fields(
    doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None
) -> list[str]:
    """Top-level required field names for this document type."""
    return list(load_schema(doc_type, acord_form, lob).get("required", []))


def schema_version(
    doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None
) -> str:
    """The schema's declared version, recorded in the corpus manifest (arch §7).

    A change here forces a corpus rebuild and a new training cycle.
    """
    schema = load_schema(doc_type, acord_form, lob)
    version = schema.get("version")
    if not version:
        raise SchemaError(f"schema for {doc_type}/{acord_form} declares no version")
    return str(version)


def resolved_schema(
    doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None
) -> dict[str, Any]:
    """The schema with every ``$ref`` inlined.

    This is what gets injected into the system prompt — the model must see the
    actual field definitions and their descriptions, not a pointer to another
    file it cannot follow.
    """
    registry = _registry()
    resolver = registry.resolver()

    def _resolve(node: Any, depth: int = 0) -> Any:
        if depth > 25:
            raise SchemaError("$ref nesting too deep — probable reference cycle")
        if isinstance(node, list):
            return [_resolve(item, depth + 1) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = resolver.lookup(node["$ref"])
            merged = _resolve(target.contents, depth + 1)
            # Keep any sibling keys that sat alongside the $ref.
            extra = {k: _resolve(v, depth + 1) for k, v in node.items() if k != "$ref"}
            return {**merged, **extra} if extra else merged
        return {k: _resolve(v, depth + 1) for k, v in node.items()}

    return _resolve(load_schema(doc_type, acord_form, lob))


def iter_described_fields(schema: dict[str, Any], prefix: str = "") -> Iterator[tuple[str, str | None]]:
    """Walk a resolved schema yielding ``(field_path, description)`` for each field.

    Recurses into object properties and array items so nested fields —
    ``claims[].total_incurred``, ``coverage_schedule[].limit`` — are covered too.
    """
    for name, node in (schema.get("properties") or {}).items():
        if not isinstance(node, dict):
            continue
        path = f"{prefix}{name}"
        yield path, node.get("description")
        if node.get("properties"):
            yield from iter_described_fields(node, prefix=f"{path}.")
        items = node.get("items")
        if isinstance(items, dict) and items.get("properties"):
            yield from iter_described_fields(items, prefix=f"{path}[].")


def assert_all_fields_described(doc_type: str, acord_form: str | None = None) -> None:
    """Fail loudly if any field lacks a ``description``.

    Descriptions are prompt text, so a missing one silently removes the model's
    only semantic anchor for that field (arch §0c). This is a SPEC_01 acceptance
    criterion, enforced here so it is checkable rather than aspirational.
    """
    missing = [
        path
        for path, desc in iter_described_fields(resolved_schema(doc_type, acord_form))
        if not (desc or "").strip()
    ]
    if missing:
        label = f"{doc_type}" + (f"/{acord_form}" if acord_form else "")
        raise SchemaError(
            f"{label}: {len(missing)} field(s) have no description, so the model gets no "
            f"semantic anchor for them (arch §0c): {', '.join(sorted(missing))}"
        )


def all_schema_keys() -> list[tuple[str, str | None]]:
    """Every (doc_type, acord_form) pair with a schema — for tests and sweeps."""
    out: list[tuple[str, str | None]] = []
    for key in _SCHEMA_FILES:
        if ":" in key:
            doc_type, form = key.split(":", 1)
            out.append((doc_type, form))
        else:
            out.append((key, None))
    return out
