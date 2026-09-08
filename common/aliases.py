"""Alias registry — surface labels and confusables per canonical field.

.. warning::

   **No serving or inference path may import this module.** Mapping extracted
   labels onto canonical keys with a lookup table at request time caps the system
   at whatever list someone wrote, cannot handle the phrasing nobody anticipated,
   and defeats the reason for fine-tuning a VLM at all (master §1.4 anti-pattern).
   ``tests/test_no_runtime_aliases.py`` enforces this with an import check rather
   than trusting discipline.

The registry has exactly three legitimate consumers:

* **SPEC_04** — annotator guidance, and rejecting a label whose ``field_provenance``
  names a registered confusable.
* **SPEC_05** — ``alias_coverage`` counting in the corpus manifest.
* **SPEC_08** — per-alias evaluation slicing and confusable misattribution.

It is **never rendered into the prompt**. Alias lists give the model a lexical
prior that makes confusable errors worse — tell it ``insured_name`` may appear as
"Insured" and a page containing "Additional Insured" substring-matches. The
prompt carries semantic glosses instead, which encode the *boundary* an alias
list structurally cannot (arch §0c).

Adding a newly observed surface form here is a **registry edit, not a schema
edit** — it triggers no corpus rebuild. That asymmetry is deliberate: the
registry grows freely as annotators meet new phrasings, while the glosses that
*are* prompt text stay stable.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import NamedTuple

from common.normalize import normalize_text

ALIAS_DIR = Path(__file__).resolve().parent.parent / "schemas" / "aliases"


class AliasRegistryError(RuntimeError):
    """Raised when a registry is missing or internally contradictory."""


class FieldAliases(NamedTuple):
    """Observed surface labels for one canonical field, and its confusables."""

    field: str
    aliases: tuple[str, ...]
    confusables: tuple[str, ...]


def _registry_path(doc_type: str) -> Path:
    return ALIAS_DIR / f"{doc_type.lower()}.aliases.json"


@cache
def load_registry(doc_type: str) -> dict[str, FieldAliases]:
    """Load and validate one document type's registry.

    An empty registry is legitimate — it means nothing has been derived yet
    (SPEC_04 ``derive_aliases`` builds it from labeled documents). A *missing*
    file is also legitimate for the same reason, so it returns empty rather than
    raising: the registry is diagnostics infrastructure, and its absence must not
    block a corpus build.
    """
    path = _registry_path(doc_type)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    registry: dict[str, FieldAliases] = {}
    for field, entry in raw.items():
        if field.startswith("$"):  # metadata keys
            continue
        aliases = tuple(entry.get("aliases", []) or [])
        confusables = tuple(entry.get("confusables", []) or [])
        overlap = (
            {n for a in aliases if (n := normalize_text(a))}
            & {n for c in confusables if (n := normalize_text(c))}
        )
        if overlap:
            raise AliasRegistryError(
                f"{path.name}: field {field!r} lists {sorted(overlap)} as BOTH an alias and a "
                "confusable. A label cannot both denote the field and be something it must not "
                "be confused with — one of the two entries is wrong."
            )
        registry[field] = FieldAliases(field, aliases, confusables)
    return registry


def aliases_for(doc_type: str, field: str) -> tuple[str, ...]:
    """Observed surface labels for a canonical field."""
    entry = load_registry(doc_type).get(field)
    return entry.aliases if entry else ()


def confusables_for(doc_type: str, field: str) -> tuple[str, ...]:
    """Labels whose values must not be returned as this field."""
    entry = load_registry(doc_type).get(field)
    return entry.confusables if entry else ()


def canonical_for(doc_type: str, surface_label: str) -> str | None:
    """Reverse lookup: a surface label -> the canonical field it denotes.

    **For labeling and evaluation only.** Using this at inference is the
    anti-pattern this module's docstring prohibits.
    """
    target = normalize_text(surface_label)
    if target is None:
        return None
    for field, entry in load_registry(doc_type).items():
        if any(normalize_text(a) == target for a in entry.aliases):
            return field
    return None


def is_confusable(doc_type: str, field: str, surface_label: str) -> bool:
    """Whether a label is a registered confusable for a field.

    SPEC_04 uses this to reject a golden label claiming ``insured_name`` was
    found under "Certificate Holder" — the highest-value annotation check there
    is, because that mistake teaches the model to conflate distinct parties.
    """
    target = normalize_text(surface_label)
    if target is None:
        return False
    return any(normalize_text(c) == target for c in confusables_for(doc_type, field))


def all_confusable_pairs(doc_type: str) -> list[tuple[str, str]]:
    """Every ``(canonical_field, confusable_label)`` pair, for eval and reporting."""
    return [
        (field, label)
        for field, entry in load_registry(doc_type).items()
        for label in entry.confusables
    ]


def registry_stats(doc_type: str) -> dict[str, int]:
    """Coarse registry size, for the corpus manifest and coverage reports."""
    registry = load_registry(doc_type)
    return {
        "fields": len(registry),
        "aliases": sum(len(e.aliases) for e in registry.values()),
        "confusables": sum(len(e.confusables) for e in registry.values()),
    }
