"""Build ``schemas/aliases/policy.aliases.json`` from the canonical schemas and
the original documents.

**One entry per canonical field.** The schemas mention the same field in many
places — ``effective_date`` sits under the policy, under each coverage part,
under every scheduled underlying policy — and a registry keyed by location
repeats the field once per place. Those copies then disagree: the entry under
the policy knew ten labels for an effective date while the one under a coverage
part knew two, so the same printed label counted as an alias in one place and
not in another. Keyed by the field itself, there is one list and it cannot
disagree with itself. ``occurs_at`` records where the field is used.

Two sources, doing different jobs:

* **The canonical schemas** (``configs/canonical schema/LOB Schema/``) carry
  ``fideon:aliases`` — the labels forms print for each field. They say what to
  look for. A SPEC_21 line overlay is read through the bundle it loads as
  (:func:`walk_bundle`): its fields are the common model's, and so are most of
  its aliases, with the overlay's own listed by field path.
* **The original documents** (``training data/original data/``) say which of
  those labels real pages actually print, and how often. An alias with no
  document evidence is either a phrasing this book of business does not use, or
  a mistake in the schema; a reader should be able to tell it apart from one
  seen on four hundred pages.

**A field's own name counts as a candidate label, but only when a document
prints it.** ``expiration_date`` is printed as "Expiration Date" on 325 of these
documents, so that is a fact rather than a guess — while a field simply called
``name`` or ``date`` is skipped, because the word appears on every page and says
nothing about which field it labels. Attaching "Name" to the twenty fields
called ``name`` would destroy the one thing this file is for: telling fields
apart.

**What this registry is, and is not.** It records which printed phrasings map to
which canonical field, so corpus coverage can be measured and evaluation can
report per alias. It is **never rendered into a prompt and never consulted at
inference** (master §1.4, enforced by ``tests/test_no_runtime_aliases.py``): an
alias table at inference is a lookup pretending to be comprehension, and it fails
silently on the first phrasing nobody listed.

Run after a schema change, or when documents are added::

    python -m scripts.derive_aliases_from_canonical --write
    python -m scripts.derive_aliases_from_canonical --write --no-documents   # skip mining
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CANONICAL = ROOT / "configs" / "canonical schema" / "LOB Schema"
DOCUMENTS = ROOT / "training data" / "original data"
TARGET = ROOT / "schemas" / "aliases" / "policy.aliases.json"

#: An alias shorter than this matches inside other words, so counting it would
#: report noise as evidence ("To", "From"). Such aliases stay in the registry;
#: only their evidence is skipped.
MIN_EVIDENCE_LENGTH = 4

#: Hand-written aliases, kept because the canonical schemas carry aliases for the
#: SPECIFIC fields (prior_policy_number, certificate_number) and none at all for
#: the obvious ones — policy_number, company_name and agency_name have no
#: ``fideon:aliases`` in any schema. Dropping these would leave the fields every
#: document prints with no phrasings recorded.
SEED_ALIASES: dict[str, list[str]] = {
    "primary_name": [
        "Insured Name", "Named Insured", "Insured", "Applicant",
        "Name of Applicant", "Applicant Name", "Name of Insured", "Name Insured",
    ],
    "policy_number": [
        "Policy Number", "Policy No.", "Policy #", "Policy No", "Pol. No.",
    ],
    "company_name": [
        "Carrier", "Insurer", "Insurance Company", "Underwritten By", "Company",
    ],
    "agency_name": [
        "Producer", "Agency", "Broker", "Agent", "Producer Name",
    ],
    "effective_date": [
        "Effective Date", "Effective", "Policy Effective Date", "Inception Date", "From",
    ],
    "expiration_date": [
        "Expiration Date", "Expiration", "Policy Expiration Date", "Expiry Date", "To",
    ],
    "total_policy_premium": [
        "Total Premium", "Policy Premium", "Total Policy Premium", "Premium",
    ],
}

#: Hand-written, because a schema cannot state that two labels look alike and
#: mean different things — the distinction the misattribution metric scores.
CONFUSABLES: dict[str, list[str]] = {
    "primary_name": [
        "Certificate Holder", "Producer", "Agency", "Additional Insured",
        "Loss Payee", "Mortgagee", "Carrier", "Insurer",
    ],
    "policy_number": [
        "Quote Number", "Binder Number", "Claim Number", "Submission Number",
        "Certificate Number",
    ],
    "company_name": [
        "Producer", "Agency", "Broker", "Named Insured", "Administrator",
    ],
    "agency_name": [
        "Carrier", "Insurer", "Named Insured", "Underwriter",
    ],
    "effective_date": [
        "Issue Date", "Date Printed", "Expiration Date", "Bind Date",
    ],
    "expiration_date": [
        "Effective Date", "Cancellation Date", "Renewal Date",
    ],
    "total_policy_premium": [
        "Deposit Premium", "Minimum Premium", "Estimated Premium",
        "Premium by Coverage", "Taxes and Fees",
    ],
}

#: The repo's own policy schema names these fields differently, and three
#: consumers look them up that way: the confusable co-occurrence count
#: (corpus_manifest), the provenance check that refuses a label naming a
#: confusable (export_golden_labels), and the review tool's "NEVER take from"
#: instruction. Until the canonical schemas ARE the repo's schemas, both namings
#: must be present — a registry serving only the canonical one silently empties
#: all three, which the test suite catches.
LEGACY_KEYS: dict[str, str] = {
    "insured_name": "primary_name",
    "carrier": "company_name",
    "producer": "agency_name",
    "total_premium": "total_policy_premium",
}

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Case- and whitespace-insensitive form, for matching a label on a page."""
    return _WS.sub(" ", text.replace(" ", " ")).strip().casefold()


def field_name(field_path: str) -> str:
    """The field itself, without where it sits: ``policy.effective_date`` -> ``effective_date``."""
    return field_path.split(".")[-1].replace("[]", "")


def own_label(name: str) -> str | None:
    """The printed label a field's own name suggests, or ``None``.

    Multi-word names only. A field called ``name`` or ``date`` shares its word
    with every page in the corpus, so matching it says nothing about which field
    a page labels — and adding it would attach one string to twenty fields.
    """
    words = [w for w in name.split("_") if w]
    if len(words) < 2:
        return None
    return " ".join(w.capitalize() for w in words)


def walk(node: Any, path: str = "") -> Iterator[tuple[str, str, list[str]]]:
    """Yield ``(field_path, kind, aliases)`` for every field in a schema."""
    if not isinstance(node, dict):
        return
    aliases = list(node.get("fideon:aliases") or [])

    if node.get("$ref", "").endswith("FieldValue"):
        yield path, "leaf", aliases
        return
    if node.get("type") == "array":
        items = node.get("items")
        if isinstance(items, dict):
            yield path, "table", aliases
            yield from walk(items, f"{path}[]")
        return

    properties = node.get("properties") or {}
    if path:
        yield path, "section" if properties else "leaf", aliases
    for key, value in properties.items():
        yield from walk(value, f"{path}.{key}" if path else key)


def unify(field_path: str, line_block: str | None) -> str:
    """Strip the per-LOB block, so one path is one path across every line."""
    if line_block and field_path.startswith(f"{line_block}."):
        return field_path[len(line_block) + 1:]
    return field_path


def common_model_bundle(line: str) -> dict[str, Any] | None:
    """The bundle a SPEC_21 overlay loads as, or ``None`` for any other file.

    Only a file the loader selects for its own line counts. ``classic_auto`` is
    read as personal auto, so the common-model switch answers for personal
    auto's overlay, while ``classic_auto.json`` is a self-contained file of its
    own; ``_common`` is no line's file. Both are read as they are.
    ``_fallback`` is no line's file either, but the loader reads it for a policy
    of no line, so it counts once it is an overlay (common model 1.1.0).
    """
    from common import schemas

    if line == schemas.CANONICAL_FALLBACK:
        if not schemas.is_common_model("policy", None, None):
            return None
        return schemas.load_schema("policy", None, None)
    if schemas.schema_key("policy", None, line) != f"policy:{line}":
        return None
    if not schemas.is_common_model("policy", None, line):
        return None
    return schemas.load_schema("policy", None, line)


def walk_bundle(bundle: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    """:func:`walk` for a SPEC_21 line, read through the bundle its overlay loads as.

    Each block of an overlay is a ``$ref`` into the common model, so :func:`walk`
    would record every block as a leaf with no aliases and reach none of the
    line's fields. Here each ``#/$defs/...`` reference is followed into the
    definition it names; a value (``FieldValue`` or a typed one, ``MoneyValue``,
    ``DateValue``...) is one leaf; and the ``fideon:`` properties (provenance)
    are not fields.

    SPEC_21 keeps aliases in two places, and both are read: on the common
    model's fields and blocks, and in the overlay's ``fideon:aliases``, keyed by
    field path. An overlay alias for a path the line does not declare is refused
    rather than dropped, since nothing else would show it was lost.
    """
    defs = bundle.get("$defs") or {}
    rows = [
        row
        for key, node in (bundle.get("properties") or {}).items()
        if not key.startswith("fideon:")
        for row in _walk_refs(node, defs, key, frozenset())
    ]
    line_aliases: dict[str, list[str]] = bundle.get("fideon:aliases") or {}
    if unknown := sorted(set(line_aliases) - {field_path for field_path, _, _ in rows}):
        raise ValueError(
            f"{bundle.get('fideon:lob')}: fideon:aliases names fields the schema does not "
            f"declare: {unknown}"
        )
    return [
        (field_path, kind, [*aliases, *line_aliases.get(field_path, [])])
        for field_path, kind, aliases in rows
    ]


def _walk_refs(
    node: Any, defs: dict[str, Any], path: str, open_defs: frozenset[str]
) -> Iterator[tuple[str, str, list[str]]]:
    """:func:`walk` for one field of a bundle, following its local references."""
    if not isinstance(node, dict):
        return
    aliases = list(node.get("fideon:aliases") or [])

    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        name = ref.split("/")[2]
        target = defs.get(name)
        # A value is one field, never its raw/parsed/page_ref. A definition
        # already open on this path contains itself: recorded, not entered again.
        if not isinstance(target, dict) or _is_value(target) or name in open_defs:
            yield path, "leaf", aliases
            return
        # A block's aliases sit on its definition; a field's beside its reference.
        merged = {**target, **{k: v for k, v in node.items() if k != "$ref"},
                  "fideon:aliases": [*aliases, *(target.get("fideon:aliases") or [])]}
        yield from _walk_refs(merged, defs, path, open_defs | {name})
        return
    if node.get("type") == "array":
        items = node.get("items")
        if isinstance(items, dict):
            yield path, "table", aliases
            yield from _walk_refs(items, defs, f"{path}[]", open_defs)
        return

    properties = {
        key: value for key, value in (node.get("properties") or {}).items()
        if not key.startswith("fideon:")
    }
    yield path, "section" if properties else "leaf", aliases
    for key, value in properties.items():
        yield from _walk_refs(value, defs, f"{path}.{key}", open_defs)


def _is_value(definition: dict[str, Any]) -> bool:
    """A value has ``raw``, ``parsed`` and ``page_ref``: ``FieldValue`` and every
    typed value alike, the test :mod:`common.model_view` reads them by."""
    return {"raw", "parsed", "page_ref"} <= set(definition.get("properties") or {})


def collect_schemas() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Every canonical field, keyed by the field itself rather than by location.

    A SPEC_21 line overlay is read through its bundle (:func:`walk_bundle`) and
    versioned as one; every other file is read as it is (:func:`walk`).
    """
    from common.schemas import BUNDLE_VERSION_KEY

    fields: dict[str, dict[str, Any]] = {}
    versions: dict[str, str] = {}

    for path in sorted(CANONICAL.glob("*.json")):
        bundle = common_model_bundle(path.stem)
        found: Iterable[tuple[str, str, list[str]]]
        if bundle is not None:
            # The overlay's version and the common model's together. An overlay
            # has no per-line block to strip: its fields are the common model's.
            versions[path.stem] = str(bundle.get(BUNDLE_VERSION_KEY, "unknown"))
            line_block = None
            found = walk_bundle(bundle)
        else:
            schema = json.loads(path.read_text(encoding="utf-8"))
            source = schema.get("fideon:source") or {}
            versions[path.stem] = str(source.get("version", "unknown"))
            line_block = source.get("line_specific_block")
            found = walk(schema)

        for field_path, kind, aliases in found:
            unified = unify(field_path, line_block)
            name = field_name(unified)
            if not name:
                continue
            entry = fields.setdefault(
                name,
                {"aliases": set(), "paths": set(), "kinds": set(), "schemas": set(),
                 "seeded": False},
            )
            entry["aliases"].update(a.strip() for a in aliases if a and a.strip())
            entry["paths"].add(unified)
            entry["kinds"].add(kind)
            entry["schemas"].add(path.stem)

    for name, aliases in SEED_ALIASES.items():
        entry = fields.setdefault(
            name,
            {"aliases": set(), "paths": set(), "kinds": {"leaf"}, "schemas": set(),
             "seeded": False},
        )
        entry["aliases"].update(aliases)
        entry["seeded"] = True
    return fields, versions


def read_documents() -> list[str]:
    """The normalised text of every original document.

    Only the text is kept. Counting how many documents print a label is evidence;
    recording WHICH ones would put client document paths into a checked-in file
    for no gain — the count is what a reader acts on.
    """
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - optional
        raise SystemExit(
            "PyMuPDF is needed to mine the documents. Install it, or pass --no-documents "
            f"to build the registry from the schemas alone ({exc})."
        ) from exc

    documents: list[str] = []
    pdfs = sorted(DOCUMENTS.rglob("*.pdf"))
    for index, pdf in enumerate(pdfs, start=1):
        try:
            with pymupdf.open(pdf) as doc:
                text = " ".join(page.get_text() for page in doc)
        except Exception as exc:  # noqa: BLE001 - one unreadable PDF must not stop the mine
            print(f"  skipped {pdf.name}: {exc}", file=sys.stderr)
            continue
        documents.append(normalise(text))
        if index % 100 == 0:
            print(f"  read {index}/{len(pdfs)} documents", file=sys.stderr)
    return documents


def count_in(documents: list[str], phrases: set[str]) -> dict[str, int]:
    """How many documents print each phrase."""
    counts: dict[str, int] = {}
    for phrase in sorted(phrases):
        if len(phrase) < MIN_EVIDENCE_LENGTH:
            continue
        needle = normalise(phrase)
        seen = sum(1 for text in documents if needle in text)
        if seen:
            counts[phrase] = seen
    return counts


def render(documents: list[str] | None) -> dict[str, Any]:
    fields, versions = collect_schemas()

    # A field's own name is a candidate label; the documents decide. Added before
    # evidence is attached, so a kept candidate carries its count like any other.
    candidates = {
        name: label for name in fields if (label := own_label(name))
        and normalise(label) not in {normalise(a) for a in fields[name]["aliases"]}
    }
    evidence: dict[str, int] = {}
    if documents:
        wanted = {a for entry in fields.values() for a in entry["aliases"]}
        evidence = count_in(documents, wanted | set(candidates.values()))
        for name, label in candidates.items():
            if label in evidence:
                fields[name]["aliases"].add(label)

    for legacy, canonical in LEGACY_KEYS.items():
        if canonical in fields:
            fields[legacy] = {**fields[canonical], "alias_of": canonical}

    registry: dict[str, Any] = {
        "$comment": (
            "DERIVED by scripts/derive_aliases_from_canonical.py from the canonical schemas "
            "and the original documents - do not hand-edit; fix the source and regenerate. "
            "ONE ENTRY PER CANONICAL FIELD, not per location: the schemas mention "
            "effective_date in ten places, and ten copies of one field disagreed about which "
            "labels were its own. `occurs_at` says where the field is used. Aliases are unified "
            "across lines of business. `documents_per_alias` is HOW MANY original documents "
            "print that alias - not which, since that would put client document paths in the "
            "repo. An alias with none is a phrasing this book does not use, or a mistake in the "
            "schema. Confusables are hand-written, because a schema cannot state that two labels "
            "look alike and mean different things. NEVER rendered into a prompt and NEVER "
            "consulted at inference (master 1.4)."
        ),
        "$doc_type": "policy",
        "$source": {
            "kind": "canonical_schema+documents",
            "keyed_by": "canonical field name",
            "schema_versions": versions,
            "documents_scanned": len(documents or []),
            "field_count": len(fields),
        },
    }

    for name in sorted(fields):
        entry = fields[name]
        aliases = sorted(entry["aliases"])
        attested = {a: evidence[a] for a in aliases if a in evidence}
        paths = sorted(entry["paths"])

        rendered: dict[str, Any] = {
            "aliases": aliases,
            "confusables": (
                CONFUSABLES.get(name)
                or CONFUSABLES.get(str(entry.get("alias_of") or ""), [])
            ),
            "hand_seeded": bool(entry.get("seeded")),
            "kinds": sorted(entry["kinds"]),
            # Where this field is used. `premium` sits in seventy-four places;
            # one entry with the list is the point of keying by field.
            "occurs_at": paths,
            "schemas": sorted(entry["schemas"]),
        }
        if documents:
            rendered["evidence"] = {
                "attested_aliases": len(attested),
                "unattested_aliases": [
                    a for a in aliases
                    if a not in attested and len(a) >= MIN_EVIDENCE_LENGTH
                ],
                "documents_per_alias": {
                    a: attested[a] for a in sorted(attested, key=lambda x: -attested[x])
                },
            }
        if entry.get("alias_of"):
            rendered["canonical_field"] = entry["alias_of"]
        registry[name] = rendered
    return registry


def report(registry: dict[str, Any]) -> None:
    """What a reader should know before trusting this file."""
    fields = {k: v for k, v in registry.items() if not k.startswith("$")}
    aliases = {a for v in fields.values() for a in v["aliases"]}
    with_aliases = sum(1 for v in fields.values() if v["aliases"])
    attested = {a for v in fields.values() for a in (v.get("evidence", {}).get("documents_per_alias") or {})}
    scanned = registry["$source"]["documents_scanned"]

    print(
        f"{len(fields)} field(s), {with_aliases} with at least one alias, "
        f"{len(aliases)} distinct alias(es)"
    )
    if not scanned:
        print("documents not mined; aliases carry no evidence")
        return

    unattested = sorted(a for a in aliases - attested if len(a) >= MIN_EVIDENCE_LENGTH)
    print(f"{len(attested)} attested in {scanned} document(s), {len(unattested)} unattested")
    if unattested:
        print("\nUnattested - no original document prints these. Either this book does not use")
        print("the phrasing, or the schema is wrong. First 20:")
        for alias in unattested[:20]:
            print(f"  {alias!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive the policy alias registry")
    parser.add_argument("--write", action="store_true", help="write the file rather than checking")
    parser.add_argument("--no-documents", action="store_true",
                        help="skip document mining; aliases carry no evidence")
    args = parser.parse_args(argv)

    documents = None if args.no_documents else read_documents()
    registry = render(documents)
    body = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"

    if args.write:
        TARGET.write_text(body, encoding="utf-8")
        print(f"wrote {TARGET.relative_to(ROOT)}")
        report(registry)
        return 0

    if TARGET.exists() and TARGET.read_text(encoding="utf-8") == body:
        print(f"{TARGET.name} is up to date")
        return 0
    print(f"{TARGET.name} is stale; regenerate with --write", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
