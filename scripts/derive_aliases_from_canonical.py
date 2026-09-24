"""Build ``schemas/aliases/policy.aliases.json`` from the canonical schemas and
the original documents.

Two sources, doing different jobs:

* **The canonical schemas** (``configs/canonical schema/policy_check/``) carry
  ``fideon:aliases`` — the labels forms print for each field. They say what to
  look for.
* **The original documents** (``training data/original data/``) say which of
  those labels real pages actually print, and how often. An alias with no
  document evidence is either a phrasing this book of business does not use, or
  a mistake in the schema; either way a reader should be able to tell it apart
  from one seen on sixty pages.

**Aliases are UNIFIED across lines of business.** The canonical schemas nest
line-specific fields under a per-LOB block (``homeowners.dwelling``,
``auto.vehicles``), which would split one field into thirty-four. The block
prefix is stripped, so ``discounts[].discount_name`` is one entry carrying every
phrasing any line prints for it. A label means the same thing whichever policy
prints it, and splitting them would mean the same discount name had to be
learned once per line.

**What this registry is, and is not.** It records which printed phrasings map to
which canonical field, so corpus coverage can be measured and evaluation can
report per alias. It is **never rendered into a prompt and never consulted at
inference** (master §1.4, enforced by ``tests/test_no_runtime_aliases.py``): an
alias table at inference is a lookup pretending to be comprehension, and it fails
silently on the first phrasing nobody listed.

**It never invents an alias.** Every string here appears in a canonical schema or
in the hand-written seed below. Document mining only counts what is already
listed — adding a discovered string automatically would be the registry learning
from a guess.

Run after a schema change, or when documents are added::

    python -m scripts.derive_aliases_from_canonical --write
    python -m scripts.derive_aliases_from_canonical --write --no-documents   # skip mining
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CANONICAL = ROOT / "configs" / "canonical schema" / "policy_check"
DOCUMENTS = ROOT / "training data" / "original data"
TARGET = ROOT / "schemas" / "aliases" / "policy.aliases.json"

#: An alias shorter than this matches inside other words, so counting it would
#: report noise as evidence ("To", "From"). Such aliases stay in the registry;
#: only their evidence is skipped.
MIN_EVIDENCE_LENGTH = 4

#: Hand-written aliases, kept because the canonical schemas carry aliases for the
#: SPECIFIC fields (prior_policy_number, certificate_number) and none at all for
#: the obvious ones — policy_number, carrier.company_name and
#: producer.agency_name have no ``fideon:aliases`` in any schema. Dropping these
#: would leave the fields every document prints with no phrasings recorded.
SEED_ALIASES: dict[str, list[str]] = {
    "named_insured.primary_name": [
        "Insured Name", "Named Insured", "Insured", "Applicant",
        "Name of Applicant", "Applicant Name", "Name of Insured", "Name Insured",
    ],
    "policy.policy_number": [
        "Policy Number", "Policy No.", "Policy #", "Policy No", "Pol. No.",
    ],
    "carrier.company_name": [
        "Carrier", "Insurer", "Insurance Company", "Underwritten By", "Company",
    ],
    "producer.agency_name": [
        "Producer", "Agency", "Broker", "Agent", "Producer Name",
    ],
    "policy.effective_date": [
        "Effective Date", "Effective", "Policy Effective Date", "Inception Date", "From",
    ],
    "policy.expiration_date": [
        "Expiration Date", "Expiration", "Policy Expiration Date", "Expiry Date", "To",
    ],
    "premium.total_policy_premium": [
        "Total Premium", "Policy Premium", "Total Policy Premium", "Premium",
    ],
}

#: Hand-written, because a schema cannot state that two labels look alike and
#: mean different things — the distinction the misattribution metric scores.
CONFUSABLES: dict[str, list[str]] = {
    "named_insured.primary_name": [
        "Certificate Holder", "Producer", "Agency", "Additional Insured",
        "Loss Payee", "Mortgagee", "Carrier", "Insurer",
    ],
    "policy.policy_number": [
        "Quote Number", "Binder Number", "Claim Number", "Submission Number",
        "Certificate Number",
    ],
    "carrier.company_name": [
        "Producer", "Agency", "Broker", "Named Insured", "Administrator",
    ],
    "producer.agency_name": [
        "Carrier", "Insurer", "Named Insured", "Underwriter",
    ],
    "policy.effective_date": [
        "Issue Date", "Date Printed", "Expiration Date", "Bind Date",
    ],
    "policy.expiration_date": [
        "Effective Date", "Cancellation Date", "Renewal Date",
    ],
    "premium.total_policy_premium": [
        "Deposit Premium", "Minimum Premium", "Estimated Premium",
        "Premium by Coverage", "Taxes and Fees",
    ],
}

#: The repo's own policy schema still names these fields flatly, and three
#: consumers look them up that way: the confusable co-occurrence count
#: (corpus_manifest), the provenance check that refuses a label naming a
#: confusable (export_golden_labels), and the review tool's "NEVER take from"
#: instruction. Until the canonical schemas ARE the repo's schemas, both keyings
#: must be present — a registry serving only the future one silently empties all
#: three, which the test suite catches.
LEGACY_KEYS: dict[str, str] = {
    "insured_name": "named_insured.primary_name",
    "policy_number": "policy.policy_number",
    "carrier": "carrier.company_name",
    "producer": "producer.agency_name",
    "effective_date": "policy.effective_date",
    "expiration_date": "policy.expiration_date",
    "total_premium": "premium.total_policy_premium",
}

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Case- and whitespace-insensitive form, for matching a label on a page."""
    return _WS.sub(" ", text.replace(" ", " ")).strip().casefold()


def walk(node: Any, path: str = "") -> Iterator[tuple[str, list[str]]]:
    """Yield ``(field_path, aliases)`` for every node carrying ``fideon:aliases``."""
    if not isinstance(node, dict):
        return
    aliases = node.get("fideon:aliases")
    if aliases:
        yield path, list(aliases)
    for key, value in (node.get("properties") or {}).items():
        yield from walk(value, f"{path}.{key}" if path else key)
    items = node.get("items")
    if isinstance(items, dict):
        yield from walk(items, f"{path}[]")


def unify(field_path: str, line_block: str | None) -> str:
    """Strip the per-LOB block, so one field is one entry across every line."""
    if line_block and field_path.startswith(f"{line_block}."):
        return field_path[len(line_block) + 1:]
    return field_path


def collect_schemas() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Every aliased field across every canonical schema, unified by field path."""
    fields: dict[str, dict[str, Any]] = {}
    versions: dict[str, str] = {}

    for path in sorted(CANONICAL.glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        source = schema.get("fideon:source") or {}
        versions[path.stem] = str(source.get("version", "unknown"))
        line_block = source.get("line_specific_block")

        for field_path, aliases in walk(schema):
            unified = unify(field_path, line_block)
            entry = fields.setdefault(
                unified, {"aliases": set(), "schemas": set(), "seeded": False}
            )
            entry["aliases"].update(a.strip() for a in aliases if a and a.strip())
            entry["schemas"].add(path.stem)

    for field_path, aliases in SEED_ALIASES.items():
        entry = fields.setdefault(field_path, {"aliases": set(), "schemas": set(), "seeded": False})
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


def mine(fields: dict[str, dict[str, Any]], documents: list[str]) -> dict[str, int]:
    """How many original documents print each alias."""
    evidence: dict[str, int] = {}
    wanted = sorted({a for entry in fields.values() for a in entry["aliases"]})

    for alias in wanted:
        if len(alias) < MIN_EVIDENCE_LENGTH:
            continue
        needle = normalise(alias)
        seen = sum(1 for text in documents if needle in text)
        if seen:
            evidence[alias] = seen
    return evidence


def render(documents: list[str] | None) -> dict[str, Any]:
    fields, versions = collect_schemas()
    evidence = mine(fields, documents) if documents else {}

    for legacy, canonical in LEGACY_KEYS.items():
        if canonical in fields:
            fields[legacy] = {**fields[canonical], "alias_of": canonical}

    registry: dict[str, Any] = {
        "$comment": (
            "DERIVED by scripts/derive_aliases_from_canonical.py from the canonical schemas "
            "and the original documents - do not hand-edit; fix the source and regenerate. "
            "Aliases are UNIFIED across lines of business: the per-LOB block is stripped, so a "
            "field is one entry carrying every phrasing any line prints for it. "
            "`documents_per_alias` is HOW MANY original documents print that alias - not which, "
            "since that would put client document paths in the repo. An alias with none is a "
            "phrasing this book does not use, or a mistake in the schema. Confusables are hand-written, because "
            "a schema cannot state that two labels look alike and mean different things. "
            "NEVER rendered into a prompt and NEVER consulted at inference (master 1.4)."
        ),
        "$doc_type": "policy",
        "$source": {
            "kind": "canonical_schema+documents",
            "schema_versions": versions,
            "documents_scanned": len(documents or []),
            "field_count": len(fields),
        },
    }

    for field_path in sorted(fields):
        entry = fields[field_path]
        aliases = sorted(entry["aliases"])
        attested = {a: evidence[a] for a in aliases if a in evidence}

        rendered: dict[str, Any] = {
            "aliases": aliases,
            "confusables": (
                CONFUSABLES.get(field_path)
                or CONFUSABLES.get(str(entry.get("alias_of") or ""), [])
            ),
            "hand_seeded": bool(entry.get("seeded")),
            # Which canonical schemas declare this field. One schema means a
            # line-specific value; thirty means a header field, and the
            # difference matters when reading a coverage report.
            "schemas": sorted(entry["schemas"]),
        }
        if documents:
            rendered["evidence"] = {
                "attested_aliases": len(attested),
                "unattested_aliases": [
                    a for a in aliases
                    if a not in attested and len(a) >= MIN_EVIDENCE_LENGTH
                ],
                # Counts only: how many documents print this label. Which ones is
                # not recorded — it would put client document paths in the repo.
                "documents_per_alias": {
                    a: attested[a] for a in sorted(attested, key=lambda x: -attested[x])
                },
            }
        if entry.get("alias_of"):
            rendered["canonical_path"] = entry["alias_of"]
        registry[field_path] = rendered
    return registry


def report(registry: dict[str, Any]) -> None:
    """What a reader should know before trusting this file."""
    fields = {k: v for k, v in registry.items() if not k.startswith("$")}
    aliases = {a for v in fields.values() for a in v["aliases"]}
    attested = {
        a for v in fields.values()
        for a in (v.get("evidence", {}).get("documents_per_alias") or {})
    }
    scanned = registry["$source"]["documents_scanned"]

    if not scanned:
        print(f"{len(fields)} field(s), {len(aliases)} alias(es); documents not mined")
        return

    unattested = sorted(a for a in aliases - attested if len(a) >= MIN_EVIDENCE_LENGTH)
    print(
        f"{len(fields)} field(s), {len(aliases)} alias(es); "
        f"{len(attested)} attested in {scanned} document(s), {len(unattested)} unattested"
    )
    if unattested:
        print("\nUnattested - no original document prints these. Either this book does not use")
        print("the phrasing, or the schema is wrong. First 25:")
        for alias in unattested[:25]:
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
