"""Build ``schemas/aliases/policy.aliases.json`` from the canonical LOB schemas.

The canonical schemas under ``configs/canonical schema/policy_check/`` carry
``fideon:aliases`` on their fields — the labels real forms print for each value,
collected from actual documents. That is far better seed material than a
hand-written list, so this derives the registry rather than anyone maintaining
it by hand.

**What this registry is, and is not.** It records which printed phrasings map to
which canonical field, so corpus coverage can be measured and evaluation can
report per alias. It is **never rendered into a prompt and never consulted at
inference** (master §1.4, enforced by ``tests/test_no_runtime_aliases.py``): an
alias table at inference is a lookup pretending to be comprehension, and it fails
silently on the first phrasing nobody listed.

**Three things it deliberately does not do.**

* It does not invent aliases. Every string here appears in a canonical schema.
* It does not derive **confusables** — "Certificate Holder looks like Named
  Insured but is a different party" is judgement, not something a schema states.
  The hand-written ones are carried across to the canonical path they belong to
  and the rest are left empty, to be filled by whoever knows the answer.
* It does not carry **evidence**. Document counts and example source_ids come
  from ``data_pipeline/labeling/derive_aliases.py`` once labeled documents exist
  (SPEC_04 §3). A schema-derived alias says "a form prints this somewhere", not
  "N documents in this corpus print it".

Run after any schema change::

    python -m scripts.derive_aliases_from_canonical --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CANONICAL = ROOT / "configs" / "canonical schema" / "policy_check"
TARGET = ROOT / "schemas" / "aliases" / "policy.aliases.json"

#: The LOBs whose schemas are derived from. Personal lines is the family being
#: trained; the others join when their adapter does, so the registry describes
#: what the corpus actually contains rather than every schema on disk.
LOBS: tuple[str, ...] = (
    "homeowners",
    "personal_auto",
    "dwelling_fire",
    "ocean_marine",
    "classic_auto",
    "motorcycle",
    "recreational_vehicle",
    "personal_umbrella",
)

#: Hand-written aliases from the seed registry, re-keyed onto their canonical
#: paths. Kept because the canonical schemas carry aliases for the SPECIFIC
#: fields (prior_policy_number, certificate_number) and none at all for the
#: obvious ones — policy_number, carrier.company_name, producer.agency_name have
#: no `fideon:aliases` in any of the eight. Dropping these would leave the three
#: fields every document prints with no phrasings recorded at all.
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

#: Hand-written confusables from the seed registry, re-keyed onto the canonical
#: path each one belongs to. These are the pairs that look alike on a page and
#: mean different things — the distinction the misattribution metric scores, and
#: the one thing in the old file worth keeping.
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
#: have to be present — a registry that serves only the future one silently
#: empties all three.
LEGACY_KEYS: dict[str, str] = {
    "insured_name": "named_insured.primary_name",
    "policy_number": "policy.policy_number",
    "carrier": "carrier.company_name",
    "producer": "producer.agency_name",
    "effective_date": "policy.effective_date",
    "expiration_date": "policy.expiration_date",
    "total_premium": "premium.total_policy_premium",
}


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


def collect() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Every aliased field across the LOB schemas, and the schema versions used."""
    fields: dict[str, dict[str, Any]] = {}
    versions: dict[str, str] = {}

    for lob in LOBS:
        path = CANONICAL / f"{lob}.json"
        if not path.exists():
            raise SystemExit(f"no canonical schema for {lob} at {path}")
        schema = json.loads(path.read_text(encoding="utf-8"))
        versions[lob] = str((schema.get("fideon:source") or {}).get("version", "unknown"))

        for field_path, aliases in walk(schema):
            entry = fields.setdefault(field_path, {"aliases": set(), "lobs": set(), "seeded": False})
            entry["aliases"].update(a.strip() for a in aliases if a and a.strip())
            entry["lobs"].add(lob)

    # The header fields every document prints, which the schemas leave unaliased.
    for field_path, aliases in SEED_ALIASES.items():
        entry = fields.setdefault(
            field_path, {"aliases": set(), "lobs": set(LOBS), "seeded": False}
        )
        entry["aliases"].update(aliases)
        entry["seeded"] = True
    return fields, versions


def render() -> dict[str, Any]:
    fields, versions = collect()

    registry: dict[str, Any] = {
        "$comment": (
            "DERIVED from the canonical LOB schemas by "
            "scripts/derive_aliases_from_canonical.py — do not hand-edit; fix the schema and "
            "regenerate. Aliases are the labels real forms print for a field. Confusables are "
            "hand-written, because a schema cannot state that two labels look alike and mean "
            "different things. NO document evidence: these say a form prints this somewhere, "
            "not that N documents in this corpus do — that comes from "
            "data_pipeline/labeling/derive_aliases.py once labeled documents exist (SPEC_04 §3). "
            "NEVER rendered into a prompt and NEVER consulted at inference (master §1.4)."
        ),
        "$doc_type": "policy",
        "$source": {
            "kind": "canonical_schema",
            "lobs": {lob: versions[lob] for lob in LOBS},
            "field_count": len(fields),
        },
    }

    # Both keyings, deliberately. The flat entry is what today's schema and its
    # three consumers read; the canonical path is what they will read once the
    # canonical schemas are registered. They carry the same strings, so there is
    # one source of truth and two ways in.
    for legacy, canonical in LEGACY_KEYS.items():
        if canonical in fields:
            fields[legacy] = {
                **fields[canonical],
                "alias_of": canonical,
            }

    for field_path in sorted(fields):
        entry = fields[field_path]
        registry[field_path] = {
            "aliases": sorted(entry["aliases"]),
            # Resolved through the canonical path for a legacy entry, or the
            # flat copy would carry aliases and no confusables — which reads as
            # "this field has no look-alikes" rather than "look them up over
            # there", and the misattribution check would quietly score nothing.
            "confusables": (
                CONFUSABLES.get(field_path)
                or CONFUSABLES.get(str(entry.get("alias_of") or ""), [])
            ),
            # Hand-written rather than schema-derived, so a later reader knows
            # which entries a schema regeneration will NOT refresh.
            "hand_seeded": bool(entry.get("seeded")),
            # Which lines print this field. A field carried by one LOB is a
            # line-specific value; one carried by all eight is a header field,
            # and the distinction matters when reading a coverage report.
            "lobs": sorted(entry["lobs"]),
        }
        if entry.get("alias_of"):
            # This entry exists under the repo's flat field name; the canonical
            # path is where it will live once the schemas are adopted.
            registry[field_path]["canonical_path"] = entry["alias_of"]

    missing = sorted(set(CONFUSABLES) - set(fields))
    if missing:
        # A confusable list keyed to a path no schema has would never be read,
        # which is worse than not having it: it reads as a guard that is in place.
        print(f"WARNING: confusables for paths no schema carries: {missing}", file=sys.stderr)
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive the policy alias registry")
    parser.add_argument("--write", action="store_true", help="write the file rather than checking")
    args = parser.parse_args(argv)

    registry = render()
    body = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"

    if args.write:
        TARGET.write_text(body, encoding="utf-8")
        print(
            f"wrote {TARGET.relative_to(ROOT)}: {registry['$source']['field_count']} field(s), "
            f"{sum(len(v['aliases']) for k, v in registry.items() if not k.startswith('$'))} alias(es)"
        )
        return 0

    if TARGET.exists() and TARGET.read_text(encoding="utf-8") == body:
        print(f"{TARGET.name} is up to date")
        return 0
    print(f"{TARGET.name} is stale; regenerate with --write", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
