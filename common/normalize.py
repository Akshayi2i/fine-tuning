"""Field-value normalizers — one definition of "matches", shared by three specs.

Comparison must reflect *correctness*, not formatting. Insurance fields need this
or you systematically under-count correct extractions: the golden label says
``2026-04-01`` and the page says ``04/01/2026``; the label says ``12400.0`` and
the page says ``$12,400.00``.

Built in SPEC_01 rather than SPEC_08 because three consumers need it and the
earliest is SPEC_04:

* **SPEC_04** ``derive_aliases`` — anchoring a golden value to its position in the
  OCR text, which is how the surface label is discovered at all.
* **SPEC_08** the promotion gate — what counts as a correct field.
* **SPEC_12** the testing harness — must agree with the gate, or "correct" means
  two different things in test and in promotion.

Pure and dependency-free by design.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any

__all__ = [
    "normalize_text",
    "normalize_date",
    "normalize_currency",
    "normalize_identifier",
    "normalize_entity_name",
    "normalize_value",
    "values_match",
]

# Legal-form suffixes, stripped when comparing organisation names so
# "Acme Mfg LLC" and "ACME MANUFACTURING LLC" can match.
_ORG_SUFFIXES = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp",
    "corporation", "co", "company", "plc", "pllc", "pc", "pa", "gmbh", "sa", "nv", "bv",
}

# Common abbreviations in organisation names.
_ORG_ABBREV = {
    "mfg": "manufacturing", "mfrs": "manufacturers", "intl": "international",
    "natl": "national", "assoc": "associates", "assn": "association",
    "bros": "brothers", "svcs": "services", "svc": "service", "sys": "systems",
    "tech": "technologies", "grp": "group", "ent": "enterprises",
    "constr": "construction", "dev": "development", "mgmt": "management",
    "prop": "properties", "ins": "insurance", "&": "and",
}

_DATE_FORMATS = (
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y",
    "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y%m%d",
)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def _base(text: str) -> str:
    """Unicode-normalise, collapse whitespace, casefold."""
    text = unicodedata.normalize("NFKC", str(text))
    return _WS.sub(" ", text).strip().casefold()


def normalize_text(value: Any) -> str | None:
    """Generic text normalisation: NFKC, collapsed whitespace, casefolded."""
    if value is None:
        return None
    return _base(value) or None


def normalize_date(value: Any) -> str | None:
    """Parse a date in any common layout to ``YYYY-MM-DD``.

    Returns ``None`` when the value is not a recognisable date — callers must
    treat that as "not comparable as a date", not as a match.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raw = _WS.sub(" ", str(value)).strip()
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
        # Two-digit years: assume the 2000s, which is right for policy dates.
        if parsed.year < 100:
            parsed = parsed.replace(year=parsed.year + 2000)
        return parsed.isoformat()
    return None


def normalize_currency(value: Any) -> float | None:
    """Parse a monetary or numeric value to a float.

    Handles currency symbols, thousands separators, and accounting-style
    parenthesised negatives — ``(1,200.00)`` is ``-1200.0``, which matters on
    loss runs where recoveries appear that way.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    if negative:
        raw = raw[1:-1]
    raw = re.sub(r"[^\d.\-]", "", raw)
    if raw in ("", "-", ".", "-."):
        return None
    try:
        amount = float(raw)
    except ValueError:
        return None
    return -amount if negative else amount


def normalize_identifier(value: Any) -> str | None:
    """Normalise a policy/claim/certificate number.

    Separators and case are presentation, not identity: ``WC-8842317-01``,
    ``wc 8842317 01`` and ``WC884231701`` are the same policy. Note this is
    deliberately lossy — use it for comparison, never for storage.
    """
    if value is None:
        return None
    cleaned = _NON_ALNUM.sub("", str(value)).upper()
    return cleaned or None


def normalize_entity_name(value: Any) -> str | None:
    """Normalise an organisation or person name for comparison.

    Casefolds, drops punctuation, expands common abbreviations, and strips legal
    suffixes, so ``Acme Mfg LLC`` matches ``ACME MANUFACTURING LLC``.

    Kept conservative on purpose: over-aggressive normalisation would collapse
    genuinely different entities, and on these documents the confusable parties
    often share words (``Rivera Fabrication`` vs ``Rivera Fabrication Holdings``).
    """
    if value is None:
        return None
    text = _PUNCT.sub(" ", _base(value))
    tokens = [_ORG_ABBREV.get(t, t) for t in text.split()]
    while tokens and tokens[-1] in _ORG_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


#: Field-name suffixes mapped to the normalizer they should use.
_FIELD_KIND_SUFFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("_date", "date"), "date"),
    (("_number", "_no", "_num", "number"), "identifier"),
    (("premium", "limit", "deductible", "paid", "reserved", "incurred",
      "amount", "revenue", "footage"), "currency"),
    (("_name", "name", "carrier", "producer", "insured", "holder"), "entity"),
)


def infer_field_kind(field_path: str) -> str:
    """Guess which normalizer a field wants from its canonical name.

    Deliberately name-based: the alternative is a hand-maintained per-field map
    that silently goes stale when the schema gains a field.
    """
    leaf = field_path.rsplit(".", 1)[-1].rstrip("[]").casefold()
    for suffixes, kind in _FIELD_KIND_SUFFIXES:
        if any(leaf.endswith(s) or leaf == s for s in suffixes):
            return kind
    return "text"


def normalize_value(value: Any, field_path: str | None = None, kind: str | None = None) -> Any:
    """Normalise a value using the normalizer appropriate to its field."""
    resolved = kind or (infer_field_kind(field_path) if field_path else "text")
    if resolved == "date":
        return normalize_date(value)
    if resolved == "currency":
        return normalize_currency(value)
    if resolved == "identifier":
        return normalize_identifier(value)
    if resolved == "entity":
        return normalize_entity_name(value)
    return normalize_text(value)


def values_match(
    expected: Any,
    actual: Any,
    field_path: str | None = None,
    kind: str | None = None,
) -> bool:
    """Whether two values match after normalisation.

    Both ``None`` counts as a match: a correctly-absent field is a correct
    extraction, and scoring it otherwise would penalise the null-handling the
    corpus explicitly teaches.
    """
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    a = normalize_value(expected, field_path, kind)
    b = normalize_value(actual, field_path, kind)
    if a is None or b is None:
        # Unparseable under the chosen normalizer — fall back to text equality
        # rather than silently declaring a match.
        return normalize_text(expected) == normalize_text(actual)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) < 0.005
    return a == b
