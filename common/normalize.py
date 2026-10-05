"""Field-value normalizers — one definition of "matches", shared by three specs.

Comparison must reflect *correctness*, not formatting. Insurance fields need this
or you systematically under-count correct extractions: the golden label says
``2026-04-01`` and the page says ``04/01/2026``; the label says ``12400.0`` and
the page says ``$12,400.00``.

Built in IMPL-01 rather than IMPL-08 because three consumers need it and the
earliest is IMPL-04:

* **IMPL-04** ``derive_aliases`` — anchoring a golden value to its position in the
  OCR text, which is how the surface label is discovered at all.
* **IMPL-08** the promotion gate — what counts as a correct field.
* **IMPL-12** the testing harness — must agree with the gate, or "correct" means
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
    "format_output_date",
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
    "prop": "properties", "ins": "insurance",
    # NOTE: "&" is handled in normalize_entity_name before punctuation is
    # stripped. It cannot live here — by the time these tokens are consulted,
    # _PUNCT has already removed it.
}

#: How far ahead a two-digit year may land. `27` is 2027 - an expiration, a
#: renewal - but `85` is 1985: a date of birth, a date licensed, a year built.
#: Reading every two-digit year as this century wrote `07/04/85` as 2085, into
#: training targets and served output alike. Ten years covers policy terms and
#: the dates printed about them; further out, the last century is meant.
_TWO_DIGIT_YEAR_HORIZON = 10

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
        if "%y" in fmt:
            # Not Python's own pivot (69 -> 1969, 68 -> 2068): this century
            # unless that is beyond the horizon, then the last one.
            year = 2000 + parsed.year % 100
            if year > date.today().year + _TWO_DIGIT_YEAR_HORIZON:
                year -= 100
            try:
                parsed = parsed.replace(year=year)
            except ValueError:  # 29 February in a year that has none
                continue
        return parsed.isoformat()
    return None


#: How every date leaves the model and the pipeline: `MM/DD/YYYY`. One
#: definition, read by the prompt, the training targets and the serving
#: post-process, because a target written one way and a request told another is
#: a model trained against its own instructions.
OUTPUT_DATE_LABEL = "MM/DD/YYYY"
OUTPUT_DATE_FORMAT = "%m/%d/%Y"


def format_output_date(value: Any) -> str | None:
    """Render a date in the output format, ``MM/DD/YYYY``.

    Accepts anything :func:`normalize_date` can parse. Returns ``None`` when the
    value is not a recognisable date, so a caller keeps the original rather than
    replacing a value it could not read with nothing.

    Comparison is unaffected: :func:`normalize_date` still reduces both sides to
    ISO before they are compared or ordered, and it reads ``MM/DD/YYYY`` back.
    Output format and comparison format are deliberately separate things.
    """
    iso = normalize_date(value)
    if iso is None:
        return None
    return datetime.strptime(iso, "%Y-%m-%d").strftime(OUTPUT_DATE_FORMAT)


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
    # "&" is expanded first, because _PUNCT strips it: the mapping in
    # _ORG_ABBREV could never fire, so `Smith & Jones Inc` did not match
    # `Smith and Jones Inc` — a very common pair on these documents, and one
    # that under-counted a gating metric every time it appeared.
    text = _PUNCT.sub(" ", _base(value).replace("&", " and "))
    tokens = [_ORG_ABBREV.get(t, t) for t in text.split()]
    while tokens and tokens[-1] in _ORG_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


#: Words that name what kind of insurer a company is, not which one. Trailing
#: runs of them are dropped by :func:`normalize_carrier`.
_CARRIER_TRAILING = frozenset({
    "insurance", "ins", "company", "co", "companies", "indemnity", "casualty",
    "surety", "assurance", "underwriters", "group", "mutual", "fire", "and", "of",
    "america", "corp", "corporation", "exchange",
})


def normalize_carrier(value: Any) -> str | None:
    """One key per insurer, for the held-out-carrier split and family grouping.

    ``The Travelers Indemnity Company``, ``Travelers Casualty and Surety Company
    of America`` and ``TRAVELERS`` are one carrier. Compared as written they were
    three, so holding one out of train left the other two in it — the carrier's
    templates leaked across the split the hold-out exists to keep clean — and
    each variant looked like a small carrier, skewing which ones were chosen.

    Conservative past that: only a TRAILING run of insurer words goes, so
    ``Great American`` and ``Great Northern`` stay apart.
    """
    tokens = (normalize_entity_name(value) or "").split()
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in _CARRIER_TRAILING:
        tokens.pop()
    return " ".join(tokens) or None


#: Field-name suffixes mapped to the normalizer they should use.
_FIELD_KIND_SUFFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    # Dates are matched by word in infer_field_kind: a suffix "date" also
    # matched `candidate` and `update`.
    (("_number", "_no", "_num", "number"), "identifier"),
    (("premium", "limit", "deductible", "paid", "reserved", "incurred",
      "amount", "revenue", "footage"), "currency"),
    (("_name", "name", "carrier", "producer", "insured", "holder"), "entity"),
)


_DATE_WORDS = frozenset({"date", "dates", "dated", "dob"})
#: A list index anywhere in a path segment: `report_due_dates[0]`. The old
#: `rstrip("[]")` stripped brackets but not the digit between them, so an
#: element of a list of dates was never recognised as one.
_INDEX = re.compile(r"\[\d*\]")


def infer_field_kind(field_path: str) -> str:
    """Guess which normalizer a field wants from its canonical name.

    Deliberately name-based: the alternative is a hand-maintained per-field map
    that silently goes stale when the schema gains a field.
    """
    leaf = _INDEX.sub("", field_path.rsplit(".", 1)[-1]).casefold()
    # A date by any word of its name, not by suffix: the canonical schemas name
    # dates `date_of_birth`, `date_licensed`, `replaces_prior_declaration_dated`
    # and `report_due_dates`, and a suffix rule left all of them unformatted —
    # MM/DD/YYYY was a guarantee for most dates and not for these. Whole words,
    # so `update_reason` or `candidate` is not a date.
    if _DATE_WORDS & set(leaf.split("_")):
        return "date"
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

    # A list-valued field is a SET, not a sequence (arch v2.1 §0b —
    # line_of_business, line_of_business_other). Comparing them positionally
    # would score a correct extraction as a miss whenever the model emitted the
    # same lines in another order, and no ordering is printed on the document for
    # it to have got wrong. An empty list and an absent field both mean "this
    # document determines none", so they match.
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        left = expected if isinstance(expected, (list, tuple)) else ([] if expected is None else [expected])
        right = actual if isinstance(actual, (list, tuple)) else ([] if actual is None else [actual])
        if any(isinstance(v, dict) for v in (*left, *right)):
            # A list of objects (claims, coverages) is row-aligned by the
            # evaluation matcher, not compared as a set here.
            return False
        return {normalize_text(v) for v in left} == {normalize_text(v) for v in right}

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
