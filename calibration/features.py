"""Per-field confidence features (arch v2.1 §5.1–5.2).

**Why min-logprob was replaced.** v1 collapsed a field's span to the minimum
token probability, on the reasoning that the weakest link is what makes a field
worth reviewing. That is a good instinct and a broken statistic: the minimum of
*n* draws falls as *n* grows, so a long correct value scores lower than a short
wrong one. ``ABC-1234567-01`` is nine tokens and ``2026`` is one — under
min-logprob the policy number looks less trustworthy than the year, every time,
on every document.

The fix is not a better aggregation. It is to stop pretending one number carries
the signal, and hand a **feature vector** to a calibrator that learns what the
combination means — including the length that made the minimum misleading.

Four logprob features, and five that have nothing to do with logprobs:

* **OCR agreement.** Is the normalized value actually present in the text of the
  page it was attributed to? A model can be confident about something it
  invented; it cannot make the value appear on the page.
* **Rule checks.** Effective date before expiration, totals summing to their
  rows, enum validity, identifier shape. These catch the confident-but-inconsistent
  case, which logprobs by construction cannot see.
* **Null flag.** A ``null`` has no tokens, so it has no logprob signal at all —
  and a false null is the failure mode that looks like ordinary imperfection
  while quietly halving a form's usefulness. Calibrated as its own class.
* **Field type.** Selects the calibrator. An identifier and a free-text
  description miscalibrate differently and should not share a curve.
* **Cross-mode agreement.** Optional, for fields the owner marks high-value:
  does the image-only run agree with the OCR+image one? Disagreement predicts
  error better than either run's own confidence does.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from common.normalize import values_match

log = logging.getLogger(__name__)

#: Field types, each with its own calibrator (arch v2.1 §5.2). Identifiers and
#: money are exact-match and unforgiving; names and addresses are fuzzy-matched;
#: free text is reported but never gated. ``number`` is a count, a year or a
#: percentage: exact-match like money, but not an amount.
FIELD_TYPES = (
    "identifier", "money", "date", "number", "entity", "address", "enum", "free_text",
)

#: The reviewed type of every canonical policy field (proposed by
#: scripts/propose_field_types.py, decided by a person). A field's type sets
#: the error it may carry when accepted without review, so the table outranks
#: the name heuristic, which put 130 money, number and identifier fields into
#: free text.
FIELD_TYPE_TABLE = Path(__file__).resolve().parent.parent / "configs" / "field_types.yaml"

_IDENTIFIER_SHAPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-/ .]{2,}$")


@dataclass
class FieldFeatures:
    """One field's feature vector, ready for its field-type calibrator."""

    field_path: str
    field_type: str
    value: Any

    # --- logprob features (§5.1) -----------------------------------------
    min_logprob: float = 0.0
    mean_logprob: float = 0.0
    first_token_logprob: float = 0.0

    #: Always included. The minimum falls as values get longer even when they are
    #: correct, so without length the calibrator cannot tell a long right answer
    #: from a short wrong one — which is exactly what v1 got wrong.
    token_count: int = 0

    # --- non-logprob features (§5.2) -------------------------------------
    ocr_agreement: float | None = None
    rule_checks_passed: bool | None = None
    is_null: bool = False
    cross_mode_agreement: float | None = None

    #: Set when the span could not be located in the generation. Such a field
    #: gets no confidence at all rather than a default one — a default here is a
    #: number nobody measured, presented as if it were measured.
    mapped: bool = True
    reason: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.mapped

    def vector(self) -> list[float]:
        """The ordered numeric vector the calibrator consumes.

        Missing non-logprob features encode as a **neutral 0.5 plus an explicit
        presence flag**, rather than as 0. Encoding "not checked" as 0 would make
        it indistinguishable from "checked and failed", and the calibrator would
        learn to distrust every field in image-only mode — where OCR agreement
        cannot be computed at all.
        """
        return [
            self.min_logprob,
            self.mean_logprob,
            self.first_token_logprob,
            math.log1p(max(self.token_count, 0)),
            self.ocr_agreement if self.ocr_agreement is not None else 0.5,
            1.0 if self.ocr_agreement is not None else 0.0,
            float(bool(self.rule_checks_passed)) if self.rule_checks_passed is not None else 0.5,
            1.0 if self.rule_checks_passed is not None else 0.0,
            1.0 if self.is_null else 0.0,
            self.cross_mode_agreement if self.cross_mode_agreement is not None else 0.5,
            1.0 if self.cross_mode_agreement is not None else 0.0,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "min_logprob", "mean_logprob", "first_token_logprob", "log_token_count",
            "ocr_agreement", "ocr_agreement_present",
            "rule_checks_passed", "rule_checks_present",
            "is_null",
            "cross_mode_agreement", "cross_mode_present",
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_path": self.field_path,
            "field_type": self.field_type,
            "is_null": self.is_null,
            "mapped": self.mapped,
            "reason": self.reason,
            **dict(zip(self.feature_names(), self.vector(), strict=True)),
        }


@cache
def field_type_table(path: Path = FIELD_TYPE_TABLE) -> dict[str, str]:
    """The reviewed field types, by path without list markers; empty until the
    table has been reviewed and committed. An unknown type is an error, not a
    silent free_text: a typo there would quietly change what is auto-accepted."""
    if not path.exists():
        return {}
    import yaml

    table = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("fields") or {}
    unknown = {f: t for f, t in table.items() if t not in FIELD_TYPES}
    if unknown:
        raise ValueError(f"{path}: unknown field types {unknown}; allowed: {FIELD_TYPES}")
    return {str(f): str(t) for f, t in table.items()}


#: The common model's typed values (SPEC_21) and the calibrator each belongs to.
#: A plain FieldValue says nothing about its kind, so it is left to the name.
VALUE_TYPE_FIELD_TYPES: dict[str, str] = {
    "DateValue": "date",
    "MoneyValue": "money",
    "PercentValue": "number",
    "NumberValue": "number",
    "YearValue": "number",
    "NaicValue": "identifier",
    "PostalCodeValue": "identifier",
    "StateValue": "enum",
    "BoolValue": "enum",
    "TransactionTypeValue": "enum",
    "CoverageTriggerValue": "enum",
    "IncludedValue": "enum",
    "ValuationValue": "enum",
    "AdmittedStatusValue": "enum",
    "DriverStatusValue": "enum",
}


@cache
def common_model_field_types() -> dict[str, str]:
    """Field type by path (list markers removed), for every field the
    common-model lines declare with a typed value. One table for every such
    line: the common model gives a path one type wherever it appears."""
    from common.schemas import is_common_model, resolved_schema, schema_selectors

    out: dict[str, str] = {}
    for doc_type, form, lob in schema_selectors():
        if doc_type == "policy" and lob and is_common_model(doc_type, form, lob):
            view = resolved_schema(doc_type, form, lob)
            _collect_typed(view, view.get("$defs") or {}, "", out, frozenset())
    return out


def _collect_typed(node: Any, defs: dict[str, Any], path: str, out: dict[str, str],
                   seen: frozenset[str]) -> None:
    from common.schemas import resolve_local

    variants = [node, *(node.get("anyOf") or [])] if isinstance(node, dict) else []
    for variant in variants:
        for name, sub in (variant.get("properties") or {}).items():
            here = f"{path}.{name}" if path else name
            ref = sub.get("$ref", "") if isinstance(sub, dict) else ""
            target = ref.rsplit("/", 1)[-1] if ref else ""
            if target in VALUE_TYPE_FIELD_TYPES:
                out.setdefault(here, VALUE_TYPE_FIELD_TYPES[target])
                continue
            if target in seen:
                continue
            resolved = resolve_local(sub, defs)
            if not isinstance(resolved, dict):
                continue
            items = resolve_local(resolved.get("items"), defs) if "items" in resolved else None
            for child in (resolved, items):
                if isinstance(child, dict) and (child.get("properties") or child.get("anyOf")):
                    _collect_typed(child, defs, here, out, seen | ({target} if target else set()))


def infer_field_type(field_path: str, value: Any = None, *, common_model: bool = False) -> str:
    """Which calibrator this field belongs to.

    The reviewed table first (configs/field_types.yaml); then, for a field of a
    ``common_model`` document, the type its schema declares (a MoneyValue is
    money) and an overflow value by what was parsed. Otherwise name-based,
    reusing ``common.normalize``'s inference so a field is *normalised* and
    *calibrated* under the same notion of what it is. Two different answers to
    "what kind of field is this" is how a money field gets compared as text and
    calibrated as a number.

    Only for a common-model document: the self-contained lines and ACORD 140
    share many of its paths (``billing.amount_due``, ``*.address.state``,
    ``buildings.year_built``), and read through its table those fields moved
    to another calibrator and another error target.
    """
    from common.normalize import infer_field_kind

    bare = re.sub(r"\[\d*\]", "", field_path)
    reviewed = field_type_table().get(bare)
    if reviewed:
        return reviewed
    if common_model:
        declared = common_model_field_types().get(bare)
        if declared:
            return declared
        if (bare == "additional_fields.value" and isinstance(value, (int, float))
                and not isinstance(value, bool)):
            return "number"
    kind = infer_field_kind(field_path)
    mapping = {
        "identifier": "identifier",
        "currency": "money",
        "date": "date",
        "entity": "entity",
    }
    if kind in mapping:
        return mapping[kind]
    leaf = field_path.rsplit(".", 1)[-1].casefold()
    # By any path segment, not the leaf alone: a canonical address is split into
    # components (``carrier.address.city``, ``named_insured.mailing_address.line_1``)
    # whose leaf never says "address", so they all fell into free_text — which
    # has no error target and is never auto-accepted.
    if any("address" in part for part in field_path.casefold().replace("[]", "").split(".")):
        return "address"
    if "line_of_business" in leaf or leaf.endswith("_type") or leaf.endswith("_status"):
        return "enum"
    return "free_text"


def ocr_agreement(value: Any, page_text: str | None) -> float | None:
    """Whether the value appears in the text of the page it was attributed to.

    ``None`` — not 0.0 — when there is no OCR text. Image-only mode has none by
    definition, and scoring it as disagreement would teach the calibrator that
    every image-only field is untrustworthy, which is a statement about the
    input mode rather than about the extraction.

    Matched as ``common.grounding`` matches - on word boundaries, so "1" no
    longer agrees with every page holding a 1 - and ``None`` for a value too
    short to look for at all.
    """
    from common import grounding

    if page_text is None or not grounding.checkable(value):
        return None
    return 1.0 if grounding.appears(value, page_text) else 0.0


def rule_checks(field_path: str, value: Any, document: dict[str, Any]) -> bool | None:
    """Consistency checks a confident model can still fail.

    Returns ``None`` when no rule applies to this field — distinct from ``False``,
    which means a rule applied and the value broke it.
    """
    from common.normalize import normalize_currency, normalize_date
    from evaluation.metrics.field_accuracy import flatten_scalars

    leaf = field_path.rsplit(".", 1)[-1].casefold()
    # Siblings are read beside the field, wherever it sits: a canonical policy's
    # period is policy.effective_date / policy.expiration_date, and reading the
    # top level found nothing, so the date-order rule never fired for a policy.
    flat = flatten_scalars(document)
    parent = field_path[: len(field_path) - len(field_path.rsplit(".", 1)[-1])]

    if leaf in ("effective_date", "expiration_date"):
        start = normalize_date(flat.get(f"{parent}effective_date"))
        end = normalize_date(flat.get(f"{parent}expiration_date"))
        if start and end:
            return start < end
        return None

    if leaf.startswith("total_") or leaf.endswith("_total"):
        rows = document.get("claims") or document.get("line_items") or []
        component = leaf.replace("total_", "").replace("_total", "")
        parts = [
            normalize_currency(row.get(component))
            for row in rows if isinstance(row, dict) and row.get(component) is not None
        ]
        stated = normalize_currency(value)
        if stated is not None and parts:
            # A tolerance, not equality: printed totals round, and a rounding
            # difference is not the inconsistency this check exists to catch.
            return abs(sum(p for p in parts if p is not None) - stated) < 0.51
        return None

    if leaf.endswith(("_number", "_no", "_num")):
        text = str(value or "").strip()
        return bool(_IDENTIFIER_SHAPE.match(text)) if text else None

    return None


def build_features(
    *,
    field_path: str,
    value: Any,
    logprobs: list[float] | None,
    document: dict[str, Any],
    page_text: str | None = None,
    cross_mode_value: Any = None,
    mapped: bool = True,
    reason: str | None = None,
    printed_value: Any = None,
    common_model: bool = False,
) -> FieldFeatures:
    """Assemble one field's feature vector.

    A ``null`` value carries no tokens, so its logprob features are left at zero
    and ``is_null`` carries the signal instead. That is the whole reason nulls
    are calibrated as their own class: there is no span to be uncertain about,
    and treating the absence as maximum confidence is how a false null ships
    unreviewed.

    ``common_model``: the document is on a common-model line, so its fields are
    typed as its schema declares them (:func:`infer_field_type`).
    """
    is_null = (
        value is None
        or (isinstance(value, str) and not value.strip())
        # An empty scalar list (``line_of_business: []``) emits no value tokens
        # either, so it is calibrated with the nulls rather than as a field the
        # span mapper failed to find.
        or (isinstance(value, list) and not value)
    )
    spans = list(logprobs or [])

    features = FieldFeatures(
        field_path=field_path,
        field_type=infer_field_type(field_path, value, common_model=common_model),
        value=value,
        token_count=len(spans),
        is_null=is_null,
        mapped=mapped,
        reason=reason,
        # Against what the page PRINTS. ``value`` is the normalised form — a date
        # rewritten to MM/DD/YYYY, a figure stripped of "$" and "," — which is
        # not on the page, so every reformatted value scored 0 agreement.
        ocr_agreement=None if is_null else ocr_agreement(
            value if printed_value is None else printed_value, page_text
        ),
        rule_checks_passed=None if is_null else rule_checks(field_path, value, document),
    )
    if spans:
        features.min_logprob = min(spans)
        features.mean_logprob = sum(spans) / len(spans)
        features.first_token_logprob = spans[0]
    if cross_mode_value is not None and not is_null:
        features.cross_mode_agreement = float(
            values_match(cross_mode_value, value, field_path=field_path)
        )
    return features


def build_document_features(
    *,
    extraction: dict[str, Any],
    spans: dict[str, list[float]],
    page_text: str | None = None,
    cross_mode: dict[str, Any] | None = None,
    common_model: bool = False,
) -> list[FieldFeatures]:
    """Feature vectors for every scalar field in one extraction.

    Fields the span mapper could not locate are included with ``mapped=False``,
    not dropped: a field that silently vanishes between generation and
    confidence is one nobody reviews and nobody counts.

    ``common_model`` says the extraction is on a common-model line. Serving and
    calibration fitting both pass it for the document's own line, so a field is
    fitted and served under one calibrator.
    """
    from common.canonical import printed_view, values_view
    from evaluation.metrics.field_accuracy import flatten_scalars

    # Envelopes in, values out: features are keyed and valued on the value view
    # (both views are no-ops on a flat extraction). The printed view keeps each
    # value's printed form — a canonical envelope's ``raw`` — for OCR agreement.
    printed = flatten_scalars(printed_view(extraction))
    extraction = values_view(extraction)
    out: list[FieldFeatures] = []
    for path, value in sorted(flatten_scalars(extraction).items()):
        logprobs = _span_logprobs(path, value, spans)
        located = logprobs is not None
        empty = value is None or (isinstance(value, list) and not value)
        out.append(build_features(
            field_path=path,
            value=value,
            logprobs=logprobs,
            document=extraction,
            page_text=page_text,
            cross_mode_value=(cross_mode or {}).get(path),
            mapped=located or empty,
            reason=None if located or empty else "no span located in the generation",
            printed_value=printed.get(path),
            common_model=common_model,
        ))
    return out


def _span_logprobs(
    path: str, value: Any, spans: dict[str, list[float]]
) -> list[float] | None:
    """The token logprobs behind one flattened field.

    ``flatten_scalars`` keeps a list of scalars whole (``line_of_business``), but
    the span mapper locates each element separately (``line_of_business[0]``,
    ``[1]`` ...), because each value has its own tokens. Looking up the whole path
    alone found nothing, so every document with a line of business went to
    review at confidence 0. The elements' tokens are pooled: the minimum is then
    the weakest value's, which is what makes a set worth reviewing.

    Every element must be located. Pooling only the ones that were would score
    the list on the values the mapper happened to find.
    """
    if path in spans:
        return spans[path]
    if isinstance(value, list) and value:
        elements = [spans.get(f"{path}[{index}]") for index in range(len(value))]
        if all(e is not None for e in elements):
            return [lp for element in elements for lp in (element or [])]
    return None


@dataclass
class FeatureSet:
    """Training rows for one field type's calibrator."""

    field_type: str
    vectors: list[list[float]] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.vectors)

    def add(self, features: FieldFeatures, was_correct: bool) -> None:
        self.vectors.append(features.vector())
        self.correct.append(bool(was_correct))


def group_by_field_type(
    labelled: list[tuple[FieldFeatures, bool]],
) -> dict[str, FeatureSet]:
    """Bucket labelled features by field type, one calibrator each.

    Pooled across DOCUMENT types deliberately (§5.3): a money field behaves like
    a money field whether it came off an ACORD or a Loss Run, and splitting by
    document type as well would quarter the data each calibrator sees at exactly
    the volume where that is unaffordable.
    """
    sets: dict[str, FeatureSet] = {}
    for features, was_correct in labelled:
        if not features.is_usable:
            continue
        sets.setdefault(features.field_type, FeatureSet(features.field_type)).add(
            features, was_correct
        )
    return sets


_PAGE_MARKER = re.compile(r"^<page \d+ of \d+>\s*")


def shown_ocr_text(page_texts: list[str | None]) -> str | None:
    """The OCR text the model was shown, as the OCR-agreement feature reads it.

    ONE definition for fitting (validation rows) and serving (requests): the page
    texts of the pages sent, page markers stripped, ``None`` when there is no OCR
    text at all (image_only). Fitting used to join every user text block — the
    markers alone for an image-only row, so agreement was "present" and almost
    always 0 — while serving passed ``request.ocr_text``, ``None`` for any
    multi-page request. The calibrators learned a feature serving never sent.
    """
    from inference_core.input_builder import EMPTY_PAGE_TEXT

    texts = [_PAGE_MARKER.sub("", text or "").strip() for text in page_texts]
    # The blank-page placeholder is prompt text, not OCR: nothing agrees with it.
    joined = "\n".join(t for t in texts if t and t != EMPTY_PAGE_TEXT)
    return joined or None
