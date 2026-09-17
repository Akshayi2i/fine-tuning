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
from typing import Any

from common.normalize import normalize_text, values_match

log = logging.getLogger(__name__)

#: Field types, each with its own calibrator (arch v2.1 §5.2). Identifiers and
#: money are exact-match and unforgiving; names and addresses are fuzzy-matched;
#: free text is reported but never gated.
FIELD_TYPES = (
    "identifier", "money", "date", "entity", "address", "enum", "free_text",
)

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


def infer_field_type(field_path: str, value: Any = None) -> str:
    """Which calibrator this field belongs to.

    Name-based, reusing ``common.normalize``'s inference so a field is
    *normalised* and *calibrated* under the same notion of what it is. Two
    different answers to "what kind of field is this" is how a money field gets
    compared as text and calibrated as a number.
    """
    from common.normalize import infer_field_kind

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
    if "address" in leaf:
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
    """
    if page_text is None:
        return None
    needle = normalize_text(value)
    if not needle:
        return None
    return 1.0 if needle in (normalize_text(page_text) or "") else 0.0


def rule_checks(field_path: str, value: Any, document: dict[str, Any]) -> bool | None:
    """Consistency checks a confident model can still fail.

    Returns ``None`` when no rule applies to this field — distinct from ``False``,
    which means a rule applied and the value broke it.
    """
    from common.normalize import normalize_currency, normalize_date

    leaf = field_path.rsplit(".", 1)[-1].casefold()

    if leaf in ("effective_date", "expiration_date"):
        start = normalize_date(document.get("effective_date"))
        end = normalize_date(document.get("expiration_date"))
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
) -> FieldFeatures:
    """Assemble one field's feature vector.

    A ``null`` value carries no tokens, so its logprob features are left at zero
    and ``is_null`` carries the signal instead. That is the whole reason nulls
    are calibrated as their own class: there is no span to be uncertain about,
    and treating the absence as maximum confidence is how a false null ships
    unreviewed.
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
        field_type=infer_field_type(field_path, value),
        value=value,
        token_count=len(spans),
        is_null=is_null,
        mapped=mapped,
        reason=reason,
        ocr_agreement=None if is_null else ocr_agreement(value, page_text),
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
) -> list[FieldFeatures]:
    """Feature vectors for every scalar field in one extraction.

    Fields the span mapper could not locate are included with ``mapped=False``,
    not dropped: a field that silently vanishes between generation and
    confidence is one nobody reviews and nobody counts.
    """
    from evaluation.metrics.field_accuracy import flatten_scalars

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
