"""Field-level accuracy, list F1, and list recall (arch §15).

Grouped in one module because they share a comparison core: every one of them
asks "does this extracted value match the golden value", and that question has
exactly one answer, supplied by ``common.normalize``. Splitting them would mean
three copies of the matching logic and three chances for them to drift.

**Normalized match, not string equality.** Insurance fields need it or you
systematically under-count correct extractions: `01/01/2026` vs `2026-01-01`,
`$1,200.00` vs `1200.0`, `Acme Mfg LLC` vs `ACME MANUFACTURING LLC`.

**List recall is separate from list F1 on purpose.** F1 over matched rows can
look healthy while whole rows are missing, and a missing row generates no tokens,
so per-field confidence is blind to it (arch §5). Recall is the metric that sees
a dropped Loss Run claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.normalize import values_match

#: Fields that are containers, not values — scored by their rows, not directly.
_CONTAINER_TYPES = (list, dict)


@dataclass
class FieldResult:
    """One field's outcome, kept per-field so failures are attributable."""

    field_path: str
    expected: Any
    got: Any
    correct: bool
    exact: bool


@dataclass
class AccuracyReport:
    """Scalar-field accuracy across a document or a corpus."""

    results: list[FieldResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def exact_match(self) -> float:
        return sum(r.exact for r in self.results) / self.total if self.total else 0.0

    @property
    def normalized_match(self) -> float:
        return sum(r.correct for r in self.results) / self.total if self.total else 0.0

    def failures(self) -> list[FieldResult]:
        return [r for r in self.results if not r.correct]

    def by_field(self) -> dict[str, float]:
        """Per-field accuracy — an aggregate hides which field is failing."""
        buckets: dict[str, list[bool]] = {}
        for result in self.results:
            buckets.setdefault(result.field_path, []).append(result.correct)
        return {path: sum(v) / len(v) for path, v in sorted(buckets.items())}


def flatten_scalars(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten to scalar leaves, keeping list ROWS addressable.

    ``claims[0].paid`` rather than ``claims`` — per-value accuracy inside a row
    is meaningful, an aggregate over a whole table is not.

    **A list of scalars is one field, not a table.** ``line_of_business`` is a
    set of enum values (arch v2.1 §0b), so exploding it into
    ``line_of_business[0]`` would make position part of its identity — and since
    ``score_fields`` skips any path containing ``[``, it also made the field
    disappear from scoring entirely the moment it stopped being a scalar. Kept
    whole, ``values_match`` compares it as the set it is.

    **A canonical ``FieldValue`` envelope is one field, scored on its value.**
    Flattened as an object it would score ``raw``, ``page_ref`` and the
    pipeline's own ``confidence`` as though they were extracted fields; collapsed
    through :func:`common.canonical.values_view`, a canonical label and a
    canonical extraction compare exactly as two flat documents do.
    """
    from common.canonical import is_field_value, values_view

    out: dict[str, Any] = {}
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if is_field_value(value):
            out[path] = values_view(value)
        elif isinstance(value, dict):
            out.update(flatten_scalars(value, f"{path}."))
        elif isinstance(value, list):
            if value and all(is_field_value(item) for item in value):
                # A list of envelopes is a list of scalars: one field, a set.
                out[path] = values_view(value)
            elif any(isinstance(item, dict) for item in value):
                # A table: rows are addressable, and an empty one contributes
                # nothing rather than a phantom field.
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        out.update(flatten_scalars(item, f"{path}[{index}]."))
                    else:
                        out[f"{path}[{index}]"] = item
            else:
                out[path] = value
        else:
            out[path] = value
    return out


def score_fields(
    expected: dict[str, Any],
    got: dict[str, Any],
    *,
    skip_lists: bool = True,
) -> AccuracyReport:
    """Score scalar fields.

    List rows are excluded by default and scored by :func:`score_list_field`
    instead — mixing them would let a 40-row Loss Run dominate the document's
    accuracy over its half-dozen header fields.

    **A value the golden label does not state is scored as wrong.** Canonical
    targets are sparse — a field the document does not state is omitted — so a
    loop over the golden paths alone never saw an invented value: a checkpoint
    that hallucinated fields lost nothing, and checkpoint selection could prefer
    it. Flat schemas emit every key, so for them this changes nothing.
    """
    expected_flat = flatten_scalars(expected)
    got_flat = flatten_scalars(got)

    report = AccuracyReport()
    for path, expected_value in sorted(expected_flat.items()):
        if skip_lists and "[" in path:
            continue
        got_value = got_flat.get(path) if path in got_flat else _rows_at(got_flat, path)
        report.results.append(
            FieldResult(
                field_path=path,
                expected=expected_value,
                got=got_value,
                correct=values_match(expected_value, got_value, field_path=path),
                exact=expected_value == got_value,
            )
        )
    for path, got_value in sorted(got_flat.items()):
        if path in expected_flat or (skip_lists and "[" in path) or _is_empty(got_value):
            continue
        report.results.append(
            FieldResult(field_path=path, expected=None, got=got_value, correct=False, exact=False)
        )
    return report


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _rows_at(flat: dict[str, Any], path: str) -> list[dict[str, Any]] | None:
    """The table a flattened extraction holds at ``path``, or ``None``.

    An empty expected table (``claims: []``) flattens to one field, while a
    populated one flattens to ``claims[0].paid`` ... with no ``claims`` key at all.
    Looking the field up by path alone therefore found ``None``, and ``[]`` vs
    ``None`` matches — so a model that invented claims for a document with none
    scored that field correct. Rebuilding the rows makes it compare as a
    non-empty table, which does not match an empty one.
    """
    marker = f"{path}["
    rows: dict[int, dict[str, Any]] = {}
    for key, value in flat.items():
        if not key.startswith(marker):
            continue
        index_text, _, rest = key[len(marker):].partition("]")
        if not index_text.isdigit():
            continue
        rows.setdefault(int(index_text), {})[rest.lstrip(".") or "value"] = value
    return [rows[i] for i in sorted(rows)] if rows else None


# --------------------------------------------------------------------------
# List fields
# --------------------------------------------------------------------------

@dataclass
class ListFieldReport:
    """Precision, recall and F1 for one repeating structure."""

    field_name: str
    expected_rows: int
    got_rows: int
    matched_rows: int
    row_field_accuracy: float = 0.0

    @property
    def precision(self) -> float:
        return self.matched_rows / self.got_rows if self.got_rows else 0.0

    @property
    def recall(self) -> float:
        """The metric that sees a dropped row.

        A missing row produces no tokens, so per-field confidence cannot flag it
        — recall is the only signal that catches it (arch §5).
        """
        return self.matched_rows / self.expected_rows if self.expected_rows else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def missed_rows(self) -> int:
        return max(0, self.expected_rows - self.matched_rows)


def _key_text(value: Any) -> str:
    """An identifier as compared: case, spacing and separators ignored.

    ``HO 00 03`` and ``HO-0003``, ``1HGCM82633A004352`` and ``1hgcm 8263 3a004352``
    name the same form and the same car. A number read as ``1`` and ``1.0`` too.
    """
    import re

    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return re.sub(r"[\s\-./_]", "", str(value if value is not None else "")).casefold()


def _row_key(row: Any, key_fields: list[str]) -> tuple:
    """A row's identity. Non-object rows key on their own value.

    Calling ``.get`` unconditionally raised ``AttributeError`` on a list of
    scalars — a schema-invalid but perfectly possible generation, and scoring
    runs before schema validation — which destroyed every other document's score
    in the same run.
    """
    if not isinstance(row, dict):
        return ("scalar:", str(row).strip().casefold())
    return tuple(_key_text(row.get(f)) for f in key_fields)


def score_list_field(
    expected_rows: list[Any],
    got_rows: list[Any],
    field_name: str,
    *,
    key_fields: list[str] | None = None,
) -> ListFieldReport:
    """Match rows by identity, then score.

    Rows are matched on an identifying subset (a claim number, say) rather than
    by position, because a model that drops row 2 would otherwise mis-align every
    row after it and score near zero for one omission.
    """
    keys = key_fields or _infer_key_fields(expected_rows)
    # Keys are counted, not deduplicated. Indexing rows into a dict collapsed
    # duplicates, so two claim rows sharing a claim number scored recall 0.5 on a
    # byte-perfect extraction — and if a key field was null in every row they all
    # collapsed to one. Both are ordinary in real Loss Runs.
    expected_index = _index_rows(expected_rows, keys)
    got_index = _index_rows(got_rows, keys)

    # Matched by multiplicity: three expected rows sharing a key and two
    # extracted ones count as two matches, not one. Counting distinct keys made
    # a perfect extraction of a duplicated row look like a 50% recall failure.
    matched_rows = 0
    field_scores: list[float] = []
    for key in sorted(set(expected_index) & set(got_index)):
        expected_group, got_group = expected_index[key], got_index[key]
        matched_rows += min(len(expected_group), len(got_group))
        for expected_row, got_row in zip(expected_group, got_group, strict=False):
            if not isinstance(expected_row, dict) or not isinstance(got_row, dict):
                field_scores.append(1.0 if expected_row == got_row else 0.0)
                continue
            report = score_fields(expected_row, got_row, skip_lists=False)
            if report.total:
                field_scores.append(report.normalized_match)

    return ListFieldReport(
        field_name=field_name,
        expected_rows=len(expected_rows),
        got_rows=len(got_rows),
        matched_rows=matched_rows,
        row_field_accuracy=sum(field_scores) / len(field_scores) if field_scores else 0.0,
    )


def _index_rows(rows: list[Any], keys: list[str]) -> dict[tuple, list[Any]]:
    """Group rows by identity, keeping every row that shares a key."""
    grouped: dict[tuple, list[Any]] = {}
    for row in rows:
        grouped.setdefault(_row_key(row, keys), []).append(row)
    return grouped


#: Fields that identify a row, strongest first. A table is matched on the first
#: one filled in for most of its expected rows. Loss Runs key on claim numbers;
#: the canonical policy tables on form numbers, VINs, numbered rows and names.
#: Knowing only the Loss Run keys, every policy table fell back to matching on
#: ALL its fields at once, so one wrong character anywhere dropped the whole row
#: from recall - a vehicle with a wrong premium counted as a missing vehicle,
#: and list recall read 0.0 on every smoke-run checkpoint.
ROW_IDENTIFIERS: tuple[str, ...] = (
    "claim_number", "policy_number", "form_number", "vin", "vin_or_hull_id",
    "hull_identification_number", "serial_number", "loan_number", "license_number",
    "docket_number", "vehicle_number", "driver_number", "unit_number", "motor_number",
    "item_number", "installment_number", "location_number", "structure_number",
    "residence_number", "object_number", "project_number", "agreement_number",
    "class_number", "question_number", "blanket_number", "coverage_code", "coverage_type",
    "coverage_name", "coverage_part", "discount_name", "exclusion_name", "benefit_name",
    "endorsement_name", "name", "individual_name", "entity_name", "identifier_type",
    "device_type", "device_description", "location_reference", "change_description",
    "field_changed", "exposure_type", "livestock_type", "plan_name", "option_name",
    "service_name", "vendor_name", "item_title", "rank", "description", "state", "label",
)

#: Identifiers that are only unique together with a companion: a location's
#: buildings are numbered 1, 2, … within each location.
_COMPANIONS: dict[str, tuple[str, ...]] = {"location_number": ("building_number",)}


def _filled_share(rows: list[dict[str, Any]], field_name: str) -> float:
    filled = sum(1 for r in rows if _key_text(r.get(field_name)))
    return filled / len(rows) if rows else 0.0


def _infer_key_fields(rows: list[Any]) -> list[str]:
    """Pick identifying fields for row matching.

    The first :data:`ROW_IDENTIFIERS` entry filled in at least half the rows,
    with its companion when that is filled too. Otherwise every scalar field any
    row carries, which makes matching strict rather than guessing at identity.

    Read across all rows: a label leaves an unstated value out of its row, so a
    first mortgagee with no name made a table whose other rows all had one fall
    back to matching on every field.
    """
    if not rows:
        return []
    # The non-dict guard comes FIRST. `candidate in rows[0]` was evaluated
    # before it, so a list of scalars — the exact case `_row_key` was hardened
    # for — raised TypeError from inside the scorer and took the whole eval run
    # down with it.
    dict_rows = [r for r in rows if isinstance(r, dict)]
    if not dict_rows:
        return []
    for candidate in ROW_IDENTIFIERS:
        if _filled_share(dict_rows, candidate) >= 0.5:
            companions = [c for c in _COMPANIONS.get(candidate, ())
                          if _filled_share(dict_rows, c) >= 0.5]
            return [candidate, *companions]
    return sorted({k for row in dict_rows for k, v in row.items()
                   if not isinstance(v, _CONTAINER_TYPES)})


def find_list_fields(obj: dict[str, Any], prefix: str = "") -> dict[str, list[dict[str, Any]]]:
    """Every repeating structure in a document, nested ones included.

    Keyed by dotted path (``auto.vehicles``, ``premium.taxes_and_fees``). A
    canonical policy keeps most of its tables inside sections, and a top-level
    scan found none of them — they were never scored at all.
    """
    found: dict[str, list[dict[str, Any]]] = {}
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, list) and value and isinstance(value[0], dict):
            found[path] = value
        elif isinstance(value, dict):
            found.update(find_list_fields(value, f"{path}."))
    return found


def _at(obj: dict[str, Any], dotted: str) -> Any:
    node: Any = obj
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def score_all_list_fields(
    expected: dict[str, Any], got: dict[str, Any]
) -> dict[str, ListFieldReport]:
    """Score every list field. Fields absent from the output score zero recall,
    which is the correct reading: the rows were expected and not produced.

    Both sides are compared on VALUES (``values_view``). A canonical row is a
    dict of envelopes, and keying rows on the envelope made two readings of one
    claim match only when raw text and page_ref were byte-identical — and never
    against a stored golden, which also carries confidence and flagged — while
    unkeyed tables matched on nothing, so any rows counted as the right rows.
    """
    from common.canonical import values_view

    expected, got = values_view(expected), values_view(got)
    return {
        name: score_list_field(rows, _at(got, name) or [], name)
        for name, rows in find_list_fields(expected).items()
    }
