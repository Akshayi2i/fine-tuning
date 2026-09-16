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
    """
    out: dict[str, Any] = {}
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_scalars(value, f"{path}."))
        elif isinstance(value, list):
            if any(isinstance(item, dict) for item in value):
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
    """
    expected_flat = flatten_scalars(expected)
    got_flat = flatten_scalars(got)

    report = AccuracyReport()
    for path, expected_value in sorted(expected_flat.items()):
        if skip_lists and "[" in path:
            continue
        got_value = got_flat.get(path)
        report.results.append(
            FieldResult(
                field_path=path,
                expected=expected_value,
                got=got_value,
                correct=values_match(expected_value, got_value, field_path=path),
                exact=expected_value == got_value,
            )
        )
    return report


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


def _row_key(row: Any, key_fields: list[str]) -> tuple:
    """A row's identity. Non-object rows key on their own value.

    Calling ``.get`` unconditionally raised ``AttributeError`` on a list of
    scalars — a schema-invalid but perfectly possible generation, and scoring
    runs before schema validation — which destroyed every other document's score
    in the same run.
    """
    if not isinstance(row, dict):
        return ("scalar:", str(row).strip().casefold())
    return tuple(str(row.get(f) or "").strip().casefold() for f in key_fields)


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


def _infer_key_fields(rows: list[Any]) -> list[str]:
    """Pick identifying fields for row matching.

    Prefers an explicit identifier; falls back to every scalar field, which makes
    matching strict rather than guessing at identity.
    """
    if not rows:
        return []
    # The non-dict guard comes FIRST. `candidate in rows[0]` was evaluated
    # before it, so a list of scalars — the exact case `_row_key` was hardened
    # for — raised TypeError from inside the scorer and took the whole eval run
    # down with it.
    first = next((r for r in rows if isinstance(r, dict)), None)
    if first is None:
        return []
    for candidate in ("claim_number", "policy_number", "coverage_type", "location_number"):
        if candidate in first:
            return [candidate]
    return sorted(k for k, v in first.items() if not isinstance(v, _CONTAINER_TYPES))


def find_list_fields(obj: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Every repeating structure in a document."""
    return {
        key: value
        for key, value in obj.items()
        if isinstance(value, list) and value and isinstance(value[0], dict)
    }


def score_all_list_fields(
    expected: dict[str, Any], got: dict[str, Any]
) -> dict[str, ListFieldReport]:
    """Score every list field. Fields absent from the output score zero recall,
    which is the correct reading: the rows were expected and not produced."""
    return {
        name: score_list_field(rows, got.get(name) or [], name)
        for name, rows in find_list_fields(expected).items()
    }
