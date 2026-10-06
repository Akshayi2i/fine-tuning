"""Base model vs base + adapter vs gold, field by field, as an Excel workbook.

Each document's three JSONs are flattened to one row per field. Table rows are
matched by their identifier (a VIN, a form number, a coverage name - the same
identifiers evaluation matches rows on), not by position, so a vehicle the model
listed second is compared with the gold's vehicle with the same VIN:
``auto.vehicles[vin=1hgcm82633a004352].year``. A table with no identifier is
matched by position (``[#2]``).

Each model's value gets a result against the gold value:

* **correct** - the same value (``values_match``, the comparison every accuracy
  metric uses: dates, amounts and names normalised);
* **wrong** - a different value;
* **missed** - the gold has a value, the model left it empty;
* **invented** - the model wrote a value where the gold has none.

Gold is compared as training and scoring use it: narrowed to what the line's
schema can hold (``common.canonical.schema_label``). The system-filled fields
(file name, page count) are left out of both sides.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULTS = ("correct", "wrong", "missed", "invented")

#: Cell colours per result.
FILLS = {"correct": "C6EFCE", "wrong": "FFC7CE", "missed": "FFEB9C", "invented": "E4D2F5"}


@dataclass
class FieldRow:
    path: str
    gold: Any
    base: Any
    adapter: Any
    base_result: str | None
    adapter_result: str | None
    gold_pages: list[int] = field(default_factory=list)

    @property
    def change(self) -> str:
        """``fixed`` when the adapter got right what the base did not; ``broke`` the reverse."""
        if self.gold in (None, "", []) and self.base_result is None and self.adapter_result is None:
            return ""
        base_ok, adapter_ok = self.base_result == "correct", self.adapter_result == "correct"
        if adapter_ok and not base_ok:
            return "fixed"
        if base_ok and not adapter_ok:
            return "broke"
        return ""


@dataclass
class ModelScore:
    correct: int = 0
    wrong: int = 0
    missed: int = 0
    invented: int = 0

    def add(self, result: str | None) -> None:
        if result:
            setattr(self, result, getattr(self, result) + 1)

    @property
    def gold_values(self) -> int:
        return self.correct + self.wrong + self.missed

    @property
    def written(self) -> int:
        return self.correct + self.wrong + self.invented

    @property
    def precision(self) -> float | None:
        return self.correct / self.written if self.written else None

    @property
    def recall(self) -> float | None:
        return self.correct / self.gold_values if self.gold_values else None

    @property
    def f1(self) -> float | None:
        total = self.written + self.gold_values
        return 2 * self.correct / total if total else None


@dataclass
class Comparison:
    rows: list[FieldRow]
    base: ModelScore
    adapter: ModelScore
    gold_outside_schema: int = 0


# --------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _is_rows(node: Any) -> bool:
    from common.canonical import is_field_value

    return (isinstance(node, list) and bool(node)
            and all(isinstance(r, dict) and not is_field_value(r) for r in node))


def _key_field(rows: list[dict[str, Any]]) -> str | None:
    """The identifier a table's rows are matched on, from the gold's rows when it has any."""
    from common.canonical import values_view
    from evaluation.metrics.field_accuracy import ROW_IDENTIFIERS, _infer_key_fields

    keys = _infer_key_fields(values_view(rows))
    return keys[0] if keys and keys[0] in ROW_IDENTIFIERS else None


def _labelled(rows: list[dict[str, Any]], key: str | None) -> list[tuple[str, dict[str, Any]]]:
    from common.canonical import values_view
    from evaluation.metrics.field_accuracy import _key_text

    out, seen = [], {}
    for index, row in enumerate(rows, start=1):
        text = _key_text(values_view(row.get(key))) if key else ""
        label = f"{key}={text}" if text else f"#{index}"
        seen[label] = seen.get(label, 0) + 1
        out.append((label if seen[label] == 1 else f"{label}~{seen[label]}", row))
    return out


def keyed_flatten(node: Any, gold: Any = None, prefix: str = "",
                  out: dict[str, Any] | None = None) -> dict[str, Any]:
    """``{path: leaf}`` with table rows labelled by their identifier.

    A leaf is a field envelope, a list of envelopes (a set such as
    ``line_of_business``) or a bare value. ``gold`` is the gold label's node at
    the same place: its rows decide which field identifies a table, so the gold
    and both models label the same row the same way.
    """
    from common.canonical import is_field_value

    out = {} if out is None else out
    if is_field_value(node):
        out[prefix] = node
    elif isinstance(node, dict):
        source = gold if isinstance(gold, dict) else {}
        for key, value in node.items():
            keyed_flatten(value, source.get(key), f"{prefix}.{key}" if prefix else key, out)
    elif _is_rows(node):
        gold_rows = gold if _is_rows(gold) else []
        key = _key_field(gold_rows or node)
        gold_by_label = dict(_labelled(gold_rows, key))
        for label, row in _labelled(node, key):
            keyed_flatten(row, gold_by_label.get(label), f"{prefix}[{label}]", out)
    else:
        out[prefix] = node
    return out


def _value(leaf: Any) -> Any:
    from common.canonical import values_view

    return values_view(leaf)


def _pages(leaf: Any) -> list[int]:
    from common.canonical import is_field_value

    leaves = leaf if isinstance(leaf, list) else [leaf]
    pages = {int(p) for item in leaves if is_field_value(item) for p in (item.get("page_ref") or [])}
    return sorted(pages)


def _result(gold: Any, got: Any, path: str) -> str | None:
    from common.normalize import values_match

    if _empty(gold):
        return "invented" if not _empty(got) else None
    if _empty(got):
        return "missed"
    return "correct" if values_match(gold, got, field_path=path) else "wrong"


def _stated(node: Any) -> int:
    from common.canonical import is_field_value

    if is_field_value(node):
        return int(node.get("raw") is not None or node.get("parsed") is not None)
    if isinstance(node, dict):
        return sum(_stated(v) for v in node.values())
    if isinstance(node, list):
        return sum(_stated(v) for v in node)
    return 0


def compare(gold_label: dict[str, Any] | None, base: dict[str, Any], adapter: dict[str, Any],
            *, doc_type: str = "policy", acord_form: str | None = None, lob: Any = None) -> Comparison:
    """One row per field any of the three holds a value for, with each model's result."""
    from common.canonical import schema_label, without_system_fields
    from common.label_mapping import map_label

    from common.structural_ids import comparable_for

    gold = without_system_fields(schema_label(gold_label or {}, doc_type, acord_form, lob))
    outside = _stated(without_system_fields(map_label(gold_label or {}, lob))) - _stated(gold)
    base, adapter = without_system_fields(base or {}), without_system_fields(adapter or {})
    # A common-model line's ids are each writer's own numbering: compared by
    # what its links name, never by id (common.structural_ids.comparable_view).
    gold, base, adapter = (comparable_for(doc, doc_type, acord_form, lob) for doc in (gold, base, adapter))

    flat_gold = keyed_flatten(gold, gold)
    flat_base = keyed_flatten(base, gold)
    flat_adapter = keyed_flatten(adapter, gold)

    rows: list[FieldRow] = []
    scores = (ModelScore(), ModelScore())
    for path in dict.fromkeys([*flat_gold, *flat_adapter, *flat_base]):
        gold_value = _value(flat_gold.get(path))
        base_value, adapter_value = _value(flat_base.get(path)), _value(flat_adapter.get(path))
        if _empty(gold_value) and _empty(base_value) and _empty(adapter_value):
            continue
        base_result, adapter_result = _result(gold_value, base_value, path), _result(gold_value, adapter_value, path)
        scores[0].add(base_result)
        scores[1].add(adapter_result)
        rows.append(FieldRow(path, gold_value, base_value, adapter_value, base_result, adapter_result,
                             _pages(flat_gold.get(path))))
    return Comparison(rows, scores[0], scores[1], max(outside, 0))


# --------------------------------------------------------------------------
# Workbooks
# --------------------------------------------------------------------------


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _percent(value: float | None) -> Any:
    return round(value, 4) if value is not None else ""


def _style_header(sheet, row: int = 1) -> None:
    from openpyxl.styles import Font, PatternFill

    for cell in sheet[row]:
        if cell.value is not None:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F3864")


def write_comparison(path: Path, comparison: Comparison, *, document: str, line: str,
                     base_label: str = "Base", adapter_label: str = "Base + adapter") -> Path:
    """``comparison.xlsx``: a Summary sheet and a Fields sheet."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    book = Workbook()
    summary = book.active
    summary.title = "Summary"
    summary.append(["Document", document])
    summary.append(["Line of business", line])
    summary.append(["Gold values compared", comparison.base.gold_values])
    summary.append(["Gold values outside the schema (not compared)", comparison.gold_outside_schema])
    summary.append([])
    header_row = summary.max_row + 1
    summary.append(["Measure", base_label, adapter_label, "Change"])
    _style_header(summary, header_row)
    for name in ("correct", "wrong", "missed", "invented"):
        a, b = getattr(comparison.base, name), getattr(comparison.adapter, name)
        summary.append([name.capitalize(), a, b, b - a])
    for name in ("precision", "recall", "f1"):
        a, b = getattr(comparison.base, name), getattr(comparison.adapter, name)
        change = round(b - a, 4) if a is not None and b is not None else ""
        summary.append([name.upper() if name == "f1" else name.capitalize(), _percent(a), _percent(b), change])
        for cell in summary[summary.max_row][1:4]:
            cell.number_format = "0.0%"
    fixed = sum(r.change == "fixed" for r in comparison.rows)
    broke = sum(r.change == "broke" for r in comparison.rows)
    summary.append(["Fields the adapter fixed / broke", fixed, broke, ""])
    summary.append([])
    for line_text in (
        "Correct: same value as gold (dates, amounts and names normalised). Wrong: a different value.",
        "Missed: gold has a value, the model left it empty. Invented: a value where gold has none.",
        "Precision = correct / values written. Recall = correct / gold values. F1 combines both.",
        "Table rows are matched by identifier (VIN, form number, coverage name ...), not position.",
    ):
        summary.append([line_text])
        summary[summary.max_row][0].font = Font(italic=True, color="595959")
    summary.column_dimensions["A"].width = 46
    for column in "BCD":
        summary.column_dimensions[column].width = 18

    fields = book.create_sheet("Fields")
    fields.append(["Field", "Gold", base_label, f"{base_label} result", adapter_label,
                   f"{adapter_label} result", "Change", "Gold page"])
    _style_header(fields)
    for row in comparison.rows:
        fields.append([row.path, _cell(row.gold), _cell(row.base), row.base_result or "",
                       _cell(row.adapter), row.adapter_result or "", row.change, _cell(row.gold_pages)])
        for column in (4, 6):
            result = fields.cell(fields.max_row, column).value
            if result in FILLS:
                fields.cell(fields.max_row, column).fill = PatternFill("solid", fgColor=FILLS[result])
    for column, width in zip("ABCDEFGH", (60, 32, 32, 14, 32, 16, 9, 10), strict=True):
        fields.column_dimensions[column].width = width
    for cells in fields.iter_rows(min_row=2):
        for cell in cells:
            cell.alignment = Alignment(vertical="top", wrap_text=cell.column in (2, 3, 5))
    fields.freeze_panes = "B2"
    fields.auto_filter.ref = fields.dimensions

    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return path


def write_overview(path: Path, documents: Iterable[tuple[str, str, Comparison]]) -> Path:
    """``summary.xlsx``: one row per document, and the pooled totals."""
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.title = "Documents"
    sheet.append(["Document", "Line", "Gold values", "Base F1", "Adapter F1", "F1 change",
                  "Base recall", "Adapter recall", "Base precision", "Adapter precision",
                  "Adapter fixed", "Adapter broke"])
    _style_header(sheet)
    pooled = (ModelScore(), ModelScore())
    for name, line, comparison in documents:
        for score, source in zip(pooled, (comparison.base, comparison.adapter), strict=True):
            for result in RESULTS:
                setattr(score, result, getattr(score, result) + getattr(source, result))
        base, adapter = comparison.base, comparison.adapter
        sheet.append([name, line, base.gold_values, _percent(base.f1), _percent(adapter.f1),
                      _percent(adapter.f1 - base.f1) if base.f1 is not None and adapter.f1 is not None else "",
                      _percent(base.recall), _percent(adapter.recall),
                      _percent(base.precision), _percent(adapter.precision),
                      sum(r.change == "fixed" for r in comparison.rows),
                      sum(r.change == "broke" for r in comparison.rows)])
    base, adapter = pooled
    sheet.append(["ALL DOCUMENTS (pooled)", "", base.gold_values, _percent(base.f1), _percent(adapter.f1),
                  _percent(adapter.f1 - base.f1) if base.f1 is not None and adapter.f1 is not None else "",
                  _percent(base.recall), _percent(adapter.recall),
                  _percent(base.precision), _percent(adapter.precision), "", ""])
    for cells in sheet.iter_rows(min_row=2, min_col=4, max_col=10):
        for cell in cells:
            cell.number_format = "0.0%"
    sheet.column_dimensions["A"].width = 48
    sheet.freeze_panes = "B2"
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return path
