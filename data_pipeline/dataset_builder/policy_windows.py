"""How a policy is read: its windows, and each window's training target.

Every canonical policy — short or long — is read as the cross product of section
groups and page windows (``configs/schema_sections.yaml``): ``decl`` over the
declarations pages, ``arrays`` and ``lineblk`` over every routed page, ``dtd``
where the line carries it. Always, never above a length threshold: a threshold
computed from prompt length moves whenever the template, a schema or the vision
budget does, and would silently re-shape documents between two corpus builds.

**One planner, two callers.** :func:`plan_windows` is what the dataset build
expands a document with *and* what ``serving.pipeline`` extracts one with. It is
a function of the line, the routed pages and where the declarations were found —
never of a gold label — so the windows a model trains on are the windows it is
served. The routed pages come from the same place on both sides too:
``serving.page_router.plan_pages`` over the OCR text the row actually carries, or
every page when there is none.

**The target is the window's slice of the gold label** (:func:`window_target`):
the group's sections, narrowed to values printed on the window's own pages. A
value on a page the window was not shown is not in its target — asking the model
for it would teach it to invent. A row that spans a window boundary keeps the
fields each window can see, in both windows; the serving merge collapses the
halves. ``page_ref`` keeps the DOCUMENT's page numbers: the prompt's markers say
``<page 140 of 200>``, and a window-relative number would be wrong by an offset
nobody would notice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from common.canonical import is_field_value, to_model_target
from common.constants import DEFAULT_LONG_DOC_PAGE_THRESHOLD


@dataclass(frozen=True)
class PolicyWindowPlan:
    """One extraction call: a section group over a set of pages."""

    group: str
    window_index: int
    pages: tuple[int, ...]
    #: The group reads its pages in one window. A value with no recorded page
    #: can only be placed when there is one window to place it in.
    single: bool

    @property
    def task(self) -> str:
        from common.schema_sections import task_for

        return task_for(self.group)


@dataclass
class TargetReport:
    """What slicing a gold label into windows left out, and why."""

    #: Values with no ``page_ref`` in a group read over several windows: there
    #: is no page to decide which window may be asked for them.
    unplaced: list[str] = field(default_factory=list)
    #: Values printed only on pages no window of their group reads. The same
    #: value is unreachable at serving; counting it is how a router or a page
    #: rule that misses real content shows up at corpus build.
    unread: list[str] = field(default_factory=list)


def routed_pages(
    page_texts: Mapping[int, str] | Sequence[str] | None,
    page_count: int,
    *,
    page_threshold: int = DEFAULT_LONG_DOC_PAGE_THRESHOLD,
) -> tuple[list[int], int | None]:
    """``(pages to read, detected declarations page)`` — serving's own rule.

    No OCR text (``image_only``) means every page: there is nothing to score.
    """
    from serving.page_router import plan_pages

    if not page_texts:
        return list(range(1, page_count + 1)), None
    texts = (
        dict(page_texts) if isinstance(page_texts, Mapping)
        else {index: text for index, text in enumerate(page_texts, start=1)}
    )
    plan = plan_pages(texts, page_threshold=page_threshold)
    return list(plan.pages), plan.declarations_page


def plan_windows(
    lob: str | list[str] | None,
    routed: Sequence[int],
    declarations_page: int | None = None,
) -> list[PolicyWindowPlan]:
    """Every window a policy is read with, in group order then page order."""
    from common.schema_sections import groups_for, pages_for
    from data_pipeline.dataset_builder.expand_tasks import (
        pages_per_extraction_call,
        plan_policy_windows,
    )

    plans: list[PolicyWindowPlan] = []
    for group in groups_for(lob):
        pages = pages_for(group, list(routed), declarations_page=declarations_page)
        capacity = pages_per_extraction_call("policy", None, lob, group)
        windows = plan_policy_windows(pages, pages_per_window=capacity)
        plans.extend(
            PolicyWindowPlan(group, index, tuple(window), single=len(windows) == 1)
            for index, window in enumerate(windows)
        )
    return plans


def window_target(
    label: dict[str, Any],
    lob: str | list[str] | None,
    plan: PolicyWindowPlan,
    report: TargetReport | None = None,
) -> dict[str, Any]:
    """The model-form target for one window: its sections, on its pages."""
    from common.schema_sections import sections_for
    from common.schemas import required_fields

    pages = set(plan.pages)
    sliced = {
        name: _within(label[name], name, pages, plan, report)
        for name in sections_for(plan.group, lob)
        if name in label
    }
    sliced = {k: v for k, v in sliced.items() if v not in (None, {}, [])}
    return to_model_target(sliced, required=required_fields("policy", None, lob, plan.group))


def _within(
    node: Any, path: str, pages: set[int], plan: PolicyWindowPlan, report: TargetReport | None
) -> Any:
    if is_field_value(node):
        if node.get("raw") is None and node.get("parsed") is None:
            return None
        refs = {int(p) for p in node.get("page_ref") or []}
        if not refs:
            if plan.single:
                return node
            if report is not None and plan.window_index == 0:
                report.unplaced.append(f"{plan.group}:{path}")
            return None
        return node if refs & pages else None
    if isinstance(node, dict):
        kept = {}
        for key, value in node.items():
            child = _within(value, f"{path}.{key}", pages, plan, report)
            if child not in (None, {}, []):
                kept[key] = child
        return kept
    if isinstance(node, list):
        rows = [_within(item, f"{path}[{i}]", pages, plan, report) for i, item in enumerate(node)]
        return [row for row in rows if row not in (None, {}, [])]
    return node


def unread_values(
    label: dict[str, Any], lob: str | list[str] | None, plans: Sequence[PolicyWindowPlan]
) -> list[str]:
    """Gold values printed on pages no window of their group reads."""
    from common.schema_sections import groups_for, sections_for

    out: list[str] = []
    for group in groups_for(lob):
        read = {p for plan in plans if plan.group == group for p in plan.pages}
        for name in sections_for(group, lob):
            if name in label:
                _collect_unread(label[name], name, read, out)
    return out


def _collect_unread(node: Any, path: str, read: set[int], out: list[str]) -> None:
    if is_field_value(node):
        refs = {int(p) for p in node.get("page_ref") or []}
        stated = node.get("raw") is not None or node.get("parsed") is not None
        if stated and refs and not refs & read:
            out.append(path)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _collect_unread(value, f"{path}.{key}", read, out)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _collect_unread(item, f"{path}[{index}]", read, out)
