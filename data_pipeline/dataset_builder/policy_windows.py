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
nobody would notice - and only the window's own pages: a value printed on pages
1 and 30 is cited as ``[1]`` by the window showing page 1, which cannot see page
30 (Fideon SPEC_09 amendment item 1). The serving merge joins a value's pages
across windows again (``serving.policy_merge``).

Two sections are handled apart (Fideon SPEC_21): ``text_sections`` is never in
a target - the section builder attaches it at inference - and each
``additional_fields`` entry (label, value, section_hint, page_ref) is placed by
its own ``page_ref``, as a field value is. Both only reach a target through a
schema that defines them; no schema in this repository does yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from common.canonical import in_schema_order, is_field_value, to_model_target
from common.constants import DEFAULT_LONG_DOC_PAGE_THRESHOLD

#: Never in an assistant target: attached at inference by the section builder
#: (Fideon SPEC_21, ``fideon:fsm_exclude``; SPEC_09 amendment item 1).
EXCLUDED_FROM_TARGETS = frozenset({"text_sections"})

#: Entries placed by their own page_ref (Fideon SPEC_21 §additional_fields).
ADDITIONAL_FIELDS = "additional_fields"


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
    #: Row fragments left out of a window because they hold none of their row's
    #: identifiers there (a premium with no coverage name): no window, and no
    #: merge, can tell which row such a fragment belongs to.
    orphaned: list[str] = field(default_factory=list)
    #: Values with no recorded page that were given the one page whose OCR
    #: text prints them (:func:`with_inferred_pages`).
    inferred: list[str] = field(default_factory=list)


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
    from common.schema_sections import groups_for, pages_for, reads_declarations
    from data_pipeline.dataset_builder.expand_tasks import (
        pages_per_extraction_call,
        plan_policy_windows,
    )

    plans: list[PolicyWindowPlan] = []
    for group in groups_for(lob):
        pages = pages_for(group, list(routed), declarations_page=declarations_page)
        capacity = pages_per_extraction_call("policy", None, lob, group)
        # A declarations group's pages all lead — pages 1-3 and the page the
        # declarations were found on, read together. A routed group packs by
        # runs alone.
        windows = plan_policy_windows(
            pages, pages_per_window=capacity,
            leading=pages if reads_declarations(group) else (),
        )
        plans.extend(
            PolicyWindowPlan(group, index, tuple(window), single=len(windows) == 1)
            for index, window in enumerate(windows)
        )
    return plans


#: A printed value shorter than this is too common to place by searching:
#: "1", "Yes" and "100" are on most pages of a policy.
MIN_PLACEABLE_CHARS = 4


def multi_window_sections(lob: str | list[str] | None, plans: Sequence[PolicyWindowPlan]) -> set[str]:
    """The label sections read over several windows: the only ones where a
    value with no recorded page cannot be placed."""
    from common.schema_sections import sections_for

    return {name for plan in plans if not plan.single for name in sections_for(plan.group, lob)}


def with_inferred_pages(
    label: dict[str, Any], page_texts: Sequence[str] | None, report: TargetReport | None = None,
    *, sections: set[str] | None = None,
) -> dict[str, Any]:
    """``label`` with a page given to each value that records none, where the
    document's OCR text prints it on exactly one page.

    A value with no page in a group read over several windows can be asked of
    no window, so it was left out of training - 1,589 values on the delivered
    bundles. Placed only when the search is unambiguous: the value's printed
    text, at least :data:`MIN_PLACEABLE_CHARS` characters, matched on word
    boundaries, on one page and no other. On none or several it stays
    unplaced; a guessed page would teach the model to cite the wrong one.
    A copy when anything is placed; ``label`` itself otherwise.

    ``sections`` limits the search to those top-level sections
    (:func:`multi_window_sections`). A group read in ONE window keeps a value
    with no page as it is; given a page there, a declarations value whose only
    exact match was a notice on page 7 was moved to page 7 and then dropped
    from the declarations window it had always been taught in.
    """
    import copy
    import re

    from common.normalize import normalize_text

    if not page_texts:
        return label
    pages = [normalize_text(text) or "" for text in page_texts]
    placed: list[tuple[list[str], int]] = []

    def find(value: Any) -> int | None:
        needle = normalize_text(value)
        if not needle or len(needle) < MIN_PLACEABLE_CHARS:
            return None
        pattern = re.compile(rf"(?<![0-9a-z]){re.escape(needle)}(?![0-9a-z])")
        hits = [number for number, text in enumerate(pages, start=1) if pattern.search(text)]
        return hits[0] if len(hits) == 1 else None

    def walk(node: Any, path: list[Any]) -> None:
        if is_field_value(node):
            if node.get("page_ref") or (node.get("raw") is None and node.get("parsed") is None):
                return
            page = find(node.get("raw") if node.get("raw") is not None else node.get("parsed"))
            if page is not None:
                placed.append((path, page))
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, [*path, key])
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, [*path, index])

    for name, section in label.items():
        if sections is None or name in sections:
            walk(section, [name])
    if not placed:
        return label
    out = copy.deepcopy(label)
    for path, page in placed:
        node = out
        for step in path:
            node = node[step]
        node["page_ref"] = [page]
        if report is not None:
            report.inferred.append("".join(f"[{s}]" if isinstance(s, int) else f".{s}" for s in path)[1:])
    return out


def window_target(
    label: dict[str, Any],
    lob: str | list[str] | None,
    plan: PolicyWindowPlan,
    report: TargetReport | None = None,
) -> dict[str, Any]:
    """The model-form target for one window: its sections, on its pages."""
    from common.canonical import schema_label
    from common.schema_sections import sections_for
    from common.schemas import required_fields, resolved_schema

    # A label written for another line's schema, moved into this line's block
    # (configs/label_mappings.yaml). Then narrowed to what the schema can hold: a key the grammar refuses must not
    # be taught (common.canonical.within_schema).
    label = schema_label(label, "policy", None, lob)
    pages = set(plan.pages)
    sliced = {
        name: (_additional_within if name == ADDITIONAL_FIELDS else _within)(
            label[name], name, pages, plan, report)
        for name in sections_for(plan.group, lob)
        if name in label and name not in EXCLUDED_FROM_TARGETS
    }
    sliced = {k: v for k, v in sliced.items() if v not in (None, {}, [])}
    # Entries in SPEC_21's own shape (label, value, section_hint, page_ref), not
    # FieldValue envelopes: kept as they are rather than slimmed as envelopes.
    additional = sliced.pop(ADDITIONAL_FIELDS, None)
    target = to_model_target(sliced, required=required_fields("policy", None, lob, plan.group))
    if additional:
        target[ADDITIONAL_FIELDS] = [
            {key: entry[key] for key in ("label", "value", "section_hint", "page_ref") if key in entry}
            for entry in additional
        ]
    # In the order the decoding grammar writes keys (in_schema_order): the
    # window's own schema slice, the one it is constrained to.
    return in_schema_order(target, resolved_schema("policy", None, lob, plan.group))


def _within(
    node: Any, path: str, pages: set[int], plan: PolicyWindowPlan, report: TargetReport | None
) -> Any:
    if is_field_value(node):
        if node.get("raw") is None and node.get("parsed") is None:
            return None
        return _on_pages(node, path, pages, plan, report)
    if isinstance(node, dict):
        kept = {}
        for key, value in node.items():
            child = _within(value, f"{path}.{key}", pages, plan, report)
            if child not in (None, {}, []):
                kept[key] = child
        return kept
    if isinstance(node, list):
        keys = _row_identifiers(node, path)
        rows = []
        for i, item in enumerate(node):
            row = _within(item, f"{path}[{i}]", pages, plan, report)
            if row in (None, {}, []):
                continue
            # A fragment that kept none of its row's identifiers in this window
            # is an orphan: labels record a short value such as a premium of
            # "1.0" on every page it happens to be printed on, so it reaches
            # windows that never show its coverage. Taught, the model writes
            # fragments the merge cannot place; the window that shows the
            # identifier carries the whole row.
            #
            # Unless it holds table rows of its own that ARE identified here: a
            # vehicle whose VIN is on page 6 and whose coverages table is on
            # page 7 is real content of the page-7 window. Dropping it taught
            # those coverages in no window at all. The serving merge joins such
            # a fragment to its row when only one row can own it.
            if (keys and _identified(item, keys) and not _identified(row, keys)
                    and not _holds_rows(row)):
                if report is not None:
                    report.orphaned.append(f"{plan.group}:{path}[{i}]")
                continue
            rows.append(row)
        return rows
    return node


def _on_pages(
    node: dict[str, Any], path: str, pages: set[int], plan: PolicyWindowPlan, report: TargetReport | None
) -> dict[str, Any] | None:
    """``node`` as this window sees it, or None when it is printed on none of its pages.

    A copy, with ``page_ref`` narrowed to the window's pages: the label is
    cached per document and read by every window and mode, so trimming it in
    place would trim it for the windows after this one.
    """
    refs = {int(p) for p in node.get("page_ref") or []}
    if not refs:
        if plan.single:
            return node
        if report is not None and plan.window_index == 0:
            report.unplaced.append(f"{plan.group}:{path}")
        return None
    seen = refs & pages
    return {**node, "page_ref": sorted(seen)} if seen else None


def _additional_within(
    node: Any, path: str, pages: set[int], plan: PolicyWindowPlan, report: TargetReport | None
) -> list[Any]:
    """The ``additional_fields`` entries printed on this window's pages, page_ref trimmed."""
    if not isinstance(node, list):
        return []
    kept = []
    for index, entry in enumerate(node):
        if not isinstance(entry, dict) or entry.get("value") in (None, "", [], {}):
            continue
        placed = _on_pages(entry, f"{path}[{index}]", pages, plan, report)
        if placed is not None:
            kept.append(placed)
    return kept


def _row_identifiers(rows: list[Any], path: str) -> list[str]:
    """The fields that identify this table's rows: the identifier scoring
    matches them on, plus a top-level section's declared key
    (``array_keys`` in configs/schema_sections.yaml), whose fields the merge
    joins on - a location's address on the page after its number is a half the
    merge can place."""
    from common.canonical import values_view
    from common.schema_sections import array_key
    from evaluation.metrics.field_accuracy import ROW_IDENTIFIERS, _infer_key_fields

    if not rows or not all(isinstance(r, dict) for r in rows):
        return []
    keys = _infer_key_fields(values_view(rows))
    # Only a named identifier, never the all-fields fallback: every field
    # being "the key" would make every partial row an orphan.
    named = [k for k in keys if k in ROW_IDENTIFIERS or k == "building_number"]
    declared = list(array_key(path)) if "." not in path and "[" not in path else []
    return [*named, *(k for k in declared if k not in named)]


def _holds_rows(row: Any) -> bool:
    """Whether a row still carries a table of its own after slicing. Nested
    rows went through the same rule, so any that remain are identified on
    these pages (or belong to a table nothing identifies)."""
    return isinstance(row, dict) and any(
        isinstance(value, list) and value and all(isinstance(r, dict) and not is_field_value(r) for r in value)
        for value in row.values()
    )


def _identified(row: Any, keys: list[str]) -> bool:
    from common.canonical import values_view

    if not isinstance(row, dict):
        return False
    return any(values_view(row.get(k)) not in (None, "", []) for k in keys)


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
