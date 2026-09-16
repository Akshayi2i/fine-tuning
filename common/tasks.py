"""The six tasks the unified model is trained on (arch v2.1 §7b).

v1 had one task: extract a whole document in one call. That does not survive
contact with a 40-page policy or a dense Loss Run, so v2.1 decomposes the work
and trains **one adapter on all of it** — the task is declared in the prompt, so
the model is never guessing which output shape is wanted.

Two consequences run through the rest of the codebase:

* **Vision and sequence budgets are per task, not global.** A classification
  thumbnail and a six-page policy extraction have nothing in common in either
  dimension, and one global cap has to be wrong for one of them. See
  ``configs/shared/vision.yaml`` and ``configs/shared/sequence.yaml``.
* **A document expands into several training examples, of several tasks.** They
  all share the document's split (arch §8.2), or the same page appears on both
  sides of it.
"""

from __future__ import annotations

from enum import StrEnum


class Task(StrEnum):
    """What a single model call is being asked to do."""

    #: Page 1 at thumbnail resolution -> doc_type, acord_form, acord_edition.
    #: Selects the prompt, the schema and (after graduation) the adapter, so a
    #: wrong answer wastes everything downstream of it (arch §4a).
    CLASSIFY = "classify"

    #: Thumbnails of every page -> the page numbers carrying schema fields.
    #: Labels come free from field provenance: the pages the golden values were
    #: found on (arch §7b).
    PAGE_SELECT = "page_select"

    #: The whole document, or the selected pages, in ONE call. Not one call per
    #: page: a routed request has to be a subsequence of the full-document shape
    #: the model trained on, or it is a shape no training row ever had.
    EXTRACT = "extract"

    #: Loss Run pages 1-2 -> carrier, insured, policy periods, valuation date.
    LOSSRUN_HEADER = "lossrun_header"

    #: An adaptive 1-3 page window -> claim rows, plus any total/subtotal rows on
    #: those pages tagged with their row_type. Window size falls as row density
    #: rises, because a dense page overflows the output budget, not the input one.
    LOSSRUN_ROWS = "lossrun_rows"

    #: The last 2 pages -> grand totals and per-period totals. Separate from
    #: LOSSRUN_ROWS because printed totals sit at the end of the report, not on
    #: the first pages where v2.0 looked for them (v2.1 correction).
    LOSSRUN_TOTALS = "lossrun_totals"


#: Tasks that read page images at full extraction resolution. The two thumbnail
#: tasks are excluded: they recognise layout, and paying extraction resolution
#: for that is the single most wasteful thing this pipeline could do.
FULL_RESOLUTION_TASKS: frozenset[Task] = frozenset({
    Task.EXTRACT, Task.LOSSRUN_HEADER, Task.LOSSRUN_ROWS, Task.LOSSRUN_TOTALS,
})

#: The Loss Run decomposition, in the order the serving flow runs it.
LOSSRUN_TASKS: tuple[Task, ...] = (
    Task.LOSSRUN_HEADER, Task.LOSSRUN_ROWS, Task.LOSSRUN_TOTALS,
)


def parse(value: str | Task) -> Task:
    """Coerce a string to a ``Task``, naming the valid set when it is not one."""
    if isinstance(value, Task):
        return value
    try:
        return Task(value)
    except ValueError as exc:
        raise ValueError(
            f"unknown task {value!r}; expected one of {[t.value for t in Task]}"
        ) from exc
