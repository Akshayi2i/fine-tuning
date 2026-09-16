"""One document into many task examples (arch v2.1 §7b).

v1 had a single task: extract a whole document in one call. That does not survive
a forty-page policy or a dense Loss Run, and the v1 answer — route pages and ask
for a document-level JSON from each one — asked for a shape no training row ever
had.

v2.1 decomposes the work instead, and **trains one adapter on all of it**. The
task is declared in the prompt, so the model is never guessing which output shape
is wanted, and a routed request stays a subsequence of the full-document shape.

    ACORD      classify -> extract (all pages, one call)
    Policy     classify -> page_select -> extract (selected pages, one call)
    Loss Run   classify -> lossrun_header -> lossrun_rows (windows) ->
                           lossrun_totals -> deterministic merge

Every example a document produces shares that document's split, because they all
share its group (§8.2). Getting that wrong would leak a document's header into
train and its rows into test.

**The window planner is output-bound, not input-bound.** A Loss Run page holding
forty claim rows produces far more output than a page holding eight, and the
constraint that binds first is the reserved output budget — not the page images.
So the window shrinks as row density rises, and the planner checks its own
estimate against the budget before committing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from common.tasks import Task

log = logging.getLogger(__name__)

#: Row density -> (window pages, overlap pages). arch v2.1 §7b.
#: Overlap exists so a row split across a page break is seen whole by at least
#: one window; at density above the last band there is no overlap and the merge
#: rejoins the halves by matching the continued row.
WINDOW_BANDS: tuple[tuple[int, int, int], ...] = (
    (12, 3, 1),      # <= 12 rows/page -> 3-page window, 1-page overlap
    (25, 2, 1),      # 13-25          -> 2-page window, 1-page overlap
)
DENSE_WINDOW = (1, 0)   # > 25 rows/page -> single page, no overlap

#: Pages always added to a policy's routed set — the declarations area carries
#: the policy-level fields whatever the page selector says (§7b).
DECLARATIONS_PAGES = 3
MAX_ROUTED_PAGES = 6

#: Above this, thumbnails are sent in chunks and the selections unioned.
PAGE_SELECT_CHUNK = 60

#: Estimated output tokens per claim row. Used by the planner to shrink a window
#: before it is built. TODO Phase 0 (spike item 4): measure on real Loss Runs.
TOKENS_PER_CLAIM_ROW = 120

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


class ExpansionError(RuntimeError):
    """Raised when a document cannot be expanded into task examples."""


@dataclass
class TaskExample:
    """One training example: a task, its pages, and what it should produce."""

    task: Task
    source_id: str
    doc_type: str
    pages: list[int]                      # 1-based page numbers, in order
    target: dict[str, Any] | list[Any]
    acord_form: str | None = None
    window_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def detect_row_density(page_markdown: str) -> int:
    """Claim rows on one page, from its MinerU markdown.

    Counts pipe-table body rows, discounting the header and its separator. In
    image-only mode there is no markdown, and the caller passes a density
    measured from the rendered page instead — the planner takes a number, not a
    source, precisely so both modes reach it the same way.
    """
    if not page_markdown:
        return 0
    rows = [line for line in page_markdown.splitlines() if _TABLE_ROW.match(line)]
    separators = [r for r in rows if set(r.replace("|", "").strip()) <= {"-", ":", " "}]
    return max(0, len(rows) - len(separators) - 1)


def plan_windows(
    densities: list[int],
    *,
    output_budget: int,
    tokens_per_row: int = TOKENS_PER_CLAIM_ROW,
) -> list[list[int]]:
    """Page windows for ``lossrun_rows``, sized from row density.

    ``densities`` is one count per page, in page order. Returns 1-based page
    windows, overlapping per the band table.

    The band gives a starting size and the **output estimate shrinks it**: three
    sparse pages that together hold ninety rows still overflow a budget sized for
    sixty-five, and the page count alone cannot see that.
    """
    if not densities:
        return []
    if output_budget <= 0:
        raise ExpansionError("output_budget must be positive to plan a window")

    windows: list[list[int]] = []
    page = 0
    total_pages = len(densities)

    while page < total_pages:
        density = densities[page]
        size, overlap = DENSE_WINDOW
        for threshold, band_size, band_overlap in WINDOW_BANDS:
            if density <= threshold:
                size, overlap = band_size, band_overlap
                break

        # Shrink until the estimated output fits. A window of one page is the
        # floor: if a single page's rows do not fit the budget, the budget or the
        # schema is wrong, and that surfaces in cap_check rather than here.
        while size > 1:
            estimated = sum(densities[page:page + size]) * tokens_per_row
            if estimated <= output_budget:
                break
            size -= 1

        window = list(range(page + 1, min(page + size, total_pages) + 1))
        if window:
            windows.append(window)

        step = max(1, size - overlap) if size > 1 else 1
        page += step

    return windows


def expand_lossrun(
    *,
    source_id: str,
    golden_label: dict[str, Any],
    page_markdown: list[str],
    output_budget: int,
    densities: list[int] | None = None,
) -> list[TaskExample]:
    """Header, row windows and totals for one Loss Run (arch v2.1 §7b).

    Totals come from **two** sources and both are produced: total and subtotal
    rows captured inside row windows, and the dedicated ``lossrun_totals`` task on
    the last two pages. v2.0 looked for printed totals on pages 1-2, where they
    are not: they sit at the end of the report and per policy period (v2.1
    correction).
    """
    if not page_markdown:
        raise ExpansionError(f"{source_id}: a Loss Run needs at least one page")

    measured = densities or [detect_row_density(p) for p in page_markdown]
    total_pages = len(page_markdown)
    claims = list(golden_label.get("claims") or [])
    examples: list[TaskExample] = []

    header_fields = {
        k: v for k, v in golden_label.items()
        if k not in {"claims", "totals"}
    }
    examples.append(TaskExample(
        task=Task.LOSSRUN_HEADER,
        source_id=source_id,
        doc_type="lossrun",
        pages=list(range(1, min(2, total_pages) + 1)),
        target=header_fields,
    ))

    windows = plan_windows(measured, output_budget=output_budget)
    # Rows are apportioned to windows by the page they were found on where
    # provenance records it, and evenly otherwise. Even apportionment is a
    # fallback, not a design: it is right on a uniform report and approximate on
    # a lumpy one, which is why provenance is preferred.
    for index, window in enumerate(windows):
        rows_in_window = [
            claim for claim in claims
            if _claim_page(claim) in window
        ] if any(_claim_page(c) for c in claims) else _apportion(claims, windows, index)

        examples.append(TaskExample(
            task=Task.LOSSRUN_ROWS,
            source_id=source_id,
            doc_type="lossrun",
            pages=window,
            target=rows_in_window,
            window_index=index,
            metadata={
                "window_pages": len(window),
                "row_density": [measured[p - 1] for p in window],
                # The first page's table header travels with every window, or a
                # window starting mid-table has unlabelled columns (§7b).
                "carries_table_header": True,
            },
        ))

    examples.append(TaskExample(
        task=Task.LOSSRUN_TOTALS,
        source_id=source_id,
        doc_type="lossrun",
        pages=list(range(max(1, total_pages - 1), total_pages + 1)),
        target=golden_label.get("totals") or {},
    ))
    return examples


def _claim_page(claim: Any) -> int | None:
    if not isinstance(claim, dict):
        return None
    page = claim.get("_page") or claim.get("page")
    return int(page) if isinstance(page, (int, str)) and str(page).isdigit() else None


def _apportion(claims: list[Any], windows: list[list[int]], index: int) -> list[Any]:
    """Split rows evenly across windows when no page provenance exists."""
    if not windows:
        return []
    per_window = max(1, len(claims) // len(windows))
    start = index * per_window
    end = start + per_window if index < len(windows) - 1 else len(claims)
    return claims[start:end]


def select_policy_pages(
    provenance_pages: list[int],
    total_pages: int,
    *,
    distractors: list[int] | None = None,
) -> list[int]:
    """The routed page set for a policy extraction (arch v2.1 §7b).

    Pages 1-3 are **always** included: the declarations area carries the
    policy-level fields, and a selector that misses it produces an extraction with
    no policy number. Capped at six pages; beyond that the highest-relevance pages
    are kept and the rest logged, because an uncapped set defeats the routing.
    """
    if total_pages <= 0:
        raise ExpansionError("a policy needs at least one page")

    declarations = list(range(1, min(DECLARATIONS_PAGES, total_pages) + 1))
    selected = sorted(set(declarations) | {p for p in provenance_pages if 1 <= p <= total_pages})
    selected += [p for p in (distractors or []) if 1 <= p <= total_pages and p not in selected]
    selected = sorted(set(selected))

    if len(selected) > MAX_ROUTED_PAGES:
        kept = declarations + [p for p in selected if p not in declarations]
        dropped = kept[MAX_ROUTED_PAGES:]
        selected = sorted(kept[:MAX_ROUTED_PAGES])
        log.warning(
            "policy page selection returned %d pages; kept %s and dropped %s. A routed set that "
            "is not capped defeats the routing (arch v2.1 §7b).",
            len(kept), selected, dropped,
        )
    return selected


def page_select_chunks(total_pages: int, chunk: int = PAGE_SELECT_CHUNK) -> list[list[int]]:
    """Thumbnail batches for page selection, unioned by the caller.

    A 200-page policy cannot have every page thumbnailed in one call even at 256
    tokens a page, so selection runs in chunks and the results are unioned.
    """
    return [
        list(range(start + 1, min(start + chunk, total_pages) + 1))
        for start in range(0, max(total_pages, 0), chunk)
    ]
