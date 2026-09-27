"""Long-document page selection and merge (arch §7).

Primarily a **Policy-document** concern. A policy can run 50+ pages, each costing
1,000–2,000 vision tokens at full resolution, so sending the whole document would
blow the context window *and* spend most of it on pages containing nothing
extractable. Most policy schemas draw from a minority of pages: the declarations
page, schedule pages, and specific endorsements.

Three steps:

1. **Page routing** — a cheap first pass over per-page OCR text picks the pages
   likely to carry the schema's fields.
2. **Scoped extraction** — the model sees only those pages.
3. **Merge** — per-page results combine into one document, with a defined
   conflict rule: **the declarations page wins for policy-level fields**.

Short documents skip all of it. ACORD forms and most Loss Runs are a few pages,
and routing them would add latency and a failure mode for no benefit.

Which pages fed the extraction is **recorded** (``pages_used``), so a
low-confidence field can be traced to the page it came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from common.constants import DEFAULT_LONG_DOC_PAGE_THRESHOLD

log = logging.getLogger(__name__)

#: Section markers that identify a page worth reading, weighted by how strongly
#: they indicate policy-level fields. Declarations outranks everything because
#: it is where the canonical values live.
PAGE_SIGNALS: dict[str, float] = {
    "declaration": 10.0,
    "declarations page": 10.0,
    "policy declarations": 10.0,
    "common policy declarations": 10.0,
    "schedule": 5.0,
    "coverage schedule": 6.0,
    "named insured": 4.0,
    "policy number": 4.0,
    "policy period": 4.0,
    "premium": 3.0,
    "limits of insurance": 3.0,
    "endorsement": 2.0,
    "producer": 2.0,
}

#: Pages scoring at least this are selected. Low on purpose: missing a page that
#: carried a field is worse than reading one that did not, because a missing
#: field is silent while a wasted page only costs tokens.
SELECTION_THRESHOLD = 2.0

#: Always read the first page even if it scores nothing — the header carries the
#: document's identity, and a scan with poor OCR can score zero while still
#: being the declarations page.
ALWAYS_INCLUDE_FIRST_PAGE = True


@dataclass
class PageScore:
    page: int
    score: float
    matched: list[str] = field(default_factory=list)
    is_declarations: bool = False


@dataclass
class RoutingPlan:
    """Which pages to read, and why."""

    pages: list[int]
    scores: list[PageScore] = field(default_factory=list)
    declarations_page: int | None = None
    routed: bool = False
    reason: str = ""

    @property
    def page_count(self) -> int:
        return len(self.pages)


def score_page(text: str, page_number: int) -> PageScore:
    """Score one page by the section markers it contains."""
    lowered = (text or "").lower()
    score = 0.0
    matched: list[str] = []
    for signal, weight in PAGE_SIGNALS.items():
        if signal in lowered:
            score += weight
            matched.append(signal)

    is_declarations = any("declaration" in m for m in matched)
    return PageScore(page=page_number, score=score, matched=matched, is_declarations=is_declarations)


def plan_pages(
    page_texts: dict[int, str],
    *,
    page_threshold: int = DEFAULT_LONG_DOC_PAGE_THRESHOLD,
    selection_threshold: float = SELECTION_THRESHOLD,
    max_pages: int | None = None,
) -> RoutingPlan:
    """Decide which pages to send to the model.

    Args:
        page_texts: page number -> OCR text, 1-based.
        page_threshold: documents at or below this length are not routed.
        max_pages: cap on selected pages, highest-scoring first.
    """
    total = len(page_texts)
    # Scored whatever the length. The declarations page is what the decl group
    # reads beyond pages 1-3, and a short policy behind a fax cover sheet — its
    # declarations on page 4 or 5 of 5 — reported none, so its policy-level
    # fields were asked of three pages that do not print them.
    scores = [score_page(text, page) for page, text in sorted(page_texts.items())]
    declarations = next((s.page for s in scores if s.is_declarations), None)

    if total <= page_threshold:
        return RoutingPlan(
            pages=sorted(page_texts),
            scores=scores,
            declarations_page=declarations,
            routed=False,
            reason=f"{total} page(s) is at or below the {page_threshold}-page threshold — "
                   "single-pass extraction, no routing",
        )

    matched = {s.page for s in scores if s.score >= selection_threshold}

    # Check for an empty selection BEFORE adding the always-include first page:
    # otherwise a document where the heuristic recognised nothing would still
    # "route", to a single page, and silently drop everything else.
    if not matched:
        log.warning(
            "no page scored above %.1f in a %d-page document — falling back to all pages. "
            "The routing heuristic did not recognise this layout, and reading everything is "
            "better than reading one arbitrary page.",
            selection_threshold, total,
        )
        return RoutingPlan(
            pages=sorted(page_texts), scores=scores, declarations_page=declarations,
            routed=False,
            reason="no page matched the selection signals; reading all pages",
        )

    selected = set(matched)
    if ALWAYS_INCLUDE_FIRST_PAGE and page_texts:
        # The header carries the document's identity, and a poorly-OCR'd scan
        # can score zero while still being the declarations page.
        selected.add(min(page_texts))

    if max_pages and len(selected) > max_pages:
        # The first page and the declarations page survive the cap regardless of
        # score. Dropping them contradicts the plan's own reason for selecting
        # them, and would still report a declarations_page the model never saw.
        pinned = {min(page_texts)} if ALWAYS_INCLUDE_FIRST_PAGE else set()
        if declarations is not None:
            pinned.add(declarations)
        ranked = sorted(
            (s for s in scores if s.page in selected and s.page not in pinned),
            key=lambda s: -s.score,
        )
        selected = pinned | {s.page for s in ranked[: max(0, max_pages - len(pinned))]}
    plan = RoutingPlan(
        pages=sorted(selected),
        scores=scores,
        declarations_page=declarations,
        routed=True,
        reason=f"selected {len(selected)} of {total} pages by section signals",
    )
    log.info("page routing: %s (declarations page: %s)", plan.reason, declarations)
    return plan


# --------------------------------------------------------------------------
# Merging scoped extractions
# --------------------------------------------------------------------------

#: Fields whose canonical value lives on the declarations page when pages
#: disagree. A schedule page can repeat a policy number in a different format;
#: the declarations page is the authority.
POLICY_LEVEL_FIELDS = frozenset({
    "insured_name", "insured_address", "policy_number", "effective_date",
    "expiration_date", "carrier", "producer", "total_premium", "line_of_business",
})


@dataclass
class MergeResult:
    """A merged document, and where each conflicting field came from."""

    extraction: dict[str, Any] = field(default_factory=dict)
    field_sources: dict[str, int] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)


def merge_page_extractions(
    per_page: dict[int, dict[str, Any]],
    *,
    declarations_page: int | None = None,
) -> MergeResult:
    """Combine per-page extractions into one document.

    **No longer on the serving path.** A routed request now sends its selected
    pages interleaved in a single call, so the model returns one document-level
    object and there is nothing to merge. The declarations-precedence rule this
    function implements is stated to the model directly instead (see
    ``prompts/doc_types/policy.jinja``), which keeps it in one place rather than
    two that can disagree.

    Kept because it is the only implementation of that conflict rule in code, and
    because any future path that does produce per-page results — a batched
    fallback, or a document too long for one context — needs exactly this.

    Conflict rules, in order:

    1. **The declarations page wins** for policy-level fields (arch §7).
    2. Otherwise the first non-null value in page order wins — earlier pages
       carry the document's identity.
    3. **List fields concatenate** rather than overwrite: a Loss Run's claims
       table spanning three pages must produce all its rows, and overwriting
       would silently drop two thirds of them.
    """
    result = MergeResult()

    for page in sorted(per_page):
        for key, value in (per_page[page] or {}).items():
            if value is None:
                continue

            if isinstance(value, list):
                existing = result.extraction.setdefault(key, [])
                if isinstance(existing, list):
                    existing.extend(value)
                    result.field_sources.setdefault(key, page)
                else:
                    # An earlier page gave a scalar and this one gives rows. The
                    # rows were previously dropped with no conflict recorded, so
                    # no review flag fired — the only disagreement path in this
                    # function that lost data silently.
                    result.conflicts.append(
                        f"{key}: page {result.field_sources.get(key)} gave a scalar "
                        f"{existing!r} but page {page} gave {len(value)} row(s); keeping the rows"
                    )
                    result.extraction[key] = list(value)
                    result.field_sources[key] = page
                continue

            if key not in result.extraction:
                result.extraction[key] = value
                result.field_sources[key] = page
                continue

            if result.extraction[key] == value:
                continue

            # A genuine disagreement between pages.
            result.conflicts.append(
                f"{key}: page {result.field_sources[key]} says {result.extraction[key]!r}, "
                f"page {page} says {value!r}"
            )
            if key in POLICY_LEVEL_FIELDS and page == declarations_page:
                result.extraction[key] = value
                result.field_sources[key] = page

    if result.conflicts:
        log.info("merged %d pages with %d field conflict(s)", len(per_page), len(result.conflicts))
    return result
