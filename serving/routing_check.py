"""The release step's routing check (Fideon SPEC_09 handoff item 6).

Before a release is written, every line of business is routed through the
serving plan as it will stand with the release promoted: to a release's
adapter, or to the base model with ``_fallback.json``. The table goes into the
release bundle, and the release step fails when a line cannot be routed or a
line the release was trained for would not reach it.

The lines are this repository's (``configs/layout_families.yaml``); the 48 L1
codes of the main repository's ``config/lob_registry.yaml`` replace them once
that file is here.
"""

from __future__ import annotations

from typing import Any

from common.constants import ACTIVE_DOC_TYPES
from serving.release_router import ServingPlan, UnservedDocType, _as_release, _best, assert_one_base_model


class RoutingCheckError(RuntimeError):
    """Raised when a line cannot be routed, or would not reach the release trained for it."""


def plan_with_candidate(plan: ServingPlan, bundle: dict[str, Any], fmt: str = "bf16") -> ServingPlan:
    """``plan`` as it stands once ``bundle`` is promoted, by the same precedence rule.

    Whatever the bundle's own status: the table says where each line would go
    if this release is served, which is the question a gated release raises too.
    A type pinned in ``plan`` stays pinned.
    """
    candidate = _as_release(bundle, fmt)
    grown = ServingPlan(releases={**plan.releases, candidate.release_id: candidate},
                        pinned=set(plan.pinned))
    for doc_type in ACTIVE_DOC_TYPES:
        if doc_type in plan.pinned:
            grown.by_doc_type[doc_type] = plan.by_doc_type[doc_type]
            continue
        covering = [r for r in grown.releases.values() if doc_type in r.covers and not r.lines]
        if covering:
            grown.by_doc_type[doc_type] = _best(covering)
    # The endpoint refuses a plan whose releases sit on different bases; so does the check.
    assert_one_base_model(grown)
    return grown


def routing_table(plan: ServingPlan, release_id: str, release_lines: list[str] | tuple[str, ...]) -> list[dict]:
    """One row per line: its family and where it goes. Raises :class:`RoutingCheckError`."""
    from common.config import lob_to_layout_family

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for line, family in sorted(lob_to_layout_family().items()):
        try:
            routed = plan.route("policy", line)
        except UnservedDocType as exc:
            problems.append(f"{line}: {exc}")
            continue
        release = routed.release
        rows.append({
            "lob": line,
            "layout_family": family,
            "release_id": release.release_id if release else None,
            "adapter": release.adapter if release else None,
            "lob_fallback_used": routed.lob_fallback_used,
        })
        if line in release_lines and (release is None or release.release_id != release_id):
            problems.append(f"{line} is a line {release_id} was trained for, but routes to "
                            f"{release.release_id if release else 'the base model'}")
    if problems:
        raise RoutingCheckError("routing check failed: " + "; ".join(problems))
    return rows
