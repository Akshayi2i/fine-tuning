"""Which canonical sections one extraction call asks for (``configs/schema_sections.yaml``).

A canonical policy schema does not fit in a prompt alongside the pages it
describes. This module cuts it into named slices — ``decl``, ``arrays``,
``lineblk``, ``dtd`` — so a window's prompt describes only what that window can
answer, and so a window has room for seven or eight pages instead of four.

**A slice is a pure function of (lob, group).** No gold, no page list, no
document. That is the property serving depends on: it has no label to consult,
so anything derived from one cannot be reproduced at inference time, and a
training row whose shape cannot be reproduced is a row that teaches a shape the
model will never be asked for again.

The group a section belongs to is declared. The *pages* a group reads are
computed — from the routed set and, for the declarations group, from where the
declarations were actually found.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

#: The slice that reads the pages the declarations are printed on.
DECLARATIONS_GROUP = "decl"

#: Written in the YAML where a group takes every section no other group claims.
REMAINDER = "remainder"


class SectionMapError(RuntimeError):
    """Raised when the section map and a canonical schema disagree."""


@lru_cache(maxsize=1)
def _map() -> dict[str, Any]:
    from common.config import CONFIG_DIR, load_yaml

    return load_yaml(CONFIG_DIR / "schema_sections.yaml")


def group_names() -> tuple[str, ...]:
    """Every declared group, in declaration order.

    Order is the order windows are built in, so it is also the order a merge
    sees them. ``decl`` leads: the declarations value wins a conflict, and a
    merge that reads it first can apply that rule by construction.
    """
    return tuple(_map()["groups"])


def task_for(group: str) -> str:
    return str(_declared(group)["task"])


def _declared(group: str) -> dict[str, Any]:
    try:
        return _map()["groups"][group]
    except KeyError:
        raise SectionMapError(
            f"no section group {group!r}; declared groups: {list(group_names())}"
        ) from None


def sections_for(group: str, lob: str | list[str] | None = None) -> tuple[str, ...]:
    """The top-level sections this group asks for, in schema order.

    Schema order rather than the order written in the YAML, so the sliced schema
    reads the way the client's file does and two slices of one schema cannot
    disagree about how it is laid out.

    A group declared ``remainder`` takes every section no other group claims.
    That is what lets a line this repo has never seen — a new section named
    ``covered_classes`` — still have it asked for, rather than silently dropped
    because nobody added it to a list.
    """
    from common.schemas import load_schema

    properties = list(load_schema("policy", None, lob).get("properties") or {})
    declared = _declared(group)["sections"]

    if declared == REMAINDER:
        claimed = {
            name
            for other in group_names()
            if _declared(other)["sections"] != REMAINDER
            for name in _declared(other)["sections"]
        }
        return tuple(name for name in properties if name not in claimed)

    wanted = set(declared)
    return tuple(name for name in properties if name in wanted)


def groups_for(lob: str | list[str] | None = None) -> tuple[str, ...]:
    """The groups that have anything to ask for on this line.

    ``dtd`` is carried by three lines out of thirty-three, and ``lineblk`` is
    empty on the fallback schema. A group with no sections would render a prompt
    with an empty schema and ask the model for ``{}`` — a wasted call whose
    output is indistinguishable from a page holding nothing.
    """
    return tuple(name for name in group_names() if sections_for(name, lob))


def pages_for(
    group: str,
    routed_pages: list[int],
    *,
    declarations_page: int | None = None,
) -> list[int]:
    """The pages this group reads, out of the routed set.

    ``routed`` groups read all of them. The declarations group reads the leading
    pages plus, when it falls outside them, the page the declarations were
    actually found on.

    That last clause is the difference between working and not on scanned
    intake. A policy that opens with a fax cover sheet or a billing notice has
    its declarations on page 4 or 5; a group pinned to pages 1-3 would never ask
    for the policy-level fields, and every one of them would come back null on
    exactly the documents most likely to be messy.
    """
    rule = str(_declared(group)["pages"])
    if rule == "routed":
        return sorted(set(routed_pages))
    if rule != "declarations":
        raise SectionMapError(f"group {group!r} declares an unknown page rule {rule!r}")

    leading = int(_map()["pages"]["declarations_leading"])
    pages = {p for p in routed_pages if p <= leading}
    if declarations_page and declarations_page in routed_pages:
        pages.add(declarations_page)
    # Never empty: a routed set that somehow holds no leading page still has to
    # be asked for the policy-level fields somewhere.
    return sorted(pages) or sorted(routed_pages)[:1]


def array_key(section: str) -> tuple[str, ...]:
    """The fields that identify one row of ``section``, for de-duplication.

    An array is asked over several page windows, so a table spanning a window
    boundary comes back twice — legitimately. Comparison is on normalised
    values, so ``"LOC 1"`` and ``"1"`` are one location.
    """
    return tuple(_map().get("array_keys", {}).get(section, ()))


def assert_sections_cover_the_schema(lob: str | list[str] | None = None) -> None:
    """Every top-level section is asked for by exactly one group.

    The check that keeps this file honest when the client ships a new schema
    version. A section in no group is a section no window asks for — the field
    returns null and the extraction still validates, which is the failure this
    whole design exists to avoid. A section in two groups is two windows both
    claiming to own it, and a merge with a conflict that should not exist.

    A group naming a section the schema does not have is also refused: it means
    the client renamed something and this map still describes the old shape.
    """
    from common.schemas import load_schema

    properties = set(load_schema("policy", None, lob).get("properties") or {})
    seen: dict[str, str] = {}
    for group in group_names():
        declared = _declared(group)["sections"]
        if declared != REMAINDER:
            missing = set(declared) - properties
            if missing and group != "dtd":
                raise SectionMapError(
                    f"group {group!r} names {sorted(missing)}, which {lob or 'the fallback'} does "
                    "not have. The client's schema was renamed and this map still describes the "
                    "old shape."
                )
        for section in sections_for(group, lob):
            if section in seen:
                raise SectionMapError(
                    f"{section!r} is claimed by both {seen[section]!r} and {group!r}, so two "
                    "windows would own it and the merge would have a conflict that should not "
                    "exist."
                )
            seen[section] = group

    unclaimed = properties - set(seen)
    if unclaimed:
        raise SectionMapError(
            f"no group asks for {sorted(unclaimed)} on {lob or 'the fallback'}, so no window would "
            "request those fields. They would come back null and the extraction would still "
            "validate."
        )
