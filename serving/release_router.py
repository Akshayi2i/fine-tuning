"""Which release serves which document type (arch v2.1 §12.3, §4.1).

Once scopes exist, more than one release can be promoted at a time: a policy
release covering policies, an older unified release still covering ACORDs and
Loss Runs. Something has to decide, per document type, which one answers — and
refuse a type nothing covers.

**Narrowest coverage wins, newest breaks ties.** A release naming its types is a
statement that it was trained and gated on them; a unified release covers
everything by default. So a policy release takes policies from the unified one it
was trained to replace for that type, while the unified release keeps the rest.

**A type no promoted release covers is refused, not extracted.** The alternative
is extracting it with whichever model is loaded, against a schema that model
never trained on, and returning confident nonsense — which nothing downstream
reports as anything but a bad extraction.

**One base model per plan.** vLLM serves one base; every scope trains a LoRA on
the same pinned base, so releases are served as adapters over that shared base.
A plan whose releases disagree about the base is refused here rather than at the
first request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.constants import ACTIVE_DOC_TYPES

log = logging.getLogger(__name__)


class UnservedDocType(RuntimeError):
    """Raised when no promoted release covers a document type."""


class ServingPlanError(RuntimeError):
    """Raised when the promoted releases cannot be served together."""


@dataclass(frozen=True)
class ServedRelease:
    """One promoted release, as serving needs it."""

    release_id: str
    scope: str
    base_model: str
    merged_model: str
    #: Which types this release answers for. Empty means every active type —
    #: which is what every bundle written before scopes existed means.
    doc_types: tuple[str, ...] = ()
    adapter: str | None = None
    calibrators: str | None = None
    prompt_hash: str = ""
    ocr_pin: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    #: Lines of business this release serves. Empty = every line.
    lines: tuple[str, ...] = ()

    @property
    def covers(self) -> tuple[str, ...]:
        return self.doc_types or tuple(ACTIVE_DOC_TYPES)

    def covers_lob(self, lob: object) -> bool:
        """Every line when unrestricted; otherwise all of the document's lines."""
        if not self.lines:
            return True
        from common.scopes import lob_lines

        found = lob_lines(lob)
        return bool(found) and found <= set(self.lines)

    @property
    def breadth(self) -> int:
        return len(self.covers)

    def describe(self) -> str:
        lines = f", lines {list(self.lines)}" if self.lines else ""
        return f"{self.release_id} ({self.scope}) covering {list(self.covers)}{lines}"


@dataclass(frozen=True)
class Routed:
    """Where one document goes: a release, or the base model with no LoRA.

    ``lob_fallback_used`` means the base model reads the policy against the
    canonical ``_fallback.json``: its line's layout family has no promoted
    adapter, the line has no family, or the caller chose the fallback for a
    line that cannot be routed (Fideon SPEC_06 §9a, handoff item 4).
    """

    release: ServedRelease | None
    layout_family: str | None = None
    lob_fallback_used: bool = False


def _unroutable(lob: object) -> str | None:
    """Why a policy with this line cannot be routed by family, or None when it can."""
    from common.config import lob_to_layout_family
    from common.scopes import lob_lines

    lines = lob_lines(lob)
    if not lines:
        return ("a policy with no line of business is not routed: its layout family, and so its "
                "adapter, cannot be known. Send the policy's `lob`, or ask for the base-model "
                "fallback (allow_lob_fallback) to read it against _fallback.json.")
    unknown = sorted(line for line in lines if line not in lob_to_layout_family())
    if unknown:
        return (f"line(s) of business {unknown} are in no layout family (configs/"
                "layout_families.yaml), so no adapter can be chosen and none is guessed. Ask for "
                "the base-model fallback (allow_lob_fallback) to read the policy against "
                "_fallback.json.")
    return None


def _family(lob: object) -> str | None:
    """The one layout family of the policy's lines; None for a line with none, or a
    package whose lines fall in several (read by the base model)."""
    from common.config import lob_to_layout_family
    from common.scopes import lob_lines

    families = {lob_to_layout_family()[line] for line in lob_lines(lob)}
    return families.pop() if len(families) == 1 else None


@dataclass
class ServingPlan:
    """What the endpoint serves, per document type."""

    #: The release answering for each type across every line (no line restriction).
    by_doc_type: dict[str, ServedRelease] = field(default_factory=dict)
    releases: dict[str, ServedRelease] = field(default_factory=dict)
    #: Types an operator pinned (``routing.release_pins``). A pin is an explicit
    #: instruction — typically a rollback — so it outranks line-scoped releases.
    pinned: set[str] = field(default_factory=set)

    @property
    def line_releases(self) -> list[ServedRelease]:
        """Promoted releases narrowed by line of business (e.g. personal lines)."""
        return [r for r in self.releases.values() if r.lines]

    @property
    def served(self) -> list[ServedRelease]:
        """Every release this plan can route to, once each."""
        seen = {r.release_id: r for r in self.by_doc_type.values()}
        seen.update({r.release_id: r for r in self.line_releases})
        return list(seen.values())

    @property
    def served_doc_types(self) -> tuple[str, ...]:
        types = set(self.by_doc_type)
        types.update(d for r in self.line_releases for d in r.covers)
        return tuple(sorted(types))

    @property
    def unserved_doc_types(self) -> tuple[str, ...]:
        return tuple(sorted(set(ACTIVE_DOC_TYPES) - set(self.served_doc_types)))

    def release_for(self, doc_type: str, lob: object = None) -> ServedRelease:
        """The release that answers for this document, or raise.

        A release narrowed by line answers first for a document all of whose lines
        it covers (the narrowest such, newest on a tie); otherwise the type's
        unrestricted release. A document no release covers — a type nothing
        serves, or a line only a line-scoped release could have taken — is refused.
        """
        if doc_type in self.pinned:
            # A rollback pin sends EVERY document of the type to the pinned
            # release; checking line releases first would silently ignore it for
            # exactly the documents the rollback was for.
            return self.by_doc_type[doc_type]
        by_line = [r for r in self.line_releases if doc_type in r.covers and r.covers_lob(lob)]
        if by_line:
            fewest = min(len(r.lines) for r in by_line)
            return max((r for r in by_line if len(r.lines) == fewest),
                       key=lambda r: (r.created_at, r.release_id))
        if doc_type not in self.by_doc_type and any(doc_type in r.covers for r in self.line_releases):
            raise UnservedDocType(
                f"no promoted release serves {doc_type!r} with line of business {lob!r}. The "
                f"releases for {doc_type!r} cover only "
                f"{sorted({line for r in self.line_releases for line in r.lines})}; a document of "
                "another line — or with no line given — would be read by a model that never trained "
                "on it. Send the policy's `lob`, or promote a release covering that line."
            )
        try:
            return self.by_doc_type[doc_type]
        except KeyError:
            raise UnservedDocType(
                f"no promoted release covers {doc_type!r}. Served types are "
                f"{list(self.served_doc_types)}. Extracting it anyway would run a model that "
                "never trained on this type against a schema it has never seen, and return a "
                "confident answer nobody can tell apart from a real one."
            ) from None

    def route(self, doc_type: str, lob: object = None, *, allow_fallback: bool = False) -> Routed:
        """Where this document goes (Fideon SPEC_06 §9a, handoff item 4).

        A policy's line decides its layout family, and the release serving that
        family answers. When the family has no promoted release - or the line
        has no family - the base model reads it with no LoRA against
        ``_fallback.json`` (``lob_fallback_used``). A policy with no line, or a
        line in no family, is never routed silently: refused, unless the caller
        chose the fallback. Other types route as :meth:`release_for` does.
        """
        if doc_type != "policy":
            return Routed(self.release_for(doc_type, lob))
        problem = _unroutable(lob)
        if problem:
            if allow_fallback:
                return Routed(None, None, lob_fallback_used=True)
            raise UnservedDocType(problem)
        family = _family(lob)
        try:
            return Routed(self.release_for(doc_type, lob), family)
        except UnservedDocType:
            # A known line no promoted release covers: read by the base model,
            # flagged, rather than refused or given to another family's adapter.
            return Routed(None, family, lob_fallback_used=True)

    def describe(self) -> str:
        lines = [f"{dt} -> {r.describe()}" for dt, r in sorted(self.by_doc_type.items())]
        lines += [f"by line -> {r.describe()}" for r in sorted(self.line_releases,
                                                               key=lambda r: r.release_id)]
        if self.unserved_doc_types:
            lines.append(f"unserved: {list(self.unserved_doc_types)}")
        return "; ".join(lines) or "nothing promoted"


def _best(covering: list[ServedRelease]) -> ServedRelease:
    """The release that answers for a type: narrowest coverage, newest on a tie.

    Narrowest first because a release that names a type was trained and gated on
    that type specifically, while a unified release covers it as one of many.
    """
    narrowest = min(r.breadth for r in covering)
    return max(
        (r for r in covering if r.breadth == narrowest),
        key=lambda r: (r.created_at, r.release_id),
    )


def _as_release(bundle: dict[str, Any], fmt: str) -> ServedRelease:
    serving_formats = bundle.get("serving_formats") or {}
    return ServedRelease(
        release_id=str(bundle.get("release_id", "")),
        scope=str(bundle.get("scope") or "unified"),
        base_model=str(bundle.get("base_model", "")),
        merged_model=str(serving_formats.get(fmt) or bundle.get("merged_model", "")),
        doc_types=tuple(bundle.get("doc_types") or ()),
        adapter=bundle.get("adapter"),
        calibrators=(bundle.get("calibrators") or {}).get(fmt),
        prompt_hash=str(bundle.get("prompt_hash", "")),
        ocr_pin=dict(bundle.get("ocr_pin") or {}),
        created_at=str(bundle.get("created_at", "")),
        lines=tuple(bundle.get("lines") or ()),
    )


def build_serving_plan(
    client: BlobClient,
    *,
    tenant_id: str | None = None,
    fmt: str = "bf16",
    pins: dict[str, str] | None = None,
) -> ServingPlan:
    """Read the release index and decide which release answers for each type.

    ``pins`` (``{doc_type: release_id}``, from ``routing.release_pins``) override
    the precedence rule. A pin naming a release that does not cover its type is
    an error: a pin is an explicit instruction, and silently ignoring one would
    serve a type from a release the operator deliberately routed away from.
    """
    plan = ServingPlan()
    index_key = paths.release_index(tenant_id)
    if not client.exists(index_key):
        log.warning("no release index at %s — nothing is promoted yet", index_key)
        return plan

    for row in client.read_json(index_key) or []:
        if row.get("status") != "promoted":
            continue
        release_id = str(row.get("release_id", ""))
        bundle_key = paths.release_bundle(release_id, tenant_id)
        if not client.exists(bundle_key):
            log.warning("release %s is indexed as promoted but has no bundle", release_id)
            continue
        release = _as_release(client.read_json(bundle_key), fmt)
        if not release.merged_model:
            log.warning("release %s serves no %s artifact; skipping", release_id, fmt)
            continue
        plan.releases[release_id] = release

    for doc_type in ACTIVE_DOC_TYPES:
        # Line-scoped releases answer only for their lines (release_for); the
        # type-level choice is among releases that serve every line.
        covering = [r for r in plan.releases.values() if doc_type in r.covers and not r.lines]
        if covering:
            plan.by_doc_type[doc_type] = _best(covering)

    for doc_type, release_id in (pins or {}).items():
        pinned = plan.releases.get(release_id)
        if pinned is None:
            raise ServingPlanError(
                f"routing.release_pins sends {doc_type!r} to {release_id!r}, which is not a "
                f"promoted release. Promoted: {sorted(plan.releases)}"
            )
        if pinned.lines:
            raise ServingPlanError(
                f"routing.release_pins sends every {doc_type!r} to {release_id!r}, which serves only "
                f"lines {list(pinned.lines)}. A per-type pin cannot route other lines to it."
            )
        if doc_type not in pinned.covers:
            raise ServingPlanError(
                f"routing.release_pins sends {doc_type!r} to {release_id!r}, which covers "
                f"{list(pinned.covers)}. A pin is an explicit instruction, so this is refused "
                "rather than quietly ignored."
            )
        plan.by_doc_type[doc_type] = pinned
        plan.pinned.add(doc_type)

    assert_one_base_model(plan)
    if plan.unserved_doc_types:
        log.warning(
            "no promoted release covers %s — documents of those types will be refused rather "
            "than extracted by a model that never trained on them.", list(plan.unserved_doc_types),
        )
    log.info("serving plan: %s", plan.describe())
    return plan


def assert_one_base_model(plan: ServingPlan) -> None:
    """Every served release must sit on the same base.

    vLLM loads one base and applies a LoRA per request, so two releases trained
    on different bases cannot be served by one endpoint. Caught here rather than
    as a load failure on the first request routed to the odd one out.
    """
    bases = {r.base_model for r in plan.served if r.base_model}
    if len(bases) > 1:
        raise ServingPlanError(
            f"the promoted releases name {len(bases)} different base models ({sorted(bases)}), "
            "and vLLM serves one base with a LoRA per request. Promote releases trained on one "
            "base, or serve them from separate endpoints."
        )
