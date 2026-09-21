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

    @property
    def covers(self) -> tuple[str, ...]:
        return self.doc_types or tuple(ACTIVE_DOC_TYPES)

    @property
    def breadth(self) -> int:
        return len(self.covers)

    def describe(self) -> str:
        return f"{self.release_id} ({self.scope}) covering {list(self.covers)}"


@dataclass
class ServingPlan:
    """What the endpoint serves, per document type."""

    by_doc_type: dict[str, ServedRelease] = field(default_factory=dict)
    releases: dict[str, ServedRelease] = field(default_factory=dict)

    @property
    def served_doc_types(self) -> tuple[str, ...]:
        return tuple(sorted(self.by_doc_type))

    @property
    def unserved_doc_types(self) -> tuple[str, ...]:
        return tuple(sorted(set(ACTIVE_DOC_TYPES) - set(self.by_doc_type)))

    def release_for(self, doc_type: str) -> ServedRelease:
        """The release that answers for this type, or raise."""
        try:
            return self.by_doc_type[doc_type]
        except KeyError:
            raise UnservedDocType(
                f"no promoted release covers {doc_type!r}. Served types are "
                f"{list(self.served_doc_types)}. Extracting it anyway would run a model that "
                "never trained on this type against a schema it has never seen, and return a "
                "confident answer nobody can tell apart from a real one."
            ) from None

    def describe(self) -> str:
        lines = [f"{dt} -> {r.describe()}" for dt, r in sorted(self.by_doc_type.items())]
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
        covering = [r for r in plan.releases.values() if doc_type in r.covers]
        if covering:
            plan.by_doc_type[doc_type] = _best(covering)

    for doc_type, release_id in (pins or {}).items():
        pinned = plan.releases.get(release_id)
        if pinned is None:
            raise ServingPlanError(
                f"routing.release_pins sends {doc_type!r} to {release_id!r}, which is not a "
                f"promoted release. Promoted: {sorted(plan.releases)}"
            )
        if doc_type not in pinned.covers:
            raise ServingPlanError(
                f"routing.release_pins sends {doc_type!r} to {release_id!r}, which covers "
                f"{list(pinned.covers)}. A pin is an explicit instruction, so this is refused "
                "rather than quietly ignored."
            )
        plan.by_doc_type[doc_type] = pinned

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
    bases = {r.base_model for r in plan.by_doc_type.values() if r.base_model}
    if len(bases) > 1:
        raise ServingPlanError(
            f"the promoted releases name {len(bases)} different base models ({sorted(bases)}), "
            "and vLLM serves one base with a LoRA per request. Promote releases trained on one "
            "base, or serve them from separate endpoints."
        )
