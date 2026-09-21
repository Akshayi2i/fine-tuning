"""Run ids — one shape, one place (arch v2.1 §12).

A run id is ``{lineage}-{version}``: ``extractor-v2``, ``policy-v2``,
``foundation-v1``, ``lossrun-adapter-v3``. The lineage says which training run
this is; the version is the artifact tag everything else resolves by.

**Why this module exists.** The id was minted by f-string in three unconnected
places (``training/train.py``, and twice in ``orchestration/pipeline_dag.py``)
and taken apart by ``rsplit("-", 1)`` in two more. Five independent opinions
about one format: the trainer could write an id the gate could not reconstruct,
and nothing would notice until the gate looked for a manifest that was never
written under that name. Scoped runs make that worse, because the lineage stops
being the constant ``extractor``.

The version grammar is ``v`` followed by dotted digits, matching
``artifact_registry.paths._VERSION_RE``, so ``extractor-v2.1`` parses as
``("extractor", "v2.1")`` — ``rsplit("-", 1)`` got that right by luck and
``_tag_of`` got ``foundation-v10`` right only because nothing had reached v10.
"""

from __future__ import annotations

import re
from typing import NamedTuple

#: The lineage the unified run has always used. Kept as a constant so the scope
#: work can hand ``extractor`` to the unified scope and its own name to every
#: other one, without any call site rebuilding the string.
UNIFIED_LINEAGE = "extractor"

#: ``{lineage}-{version}``. The lineage is non-greedy so the LAST ``-v…`` group
#: wins: ``lossrun-adapter-v3`` is lineage ``lossrun-adapter``, not ``lossrun``.
RUN_ID_RE = re.compile(r"^(?P<lineage>[a-z][a-z0-9_-]*?)-(?P<version>v\d+(?:\.\d+)*)$")


class RunIdError(ValueError):
    """Raised on a malformed run id."""


class ParsedRunId(NamedTuple):
    lineage: str
    version: str
    raw: str


def build_run_id(lineage: str, version: str) -> str:
    """``("extractor", "v2")`` -> ``"extractor-v2"``.

    Validated on the way out rather than trusted: an id that does not parse is
    one the registry cannot look up again, and the failure would surface at the
    gate, long after the weights were written.
    """
    run_id = f"{lineage.strip().lower()}-{version.strip().lower()}"
    if not RUN_ID_RE.match(run_id):
        raise RunIdError(
            f"{run_id!r} is not a valid run id; expected {{lineage}}-v{{n}} such as "
            "'extractor-v2' or 'policy-v2.1'"
        )
    return run_id


def parse_run_id(run_id: str) -> ParsedRunId:
    """``"extractor-v2.1"`` -> ``ParsedRunId("extractor", "v2.1", …)``."""
    if not isinstance(run_id, str):
        raise RunIdError(f"run_id must be a string, got {type(run_id).__name__}")
    match = RUN_ID_RE.match(run_id.strip().lower())
    if not match:
        raise RunIdError(
            f"malformed run id {run_id!r}; expected {{lineage}}-v{{n}} such as 'extractor-v2'"
        )
    return ParsedRunId(match.group("lineage"), match.group("version"), run_id.strip().lower())


def is_valid_run_id(run_id: str) -> bool:
    """Non-raising form of :func:`parse_run_id`."""
    try:
        parse_run_id(run_id)
    except RunIdError:
        return False
    return True


def version_of(run_id: str) -> str:
    """The artifact tag a run id resolves to: ``extractor-v2.1`` -> ``v2.1``."""
    return parse_run_id(run_id).version


def lineage_of(run_id: str) -> str:
    """The training lineage: ``policy-v2`` -> ``policy``."""
    return parse_run_id(run_id).lineage
