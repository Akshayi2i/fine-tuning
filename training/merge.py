"""Fold the unified adapter into the base (arch v2.1 §13 step 7, SPEC_10).

PEFT's ``merge_and_unload()`` produces one standalone bf16 model. It is what the
first serving cycle ships, and it is also what FP8 quantizes *from* (§13a), so
every serving format in a release descends from this one artifact.

**One adapter, one merge.** v1 applied a Foundation LoRA and then a per-type LoRA
on top, producing a model per document type — and the order mattered, because
reversing it produced a different model from the one that was evaluated. That
whole ordering problem is gone: there is one adapter (arch v2.1 §4.1).

A graduated per-type adapter (§4.2) is **never merged**. It is applied at serving
time on top of these merged weights, one LoRA per request — which is the only
shape vLLM can serve, and the reason the v1 stack could not be served at all.

**The base is loaded in bf16, and that is now the aligned case rather than a
compromise.** Under the §9 default the adapter trained against a bf16 base, so
merging into bf16 applies it to exactly the weights it saw. That was not true
while training ran in 4-bit: the adapter then carried a delta partly
compensating for quantization error in weights it was never merged into. If a
run sets ``load_in_4bit: true``, that gap comes back, and this merge is where it
lands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from artifact_registry import paths

log = logging.getLogger(__name__)

Dtype = Literal["fp16", "bf16"]


class MergeError(RuntimeError):
    """Raised when an adapter cannot be merged."""


@dataclass
class MergePlan:
    """What a merge will do, inspectable before it runs."""

    base_model: str
    adapter: str
    output_dir: str
    dtype: Dtype

    #: The checkpoint the §11.2 selector picked, when one was chosen. Recorded
    #: because "which checkpoint shipped" is otherwise unanswerable once the
    #: staging volume is reclaimed — the merged weights do not say.
    selected_checkpoint: str | None = None

    def describe(self) -> str:
        source = self.selected_checkpoint or self.adapter
        return f"{self.base_model} + {source} -> {self.output_dir} ({self.dtype})"


def plan_merge(
    *,
    base_model: str,
    version: str,
    dtype: Dtype = "bf16",
    selected_checkpoint: str | None = None,
    scope: str | None = None,
) -> MergePlan:
    """Assemble a merge plan without touching any weights.

    ``dtype`` defaults to **bf16**, not fp16: the adapter trained in bf16 against
    a bf16 base, and FP8 is quantized from this artifact (§13a). Merging to fp16
    would introduce a precision change between training and every serving format
    for no reason.

    ``scope`` selects which run's adapter is folded in and where the merged model
    lands. It defaults to the unified scope, whose paths are unchanged — and it
    matters because two scoped runs can share a version, so an unscoped merge
    would fold whichever adapter happened to be staged at that tag.
    """
    return MergePlan(
        base_model=base_model,
        adapter=selected_checkpoint or paths.scoped_staging_adapter_dir(scope, version),
        output_dir=paths.staging_merged_model_dir(version, scope=scope),
        dtype=dtype,
        selected_checkpoint=selected_checkpoint,
    )


def merge(plan: MergePlan, *, dry_run: bool = False) -> str:
    """Execute a merge. Returns the output directory."""
    log.info("merging %s", plan.describe())
    if dry_run:
        return plan.output_dir

    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional heavy dep
        raise MergeError(
            'PEFT and torch are required to merge. Install the [train] extra on the pod: '
            'pip install -e ".[train]"'
        ) from exc

    raise NotImplementedError(  # pragma: no cover - Phase 8 GPU milestone
        "Wire PEFT merge_and_unload() here in the Phase 8 GPU milestone. Load the base in "
        f"{plan.dtype} — NOT 4-bit, which would lose the precision the adapter was trained to "
        "add — apply the adapter, merge_and_unload(), and save. One adapter, one merge: the v1 "
        "Foundation-then-per-type ordering does not apply (arch v2.1 §4.1)."
    )
