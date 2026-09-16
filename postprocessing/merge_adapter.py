"""Merge a LoRA adapter into the base model (SPEC_10, arch §13 step 7).

PEFT's ``merge_and_unload()`` folds the adapter weights into the base, producing
one standalone model. Required before quantization, and it is what the first
serving cycle actually ships: **vLLM on the merged fp16/bf16 model** is the
primary path, with GGUF as the portable/edge option (arch §13a).

Loading the base here in bf16/fp16 is the *aligned* case, not a compromise: under
the arch §9 default the adapter was trained against a bf16 base, so merging into
bf16 applies it to exactly the weights it saw. That was not true while training
ran in 4-bit — the adapter then carried a delta partly compensating for
quantization error in weights it was never merged into. If a run sets
``load_in_4bit: true``, that gap comes back, and this merge is where it lands.

Writes to the **staging volume**, not to Blob. `package` (SPEC_13 command 2)
pushes it. The merged model is ~16 GB and quantization also runs on RunPod, so
pushing it to Azure here and pulling it back there is a 32 GB round trip for
nothing (master §12a).
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
    foundation_adapter: str
    type_adapter: str | None
    output_dir: str
    dtype: Dtype
    doc_type: str | None

    @property
    def is_unified(self) -> bool:
        """Foundation-only: one model serving every document type.

        Worth keeping available. At pilot volume a per-type adapter trains on ~25
        documents the Foundation already saw, so it may add nothing — and
        Foundation-only means one merged model instead of three (arch §4).
        """
        return self.type_adapter is None


def plan_merge(
    *,
    base_model: str,
    foundation_version: str,
    out_version: str,
    doc_type: str | None = None,
    adapter_version: str | None = None,
    dtype: Dtype = "fp16",
) -> MergePlan:
    """Assemble a merge plan without touching any weights."""
    return MergePlan(
        base_model=base_model,
        foundation_adapter=paths.staging_adapter_dir("foundation", foundation_version),
        type_adapter=(
            paths.staging_adapter_dir("doc_type", adapter_version or out_version, doc_type)
            if doc_type else None
        ),
        output_dir=paths.staging_merged_model_dir(out_version, doc_type),
        dtype=dtype,
        doc_type=doc_type,
    )


def merge(plan: MergePlan, *, dry_run: bool = False) -> str:
    """Execute a merge. Returns the output directory.

    Adapters are applied in order — Foundation first, then the per-type LoRA on
    top — because that is the order they were trained in. Reversing it would
    produce a different model from the one that was evaluated.
    """
    log.info(
        "merging %s + %s%s -> %s (%s)",
        plan.base_model, plan.foundation_adapter,
        f" + {plan.type_adapter}" if plan.type_adapter else "",
        plan.output_dir, plan.dtype,
    )
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

    raise NotImplementedError(
        "Wire PEFT merge_and_unload() here in the Phase 8 GPU milestone. Load the base in the "
        "target dtype (NOT 4-bit — merging into a quantized base loses precision the adapter "
        "was trained to add), apply the Foundation adapter, then the per-type adapter, then "
        "merge_and_unload() and save."
    )
