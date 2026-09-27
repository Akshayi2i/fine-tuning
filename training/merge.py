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


def _split_ref(base_model: str) -> tuple[str, str | None]:
    """``Qwen/Qwen3-VL-8B-Instruct@<sha>`` -> ``(id, sha)``."""
    model_id, _, revision = base_model.partition("@")
    return model_id, revision or None


def merge(plan: MergePlan, *, dry_run: bool = False) -> str:
    """Execute a merge. Returns the output directory.

    Written to ``<output>.partial`` and renamed into place only once the saved
    model has its config and weights, so an interrupted merge never leaves a
    directory that ``_is_merged`` counts as done.
    """
    log.info("merging %s", plan.describe())
    if dry_run:
        return plan.output_dir

    import shutil
    from pathlib import Path

    adapter = Path(plan.adapter)
    if not (adapter / "adapter_config.json").is_file():
        raise MergeError(
            f"{adapter} holds no adapter_config.json, so there is no adapter to merge. Point "
            "the merge at a checkpoint directory ms-swift wrote (checkpoint-N), not its root."
        )
    from common.config import base_model_dir

    model_id, revision = _split_ref(plan.base_model)
    local = base_model_dir()
    if local is None and revision in (None, "", "PIN_ME"):
        raise MergeError(
            f"the base model revision is {revision!r}. Merging into a floating revision folds "
            "the adapter into weights that may not be the ones it trained against; pin "
            "configs/base_model.yaml to a commit SHA first."
        )

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional heavy dep
        raise MergeError(
            'PEFT, torch and transformers are required to merge. Install the [train] extra on '
            'the pod: pip install -e ".[train]"'
        ) from exc

    output = Path(plan.output_dir)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)

    # The base in the merge dtype, never 4-bit: merging into quantized weights
    # loses the precision the adapter was trained to add. On the CPU, because a
    # merge is arithmetic, not inference, and the GPU may still hold an engine.
    dtype = torch.bfloat16 if plan.dtype == "bf16" else torch.float16
    # The pod's local copy when there is one; the pinned Hub revision otherwise.
    source, source_revision = (str(local), None) if local is not None else (model_id, revision)
    base = AutoModelForImageTextToText.from_pretrained(  # pragma: no cover - needs weights
        source, revision=source_revision, torch_dtype=dtype, device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(str(partial), safe_serialization=True, max_shard_size="5GB")
    # The processor travels with the weights: vLLM reads the chat template and the
    # image processor config from the model directory it is pointed at.
    AutoProcessor.from_pretrained(source, revision=source_revision).save_pretrained(str(partial))

    if not (partial / "config.json").is_file() or not any(partial.glob("*.safetensors")):
        raise MergeError(f"the merge wrote no config or weights to {partial}")
    if output.exists():
        shutil.rmtree(output)
    partial.rename(output)
    log.info("merged model saved to %s", output)
    return str(output)
