"""GGUF quantization export (SPEC_10, arch §13a).

A **configurable export step, not a fixed choice**. All formats derive from the
same fp16 GGUF conversion, so one or many can be produced in a single run.

**What is produced routinely:** supporting all six formats is a capability, not a
per-cycle obligation. The default is **fp16 as the accuracy baseline plus one
serving format** — regenerating and re-validating all six every cycle wastes eval
compute for formats nobody deploys.

**Multimodal caveat (arch §13a).** Qwen3-VL is a VLM, so a GGUF export needs the
quantized LLM **plus a separate ``mmproj`` file** for the vision encoder and
projector, served together. Whether llama.cpp supports this for Qwen3-VL is
**unverified** — which is exactly why GGUF is the portable/edge path and vLLM on
the merged fp16/bf16 model is the primary one. Emitting a GGUF without a
verified mmproj would produce a model that loads and cannot see.

Threshold validation is **deferred** this cycle (SPEC_10): nothing is served
quantized yet, so there is nothing to validate against. The thresholds live in
the spec until a format is actually deployed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from artifact_registry import paths

log = logging.getLogger(__name__)

Format = Literal["fp16", "bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m"]

ALL_FORMATS: tuple[Format, ...] = ("fp16", "bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m")

#: fp16 is the accuracy baseline every quantized variant is measured against;
#: q5_k_m is the default serving target when GGUF is eventually deployed.
DEFAULT_FORMATS: tuple[Format, ...] = ("fp16", "q5_k_m")

#: Approximate size relative to the fp16 merged model. Indicative only — measure
#: on the real model rather than quoting these.
RELATIVE_SIZE: dict[Format, float] = {
    "fp16": 1.00, "bf16": 1.00, "q8_0": 0.53,
    "q6_k": 0.44, "q5_k_m": 0.38, "q4_k_m": 0.32,
}


class QuantizationError(RuntimeError):
    """Raised when an export cannot be produced safely."""


@dataclass
class QuantizationPlan:
    """What an export run will produce."""

    merged_model: str
    formats: list[Format]
    output_dirs: dict[Format, str] = field(default_factory=dict)
    version: str = ""
    doc_type: str | None = None
    mmproj_required: bool = True
    mmproj_verified: bool = False

    def estimated_sizes_gb(self, fp16_size_gb: float = 16.0) -> dict[Format, float]:
        return {f: round(fp16_size_gb * RELATIVE_SIZE[f], 1) for f in self.formats}


def plan_quantization(
    *,
    version: str,
    formats: list[str] | None = None,
    doc_type: str | None = None,
    mmproj_verified: bool = False,
) -> QuantizationPlan:
    """Assemble an export plan.

    Args:
        mmproj_verified: set only once the Phase 0 spike has confirmed llama.cpp
            produces a working multimodal projector for Qwen3-VL. Defaulting to
            ``False`` is deliberate — the safe direction is refusing to ship a
            model that cannot see.
    """
    requested = [f.strip().lower() for f in (formats or DEFAULT_FORMATS)]
    unknown = [f for f in requested if f not in ALL_FORMATS]
    if unknown:
        raise QuantizationError(
            f"unknown quantization format(s) {unknown}; expected any of {list(ALL_FORMATS)}"
        )

    resolved: list[Format] = [f for f in ALL_FORMATS if f in requested]  # canonical order
    return QuantizationPlan(
        merged_model=paths.staging_merged_model_dir(version, doc_type),
        formats=resolved,
        output_dirs={f: paths.staging_quantized_model_dir(version, f, doc_type) for f in resolved},
        version=version,
        doc_type=doc_type,
        mmproj_verified=mmproj_verified,
    )


def quantize(plan: QuantizationPlan, *, dry_run: bool = False, allow_unverified_mmproj: bool = False) -> dict[Format, str]:
    """Export the requested formats. Returns format -> output directory.

    Pipeline: merged fp16/bf16 -> base GGUF via ``convert_hf_to_gguf.py`` ->
    ``llama-quantize`` per format.
    """
    if plan.mmproj_required and not plan.mmproj_verified and not allow_unverified_mmproj:
        raise QuantizationError(
            "Qwen3-VL is multimodal, so a GGUF export needs an mmproj file for the vision "
            "encoder and projector — and llama.cpp support for it is unverified (arch §13a). "
            "Exporting without one produces a model that loads and cannot see, which would fail "
            "silently on every image-only document. Confirm support in the Phase 0 spike and set "
            "mmproj_verified=True, or keep serving the merged fp16/bf16 model through vLLM, which "
            "is the primary path anyway."
        )

    if len(plan.formats) > len(DEFAULT_FORMATS):
        log.warning(
            "exporting %d formats. Producing and re-validating every format each cycle wastes "
            "eval compute — converge on fp16 plus one serving format and generate the rest on "
            "demand (arch §13a).", len(plan.formats),
        )

    log.info("quantizing %s -> %s", plan.merged_model, plan.formats)
    if dry_run:
        return dict(plan.output_dirs)

    raise NotImplementedError(
        "Wire llama.cpp here once the Phase 0 spike confirms Qwen3-VL GGUF support: "
        "convert_hf_to_gguf.py for the base conversion, llama-quantize per format, and the "
        "mmproj export alongside. Until then the serving path is vLLM on the merged model, and "
        "nothing downstream depends on this running."
    )
