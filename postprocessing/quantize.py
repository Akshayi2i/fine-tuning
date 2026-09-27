"""Serving-format quantization (arch v2.1 §13a, SPEC_10).

**What this produces, and for what runtime.** The serving endpoint is vLLM. So
the formats this module produces are the ones vLLM loads natively:

    merged bf16 model (~16 GB)
        │
        ├── served as-is                     — the reference, and cycle 1's format
        ├── FP8 W8A8 via llm-compressor      — ~8 GB, the default from cycle 2
        └── AWQ INT4 via llm-compressor      — VRAM-constrained serving only

**Why GGUF stopped being the target.** GGUF is llama.cpp's format. v1 produced
``fp16`` + ``q5_k_m`` GGUF files every cycle while the endpoint ran vLLM, which
cannot load them — so the pipeline spent conversion and eval compute on an
artifact nothing could deploy, and there was no small format the server *could*
open. bf16 or nothing.

v1's own docstring had the right of it: *"GGUF is the portable/edge path and vLLM
on the merged model is the primary one."* It simply never followed through, and
left ``DEFAULT_FORMATS = ("fp16", "q5_k_m")`` making a GGUF the per-cycle
deliverable. GGUF is now an **on-request export** for an offline or edge
deployment, validated separately in llama.cpp, never on the serving path.

**The rule that constrains every format.** Quantization applies to the LANGUAGE
DECODER only. The vision encoder, every merger and ``lm_head`` stay bf16 (§13a).
Compressing the vision path produces a model that loads cleanly and reads pages
badly — very hard to notice, because it still emits well-formed JSON. This is the
same failure the mmproj guard below refuses to ship, arriving by a different
route.

**Nothing is served on a format that has not been gated.** Every serving format
gets its own calibrator fit and its own gate run (§13a), because quantization
degrades exactly what was fine-tuned in: an FP8 model inherits nothing from
bf16's result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from artifact_registry import paths

log = logging.getLogger(__name__)

#: Formats the vLLM serving endpoint loads natively.
ServingFormat = Literal["bf16", "fp8", "awq_int4"]

SERVING_FORMATS: tuple[ServingFormat, ...] = ("bf16", "fp8", "awq_int4")

#: bf16 is the reference every quantized format's drop is measured against, and
#: it is cycle 1's serving format — FP8 is the plan until Phase 0 spike item 9
#: confirms a decoder-only FP8 export loads and runs in vLLM.
DEFAULT_FORMATS: tuple[ServingFormat, ...] = ("bf16",)

#: What to produce once FP8 is verified. Named rather than inlined so the switch
#: is one constant, not an edit across the orchestration layer.
VERIFIED_FORMATS: tuple[ServingFormat, ...] = ("bf16", "fp8")

#: Approximate size relative to the bf16 merged model. Indicative — measure on
#: the real model rather than quoting these. AWQ INT4 is lower than a naive 1/4
#: because the vision tower and lm_head stay bf16 and are not compressed at all.
RELATIVE_SIZE: dict[ServingFormat, float] = {"bf16": 1.00, "fp8": 0.55, "awq_int4": 0.35}

#: Modules never quantized, in any format (§13a). Passed to llm-compressor as
#: its ignore list.
NEVER_QUANTIZED: tuple[str, ...] = ("visual.*", "*merger*", "*aligner*", "lm_head")

#: GGUF remains available as an edge export. Kept in this module rather than
#: deleted, because "we support an offline deployment" is a real capability —
#: it is simply not the serving path.
GgufFormat = Literal["fp16", "bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m"]
GGUF_FORMATS: tuple[GgufFormat, ...] = ("fp16", "bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m")


class QuantizationError(RuntimeError):
    """Raised when an export cannot be produced safely."""


@dataclass
class QuantizationPlan:
    """What a serving-format export run will produce."""

    merged_model: str
    formats: list[ServingFormat]
    output_dirs: dict[ServingFormat, str] = field(default_factory=dict)
    version: str = ""

    #: Set only once Phase 0 spike item 9 has confirmed a decoder-only FP8 export
    #: loads and runs in vLLM. Defaulting to ``False`` keeps cycle 1 on bf16,
    #: which is the safe direction: an unverified serving format is a deployment
    #: that fails at load, or worse, one that loads and reads badly.
    fp8_verified: bool = False

    @property
    def quantized_formats(self) -> list[ServingFormat]:
        """Everything except the bf16 reference — the formats actually compressed."""
        return [f for f in self.formats if f != "bf16"]

    def estimated_sizes_gb(self, bf16_size_gb: float = 16.0) -> dict[ServingFormat, float]:
        return {f: round(bf16_size_gb * RELATIVE_SIZE[f], 1) for f in self.formats}

    def describe(self) -> str:
        sizes = self.estimated_sizes_gb()
        return ", ".join(f"{f} (~{sizes[f]}GB)" for f in self.formats)


def plan_quantization(
    *,
    version: str,
    formats: list[str] | None = None,
    fp8_verified: bool = False,
    scope: str | None = None,
) -> QuantizationPlan:
    """Assemble a serving-format export plan.

    ``scope`` addresses the scope's own merged model and outputs. Without it a
    scoped run quantized the unified model's path — which merge never wrote for
    that scope — and wrote its formats where ``_is_quantized`` never looks.

    Defaults to bf16 alone. That is not conservatism for its own sake: FP8 is
    unverified for this model until the Phase 0 spike runs, and a serving format
    nobody has loaded is not a serving format.
    """
    requested = [f.strip().lower() for f in (formats or DEFAULT_FORMATS)]

    gguf_asked_for = [f for f in requested if f in GGUF_FORMATS and f not in SERVING_FORMATS]
    if gguf_asked_for:
        raise QuantizationError(
            f"{gguf_asked_for} are GGUF formats, which llama.cpp loads and vLLM does not. The "
            "serving endpoint runs vLLM, so producing these per cycle spends conversion and "
            "eval compute on an artifact nothing can deploy (arch v2.1 §13a). Use "
            f"{list(SERVING_FORMATS)} for serving, or plan_gguf_export() for an offline build."
        )

    unknown = [f for f in requested if f not in SERVING_FORMATS]
    if unknown:
        raise QuantizationError(
            f"unknown serving format(s) {unknown}; expected any of {list(SERVING_FORMATS)}"
        )

    resolved: list[ServingFormat] = [f for f in SERVING_FORMATS if f in requested]
    if "bf16" not in resolved:
        # Every §13b threshold is an ABSOLUTE margin against bf16, so a run that
        # produces only a quantized format has nothing to measure its own drop
        # from — and the gate would have no reference to compare against.
        raise QuantizationError(
            f"formats {resolved} omit the bf16 reference. Every quantization threshold in "
            "§13b is an absolute margin against bf16, so without it a quantized format "
            "cannot be validated at all."
        )

    return QuantizationPlan(
        merged_model=paths.staging_merged_model_dir(version, None, scope=scope),
        formats=resolved,
        output_dirs={
            f: paths.staging_quantized_model_dir(version, f, scope=scope)
            for f in resolved if f != "bf16"
        },
        version=version,
        fp8_verified=fp8_verified,
    )


def quantize(plan: QuantizationPlan, *, dry_run: bool = False) -> dict[str, str]:
    """Produce the requested serving formats. Returns format -> directory.

    bf16 is the merged model itself and is never re-exported — a copy would be a
    second 16 GB artifact identical to the first.
    """
    if "fp8" in plan.formats and not plan.fp8_verified:
        raise QuantizationError(
            "FP8 is not verified for this model. Phase 0 spike item 9 exists to confirm that a "
            "decoder-only FP8 export — with visual.*, the mergers and lm_head excluded — loads "
            "and runs in vLLM. Until it has, cycle 1 serves the bf16 merged model, which is "
            "slower and certain rather than fast and unproven."
        )

    produced: dict[str, str] = {"bf16": plan.merged_model}
    log.info("serving formats for %s: %s", plan.version, plan.describe())
    if dry_run:
        produced.update({str(k): v for k, v in plan.output_dirs.items()})
        return produced

    for fmt in plan.quantized_formats:
        produced[fmt] = _export(plan, fmt)
    return produced


def _export(plan: QuantizationPlan, fmt: ServingFormat) -> str:  # pragma: no cover - needs a GPU
    """One llm-compressor export, decoder only."""
    try:
        import llmcompressor  # noqa: F401
    except ImportError as exc:
        raise QuantizationError(
            'llm-compressor is required for FP8 and AWQ export. Install the [serve] extra on '
            'the pod: pip install -e ".[serve]"'
        ) from exc

    raise NotImplementedError(
        f"Wire llm-compressor's {fmt} recipe here in the Phase 8 GPU milestone. "
        f"QuantizationModifier over {plan.merged_model}, with ignore={list(NEVER_QUANTIZED)} — "
        "the vision encoder, every merger and lm_head stay bf16, because compressing the vision "
        "path produces a model that loads cleanly and reads pages badly while still emitting "
        f"well-formed JSON. Save to {plan.output_dirs.get(fmt)}, then load it in vLLM before "
        "anything downstream depends on it."
    )


# --------------------------------------------------------------------------
# GGUF — the edge export, on request only
# --------------------------------------------------------------------------

@dataclass
class GgufExportPlan:
    """An offline/edge build. Never the serving path (arch v2.1 §13a)."""

    merged_model: str
    formats: list[GgufFormat]
    output_dirs: dict[GgufFormat, str] = field(default_factory=dict)
    version: str = ""
    mmproj_verified: bool = False


def plan_gguf_export(
    *,
    version: str,
    formats: list[str],
    mmproj_verified: bool = False,
) -> GgufExportPlan:
    """Plan an on-request GGUF build for an offline or edge deployment.

    Not produced per cycle and not validated by the serving gate: a GGUF is
    validated separately in llama.cpp, against the §13b AWQ INT4 column.
    """
    requested = [f.strip().lower() for f in formats]
    unknown = [f for f in requested if f not in GGUF_FORMATS]
    if unknown:
        raise QuantizationError(
            f"unknown quantization format {unknown[0]!r}; expected one of {list(GGUF_FORMATS)}"
        )
    resolved: list[GgufFormat] = [f for f in GGUF_FORMATS if f in requested]
    return GgufExportPlan(
        merged_model=paths.staging_merged_model_dir(version, None),
        formats=resolved,
        # runtime is explicit: bf16 is also a serving format, and inferring the
        # runtime from the name wrote a bf16 GGUF into the vLLM serving directory.
        output_dirs={
            f: paths.staging_quantized_model_dir(version, f, runtime="gguf") for f in resolved
        },
        version=version,
        mmproj_verified=mmproj_verified,
    )


def export_gguf(plan: GgufExportPlan, *, dry_run: bool = False) -> dict[str, str]:
    """Export GGUF for an offline build.

    The mmproj guard is unchanged from v1 and stays unchanged: Qwen3-VL is
    multimodal, so a GGUF needs the quantized LLM **plus** a separate ``mmproj``
    file for the vision encoder and projector. Exporting without a verified one
    produces a model that loads and cannot see, which fails silently on every
    image-only document.
    """
    if not plan.mmproj_verified:
        raise QuantizationError(
            "Qwen3-VL is multimodal, so a GGUF export needs an mmproj file for the vision "
            "encoder and projector — and llama.cpp support for it is unverified (arch v2.1 "
            "§13a). Exporting without one produces a model that loads and cannot see, which "
            "fails silently on every image-only document. Confirm support in the Phase 0 spike "
            "and set mmproj_verified=True. Nothing on the serving path needs this: vLLM loads "
            "the bf16 merged model and, once verified, FP8."
        )
    log.info("GGUF edge export %s -> %s", plan.merged_model, plan.formats)
    if dry_run:
        return {str(k): v for k, v in plan.output_dirs.items()}

    raise NotImplementedError(  # pragma: no cover - needs llama.cpp
        "Wire llama.cpp here: convert_hf_to_gguf.py for the base conversion, llama-quantize "
        "per format, and the mmproj export alongside. Validate the result IN llama.cpp against "
        "the §13b AWQ INT4 column — the serving gate does not cover it, because the serving "
        "endpoint never loads it."
    )
