"""How the frozen base is held during training — bf16 by default, 4-bit on request.

**The default is plain LoRA on a bf16 base, not QLoRA** (arch §9). The reason is
train/serve alignment: *both* serving paths hold the base in bf16/fp16 — the
merged model (SPEC_10) and vLLM's LoRA hot-swap (SPEC_11). An adapter trained
against a 4-bit base learns a delta that partly compensates for quantization
error in weights it is then never served against. Training in bf16 removes that
gap entirely: you train on exactly the weights you serve.

4-bit NF4 stays available behind ``quantization.load_in_4bit`` in
``configs/base_model.yaml``, because it is the right answer on a VRAM-constrained
pod. It saves roughly 11 GB of base weights (~16 GB → ~5.5 GB) and costs the
mismatch above plus ~20-40% slower steps, since NF4 weights dequantize on every
matmul.

This module exists so the two trainers cannot drift apart on the question. It was
previously duplicated in both, with ``quantization_bit`` hardcoded to ``4`` in
each while ``load_in_4bit`` sat in the YAML being read by nothing — so the
manifest recorded a setting that no run could actually vary.
"""

from __future__ import annotations

from typing import Any, Literal

Technique = Literal["LoRA", "QLoRA"]

#: What the manifest records when the base is held in bf16. Not a free-text
#: label: `evaluation` and the promotion gate compare runs on it, so a run
#: trained in 4-bit and one trained in bf16 must never describe themselves
#: the same way.
BF16_DESCRIPTOR = "bf16_frozen_base"


def loads_in_4bit(base: dict[str, Any]) -> bool:
    """Whether this run holds the base in 4-bit. Default False (arch §9)."""
    return bool(base.get("quantization", {}).get("load_in_4bit", False))


def technique(base: dict[str, Any]) -> Technique:
    """``"QLoRA"`` when the base is quantized, ``"LoRA"`` when it is bf16."""
    return "QLoRA" if loads_in_4bit(base) else "LoRA"


def swift_quantization_args(base: dict[str, Any]) -> dict[str, Any]:
    """ms-swift arguments for base-weight precision.

    ms-swift 3 names: ``quant_method`` + ``quant_bits``. The 2.x
    ``quantization_bit`` is not an ms-swift 3 argument, and its parser refuses an
    unknown flag, so passing it killed every run at argument parsing.

    bf16 emits NOTHING. ms-swift 3's ``quant_bits`` defaults to None — no
    quantization — and has no "0" value to state it explicitly with.

    The ``bnb_4bit_*`` arguments are emitted **only** under 4-bit, so the rendered
    command, and therefore the run log, never carries settings that describe
    nothing the run did.
    """
    if not loads_in_4bit(base):
        return {}

    quant = base["quantization"]
    return {
        "quant_method": "bnb",
        "quant_bits": 4,
        "bnb_4bit_quant_type": quant["bnb_4bit_quant_type"],
        "bnb_4bit_use_double_quant": quant["bnb_4bit_use_double_quant"],
        "bnb_4bit_compute_dtype": quant["bnb_4bit_compute_dtype"],
    }


def manifest_descriptor(base: dict[str, Any]) -> str:
    """The ``TrainingConfig.base_quantization`` value for this run.

    Built from the config that actually ran, so a manifest cannot claim NF4
    double-quant while the run held the base in bf16.
    """
    if not loads_in_4bit(base):
        return BF16_DESCRIPTOR

    quant = base["quantization"]
    double = "double_quant" if quant["bnb_4bit_use_double_quant"] else "single_quant"
    return f"{quant['bnb_4bit_quant_type']}_{double}_{quant['bnb_4bit_compute_dtype']}_compute"
