"""Phase 0 dependency spike — run this on a throwaway RunPod pod.

Answers, in one command, every question the build is currently guessing at. Each
one de-risks a decision that is expensive to reverse later:

======================================  ===================================================
check                                   what its answer changes
======================================  ===================================================
CUDA and GPU class                      whether anything below can run at all
ms-swift + Qwen3-VL multimodal LoRA     the Layer-3 entrypoint choice (arch §10). If it
                                        fails, the documented fallback is TRL SFTTrainer
flash-attn                              install pain is the usual blocker; sdpa is the
                                        fallback, at a speed cost
interleaved image/text content          the corpus row format (SPEC_07). If a trainer or
                                        server rejects it, page pairing needs the
                                        single-block fallback with inline markers
vLLM multi-LoRA hot-swap                SPEC_11's serving design. If it fails, serve merged
                                        per-type models instead
MinerU on CUDA                          the whole data pipeline; GPU is the default mode
llama.cpp mmproj for Qwen3-VL           whether GGUF export can produce a model that can
                                        see. Until then, quantization stays deferred
======================================  ===================================================

**Every check is independent and none aborts the run.** A spike that stops at the
first failure tells you one thing; this tells you all of them in one pod-hour.

    pip install -e ".[data,train,serve]"        # expect flash-attn to be the hard part
    python scripts/phase0_spike.py --pdf sample.pdf --out spike_report.json

The report is written as JSON and printed as prose. Paste the prose back.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class Result:
    """One check. ``ok=None`` means it could not be determined, which is not a pass."""

    name: str
    ok: bool | None = None
    detail: str = ""
    decides: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "UNKNOWN" if self.ok is None else ("PASS" if self.ok else "FAIL")


def run(name: str, decides: str, fn) -> Result:
    """Run one check, converting any exception into a reported failure."""
    result = Result(name=name, decides=decides)
    started = time.perf_counter()
    try:
        fn(result)
    except Exception as exc:  # noqa: BLE001 - a check that raises is a check that failed
        result.ok = False
        result.detail = f"{type(exc).__name__}: {exc}"
        result.data["traceback"] = traceback.format_exc(limit=4)
    result.data["seconds"] = round(time.perf_counter() - started, 1)
    print(f"  [{result.status:7}] {name}: {result.detail[:150]}")
    return result


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_gpu(r: Result) -> None:
    import torch

    r.data["torch"] = torch.__version__
    if not torch.cuda.is_available():
        r.ok = False
        r.detail = "no CUDA device visible — nothing below can be trusted"
        return
    r.ok = True
    r.data["gpu"] = torch.cuda.get_device_name(0)
    r.data["count"] = torch.cuda.device_count()
    r.data["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    r.detail = f"{r.data['gpu']} x{r.data['count']}, {r.data['vram_gb']}GB, torch {torch.__version__}"


def check_flash_attn(r: Result) -> None:
    try:
        import flash_attn

        r.ok = True
        r.detail = f"flash-attn {getattr(flash_attn, '__version__', 'installed')}"
    except ImportError as exc:
        # Not fatal: sdpa works. Recorded because it is the usual install failure
        # and its absence costs throughput, which changes pod-hour estimates.
        r.ok = False
        r.detail = f"not importable ({exc}); fall back to attn_implementation='sdpa' and note the cost"


def check_ms_swift(r: Result) -> None:
    import shutil

    if shutil.which("swift") is None:
        r.ok = False
        r.detail = "the `swift` CLI is not on PATH; install the [train] extra"
        return

    out = subprocess.run(["swift", "sft", "--help"], capture_output=True, text=True, timeout=120)
    help_text = (out.stdout + out.stderr).lower()
    r.data["exit_code"] = out.returncode

    # The three arguments the training entrypoints actually emit. A missing one
    # means the config this repo renders would not be accepted.
    required = {
        "--dataset": "--dataset" in help_text,
        "--val_dataset": "val_dataset" in help_text,
        "--freeze_vit": "freeze_vit" in help_text,
        "--lora_target_modules": "lora_target_modules" in help_text,
    }
    r.data["arguments"] = required
    missing = [k for k, present in required.items() if not present]
    r.ok = not missing
    r.detail = (
        "ms-swift accepts every argument the trainer emits"
        if r.ok else f"ms-swift does not accept {missing} — check the fallback in arch §10"
    )


def check_model_loads(r: Result) -> None:
    """Config-only, not weights. Confirms the model id resolves and the class exists."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    r.ok = True
    r.data["architectures"] = getattr(config, "architectures", None)
    r.data["model_type"] = getattr(config, "model_type", None)
    r.detail = f"{r.data['model_type']} / {r.data['architectures']}"


def check_interleaved_content(r: Result) -> None:
    """Does the processor accept image, text, image, text in one user turn?

    This is the corpus row format. If a trainer or server flattens or rejects it,
    page pairing falls back to one text block with inline `<page N of M>` markers
    — which fixes the split-table problem but not the image/text correspondence.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    messages = [
        {"role": "system", "content": "extract"},
        {"role": "user", "content": [
            {"type": "image", "image": "page_1.png"},
            {"type": "text", "text": "<page 1 of 2>\n\nfirst page"},
            {"type": "image", "image": "page_2.png"},
            {"type": "text", "text": "<page 2 of 2>\n\nsecond page"},
        ]},
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    r.data["rendered_chars"] = len(rendered)
    r.data["image_placeholders"] = rendered.count("<|image_pad|>") or rendered.count("<image>")
    # Both pages' text must survive, and in order.
    first, second = rendered.find("first page"), rendered.find("second page")
    r.ok = first >= 0 and second > first
    r.detail = (
        f"interleaved content preserved in order ({r.data['image_placeholders']} image slots)"
        if r.ok else "the template dropped or reordered interleaved blocks — see the fallback above"
    )
    r.data["sample"] = rendered[:400]


def check_vllm_multi_lora(r: Result) -> None:
    """Does vLLM support multi-LoRA for this architecture?

    Checked by capability, not by serving: loading an 8B model twice is not worth
    a pod-hour when the question is whether the feature exists for this model class.
    """
    import vllm
    from vllm.lora.request import LoRARequest  # noqa: F401

    r.data["vllm"] = vllm.__version__
    from vllm import EngineArgs

    fields = set(getattr(EngineArgs, "__dataclass_fields__", {}))
    r.data["enable_lora"] = "enable_lora" in fields
    r.data["max_loras"] = "max_loras" in fields
    r.ok = r.data["enable_lora"] and r.data["max_loras"]
    r.detail = (
        f"vLLM {vllm.__version__} exposes enable_lora/max_loras — confirm hot-swap on a real "
        "adapter before relying on it" if r.ok
        else f"vLLM {vllm.__version__} lacks multi-LoRA args; serve merged per-type models (SPEC_11)"
    )


def _mineru(pdf: Path) -> tuple[str, float]:
    """Run MinerU once on GPU. Returns (markdown, seconds).

    No device parameter: OCR is GPU-only, because MinerU's CPU path uses lighter
    model variants and produces different markdown from the same PDF (arch §8a).
    """
    from data_pipeline.ocr.run_mineru import MinerUEngine

    started = time.perf_counter()
    pages = MinerUEngine().process(pdf.read_bytes(), device="cuda", max_long_side_px=1792)
    return "\n\n".join(p.markdown for p in pages), time.perf_counter() - started


def check_mineru_gpu(pdf: Path):
    def _check(r: Result) -> None:
        markdown, seconds = _mineru(pdf)
        r.ok = bool(markdown.strip())
        r.data["seconds"] = round(seconds, 1)
        r.data["chars"] = len(markdown)
        r.detail = f"{len(markdown)} chars in {seconds:.1f}s on cuda"
    return _check


def check_llama_cpp_mmproj(r: Result) -> None:
    """Can llama.cpp export a vision projector for this architecture?

    Without a working mmproj a GGUF loads and cannot see, failing silently on
    every image-only document. Checked by inspecting the converter's supported
    architectures rather than by running a 16GB conversion.
    """
    import shutil

    # Bounded search. Globbing from the filesystem root walks every mounted
    # volume and dies on the first unreadable directory — which is what it did.
    converter = shutil.which("convert_hf_to_gguf.py")
    if converter is None:
        for root in (Path.cwd(), Path.home(), Path("/workspace"), Path("/opt"), Path("/usr/local")):
            try:
                found = next(root.glob("*/convert_hf_to_gguf.py"), None) or next(
                    root.glob("*/*/convert_hf_to_gguf.py"), None
                )
            except (OSError, PermissionError):
                continue
            if found:
                converter = str(found)
                break
    if converter is None:
        r.ok = None
        r.detail = (
            "llama.cpp not found (searched PATH, cwd, home, /workspace, /opt, /usr/local) — "
            "clone it to answer this, or leave GGUF deferred"
        )
        return

    text = Path(converter).read_text(encoding="utf-8", errors="replace").lower()
    r.data["mentions_qwen3vl"] = "qwen3vl" in text or "qwen3_vl" in text
    r.data["mentions_mmproj"] = "mmproj" in text
    r.ok = r.data["mentions_qwen3vl"] and r.data["mentions_mmproj"]
    r.detail = (
        "the converter references Qwen3-VL and mmproj — attempt a real export next"
        if r.ok else
        "no Qwen3-VL + mmproj support found; GGUF stays deferred and vLLM on the merged "
        "model remains the serving path (arch §13a)"
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def render(results: list[Result]) -> str:
    lines = [
        "PHASE 0 DEPENDENCY SPIKE",
        f"  {datetime.now(UTC).isoformat()}  ·  python {platform.python_version()}  ·  {platform.platform()}",
        "",
    ]
    for r in results:
        lines.append(f"[{r.status:7}] {r.name}")
        lines.append(f"          {r.detail}")
        if r.decides:
            lines.append(f"          decides: {r.decides}")
    failed = [r for r in results if r.ok is False]
    unknown = [r for r in results if r.ok is None]

    lines += ["", f"{len(results) - len(failed) - len(unknown)} passed, "
                  f"{len(failed)} failed, {len(unknown)} undetermined."]
    blocking = [r for r in failed if r.name == "cuda_available"]
    recoverable = [r for r in failed if r not in blocking]
    if blocking:
        lines += ["", "BLOCKING — nothing below this can be trusted:"]
        lines += [f"  - {r.name}: {r.detail}" for r in blocking]
    if recoverable:
        lines += ["", "Each of these has a documented fallback — none stops the project:"]
        lines += [f"  - {r.name}: {r.decides}" for r in recoverable]
    if unknown:
        lines.append("")
        lines.append("Undetermined is not a pass. These still need an answer:")
        lines += [f"  - {r.name}" for r in unknown]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 dependency spike")
    parser.add_argument("--pdf", type=Path, help="a real insurance PDF for the MinerU checks")
    parser.add_argument("--out", type=Path, default=Path("spike_report.json"))
    parser.add_argument("--skip-mineru", action="store_true")
    args = parser.parse_args(argv)

    print("Phase 0 spike — every check is independent; none aborts the run.\n")
    results = [
        run("cuda_available", "whether anything below can run", check_gpu),
        run("flash_attn", "throughput; sdpa is the fallback", check_flash_attn),
        run("ms_swift_arguments", "the Layer-3 entrypoint; TRL SFTTrainer is the fallback (arch §10)",
            check_ms_swift),
        run("model_config_resolves", "the pinned model id and revision", check_model_loads),
        run("interleaved_image_text", "the corpus row format; single-block + markers is the fallback",
            check_interleaved_content),
        run("vllm_multi_lora", "SPEC_11 serving; merged per-type models are the fallback",
            check_vllm_multi_lora),
        run("llama_cpp_mmproj", "whether GGUF export can produce a model that can see",
            check_llama_cpp_mmproj),
    ]

    if args.pdf and not args.skip_mineru:
        results.append(run("mineru_cuda", "the whole data pipeline", check_mineru_gpu(args.pdf)))
        # No device-parity check: OCR is GPU-only, so there is no second device
        # whose output could differ. What was a measurement is now a constraint.
    else:
        print("  [SKIPPED] MinerU checks — pass --pdf to answer the ocr_device pin question")

    report = render(results)
    print("\n" + report)

    args.out.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "results": [asdict(r) | {"status": r.status} for r in results],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # Non-zero only when a check genuinely failed. An undetermined check is
    # reported loudly but does not fail the run, because the usual cause is a
    # tool that is simply not installed on this pod yet.
    return 1 if any(r.ok is False for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
