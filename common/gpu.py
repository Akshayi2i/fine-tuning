"""GPU only: model compute never falls back to the CPU.

Every step that runs a model — training, the merge, quantization, vLLM
generation (checkpoint selection, calibration, the golden eval, serving), the
HF fallback backend, MinerU OCR — calls :func:`require_cuda` before it loads
anything. A CPU fallback is never a slower success here: an 8B vision model on a
CPU turns an hour into days, and a job that quietly took that path would look
healthy while it ran. So the step refuses, with the reason, at once.

What stays on the CPU is work with no GPU form at all — reading and writing
JSON, tokenizer counts, downloads, the corpus build's bookkeeping — and it is
not where the time goes.

Also the cheap, import-free probe :func:`gpu_present`, used to recognise the pod
without loading torch.
"""

from __future__ import annotations

import shutil
from pathlib import Path


class GPUError(RuntimeError):
    """Raised when model compute would run without a CUDA device."""


def gpu_present() -> bool:
    """Whether this machine has an NVIDIA GPU, without importing torch."""
    return Path("/dev/nvidia0").exists() or shutil.which("nvidia-smi") is not None


def require_cuda(what: str) -> str:
    """Refuse ``what`` unless torch sees a CUDA device. Returns the device's name."""
    try:
        import torch
    except ImportError as exc:
        raise GPUError(
            f"{what} runs on the GPU only, and torch is not installed here. Run it on the pod "
            "after `bash scripts/setup_pod.sh <role>`."
        ) from exc
    if not torch.cuda.is_available():
        hint = (
            "torch is a CPU-only build — reinstall with scripts/setup_pod.sh"
            if torch.version.cuda is None
            else "no CUDA device is visible (check nvidia-smi, and that CUDA_VISIBLE_DEVICES "
            "is not empty)"
        )
        raise GPUError(
            f"{what} runs on the GPU only and will not fall back to the CPU, which would take "
            f"days instead of hours: {hint}."
        )
    return torch.cuda.get_device_name(0)
