"""Latency and GPU memory, measured where a release is evaluated (Fideon SPEC_09 handoff item 6).

Recorded in the eval report and copied into the release bundle; not gated
until SPEC_06 sets a target. Measured during the golden eval, which serves
every eval document through :func:`serving.pipeline.extract` on the target
GPU: the P95 of every generation call - one window of a policy or a Loss Run,
or a whole short document - per adapter, and the GPU memory in use at the end.

What this is not: a load test with every adapter of every promoted release
loaded at once. The eval serves this release's merged model alone; the bundle
says so (``measured_with``).
"""

from __future__ import annotations

import logging
import math
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

log = logging.getLogger(__name__)

#: How a call served by the base model, with no LoRA, is labelled.
BASE_MODEL = "base model"


def p95(values: Sequence[float]) -> float | None:
    """The 95th percentile by the nearest-rank method; None for no values."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def latency_by_adapter(scored: Sequence[tuple[Any, Any, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Per adapter: the P95 latency of one call in milliseconds, and how many calls."""
    calls: dict[str, list[float]] = defaultdict(list)
    for _expected, _got, metadata in scored:
        adapter = metadata.get("adapter") or BASE_MODEL
        calls[adapter].extend(float(v) for v in metadata.get("window_latencies_ms") or [] if v is not None)
    return {adapter: {"p95_ms": round(p95(values), 1), "calls": len(values)}
            for adapter, values in sorted(calls.items()) if values}


def peak_gpu_memory_mb() -> float | None:
    """The most memory in use on any GPU, from ``nvidia-smi``; None without one.

    vLLM reserves its share of the card when it starts (gpu_memory_utilization),
    so this reads that reservation plus whatever else the process holds.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("GPU memory not measured: %s", exc)
        return None
    readings = [float(line) for line in out.split() if line.strip().replace(".", "", 1).isdigit()]
    return max(readings) if readings else None


def measurements(scored: Sequence[tuple[Any, Any, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "latency_p95_ms_by_adapter": latency_by_adapter(scored),
        "peak_gpu_memory_mb": peak_gpu_memory_mb(),
        "measured_with": "the golden eval: this release's merged bf16 model alone, on the target GPU",
    }
