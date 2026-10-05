"""MinerU for one request, within a time budget (Fideon SPEC_05 §4).

The serving-side entry point. The caller passes the document's modality and
``force_ocr`` as its routing stage decided them; nothing is detected here. A
native document reads within ``NATIVE_TIMEOUT_S``; a scanned one, or a native
one forced through OCR because its figures are drawn (§4.1), within
``SCANNED_TIMEOUT_S`` - it is doing the scanned path's work. Past its budget,
or on any failure, the call raises :class:`MinerUError` and the document goes
to review (§9).

This repository's own serving endpoint runs no OCR: its requests arrive with
their page texts (``serving.vllm_entrypoint``). This is what produces them
where a PDF arrives instead.

**A timeout stops the wait, not MinerU.** The call runs in a worker thread, and
a thread cannot be interrupted: past its budget the request fails at once, but
MinerU finishes in the background and its result is dropped. A caller that
must reclaim the GPU at the deadline runs MinerU in a worker process.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

from data_pipeline.ocr.modality import MODALITIES, NATIVE
from data_pipeline.ocr.run_mineru import MinerUEngine, OcrEngine, OcrError, PageOutput

NATIVE_TIMEOUT_S = 30
SCANNED_TIMEOUT_S = 90   # OCR is slower; a GPU brings it nearer 20 s

#: Fewer characters than this across the document is no reading at all (§9).
MIN_TEXT_CHARS = 10


class MinerUError(RuntimeError):
    """MinerU failed, timed out or read nothing; the document goes to review."""


@dataclass
class MinerUOutput:
    pages: list[PageOutput]
    path_used: Literal["native", "scanned"]
    duration_ms: int
    errors: list[str] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return len(self.pages)


async def run_mineru(
    pdf_bytes: bytes,
    modality: str,
    force_ocr: bool = False,
    *,
    engine: OcrEngine | None = None,
    max_long_side_px: int | None = None,
) -> MinerUOutput:
    """Read one PDF with MinerU on the path its modality and ``force_ocr`` choose."""
    if modality not in MODALITIES:
        raise MinerUError(f"modality {modality!r} is neither of {list(MODALITIES)}")
    native = modality == NATIVE and not force_ocr
    timeout = NATIVE_TIMEOUT_S if native else SCANNED_TIMEOUT_S
    if max_long_side_px is None:
        from common.config import resolution_cap_px

        max_long_side_px = resolution_cap_px()
    engine = engine or MinerUEngine()

    started = time.monotonic()
    try:
        pages = await asyncio.wait_for(
            asyncio.to_thread(engine.process, pdf_bytes, device="cuda",
                              max_long_side_px=max_long_side_px,
                              modality=modality, force_ocr=force_ocr),
            timeout=timeout,
        )
    except TimeoutError:
        raise MinerUError(
            f"MinerU timed out after {timeout}s on the {'native' if native else 'scanned'} path"
        ) from None
    except OcrError as exc:
        raise MinerUError(str(exc)) from exc
    if sum(len(page.markdown.strip()) for page in pages or []) < MIN_TEXT_CHARS:
        raise MinerUError(f"MinerU read fewer than {MIN_TEXT_CHARS} characters from the document")
    return MinerUOutput(
        pages=list(pages),
        path_used="native" if native else "scanned",
        duration_ms=int((time.monotonic() - started) * 1000),
        errors=[f"page {p.page_number}: nothing read" for p in pages if p.ocr_failed],
    )
