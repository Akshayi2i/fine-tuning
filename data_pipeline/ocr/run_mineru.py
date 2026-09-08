"""MinerU OCR and page rendering, GPU-accelerated (arch §14, §8a).

Reads ``raw-documents/``, writes ``processed/``. Never writes back to the raw
layer, and nothing downstream of here touches raw documents again.

Two things are non-negotiable:

**The resolution cap comes from config and is recorded.** Image token count is a
direct function of page resolution — the single biggest cost and latency lever
(arch §11) — and the cap must be *identical* here and at production inference. A
mismatch means serving the model a distribution it never trained on, which shows
up only as degraded accuracy with no error anywhere.

**The MinerU version and device are recorded on every document.** The model
learns how MinerU formats its output, so both are part of the corpus pin
(:mod:`~data_pipeline.ocr.mineru_version`).

The MinerU invocation is deliberately behind a swappable interface: it may run as
a library call or a subprocess, and the Phase 0 spike may replace it outright.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, for_ocr
from common.config import resolution_cap_px
from common.constants import ACTIVE_DOC_TYPES
from data_pipeline.ocr.mineru_version import Device, current_environment

log = logging.getLogger(__name__)


class OcrError(RuntimeError):
    """Raised when a document cannot be processed."""


@dataclass
class PageOutput:
    """One page's OCR text and rendered image."""

    page_number: int          # 1-based; ordering is positionally meaningful
    markdown: str
    image_bytes: bytes
    table_row_count: int = 0  # feeds the row-completeness signal (SPEC_09)
    ocr_failed: bool = False


class OcrEngine(Protocol):
    """The swappable OCR interface.

    Implemented by :class:`MinerUEngine` in production and by a stub in tests, so
    the whole pipeline is verifiable without MinerU installed.
    """

    def process(self, pdf_bytes: bytes, *, device: Device, max_long_side_px: int) -> list[PageOutput]:
        ...


class MinerUEngine:
    """Real MinerU. Imported lazily so CI needs neither MinerU nor CUDA."""

    def __init__(self, device: Device = "cuda") -> None:
        self.device = device

    def process(self, pdf_bytes: bytes, *, device: Device, max_long_side_px: int) -> list[PageOutput]:
        try:
            from magic_pdf.data.dataset import PymuDocDataset  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional heavy dep
            raise OcrError(
                "MinerU (magic-pdf) is not installed. It runs on GPU by default (arch §14); "
                'install the [data] extra and MinerU on the pod: pip install -e ".[data]"'
            ) from exc
        raise NotImplementedError(
            "Wire MinerU here once the Phase 0 spike confirms the GPU path, the model-weight "
            "download, and whether GPU and CPU output differ (mineru_version.compare_devices). "
            "The interface above is what the rest of the pipeline depends on — keep it stable."
        )


def count_table_rows(markdown: str) -> int:
    """Count markdown table body rows.

    Consumed by the list-completeness cross-check (SPEC_09): if the model extracts
    six claims from a page MinerU saw eight rows on, that is a recall failure the
    per-field confidence cannot see, because the missing rows generate no tokens.
    """
    rows = 0
    tables = 0
    in_table = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            in_table = False          # a non-table line ends the current table
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells if c):   # separator row
            continue
        if not in_table:
            # The first row of a run is that table's header.
            in_table = True
            tables += 1
        rows += 1
    # One header per table, not per page. Discounting a single header for the
    # whole page over-reported a two-table page by one row, which raises a false
    # row-completeness flag — and active learning treats that flag as an
    # unconditional override to full manual review.
    return max(0, rows - tables)


def process_document(
    doc_type: str,
    source_id: str,
    client: BlobClient,
    engine: OcrEngine,
    *,
    tenant_id: str | None = None,
    device: Device | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """OCR and render one document into ``processed/``.

    Idempotent: skips when output already exists **and** the source checksum,
    MinerU version, and device all still match. A change to any of the three
    forces reprocessing, because each one changes the input distribution the
    model was trained to arbitrate against.
    """
    env = current_environment(device, strict=device == "cuda")
    cap = resolution_cap_px()
    meta_key = paths.ocr_meta(doc_type, source_id, tenant_id)

    if not force and client.exists(meta_key):
        existing = client.read_json(meta_key)
        raw_meta = client.read_json(paths.raw_metadata(doc_type, source_id, tenant_id))
        unchanged = (
            # A render-only meta is written by render_only.py at this same key
            # and satisfies every check below, so a document rendered first was
            # skipped by OCR forever — no markdown was ever produced, and the
            # CLI reported it as processed.
            not existing.get("render_only")
            and existing.get("source_checksum") == raw_meta.get("checksum_sha256")
            and existing.get("mineru_version") == env.mineru_version
            and existing.get("ocr_device") == env.device
            and existing.get("resolution_cap_px") == cap
        )
        if unchanged:
            log.info("skipping %s: already processed and unchanged", source_id)
            return existing

    raw_meta = client.read_json(paths.raw_metadata(doc_type, source_id, tenant_id))
    pdf_bytes = client.read_bytes(paths.raw_pdf(doc_type, source_id, tenant_id))

    try:
        pages = engine.process(pdf_bytes, device=env.device, max_long_side_px=cap)
    except Exception as exc:
        raise OcrError(f"OCR failed for {source_id}: {exc}") from exc
    if not pages:
        raise OcrError(f"OCR produced no pages for {source_id}")

    for page in sorted(pages, key=lambda p: p.page_number):
        client.write_bytes(
            paths.processed_page(doc_type, source_id, page.page_number, "png", tenant_id),
            page.image_bytes,
        )
        client.write_text(
            paths.processed_page(doc_type, source_id, page.page_number, "md", tenant_id),
            page.markdown,
        )

    ocr_meta = {
        "source_id": source_id,
        "doc_type": doc_type,
        "tenant_id": paths._tenant(tenant_id),
        "page_count": len(pages),
        "mineru_version": env.mineru_version,
        "ocr_device": env.device,
        "gpu_name": env.gpu_name,
        "preprocessing_date": datetime.now(UTC).isoformat(),
        "resolution_cap_px": cap,
        "source_checksum": raw_meta.get("checksum_sha256"),
        "table_row_counts": {str(p.page_number): p.table_row_count for p in pages},
        "failed_pages": [p.page_number for p in pages if p.ocr_failed],
    }
    client.write_json(meta_key, ocr_meta)
    # The marker distinguishes a real OCR pass from the stored metadata returned
    # on a skip, so it is added to the RETURNED copy only. Writing it into blob
    # meant the skip path read it straight back and process_batch filed every
    # skipped document under "processed" — a rerun that did no OCR at all still
    # reported "N processed, 0 skipped".
    ocr_meta = {**ocr_meta, "_reprocessed": True}
    log.info("processed %s: %d pages on %s at %dpx", source_id, len(pages), env.device, cap)
    return ocr_meta


def process_batch(
    source_ids: Iterable[str],
    doc_type: str,
    client: BlobClient,
    engine: OcrEngine,
    *,
    tenant_id: str | None = None,
    device: Device | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Process many documents. One failure never stops the batch."""
    processed: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    for source_id in source_ids:
        try:
            meta = process_document(
                doc_type, source_id, client, engine,
                tenant_id=tenant_id, device=device, force=force,
            )
            # `process_document` returns the stored meta unchanged when it
            # skips, so there is no marker inside it to read — the previous
            # check looked for a `_skipped` key nothing ever wrote, and a run
            # that did no OCR at all reported every document as processed.
            (skipped if meta.get("_reprocessed") is not True else processed).append(source_id)
        except OcrError as exc:
            failed.append((source_id, str(exc)))
            log.warning("OCR failed for %s: %s", source_id, exc)
    return {"processed": processed, "skipped": skipped, "failed": failed}


def find_unprocessed(client: BlobClient, doc_type: str, tenant_id: str | None = None) -> list[str]:
    """Ingested documents with no ``ocr_meta.json`` yet."""
    tenant = paths._tenant(tenant_id)
    ingested = {
        key.split("/")[3]
        for key in client.list(f"raw-documents/{tenant}/{doc_type}/")
        if key.endswith("metadata.json")
    }
    # A render-only ocr_meta.json lives at the same key but records no OCR at
    # all. `process_document` already refuses to skip those; counting them as
    # done here filtered them out of the work list, so `--all-unprocessed`
    # printed "nothing to process" and no page_*.md was ever produced for a
    # document that had only been rendered.
    done = set()
    for key in client.list(f"processed/{tenant}/{doc_type}/"):
        if not key.endswith("ocr_meta.json"):
            continue
        try:
            if client.read_json(key).get("render_only"):
                continue
        except Exception:  # noqa: BLE001 - unreadable metadata is not "done"
            continue
        done.add(key.split("/")[3])
    return sorted(ingested - done)


def benchmark_gpu(
    pdf_bytes: bytes,
    engine_factory: Callable[[Device], OcrEngine],
    *,
    max_long_side_px: int | None = None,
) -> dict[str, Any]:
    """Time one document on GPU, for pod sizing and cost per document.

    There is no CPU comparison to make. MinerU's CPU path uses lighter model
    variants and produces different markdown from the same PDF, so a corpus
    spanning both devices is built from two distributions (arch §8a) — the
    choice was never speed against cost.
    """
    import time

    cap = max_long_side_px or resolution_cap_px()
    started = time.perf_counter()
    pages = engine_factory("cuda").process(pdf_bytes, device="cuda", max_long_side_px=cap)
    seconds = time.perf_counter() - started

    return {
        "resolution_cap_px": cap,
        "device": "cuda",
        "seconds": round(seconds, 2),
        "pages": len(pages),
        "seconds_per_page": round(seconds / len(pages), 2) if pages else None,
    }

def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Run MinerU OCR + page rendering (GPU by default)")
    parser.add_argument("--doc-type", required=True, choices=list(ACTIVE_DOC_TYPES))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-ids", nargs="+")
    group.add_argument("--all-unprocessed", action="store_true")
    # No --device. OCR is GPU-only: MinerU's CPU path uses lighter model variants
    # and produces different markdown from the same PDF, so a corpus spanning
    # both devices is built from two distributions (arch §8a). An option that
    # accepts a single value is a way to be surprised later, not a choice.
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--force-reprocess", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = for_ocr()
    engine = MinerUEngine(device="cuda")

    source_ids = (
        find_unprocessed(client, args.doc_type, args.tenant)
        if args.all_unprocessed else args.source_ids
    )
    if not source_ids:
        print("nothing to process")
        return 0

    result = process_batch(
        source_ids, args.doc_type, client, engine,
        tenant_id=args.tenant, force=args.force_reprocess,
    )
    print(f"{len(result['processed'])} processed, {len(result['skipped'])} skipped, "
          f"{len(result['failed'])} failed")
    for source_id, reason in result["failed"]:
        print(f"  FAILED {source_id}: {reason}")
    return 1 if result["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
