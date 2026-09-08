# SPEC 03 — Ingestion + MinerU OCR

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02.

## Goal

Ingest raw insurance PDFs into the immutable `raw-documents/` layer, then run MinerU OCR + page-image rendering into `processed/`. This is the front of both the training-data pipeline and (reused) the inference pipeline.

## Deliverables

### 1. `data_pipeline/ingestion/pull_raw_pdfs.py`
- Takes a local folder or a source location, and for each PDF:
  - Assigns/validates a `source_id` (`{doc_type}_{index}`; doc_type either provided or inferred later — allow an `--unclassified` bucket).
  - Computes SHA-256 checksum; **dedup**: if checksum already ingested, skip and log.
  - Writes `raw-documents/{doc_type}/{source_id}/original.pdf` (immutable — never overwrite; a corrected doc becomes a new source_id).
  - Writes `metadata.json`: ingestion timestamp, source, checksum, page count, digital-vs-scanned flag (detect via presence of an embedded text layer), PII flags placeholder.
- Idempotent and re-runnable. CLI with `--input`, `--doc-type`, `--dest-container`.
- Uses the tighter-RBAC container for `raw-documents/` (master §8).

### 2. `data_pipeline/ocr/run_mineru.py`
- Batch-process documents from `raw-documents/` (or a passed list of source_ids):
  - Run **MinerU** to produce per-page markdown/text.
  - Render each page to a PNG at the **resolution cap** from `configs/base_model.yaml` (identical to training + inference — this consistency is mandatory).
  - Write `processed/{doc_type}/{source_id}/page_{n}.md` and `page_{n}.png`.
- Detect + record MinerU failures/low-confidence pages in a per-doc `ocr_meta.json` (needed later so the dataset builder and eval can account for imperfect OCR).
- Idempotent (skip if processed output exists and source checksum unchanged).
- Make MinerU invocation configurable (it may run as a subprocess or library call); keep the interface swappable.
- CLI: `--source-ids ... | --all-unprocessed`, `--doc-type`.

### 3. `data_pipeline/ocr/render_only.py` (helper)
- Path for `image_only` mode / inference: render page images without requiring OCR text, at the same resolution cap.

## Constraints
- `raw-documents/` is write-once; assert before writing.
- Resolution cap MUST come from config, never hardcoded, and be logged.
- No PII in logs (don't print OCR text).
- All Blob I/O via SPEC_02.

## Acceptance checklist
- [ ] Ingesting the same PDF twice results in one stored copy (dedup by checksum).
- [ ] Digital vs scanned flag is set correctly on sample inputs.
- [ ] MinerU output produces `page_*.md` + `page_*.png` at the configured resolution.
- [ ] Re-running OCR on already-processed docs is a no-op.
- [ ] `ocr_meta.json` records page count and any OCR failures.
- [ ] `render_only` produces images with no OCR dependency.
