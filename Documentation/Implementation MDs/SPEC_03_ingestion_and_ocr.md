# SPEC 03 — Ingestion + MinerU OCR (GPU, with Version + Device Pinning)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02.
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §7 (page images, multi-page), §8a (**MinerU version pinning**), §11 (resolution cap), §13 steps 1–2, §18a (raw-document storage, dedup, retention, access pattern).

## Goal

Ingest raw insurance PDFs into the immutable, tenant-scoped `raw-documents/` layer, then run MinerU OCR + page-image rendering into `processed/` — recording the exact MinerU version used, because the model is fine-tuned partly on *how MinerU formats its output*.

This is the front of both the training-data pipeline and (reused) the inference pipeline.

## Deliverables

### 1. `data_pipeline/ingestion/pull_raw_pdfs.py`
For each PDF from a local folder or source location:
- Assign/validate a `source_id` (`{doc_type}_{index}`; doc_type either provided or deferred — allow an `--unclassified` bucket resolved later by the classifier). **`build_source_id` and `paths.raw_pdf` must accept it**, not only the guard in `ingest_directory` — accepting it in one place and rejecting it in the others made every document fail one at a time into `failed`, so a broken mode read as bad input.
- Compute a **SHA-256 checksum** and **dedup** (arch §18a): if the checksum was already ingested, skip and log. Insurance documents — Loss Runs and renewal policies especially — are frequently re-submitted with only minor changes; de-duping prevents redundant labeling effort and corpus bloat.
- Write `raw-documents/{tenant_id}/{doc_type}/{source_id}/original.pdf` — **immutable, exactly as received, never overwritten**. A corrected version of a document is ingested as a **new `source_id`**, so historical training runs remain reproducible against the exact bytes they trained on.
- Write `metadata.json`: ingestion timestamp, `tenant_id`, source system, checksum, page count, **digital-vs-scanned flag** (detect via presence of an embedded text layer — this flag feeds the scanned-PDF eval subset in SPEC_08 and the ViT gate in SPEC_06), PII flags placeholder, retention class.
- Idempotent and re-runnable. CLI: `--input`, `--doc-type`, `--dest-container` (`--tenant` optional, defaults from env).
- Uses the tighter-RBAC container for `raw-documents/` (master §8). **This is the only pipeline component that writes this layer.**


### 2. `data_pipeline/ocr/mineru_version.py` *(new — arch §8a)*
- `get_mineru_version()` → the exact installed MinerU version string (e.g. `"1.4.2"`).
- `assert_version_matches(corpus_manifest)` → raises when the running MinerU version **or device** differs from what a corpus manifest pins. Device is included because §8a's rule is about the *output distribution*, and GPU and CPU MinerU can format differently — confirm empirically and relax the device check if they prove identical.
- **Serving a model trained on MinerU v{n} output against documents processed by MinerU v{n+1} is distribution shift, and is treated as a regression trigger — not a routine dependency bump.** This helper is called by the dataset builder (SPEC_05), the serving pipeline (SPEC_11), and the testing harness (SPEC_12); a mismatch is a loud failure with the remediation stated (reprocess affected documents, increment the corpus version, retrain).

### 3. `data_pipeline/ocr/run_mineru.py`
Batch-process documents from `raw-documents/` (or a passed list of source_ids):
- Run **MinerU on GPU** to produce per-page markdown/text. GPU is the **only** execution mode (arch §14): layout detection, table/formula recognition and the OCR models are GPU-bound, and the CPU path falls back to lighter model variants that are both slower and less accurate. `cpu` is refused (`mineru_version.resolve_device`), and the device is **recorded** in `ocr_meta.json`, never assumed.
- Render each page to PNG at the **resolution cap from `configs/base_model.yaml`** — identical in training data prep and production inference. This consistency is mandatory (arch §11): image token count is a direct function of page resolution and is the single biggest cost/latency lever.
- Write `processed/{tenant_id}/{doc_type}/{source_id}/page_{n}.md` and `page_{n}.png`.
- **Preserve page order explicitly** — Interleaved-MRoPE means multi-page ordering is positionally meaningful (arch §3). Page numbering must be stable and 1-based.
- Write per-doc `ocr_meta.json`: page count, **`mineru_version`**, **`ocr_device`** (`cuda`; `cpu` appears only on corpora predating the GPU-only rule), **`preprocessing_date`** (ISO 8601, arch §8a), resolution cap used, per-page OCR failure/low-confidence flags, and **detected table row counts per page** (consumed by the list-completeness signal in SPEC_09).
- Idempotent — skip if processed output exists, the source checksum is unchanged, **and** the MinerU version and device match; a change to either forces reprocessing.
- Make MinerU invocation configurable (subprocess or library call); keep the interface swappable.
- CLI: `--source-ids ... | --all-unprocessed`, `--doc-type`, `--force-reprocess`.

### 4. `data_pipeline/ocr/render_only.py` (helper)
- Renders page images without requiring OCR text, at the same resolution cap — the `image_only` mode path (arch §6) for both corpus build and inference.
- Writes the same `ocr_meta.json` skeleton (page count, resolution, version) minus OCR content, so `image_only` rows are still traceable.

## Constraints
- `raw-documents/` is write-once and tenant-prefixed; assert before writing.
- Resolution cap MUST come from config, never hardcoded, and be logged and recorded in `ocr_meta.json`.
- MinerU version MUST be recorded on every processed document.
- No PII in logs (never print OCR text).
- All Blob I/O via SPEC_02.

## GPU only — not a default, a constraint

MinerU runs on **GPU exclusively**. This is not a performance default with an
opt-out; the CPU path is refused.

The reason is correctness, not speed. MinerU's CPU path selects **lighter model
variants**, so the same PDF yields *different markdown*: different table splits,
different cell boundaries, different header handling. The model learns how MinerU
formats its output (arch §8a), so a corpus built across both devices is built
from two distributions — and serving from whichever the pod happened to have is a
third. A speed/cost trade would be a decision; this is a defect.

Consequences, all enforced in code:

- `resolve_device("cpu")` raises. There is no `--device` flag on the OCR CLI.
- With no CUDA device visible, the stage **fails** rather than falling back. A
  silent fallback would finish the job, write markdown from the wrong
  distribution, and report success — invisible until accuracy is unexplainably
  poor.
- `ocr_device` is still recorded on every document. It is now always `cuda`, and
  that is provenance rather than a variable.
- `assert_gpu_only` refuses to *extend* a corpus that records `cpu` — such a
  corpus predates this rule, stays readable, and cannot take new documents
  without mixing two markdown formats in one training set.
- CI simulates a GPU pod to exercise these paths, because they now refuse to run
  without one.

## Acceptance checklist
- [ ] Ingesting the same PDF twice results in one stored copy (dedup by checksum).
- [ ] Ingesting a corrected document creates a **new `source_id`**; the original is untouched.
- [ ] Digital vs scanned flag is set correctly on sample inputs.
- [ ] MinerU output produces `page_*.md` + `page_*.png` at the configured resolution, 1-based and ordered.
- [ ] `ocr_meta.json` records page count, `mineru_version`, **`ocr_device`**, `preprocessing_date`, resolution cap, OCR failures, and per-page table row counts.
- [ ] The row count discounts **one header per table**, not one per page: a page with two tables otherwise over-reports by one, and the SPEC_09 cross-check then raises a false row-completeness flag — which active learning treats as an unconditional override to full manual review.
- [ ] A **render-only** `ocr_meta.json` (written by `render_only.py` at the same key) does not satisfy the OCR idempotence check. Otherwise a document rendered first is never OCR'd, no markdown is ever produced, and the CLI reports it as processed.
- [ ] `process_batch` reports genuinely skipped documents as skipped rather than counting them as processed.
- [ ] **OCR runs on GPU only.** There is no `--device` flag and no CPU fallback: `resolve_device` refuses `cpu`, and refuses to run at all when no CUDA device is visible.
- [ ] GPU throughput is measured per document (`benchmark_gpu`), for pod sizing and cost per document. There is no CPU comparison, because there is no CPU path.
- [ ] Re-running OCR on already-processed docs is a no-op; **changing the MinerU version forces reprocessing**.
- [ ] `assert_version_matches` raises on a seeded version mismatch with a remediation message.
- [ ] `render_only` produces images with no OCR dependency.
- [ ] Running `render_only` over an **already-OCR'd** document re-renders its images but **preserves the OCR metadata** — `table_row_counts` survives and the document is not relabelled `render_only`. Overwriting it silently switched off the SPEC_09 detected-row cross-check while the `page_*.md` files still sat there.
- [ ] `find_unprocessed` applies the same render-only test `process_document` does. Counting a render-only `ocr_meta.json` as done filtered the document out of `--all-unprocessed`, so it was never OCR'd and the CLI reported nothing to do.
- [ ] The `_reprocessed` marker is added to the **returned** metadata only, never written to Blob — persisting it made the skip path read it straight back, so `process_batch` reported every skipped document as processed.

---

## Current implementation (2026-09-27)

- **MinerU 1.x**, installed by the `ocr` dependency group as `magic-pdf[full]>=1.3,<2` — the code imports
  `magic_pdf` and records the `magic-pdf` version, and MinerU 2.x renamed the package. Model weights are a
  separate download (MinerU's `download_models_hf.py`); `setup_pod.sh ocr` prints the step.
- **Delivered batches travel through Blob** (the ~20 GB training folder cannot go through git): staged by
  azcopy under `intake/{batch}/` in the **raw container** (`paths.intake_batch_dir`; `requires_raw_container`
  covers `intake/`, so only an ingestion or OCR client can read it), pulled onto the volume by
  `python -m data_pipeline.ingestion.pull_intake --batch <batch>` (to `/workspace/intake/<batch>`, 16 parallel
  downloads, `.part` then rename, PDFs already present skipped, labels always re-fetched, `--refresh` for
  everything), then imported from that folder. `import_labeled_pdfs --from-blob <batch>` pulls and imports in
  one command. Both run detached on the pod.
- **Its own environment.** MinerU 1.x cannot share an environment with vLLM 0.11 / torch 2.8, so on the pod
  it is installed in `/workspace/venv-ocr` (`scripts/pod_bootstrap.sh`) and OCR runs there first:
  `/workspace/venv-ocr/bin/python -m data_pipeline.ocr.run_mineru --doc-type policy --all-unprocessed`.
  The pipeline's preprocessing stage then finds every document done (`find_unprocessed` checks for
  `ocr_meta.json`). Run from the training environment with documents still pending, it raises a
  `PipelineError` naming that command instead of failing on `import magic_pdf`.
- **Its config on the volume**: MinerU's download writes `~/magic-pdf.json` on the container disk, which a
  pod stop wipes. The bootstrap moves it to `/workspace/magic-pdf.json` and sets
  `MINERU_TOOLS_CONFIG_JSON` to that absolute path (MinerU joins it to the home directory, and an absolute
  path wins); `HF_HOME` on the volume keeps the weights.
- **GPU only**, with no CPU option (see the GPU section above and `common/gpu.py` for the same rule on every
  other model step).
- **Render-only documents** (`ocr_meta.render_only: true`, rendered for labeling without OCR) and documents
  whose meta records **no pages** are skipped by the corpus build with a logged reason; they are never
  built into `ocr_plus_image` rows with blank text.
- On the pod, `run_mineru`, `render_only` and all three importers run **detached in tmux**
  (`orchestration/detach.py`) — a closed laptop does not stop an OCR batch.

**MinerU engine, wired** (`data_pipeline/ocr/run_mineru.py::MinerUEngine`, MinerU 1.x public API):
- refuses anything but CUDA; renders page images with the shared renderer (`render_only.render_pdf_pages`)
  at the configured cap, so OCR'd and image-only documents are the same pixels;
- `PymuDocDataset(pdf).classify()` picks OCR mode for scans and text mode for a text layer; the result's
  **content list** becomes **one markdown string per page** (`pages_from_content_list`: headings by
  `text_level`, tables as MinerU's HTML with captions and footnotes, image captions only, equations as text).
  MinerU's own `get_markdown` returns one blob per document, which cannot be split back into pages;
- **table rows are counted in HTML tables** as well as pipe tables (`count_table_rows`) — MinerU 1.x writes
  HTML, and counting pipe tables alone read every page as row-less, so the row-completeness check never fired;
- a page MinerU returns nothing for is kept and flagged `ocr_failed`.

Verified here with a stand-in MinerU on a real PDF (`tests/test_mineru_engine.py`); the real library, its model
weights and GPU output are verified on the pod by the Phase 0 spike (`check_mineru_gpu`). Serving callers that
supply `page_texts` must produce them with this engine, or the model reads a formatting it never trained on.

**Scanned is MinerU's classification**, recorded as `ocr_meta.is_scanned` (OCR mode = scan). An empty page
is an OCR failure only on a scan; on a text layer it is a blank or picture-only page, and no longer makes the
document count as scanned (older metas fall back to "any failed page"). **HTML table rows are parsed**, not
regex-matched: a row is a header only when it has header cells and no data cells (so row-header schedules
count), and nested tables no longer cut the outer table short.

**MinerU's own device setting** (`data_pipeline/ocr/mineru_config.py`): MinerU 1.x takes its device from
`device-mode` in `~/magic-pdf.json` (file name overridable by `MINERU_TOOLS_CONFIG_JSON`), whose shipped default
is `cpu` — a GPU being present does not make MinerU use it. The engine refuses unless the config says `cuda`;
`python -m data_pipeline.ocr.mineru_config --cuda` sets it, and `setup_pod.sh ocr` runs that once the weights
are downloaded. **Mixed documents**: any page without a text layer sends the whole document through OCR mode
(text mode left scanned endorsement pages of a typed policy unread and unflagged); `scanned` is recorded per
page and `is_scanned` is true when any page is a scan.
