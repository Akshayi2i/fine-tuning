# SPEC 05 — Dataset Builder (JSONL, Modality, Split)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_04.

## Goal

Compile processed docs + golden labels into versioned, chat-format JSONL training corpora — applying the 3-regime modality split and leakage-safe train/val/test splitting. (Arch §6, §7, §10.)

## Deliverables

### 1. `data_pipeline/dataset_builder/split_train_val_test.py`
- **Split at `source_id` level BEFORE any modality expansion** (master §10 — this prevents the same document leaking across splits via different modality variants).
- Ratio scales with per-type volume (arch §7): small pilot ≈ 70/18/12; at scale 80/10/10. Configurable, seeded, logged.
- Output: a deterministic assignment `source_id → split`, saved for reproducibility.

### 2. `data_pipeline/dataset_builder/modality_dropout.py`
- For each `source_id`, generate the 3 modality variants with the mix ratios (50/20/30 — master §10). Each variant becomes one JSONL row in the split its source_id belongs to.
- `image_only` rows omit the OCR text block and use the image-only system prompt.

### 3. `data_pipeline/dataset_builder/noisy_ocr_augment.py`
- For `noisy_ocr_image` rows: take the real MinerU OCR and inject realistic corruptions (digit↔letter confusions like O/0, l/1; merged/split table cells; dropped headers) while keeping the **golden JSON as the correct target** — teaching image-over-OCR arbitration (arch §5, §6).
- Corruptions parameterized and seeded; keep them plausible (mirror MinerU's real failure modes), not random noise.

### 4. `data_pipeline/dataset_builder/build_jsonl.py`
- Orchestrates: for each split and doc_type, assemble chat-format rows using the exact data contract (master §9):
  - `system` = rendered prompt (schema + modality instruction) via `common.prompts`.
  - `user` = image block (+ OCR text unless image_only).
  - `assistant` = golden JSON string.
- Carry `doc_type`, `acord_form`, `modality_mode`, `source_id`, `split` on every row.
- Write `corpus/v{n}/{doc_type}/{train,val,test}.jsonl` to Blob.
- Reference images/OCR by their Blob URIs or a training-time-resolvable path (document the choice).

### 5. `data_pipeline/corpus_manifest.py`
- Writes `manifest.json` per corpus version: example counts per doc_type × split × modality_mode, source_id lists per split, seed, ratios, builder git commit. This is what training + eval read to know exactly what a corpus version contains.

## Constraints
- Zero split leakage — assert no `source_id` appears in more than one split.
- Every row schema-checkable (the assistant JSON validates against the doc_type schema).
- Deterministic + seeded; re-running yields identical corpora.

## Acceptance checklist
- [ ] No `source_id` crosses splits (automated assertion).
- [ ] Each source produces exactly 3 rows with the correct modality distribution across the dataset.
- [ ] `noisy_ocr_image` rows have corrupted OCR but correct golden JSON targets.
- [ ] `image_only` rows contain no OCR text and use the image-only prompt.
- [ ] `manifest.json` counts match the actual JSONL row counts.
- [ ] Rebuild with same seed produces byte-identical output.
