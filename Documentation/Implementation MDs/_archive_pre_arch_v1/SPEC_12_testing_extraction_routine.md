# SPEC 12 — Testing / Extraction Routine (CLI Harness)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_03 (OCR), SPEC_07 (inference core), SPEC_08 (metrics), SPEC_09 (calibration), SPEC_11 (serving pipeline).

## Goal

A CLI harness to test any fine-tuned model version on real PDFs post-training, producing structured JSON + per-field/overall confidence + quality metrics, with version-organized outputs and provenance. **Reuses the serving pipeline (SPEC_11) so test == prod.** (Arch §15.)

## Deliverables

### 1. Folder layout under `testing/` (create dirs with `.gitkeep`)
```
testing/
├── run_extraction.py
├── test_data/           # INPUT: drop test PDFs (scanned or digital)
├── prompts/             # acord.prompt.txt, policy.prompt.txt, lossrun.prompt.txt
├── ocr_cache/           # cached MinerU output per doc
├── results/{version}/   # OUTPUT: e.g. results/v2/abcLossRun.json
├── metrics/{version}/    # OUTPUT: per-doc + _run_summary.json
└── extraction_registry.json
```

### 2. `testing/prompts/*.prompt.txt`
- One per active doc type (acord, policy, lossrun): role framing + injected schema (from registry, kept in sync) + extraction rules + modality instruction line. **Same prompts serving uses** (they should resolve to the same rendered prompt as `common.prompts`) — no drift. (Arch §15.)

### 3. `testing/run_extraction.py` — the CLI
- Flags:
  - `--model vN` — **user selects the version** (resolved via SPEC_02).
  - `--input <file|dir>` — test PDFs.
  - `--mode ocr_plus_image|image_only`.
  - `--ground-truth <dir>` — optional; if omitted, still emit JSON + confidence, no accuracy metrics.
- Per PDF: OCR (SPEC_03) → **call `serving/pipeline.py` (SPEC_11)** for classify → route → inference (SPEC_07) → calibrated confidence (SPEC_09) → validate → write `results/{version}/{doc_stem}.json` → metrics (SPEC_08, if ground truth) → append `extraction_registry.json`.
- The routine is a thin CLI wrapper over the serving pipeline + OCR + metrics; it must NOT re-implement inference (that would break test==prod).

### 4. Metrics output
- Per-doc `*.metrics.json`: per-field {value, confidence, correct?}, overall_confidence, schema_valid, field_exact_match_rate, list_field_f1, list_recall (SPEC_08 metrics + `common.normalize`).
- `metrics/{version}/_run_summary.json`: batch aggregates — mean field-exact-match, list F1, list recall, schema validity, ECE, per-doc-type breakdown, mean latency/doc. The cross-version comparison file.

### 5. `testing/extraction_registry.json`
- One entry per extraction: document, doc_type, model_version, mode, result_path, metrics_path, overall_confidence, schema_valid, extracted_at. Ties `model_version` back to the training RunManifest (SPEC_02) so a result traces to the corpus/commit that produced its model. (Arch §15.)

## Constraints
- Confidence emitted even without ground truth.
- Same prompts + same pipeline as serving (reuse, don't reimplement).
- No PII in logs; results dir gitignored.

## Acceptance checklist
- [ ] `run_extraction --model v2 --input test_data/` writes `results/v2/*.json`.
- [ ] `--mode image_only` runs with no OCR and still extracts.
- [ ] Per-field + overall confidence present with and without ground truth.
- [ ] With `--ground-truth`, per-doc + summary metrics computed.
- [ ] `extraction_registry.json` links result → model version.
- [ ] Output equals the serving pipeline's output on the same input (parity test).
