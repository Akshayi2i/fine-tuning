# SPEC 04 — Labeling + Golden JSON Generation

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-03.

## Goal

Turn OCR'd documents into human-verified **golden JSON** labels via a bootstrap pre-annotation + human-review workflow, with provenance tracking. (Arch §6 "Golden JSON Generation Mechanism".)

## Deliverables

### 1. `data_pipeline/labeling/pre_annotate.py`
- Generates a first-pass ("silver") JSON draft for a document using a configurable annotator backend:
  - `base_qwen3vl` (default, self-hosted) — prompt the base/current model with schema + image (+OCR).
  - `own_finetuned` — once a v1 exists, use the current promoted model (better drafts each cycle).
  - `external_frontier` — **disabled by default**; guarded by an explicit `--allow-external` flag AND an env permission flag, because sending PII to third-party APIs may violate compliance (master §8). If used, require a zero-retention endpoint config.
- Output draft JSON is **never trusted as-is** — it's only a starting point for review.

### 2. `data_pipeline/labeling/review_tool/`
- Integration config for an external labeling tool (Label Studio or Argilla) OR a lightweight custom review UI:
  - Presents the page image(s) + OCR + draft JSON side-by-side.
  - Reviewer corrects every field; output is the golden JSON.
  - Support **double-annotation** on a configurable sample (default 10–20%): second independent label + adjudication, producing an inter-annotator agreement score.
- Provide a task-export/import adapter so labels round-trip cleanly.

### 3. `data_pipeline/labeling/export_golden_labels.py`
- Writes verified labels to `golden-labels/{doc_type}/{source_id}/golden.json` and `label_metadata.json` containing: reviewer id, review date, draft source (which annotator backend), double-annotated flag, agreement score.
- **Validates every golden.json against its schema** (IMPL-01) before writing — reject invalid labels.
- Idempotent; re-export overwrites only with a new label_metadata version.

### 4. `data_pipeline/labeling/active_learning.py`
- After a model version exists: run it over new unlabeled docs, use **calibrated confidence** (IMPL-08) to route — high-confidence → light spot-check, low-confidence fields → full manual review. Emits a prioritized review queue. (Arch §6 step 6; ties to §12 step 11 feedback loop.)

## Constraints
- External pre-annotation OFF unless explicitly permitted.
- Golden labels must be schema-valid.
- Store provenance for every label.
- No PII in logs.

## Acceptance checklist
- [ ] `pre_annotate` produces a draft with the self-hosted backend; external backend refuses without the permission flag.
- [ ] Review workflow exports a corrected golden JSON that passes schema validation.
- [ ] Double-annotation path produces an agreement score on the sampled subset.
- [ ] `label_metadata.json` captures full provenance.
- [ ] `active_learning` orders a queue by ascending confidence.
