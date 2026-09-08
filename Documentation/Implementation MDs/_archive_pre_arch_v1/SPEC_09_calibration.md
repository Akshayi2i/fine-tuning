# SPEC 09 — Confidence Calibration

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_07, SPEC_08. Uses inference core (SPEC_07) for logprobs/spans and the evaluation normalizer (SPEC_08) for the correctness signal. No forward dependency on serving/testing.

## Goal

Turn raw token logprobs into trustworthy per-field confidence via post-hoc calibration, plus the list-field completeness signal. (Arch §4.)

## Deliverables

### 1. `calibration/logprob_confidence.py`
- Consumes field spans + logprobs from SPEC_07 `span_map`, aggregates to raw per-field confidence. Default = **min token prob** in span; pluggable (min/mean/geomean).
- Handles nested + list fields (per-value confidence within rows).

### 2. `calibration/fit_calibration.py`
- On a **held-out validation set** (never training data), compare raw confidence vs correctness (via `common.normalize` from SPEC_08) and fit a transform:
  - `temperature` scaling (default) and `isotonic` regression (flexible), selectable.
  - Fit **per doc_type** (optionally per field-type).
- Persist to `calibration/calibration_store/{version}/{doc_type}.json` + push to Blob.

### 3. `calibration/apply_calibration.py`
- Load the fitted transform for a model version + doc_type; map raw → calibrated confidence at inference time. Used by serving (SPEC_11) and testing (SPEC_12).

### 4. `calibration/list_completeness.py`
- Separate "did we get all rows?" confidence: cross-check extracted row count vs a document-derived count (stated "total: N" or MinerU-detected table rows). Disagreement → flag the whole list regardless of per-value confidence. Optionally calibrate against ground-truth row counts on validation. (Arch §4.)

## Constraints
- Calibration fit ONLY on held-out data.
- Params versioned per model version + doc_type.
- Correctness signal reuses SPEC_08 `common.normalize`.

## Acceptance checklist
- [ ] Raw per-field confidence extracted from a sample generation + logprobs (via SPEC_07 spans).
- [ ] Temperature + isotonic both reduce ECE on a synthetic miscalibrated set.
- [ ] Fitted params persist/reload and apply deterministically.
- [ ] `list_completeness` flags a list when extracted rows < document-stated count.
- [ ] Calibrated confidence available with no ground truth (inference-time).
