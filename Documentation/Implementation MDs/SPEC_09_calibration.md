# SPEC 09 — Confidence Calibration

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_07, SPEC_08. Uses the inference core (SPEC_07) for logprobs/spans and the evaluation normalizer (SPEC_08) for the correctness signal. No forward dependency on serving/testing.
>
> **Architecture refs:** `finetuning-architecture-v1.md` §5 (**confidence flow, calibration, list-field second signal**), §0b (LoB carries confidence too), §15 (ECE as a gating metric).

## Goal

Turn raw token logprobs into trustworthy per-field confidence via post-hoc calibration, plus the list-field completeness signal that logprobs are structurally blind to.

## The end-to-end flow being implemented (arch §5)

```
Generation ──> token logprobs ──> map to field spans ──> aggregate per field
                                                              │
                                                        raw confidence
                                                              │
                                              post-hoc calibration transform
                                              (temperature / isotonic, per type)
                                                              │
                                                     calibrated confidence
                                                       │              │
                                              scalar fields      list fields
                                                       │              │
                                              threshold check   + row-completeness
                                                       │              │
                                                       └──> review routing <──┘
```

**Why calibration is not optional:** fine-tuned generative models — especially after LoRA fine-tuning on a narrow task — become **overconfident**. Correct and incorrect outputs carry similarly high token probabilities, because the model has learned to be fluent in the target format even when it is wrong about the content. Raw logprobs alone are not a usable confidence signal.

**Why not a separate verifier model:** another model to train, version, evaluate, and keep in sync with every Foundation/adapter version, plus doubled inference latency. It stays a documented escalation path **if** calibrated logprobs prove insufficient in evaluation (SPEC_08 ECE) — start with the cheap, well-understood approach.

## Deliverables

### 1. `calibration/logprob_confidence.py`
- Consumes field spans + logprobs from SPEC_07 `span_map` and aggregates to a raw per-field confidence.
- **Default aggregation = minimum token probability within the span** — most sensitive to the weakest link, which is what you want for flagging risky fields. Pluggable (`min` | `mean` | `geomean`); the exact choice is an empirical tuning detail, not an architectural one, so make it a config value and record which was used.
- Handles nested fields and list fields (per-value confidence within each row).
- **`line_of_business` gets confidence like any other field** (arch §0b) and is emitted in the same `{value, confidence}` shape.
- Costs nothing extra at training or inference time — the logprobs are already there.

### 2. `calibration/fit_calibration.py`
- On a **held-out labeled validation set — never training data** — compare raw confidence against actual field-level correctness (exact / normalized match via `common.normalize` from SPEC_08) and fit a transform:
  - **`temperature` scaling** (default: simple, one parameter, works well here)
  - **`isotonic` regression** (more flexible; handles non-monotonic miscalibration)
- Fit **per doc_type**, optionally per field-type. Fit for a specific model version — calibration is not transferable across versions.
- Persist to `calibration/calibration_store/{version}/{doc_type}.json` and push to Blob (SPEC_02). This is a small lookup/transform applied at serving time **after** the model call, before the JSON is returned.
- Report the ECE before and after fitting, so a fit that fails to improve calibration is visible rather than assumed.

### 3. `calibration/apply_calibration.py`
- Load the fitted transform for a model version + doc_type; map raw → calibrated confidence at inference time. Used by serving (SPEC_11) and testing (SPEC_12).
- **Fails loudly when no calibration exists for a version/doc_type** rather than silently passing raw (overconfident) values through as if calibrated. A missing calibration is an operational error, not a fallback.
- Applies the review threshold (default 0.7, tuned — arch §5) and emits `review_flags` for sub-threshold fields, which is the practical payoff of doing calibration properly.

### 4. `calibration/list_completeness.py`
The second confidence signal that per-token logprobs structurally cannot provide (arch §5).

**The failure mode:** if the model extracts 6 of 8 claims, the 2 missing claims have **no generated tokens**, so there is no low probability to flag. Per-field confidence on the 6 extracted rows can all be high while the extraction is silently incomplete. This is a recall failure, and token confidence is blind to it.

Implement **two independent cross-checks**, either of which flags the list:
1. **Document-stated count** — a `total claims: N` style field extracted from the document.
2. **Structure-derived count** — the number of table rows MinerU detected on the relevant pages (recorded in `ocr_meta.json` by SPEC_03).

When the model's row count disagrees with either, **flag the whole list for review regardless of per-value confidence**. Optionally calibrate a separate `row_completeness_confidence` against ground-truth row counts on the validation set.

This matters most for **Loss Runs**, where a missed claim row is both easy to make and expensive to miss.

## Constraints
- Calibration fit ONLY on held-out data — assert the validation `source_id`s are disjoint from training.
- Params versioned per model version + doc_type; never reused across versions.
- Correctness signal reuses SPEC_08 `common.normalize` — the same definition of "correct" as the promotion gate.
- Raw confidence is never returned to callers as if it were calibrated.
- No PII in logs (confidence records reference field paths, not field values).

## Acceptance checklist
- [ ] Raw per-field confidence extracted from a sample generation + logprobs (via SPEC_07 spans), including a `claims[i].amount` row field and `line_of_business`.
- [ ] Aggregation strategy is configurable and the choice is recorded.
- [ ] Temperature and isotonic both reduce ECE on a synthetic miscalibrated set; before/after ECE is reported.
- [ ] Fitted params persist/reload and apply deterministically.
- [ ] `apply_calibration` **raises** when no calibration exists for a version/doc_type.
- [ ] `list_completeness` flags a list when extracted rows < the document-stated count, and separately when < the MinerU-detected row count.
- [ ] A flagged list is routed to review even when every per-value confidence is high.
- [ ] Calibrated confidence is available with **no ground truth** — the inference-time case that makes production review routing possible.
- [ ] Fitting on data overlapping the training split fails the assertion.
