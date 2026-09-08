# SPEC 08 — Evaluation (Metrics + Promotion Gating)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_06, SPEC_07. Uses the shared inference core (SPEC_07) for all model runs — no forward dependency on serving or testing.

## Goal

Evaluate any model/adapter version against the frozen golden eval set and gate promotion so no regression ships. (Arch §14.)

## Deliverables

### 1. `common/normalize.py` (shared normalizer — create here, reused by testing)
- Field-type normalizers so `01/01/2026` == `2026-01-01`, `$1,200.00` == `1200.0`, `Acme Mfg LLC` ≈ `ACME MANUFACTURING LLC`. Evaluation and testing (SPEC_12) MUST use this identical logic.

### 2. `evaluation/metrics/` — one module per metric, each pure and testable
- `field_exact_match.py` — exact + normalized match (via `common.normalize`). Per-field and aggregate.
- `field_f1.py` — precision/recall/F1 for list fields (claims, schedule rows).
- `list_recall.py` — **row completeness**: extracted row count vs ground-truth count; missed-row rate (arch §4 list-field gap).
- `schema_validity.py` — parse + validate output against the doc_type schema.
- `calibration_error.py` — Expected Calibration Error (ECE) on confidence vs correctness.
- `ocr_arbitration_accuracy.py` — on the noisy-OCR subset: did the model override bad OCR using the image?
- `mode_accuracy.py` — separate accuracy for `image_only` and `scanned` subsets (feed the ViT gate, SPEC_06).
- `classifier_accuracy.py` — doc-type + ACORD-form classification accuracy (arch §3a).

### 3. `evaluation/run_eval.py`
- Given `--model vN` (resolved via SPEC_02) + the golden eval set, run inference **via SPEC_07 inference core** across all eval docs and relevant modes, compute every metric, write `eval-reports/v{n}/{doc_type}/report.json` + a top-level summary. Per doc_type and per modality mode.
- Note: classifier accuracy uses the classifier from serving (SPEC_11) once it exists; until then, `run_eval` can score extraction with the true doc_type supplied, and classifier metrics get filled in on a later pass. (Keeps eval runnable before serving is built.)

### 4. `evaluation/gating.py`
- `promotion_gate(candidate_report, current_report, doc_type)` → candidate must match/beat current on every gating metric. Per-metric deltas + overall decision (arch §14). Writes the decision into the candidate's RunManifest (SPEC_02).

## Constraints
- Golden eval set is frozen + versioned separately; never train on it.
- Metrics deterministic + unit-testable on synthetic inputs.
- All model runs go through SPEC_07 (no bespoke inference here).

## Acceptance checklist
- [ ] Each metric returns correct values on hand-constructed synthetic cases.
- [ ] Normalized match handles dates/currency/names via `common.normalize`.
- [ ] `list_recall` flags a deliberately dropped claim row.
- [ ] `run_eval` produces a per-doc-type report with mode breakdowns, using SPEC_07.
- [ ] `gating` blocks a candidate that regresses any single gate metric.
- [ ] Promotion decision written to the RunManifest.
