# SPEC 14 — Unit Tests + CI Fixtures

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: all prior specs (01–13).

## Goal

A test suite + CI that verifies the correctness-critical pieces without GPUs or live Azure, plus small fixtures.

## Deliverables

### 1. `tests/fixtures/`
- 2–3 tiny sample PDFs per active doc type (or synthetic stand-ins), MinerU-style OCR markdown, matching golden JSON — CI-small.
- Seeded synthetic logprob/generation samples for calibration + confidence tests.
- In-memory mock Blob backend (no live Azure).

### 2. Core tests (mirror each spec's acceptance checks)
- `test_data_collator.py` — **label masking correct**: loss only on assistant JSON tokens; system/image/OCR masked to `-100` (SPEC_06). Highest-value test.
- `test_schema_validity.py` — schemas load; valid examples pass; malformed/wrong-type fail (SPEC_01/08).
- `test_split_leakage.py` — no `source_id` crosses train/val/test; expansion after split (SPEC_05).
- `test_modality_mix.py` — 50/20/30 holds; image_only rows omit OCR (SPEC_05).
- `test_inference_core.py` — `build_messages` identical across contexts; `map_field_spans` correct (SPEC_07). **This guards test==prod at the primitive level.**
- `test_metrics.py` — normalized match (dates/currency/names), list recall, F1 correct (SPEC_08).
- `test_gating.py` — a regression on any single metric blocks promotion (SPEC_08).
- `test_calibration.py` — temperature + isotonic reduce ECE; params persist/reload; list-completeness flags dropped rows (SPEC_09).
- `test_run_registry.py` — RunManifest validates; `adapters_depending_on` + `resolve_model_version` correct (SPEC_02).
- `test_router.py` — low classifier confidence → Foundation-only fallback + review flag; deterministic selection (SPEC_11).
- `test_source_id.py` — id build/parse/validate round-trips (SPEC_01).
- `test_test_prod_parity.py` — testing output == serving pipeline output on the same fixture (SPEC_11/12). Since testing reuses `serving/pipeline.py`, this asserts they don't diverge.

### 3. CI
- `.github/workflows/ci.yml`: install `[dev]` extras, run `ruff`, `mypy`, `pytest` on every push. Mark GPU + live-Azure tests to skip so CI is CPU-only + fast.
- Coverage gate on correctness-critical modules (collator masking, split, inference-core, metrics, calibration, gating, registry).

## Constraints
- Runs without GPU or live Azure (mock/stub heavy deps).
- Fixtures contain no real PII.
- Deterministic (seeded) everywhere.

## Acceptance checklist
- [ ] `pytest` passes green on CPU-only CI.
- [ ] Collator masking test fails if masking is broken (mutation-check it).
- [ ] Split-leakage test fails if expansion moves before split.
- [ ] `test_inference_core` + `test_test_prod_parity` pass on fixtures.
- [ ] `ruff` + `mypy` clean.
