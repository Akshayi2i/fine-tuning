# SPEC 07 — Inference Core (Shared Model-Runner Primitive)

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02. **This module exists to break the cycle between evaluation, calibration, serving, and testing** — they all need one shared way to run a model version over a document. Build it here, once; everything downstream imports it.

## Why this module exists

Evaluation (IMPL-08), calibration (IMPL-09), serving (IMPL-11), and testing (IMPL-12) must all run the model **identically** (same prompt assembly, same generation, same logprob capture) or their numbers won't agree and "test == prod" breaks. That shared primitive lives here, with **no dependency on calibration, classification, or routing** (those are higher-level concerns layered on top later).

## Goal

A clean, low-level inference engine: given a resolved model version + a prepared document input + a doc_type, produce raw generated JSON text and per-token logprobs. No confidence calibration, no classification, no routing here — just correct, reproducible generation.

## Deliverables

### 1. `inference_core/model_runner.py`
- `load_model(version, backend="vllm"|"hf"|"gguf")` — resolves the version to concrete artifacts via IMPL-02 (`resolve_model_version`), loads base + Foundation (+ optional per-type adapter) or a merged/quantized model. Backend-swappable.
- `generate(model, messages, want_logprobs=True, **gen_kwargs)` → returns `{text, token_logprobs, tokens}`. Logprobs are mandatory-capable (needed by calibration).
- Deterministic given seed + greedy/temperature settings; logs the generation config.

### 2. `inference_core/input_builder.py`
- `build_messages(doc_type, image_paths, ocr_text_or_none, modality_mode, schema)` → the exact chat-format `messages` (master §9), using `common.prompts`. This is the single place prompt+image+OCR get assembled, so evaluation/serving/testing are guaranteed identical.
- Handles `image_only` (omit OCR block + image-only prompt) vs `ocr_plus_image`.
- Applies the resolution cap from config to any image passed in.

### 3. `inference_core/span_map.py`
- `map_field_spans(generated_text, tokens, token_logprobs)` → for each JSON field path (including nested + list rows), the token span and its logprobs. This is the raw material calibration (IMPL-09) turns into confidence. Pure/testable; no calibration logic itself.

### 4. `inference_core/runner_config.py`
- Central generation + backend config (model tag, backend, temperature, max_new_tokens, logprob settings), loaded from `configs/inference/`.

## Constraints
- **No imports from** calibration, serving, evaluation, or testing (they import this, not vice-versa).
- Backend-agnostic interface (vLLM for serving, HF/GGUF acceptable for local eval) so the same calls work in every context.
- No PII in logs.

## Acceptance checklist
- [ ] `load_model("v2")` resolves + loads a version (foundation+adapter and merged paths both work).
- [ ] `build_messages(...)` yields identical structure for eval, serving, testing given the same inputs.
- [ ] `generate(...)` returns text + aligned token logprobs.
- [ ] `map_field_spans(...)` correctly maps a field value to its tokens on a sample generation.
- [ ] Module imports without pulling in calibration/serving/eval/testing.
