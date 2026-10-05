# SPEC 11 — Serving (vLLM Endpoint + Classifier + Routing)

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-07 (inference core), IMPL-09 (calibration), IMPL-10 (postprocessing).

## Goal

Serve the promoted model on a RunPod Serverless vLLM endpoint with per-doc-type adapter routing, document-type classification, long-document page routing, and calibrated confidence post-processing. Built on the shared inference core so serving == eval == testing. (Arch §3a, §3b, §4, §6, §13.)

## Deliverables

### 1. `serving/vllm_entrypoint.py`
- RunPod Serverless handler wrapping vLLM serving the promoted merged/quantized model, OpenAI-compatible, **logprobs enabled**.
- **LoRA hot-swap**: Foundation + per-request per-type adapter (vLLM multi-LoRA), chosen from the classifier result.
- Pulls the promoted artifact from Blob on cold start (IMPL-02); version configurable.
- Uses IMPL-07 `model_runner`/`input_builder` for generation so output matches eval/testing exactly.
- **No plaintext PII in logs** (master §8).

### 2. `serving/doc_type_classifier.py`
- Identifies doc type ∈ {acord, policy, lossrun} + ACORD form number (two-level, arch §3b).
- Default: **zero-shot via the Foundation model** (arch §3a Option C) using `doc_type_classifier_prompt.jinja`; interface swappable for a dedicated vision classifier later. Returns label + confidence.

### 3. `serving/adapter_router.py`
- Maps classifier result → (adapter path, prompt file, schema). **Low-confidence fallback**: fall back to **Foundation-only extraction** with a generic prompt and flag for human routing review — never silently guess (arch §3a).

### 4. `serving/page_router.py`
- Long-doc handling for Policy (arch §6): if page count > threshold, select likely-relevant pages (keyword/section over per-page OCR, or a light page classifier), scoped extraction, then merge with a conflict rule (declarations page wins for policy-level fields). Short docs skip. Record which pages fed each field.

### 5. `serving/confidence_postprocess.py`
- Applies IMPL-09 calibration to raw logprob confidence, adds list-completeness, shapes the final output contract (master §9), flags sub-threshold fields for review.

### 6. `serving/pipeline.py` (request orchestrator)
- Per request: (OCR if provided) → classify → route adapter/prompt → (page-route if long) → IMPL-07 inference → IMPL-09 confidence → final JSON. **This is the canonical pipeline testing (IMPL-12) reuses**, guaranteeing test == prod.

## Constraints
- Non-GPU orchestration designed to run **outside** RunPod; the RunPod handler stays scoped to inference (arch §13). Document the boundary.
- Deterministic adapter/prompt/schema selection.
- Calibrated (not raw) confidence in responses.

## Acceptance checklist
- [ ] Endpoint serves a request end-to-end → schema-shaped JSON + calibrated confidence.
- [ ] Correct per-type adapter hot-swapped from classification.
- [ ] Low classifier confidence → Foundation-only fallback + review flag.
- [ ] Long policy doc triggers page routing; short docs skip.
- [ ] No PII in logs.
- [ ] `serving/pipeline.py` is the single path reused by testing.
