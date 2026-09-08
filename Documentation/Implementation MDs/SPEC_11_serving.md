# SPEC 11 — Serving (vLLM Endpoint + Classifier + Routing)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_07 (inference core), SPEC_09 (calibration), SPEC_10 (postprocessing).
>
> **Naming hazard (master §0.1):** the architecture's "**Fideon SPEC_11**" is the external Presidio de-identification spec (implemented in our SPEC_05). This file is our serving layer. Different documents.
>
> **Architecture refs:** `finetuning-architecture-v1.md` §0a (audit gate contract), §0b (LoB output), §4a (**classifier as a load-bearing component**), §4b (ACORD two-level), §4c (day-zero classifier), §5 (confidence), §7 (long-document page routing), §14 (**RunPod split: what runs where**).

## Goal

Serve the promoted model on a RunPod Serverless vLLM endpoint with per-doc-type adapter routing, document-type classification, long-document page routing, and calibrated confidence post-processing — built on the shared inference core so **serving == eval == testing**.

## The infrastructure boundary (arch §14) — get this right or the design leaks

```
Training (ephemeral pod)              Serving (persistent endpoint)
─────────────────────────             ─────────────────────────────
clone repo @ commit                   vLLM + promoted model
pull base + corpus (Blob)                     │
run QLoRA training                    request ──> classify doc type
push adapter + manifest (Blob)                │
terminate                             select + hot-swap adapter
                                              │
External orchestrator                 inference + logprobs
triggers pods via RunPod API                  │
                                       calibrate confidence
Application logic (CPU infra,                 │
outside RunPod): OCR, prompt          return JSON + confidence
assembly, endpoint calls,
confidence calibration
```

**RunPod is scoped purely to GPU-bound work.** Application/orchestration logic — calling MinerU, assembling the prompt, calling the endpoint, applying the calibration transform, returning final JSON — runs in **your own service** (Azure Function, App Service, or AKS), not inside RunPod. This keeps business logic on cheap CPU infra, keeps it portable if the GPU provider changes, and keeps the RunPod handler small. Document this boundary in the module docstring; it is an architectural constraint, not a deployment preference.

## Deliverables

### 1. `serving/vllm_entrypoint.py`
- RunPod Serverless handler wrapping vLLM, serving the promoted merged/quantized model over an OpenAI-compatible API, **with logprobs enabled** (calibration depends on them).
- **LoRA hot-swap**: Foundation + a per-request per-type adapter via vLLM multi-LoRA, chosen from the classifier result — no base-model reload per request (arch §4).
- Pulls the promoted artifact from Blob on cold start (SPEC_02); version configurable.
- Uses SPEC_07 `model_runner` / `input_builder` for generation so output matches eval and testing exactly.
- **Asserts the serving MinerU version matches the training corpus pin** (SPEC_03 `assert_version_matches`) — a mismatch is distribution shift and a regression trigger (arch §8a), not a warning to ignore.
- **No plaintext PII in logs**; do not persist raw request/response bodies (master §8).

### 2. `serving/doc_type_classifier.py`
**This is load-bearing, not preprocessing (arch §4a).** If classification is wrong, you load the wrong adapter *and* the wrong prompt *and* the wrong schema, and the extraction fails no matter how good the model is.

```
document ──> classifier ──> confidence >= threshold ? ──> yes ──> select adapter + prompt + schema
                                     │                              (acord -> + form number)
                                     └── no ──> Foundation-only extraction + human routing review
```

- Identifies doc type ∈ {`acord`, `policy`, `lossrun`} **and the ACORD form number** — the two-level classification of arch §4b. Returns label + confidence.
- **Default implementation: zero-shot via the Foundation model** (arch §4a Option C) using `doc_type_classifier_prompt.jinja` — zero extra infrastructure, and the Foundation already understands these document types.
- Interface is swappable for a **dedicated vision classifier** (Option B) later, which is the right escalation if classification accuracy becomes a *measured* bottleneck — it works in both OCR and image-only modes, unlike a text-feature classifier.
- **Day-zero (arch §4c):** before Foundation v1.0 exists, the classifier runs against the **base** `Qwen3-VL-8B-Instruct`. This path is explicitly temporary and is superseded on promotion of Foundation v1.0, after which the base model is retired from the production inference path.
- **Has its own accuracy target and is measured on the golden eval set** alongside extraction metrics (SPEC_08 `classifier_accuracy`) — a classifier at 92% caps the whole system at 92%.

### 3. `serving/adapter_router.py`
- Maps classifier result → (adapter path, prompt file, schema). Deterministic selection.
- **Low-confidence fallback (arch §4a):** when the classifier isn't confident, **don't silently guess**. Fall back to **Foundation-only extraction** with a generic schema-agnostic prompt and flag the document for human routing review. A wrong-adapter extraction is worse than a slightly-generic one.
- Runs **before** adapter selection in both the serving path and the testing routine.
- For ACORD, resolves the form number to the matching schema even when a single shared `acord` adapter is in use (arch §4b Option 1) — the classifier must identify the specific form so the correct schema is always selected.

### 4. `serving/page_router.py`
Long-document handling, primarily a **Policy-document** concern (arch §7). Policy documents can run 50+ pages; each page at full resolution costs 1,000–2,000+ vision tokens, so a naive "every page image + all OCR in one prompt" approach blows the context window and wastes compute on pages containing no extractable fields.

1. **Page routing** — for a document over the page threshold (default `>5` pages, from `vllm_serving.yaml`), identify which pages contain the target fields via a cheap first pass: keyword/section detection over MinerU's per-page OCR text, or a lightweight page classifier. Most policy schemas draw from a minority of pages (declarations, schedules, specific endorsements).
2. **Scoped extraction** — run the model on the selected pages **in a single call**, interleaved `[img_3][<page 3 of 20> + md_3][img_9][<page 9 of 20> + md_9]…` (SPEC_07). One call per page was the earlier design and was wrong twice over: it asked the model to produce a complete document-level JSON from one page — a shape no training row ever had — and it prevented the model from seeing that a table on page 9 continues on page 14. Interleaving makes a routed request a **subsequence of the full document**, so the served shape is the trained shape with fewer pairs. It is also one call instead of N.
3. **Merge — no longer on the serving path.** A single call returns one document-level object, so there is nothing to merge. The conflict rule it implemented (*the declarations page wins for policy-level fields*) is stated to the model directly in the Policy prompt block, which keeps it in one place rather than two that can disagree. `merge_page_extractions` is retained as the only implementation of that rule in code, for any future path that does produce per-page results.

The page numbers in the markers are the document's real numbers, so a gap tells the model it is holding a fragment: fields living on pages it was not shown are absent, not missing. `image_only` carries the markers without markdown — the guarantee is *no OCR text*, not *no text at all*.

- **Short documents skip routing entirely** — ACORD forms and most Loss Runs are 1–few pages and are extracted single-pass.
- **Record which pages fed each extraction** in the output metadata (`pages_used`, master §9), so a low-confidence field can be traced to the specific page it came from.

### 5. Confidence post-processing — composed in `serving/pipeline.py`
*(`calibration/apply_calibration.py` + `calibration/list_completeness.py`, called from the pipeline, rather than a `serving/confidence_postprocess.py` wrapper. The steps below are all implemented; a serving-side wrapper around two calibration modules would be a second place for the output contract to be shaped, and the contract has to have exactly one.)*
- Applies SPEC_09 calibration to raw logprob confidence, adds the **list-completeness** signal, shapes the final output contract (master §9), and flags sub-threshold fields for review (default threshold 0.7).
- Emits `line_of_business` in the standard `{value, confidence}` shape (arch §0b), `null` when undetermined.
- **Validates the final JSON against the doc_type schema before returning** — this mirrors the Fideon SPEC_07 Stage 3 audit gate (arch §0a). A schema-invalid response is an error, not a returned result.
- Never returns raw (uncalibrated) confidence as if calibrated — SPEC_09 raises when calibration is missing.

### 6. `serving/pipeline.py` (request orchestrator)
Per request: (OCR if provided) → classify → route adapter/prompt/schema → (page-route if long) → SPEC_07 inference → SPEC_09 confidence → schema validation → final JSON.

**This is the canonical pipeline the testing harness (SPEC_12) reuses**, which is what guarantees test == prod. Testing must call it, never re-implement it.

## Constraints
- Non-GPU orchestration runs **outside** RunPod; the RunPod handler stays scoped to inference (arch §14). Document the boundary.
- Deterministic adapter/prompt/schema selection.
- Calibrated (never raw) confidence in responses.
- Every response schema-validated before return.
- MinerU version parity with the training corpus asserted.
- No PII in logs; no persisted raw request/response bodies.

## Acceptance checklist
- [ ] Endpoint serves a request end-to-end → schema-shaped JSON + calibrated confidence + `line_of_business`.
- [ ] The correct per-type adapter is hot-swapped from the classification result.
- [ ] ACORD classification returns both `doc_type` and `acord_form`, and the form selects the schema.
- [ ] Low classifier confidence → **Foundation-only fallback + review flag**, never a silent guess.
- [ ] A long policy doc triggers page routing, records `pages_used`, and sends its selected pages in **one** interleaved call; a short doc skips routing.
- [ ] A routed request's user turn is a **subsequence** of the full document's — the same blocks, fewer pairs. Comparing system prompts alone does not catch a divergence here.
- [ ] `pages_used` reflects the pages actually sent, including for `image_only` documents that carry no per-page OCR text.
- [ ] A page cap never discards the first page or the declarations page, and never reports a `declarations_page` the model was not shown.
- [ ] A schema-invalid generation is rejected, not returned.
- [ ] Missing calibration for the served version raises rather than returning raw confidence.
- [ ] MinerU version mismatch against the corpus pin fails loudly.
- [ ] No PII in logs; no raw bodies persisted.
- [ ] `serving/pipeline.py` is the single path reused by SPEC_12.
