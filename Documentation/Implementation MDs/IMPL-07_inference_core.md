# SPEC 07 — Inference Core (Shared Model-Runner Primitive)

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02. **This module exists to break the cycle between evaluation, calibration, serving, and testing** — they all need one shared way to run a model version over a document. Build it here, once; everything downstream imports it.
>
> **Naming hazard (master §0.1):** in this folder `IMPL-07` is the inference core. The architecture's "**Fideon SPEC_07 Stage 3**" is the external production **audit gate** — a different document. They are unrelated.
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §5 (logprobs → field spans), §6 (modality modes), §7 (prompt template identity, multi-page inputs), §11 (resolution cap).

## Why this module exists

Evaluation (IMPL-08), calibration (IMPL-09), serving (IMPL-11), and testing (IMPL-12) must all run the model **identically** — same prompt assembly, same generation config, same logprob capture — or their numbers won't agree and "test == prod" breaks. Divergence between training-time and inference-time prompts is the single most common cause of post-fine-tuning degradation, and it is invisible in training metrics because it only manifests at serving time (arch §7). That shared primitive lives here, with **no dependency on calibration, classification, or routing** — those are higher-level concerns layered on later.

## Goal

A clean, low-level inference engine: given a resolved model version + a prepared document input + a doc_type, produce raw generated JSON text and per-token logprobs. No confidence calibration, no classification, no routing here — just correct, reproducible generation.

## Deliverables

### 1. `inference_core/model_runner.py`
- `load_model(version, backend="vllm"|"hf"|"gguf")` — resolves the version to concrete artifacts via IMPL-02 `resolve_model_version`, and loads the **merged** model, optionally with ONE graduated per-type adapter applied per request. Backend-swappable: vLLM for serving, HF for local eval, GGUF for the offline/edge path only.
- `generate(model, messages, want_logprobs=True, **gen_kwargs)` → `{text, tokens, token_logprobs}`.
  - **Logprobs are mandatory-capable** — vLLM supports them natively and HF via `generate(output_scores=True)`. Calibration (IMPL-09) is built entirely on them, so a backend that cannot return them is not a valid backend for anything but smoke testing.
- Deterministic given seed + greedy/temperature settings; logs and returns the generation config so an output can be reproduced.
- **Adapter stacking:** loading Foundation + one per-type LoRA on top is the normal case (arch §4). Foundation-only loading must also work — it is the classifier's low-confidence fallback path (IMPL-11).

### 2. `inference_core/input_builder.py`
- `build_messages(doc_type, image_paths, ocr_pages, modality_mode, acord_form=None, page_numbers=None, total_pages=None)` → the exact chat-format `messages` (master §9), rendered through **`common.prompts`**. This is the single place prompt + image + OCR get assembled, so dataset build, evaluation, serving, and testing are guaranteed identical.
  - **`ocr_pages` is one markdown string per image, not one joined blob.** A mismatched length is refused: a short or long list shifts every page's text onto the wrong image, and every value after the shift is then attributed to the wrong page.
  - The user turn is **interleaved** — `[img_1][<page 1 of M> + md_1][img_2][<page 2 of M> + md_2]…` — so which text belongs to which image is positional rather than something the model infers. Joining destroyed that, and (because a blank line ends a Markdown table) split every table crossing a page boundary in two.
  - `page_numbers` / `total_pages` let a routed request label pages truthfully (`<page 9 of 20>`). The gap in the numbering is what tells the model it holds a fragment, so fields living on unshown pages are absent rather than missing.
  - `image_only` carries the markers and no markdown. The guarantee is *no OCR text*, not *no text at all*.
- Handles `image_only` (omit the OCR block, use the image-only prompt with its explicit "no OCR provided" declaration) vs `ocr_plus_image` (arch §6).
- **Multi-page inputs:** accepts an ordered list of page images and emits them as ordered `image` blocks with concatenated OCR text. **Page order is positionally meaningful** under Interleaved-MRoPE (arch §3) — ordering must be explicit and stable, never set-like.
- Applies the **resolution cap from config** to any image passed in — identical to the cap used at corpus build (arch §11). Assert the cap matches `configs/base_model.yaml`; a silent mismatch between training prep and inference is a correctness bug, not a tuning knob. The vLLM engine additionally receives the **pixel budget** as `mm_processor_kwargs` from `common.config.pixel_budget()`, the function training takes its budget from.
- A page whose OCR text is blank gets `EMPTY_PAGE_TEXT` — the same placeholder in training rows and serving requests.
- Exposes `prompt_template_version` and `schema_version` on the built message set so callers can verify they match the corpus the model was trained on.

### 3. `inference_core/span_map.py`
- `map_field_spans(generated_text, tokens, token_logprobs)` → for each JSON field path — **including nested fields and individual list rows** (e.g. `claims[0].amount`) — the token span and its logprobs.
- This is the raw material calibration (IMPL-09) turns into confidence. Pure and testable; contains **no calibration logic itself**.
- Must handle: values spanning multiple tokens, `null` values, numeric vs string values, and rows inside arrays. A field whose span cannot be mapped is reported explicitly rather than silently dropped — an unmapped field would otherwise get no confidence and look like a clean extraction.

### 4. `inference_core/runner_config.py`
- Central generation + backend config (model tag, backend, temperature, `max_new_tokens`, logprob settings, seed), loaded from `configs/inference/`. One source of truth so eval, serving, and testing cannot drift on generation parameters either.

## Constraints
- **No imports from** calibration, serving, evaluation, or testing (they import this, not vice-versa).
- Backend-agnostic interface so the same calls work in every context.
- Resolution cap and prompt rendering come from config/`common.prompts` — never inline.
- No PII in logs (never log generated field values or OCR text).

## Acceptance checklist
- [ ] `load_model("v2")` resolves + loads a version (foundation+adapter and merged paths both work); Foundation-only loading also works.
- [ ] `build_messages(...)` yields **byte-identical** structure for eval, serving, and testing given the same inputs.
- [ ] Each page image is immediately followed by that page's own text, and every text block carries its `<page N of M>` marker.
- [ ] A **routed subset is a subsequence of the full document** — the same blocks, fewer pairs — so a long policy is served in the shape it trained on. Comparing only *system* prompts does not catch this; the user turn must be compared too.
- [ ] A mismatch between the page-image count and the page-text count is refused.
- [ ] Two pages' markdown is never concatenated, so a table crossing a page boundary is not split.
- [ ] `build_messages(...)` rendering matches what IMPL-05 wrote into the corpus for the same doc_type + modality_mode (round-trip test against a corpus row).
- [ ] Multi-page input produces ordered image blocks; reversing the input order changes the output message order.
- [ ] Resolution cap mismatch between `base_model.yaml` and `vllm_serving.yaml` raises.
- [ ] `generate(...)` returns text + aligned token logprobs.
- [ ] `map_field_spans(...)` correctly maps a scalar field, a `null` field, and a `claims[i].amount` list-row field on a sample generation; an unmappable field is reported, not dropped.
- [ ] Module imports without pulling in calibration/serving/eval/testing.

---

## Current implementation (2026-09-27)

- **vLLM backend**: refuses to build an engine without CUDA (`common.gpu.require_cuda`); passes
  `mm_processor_kwargs={"min_pixels", "max_pixels"}` from `pixel_budget()`; `close()` releases the engine
  (tears down vLLM's parallel state, collects, empties the CUDA cache) and `release_model(model)` calls it.
  The pipeline releases each engine before the next stage loads one — checkpoint selection, calibration
  per format and the golden eval each load a model on one card.
- **HF backend**: `device_map="cuda"`, never `"auto"` (which silently offloads layers to the CPU when VRAM
  runs short); refuses without CUDA.
- **Base weights** resolve to the pod's local copy (`common.config.base_model_source()`); the registry used
  to hand vLLM the identity string `model_id@revision`, which no loader accepts.
- **Dates**: generations are post-processed to `MM/DD/YYYY` (`common.canonical.with_output_dates`); a field is
  a date by a whole word of its name (`date`, `dates`, `dated`, `dob`), list elements included.
- **Pinned**: vLLM `==0.11.0` on both the training and the serving pod — calibration is fitted on the logprobs
  one build produces, so serving runs the same build. 0.11 still has `GuidedDecodingParams`; moving to
  `structured_outputs` is required before raising the pin.
