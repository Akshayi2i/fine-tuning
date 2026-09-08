# SPEC 07 — Inference Core (Shared Model-Runner Primitive)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02. **This module exists to break the cycle between evaluation, calibration, serving, and testing** — they all need one shared way to run a model version over a document. Build it here, once; everything downstream imports it.
>
> **Naming hazard (master §0.1):** in this folder `SPEC_07` is the inference core. The architecture's "**Fideon SPEC_07 Stage 3**" is the external production **audit gate** — a different document. They are unrelated.
>
> **Architecture refs:** `finetuning-architecture-v1.md` §5 (logprobs → field spans), §6 (modality modes), §7 (prompt template identity, multi-page inputs), §11 (resolution cap).

## Why this module exists

Evaluation (SPEC_08), calibration (SPEC_09), serving (SPEC_11), and testing (SPEC_12) must all run the model **identically** — same prompt assembly, same generation config, same logprob capture — or their numbers won't agree and "test == prod" breaks. Divergence between training-time and inference-time prompts is the single most common cause of post-fine-tuning degradation, and it is invisible in training metrics because it only manifests at serving time (arch §7). That shared primitive lives here, with **no dependency on calibration, classification, or routing** — those are higher-level concerns layered on later.

## Goal

A clean, low-level inference engine: given a resolved model version + a prepared document input + a doc_type, produce raw generated JSON text and per-token logprobs. No confidence calibration, no classification, no routing here — just correct, reproducible generation.

## Deliverables

### 1. `inference_core/model_runner.py`
- `load_model(version, backend="vllm"|"hf"|"gguf")` — resolves the version to concrete artifacts via SPEC_02 `resolve_model_version`, and loads either base + Foundation (+ optional per-type adapter) or a merged/quantized model. Backend-swappable: vLLM for serving, HF for local eval, GGUF for the portable/edge path.
- `generate(model, messages, want_logprobs=True, **gen_kwargs)` → `{text, tokens, token_logprobs}`.
  - **Logprobs are mandatory-capable** — vLLM supports them natively and HF via `generate(output_scores=True)`. Calibration (SPEC_09) is built entirely on them, so a backend that cannot return them is not a valid backend for anything but smoke testing.
- Deterministic given seed + greedy/temperature settings; logs and returns the generation config so an output can be reproduced.
- **Adapter stacking:** loading Foundation + one per-type LoRA on top is the normal case (arch §4). Foundation-only loading must also work — it is the classifier's low-confidence fallback path (SPEC_11).

### 2. `inference_core/input_builder.py`
- `build_messages(doc_type, image_paths, ocr_pages, modality_mode, acord_form=None, page_numbers=None, total_pages=None)` → the exact chat-format `messages` (master §9), rendered through **`common.prompts`**. This is the single place prompt + image + OCR get assembled, so dataset build, evaluation, serving, and testing are guaranteed identical.
  - **`ocr_pages` is one markdown string per image, not one joined blob.** A mismatched length is refused: a short or long list shifts every page's text onto the wrong image, and every value after the shift is then attributed to the wrong page.
  - The user turn is **interleaved** — `[img_1][<page 1 of M> + md_1][img_2][<page 2 of M> + md_2]…` — so which text belongs to which image is positional rather than something the model infers. Joining destroyed that, and (because a blank line ends a Markdown table) split every table crossing a page boundary in two.
  - `page_numbers` / `total_pages` let a routed request label pages truthfully (`<page 9 of 20>`). The gap in the numbering is what tells the model it holds a fragment, so fields living on unshown pages are absent rather than missing.
  - `image_only` carries the markers and no markdown. The guarantee is *no OCR text*, not *no text at all*.
- Handles `image_only` (omit the OCR block, use the image-only prompt with its explicit "no OCR provided" declaration) vs `ocr_plus_image` (arch §6).
- **Multi-page inputs:** accepts an ordered list of page images and emits them as ordered `image` blocks with concatenated OCR text. **Page order is positionally meaningful** under Interleaved-MRoPE (arch §3) — ordering must be explicit and stable, never set-like.
- Applies the **resolution cap from config** to any image passed in — identical to the cap used at corpus build (arch §11). Assert the cap matches `configs/base_model.yaml`; a silent mismatch between training prep and inference is a correctness bug, not a tuning knob.
- Exposes `prompt_template_version` and `schema_version` on the built message set so callers can verify they match the corpus the model was trained on.

### 3. `inference_core/span_map.py`
- `map_field_spans(generated_text, tokens, token_logprobs)` → for each JSON field path — **including nested fields and individual list rows** (e.g. `claims[0].amount`) — the token span and its logprobs.
- This is the raw material calibration (SPEC_09) turns into confidence. Pure and testable; contains **no calibration logic itself**.
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
- [ ] `build_messages(...)` rendering matches what SPEC_05 wrote into the corpus for the same doc_type + modality_mode (round-trip test against a corpus row).
- [ ] Multi-page input produces ordered image blocks; reversing the input order changes the output message order.
- [ ] Resolution cap mismatch between `base_model.yaml` and `vllm_serving.yaml` raises.
- [ ] `generate(...)` returns text + aligned token logprobs.
- [ ] `map_field_spans(...)` correctly maps a scalar field, a `null` field, and a `claims[i].amount` list-row field on a sample generation; an unmappable field is reported, not dropped.
- [ ] Module imports without pulling in calibration/serving/eval/testing.
