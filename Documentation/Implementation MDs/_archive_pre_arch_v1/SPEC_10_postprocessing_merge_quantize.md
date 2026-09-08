# SPEC 10 — Postprocessing (Merge Adapter + GGUF Quantize)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_06. (Independent of the inference/eval/calibration chain — only needs a trained adapter.)

## Goal

Merge a trained LoRA adapter into the base model, then export user-selectable GGUF quantization formats for serving. (Arch §12 steps 7–9, §12a.)

## Deliverables

### 1. `postprocessing/merge_adapter.py`
- Loads base model + a Foundation adapter (and optionally a stacked per-type adapter); runs PEFT `merge_and_unload()` → full fp16/bf16 merged model.
- Pushes to `merged-models/{doc_type|unified}/v{n}/` (SPEC_02).
- Supports Foundation-only (unified) or Foundation+per-type (per doc_type) merges depending on serving strategy.
- CLI: `--foundation vN [--adapter doc_type vN] --out-version vN --dtype fp16|bf16`.

### 2. `postprocessing/quantize.py`
- GGUF export: merged fp16/bf16 → base GGUF (`convert_hf_to_gguf.py`) → `llama-quantize` per format.
- **User-selectable `--formats`**: any subset of `fp16 bf16 q8_0 q6_k q5_k_m q4_k_m` in one run.
- **Default:** produce `fp16` (baseline) + one serving format (`q5_k_m` or `q4_k_m`); others on demand (arch §12a convergence).
- **Multimodal (VL) handling:** Qwen3-VL needs vision encoder + projector exported — produce the quantized LLM GGUF **plus the `mmproj` file**, served together. **Verify current llama.cpp Qwen3-VL support at implementation time**; if missing/immature, warn clearly and keep vLLM (merged fp16/bf16) primary while GGUF is the portable/edge path. (Arch §12a.)
- Push each format to `quantized-models/{doc_type|unified}/v{n}/gguf/{format}/`; record `quantized_formats` in the RunManifest.
- CLI: `--model vN --formats q5_k_m q4_k_m fp16`.

### 3. `postprocessing/validate_quant.py`
- Runs each produced GGUF through the extraction/testing routine (SPEC_12) against the golden eval set — **every served format must be re-validated** (lower-bit quant can degrade JSON structure / field precision / calibration). Reports accuracy delta vs fp16. (Arch §12a.)
- Note: this calls into SPEC_12; run it after SPEC_12 exists, or as the validation gate before promoting a quantized format to serving.

## Constraints
- Merged models never overwrite adapters (separate paths).
- Every served quantized format passes validation first.
- Record all produced formats in the manifest.

## Acceptance checklist
- [ ] `merge_adapter` produces a loadable merged model (foundation-only and foundation+per-type).
- [ ] `quantize --formats fp16 q4_k_m` produces both GGUFs + the mmproj file.
- [ ] Immature VL-GGUF path warns clearly, no silent broken model.
- [ ] Artifacts land at correct Blob paths; manifest lists formats.
- [ ] `validate_quant` reports accuracy delta vs fp16 for each format.
