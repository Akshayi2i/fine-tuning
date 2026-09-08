# SPEC 10 — Postprocessing (Merge Adapter + GGUF Quantize)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_06. (Independent of the inference/eval/calibration chain — only needs a trained adapter.)
>
> **Architecture refs:** `finetuning-architecture-v1.md` §13 steps 7–9, §13a (**user-selectable GGUF format matrix**), §13b (quantization quality thresholds — reference only this cycle), §18 (artifact paths).

> **This spec is split across two operator commands (SPEC_13).** `merge_adapter` is the last stage of **command 1 (`finetune`)** and writes to the RunPod staging volume. `quantize` is the first stage of **command 2 (`package`)**, which then pushes adapters, merged model, and every quantized format to Azure Blob. Build both here; the command boundary is orchestration's concern, not this module's.

## Goal

Merge a trained LoRA adapter into the base model and export user-selectable GGUF quantization formats. Numeric quality thresholds are specified here and enforced when a quantized format is actually served — not in the first cycle, whose serving path is vLLM on the merged fp16/bf16 model.

## Deliverables

### 1. `postprocessing/merge_adapter.py`
- Loads base model + a Foundation adapter (and optionally a stacked per-type adapter); runs PEFT `merge_and_unload()` → a full fp16/bf16 merged model.
- Writes to the **staging volume** at `/runpod-volume/staging/merged-models/{doc_type|unified}/v{n}/`. `package` pushes it to `merged-models/{doc_type|unified}/v{n}/` in Blob (SPEC_02).
- Supports **Foundation-only (unified)** or **Foundation + per-type** merges depending on serving strategy. Merged models never overwrite adapters — separate paths.
- CLI: `--foundation vN [--adapter doc_type vN] --out-version vN --dtype fp16|bf16`.

### 2. `postprocessing/quantize.py`
Quantization is a **configurable export step, not a single fixed choice** (arch §13a). All formats derive from the same fp16 GGUF conversion, so one or many can be exported in one run.

**Pipeline:** merged fp16/bf16 → base GGUF (`convert_hf_to_gguf.py` from llama.cpp) → `llama-quantize` per requested format.

```
python postprocessing/quantize.py --model v2 --formats fp16 q6_k q5_k_m q4_k_m
python postprocessing/quantize.py --model v2 --formats q4_k_m
```

**Format matrix (arch §13a):**

| Format | Bits/weight | Relative size¹ | Quality | Typical use |
|---|---|---|---|---|
| **FP16** | 16 | 100% (~16 GB) | Reference / lossless | Accuracy baseline; what every quantized variant is measured against |
| **BF16** | 16 | 100% (~16 GB) | Reference / lossless | Same size, wider dynamic range; preferred baseline on native-bf16 hardware |
| **Q8_0** | 8 | ~53% | Near-lossless | Maximum quality at half the fp16 memory |
| **Q6_K** | ~6.6 | ~44% | Very high | Strong quality/size balance |
| **Q5_K_M** | ~5.7 | ~38% | High | **Default serving target** |
| **Q4_K_M** | ~4.8 | ~32% | Good | Most memory-efficient; only under VRAM constraint |

¹ Relative to the fp16 merged model. Actual sizes vary — measure on the real merged model.

**What to produce routinely:** supporting all six is a capability, not a per-cycle obligation. Converge on **fp16 as the accuracy baseline + one serving format** produced and validated every cycle; generate Q8_0, Q6_K, BF16 **on demand** when a specific deployment target calls for them. Regenerating and re-validating all six every cycle wastes eval compute. Make this the CLI default, not just a docstring.

**Multimodal (VL) handling — verify, don't assume (arch §13a):** Qwen3-VL is multimodal, so the GGUF export must handle the vision encoder + projector as well as the LLM. In the llama.cpp ecosystem this means producing the quantized LLM GGUF **plus a separate `mmproj` file**, served together. **Confirm current llama.cpp support for the exact Qwen3-VL version at implementation time.** If vision-side GGUF support lags:
- warn clearly and refuse to emit a silently-broken model,
- keep the **vLLM path primary** (serving the merged fp16/bf16, or an AWQ/FP8 variant),
- treat GGUF as the portable / edge / offline option.

- Reads the merged model from the staging volume, writes each format back to staging, and `package` pushes them to `quantized-models/{doc_type|unified}/v{n}/gguf/{format}/`; record `quantized_formats` in the RunManifest (SPEC_02).
- CLI: `--model vN --formats ...`, `--mmproj/--no-mmproj`, `--dry-run`.

### 3. Quantization quality thresholds — `postprocessing/quant_thresholds.py`

The primary serving path is **vLLM on the merged fp16/bf16 model** (arch §13a), so the first cycle ships nothing quantized and the gate has nothing to fire on. The table is now **data in one module** rather than prose here, because it is explicitly provisional — revised once absolute F1 values are known, since a 2% relative drop means something different at F1 0.95 than at 0.70 — and revising it must be one edit.

"Re-validate before promotion" needs numbers attached or it degrades into a judgement call per release. **Acceptable degradation relative to the fp16 reference:**

| Format | Max field F1 drop | Max ECE increase | Min JSON validity |
|---|---|---|---|
| FP16 | Reference (0%) | Reference | 100% |
| BF16 | ≤ 0.5% | ≤ 0.005 | 100% |
| Q8_0 | ≤ 1.0% | ≤ 0.010 | 100% |
| Q6_K | ≤ 1.5% | ≤ 0.015 | ≥ 99.5% |
| Q5_K_M | ≤ 2.0% | ≤ 0.020 | ≥ 99.5% |
| Q4_K_M | ≤ 4.0% | ≤ 0.030 | ≥ 99.0% |

- A format exceeding **any** threshold is **not promoted to serving**.
- **Q5_K_M is the default serving target**; Q4_K_M is used only under VRAM constraint and only when it meets threshold.
- These are initial pilot-cycle targets, **revised after the first full evaluation cycle** once absolute F1 values are known — a 2% relative drop means something different at F1 0.95 than at 0.70. Keep them in this module as data, not scattered constants, so revising them is one edit.

### 4. `postprocessing/validate_quant.py`
- Runs each produced GGUF through the extraction/testing routine (SPEC_12) against the **frozen golden eval set**, and evaluates the result against the threshold table.
- **Every quantized format intended for serving must pass before promotion.** Lower-bit quantization degrades exactly the behaviors that were fine-tuned in — strict JSON structure, precise field values (dates, currency, policy numbers), and confidence calibration. It is common to find fp16 and Q6_K statistically indistinguishable while Q4_K_M drops a point or two on field-exact-match; whether that trade is acceptable is a per-deployment decision, which is why all formats stay available rather than one being hardcoded.
- Reports the accuracy delta vs fp16 per format, writes `quant_threshold_results` (pass/fail per format) into the RunManifest, and **refuses promotion of an over-threshold format**.
- Build-order note: this calls into SPEC_12, and it only matters once a quantized format is being served. Build merge + quantize first; wire this when GGUF actually ships.

## Constraints
- Merged models never overwrite adapters (separate Blob paths).
- Every **served** quantized format passes threshold validation first — no exceptions, no override flag. (Moot in the first cycle: nothing is served quantized.)
- Record all produced formats in the manifest.
- Never emit a GGUF whose multimodal projector support is unverified without a loud warning.

## Acceptance checklist
- [ ] `merge_adapter` produces a loadable merged model (foundation-only and foundation+per-type).
- [ ] `quantize --formats fp16 q4_k_m` produces both GGUFs **plus the `mmproj` file**.
- [ ] Default invocation produces fp16 + one serving format, not all six.
- [ ] An immature / unsupported VL-GGUF path warns clearly and does not emit a silently broken model.
- [ ] `merge_adapter` writes to the staging volume; `package` lands adapters, merged model, and every format at the correct Blob paths and flips the manifest to `"published"`.
- [ ] `quant_thresholds` returns pass/fail correctly **at each format's exact boundary** — a format landing precisely on its allowance passes, and floating-point recomputation must not block it.
- [ ] `validate_quant` reports the delta vs fp16 per format, writes `quant_threshold_results` into the RunManifest, and refuses an over-threshold format **with no override path**.
- [ ] A format that was not measured does not pass, and validation without the fp16 reference is refused outright.
- [ ] **`assert_servable` refuses a serving format that produced no verdict at all.** `validate_quant` only returns a result for formats it was given metrics for, so a format nobody scored appears in neither `servable_formats` nor `blocked_formats`. Checking `blocked_formats` alone let it sail through to push with zero measurements — the precise inverse of the rule this module exists to enforce.
- [ ] The gate sits inside `package`, between quantize and push.
