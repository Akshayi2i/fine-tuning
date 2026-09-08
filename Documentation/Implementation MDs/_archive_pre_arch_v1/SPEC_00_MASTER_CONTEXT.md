# SPEC 00 — Master Context (Read First)

> **Purpose of this file.** This is the shared context every other spec assumes. When you run any individual spec (SPEC_01 … SPEC_13) in Claude Code, that spec references this file for global conventions. Read this once; each module spec then adds only what's specific to it.
>
> **Source of truth.** This spec set is derived from the architecture document `qwen3vl-insurance-extraction-finetuning-architecture.md`. Where a spec cites a section like "(§4)", it refers to that architecture document. If a spec and the architecture doc ever disagree, the architecture doc wins — flag the discrepancy rather than guessing.

---

## 1. What is being built

A production-grade fine-tuning and extraction system for **insurance document extraction** using **Qwen3-VL-8B-Instruct**. The system ingests insurance PDFs (digital and scanned), runs OCR (MinerU), and uses a fine-tuned vision-language model to extract structured **JSON with per-field confidence scores**.

**Active document types (only these three for now):** `acord`, `policy`, `lossrun`.
Quote and Endorsement are deferred — design for easy addition but do not implement them.

**Two production inference modes, one model:**
1. `ocr_plus_image` — MinerU OCR text + page image, both passed to the model.
2. `image_only` — page image only, no OCR (model must still extract).

---

## 2. Core architectural decisions (do not re-litigate — implement as stated)

| Decision | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-8B-Instruct`, pinned HF revision |
| Fine-tuning technique | **QLoRA** — 4-bit NF4 quantized base, LoRA adapters in bf16 |
| Trainable components | **Projector + LLM decoder** via LoRA. **Vision Encoder (ViT) frozen by default** — unfreeze only via the eval gate (§2 of arch doc). |
| Adapter strategy | **Hybrid**: one shared **Foundation LoRA** (rank 64, alpha 128) trained across all doc types + all 3 modality regimes, then small **per-type LoRA** adapters (rank 16, alpha 32) stacked on top. |
| Trainer stack | **ms-swift** (entrypoint) → TRL `SFTTrainer`/HF `Trainer` (loop) → PyTorch/PEFT/bitsandbytes/DeepSpeed. |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` + vision-language projector/merger. |
| Attention impl | `flash_attention_2`. |
| Confidence | Per-token logprobs → per-field aggregation → **post-hoc calibration** (temperature scaling / isotonic). List fields also get a **row-completeness** signal. |
| Quantization | **GGUF**, user-selectable format (`fp16|bf16|q8_0|q6_k|q5_k_m|q4_k_m`). Routinely produce/validate only **fp16 baseline + one serving format** (q5_k_m or q4_k_m). |
| Artifact storage | **Azure Blob** — all weights/corpus/PDFs. Repo holds code only. |
| Compute | **RunPod** — ephemeral training pods + a persistent Serverless vLLM inference endpoint. Business logic lives outside RunPod. |

---

## 3. The `source_id` join key (critical convention)

Every source document has a stable `source_id` (e.g. `acord_0001`, `lossrun_0042`, `policy_0007`). This **same id** appears at every layer — raw PDF, OCR output, golden label, every compiled training row, and every extraction result. It is the traceability spine. Never generate data at any layer without carrying its `source_id`.

Naming: `{doc_type}_{zero_padded_index}`, e.g. `acord_0001`. ACORD form number is tracked as a separate field (`acord_form: "25" | "125" | "140"`), not baked into the doc_type.

---

## 4. Azure Blob layout (the data/artifact side — code never stores these locally)

```
azure-blob://insurance-extraction/
  raw-documents/{doc_type}/{source_id}/original.pdf, metadata.json   # immutable, PII-sensitive, tighter RBAC
  processed/{doc_type}/{source_id}/page_*.png, page_*.md             # MinerU output + rendered images
  golden-labels/{doc_type}/{source_id}/golden.json, label_metadata.json
  corpus/v{n}/{doc_type}/{train,val,test}.jsonl
  base-models/qwen3-vl-8b-instruct/                                  # cached from HF, pinned revision
  adapters/foundation/v{n}/  ,  adapters/{doc_type}/v{n}/            # per-type tagged w/ dependent foundation version
  merged-models/{doc_type|unified}/v{n}/                             # fp16/bf16
  quantized-models/{doc_type|unified}/v{n}/gguf/{format}/
  registry/foundation/{run_id}/run_manifest.json
  registry/adapters/{doc_type}/{run_id}/run_manifest.json
  registry/registry_index.json
  eval-reports/v{n}/
  golden-eval-set/                                                   # frozen, versioned separately
```

---

## 5. Repository structure (code side — this is what the specs build)

```
insurance-extraction-finetuning/
├── README.md
├── pyproject.toml / requirements.txt
├── .env.example
├── configs/            base_model.yaml, training/*.yaml, deepspeed/*.json, inference/vllm_serving.yaml   → SPEC_01
├── schemas/            {acord25,acord125,lossrun,policy_doc}.schema.json                                  → SPEC_01
├── prompts/            system_prompt_template.jinja, doc_type_classifier_prompt.jinja                     → SPEC_01
├── common/             config, schemas, prompts, ids, constants, normalize                                → SPEC_01 (+ normalize in 08)
├── artifact_registry/  push_to_blob.py, pull_from_blob.py                                                 → SPEC_02
├── registry_utils/     write_run_manifest.py, query_registry.py                                           → SPEC_02
├── data_pipeline/
│   ├── ingestion/      pull_raw_pdfs.py                                                                   → SPEC_03
│   ├── ocr/            run_mineru.py, render_only.py                                                       → SPEC_03
│   ├── labeling/       pre_annotate.py, export_golden_labels.py, review_tool/, active_learning.py         → SPEC_04
│   ├── dataset_builder/ build_jsonl.py, modality_dropout.py, noisy_ocr_augment.py, split_train_val_test.py → SPEC_05
│   └── corpus_manifest.py                                                                                 → SPEC_05
├── training/           train_foundation.py, train_adapter.py, data_collator.py, vit_gate.py, callbacks/   → SPEC_06
├── inference_core/     model_runner.py, input_builder.py, span_map.py, runner_config.py                   → SPEC_07  (shared primitive)
├── evaluation/         run_eval.py, metrics/*, gating.py                                                  → SPEC_08
├── calibration/        logprob_confidence.py, fit_calibration.py, apply_calibration.py, list_completeness.py → SPEC_09
├── postprocessing/     merge_adapter.py, quantize.py, validate_quant.py                                   → SPEC_10
├── serving/            vllm_entrypoint.py, doc_type_classifier.py, adapter_router.py, page_router.py,
│                       confidence_postprocess.py, pipeline.py                                             → SPEC_11
├── testing/            run_extraction.py, prompts/, (test_data, ocr_cache, results, metrics, registry)   → SPEC_12
├── orchestration/      runpod_controller.py, pipeline_dag.py                                              → SPEC_13
└── tests/              test_*.py, fixtures/                                                               → SPEC_14
```

**Key separation principle:** the repo never stores model weights, corpus data, or PDFs. Everything data/artifact-related is pulled from / pushed to Azure Blob at runtime via `artifact_registry/`.

---

## 6. Spec build order & dependency graph

Build in this order — later specs import from earlier ones. **SPEC_07 (inference core) is deliberately placed before evaluation/calibration/serving/testing because all four share it** — this breaks what would otherwise be a circular dependency between them.

```
SPEC_01  Project scaffold, configs, schemas, prompts, common   (no deps)
SPEC_02  Azure Blob I/O + training run registry                (deps: 01)
SPEC_03  Ingestion + MinerU OCR                                (deps: 01, 02)
SPEC_04  Labeling + golden JSON generation                     (deps: 01, 02, 03)
SPEC_05  Dataset builder (JSONL, modality, split)              (deps: 01, 02, 04)
SPEC_06  Training (ms-swift QLoRA, foundation + adapters)      (deps: 01, 02, 05)
SPEC_07  Inference core (shared model-runner primitive)        (deps: 01, 02)
SPEC_08  Evaluation (metrics + gating, + common.normalize)     (deps: 01, 02, 06, 07)
SPEC_09  Confidence calibration                                (deps: 01, 02, 07, 08)
SPEC_10  Postprocessing (merge + GGUF quantize)                (deps: 01, 02, 06)
SPEC_11  Serving (vLLM endpoint, classifier, routing)          (deps: 01, 02, 07, 09, 10)
SPEC_12  Testing / extraction routine (CLI harness)            (deps: 01, 02, 03, 07, 08, 09, 11)
SPEC_13  Orchestration (RunPod controller, pipeline DAG)       (deps: all above)
SPEC_14  Unit tests + CI fixtures                              (deps: all above)
```

**Note on two cross-references that point "forward" but are not build-order violations:**
- `postprocessing/validate_quant.py` (SPEC_10) calls the testing routine (SPEC_12). This is fine — `validate_quant` is a *validation gate you run after SPEC_12 exists*, not something the core merge/quantize functions need. Build merge+quantize in SPEC_10; wire `validate_quant` once SPEC_12 is done.
- `evaluation/run_eval.py` (SPEC_08) can compute classifier-accuracy only once the classifier (SPEC_11) exists. SPEC_08 is built to run *without* it (supply the true doc_type), and classifier metrics fill in on a later pass. Not a blocker.

Everything else is strictly bottom-up.

---

## 7. Global conventions every spec must follow

- **Language/runtime:** Python 3.11+. Use type hints throughout. Prefer `pydantic` v2 for config/data models.
- **Config:** all runtime config via YAML in `configs/` + environment variables (via `.env`, loaded with `python-dotenv`). Never hardcode paths, model ids, or credentials.
- **Secrets:** Azure + RunPod credentials come from env vars only. `.env.example` lists every required var with placeholder values. Never commit real secrets.
- **Logging:** use structured logging (`structlog` or stdlib `logging` with JSON formatter). **Never log PII** (raw OCR text, extracted field values, image bytes) at INFO or above — see PII rules below.
- **Blob access:** all Azure Blob reads/writes go through `artifact_registry/` (SPEC_02). No module opens the Azure SDK directly except that module.
- **Determinism:** set and log random seeds for any stochastic step (splitting, training, augmentation).
- **CLIs:** every runnable module exposes a `argparse`/`typer` CLI with `--help`. Long-running scripts print progress.
- **Error handling:** fail loudly with clear messages on missing config, missing Blob artifacts, or schema-invalid data. Don't silently continue.
- **No network assumptions:** code may run inside a RunPod pod with a restricted egress allowlist; make external endpoints configurable, not hardcoded.
- **Idempotency:** ingestion, OCR, and dataset-build steps should be safe to re-run — skip or overwrite deterministically based on `source_id` + content checksum, never duplicate.

## 8. PII handling (non-negotiable, applies to every spec)

Insurance documents contain PII (names, TINs/SSNs, addresses, financials).
- `raw-documents/` is the most sensitive layer — tighter RBAC, encryption at rest, immutable.
- Never send PII to third-party APIs unless explicitly permitted (affects labeling pre-annotation in SPEC_04 — default to the self-hosted base model).
- Never persist raw request/response bodies or OCR text in plaintext logs (affects serving SPEC_10, testing SPEC_11).
- Scrub/access-control eval reports and error dumps.

## 9. Data contracts (shared shapes used across specs)

**Training example (one JSONL row):**
```json
{
  "doc_type": "lossrun",
  "acord_form": null,
  "modality_mode": "ocr_plus_image | noisy_ocr_image | image_only",
  "source_id": "lossrun_0042",
  "split": "train | val | test",
  "messages": [
    {"role": "system", "content": "<prompt: schema + modality instructions>"},
    {"role": "user", "content": [{"type": "image", "image": "<path/uri>"}, {"type": "text", "text": "<OCR markdown or omitted>"}]},
    {"role": "assistant", "content": "<target JSON string>"}
  ]
}
```

**Extraction output (per document, at inference/testing):**
```json
{
  "source_id": "abcLossRun",
  "doc_type": "lossrun",
  "model_version": "v2",
  "mode": "ocr_plus_image",
  "schema_valid": true,
  "overall_confidence": 0.88,
  "fields": {
    "<field>": {"value": "...", "confidence": 0.94}
  },
  "list_fields": {
    "<list_field>": {"rows": [...], "row_completeness_confidence": 0.7}
  }
}
```

**Run manifest (registry, per training run):** see SPEC_02 for the full schema.

## 10. Modality-mix for training data (used by SPEC_05)

Within the Foundation corpus, per source document generate 3 rows:
- `ocr_plus_image` — 50%
- `noisy_ocr_image` — 20% (deliberately corrupted OCR, corrected golden JSON)
- `image_only` — 30% (no OCR text block; system prompt says so explicitly)

Split at **source_id level BEFORE** modality expansion to prevent leakage.

---

## 11. How to use these specs with Claude Code

1. Start a Claude Code session at the repo root.
2. Run `SPEC_01` first to scaffold the project.
3. Run each subsequent spec **in numeric order (01 → 14)**. Each spec is self-contained but assumes earlier specs are done. The order is dependency-correct — no spec references code from a higher-numbered spec as a build prerequisite (the two "forward" mentions in SPEC_08 and SPEC_10 are explicitly non-blocking; see §6).
4. Each spec ends with an **Acceptance checklist** — verify it before moving on.
5. This master file is the context; when a spec says "per master context", it means this file.
