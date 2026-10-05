# Insurance Document Extraction — Full Build Spec (Combined)
### Qwen3-VL-8B-Instruct Fine-Tuning & Extraction System

> **This is the combined spec file** containing the master context plus all 14 module specs in dependency-correct build order. Single-file companion to the individual `IMPL-00 … IMPL-14` files. Build one module at a time from the individual files; use this for the whole picture.
>
> **Build order (dependency-verified):** 01 scaffold -> 02 blob/registry -> 03 ingestion/OCR -> 04 labeling -> 05 dataset -> 06 training -> 07 inference-core -> 08 evaluation -> 09 calibration -> 10 postprocessing -> 11 serving -> 12 testing -> 13 orchestration -> 14 tests.
>
> **Key ordering fix:** IMPL-07 (inference core) is a shared model-runner primitive placed before evaluation/calibration/serving/testing, because all four use it. This removes the circular dependency those modules would otherwise have.

---

## Table of Contents
- SPEC 00 — Master Context (read first)
- SPEC 01 — Project Scaffold, Configs, Schemas, Prompts
- SPEC 02 — Azure Blob I/O + Training Run Registry
- SPEC 03 — Ingestion + MinerU OCR
- SPEC 04 — Labeling + Golden JSON Generation
- SPEC 05 — Dataset Builder (JSONL, Modality, Split)
- SPEC 06 — Training (ms-swift QLoRA)
- SPEC 07 — Inference Core (shared model-runner primitive)
- SPEC 08 — Evaluation (Metrics + Gating)
- SPEC 09 — Confidence Calibration
- SPEC 10 — Postprocessing (Merge + GGUF Quantize)
- SPEC 11 — Serving (vLLM + Classifier + Routing)
- SPEC 12 — Testing / Extraction Routine
- SPEC 13 — Orchestration (RunPod + Pipeline DAG)
- SPEC 14 — Unit Tests + CI

---



===================================================================================

# SPEC 00 — Master Context (Read First)

> **Purpose of this file.** This is the shared context every other spec assumes. When you run any individual spec (IMPL-01 … IMPL-13) in Claude Code, that spec references this file for global conventions. Read this once; each module spec then adds only what's specific to it.
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
├── configs/            base_model.yaml, training/*.yaml, deepspeed/*.json, inference/vllm_serving.yaml   → IMPL-01
├── schemas/            {acord25,acord125,lossrun,policy_doc}.schema.json                                  → IMPL-01
├── prompts/            system_prompt_template.jinja, doc_type_classifier_prompt.jinja                     → IMPL-01
├── common/             config, schemas, prompts, ids, constants, normalize                                → IMPL-01 (+ normalize in 08)
├── artifact_registry/  push_to_blob.py, pull_from_blob.py                                                 → IMPL-02
├── registry_utils/     write_run_manifest.py, query_registry.py                                           → IMPL-02
├── data_pipeline/
│   ├── ingestion/      pull_raw_pdfs.py                                                                   → IMPL-03
│   ├── ocr/            run_mineru.py, render_only.py                                                       → IMPL-03
│   ├── labeling/       pre_annotate.py, export_golden_labels.py, review_tool/, active_learning.py         → IMPL-04
│   ├── dataset_builder/ build_jsonl.py, modality_dropout.py, noisy_ocr_augment.py, split_train_val_test.py → IMPL-05
│   └── corpus_manifest.py                                                                                 → IMPL-05
├── training/           train_foundation.py, train_adapter.py, data_collator.py, vit_gate.py, callbacks/   → IMPL-06
├── inference_core/     model_runner.py, input_builder.py, span_map.py, runner_config.py                   → IMPL-07  (shared primitive)
├── evaluation/         run_eval.py, metrics/*, gating.py                                                  → IMPL-08
├── calibration/        logprob_confidence.py, fit_calibration.py, apply_calibration.py, list_completeness.py → IMPL-09
├── postprocessing/     merge_adapter.py, quantize.py, validate_quant.py                                   → IMPL-10
├── serving/            vllm_entrypoint.py, doc_type_classifier.py, adapter_router.py, page_router.py,
│                       confidence_postprocess.py, pipeline.py                                             → IMPL-11
├── testing/            run_extraction.py, prompts/, (test_data, ocr_cache, results, metrics, registry)   → IMPL-12
├── orchestration/      runpod_controller.py, pipeline_dag.py                                              → IMPL-13
└── tests/              test_*.py, fixtures/                                                               → IMPL-14
```

**Key separation principle:** the repo never stores model weights, corpus data, or PDFs. Everything data/artifact-related is pulled from / pushed to Azure Blob at runtime via `artifact_registry/`.

---

## 6. Spec build order & dependency graph

Build in this order — later specs import from earlier ones. **IMPL-07 (inference core) is deliberately placed before evaluation/calibration/serving/testing because all four share it** — this breaks what would otherwise be a circular dependency between them.

```
IMPL-01  Project scaffold, configs, schemas, prompts, common   (no deps)
IMPL-02  Azure Blob I/O + training run registry                (deps: 01)
IMPL-03  Ingestion + MinerU OCR                                (deps: 01, 02)
IMPL-04  Labeling + golden JSON generation                     (deps: 01, 02, 03)
IMPL-05  Dataset builder (JSONL, modality, split)              (deps: 01, 02, 04)
IMPL-06  Training (ms-swift QLoRA, foundation + adapters)      (deps: 01, 02, 05)
IMPL-07  Inference core (shared model-runner primitive)        (deps: 01, 02)
IMPL-08  Evaluation (metrics + gating, + common.normalize)     (deps: 01, 02, 06, 07)
IMPL-09  Confidence calibration                                (deps: 01, 02, 07, 08)
IMPL-10  Postprocessing (merge + GGUF quantize)                (deps: 01, 02, 06)
IMPL-11  Serving (vLLM endpoint, classifier, routing)          (deps: 01, 02, 07, 09, 10)
IMPL-12  Testing / extraction routine (CLI harness)            (deps: 01, 02, 03, 07, 08, 09, 11)
IMPL-13  Orchestration (RunPod controller, pipeline DAG)       (deps: all above)
IMPL-14  Unit tests + CI fixtures                              (deps: all above)
```

**Note on two cross-references that point "forward" but are not build-order violations:**
- `postprocessing/validate_quant.py` (IMPL-10) calls the testing routine (IMPL-12). This is fine — `validate_quant` is a *validation gate you run after IMPL-12 exists*, not something the core merge/quantize functions need. Build merge+quantize in IMPL-10; wire `validate_quant` once IMPL-12 is done.
- `evaluation/run_eval.py` (IMPL-08) can compute classifier-accuracy only once the classifier (IMPL-11) exists. IMPL-08 is built to run *without* it (supply the true doc_type), and classifier metrics fill in on a later pass. Not a blocker.

Everything else is strictly bottom-up.

---

## 7. Global conventions every spec must follow

- **Language/runtime:** Python 3.11+. Use type hints throughout. Prefer `pydantic` v2 for config/data models.
- **Config:** all runtime config via YAML in `configs/` + environment variables (via `.env`, loaded with `python-dotenv`). Never hardcode paths, model ids, or credentials.
- **Secrets:** Azure + RunPod credentials come from env vars only. `.env.example` lists every required var with placeholder values. Never commit real secrets.
- **Logging:** use structured logging (`structlog` or stdlib `logging` with JSON formatter). **Never log PII** (raw OCR text, extracted field values, image bytes) at INFO or above — see PII rules below.
- **Blob access:** all Azure Blob reads/writes go through `artifact_registry/` (IMPL-02). No module opens the Azure SDK directly except that module.
- **Determinism:** set and log random seeds for any stochastic step (splitting, training, augmentation).
- **CLIs:** every runnable module exposes a `argparse`/`typer` CLI with `--help`. Long-running scripts print progress.
- **Error handling:** fail loudly with clear messages on missing config, missing Blob artifacts, or schema-invalid data. Don't silently continue.
- **No network assumptions:** code may run inside a RunPod pod with a restricted egress allowlist; make external endpoints configurable, not hardcoded.
- **Idempotency:** ingestion, OCR, and dataset-build steps should be safe to re-run — skip or overwrite deterministically based on `source_id` + content checksum, never duplicate.

## 8. PII handling (non-negotiable, applies to every spec)

Insurance documents contain PII (names, TINs/SSNs, addresses, financials).
- `raw-documents/` is the most sensitive layer — tighter RBAC, encryption at rest, immutable.
- Never send PII to third-party APIs unless explicitly permitted (affects labeling pre-annotation in IMPL-04 — default to the self-hosted base model).
- Never persist raw request/response bodies or OCR text in plaintext logs (affects serving IMPL-10, testing IMPL-11).
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

**Run manifest (registry, per training run):** see IMPL-02 for the full schema.

## 10. Modality-mix for training data (used by IMPL-05)

Within the Foundation corpus, per source document generate 3 rows:
- `ocr_plus_image` — 50%
- `noisy_ocr_image` — 20% (deliberately corrupted OCR, corrected golden JSON)
- `image_only` — 30% (no OCR text block; system prompt says so explicitly)

Split at **source_id level BEFORE** modality expansion to prevent leakage.

---

## 11. How to use these specs with Claude Code

1. Start a Claude Code session at the repo root.
2. Run `IMPL-01` first to scaffold the project.
3. Run each subsequent spec **in numeric order (01 → 14)**. Each spec is self-contained but assumes earlier specs are done. The order is dependency-correct — no spec references code from a higher-numbered spec as a build prerequisite (the two "forward" mentions in IMPL-08 and IMPL-10 are explicitly non-blocking; see §6).
4. Each spec ends with an **Acceptance checklist** — verify it before moving on.
5. This master file is the context; when a spec says "per master context", it means this file.


===================================================================================

# SPEC 01 — Project Scaffold, Configs, Schemas, Prompts

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: none. This spec creates the repo skeleton and all static configuration that later modules read.

## Goal

Scaffold the `insurance-extraction-finetuning` repo and create every config file, JSON schema, and prompt template that the rest of the system depends on.

## Deliverables

### 1. Repo root
- `README.md` — project overview, setup steps, how to run each pipeline stage (stub with section headings + the module map from master context).
- `pyproject.toml` — Python 3.11+, dependencies grouped by extra: `[data]` (azure-storage-blob, pypdf, pillow, python-dotenv, pydantic), `[train]` (ms-swift, transformers, peft, bitsandbytes, accelerate, deepspeed, trl, flash-attn), `[serve]` (vllm), `[eval]` (numpy, scikit-learn, jsonschema), `[dev]` (pytest, ruff, mypy). Pin major versions; leave a comment that exact versions are verified at implementation time.
- `requirements.txt` — generated equivalent for pip installs on RunPod.
- `.env.example` — every required env var with placeholder: `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_BLOB_CONTAINER`, `RUNPOD_API_KEY`, `HF_TOKEN`, `HF_MODEL_REVISION`, `MLFLOW_TRACKING_URI` (optional), plus any external-endpoint URLs. Comment each.
- `.gitignore` — Python, `.env`, model weights, `*.pdf`, `results/`, `ocr_cache/`, `__pycache__`.

### 2. `configs/`
- `base_model.yaml` — `model_id: Qwen/Qwen3-VL-8B-Instruct`, `revision: <pin>`, quantization block (4-bit NF4, bf16 compute), `attn_implementation: flash_attention_2`, resolution cap (`max_image_long_side_px: 1792`), `max_seq_len` (comment: set from 95th-percentile of real corpus).
- `configs/training/foundation.yaml` — LoRA rank 64 / alpha 128 / dropout 0.05, target modules list (from master §2), LR `2e-4`, cosine schedule warmup 0.03, epochs 3, effective batch 32 via grad accumulation, gradient checkpointing on, ViT frozen flag `train_vit: false`, early-stopping config. Add a top comment: "hyperparameters are sweep starting points (§10), tune on val."
- `configs/training/acord25_adapter.yaml`, `acord125_adapter.yaml`, `lossrun_adapter.yaml`, `policy_adapter.yaml` — LoRA rank 16 / alpha 32, LR `7e-5`, epochs 4, each references its `doc_type`, its schema file, and the Foundation version it builds on (`foundation_version: <set at runtime>`).
- `configs/deepspeed/zero2.json` and `zero3.json` — standard ZeRO-2 / ZeRO-3 configs for multi-GPU.
- `configs/inference/vllm_serving.yaml` — resolution cap (must match `base_model.yaml`), max_seq_len, adapter routing config (map doc_type → adapter path), enable logprobs, served model tag.

### 3. `schemas/` — JSON Schemas (Draft 2020-12)
Create strict JSON Schema files defining the extraction target for each type. Use realistic insurance fields; mark required vs optional; use proper types (string/number/array/null-unions); date fields as `string` with format note (YYYY-MM-DD).
- `acord25.schema.json` — Certificate of Liability: insured, insurer(s), policy numbers, effective/expiration dates, coverage limits (GL, auto, umbrella), certificate holder, etc.
- `acord125.schema.json` — Commercial Application: applicant, business info, premises, coverages requested.
- `lossrun.schema.json` — carrier, policy number, valuation date, and a **`claims` array** (repeating rows: claim number, loss date, status, paid, reserved, total incurred, description). This is the key list-field type.
- `policy_doc.schema.json` — named insured, policy number, effective/expiration, coverage schedule (array), premium, endorsements list.

Each schema must be loadable by `jsonschema` and validate a correct example. Include one valid example JSON per schema under `schemas/examples/{type}.example.json`.

### 4. `prompts/`
- `system_prompt_template.jinja` — parametrized by `{{ doc_type }}`, `{{ schema_json }}`, and `{{ modality_mode }}`. Must:
  - State the doc type and inject the exact schema.
  - Give extraction rules: valid JSON only, no markdown fences, missing fields → `null`, date format, how to handle repeating table rows.
  - Include the **modality instruction line** that differs by mode:
    - `ocr_plus_image`: "OCR text is provided; use it as primary source for dense text/numbers, use the image to verify layout and correct OCR errors."
    - `noisy_ocr_image`: same as above (the noise is in the data, not the instruction).
    - `image_only`: "OCR text is not provided — extract directly from the image."
- `doc_type_classifier_prompt.jinja` — zero-shot prompt that asks the model to name the doc type (`acord`/`policy`/`lossrun`) and, if acord, the form number. Output a small fixed JSON (`{"doc_type": "...", "acord_form": "..."}`).

### 5. `common/` (shared library — create this package)
- `common/config.py` — pydantic settings loader (reads env + a given YAML), one entrypoint `load_config(path)`.
- `common/schemas.py` — helpers to load a schema by doc_type, validate a JSON object against it, list required fields.
- `common/prompts.py` — render the jinja templates given doc_type + modality_mode + schema.
- `common/ids.py` — `source_id` helpers: build/parse `{doc_type}_{index}`, validate format.
- `common/constants.py` — `ACTIVE_DOC_TYPES = ["acord", "policy", "lossrun"]`, modality modes, modality mix ratios, resolution cap default.

## Constraints
- No business logic yet — this is scaffold + static assets + tiny shared helpers.
- Everything must import cleanly (`python -c "import common..."`).
- Follow all global conventions in master §7.

## Acceptance checklist
- [ ] Repo tree matches master §5 (empty dirs get a `.gitkeep`).
- [ ] `pip install -e .[dev]` succeeds (or documents which heavy deps are RunPod-only).
- [ ] All 4 schemas load and validate their example JSON via `jsonschema`.
- [ ] `common.prompts.render(...)` produces a correct system prompt for each of the 3 modality modes.
- [ ] `.env.example` lists every env var used anywhere in the design.
- [ ] `ruff` and `mypy` pass on `common/`.


===================================================================================

# SPEC 02 — Azure Blob I/O + Training Run Registry

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01. This is the ONLY module that talks to the Azure SDK directly. Everything else goes through it.

## Goal

Provide a clean, typed interface for all Azure Blob reads/writes, and implement the training run registry that records every Foundation and per-type training run.

## Deliverables

### 1. `artifact_registry/blob_client.py`
- Thin wrapper around `azure-storage-blob`, initialized from env (`AZURE_STORAGE_CONNECTION_STRING`, `AZURE_BLOB_CONTAINER`).
- Methods: `upload_file(local, blob_path)`, `download_file(blob_path, local)`, `upload_dir(local_dir, blob_prefix)`, `download_dir(blob_prefix, local_dir)`, `exists(blob_path)`, `list(prefix)`, `read_json(blob_path)`, `write_json(obj, blob_path)`.
- Retries with backoff; clear errors on auth/missing-container.
- **Tighter access note:** expose an optional `container` override so `raw-documents/` can point at a separately-permissioned container (master §8).

### 2. `artifact_registry/push_to_blob.py` / `pull_from_blob.py`
- High-level, path-aware helpers keyed to the Blob layout (master §4). Examples:
  - `push_adapter(local_dir, kind, doc_type, version)` → `adapters/{foundation|doc_type}/v{n}/`
  - `pull_adapter(kind, doc_type, version, local_dir)`
  - `push_merged_model(...)`, `push_quantized(local_dir, doc_type, version, fmt)`
  - `pull_base_model(local_dir)` (from `base-models/`, or from HF if absent, then cache to Blob)
  - `push_corpus_version(...)`, `pull_corpus_version(...)`
  - `push_eval_report(...)`, `push_golden_eval_set` / `pull_golden_eval_set`
- CLI so a RunPod pod can `python -m artifact_registry.pull_from_blob --corpus v3 --dest ./data`.

### 3. `registry_utils/models.py`
- Pydantic model `RunManifest` capturing exactly (master §2, arch §11):
  - `run_id`, `run_type` (`foundation`|`per_type_adapter`), `doc_type` (nullable), `status` (`trained|evaluated|promoted|archived|failed`), `created_at`.
  - `dependencies`: `base_model` (id@revision), `foundation_version` (nullable), `corpus_version`, `code_git_commit`.
  - `training_config`: technique, lora_rank/alpha/dropout, lr, epochs, effective_batch_size, target_modules, `vit_frozen`, resolution_cap_px, max_seq_len.
  - `data_stats`: train/val/test counts, modality_mix.
  - `eval_metrics`: field_exact_match, field_f1_list_fields, list_field_recall, schema_validity_rate, ece_confidence, ocr_arbitration_accuracy, image_only_accuracy, scanned_accuracy, doc_type_classifier_accuracy.
  - `artifacts`: adapter_weights, merged_model, quantized_model, quantized_formats[], eval_report.
  - `promotion`: gated_against, beat_previous_on_all_gates, promoted_by, promoted_at.

### 4. `registry_utils/write_run_manifest.py`
- `write_manifest(manifest: RunManifest)` → writes `registry/{run_type}/{doc_type or 'foundation'}/{run_id}/run_manifest.json` to Blob, and updates `registry/registry_index.json` (append/replace the flat row: run_id, type, status, key metrics, created_at).
- Helper `capture_git_commit()` to fill `code_git_commit`.
- Optional MLflow/W&B logging hook (guarded by env; no-op if unset).

### 5. `registry_utils/query_registry.py`
- CLI + functions:
  - `get(run_id)` → RunManifest.
  - `list_runs(run_type=None, doc_type=None, status=None)`.
  - `adapters_depending_on(foundation_version)` → list of adapter run_ids (implements the §11 dependency-upgrade query).
  - `latest_promoted(kind, doc_type)` → the version currently serving.
  - `resolve_model_version(tag)` → given a user tag like `v2`, return the concrete artifact paths (foundation + per-type adapter, or merged/quantized) — used by serving (IMPL-10) and testing (IMPL-11).

## Constraints
- Only this package imports `azure-storage-blob`.
- All Blob paths built from a single `paths.py` helper so the layout lives in one place.
- No PII in logs.

## Acceptance checklist
- [ ] `blob_client` round-trips a file and a directory against a real/emulated container.
- [ ] `RunManifest` validates the example from arch §11 and rejects malformed input.
- [ ] `write_run_manifest` creates the manifest and updates `registry_index.json`.
- [ ] `adapters_depending_on("foundation-v2.0")` returns the right list from seeded manifests.
- [ ] `resolve_model_version("v2")` returns concrete adapter/merged paths.
- [ ] Unit tests mock the Blob client (no live Azure needed for CI).


===================================================================================

# SPEC 03 — Ingestion + MinerU OCR

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02.

## Goal

Ingest raw insurance PDFs into the immutable `raw-documents/` layer, then run MinerU OCR + page-image rendering into `processed/`. This is the front of both the training-data pipeline and (reused) the inference pipeline.

## Deliverables

### 1. `data_pipeline/ingestion/pull_raw_pdfs.py`
- Takes a local folder or a source location, and for each PDF:
  - Assigns/validates a `source_id` (`{doc_type}_{index}`; doc_type either provided or inferred later — allow an `--unclassified` bucket).
  - Computes SHA-256 checksum; **dedup**: if checksum already ingested, skip and log.
  - Writes `raw-documents/{doc_type}/{source_id}/original.pdf` (immutable — never overwrite; a corrected doc becomes a new source_id).
  - Writes `metadata.json`: ingestion timestamp, source, checksum, page count, digital-vs-scanned flag (detect via presence of an embedded text layer), PII flags placeholder.
- Idempotent and re-runnable. CLI with `--input`, `--doc-type`, `--dest-container`.
- Uses the tighter-RBAC container for `raw-documents/` (master §8).

### 2. `data_pipeline/ocr/run_mineru.py`
- Batch-process documents from `raw-documents/` (or a passed list of source_ids):
  - Run **MinerU** to produce per-page markdown/text.
  - Render each page to a PNG at the **resolution cap** from `configs/base_model.yaml` (identical to training + inference — this consistency is mandatory).
  - Write `processed/{doc_type}/{source_id}/page_{n}.md` and `page_{n}.png`.
- Detect + record MinerU failures/low-confidence pages in a per-doc `ocr_meta.json` (needed later so the dataset builder and eval can account for imperfect OCR).
- Idempotent (skip if processed output exists and source checksum unchanged).
- Make MinerU invocation configurable (it may run as a subprocess or library call); keep the interface swappable.
- CLI: `--source-ids ... | --all-unprocessed`, `--doc-type`.

### 3. `data_pipeline/ocr/render_only.py` (helper)
- Path for `image_only` mode / inference: render page images without requiring OCR text, at the same resolution cap.

## Constraints
- `raw-documents/` is write-once; assert before writing.
- Resolution cap MUST come from config, never hardcoded, and be logged.
- No PII in logs (don't print OCR text).
- All Blob I/O via IMPL-02.

## Acceptance checklist
- [ ] Ingesting the same PDF twice results in one stored copy (dedup by checksum).
- [ ] Digital vs scanned flag is set correctly on sample inputs.
- [ ] MinerU output produces `page_*.md` + `page_*.png` at the configured resolution.
- [ ] Re-running OCR on already-processed docs is a no-op.
- [ ] `ocr_meta.json` records page count and any OCR failures.
- [ ] `render_only` produces images with no OCR dependency.


===================================================================================

# SPEC 04 — Labeling + Golden JSON Generation

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-03.

## Goal

Turn OCR'd documents into human-verified **golden JSON** labels via a bootstrap pre-annotation + human-review workflow, with provenance tracking. (Arch §6 "Golden JSON Generation Mechanism".)

## Deliverables

### 1. `data_pipeline/labeling/pre_annotate.py`
- Generates a first-pass ("silver") JSON draft for a document using a configurable annotator backend:
  - `base_qwen3vl` (default, self-hosted) — prompt the base/current model with schema + image (+OCR).
  - `own_finetuned` — once a v1 exists, use the current promoted model (better drafts each cycle).
  - `external_frontier` — **disabled by default**; guarded by an explicit `--allow-external` flag AND an env permission flag, because sending PII to third-party APIs may violate compliance (master §8). If used, require a zero-retention endpoint config.
- Output draft JSON is **never trusted as-is** — it's only a starting point for review.

### 2. `data_pipeline/labeling/review_tool/`
- Integration config for an external labeling tool (Label Studio or Argilla) OR a lightweight custom review UI:
  - Presents the page image(s) + OCR + draft JSON side-by-side.
  - Reviewer corrects every field; output is the golden JSON.
  - Support **double-annotation** on a configurable sample (default 10–20%): second independent label + adjudication, producing an inter-annotator agreement score.
- Provide a task-export/import adapter so labels round-trip cleanly.

### 3. `data_pipeline/labeling/export_golden_labels.py`
- Writes verified labels to `golden-labels/{doc_type}/{source_id}/golden.json` and `label_metadata.json` containing: reviewer id, review date, draft source (which annotator backend), double-annotated flag, agreement score.
- **Validates every golden.json against its schema** (IMPL-01) before writing — reject invalid labels.
- Idempotent; re-export overwrites only with a new label_metadata version.

### 4. `data_pipeline/labeling/active_learning.py`
- After a model version exists: run it over new unlabeled docs, use **calibrated confidence** (IMPL-08) to route — high-confidence → light spot-check, low-confidence fields → full manual review. Emits a prioritized review queue. (Arch §6 step 6; ties to §12 step 11 feedback loop.)

## Constraints
- External pre-annotation OFF unless explicitly permitted.
- Golden labels must be schema-valid.
- Store provenance for every label.
- No PII in logs.

## Acceptance checklist
- [ ] `pre_annotate` produces a draft with the self-hosted backend; external backend refuses without the permission flag.
- [ ] Review workflow exports a corrected golden JSON that passes schema validation.
- [ ] Double-annotation path produces an agreement score on the sampled subset.
- [ ] `label_metadata.json` captures full provenance.
- [ ] `active_learning` orders a queue by ascending confidence.


===================================================================================

# SPEC 05 — Dataset Builder (JSONL, Modality, Split)

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-04.

## Goal

Compile processed docs + golden labels into versioned, chat-format JSONL training corpora — applying the 3-regime modality split and leakage-safe train/val/test splitting. (Arch §6, §7, §10.)

## Deliverables

### 1. `data_pipeline/dataset_builder/split_train_val_test.py`
- **Split at `source_id` level BEFORE any modality expansion** (master §10 — this prevents the same document leaking across splits via different modality variants).
- Ratio scales with per-type volume (arch §7): small pilot ≈ 70/18/12; at scale 80/10/10. Configurable, seeded, logged.
- Output: a deterministic assignment `source_id → split`, saved for reproducibility.

### 2. `data_pipeline/dataset_builder/modality_dropout.py`
- For each `source_id`, generate the 3 modality variants with the mix ratios (50/20/30 — master §10). Each variant becomes one JSONL row in the split its source_id belongs to.
- `image_only` rows omit the OCR text block and use the image-only system prompt.

### 3. `data_pipeline/dataset_builder/noisy_ocr_augment.py`
- For `noisy_ocr_image` rows: take the real MinerU OCR and inject realistic corruptions (digit↔letter confusions like O/0, l/1; merged/split table cells; dropped headers) while keeping the **golden JSON as the correct target** — teaching image-over-OCR arbitration (arch §5, §6).
- Corruptions parameterized and seeded; keep them plausible (mirror MinerU's real failure modes), not random noise.

### 4. `data_pipeline/dataset_builder/build_jsonl.py`
- Orchestrates: for each split and doc_type, assemble chat-format rows using the exact data contract (master §9):
  - `system` = rendered prompt (schema + modality instruction) via `common.prompts`.
  - `user` = image block (+ OCR text unless image_only).
  - `assistant` = golden JSON string.
- Carry `doc_type`, `acord_form`, `modality_mode`, `source_id`, `split` on every row.
- Write `corpus/v{n}/{doc_type}/{train,val,test}.jsonl` to Blob.
- Reference images/OCR by their Blob URIs or a training-time-resolvable path (document the choice).

### 5. `data_pipeline/corpus_manifest.py`
- Writes `manifest.json` per corpus version: example counts per doc_type × split × modality_mode, source_id lists per split, seed, ratios, builder git commit. This is what training + eval read to know exactly what a corpus version contains.

## Constraints
- Zero split leakage — assert no `source_id` appears in more than one split.
- Every row schema-checkable (the assistant JSON validates against the doc_type schema).
- Deterministic + seeded; re-running yields identical corpora.

## Acceptance checklist
- [ ] No `source_id` crosses splits (automated assertion).
- [ ] Each source produces exactly 3 rows with the correct modality distribution across the dataset.
- [ ] `noisy_ocr_image` rows have corrupted OCR but correct golden JSON targets.
- [ ] `image_only` rows contain no OCR text and use the image-only prompt.
- [ ] `manifest.json` counts match the actual JSONL row counts.
- [ ] Rebuild with same seed produces byte-identical output.


===================================================================================

# SPEC 06 — Training (ms-swift QLoRA: Foundation + Per-Type Adapters)

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-05.

## Goal

Fine-tune Qwen3-VL-8B-Instruct with QLoRA via the locked trainer stack (ms-swift → TRL SFTTrainer → HF Trainer). Train the shared Foundation LoRA and the per-type adapters on top, freezing the ViT by default with a gated unfreeze path. (Arch §2, §3, §8, §9, §10, §11.)

## Deliverables

### 1. `training/train_foundation.py`
- Entrypoint that:
  - Pulls base model (IMPL-02) and corpus version (IMPL-02) locally.
  - Loads `configs/training/foundation.yaml` + `configs/base_model.yaml`.
  - Configures **QLoRA**: 4-bit NF4 base, bf16 LoRA, target modules = attention+MLP projections **+ vision-language projector**, **ViT frozen** (`train_vit: false`).
  - Launches **ms-swift** SFT (which runs TRL `SFTTrainer`/HF `Trainer` underneath) with DeepSpeed (ZeRO-2 default).
  - Trains across ALL doc types + ALL 3 modality regimes (the mixed corpus).
  - Saves adapter to local out dir; pushes to `adapters/foundation/v{n}/` (IMPL-02).
  - Writes a `RunManifest` (IMPL-02) with full config, data stats, git commit, corpus version.
- Flags: `--corpus vN`, `--out-version vN`, `--deepspeed zero2|zero3`, `--train-vit` (default false — the gated exception).

### 2. `training/train_adapter.py`
- Per-doc-type adapter entrypoint:
  - Loads the **current promoted Foundation** as the base (frozen) + trains a small LoRA (rank 16) on that doc type's slice of the corpus.
  - **Always trains fresh from the Foundation** — never continues from a previous adapter checkpoint (arch §11).
  - Records `foundation_version` dependency in the manifest.
  - Pushes to `adapters/{doc_type}/v{n}/`.
- Flags: `--doc-type acord|policy|lossrun`, `--foundation vN`, `--corpus vN`, `--out-version vN`.

### 3. `training/data_collator.py`
- **Override hook only.** ms-swift provides correct multimodal collation + `-100` label masking (system/image/OCR tokens masked, loss only on assistant JSON) by default. Implement this file ONLY if a custom masking need arises; otherwise it documents that the framework default is used and includes an assertion/test that verifies masking is correct on a sample batch.

### 4. `training/callbacks/early_stopping.py`
- Early stopping on val loss + field-level F1, patience 2 evals. Wire into the ms-swift/Trainer callback system.

### 5. `training/vit_gate.py` (decision helper)
- Given an eval report (IMPL-07) for a frozen-ViT Foundation, decide whether the ViT-unfreeze gate fires: image_only OR scanned accuracy below target AND failures are perception-type (not schema/reasoning). Emits a recommendation + rationale. (Arch §2 escalation gate.) Does not auto-train — surfaces the decision.

## Constraints
- ms-swift is the entrypoint (locked). Raw HF path only as documented fallback.
- ViT frozen unless `--train-vit` explicitly set (and then use LoRA-on-ViT, not full FT).
- Every run writes a manifest — no silent training.
- Resolution cap + max_seq_len come from config, consistent with data prep.

## Acceptance checklist
- [ ] Foundation training runs end-to-end on a tiny sample corpus and produces a loadable LoRA adapter.
- [ ] Label masking verified: loss computed only on assistant tokens (test on a sample batch).
- [ ] Per-type adapter trains on top of a given Foundation and records the dependency.
- [ ] Adapter + RunManifest land in Blob at the right paths.
- [ ] `--train-vit` toggles ViT LoRA; default keeps it frozen.
- [ ] `vit_gate` returns a correct fire/no-fire recommendation on seeded eval inputs.


===================================================================================

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


===================================================================================

# SPEC 08 — Evaluation (Metrics + Promotion Gating)

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-06, IMPL-07. Uses the shared inference core (IMPL-07) for all model runs — no forward dependency on serving or testing.

## Goal

Evaluate any model/adapter version against the frozen golden eval set and gate promotion so no regression ships. (Arch §14.)

## Deliverables

### 1. `common/normalize.py` (shared normalizer — create here, reused by testing)
- Field-type normalizers so `01/01/2026` == `2026-01-01`, `$1,200.00` == `1200.0`, `Acme Mfg LLC` ≈ `ACME MANUFACTURING LLC`. Evaluation and testing (IMPL-12) MUST use this identical logic.

### 2. `evaluation/metrics/` — one module per metric, each pure and testable
- `field_exact_match.py` — exact + normalized match (via `common.normalize`). Per-field and aggregate.
- `field_f1.py` — precision/recall/F1 for list fields (claims, schedule rows).
- `list_recall.py` — **row completeness**: extracted row count vs ground-truth count; missed-row rate (arch §4 list-field gap).
- `schema_validity.py` — parse + validate output against the doc_type schema.
- `calibration_error.py` — Expected Calibration Error (ECE) on confidence vs correctness.
- `ocr_arbitration_accuracy.py` — on the noisy-OCR subset: did the model override bad OCR using the image?
- `mode_accuracy.py` — separate accuracy for `image_only` and `scanned` subsets (feed the ViT gate, IMPL-06).
- `classifier_accuracy.py` — doc-type + ACORD-form classification accuracy (arch §3a).

### 3. `evaluation/run_eval.py`
- Given `--model vN` (resolved via IMPL-02) + the golden eval set, run inference **via IMPL-07 inference core** across all eval docs and relevant modes, compute every metric, write `eval-reports/v{n}/{doc_type}/report.json` + a top-level summary. Per doc_type and per modality mode.
- Note: classifier accuracy uses the classifier from serving (IMPL-11) once it exists; until then, `run_eval` can score extraction with the true doc_type supplied, and classifier metrics get filled in on a later pass. (Keeps eval runnable before serving is built.)

### 4. `evaluation/gating.py`
- `promotion_gate(candidate_report, current_report, doc_type)` → candidate must match/beat current on every gating metric. Per-metric deltas + overall decision (arch §14). Writes the decision into the candidate's RunManifest (IMPL-02).

## Constraints
- Golden eval set is frozen + versioned separately; never train on it.
- Metrics deterministic + unit-testable on synthetic inputs.
- All model runs go through IMPL-07 (no bespoke inference here).

## Acceptance checklist
- [ ] Each metric returns correct values on hand-constructed synthetic cases.
- [ ] Normalized match handles dates/currency/names via `common.normalize`.
- [ ] `list_recall` flags a deliberately dropped claim row.
- [ ] `run_eval` produces a per-doc-type report with mode breakdowns, using IMPL-07.
- [ ] `gating` blocks a candidate that regresses any single gate metric.
- [ ] Promotion decision written to the RunManifest.


===================================================================================

# SPEC 09 — Confidence Calibration

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-07, IMPL-08. Uses inference core (IMPL-07) for logprobs/spans and the evaluation normalizer (IMPL-08) for the correctness signal. No forward dependency on serving/testing.

## Goal

Turn raw token logprobs into trustworthy per-field confidence via post-hoc calibration, plus the list-field completeness signal. (Arch §4.)

## Deliverables

### 1. `calibration/logprob_confidence.py`
- Consumes field spans + logprobs from IMPL-07 `span_map`, aggregates to raw per-field confidence. Default = **min token prob** in span; pluggable (min/mean/geomean).
- Handles nested + list fields (per-value confidence within rows).

### 2. `calibration/fit_calibration.py`
- On a **held-out validation set** (never training data), compare raw confidence vs correctness (via `common.normalize` from IMPL-08) and fit a transform:
  - `temperature` scaling (default) and `isotonic` regression (flexible), selectable.
  - Fit **per doc_type** (optionally per field-type).
- Persist to `calibration/calibration_store/{version}/{doc_type}.json` + push to Blob.

### 3. `calibration/apply_calibration.py`
- Load the fitted transform for a model version + doc_type; map raw → calibrated confidence at inference time. Used by serving (IMPL-11) and testing (IMPL-12).

### 4. `calibration/list_completeness.py`
- Separate "did we get all rows?" confidence: cross-check extracted row count vs a document-derived count (stated "total: N" or MinerU-detected table rows). Disagreement → flag the whole list regardless of per-value confidence. Optionally calibrate against ground-truth row counts on validation. (Arch §4.)

## Constraints
- Calibration fit ONLY on held-out data.
- Params versioned per model version + doc_type.
- Correctness signal reuses IMPL-08 `common.normalize`.

## Acceptance checklist
- [ ] Raw per-field confidence extracted from a sample generation + logprobs (via IMPL-07 spans).
- [ ] Temperature + isotonic both reduce ECE on a synthetic miscalibrated set.
- [ ] Fitted params persist/reload and apply deterministically.
- [ ] `list_completeness` flags a list when extracted rows < document-stated count.
- [ ] Calibrated confidence available with no ground truth (inference-time).


===================================================================================

# SPEC 10 — Postprocessing (Merge Adapter + GGUF Quantize)

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-06. (Independent of the inference/eval/calibration chain — only needs a trained adapter.)

## Goal

Merge a trained LoRA adapter into the base model, then export user-selectable GGUF quantization formats for serving. (Arch §12 steps 7–9, §12a.)

## Deliverables

### 1. `postprocessing/merge_adapter.py`
- Loads base model + a Foundation adapter (and optionally a stacked per-type adapter); runs PEFT `merge_and_unload()` → full fp16/bf16 merged model.
- Pushes to `merged-models/{doc_type|unified}/v{n}/` (IMPL-02).
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
- Runs each produced GGUF through the extraction/testing routine (IMPL-12) against the golden eval set — **every served format must be re-validated** (lower-bit quant can degrade JSON structure / field precision / calibration). Reports accuracy delta vs fp16. (Arch §12a.)
- Note: this calls into IMPL-12; run it after IMPL-12 exists, or as the validation gate before promoting a quantized format to serving.

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


===================================================================================

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


===================================================================================

# SPEC 12 — Testing / Extraction Routine (CLI Harness)

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-03 (OCR), IMPL-07 (inference core), IMPL-08 (metrics), IMPL-09 (calibration), IMPL-11 (serving pipeline).

## Goal

A CLI harness to test any fine-tuned model version on real PDFs post-training, producing structured JSON + per-field/overall confidence + quality metrics, with version-organized outputs and provenance. **Reuses the serving pipeline (IMPL-11) so test == prod.** (Arch §15.)

## Deliverables

### 1. Folder layout under `testing/` (create dirs with `.gitkeep`)
```
testing/
├── run_extraction.py
├── test_data/           # INPUT: drop test PDFs (scanned or digital)
├── prompts/             # acord.prompt.txt, policy.prompt.txt, lossrun.prompt.txt
├── ocr_cache/           # cached MinerU output per doc
├── results/{version}/   # OUTPUT: e.g. results/v2/abcLossRun.json
├── metrics/{version}/    # OUTPUT: per-doc + _run_summary.json
└── extraction_registry.json
```

### 2. `testing/prompts/*.prompt.txt`
- One per active doc type (acord, policy, lossrun): role framing + injected schema (from registry, kept in sync) + extraction rules + modality instruction line. **Same prompts serving uses** (they should resolve to the same rendered prompt as `common.prompts`) — no drift. (Arch §15.)

### 3. `testing/run_extraction.py` — the CLI
- Flags:
  - `--model vN` — **user selects the version** (resolved via IMPL-02).
  - `--input <file|dir>` — test PDFs.
  - `--mode ocr_plus_image|image_only`.
  - `--ground-truth <dir>` — optional; if omitted, still emit JSON + confidence, no accuracy metrics.
- Per PDF: OCR (IMPL-03) → **call `serving/pipeline.py` (IMPL-11)** for classify → route → inference (IMPL-07) → calibrated confidence (IMPL-09) → validate → write `results/{version}/{doc_stem}.json` → metrics (IMPL-08, if ground truth) → append `extraction_registry.json`.
- The routine is a thin CLI wrapper over the serving pipeline + OCR + metrics; it must NOT re-implement inference (that would break test==prod).

### 4. Metrics output
- Per-doc `*.metrics.json`: per-field {value, confidence, correct?}, overall_confidence, schema_valid, field_exact_match_rate, list_field_f1, list_recall (IMPL-08 metrics + `common.normalize`).
- `metrics/{version}/_run_summary.json`: batch aggregates — mean field-exact-match, list F1, list recall, schema validity, ECE, per-doc-type breakdown, mean latency/doc. The cross-version comparison file.

### 5. `testing/extraction_registry.json`
- One entry per extraction: document, doc_type, model_version, mode, result_path, metrics_path, overall_confidence, schema_valid, extracted_at. Ties `model_version` back to the training RunManifest (IMPL-02) so a result traces to the corpus/commit that produced its model. (Arch §15.)

## Constraints
- Confidence emitted even without ground truth.
- Same prompts + same pipeline as serving (reuse, don't reimplement).
- No PII in logs; results dir gitignored.

## Acceptance checklist
- [ ] `run_extraction --model v2 --input test_data/` writes `results/v2/*.json`.
- [ ] `--mode image_only` runs with no OCR and still extracts.
- [ ] Per-field + overall confidence present with and without ground truth.
- [ ] With `--ground-truth`, per-doc + summary metrics computed.
- [ ] `extraction_registry.json` links result → model version.
- [ ] Output equals the serving pipeline's output on the same input (parity test).


===================================================================================

# SPEC 13 — Orchestration (RunPod Controller + Pipeline DAG)

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: all prior specs (01–12).

## Goal

Automate the full lifecycle: spin up ephemeral RunPod training pods, run the pipeline stages, and manage the persistent serving endpoint — with business/orchestration logic living **outside** RunPod. (Arch §12, §13.)

## Deliverables

### 1. `orchestration/runpod_controller.py`
- Via the RunPod API:
  - Launch an **ephemeral GPU pod** for a training/eval/quantize job (instance configurable: A100 80GB for Foundation, smaller for per-type adapters).
  - Pod clones the repo at a given commit, installs, pulls only needed artifacts from Blob (IMPL-02), runs the job, pushes results + manifest, then **terminates**.
  - Poll status, collect logs, handle failures + retries.
- Manage the **persistent Serverless vLLM endpoint** (IMPL-11): deploy/update to a promoted version, health-check.
- CLI: `train-foundation --corpus vN --commit <sha> --gpu a100-80`, `train-adapter ...`, `quantize ...`, `deploy-endpoint --model vN`.

### 2. `orchestration/pipeline_dag.py`
- The end-to-end pipeline (arch §12 stages 1–11) as an ordered DAG (Airflow OR GitHub Actions; keep stage functions reusable):
  1. ingest → 2. OCR → 3. label → 4. dataset build → 5. train (foundation/adapters) → 6. evaluate + gate → 7. merge → 8. quantize + validate → 9. push artifacts → 10. deploy endpoint → 11. feedback loop (active-learning queue).
- Idempotent, resumable stages; a failed stage doesn't corrupt state.
- **Gating (IMPL-08) is a hard stop**: no promote/deploy unless the candidate beats current.

### 3. `orchestration/config/`
- Which GPU per stage, corpus/version tags, schedule (e.g., weekday retrains), notification hooks.

## Constraints
- Training pods **ephemeral** — nothing persistent resident; code cloned fresh.
- Orchestration/business logic on cheap CPU infra outside RunPod; only GPU-bound work on RunPod (arch §13).
- Every training/eval/quantize job writes a RunManifest (IMPL-02).
- Secrets from env only.

## Acceptance checklist
- [ ] Controller launches a pod, runs a trivial job, pushes output, tears down.
- [ ] `deploy-endpoint --model vN` updates the serving endpoint.
- [ ] DAG runs stages in dependency order; a failed stage halts promotion.
- [ ] Gating blocks deploy on regression.
- [ ] No persistent process left on a training pod after completion.


===================================================================================

# SPEC 14 — Unit Tests + CI Fixtures

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: all prior specs (01–13).

## Goal

A test suite + CI that verifies the correctness-critical pieces without GPUs or live Azure, plus small fixtures.

## Deliverables

### 1. `tests/fixtures/`
- 2–3 tiny sample PDFs per active doc type (or synthetic stand-ins), MinerU-style OCR markdown, matching golden JSON — CI-small.
- Seeded synthetic logprob/generation samples for calibration + confidence tests.
- In-memory mock Blob backend (no live Azure).

### 2. Core tests (mirror each spec's acceptance checks)
- `test_data_collator.py` — **label masking correct**: loss only on assistant JSON tokens; system/image/OCR masked to `-100` (IMPL-06). Highest-value test.
- `test_schema_validity.py` — schemas load; valid examples pass; malformed/wrong-type fail (IMPL-01/08).
- `test_split_leakage.py` — no `source_id` crosses train/val/test; expansion after split (IMPL-05).
- `test_modality_mix.py` — 50/20/30 holds; image_only rows omit OCR (IMPL-05).
- `test_inference_core.py` — `build_messages` identical across contexts; `map_field_spans` correct (IMPL-07). **This guards test==prod at the primitive level.**
- `test_metrics.py` — normalized match (dates/currency/names), list recall, F1 correct (IMPL-08).
- `test_gating.py` — a regression on any single metric blocks promotion (IMPL-08).
- `test_calibration.py` — temperature + isotonic reduce ECE; params persist/reload; list-completeness flags dropped rows (IMPL-09).
- `test_run_registry.py` — RunManifest validates; `adapters_depending_on` + `resolve_model_version` correct (IMPL-02).
- `test_router.py` — low classifier confidence → Foundation-only fallback + review flag; deterministic selection (IMPL-11).
- `test_source_id.py` — id build/parse/validate round-trips (IMPL-01).
- `test_test_prod_parity.py` — testing output == serving pipeline output on the same fixture (IMPL-11/12). Since testing reuses `serving/pipeline.py`, this asserts they don't diverge.

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
