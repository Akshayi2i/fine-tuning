# SPEC 01 — Project Scaffold, Configs, Schemas, Prompts

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: none. This spec creates the repo skeleton and all static configuration that later modules read.

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
