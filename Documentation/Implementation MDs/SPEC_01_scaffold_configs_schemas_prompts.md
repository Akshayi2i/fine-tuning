# SPEC 01 — Project Scaffold, Configs, Schemas, Prompts

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: none. This spec creates the repo skeleton and all static configuration that later modules read.
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §0a (schema contract), §0b (LoB), §6 (modality instructions), §7 (dataset format, schema registry, prompt template), §9/§9a (LoRA targets), §11 (full hyperparameter spec), §11a (sweeps), §13a/§13b (quantization), §19 (repo structure).

## Goal

Scaffold the `insurance-extraction-finetuning` repo and create every config file, JSON schema, and prompt template that the rest of the system depends on.

## Deliverables

### 1. Repo root
- `README.md` — project overview, setup steps, how to run each pipeline stage (stub with section headings + the module map from master §5). State prominently that this repo implements the **L3 VLM layer** of the Fideon pipeline and that its output schema is owned by **Fideon SPEC_00** (master §1.1).
- `pyproject.toml` — Python 3.11+, dependencies grouped by extra: `[data]` (azure-storage-blob, pypdf, pillow, python-dotenv, pydantic), `[train]` (ms-swift, transformers, peft, bitsandbytes, accelerate, deepspeed, trl, flash-attn), `[serve]` (vllm), `[eval]` (numpy, scikit-learn, jsonschema), `[track]` (mlflow or wandb), `[dev]` (pytest, ruff, mypy). Pin major versions; comment that exact versions are verified at implementation time.
- `requirements.txt` — generated equivalent for pip installs on RunPod.
- `.env.example` — every required env var with placeholder: `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_BLOB_CONTAINER`, `AZURE_RAW_CONTAINER` (separately-permissioned raw-documents container, master §8), `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, `RUNPOD_VOLUME_ID`, `RUNPOD_VOLUME_MOUNT` (default `/runpod-volume`, the staging volume — master §12a), `HF_TOKEN`, `HF_MODEL_REVISION`, `MLFLOW_TRACKING_URI` / `WANDB_API_KEY` (optional), `DEFAULT_TENANT_ID` (single-tenant default), `ALLOW_EXTERNAL_PREANNOTATION` (default `false`), plus any external-endpoint URLs. Comment each.
- `.gitignore` — Python, `.env`, model weights, `*.pdf`, `results/`, `metrics/`, `ocr_cache/`, `calibration_store/`, `__pycache__`.

### 2. `configs/`
- **`base_model.yaml`** — `model_id: Qwen/Qwen3-VL-8B-Instruct`, `revision: <pin>`, quantization block (**`load_in_4bit: false` by default — the base is held in bf16**; the `bnb_4bit_*` keys stay populated so flipping the flag on a VRAM-constrained pod needs no other edit, and are read only when it is true, per arch §9.2), `attn_implementation: flash_attention_2`, resolution cap (`max_image_long_side_px: 1792`, valid range 1536–2048 per arch §11), `max_seq_len` (comment: **set from the 95th-percentile token count measured on the real corpus**, not guessed).
- **`configs/training/foundation.yaml`** — the full parameter set from arch §11, not just the summary:
  - LoRA: rank 64, alpha 128, dropout 0.05, `bias: none`, target modules from master §2.
  - Optimization: LR `2e-4` (range 1e-4–2e-4), cosine schedule, warmup ratio 0.03–0.05, epochs 3 (range 2–3), **optimizer AdamW paged 8-bit**, β₁ 0.9, β₂ 0.999, ε 1e-8, weight decay 0.01, max grad norm 1.0.
  - Batch/memory: per-device train batch 1–2, grad accumulation set to reach **effective batch 32–64**, gradient checkpointing on, mixed precision bf16.
  - Vision: `train_vit: false` (the §3 gate default), resolution cap inherited from `base_model.yaml`.
  - Eval/checkpointing: eval per fixed step interval, **early stopping on val loss + field-level F1, patience 2**, save per step interval retaining best, `metric_for_best_model: field_f1`, fixed recorded `seed`, logging interval.
  - Top comment: "hyperparameters are sweep starting points (arch §11/§11a) — tune on the validation set, not fixed truth."
- **`configs/training/{acord,lossrun,policy}_adapter.yaml`** — LoRA rank 16 / alpha 32 / dropout 0.05, LR `7e-5` (range 5e-5–1e-4), epochs 4 (range 3–5), **effective batch 16–32**; each references its `doc_type`, its schema file, and `foundation_version: <set at runtime>`.
  - **One shared `acord` adapter only** (arch §4b Option 1). Per-form adapter configs (`acord25`/`acord125`/`acord140`) are **not created** — with limited data per form, one adapter learning shared "ACORD-ness" generalizes better than several data-starved ones. Add them only when a form has 1000+ examples *and* eval shows the shared adapter underperforming. The per-form **schemas** still exist, because the classifier must select the right schema regardless.
- **`configs/sweeps/`** *(arch §11a — **author the configs now, run them after the SPEC_15 pilot**)* — the bounded 3-phase sweep definitions, as W&B Sweep / MLflow configs. A 9–12 run sweep against 25–30 docs/type mostly measures noise; arch §11a scopes it to "before the first **production** run", not before the pilot:
  - `phase1_lr.yaml` — Foundation `{5e-5, 1e-4, 2e-4}`, per-type `{2e-5, 5e-5, 1e-4}`; 1 epoch per candidate; metric = validation loss; 3 runs per adapter type.
  - `phase2_epochs.yaml` — Foundation `{2, 3, 4}`, per-type `{3, 4, 5}` with early stopping patience 2; metric = validation field-level F1; 3 runs per adapter type.
  - `phase3_rank.yaml` — Foundation rank `{32, 64, 128}`; **secondary, not run by default** — only if F1 plateaus.
  - Comment the total budget: **~9–12 training runs before the first production run**, and that **every sweep run writes a full run manifest** (SPEC_02) so sweeps are first-class registry entries, not untracked side experiments.
- **`configs/deepspeed/zero2.json`** and **`zero3.json`** — ZeRO-2 default, ZeRO-3 for VRAM-constrained multi-GPU.
- **`configs/inference/vllm_serving.yaml`** — resolution cap (**must match `base_model.yaml`**), max_seq_len, adapter routing map (doc_type → adapter path), `enable_logprobs: true`, served model tag, classifier confidence threshold, **review confidence threshold (default 0.7, arch §5)**, long-document page-count threshold (default `>5` pages, arch §7).

### 3. `schemas/` — JSON Schemas (Draft 2020-12)

**These are not invented schemas — they are the Fideon SPEC_00 Pydantic models serialised to JSON Schema** (master §1.1). Generate them from the Fideon models where available; hand-authored versions are a temporary stand-in and must be reconciled before the first real corpus build.

- `acord25.schema.json` — Certificate of Liability: insured, insurer(s), policy numbers, effective/expiration dates, coverage limits (GL, auto, umbrella), certificate holder.
- `acord125.schema.json` — Commercial Application: applicant, business info, premises, coverages requested.
- `acord140.schema.json` — Property section (per arch §4b two-level ACORD classification).
- `lossrun.schema.json` — carrier, policy number, valuation date, and a **`claims` array** (repeating rows: claim number, loss date, status, paid, reserved, total incurred, description). This is the key list-field type and the one where a **missed row** is both easy and expensive (arch §5).
- `policy_doc.schema.json` — named insured, policy number, effective/expiration, coverage schedule (array), premium, endorsements list.

**Every schema must additionally include:**
- **`line_of_business`** (arch §0b) — enum `workers_comp | general_liability | commercial_auto | property | umbrella`, **nullable**, required key in every schema. Factor the enum into `schemas/lob.enum.json` and `$ref` it so there is exactly one definition.
- **A `description` on every field** — see the semantic-gloss rules below (master §1.4). This is not documentation; it is injected into the prompt and is part of what the model is trained on.
- Explicit `null` unions for optional fields — "absent" must be representable as `null`, never as a missing key or a hallucinated value.
- Date fields as `string` with a `YYYY-MM-DD` format note.
- `required` vs optional marked deliberately.

**Field descriptions — semantic glosses with exclusions (master §1.4)**

The same field appears under many surface labels across documents (*Insured Name*, *Named Insured*, *Applicant*, *Name of Applicant*, …). The model has to map all of them onto one canonical key. Its main aid is the field's `description`, so write these as **definitions with explicit exclusions**, never as label lists:

```json
"insured_name": {
  "type": ["string", "null"],
  "description": "The party purchasing the insurance policy. Not the certificate
                  holder, the producer/agency, or an additional insured."
}
```

The exclusion clause is the part doing the real work — it teaches the **boundary** between confusable entities, which insurance documents are dense with (certificate holder, producer, additional insured, loss payee, carrier). A definition generalises to phrasings nobody listed; a list of aliases does not, and gives the model a lexical prior that makes confusable errors *worse*.

Rules for writing them:
- Define what the field **is**, in the document's own domain terms.
- Add an exclusion clause naming the confusable entities it must not be.
- One or two lines. These ride on every training row and every production request.
- **Never enumerate surface labels here.** Those live in the alias registry below, which never enters the prompt.

### 3a. `schemas/aliases/{doc_type}.aliases.json` — the alias registry

A versioned sidecar per document type, recording the surface forms actually observed and the entities that must not be confused with them:

```json
{
  "insured_name": {
    "aliases": ["Insured Name", "Named Insured", "Applicant", "Name of Applicant",
                "Applicant Name", "Name Insured", "Insured"],
    "confusables": ["Certificate Holder", "Producer", "Agency",
                    "Additional Insured", "Loss Payee", "Mortgagee", "Carrier"]
  },
  "policy_number": {
    "aliases": ["Policy Number", "Policy No.", "Policy #", "Certificate Number"],
    "confusables": ["Quote Number", "Claim Number", "Binder Number"]
  }
}
```

**Three consumers, and one hard prohibition:**

| Consumer | Use |
|---|---|
| SPEC_04 | Annotator rulebook — so three reviewers apply one consistent mapping, and a label naming a confusable is rejected |
| SPEC_05 | Alias coverage counting in the corpus manifest |
| SPEC_08 | Per-alias eval slicing and the confusable-misattribution metric |

**It is never rendered into the prompt and never consulted at inference** (master §1.4 anti-pattern). Adding a newly observed surface form is a registry edit, **not** a schema change — it triggers no corpus rebuild. That asymmetry is deliberate: the registry can grow freely as annotators encounter new phrasings, while the glosses stay stable.

Each schema must be loadable by `jsonschema` and validate a correct example. Include one valid example JSON per schema under `schemas/examples/{type}.example.json`, each exercising a populated `line_of_business` and at least one `null` field.

**Schema versioning:** each schema file carries a `$id` with a version, and the registry exposes the active version string. Master §7 — a schema change forces a corpus rebuild and a new training cycle.

### 4. `prompts/`
- **`system_prompt_template.jinja`** — parametrized by `{{ doc_type }}`, `{{ schema_json }}`, `{{ modality_mode }}`. Must:
  - State the doc type and inject the **exact schema from the registry** — never hand-typed per example (arch §7).
  - Give extraction rules: valid JSON only, no markdown fences, absent fields → `null`, date format, how to handle repeating table rows.
  - Include the **modality instruction line**, explicitly declared rather than silently omitted (arch §6):
    - `ocr_plus_image`: "The OCR text is the primary source; use the page image to verify and correct OCR errors."
    - `noisy_ocr_image`: identical to `ocr_plus_image` — **the noise lives in the data, not the instruction**.
    - `image_only`: "No OCR text is provided — extract directly from the images."
  - Carry a **template version string** rendered into the prompt metadata (not the prompt body), recorded in the corpus manifest (SPEC_05).
- **`doc_type_classifier_prompt.jinja`** — zero-shot prompt asking the model to name the doc type (`acord`/`policy`/`lossrun`) and, if acord, the form number — the **two-level classification** of arch §4b. Output a small fixed JSON: `{"doc_type": "...", "acord_form": "...", "confidence": 0.0}`.

**Non-negotiable:** `common.prompts.render(...)` is the single renderer used by dataset build (SPEC_05), serving (SPEC_11), and testing (SPEC_12). Training-time and inference-time prompts must render identically (arch §7).

- The template renders each field's **`description` alongside its type**, so the semantic glosses (§3) reach the model. No structural change is needed — descriptions ride along with the already-injected schema.
- The template **must not** reference the alias registry (master §1.4).

### 5. `common/` (shared library — create this package)
- `common/config.py` — pydantic settings loader (env + a given YAML), one entrypoint `load_config(path)`.
- `common/schemas.py` — load a schema by `doc_type` (+ `acord_form`), validate a JSON object against it, list required fields, expose the active schema version.
- `common/prompts.py` — render the jinja templates given doc_type + modality_mode + schema; expose `prompt_template_version`.
- `common/ids.py` — `source_id` helpers: build/parse `{doc_type}_{index}`, validate format.
- `common/lob.py` *(new)* — the LOB enum, validation, `null` handling, and a per-value coverage counter used by SPEC_05 and SPEC_08.
- `common/normalize.py` *(moved here from SPEC_08)* — per-field-type normalizers so comparison reflects correctness rather than formatting: dates (`01/01/2026` == `2026-01-01`), currency (`$1,200.00` == `1200.0`), entity names (`Acme Mfg LLC` ≈ `ACME MANUFACTURING LLC`), policy/claim number separators. **Built here rather than in SPEC_08 because three specs need it and the earliest is SPEC_04**: alias derivation (SPEC_04), the promotion gate (SPEC_08), and the testing routine (SPEC_12). Pure and dependency-free. One definition of "matches" shared by all three — divergence silently changes what "correct" means between them.
- `common/aliases.py` *(new)* — loads `schemas/aliases/{doc_type}.aliases.json`; `aliases_for(doc_type, field)`, `confusables_for(doc_type, field)`, `canonical_for(doc_type, surface_label)` (reverse lookup, **for labeling and eval only**), `is_confusable(doc_type, field, label)`. Mirrors the `common/lob.py` pattern. Its module docstring must state that no serving or inference path may import it.
- `common/constants.py` — `ACTIVE_DOC_TYPES = ["acord", "policy", "lossrun"]`, **`UNCLASSIFIED = "unclassified"`** (the holding bucket for documents whose type is not yet known — it lives here, not in `data_pipeline`, because `common/ids.py` and `artifact_registry/paths.py` both have to honour it and both sit below `data_pipeline`), the **`doc_type` → Fideon SPEC_00 canonical model mapping table (master §1.2)**, ACORD form list, modality modes + mix ratios, resolution cap default, split-ratio table by volume, confidence review threshold.

## Constraints
- No business logic yet — scaffold + static assets + tiny shared helpers.
- Everything imports cleanly (`python -c "import common"`).
- **The doc_type ↔ canonical-model mapping exists in exactly one place** (`common/constants.py`).
- The LOB enum exists in exactly one place (`schemas/lob.enum.json`, surfaced by `common/lob.py`).
- Follow all global conventions in master §7.

## Acceptance checklist
- [ ] Repo tree matches master §5 (empty dirs get a `.gitkeep`).
- [ ] `pip install -e .[dev]` succeeds (or documents which heavy deps are RunPod-only).
- [ ] All schemas load and validate their example JSON via `jsonschema`.
- [ ] **Every schema requires `line_of_business` with the correct enum + null union**, and rejects an out-of-enum value.
- [ ] `common.prompts.render(...)` produces a correct system prompt for each of the 3 modality modes, and `noisy_ocr_image` renders **identically** to `ocr_plus_image`.
- [ ] **Every field in every schema has a non-empty `description`**, and each description for a confusable-prone field carries an exclusion clause.
- [ ] The rendered prompt **contains** the field descriptions and **contains no alias strings**.
- [ ] `schemas/aliases/{doc_type}.aliases.json` loads for every active doc type; no string appears in both `aliases` and `confusables` for the same field.
- [ ] `common.aliases.canonical_for("policy", "Applicant")` returns `insured_name`; `is_confusable("policy", "insured_name", "Certificate Holder")` is true.
- [ ] `common.constants` resolves `lossrun → loss_run/LossRunDocument` and the other two active types.
- [ ] `configs/training/foundation.yaml` contains every parameter in arch §11's full specification tables (not just the summary table).
- [ ] `configs/sweeps/` defines the 3 phases with the documented candidate sets and budgets (authored, not executed this cycle).
- [ ] `configs/inference/vllm_serving.yaml` resolution cap equals `configs/base_model.yaml` — assert this in a test.
- [ ] `.env.example` lists every env var used anywhere in the design.
- [ ] `ruff` and `mypy` pass on `common/`.
