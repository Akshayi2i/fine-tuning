# SPEC 01 — Project Scaffold, Configs, Schemas, Prompts

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: none. This spec creates the repo skeleton and all static configuration that later modules read.
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §0a (schema contract), §0b (LoB), §6 (modality instructions), §7 (dataset format, schema registry, prompt template), §9/§9a (LoRA targets), §11 (full hyperparameter spec), §11a (sweeps), §13a/§13b (quantization), §19 (repo structure).

## Goal

Scaffold the `insurance-extraction-finetuning` repo and create every config file, JSON schema, and prompt template that the rest of the system depends on.

## Deliverables

### 1. Repo root
- `README.md` — project overview, setup steps, how to run each pipeline stage (stub with section headings + the module map from master §5). State prominently that this repo implements the **L3 VLM layer** of the Fideon pipeline and that its output schema is owned by **Fideon SPEC_00** (master §1.1).
- `pyproject.toml` — Python 3.11+. **The only place versions are declared**, in dependency groups: `[data]`, `[ocr]` (MinerU 1.x), `[train]`, `[serve]`, `[quantize]`, `[eval]`, `[track]`, `[dev]`. See *Current implementation* below for the pinned versions and why each group is separate.
- `requirements.txt` and `requirements-{ocr,train,serve,quantize}.txt` — one per machine, each only `-e .[groups]`: they choose groups, never versions. Installed on a pod by `scripts/setup_pod.sh <role>`.
- `.env.example` — every required env var with placeholder: `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_BLOB_CONTAINER`, `AZURE_RAW_CONTAINER` (separately-permissioned raw-documents container, master §8), `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, `RUNPOD_VOLUME_ID`, `RUNPOD_VOLUME_MOUNT` (default `/runpod-volume`, the staging volume — master §12a), `HF_TOKEN`, `HF_MODEL_REVISION`, `MLFLOW_TRACKING_URI` / `WANDB_API_KEY` (optional), `DEFAULT_TENANT_ID` (single-tenant default), `ALLOW_EXTERNAL_PREANNOTATION` (default `false`), `FIDEON_BASE_MODEL_DIR` (overrides the local base-model directory), `FIDEON_ALLOW_OFF_POD` (train on a non-RunPod GPU host), `FIDEON_NO_DETACH` (run one command in the foreground on the pod), plus any external-endpoint URLs. Comment each. `tests/test_repo_contracts.py` fails when the code reads a variable the template does not list, or the template lists one nothing reads.
- `.gitignore` — Python, `.env`, model weights, `*.pdf`, `results/`, `metrics/`, `ocr_cache/`, `calibration_store/`, `__pycache__`.

### 2. `configs/`
- **`base_model.yaml`** — `model_id: Qwen/Qwen3-VL-8B-Instruct`, `revision: <pin>`, **`local_dir: /workspace/models`** (where the pod keeps the weights; every loader reads them from there), quantization block (**`load_in_4bit: false` by default — the base is held in bf16**; the `bnb_4bit_*` keys stay populated so flipping the flag on a VRAM-constrained pod needs no other edit, and are read only when it is true, per arch §9.2), `attn_implementation: flash_attention_2`, resolution cap (`max_image_long_side_px: 1792`, valid range 1536–2048 per arch §11), `max_seq_len` (comment: **set from the 95th-percentile token count measured on the real corpus**, not guessed).
- **`configs/training/unified.yaml`** (was `foundation.yaml` under v1's two-level design) — the full parameter set from arch §11, not just the summary:
  - LoRA: rank 64, alpha 128, dropout 0.05, `bias: none`, target modules from master §2.
  - Optimization: LR `2e-4` (range 1e-4–2e-4), cosine schedule, warmup ratio 0.03–0.05, epochs 3 (range 2–3), **optimizer AdamW paged 8-bit**, β₁ 0.9, β₂ 0.999, ε 1e-8, weight decay 0.01, max grad norm 1.0.
  - Batch/memory: per-device train batch 1–2, grad accumulation set to reach **effective batch 32–64**, gradient checkpointing on, mixed precision bf16.
  - Vision: `train_vit: false` (the §3 gate default), resolution cap inherited from `base_model.yaml`.
  - Eval/checkpointing: eval per fixed step interval, **validation loss tracked (`metric_for_best_model: eval_loss`, `greater_is_better: false`); no early-stopping patience is passed — ms-swift 3 takes none, and what ships is chosen by generated field accuracy over every saved checkpoint**, save per step interval with `save_total_limit: 4` so checkpoint selection has its candidates — the checkpoint that ships is chosen afterwards by generated `field_normalized_match` (IMPL-06 §3), never by loss, fixed recorded `seed`, logging interval.
  - Top comment: "hyperparameters are sweep starting points (arch §11/§11a) — tune on the validation set, not fixed truth."
- **Not created under v2.1:** `configs/training/{acord,lossrun,policy}_adapter.yaml`. There is one unified adapter (arch §4.1); a per-type config arrives only with a type that passes the §4.2 graduation gate. The original v1 description, for reference — LoRA rank 16 / alpha 32 / dropout 0.05, LR `7e-5` (range 5e-5–1e-4), epochs 4 (range 3–5), **effective batch 16–32**; each references its `doc_type`, its schema file, and `foundation_version: <set at runtime>`.
  - **One shared `acord` adapter only** (arch §4b Option 1). Per-form adapter configs (`acord25`/`acord125`/`acord140`) are **not created** — with limited data per form, one adapter learning shared "ACORD-ness" generalizes better than several data-starved ones. Add them only when a form has 1000+ examples *and* eval shows the shared adapter underperforming. The per-form **schemas** still exist, because the classifier must select the right schema regardless.
- **`configs/sweeps/`** *(arch §11a — **author the configs now, run them after the IMPL-15 pilot**)* — the bounded 3-phase sweep definitions, as W&B Sweep / MLflow configs. A 9–12 run sweep against 25–30 docs/type mostly measures noise; arch §11a scopes it to "before the first **production** run", not before the pilot:
  - `phase1_lr.yaml` — Foundation `{5e-5, 1e-4, 2e-4}`, per-type `{2e-5, 5e-5, 1e-4}`; 1 epoch per candidate; metric = validation loss; 3 runs per adapter type.
  - `phase2_epochs.yaml` — Foundation `{2, 3, 4}`, per-type `{3, 4, 5}` with early stopping patience 2; metric = validation field-level F1; 3 runs per adapter type.
  - `phase3_rank.yaml` — Foundation rank `{32, 64, 128}`; **secondary, not run by default** — only if F1 plateaus.
  - Comment the total budget: **~9–12 training runs before the first production run**, and that **every sweep run writes a full run manifest** (IMPL-02) so sweeps are first-class registry entries, not untracked side experiments.
- **`configs/deepspeed/zero2.json`** and **`zero3.json`** — ZeRO-2 default, ZeRO-3 for VRAM-constrained multi-GPU.
- **`configs/inference/vllm_serving.yaml`** — resolution cap (**must match `base_model.yaml`**), max_seq_len, adapter routing map (doc_type → adapter path), `enable_logprobs: true`, served model tag, classifier confidence threshold, **review confidence threshold (default 0.7, arch §5)**. The long-document page threshold is **not** set here any more: it is `common.constants.DEFAULT_LONG_DOC_PAGE_THRESHOLD`, the value the corpus build planned its training windows with, so serving cannot drift from training.

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
| IMPL-04 | Annotator rulebook — so three reviewers apply one consistent mapping, and a label naming a confusable is rejected |
| IMPL-05 | Alias coverage counting in the corpus manifest |
| IMPL-08 | Per-alias eval slicing and the confusable-misattribution metric |

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
  - Carry a **template version string** rendered into the prompt metadata (not the prompt body), recorded in the corpus manifest (IMPL-05).
- **`doc_type_classifier_prompt.jinja`** — zero-shot prompt asking the model to name the doc type (`acord`/`policy`/`lossrun`) and, if acord, the form number — the **two-level classification** of arch §4b. Output a small fixed JSON: `{"doc_type": "...", "acord_form": "...", "confidence": 0.0}`.

**Non-negotiable:** `common.prompts.render(...)` is the single renderer used by dataset build (IMPL-05), serving (IMPL-11), and testing (IMPL-12). Training-time and inference-time prompts must render identically (arch §7).

- The template renders each field's **`description` alongside its type**, so the semantic glosses (§3) reach the model. No structural change is needed — descriptions ride along with the already-injected schema.
- The template **must not** reference the alias registry (master §1.4).

### 5. `common/` (shared library — create this package)
- `common/config.py` — pydantic settings loader (env + a given YAML), one entrypoint `load_config(path)`.
- `common/schemas.py` — load a schema by `doc_type` (+ `acord_form`), validate a JSON object against it, list required fields, expose the active schema version.
- `common/prompts.py` — render the jinja templates given doc_type + modality_mode + schema; expose `prompt_template_version`.
- `common/ids.py` — `source_id` helpers: build/parse `{doc_type}_{index}`, validate format.
- `common/lob.py` *(new)* — the LOB enum, validation, `null` handling, and a per-value coverage counter used by IMPL-05 and IMPL-08.
- `common/normalize.py` *(moved here from IMPL-08)* — per-field-type normalizers so comparison reflects correctness rather than formatting: dates (`01/01/2026` == `2026-01-01`), currency (`$1,200.00` == `1200.0`), entity names (`Acme Mfg LLC` ≈ `ACME MANUFACTURING LLC`), policy/claim number separators. **Built here rather than in IMPL-08 because three specs need it and the earliest is IMPL-04**: alias derivation (IMPL-04), the promotion gate (IMPL-08), and the testing routine (IMPL-12). Pure and dependency-free. One definition of "matches" shared by all three — divergence silently changes what "correct" means between them.
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
- [ ] `configs/training/unified.yaml` contains every parameter in arch §11's full specification tables (not just the summary table).
- [ ] `configs/sweeps/` defines the 3 phases with the documented candidate sets and budgets (authored, not executed this cycle).
- [ ] `configs/inference/vllm_serving.yaml` resolution cap equals `configs/base_model.yaml` — assert this in a test.
- [ ] `.env.example` lists every env var used anywhere in the design.
- [ ] `ruff` and `mypy` pass on `common/`.

---

## Current implementation (2026-09-27)

**Dependencies** (`pyproject.toml`; resolved for Linux with `uv` — the resolver found three failures that
would have broken the pods, fixed below):

| Group | Holds | Why separate |
|---|---|---|
| `data` | azure-storage-blob, pypdf, pillow, PyMuPDF | Blob and page rendering, everywhere |
| `ocr` | `magic-pdf[full]>=1.3,<2` | MinerU 1.x: the code imports `magic_pdf`, which 2.x removed |
| `train` | torch `>=2.8,<2.9`, transformers `>=4.57`, peft, bitsandbytes, accelerate, deepspeed, ms-swift `>=3.9,<4`, qwen-vl-utils | torch pinned to the minor vLLM 0.11 pins: the training pod also runs vLLM |
| `serve` | vLLM `==0.11.0`, transformers `>=4.57` | exact, and the same on both pods: calibration is fitted on the training pod's vLLM |
| `quantize` | `llmcompressor>=0.8` (its PyPI name) | needs datasets `>=4` (ms-swift 3 needs `<4`) and a transformers range vLLM 0.11 excludes: its own environment |
| `eval`, `track`, `dev` | numpy/scikit-learn, mlflow, pytest/ruff/mypy/pyarrow | |

flash-attn is in no group: it must be built against the installed torch with `--no-build-isolation`,
which `scripts/setup_pod.sh train` does as a second step. `tests/test_dependencies.py` fails when the code
imports a package no group installs (MinerU was exactly that), or a requirements file names an unknown group.

**Pod roles** (`scripts/setup_pod.sh ocr|train|serve|quantize|dev`): installs in order, builds flash-attn,
installs tmux, checks every expected package and that torch sees CUDA. On the pod it runs itself in tmux.

**Two environments, not one.** `magic-pdf[full]` 1.x and `vllm==0.11.0` / torch 2.8 do not resolve together,
so `ocr` and `train` are separate environments: `scripts/pod_bootstrap.sh` builds `/workspace/venv` (train,
which includes serve) and `/workspace/venv-ocr` (ocr) on the volume and records the requirements each was
built from, so a re-run reinstalls only what changed.

**Lock files** (`requirements-{train,ocr,serve,quantize}.lock`, from `scripts/lock_requirements.sh` with uv, for
Linux x86-64 / Python 3.11): `setup_pod.sh` installs each role with `-c <lock>`, so every package is pinned.
Resolving them found what ranges alone hid: the serving environment took **transformers 5.x** (vLLM 0.11.0 sets
no upper bound) and the OCR and quantize environments took the newest torch, built for a CUDA newer than the
pod's. Now `transformers>=4.57,<4.58` in train and serve, `torch>=2.8,<2.9` in every environment, `pillow<12`
and `PyMuPDF<1.25` everywhere (one renderer and one resizer for the OCR environment's training pages and for
serving), and the serve lock is resolved inside the train lock so serving runs exactly what calibration saw.
`flash-attn<3`. The OCR role installs `libgl1`/`libglib2.0-0` (OpenCV). The release bundle's `lockfile_hash` is
the train lock's. `tests/test_dependency_locks.py` fails on a stale lock or drift between environments.

**Base model revision** pinned to `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (Hub commit of 2025-10-15, four
safetensors shards, 17.5 GB); the bootstrap downloads exactly that into `local_dir`.

**`.env.example`**: the Azure connection string is quoted — the file is sourced by bash on the pod, which
would cut an unquoted value at its first `;`. `HF_HOME` is documented (on the pod,
`/workspace/.cache/huggingface`) and `MINERU_TOOLS_CONFIG_JSON` may be an absolute path.

**Vision budget** (`configs/shared/vision.yaml`): budgets are whole 32×32 visual tokens — `max_pixels`
`2483200` (was `2483712`, a Qwen2.5 28×28 figure that made `MAX_PIXELS` and `IMAGE_MAX_TOKEN_NUM` name
different budgets). `common.config.pixel_budget()` is the one source for the trainer's env and the vLLM
engine's `mm_processor_kwargs`, and refuses a non-multiple.

**Section map** (`configs/schema_sections.yaml`): the policy section groups (decl, arrays, lineblk, dtd),
their page rules and identifying keys, read by the window planner in training and serving.
