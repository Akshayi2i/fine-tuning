# SPEC 00 — Master Context (Read First)

> **Purpose of this file.** This is the shared context every other spec assumes. When you run any individual spec (SPEC_01 … SPEC_15) in Claude Code, that spec references this file for global conventions. Read this once; each module spec then adds only what's specific to it.
>
> **Source of truth.** This spec set and the architecture document **`Documentation/finetuning-architecture-v1.md`** are **in sync** as of this revision. Where a spec cites a section like "(arch §4)", it refers to that document.
>
> **Precedence.** The architecture carries the *why* — design rationale, trade-offs, the reasoning behind each decision. These specs carry the *how* — module boundaries, file paths, function signatures, acceptance criteria. Neither outranks the other: they describe the same system at different altitudes. **If they disagree, that is a bug in one of them.** Fix the disagreement rather than picking a winner, and say which one you changed.
>
> Anything genuinely build-level — a CLI flag, a module name, a test — lives here and is not expected to appear in the architecture.
>
> **Revision:** aligned to `finetuning-architecture-v1.md` (architecture v1). This supersedes the earlier spec set derived from `qwen3vl-insurance-extraction-finetuning-architecture.md`; all architecture section numbers were renumbered in v1 (see §0.2 below).

---

## 0. Read this before anything else — two naming hazards

### 0.1 "SPEC_xx" means two different things

The architecture document references **Fideon pipeline specs** that are *external to this repository*, and they happen to use the same `SPEC_nn` numbering as these implementation specs. They are not the same documents.

| Reference in the architecture doc | What it actually is | Our equivalent |
|---|---|---|
| **Fideon SPEC_00** | The canonical output-schema spec — Pydantic models in `fideon/schemas/` | Consumed by our SPEC_01 (`schemas/`) |
| **Fideon SPEC_01** | L0→L1→L2→L3 routing spec (where the VLM sits in the pipeline) | Context only; see §1.1 |
| **Fideon SPEC_07 Stage 3** | The production **audit gate** that schema-validates every VLM output | Mirrored by our SPEC_08 `schema_validity` + our SPEC_11 output validation |
| **Fideon SPEC_11** | Presidio **de-identification** step | **BLOCKED** — see SPEC_05 §1 and §8 below |
| **Fideon SPEC_12** | **Multi-tenant deployment** model | Implemented for corpus/adapters in our SPEC_02/05/06 |

**Convention for this spec set:** always write **"Fideon SPEC_nn"** when referring to the external pipeline specs, and a bare **"SPEC_nn"** when referring to a file in this folder. Never let a bare `SPEC_07` mean the audit gate — in this folder `SPEC_07` is the inference core.

### 0.2 Architecture section numbers changed in v1

The previous spec set cited the old architecture numbering. Everything has been re-cited against v1. For anyone cross-reading old notes:

| Topic | Old § | **v1 §** |
|---|---|---|
| Pipeline integration / schema contract / LoB | *(did not exist)* | **§0, §0a, §0b** |
| Qwen3-VL recap + ViT escalation gate | §2 | **§3** |
| Adapter strategy (Foundation → per-type) | §3 | **§4** |
| Doc-type classifier | §3a | **§4a** |
| ACORD sub-types | §3b | **§4b** |
| Day-zero bootstrap | *(did not exist)* | **§4c** |
| Confidence (logprobs + calibration) | §4 | **§5** |
| Dual-input-mode training | §5 | **§6** |
| Dataset format / golden JSON / long docs | §6 | **§7** |
| Corpus management + split strategy | §7 | **§8** |
| MinerU version pinning | *(did not exist)* | **§8a** |
| Multi-tenant corpus isolation | *(did not exist)* | **§8b** |
| Fine-tuning technique (LoRA on a bf16 base) | §8 | **§9** |
| LoRA target-module justification | *(did not exist)* | **§9a** |
| Trainer stack | §9 | **§10** |
| Hyperparameters | §10 | **§11** |
| Hyperparameter sweep methodology | *(did not exist)* | **§11a** |
| Versioning + run registry | §11 | **§12** |
| End-to-end pipeline | §12 | **§13** |
| GGUF format matrix | §12a | **§13a** |
| Quantization quality thresholds | *(did not exist)* | **§13b** |
| RunPod infrastructure | §13 | **§14** |
| Evaluation framework | §14 | **§15** |
| Pilot validation protocol | *(did not exist)* | **§16** |
| Extraction / testing routine | §15 | **§17** |
| Azure Blob layout | §16 | **§18, §18a** |
| Repo structure | §17 | **§19** |

---

## 1. What is being built

A production-grade fine-tuning and extraction system for **insurance document extraction** using **Qwen3-VL-8B-Instruct**. The system ingests insurance PDFs (digital and scanned), runs OCR (MinerU), and uses a fine-tuned vision-language model to extract structured **JSON with per-field confidence scores**.

**Active document types (only these three for now):** `acord`, `policy`, `lossrun`.
Quote and Endorsement are deferred — design for easy addition but do not implement them.

**Two production inference modes, one model:**
1. `ocr_plus_image` — MinerU OCR text + page image, both passed to the model.
2. `image_only` — page image only, no OCR (model must still extract).

### 1.1 Where this sits: the L3 VLM layer (arch §0)

This system is the **L3 layer** of the Fideon document-extraction pipeline. L0 → L1 → L2 → L3 routing is defined in **Fideon SPEC_01**. L3 is invoked when the cheaper upstream layers cannot handle a document (scanned input, unknown carrier, structural inference failure).

Three contractual obligations follow, and every spec downstream assumes they hold:

1. **The training target JSON is the canonical Fideon SPEC_00 schema — not an internal format.** Our `schemas/` registry (SPEC_01) is the Fideon SPEC_00 Pydantic models serialised to JSON Schema. Field names, types, nesting, and null representation must match exactly.
2. **Any Fideon SPEC_00 schema change requires a corpus rebuild and a new training cycle.** Treat a schema bump like a breaking dependency upgrade, not a patch.
3. **Every golden label is validated against the Fideon SPEC_00 JSON Schema before admission to the corpus** (SPEC_04), and every production inference output is validated by the Fideon SPEC_07 Stage 3 audit gate. Corpus/schema drift therefore surfaces as a production validation failure, not as a silent quality regression.

### 1.2 Document-type vocabulary and the canonical model mapping

The architecture uses two vocabularies: the Fideon SPEC_00 **canonical model names** (arch §0a) and the **corpus/adapter `doc_type` tags** used everywhere else (arch §7 record fields, §17 CLI, §18 Blob layout, §19 repo tree). **Resolved decision for implementation:** `doc_type` stays short and stable (`acord|policy|lossrun`), and a single mapping table resolves it to the canonical schema. Implement the mapping once in `common/constants.py`; never hardcode either name elsewhere.

| `doc_type` (ours — corpus, adapters, paths, CLI) | Fideon SPEC_00 canonical key | Fideon SPEC_00 Pydantic model | Status |
|---|---|---|---|
| `lossrun` | `loss_run` | `LossRunDocument` | active |
| `policy` | `policy_check` | `PolicyCheckDocument` | active |
| `acord` | `acord_mapping` | `ACORDMappingDocument` | active (+ `acord_form` sub-type) |
| `quote` | `quote_gen` | `QuoteGenDocument` | **deferred — do not implement** |

> If Fideon SPEC_00 keys its models differently from the table above, the mapping table is the **only** place to change. Flag it rather than renaming paths.

### 1.3 Line of Business is a first-class VLM output (arch §0b)

L1 (carrier registry) and L2 (structural inference) both attempt LoB detection first. When L3 is invoked, **the VLM must detect and output `line_of_business` itself.**

- Valid values (the Fideon SPEC_00 LOB enum): `workers_comp`, `general_liability`, `commercial_auto`, `property`, `umbrella`.
- `null` when LoB cannot be determined from the document.
- Emitted in the standard confidence-bearing shape: `{"line_of_business": {"value": "workers_comp", "confidence": 0.92}}`.
- **Present in every golden JSON, for every document type, even when null** (SPEC_04 rejects labels missing the key).
- **Corpus coverage target: ≥20% of training examples per LoB value** in the real document population (SPEC_05 reports actual coverage in the manifest; under-coverage is a loud warning, not a silent pass).
- **Reported as its own metric, per LoB value, and it is a gating metric** (SPEC_08). It is never averaged into overall field accuracy — a rare class must not hide inside a healthy aggregate.

### 1.4 Canonical field mapping — the model does the semantic work

The same real-world field appears under many surface labels across documents. The party purchasing the policy shows up as *Insured Name*, *Named Insured*, *Applicant*, *Name of Applicant*, *Applicant of Insured*, or bare *Name*. **The golden JSON always uses the one canonical key** (`insured_name`), and so does the inference output.

**The contract:**

1. **Golden labels are canonical.** Annotators map whatever the document says onto the canonical key. Surface labels never appear as keys in a golden JSON.
2. **Inference output is canonical.** The extraction returns `insured_name` regardless of how the document phrased it.
3. **The VLM performs the mapping.** This is a core thing the Foundation LoRA is trained to do (arch §4, "insurance terminology & abbreviations") — recognising that *Applicant* on this page denotes the same field as *Named Insured* on that one.

**How the model is taught it — three mechanisms, all required:**

| Mechanism | Where | What it contributes |
|---|---|---|
| **Semantic gloss in the schema** | SPEC_01 — every field carries a `description` defining what it means **and what it excludes** | A semantic anchor ("find the party purchasing the insurance"), so unseen phrasings resolve. Also lifts base-model pre-annotation quality from day one. |
| **Corpus coverage across variants** | SPEC_05 — alias coverage tracked per canonical field | The mapping is learned from examples; a variant appearing twice is learned weakly |
| **Confusable co-occurrence examples** | SPEC_05 — required edge case | Teaches the **boundary**, not just the mapping |

> ### Anti-pattern — no runtime alias lookup, ever
>
> It is tempting to map extracted labels onto canonical keys with a lookup table at inference. **Do not.** A lookup caps the system at the list someone wrote, cannot handle the phrasing nobody anticipated, and defeats the entire reason for fine-tuning a VLM. The alias registry (SPEC_01) is **training, labeling, and evaluation material only** — it is never consulted at request time and never appears in the prompt.

**Why the prompt carries a gloss and not an alias list.** An alias list gives the model a *lexical* prior, which makes confusable errors worse — tell it `insured_name` may appear as "Insured" and a document containing "Additional Insured" substring-matches. It also never closes the long tail, and since schema text is prompt text, every newly discovered alias would trigger a corpus rebuild and retrain (§7). A gloss encodes the **boundary** ("not the certificate holder, the producer, or an additional insured"), which an alias list structurally cannot, and it is written once.

**The discrimination risk this exists to manage.** Insurance documents are dense with name-like fields — certificate holder, producer/agency, additional insured, loss payee, mortgagee, carrier. A corpus that only ever teaches "name-ish label → `insured_name`" collapses them, producing confident, well-formed, wrong extractions. Confusable misattribution is therefore a **gating metric** (SPEC_08), not a nice-to-have.

---

## 2. Core architectural decisions (do not re-litigate — implement as stated)

| Decision | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-8B-Instruct`, pinned HF revision |
| Fine-tuning technique | **LoRA on a bf16 base** — bf16 frozen base weights, LoRA adapters in bf16. Both serving paths hold the base in bf16/fp16 (merged model per SPEC_10, vLLM LoRA hot-swap per SPEC_11), so training in bf16 means the adapter is applied to exactly the weights it trained against. **4-bit NF4 QLoRA remains a live flag** (`quantization.load_in_4bit`) for VRAM-constrained pods — arch §9.2 |
| Trainable components | **Projector + LLM decoder** via LoRA. **Vision Encoder (ViT) frozen by default** — escalated only via the eval gate (arch §3). **When the ViT is trained it gets a LoRA — never a full fine-tune** (arch §3). |
| Adapter strategy | **Hybrid**: one shared **Foundation LoRA** (rank 64, alpha 128) trained across all doc types + all 3 modality regimes, then small **per-type LoRA** adapters (rank 16, alpha 32) stacked on top. |
| Trainer stack | **Layer 3 ms-swift** (what you invoke) → **Layer 2 TRL `SFTTrainer`** (the real loop) → **Layer 1 PyTorch/Transformers/PEFT/bitsandbytes/Accelerate+DeepSpeed**. Locked, one option per layer (arch §10). |
| Trainer fallback | Dropping to TRL `SFTTrainer` directly at Layer 3 is a **contingency**, permitted only if ms-swift lacks a required Qwen3-VL capability at implementation time — not a parallel option. |
| Collator | ms-swift provides multimodal collation and `-100` masking; `training/data_collator.py` is an **override hook only** (arch §10). |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` in every decoder layer + vision-language projector/merger (justified per-module in arch §9a). |
| Attention impl | `flash_attention_2`. |
| Confidence | Per-token logprobs → per-field aggregation (default **min token prob** in span) → **post-hoc calibration** (temperature scaling default / isotonic). List fields also get a **row-completeness** signal. Review threshold ≈ 0.7, tuned. |
| Quantization | **GGUF**, user-selectable (`fp16\|bf16\|q8_0\|q6_k\|q5_k_m\|q4_k_m`). Routinely produce/validate only **fp16 baseline + one serving format**; **`q5_k_m` is the default serving target**; per-format quality thresholds enforced (arch §13b). |
| Artifact storage | **Azure Blob** — all weights/corpus/PDFs. Repo holds code only. |
| Compute | **RunPod** — ephemeral training pods + a persistent Serverless vLLM inference endpoint. Business logic lives outside RunPod. |
| Schema contract | Target JSON = Fideon SPEC_00 canonical schema; audit gate Fideon SPEC_07 Stage 3 validates every inference call (§1.1). |
| LoB detection | VLM outputs `line_of_business` per the Fideon SPEC_00 LOB enum; ≥20% corpus coverage per value; own gating metric (§1.3). |
| Tenant isolation | `tenant_id` path prefix reserved per Fideon SPEC_12; single-tenant for this build. No cross-tenant mixing in a corpus file (§8 below). |
| De-identification | **BLOCKED — do not implement.** Arch §8b requires Presidio de-identification of Foundation training data, but text-only de-identification corrupts the training signal. See §8 and SPEC_05. |
| MinerU | Version **pinned per corpus version**; inference must use the same version as the corpus (§7 below, arch §8a). |
| Prompt template | Versioned alongside the schema registry; **training-time and inference-time prompts must render identically**; a template change forces a corpus rebuild (arch §7). |

---

## 3. The `source_id` join key (critical convention)

Every source document has a stable `source_id` (e.g. `acord_0001`, `lossrun_0042`, `policy_0007`). This **same id** appears at every layer — raw PDF, OCR output, golden label, every compiled training row, and every extraction result. It is the traceability spine. Never generate data at any layer without carrying its `source_id`.

Naming: `{doc_type}_{zero_padded_index}`, e.g. `acord_0001`. ACORD form number is tracked as a separate field (`acord_form: "25" | "125" | "140"`), not baked into the doc_type.

Worked traceability chain (arch §7):

```
raw-documents/{tenant}/acord/acord_0001/original.pdf
processed/{tenant}/acord/acord_0001/page_1.png + page_1.md
golden-labels/{tenant}/acord/acord_0001/golden.json + label_metadata.json
corpus/{tenant}/v1/acord/train.jsonl  →  3 rows, all source_id: "acord_0001"
                                          (ocr_plus_image | noisy_ocr_image | image_only)
```

One source PDF yields **~3 compiled JSONL rows**, so N source documents produce on the order of 3N training examples.

---

## 4. Azure Blob layout (the data/artifact side — code never stores these locally)

```
azure-blob://insurance-extraction/
  # ---- tenant-partitioned (PII-bearing) ----
  raw-documents/{tenant_id}/{doc_type}/{source_id}/original.pdf, metadata.json   # immutable, tightest RBAC
  processed/{tenant_id}/{doc_type}/{source_id}/page_*.png, page_*.md, ocr_meta.json
  golden-labels/{tenant_id}/{doc_type}/{source_id}/golden.json, label_metadata.json
  corpus/{tenant_id}/v{n}/{doc_type}/{train,val,test}.jsonl
  corpus/{tenant_id}/v{n}/manifest.json

  # ---- shared / no tenant data ----
  base-models/qwen3-vl-8b-instruct/                                  # cached from HF, pinned revision
  adapters/foundation/v{n}/                                          # de-identified training data ONLY
  adapters/{doc_type}/v{n}/                                          # tagged w/ dependent foundation version
  merged-models/{doc_type|unified}/v{n}/                             # fp16/bf16
  quantized-models/{doc_type|unified}/v{n}/gguf/{format}/            # one subfolder per format
  registry/foundation/{run_id}/run_manifest.json
  registry/adapters/{doc_type}/{run_id}/run_manifest.json
  registry/registry_index.json
  calibration/{version}/{doc_type}.json
  eval-reports/v{n}/summary.json                                     # EvalReport.as_dict()
  eval-reports/v{n}/{doc_type}/report.json
  eval-reports/v{n}/gate_decision.json                              # the gate's verdict, NOT the report
  golden-eval-set/                                                   # frozen, versioned separately
```

**`gate_decision.json` is a separate key on purpose.** The promotion gate writes its verdict — pass/fail, per-metric deltas, failed gates — and the scored `EvalReport` writes `summary.json`. Sharing one key meant the gate overwrote the report it had just read, taking `by_doc_type` and every per-document error record with it; `vit_gate` then saw zero image-only and zero scanned documents and returned `insufficient_data` for ever. Two writers, two keys.

**Tenant partitioning rule (arch §8b, §18):** everything holding tenant document data is prefixed by `tenant_id`. Shared artifacts containing no tenant data (`base-models/`, `adapters/`, `registry/`) stay un-prefixed.

**Scope for the current build:** the **path shape is reserved now so no migration is needed later**, but the system runs **single-tenant**. `tenant_id` defaults from `DEFAULT_TENANT_ID` and is not a required argument on every CLI. The one rule that is live and enforced is **no cross-tenant mixing** — a corpus file must never contain rows from two tenants, because that is a training-data-composition rule, not plumbing. Per-tenant adapter lineages are **not built** until a broker actually requires one.

**Access-control split (arch §18a):** `raw-documents/` lives in a separate container (or at minimum a separate access policy) from everything else — it is the only layer holding unredacted PII. No training or serving component ever reads it directly.

---

## 5. Repository structure (code side — this is what the specs build)

```
insurance-extraction-finetuning/
├── README.md
├── pyproject.toml / requirements.txt
├── .env.example
├── configs/            base_model.yaml, training/*.yaml, sweeps/*.yaml (deferred),
│                       deepspeed/*.json, inference/vllm_serving.yaml                   → SPEC_01
├── schemas/            {acord25,acord125,acord140,lossrun,policy_doc}.schema.json,
│                       lob.enum.json, examples/                                        → SPEC_01
│                       aliases/{doc_type}.aliases.json   # NEVER in the prompt (§1.4)
│                       (all five schemas needed for form-level routing; only ONE
│                        shared `acord` ADAPTER is trained — arch §4b)
├── prompts/            system_prompt_template.jinja, doc_type_classifier_prompt.jinja  → SPEC_01
├── common/             config, schemas, prompts, ids, constants, lob, aliases,
│                       normalize                                                       → SPEC_01
├── artifact_registry/  blob_client.py, paths.py, push_to_blob.py, pull_from_blob.py    → SPEC_02
├── registry_utils/     models.py, write_run_manifest.py, query_registry.py             → SPEC_02
├── data_pipeline/
│   ├── ingestion/      pull_raw_pdfs.py                                                → SPEC_03
│   ├── ocr/            run_mineru.py, render_only.py, mineru_version.py                → SPEC_03
│   ├── labeling/       pre_annotate.py, export_golden_labels.py, review_tool/,
│   │                   active_learning.py                                              → SPEC_04
│   ├── deidentify/     (BLOCKED — see SPEC_05; do not implement yet)                   → SPEC_05
│   ├── dataset_builder/ build_jsonl.py, modality_dropout.py, noisy_ocr_augment.py,
│   │                   split_train_val_test.py                                         → SPEC_05
│   └── corpus_manifest.py                                                              → SPEC_05
├── training/           train_foundation.py, train_adapter.py, data_collator.py,
│                       vit_gate.py, callbacks/  (sweep.py deferred — see SPEC_06)      → SPEC_06
├── inference_core/     model_runner.py, input_builder.py, span_map.py, runner_config.py → SPEC_07 (shared primitive)
├── evaluation/         run_eval.py, metrics/*, gating.py                               → SPEC_08
├── calibration/        logprob_confidence.py, fit_calibration.py, apply_calibration.py,
│                       list_completeness.py, calibration_store/                        → SPEC_09
├── postprocessing/     merge_adapter.py, quantize.py
│                       (validate_quant.py deferred — see SPEC_10)                      → SPEC_10
├── serving/            vllm_entrypoint.py, doc_type_classifier.py, adapter_router.py,
│                       page_router.py, confidence_postprocess.py, pipeline.py          → SPEC_11
├── testing/            run_extraction.py, prompts/, (test_data, ocr_cache, results,
│                       metrics, extraction_registry.json)                              → SPEC_12
├── orchestration/      run.py (the 3+1 command surface), runpod_controller.py,
│                       pipeline_dag.py, config/                                        → SPEC_13
├── pilot/              zero_shot_baseline.py, smoke_test.py, pilot_report.py           → SPEC_15
└── tests/              test_*.py, fixtures/                                            → SPEC_14
```

**Key separation principle:** the repo never stores model weights, corpus data, or PDFs. Everything data/artifact-related is pulled from / pushed to Azure Blob at runtime via `artifact_registry/`.

---

## 6. Spec build order & dependency graph

Build in this order — later specs import from earlier ones. **SPEC_07 (inference core) is deliberately placed before evaluation/calibration/serving/testing because all four share it** — this breaks what would otherwise be a circular dependency between them.

```
SPEC_01  Project scaffold, configs, schemas, prompts, common   (no deps)
SPEC_02  Azure Blob I/O + training run registry                (deps: 01)
SPEC_03  Ingestion + MinerU OCR (+ version pinning)            (deps: 01, 02)
SPEC_04  Labeling + golden JSON + day-zero bootstrap gate      (deps: 01, 02, 03)
SPEC_05  Dataset builder (JSONL, modality, split)              (deps: 01, 02, 04)
SPEC_06  Training (ms-swift LoRA, foundation + adapters)       (deps: 01, 02, 05)
SPEC_07  Inference core (shared model-runner primitive)        (deps: 01, 02)
SPEC_08  Evaluation (metrics + gating)                         (deps: 01, 02, 06, 07)
SPEC_09  Confidence calibration                                (deps: 01, 02, 07, 08)
SPEC_10  Postprocessing (merge + GGUF quantize + thresholds)   (deps: 01, 02, 06)
SPEC_11  Serving (vLLM endpoint, classifier, routing)          (deps: 01, 02, 07, 09, 10)
SPEC_12  Testing / extraction routine (CLI harness)            (deps: 01, 02, 03, 07, 08, 09, 11)
SPEC_13  Orchestration (RunPod controller, pipeline DAG)       (deps: 01-12)
SPEC_14  Unit tests + CI fixtures                              (deps: 01-13)
SPEC_15  Pilot validation protocol (operational runbook)       (deps: 01-14; executed, not imported)
```

**Deferred until after SPEC_15 passes** (they are real work, just not first-cycle work):
- **Hyperparameter sweep execution** (arch §11a) — 9–12 runs on 25–30 docs/type mostly measures noise. `configs/sweeps/` is authored in SPEC_01; running it belongs *after* the pilot and *before* production-scale training.
- **Quantized-format validation** (arch §13b) — the primary serving path is vLLM on the merged fp16/bf16 model. Nothing is quantized in the first cycle, so nothing needs threshold validation yet.

**Note on cross-references that point "forward" but are not build-order violations:**
- `postprocessing/validate_quant.py` (SPEC_10) calls the testing routine (SPEC_12). `validate_quant` is a *gate you run after SPEC_12 exists*, not something merge/quantize needs. Build merge+quantize in SPEC_10; wire `validate_quant` once SPEC_12 is done.
- `evaluation/run_eval.py` (SPEC_08) can compute classifier accuracy only once the classifier (SPEC_11) exists. SPEC_08 runs *without* it (supply the true doc_type); classifier metrics fill in on a later pass.
- SPEC_15 is a **runbook**, not a library. It orders the pilot experiments (arch §16) and is executed against the built system; nothing imports it.

Everything else is strictly bottom-up.

---

## 7. Global conventions every spec must follow

- **Language/runtime:** Python 3.11+. Type hints throughout. `pydantic` v2 for config/data models.
- **Config:** all runtime config via YAML in `configs/` + environment variables (`.env`, loaded with `python-dotenv`). Never hardcode paths, model ids, or credentials.
- **Secrets:** Azure + RunPod credentials from env vars only (including `RUNPOD_VOLUME_ID` / `RUNPOD_VOLUME_MOUNT` for the staging volume, §12a). `.env.example` lists every required var with placeholders. Never commit real secrets.
- **Logging:** structured logging (`structlog` or stdlib `logging` with a JSON formatter). **Never log PII** (raw OCR text, extracted field values, image bytes) at INFO or above.
- **Blob access:** all Azure Blob reads/writes go through `artifact_registry/` (SPEC_02). No module opens the Azure SDK directly except that module.
- **Determinism:** set and log random seeds for any stochastic step (splitting, training, augmentation, generation). The seed is recorded in the run manifest.
- **CLIs:** every runnable module exposes an `argparse`/`typer` CLI with `--help`. Long-running scripts print progress.
- **Error handling:** fail loudly with clear messages on missing config, missing Blob artifacts, or schema-invalid data. Don't silently continue.
- **No network assumptions:** code may run inside a RunPod pod with a restricted egress allowlist; make external endpoints configurable, not hardcoded.
- **Idempotency and resumability (arch §13):** every pipeline stage is idempotent and resumable — a failed stage must not corrupt state and must be safe to simply re-run. Key on `source_id` + content checksum; never duplicate.
- **Tenant scoping:** Blob paths are constructed only via `artifact_registry/paths.py`, which reserves the `tenant_id` prefix. The build is **single-tenant**: `tenant_id` defaults from `DEFAULT_TENANT_ID` and is not a required CLI argument. The live rule is **no cross-tenant mixing in a corpus file** (a training-data-composition rule); the rest of the multi-tenant machinery is deferred until a second broker exists.
- **MinerU version pinning (arch §8a):** the MinerU version used to preprocess a corpus version is recorded in that corpus manifest and pinned. **Inference must use the same MinerU version as the training corpus.** A mismatch is treated as **distribution shift and a regression trigger**, not a routine dependency bump — an upgrade means reprocessing the affected documents and incrementing the corpus version *before* the next training cycle. This matters because the model is fine-tuned partly on *how MinerU formats its output* (table markdown conventions, reading order, error patterns).
- **Prompt/schema versioning (arch §7):** the system prompt template is versioned alongside the schema registry and both versions are recorded in the corpus manifest. **A change to either forces a corpus rebuild and a new training cycle.** This includes **field `description` text** — the semantic glosses of §1.4 are injected into the prompt with the schema, so editing a gloss is a schema change, not a documentation tweak. Settle the glosses before the first real corpus build. (The **alias registry** is not prompt material and is exempt: adding a newly observed surface form is a registry edit and triggers no rebuild.) Training-time and inference-time prompts must render identically — this divergence is the single most common cause of post-fine-tuning degradation and is invisible in training metrics because it only manifests at serving time.

## 8. PII, tenancy, and de-identification (non-negotiable, applies to every spec)

Insurance documents contain PII (names, TINs/SSNs, addresses, financials).

- `raw-documents/` is the most sensitive layer — separate container, tighter RBAC, encryption at rest, immutable, Cool/Archive tier once processed. Only `data_pipeline/ingestion/` writes it and only `data_pipeline/ocr/` reads it.
- **Cross-tenant mixing is prohibited.** Corpus, processed output, and golden labels are `tenant_id`-prefixed (§4). Assert that a corpus file never contains rows from two tenants.
- **De-identification is BLOCKED, not skipped** — see the blocker below and SPEC_05. Until it is resolved, PII protection rests on access control and tenancy, and that limitation must be stated to the compliance owner rather than left implicit.
- Never send PII to third-party APIs unless explicitly permitted (affects pre-annotation in SPEC_04 — default to the self-hosted base model).
- Never persist raw request/response bodies or OCR text in plaintext logs (affects SPEC_11 serving, SPEC_12 testing).
- Scrub/access-control eval reports and error dumps — a logged prompt or a failed extraction printed to a log can leak PII into low-security observability tooling.

> ### BLOCKER — Presidio de-identification must not be implemented as specified
>
> Arch §8b requires the Foundation LoRA to train only on Presidio de-identified data, but specifies de-identification of **text**, not of the **page images** the vision encoder reads. Implementing it that way is not merely an incomplete privacy control — it **corrupts the training signal**:
>
> - **`image_only` (30% of the Foundation corpus):** the image shows "John Smith", the target says `PERSON_1`. The target is **not derivable from the input**. That is an unlearnable example, and 30% of the corpus made of them is pure hallucination pressure.
> - **`ocr_plus_image` (50%):** OCR says `PERSON_1`, image says "John Smith", target says `PERSON_1` → teaches **trust-OCR-over-image**, the exact inverse of what the projector LoRA and the 20% `noisy_ocr_image` regime exist to teach (arch §3, §6).
>
> **Half-de-identifying is worse than either extreme.** Resolve before implementing — either redact page images consistently with the text, or de-identify nothing and rely on tenancy plus access control. This is a technical blocker on SPEC_05, not a compliance sign-off.

## 9. Data contracts (shared shapes used across specs)

**Training example (one JSONL row):**
```json
{
  "doc_type": "lossrun",
  "acord_form": null,
  "modality_mode": "ocr_plus_image | noisy_ocr_image | image_only",
  "source_id": "lossrun_0042",
  "tenant_id": "broker_a",
  "split": "train | val | test",
  "deidentified": true,
  "messages": [
    {"role": "system", "content": "<prompt: schema + modality instructions>"},
    {"role": "user", "content": [
      {"type": "image", "image": "<page 1 image>"},
      {"type": "text",  "text": "<page 1 of N>

<page 1 MinerU markdown>"},
      {"type": "image", "image": "<page 2 image>"},
      {"type": "text",  "text": "<page 2 of N>

<page 2 MinerU markdown>"}
    ]},
    {"role": "assistant", "content": "<target JSON string, incl. line_of_business>"}
  ]
}
```
`doc_type`, `acord_form`, `modality_mode`, `source_id`, `split` are the required top-level keys from arch §7; `tenant_id` and `deidentified` are added by the tenancy and Presidio rules (arch §8b).

**The user turn is interleaved: each page's image is immediately followed by that page's own markdown, prefixed `<page N of M>`.** Three reasons, and all three were live defects before it was:

1. **Correspondence.** With all images first and one joined text blob, nothing said where page 1's text ended. The evidence rule — *trust the image where the two disagree* — requires knowing which image, and on a 40-page policy the model cannot recover that by content matching.
2. **Markdown integrity.** A blank line terminates a Markdown table, so joining pages split every table crossing a page boundary in two, the second half without its header. Loss Runs and long policies are exactly where row completeness matters most.
3. **One shape, not two.** A page-routed request (arch §7) is now the same structure with fewer pairs — `[img_1][txt_1][img_9][txt_9]` — instead of N separate single-page calls asking for a whole-document JSON from one page, a shape no training row ever had.

`image_only` carries the marker and no markdown: the guarantee is *no OCR text*, not *no text at all*, and without it a routed image-only request cannot tell the model it holds pages 9 and 14 of 20 rather than a two-page document. A gap in the numbering is itself the signal that absent fields live on pages the model was not given.

**Loss is computed only on the assistant tokens** — system, image, and OCR-text tokens are masked with label `-100`.

**Extraction output (per document, at inference/testing):**
```json
{
  "source_id": "abcLossRun",
  "doc_type": "lossrun",
  "model_version": "v2",
  "mode": "ocr_plus_image",
  "schema_valid": true,
  "overall_confidence": 0.88,
  "line_of_business": {"value": "workers_comp", "confidence": 0.92},
  "fields": {
    "<field>": {"value": "...", "confidence": 0.94}
  },
  "list_fields": {
    "<list_field>": {"rows": [...], "row_completeness_confidence": 0.7}
  },
  "pages_used": [1, 4, 5],
  "review_flags": ["valuation_date:low_confidence", "claims:row_count_mismatch"]
}
```

**Run manifest (registry, per training run):** see SPEC_02 for the full schema.

## 10. Modality mix for training data (used by SPEC_05)

Within the Foundation corpus, per source document generate 3 rows (arch §6):

| Regime | Share | Purpose |
|---|---|---|
| `ocr_plus_image` — image + real MinerU OCR | ~50% | Baseline dual-modality behavior |
| `noisy_ocr_image` — image + deliberately imperfect OCR, **correct** golden target | ~20% | Teaches image-over-OCR arbitration |
| `image_only` — image, no OCR block at all | ~30% | Satisfies the must-work-without-OCR requirement |

The system prompt **explicitly declares which mode is active** rather than silently omitting the OCR block — that explicit signal is what lets the model switch behavior reliably instead of guessing which mode it is in.

**Split at `source_id` level BEFORE modality expansion** to prevent leakage. Split ratio scales with per-type volume (arch §8):

| Data volume per doc type | Split |
|---|---|
| Pilot batch (~25–30/type) | ~70 / 18 / 12 — metrics are directional, not final |
| 200–1000/type | 75/15/15 or 80/10/10 |
| 1000+/type (target state) | 80/10/10 |

---

## 12. The operator command surface (SPEC_13)

A cycle is run through **three commands plus one umbrella command**, all from `orchestration/run.py`. Anyone running the pipeline needs the version tag and nothing else — no paths, no pod management, no stage wiring.

| # | Command | Arch §13 stages | Ends with |
|---|---|---|---|
| **1** | `finetune` | 1 → 7 (ingest, OCR, dataset build, train, evaluate, **gate**, merge) | Adapters + merged model on the **RunPod staging volume** |
| **2** | `package` | 8 → 9 (quantize, push) | Adapters + merged + quantized model in **Azure Blob** |
| **3** | `extract` | §17 extraction routine | JSON + confidence + metrics, locally |
| **—** | `all` | `finetune` then `package` | Same as command 2 |

```bash
python -m orchestration.run finetune --input ./intake --out-version v2 --gpu a100-80
python -m orchestration.run package  --version v2 --formats fp16 q5_k_m
python -m orchestration.run extract  --model base|v1|v2 --input testing/test_data/
python -m orchestration.run all      --input ./intake --out-version v2 --formats fp16 q5_k_m
```

**`all` never includes `extract`.** Extraction is a separate concern from building a model — it runs against any chosen version, including models trained weeks earlier and the untuned base.

**Three facts this surface has to respect, and does:**

1. **Labeling is human work inside the span of command 1.** `finetune` ingests and OCRs everything, builds the corpus from only the source_ids that have a validated `golden.json`, and **reports the unlabeled backlog**. It aborts before training only when the labeled set is below `--min-labels-per-type` (default 25, matching SPEC_04's day-zero rule).
2. **The gate is a hard stop inside command 1.** A failed evaluation stops `finetune` *before* merge and exits non-zero. `all` therefore never reaches `package` on a failed gate.
3. **Command 1 fans out into 1 + N GPU jobs** — Foundation first, then one per active doc type, sequentially, because a per-type adapter cannot start before the Foundation it depends on has trained and passed evaluation (arch §12).

### 12a. Staging volume vs. registry — do not conflate these

| | **Staging volume** | **Registry** |
|---|---|---|
| Where | RunPod network volume at `/runpod-volume/staging/` | Azure Blob `registry/` |
| Holds | Working copies of adapters, merged model, eval reports between commands 1 and 2 | Durable `run_manifest.json` per run |
| Lifetime | Cleared after a verified push | Permanent |
| Guarantee | **None** — working storage | Lineage of record (arch §12) |

RunPod pods are ephemeral (arch §14), so persistence between commands 1 and 2 comes from a network volume attached to every launched pod. Staging exists because the merged model is ~16 GB and quantization also runs on RunPod — pushing it to Azure at the end of command 1 and pulling it back at the start of command 2 is a 32 GB round trip for no benefit.

**But a volume is not an artifact of record.** So command 1 **always writes its run manifest to Blob** with `artifacts.status: "staged"`, even though the weights stay on the volume; command 2 flips it to `"published"` with real Blob paths. Without that, a reclaimed volume means a training run that happened and left no trace.

---

## 11. How to use these specs with Claude Code

1. Start a Claude Code session at the repo root.
2. Run `SPEC_01` first to scaffold the project.
3. Run each subsequent spec **in numeric order (01 → 14)**. Each spec is self-contained but assumes earlier specs are done. The order is dependency-correct — no spec references code from a higher-numbered spec as a build prerequisite (the two "forward" mentions in SPEC_08 and SPEC_10 are explicitly non-blocking; see §6).
4. Each spec ends with an **Acceptance checklist** — verify it before moving on.
5. **SPEC_15 is executed, not built** — run it after SPEC_14 as the pilot validation protocol before committing to full corpus annotation.
6. This master file is the context; when a spec says "per master context", it means this file.
