# SPEC 02 — Azure Blob I/O + Training Run Registry

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01. This is the ONLY module that talks to the Azure SDK directly. Everything else goes through it.
>
> **Architecture refs:** `finetuning-architecture-v1.md` §8b (tenant isolation per Fideon SPEC_12), §12 (versioning + training run registry), §13b (quantization thresholds recorded), §15 (metrics recorded), §18/§18a (Blob layout, raw-document access control).

## Goal

Provide a clean, typed interface for all Azure Blob reads/writes, and implement the training run registry that records **every** Foundation, per-type, per-tenant, and **sweep** training run.

## Deliverables

### 1. `artifact_registry/paths.py` (build this first)
- The **single place** the Blob layout (master §4) is expressed. Every other function builds paths through it.
- **Reserves the tenant prefix** (arch §8b): `raw-documents/`, `processed/`, `golden-labels/`, and `corpus/` are `{tenant_id}`-prefixed; `base-models/`, `adapters/`, `registry/`, `golden-eval-set/` are never prefixed. `tenant_id` **defaults from `DEFAULT_TENANT_ID`** — the build is single-tenant and the prefix exists so no migration is needed later.
- Distinguishes the two adapter lineages: `adapters/foundation/v{n}/` and `adapters/{doc_type}/v{n}/`. A per-tenant lineage is **not built** until a broker requires one.
- **Also expresses the RunPod staging paths** (master §12a) — `/runpod-volume/staging/{adapters,merged-models,eval-reports,run_manifests}/` — deliberately mirroring the Blob layout so `package` (SPEC_13 command 2) copies rather than translates. Staging paths and Blob paths are built by separate functions that must never be interchanged: one is working storage, the other is the artifact of record.

### 2. `artifact_registry/blob_client.py`
- Thin wrapper around `azure-storage-blob`, initialized from env (`AZURE_STORAGE_CONNECTION_STRING`, `AZURE_BLOB_CONTAINER`).
- Methods: `upload_file`, `download_file`, `upload_dir`, `download_dir`, `exists`, `list(prefix)`, `read_json`, `write_json`.
- Retries with backoff; clear errors on auth/missing-container.
- **Separate-container support (arch §18a):** an optional `container` override so `raw-documents/` points at the separately-permissioned `AZURE_RAW_CONTAINER`. Reading `raw-documents/` from a training or serving context must be refused, not merely discouraged — expose it only to the ingestion/OCR callers.
- **Immutability guard:** writing to an existing `raw-documents/.../original.pdf` raises. Raw documents are write-once (arch §18a).

### 3. `artifact_registry/transfer.py`
*(One module, not the `push_to_blob.py` / `pull_from_blob.py` pair originally sketched: a push and its pull must agree on the path, and two files are two places for them to drift.)*
- High-level, path-aware helpers keyed to the Blob layout. Tenant-scoped helpers accept an optional `tenant_id` that defaults from env:
  - `push_adapter(local_dir, kind, doc_type, version)` → foundation / per-type path
  - `pull_adapter(kind, doc_type, version, local_dir)`
  - `push_merged_model(...)`, `push_quantized(local_dir, doc_type, version, fmt)`
  - `pull_base_model(local_dir)` (from `base-models/`, or from HF at the pinned revision if absent, then cache to Blob)
  - `push_corpus_version(...)`, `pull_corpus_version(version, ..., tenant_id=None)`
  - `push_calibration(version, doc_type, params)` / `pull_calibration(...)` (SPEC_09 store)
  - `push_eval_report(...)`, `push_golden_eval_set` / `pull_golden_eval_set`
- CLI so a RunPod pod can `python -m artifact_registry.pull_from_blob --corpus v3 --dest ./data`.

### 4. `registry_utils/models.py`
Pydantic model `RunManifest` capturing the arch §12 manifest **plus the v1 additions**:

- `run_id`, `run_type` (`foundation` | `per_type_adapter`), `doc_type` (nullable), `tenant_id` (nullable, defaulted), `status` (`trained|evaluated|promoted|archived|failed`), `created_at`.
- **`is_sweep_run`** (bool) + `sweep_id` / `sweep_phase` — reserved fields so sweep runs are first-class registry entries when sweeps are run (arch §11a; deferred past the SPEC_15 pilot).
- `dependencies`:
  - `base_model` (`qwen3-vl-8b-instruct@<hf_revision_pin>`)
  - `foundation_version` (nullable — which Foundation this adapter sits on)
  - `corpus_version`
  - `code_git_commit`
  - **`mineru_version`** (arch §8a)
  - **`schema_version`** and **`prompt_template_version`** (arch §7)
- `training_config`: technique, lora_rank/alpha/dropout, bias, base quantization (`nf4`, double-quant, bf16 compute), lr, lr_scheduler, warmup_ratio, epochs, optimizer (`adamw_paged_8bit`), betas, eps, weight_decay, max_grad_norm, per_device_batch, grad_accum, effective_batch_size, gradient_checkpointing, mixed_precision, target_modules, **`vit_trainable`** (and, when true, that it is **LoRA-on-ViT, never full fine-tune** — arch §3), resolution_cap_px, max_seq_len, **`seed`**.
- `data_stats`: train/val/test counts, `modality_mix`, **`lob_coverage`** (per-LoB-value share, arch §0b), `tenant_ids` contributing, `deidentified: true|false` (recorded; the enforcement rule is **suspended while de-identification is blocked** — see SPEC_00 §8 and SPEC_05).
- `eval_metrics` (all against the frozen golden eval set, arch §15):
  `field_exact_match`, `field_normalized_match`, `field_f1_list_fields`, `list_field_recall`, `schema_validity_rate`, `ece_confidence`, `ocr_arbitration_accuracy`, `image_only_accuracy`, `scanned_accuracy`, `doc_type_classifier_accuracy`, **`lob_detection_accuracy`** (overall **and per LoB value**), `latency_ms_per_doc`.
- `artifacts`: **`status`** (`staged` | `published`), **`staging_path`** (set while `staged`), `adapter_weights`, `merged_model`, `quantized_model`, `quantized_formats[]`, `eval_report`, `calibration_params`.
  - `finetune` (SPEC_13 command 1) writes the manifest with `status: "staged"` and weights still on the RunPod volume; `package` (command 2) flips it to `"published"` and fills the Blob paths. A manifest is written **either way**, so a reclaimed staging volume never means an untracked training run.
- `promotion`: `gated_against`, `beat_previous_on_all_gates`, `promoted_by`, `promoted_at`.

### 5. `registry_utils/write_run_manifest.py`
- `write_manifest(manifest: RunManifest)` → writes `registry/{foundation|adapters/{doc_type}}/{run_id}/run_manifest.json` to Blob, and updates `registry/registry_index.json` (a flat table: run_id, type, status, key metrics, created_at, is_sweep_run).
- `capture_git_commit()` to fill `code_git_commit`; `capture_mineru_version()` reads the pin from the corpus manifest (SPEC_05).
- `data_stats.deidentified` is **recorded but not enforced** while de-identification is blocked (SPEC_00 §8). Reinstate the refusal here — it is the right enforcement point — once the image-redaction question is resolved.
- MLflow/W&B logging hook (guarded by env; no-op if unset) — the tracker gives the queryable UI, the Blob manifest is the durable record that travels with the artifacts (arch §12).

### 6. `registry_utils/query_registry.py`
CLI + functions:
- `get(run_id)` → RunManifest.
- `list_runs(run_type=None, doc_type=None, status=None, include_sweeps=False)`.
- `adapters_depending_on(foundation_version)` → adapter run_ids — this makes the arch §12 dependency-upgrade rule **a query, not a manual audit**: when Foundation moves to v3.0, this is the exact list needing re-validation.
- `latest_promoted(kind, doc_type)` → the version currently serving.
- `resolve_model_version(tag)` → given a user tag, return the concrete artifact paths. Used by SPEC_07, SPEC_11, SPEC_12, SPEC_13.
  - **`"base"` is a valid tag** and resolves to the pinned HF base model with **no adapter** — the zero-shot path used by `extract --model base` (SPEC_13), the pilot baseline (SPEC_15), and day-zero pre-annotation (SPEC_04).
  - `"v2"` resolves Foundation + the routed per-type adapter, or the merged/quantized model.
  - Resolves from Blob for `published` versions and from the staging volume for `staged` ones, so `extract` works on a model that has not been packaged yet.
- `diff_manifests(run_a, run_b)` → field-by-field delta. This is the arch §12 regression-debugging workflow: when a doc type regresses, diff the new manifest against the last-good one and see immediately whether the corpus, foundation version, MinerU version, or a hyperparameter moved.

## Constraints
- Only this package imports `azure-storage-blob`.
- All Blob paths built via `paths.py` — the layout lives in one place and the tenant prefix is not optional.
- `raw-documents/` is write-once and reachable only from ingestion/OCR.
- No PII in logs.
- Every training/eval/quantize job writes a manifest — **no silent runs, sweeps included**.

## Promotion and the index

- **`mark_promoted` requires the affirmative gate result**, not merely the
  absence of a failure. `promotion.beat_previous_on_all_gates` defaults to
  `None` and only the gate ever sets it, so rejecting just `False` let a run that
  was **never evaluated** promote cleanly — and the index could not tell it from
  one that actually passed.
- **A Foundation run owns its adapter and nothing else.** In a per-type build the
  merged and quantized models are produced per doc type, and the `doc_type=None`
  "unified" paths exist only under `--foundation-only`; publishing them anyway
  pointed the manifest at artifacts that were never built.
- **`registry_index.json` is a shared read-modify-write.** Concurrent runs each
  read N rows and write N+1, so the later write drops the earlier run's row — and
  the index is exactly what `adapters_depending_on` and `latest_promoted` read,
  so a lost row makes a run invisible to the queries the registry exists for. The
  write re-reads and re-checks; a store with conditional writes is the real fix
  before jobs run in parallel.
- **Version tags match exactly.** A substring match makes `v1` match `v10`, so
  `resolve_model_version("v1")` can read v10's artifact status and return a
  staging path for a model published in Blob.

## Acceptance checklist
- [ ] `blob_client` round-trips a file and a directory against a real/emulated container.
- [ ] Overwriting an existing `original.pdf` raises (write-once guard).
- [ ] `RunManifest` validates the arch §12 example (extended with the v1 fields) and rejects malformed input.
- [ ] `write_run_manifest` creates the manifest and updates `registry_index.json`.
- [ ] `adapters_depending_on("foundation-v2.0")` returns the right list from seeded manifests.
- [ ] `resolve_model_version("v2")` returns concrete adapter/merged paths; `resolve_model_version("base")` returns the pinned base model with no adapter.
- [ ] A `staged` manifest resolves to staging-volume paths; a `published` one resolves to Blob paths.
- [ ] `diff_manifests` surfaces a changed `corpus_version` / `mineru_version` / hyperparameter.
- [ ] Unit tests mock the Blob client (no live Azure needed for CI).
