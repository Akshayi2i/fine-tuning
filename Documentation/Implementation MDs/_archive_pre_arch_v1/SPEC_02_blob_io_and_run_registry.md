# SPEC 02 — Azure Blob I/O + Training Run Registry

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01. This is the ONLY module that talks to the Azure SDK directly. Everything else goes through it.

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
  - `resolve_model_version(tag)` → given a user tag like `v2`, return the concrete artifact paths (foundation + per-type adapter, or merged/quantized) — used by serving (SPEC_10) and testing (SPEC_11).

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
