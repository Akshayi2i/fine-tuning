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
