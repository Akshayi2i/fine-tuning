# SPEC 06 — Training (ms-swift LoRA: ONE Unified Adapter)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_05.
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §3 (ViT escalation gate — **LoRA, never full FT**), §4.1 (**one unified adapter**), §4.2 (graduation gate for per-type adapters), §6.1 (per-epoch modality sampling), §8b (tenancy per Fideon SPEC_12; de-identification per Fideon SPEC_11 — blocked, see SPEC_05), **§9.2 (bf16 base vs 4-bit)**, §9.3 (memory settings), §9.4 / §9a (frozen ViT and mergers; LoRA targets), §10 (**locked three-layer trainer stack**), §11 (hyperparameters), §11.2 (checkpoint selection), §11a (**sweep methodology**), §12 (versioning + run registry).

## Goal

Fine-tune Qwen3-VL-8B-Instruct with **LoRA on a bf16 base** via the locked trainer stack. **One adapter**, across every document type and every task, with the ViT *and* the mergers frozen, and every run — sweeps included — recorded in the registry.

Two reversals from v1, both forced by how vLLM serves:

* **One adapter, not a Foundation plus per-type stack.** vLLM applies one LoRA per request, so the two could never both be active — the v1 topology was unservable, not merely awkward. It also did not fit the data: at 25–30 documents per type a rank-16 adapter memorises its own training set.
* **The mergers are not a LoRA target** (`freeze_aligner: true`). Tower and connector LoRA in vLLM is experimental with known mixed-adapter batching risks, and image-vs-OCR arbitration is learned in the **decoder**, where image tokens and OCR text tokens attend to each other. The mergers only map visual features and never see OCR text at all.

## The locked stack (arch §10) — not a choice to re-make

```
Layer 3   ms-swift                    ← what you invoke (CLI / config)
              │ wraps and configures ↓
Layer 2   swift.trainers.Seq2SeqTrainer   ← the real loop, on the HF Trainer
              │ (optimizer, backprop, grad accumulation, checkpointing,
              │  eval hooks, early stopping, DeepSpeed/Accelerate)
              │ built on ↓
Layer 1   PyTorch + Transformers + PEFT + Accelerate (bitsandbytes only under load_in_4bit)
```

TRL's `SFTTrainer` is **not** in this stack: ms-swift builds its own `Seq2SeqTrainer` on the HF Trainer, and TRL is a dependency for its RLHF trainers only. `training/train.py` is a **thin wrapper that assembles the ms-swift config and launches it**, not a custom training loop.

**Fallback clause:** migrating to TRL `SFTTrainer` is permitted **only** if ms-swift lacks a specific Qwen3-VL capability needed at implementation time. It is a framework migration, not a layer swap — the template, collator and masking must be re-implemented against the §10.2 parity tests. Document it if invoked.

## Deliverables

### 1. `training/train.py` — the unified extractor run

Replaces v1's `train_foundation.py` and `train_adapter.py`. One entrypoint, one run type (`unified`), run id `extractor-v{n}`.

- **Config:** `configs/training/unified.yaml` + `configs/base_model.yaml`. There are no per-type training configs.
- **Base precision** from `training/base_precision.py`. Default is a **bf16 frozen base** (`quantization_bit: 0`, passed explicitly rather than omitted, and every `bnb_4bit_*` argument suppressed so the run log never carries settings describing nothing the run did). `load_in_4bit: true` in `configs/base_model.yaml` switches to NF4 with double quantization and bf16 compute. The manifest's `technique` (`LoRA`/`QLoRA`) and `base_quantization` are derived from the same config, never defaulted.
- **LoRA:** rank 64 / alpha 128, dropout **0.10** at pilot scale (0.05 from ≥200 docs per type), `bias: none`. Targets are the attention and MLP projections of the **language-model decoder only** — `q/k/v/o_proj`, `gate/up/down_proj`. No `merger`.
- **Frozen:** `freeze_vit: true` (unless `--train-vit`, the §3 escalation — LoRA-on-ViT, never full fine-tune) and `freeze_aligner: true` always.
- **Optimisation (§11.1):** AdamW (`adamw_torch`), LR **1e-4**, cosine, **`warmup_steps: 10`** (steps, not a ratio — a ratio rounds to zero warmup at pilot volume), weight decay **0.0** (decay on a low-rank adapter is a shrinkage prior on the update, not regularisation), max grad norm 1.0, per-device batch 1 × accumulation 8 = **effective batch 8**, gradient checkpointing, bf16.
- **Memory (§9.3):** `use_logits_to_keep`, `padding_free`, `length_grouped_sampling`. `max_length` is the largest task cap **including `by_doc_type` overrides** (policy extraction is 32,768). If a cap does not fit one card, **sequence parallelism comes first**; DeepSpeed is off by default, because under LoRA ZeRO-2 shards ~1.5 GB of adapter optimizer state and is close to a no-op.
- **Datasets.** **Epochs are files, and ms-swift runs one pass over them** (§6.1). `dataset` is the first `num_train_epochs` of `corpus/{tenant}/v{n}/train/epoch_{1..4}.jsonl`, and ms-swift is given **`num_train_epochs: 1`**. Each file already holds every train document once in that epoch's modality draw, so the concatenation *is* the N-epoch run; also telling ms-swift N looped the N files N times — nine passes for a "3 epoch" run. The manifest records the logical count. Fewer epoch files than epochs is refused. `val_dataset` is `corpus/{tenant}/v{n}/val/val.jsonl`, passed separately (see below). The corpus is read for `--tenant`, never silently from the default tenant. (ms-swift shuffles across the concatenation, so epoch-2 rows can precede epoch-1 rows; each document is still seen exactly N times, each time in its drawn regime.)
- **Evaluation during training** is validation **loss**, for early stopping only (`eval_steps`/`save_steps` 50, `save_total_limit: 4`, `metric_for_best_model: eval_loss`, `greater_is_better: false`, patience 3). **Loss never selects what ships** — see §11.2 below.
- **Staging:** the adapter and its `checkpoint-*` directories go to the RunPod staging volume under `staging/adapters/foundation/v{n}/` (the slot name is historical; the manifest's `run_type` says `unified`). Pushed to Blob by `package`, or immediately with `finetune --push-adapters`.
- **Manifest:** a `RunManifest` (SPEC_02) is written at status **`training`** before launch; `launch_and_record` flips it to `trained` when ms-swift returns, or **`failed`** when it raises, so the registry never claims weights a crashed run never wrote. It records the full config, data stats, LoB coverage, seed, git commit, corpus version, MinerU version, schema/prompt template versions, and the de-identification flag (recorded, not asserted, while de-identification is blocked — SPEC_05 §1).
- **Every value in the YAML's `evaluation:` block reaches ms-swift.** `metric_for_best_model`, `greater_is_better` and `load_best_model_at_end` are passed into `swift_early_stopping_args`, which is unpacked **first** so explicit keys win.

Flags: `--corpus vN`, `--out-version vN`, `--tenant`, `--deepspeed zero2|zero3`, `--train-vit`, `--continue-from <checkpoint dir>`, `--dry-run`.

**Versioning rule (arch §12) — enforced in code:**
- **Major corpus expansion** → retrain **from the original HF base model** on the full accumulated corpus. Continued training on an existing LoRA compounds drift across cycles.
- **Minor incremental patch** → continuing from the current checkpoint is acceptable, **but promotion requires cross-type regression evidence** against the frozen golden eval set (SPEC_08). `--continue-from` sets `continued_from` on the manifest, which the gate reads.
- **`--continue-from` takes a checkpoint DIRECTORY, never a registry run-id.** It reaches ms-swift as `resume_from_checkpoint`, which reads a path; given a run-id, ms-swift finds nothing, trains from base, and the manifest records a lineage that never happened. `assert_checkpoint_path` refuses the run-id shapes this repo generates (`extractor-v2`, `extractor-v2.1`, `foundation-v…`, `{doc_type}-adapter-v…`).

### 2. `training/merge.py`

`plan_merge` / `merge` fold **one** adapter — the checkpoint §11.2 selected, recorded on the plan — into the bf16 base with PEFT `merge_and_unload()`, on the staging volume. See SPEC_10 §1.

### 3. Checkpoint selection — `evaluation/checkpoint_eval.py` (arch §11.2)

Runs as the `checkpoint_eval` stage, between training and merge.

- **Candidates:** the last 3 checkpoints by step, plus the best-loss one. After a real run, `discover_checkpoints` reads them from ms-swift's versioned output directory (`…/v{n}/v0-<timestamp>/checkpoint-*`, the most recent run), and the best-loss checkpoint from `trainer_state.json`.
- **Scoring:** each candidate is applied as a **decoder LoRA** on the bf16 base in vLLM and generates over `val/val.jsonl` through `evaluation/validation_generation.py` — the row's own prompt with the golden answer removed, constrained as serving is — and is scored by the gate's own `build_report`. The winner has the best `field_normalized_match`; ties go to the later step. A candidate whose scoring raises is skipped and recorded, never scored zero.
- **Why not loss:** averaged over every token, it is dominated by schema keys and JSON punctuation the model gets right in the first hundred steps. A checkpoint can improve on loss while getting worse at the values.

### 4. Per-type adapters — only through the §4.2 graduation gate

There is **no per-type training entrypoint** in the default topology. A per-type adapter returns only when a type has **≥500 labeled documents AND a measured >2 pp win outside the paired-bootstrap 95% CI**. Even then it trains on the **merged** foundation, decoder-only, and is applied at serving time as the request's one LoRA — never stacked, never merged. When the unified run moves to a new major version, every graduated adapter must be re-validated; `query_registry.adapters_depending_on()` (SPEC_02) produces the work list. **ACORD** stays one type with per-form schemas (arch §4b); per-form adapters are not considered until a form has 1000+ examples.

### 5. `training/data_collator.py`

**Override hook only.** ms-swift provides multimodal collation and `-100` label masking (system, image and OCR tokens masked; **loss only on the assistant JSON**). This file:
- documents that the framework default is in use;
- contains `assert_masking_correct` / `verify_batch`, which verify masking on a sample batch — the highest-value correctness check in the system, because broken masking trains the model to reproduce its own prompt and is invisible in loss curves;
- implements custom masking only if a genuine need arises (`custom_collator` raises until then).

### 6. `training/callbacks/early_stopping.py`

`swift_early_stopping_args` emits `early_stopping_patience` (3 in `unified.yaml`) with `metric_for_best_model: eval_loss`, `greater_is_better: false`. Early stopping stops a run that has stopped improving; it does not pick the checkpoint that ships (§3 above).

### 7. `training/vit_gate.py` (decision helper — arch §3)

Given an eval report (SPEC_08) for a frozen-ViT run, decide whether the ViT escalation fires:

```
1. Train with the ViT frozen.
2. Evaluate on (a) the image-only eval subset and (b) the scanned-PDF eval subset.
3. IF image-only OR scanned accuracy is below target
   AND the errors are perception errors (misread characters, missed checkboxes)
   rather than schema/reasoning errors:
      → apply LoRA to the ViT and retrain.
4. ELSE keep it frozen.
```

- The **error-type distinction is the load-bearing part**: a character read correctly but placed in the wrong field is a decoder problem, and touching the ViT won't help. `classify_error` separates perception from schema/reasoning errors, and the gate reads that — not the accuracy number alone.
- **When the ViT is trained, LoRA is applied — never full fine-tuning** (arch §3). `--train-vit` configures a ViT LoRA; no code path enables full ViT fine-tuning.
- Emits a recommendation and rationale. **Does not auto-train.**

### 8. `training/sweep.py` — built; **execution waits for production volume**

The protocol is implemented and tested against a stub trainer. What waits is *running* it: a 9–12 run sweep against 25–30 documents per type mostly measures noise. `assert_enough_data` refuses below **200 documents per type**. It is deliberately **not a pipeline stage** — nine to twelve training runs is an operator decision.

| Phase | Sweeps (the unified run — `foundation` grid key) | Held fixed | Metric | Budget |
|---|---|---|---|---|
| **1 — Learning rate** (first, highest impact) | `{5e-5, 1e-4, 2e-4}`, 1 epoch each | everything else | validation loss | 3 runs |
| **2 — Epochs** | `{2, 3, 4}` document passes, patience 2 | Phase 1 winner | validation field F1 | 3 runs |
| **3 — LoRA rank** (**only if F1 plateaus**; off by default) | `{32, 64, 128}`, rsLoRA on | Phases 1–2 winners | validation field F1 | 3 runs |

- The four materialized epoch files exist so the epoch phase compares the same data draws rather than regenerating them.
- The `per_type` grids in `configs/sweeps/` are dormant until a type graduates (§4 above).
- **Every sweep run writes a full `run_manifest.json`** with `is_sweep_run: true`. A candidate that produced no score is excluded from ranking, not defaulted.

## Constraints
- ms-swift is the entrypoint (locked). TRL `SFTTrainer` is a documented migration fallback only.
- **One adapter.** No code path trains a per-type adapter outside the §4.2 graduation gate, and none stacks two LoRAs.
- **ViT and mergers frozen** unless `--train-vit` — and then LoRA-on-ViT, never full fine-tune. The mergers are never a LoRA target.
- Every run writes a manifest — **no silent training, sweeps included**.
- Runs record (not yet assert) de-identification status; every run asserts its corpus is single-tenant.
- Vision budgets and sequence caps come from `configs/shared/` and must match data prep and serving (the §10.2 parity tests assert this).
- Seed is fixed, logged, and recorded in the manifest.

## Datasets, and the split that must stay split

The trainer receives **`--dataset` (the epoch files) and `--val_dataset` (val) as separate arguments**. Passing both to `--dataset` makes ms-swift treat the validation set as training data and carve its own eval split out of the union, so early stopping reads documents the model memorised — and nothing downstream can detect it.

**Argument rendering is part of this contract.** Two rendering rules are load-bearing:

- A `False` value renders as `--flag false`, never as an omitted flag. Dropping it silently disabled every option whose correct value is `False` — `freeze_vit=False` meant `--train-vit` never reached the trainer while the manifest recorded `vit_trainable: true`.
- A list renders as one argv element **per item**. Joining them gives `HfArgumentParser`'s `nargs="+"` one token containing every value, so the epoch files become one nonexistent filename and `lora_target_modules` matches no module.

## Acceptance checklist
- [ ] The unified run trains end-to-end on a tiny sample corpus and produces a loadable LoRA adapter (GPU milestone).
- [ ] `dataset` is the first `num_train_epochs` epoch files, ms-swift gets `num_train_epochs: 1`, and fewer files than epochs is refused.
- [ ] Every file training reads exists after a real dataset build (`tests/test_orchestration.py`).
- [ ] Train and val are passed as **separate** datasets; no path puts the validation split into `--dataset`.
- [ ] Under the default config the rendered CLI carries `quantization_bit 0` and no `bnb_4bit_*` argument; `load_in_4bit: true` switches to NF4, and the manifest's `technique` follows.
- [ ] Target modules contain no `merger`; `freeze_aligner` is always true; `--train-vit` renders `--freeze_vit false` and records a ViT **LoRA**.
- [ ] `max_length` covers the largest task cap including `by_doc_type` overrides.
- [ ] The assistant span **includes** the end-of-turn token, so EOS is supervised.
- [ ] **Label masking verified**: loss only on assistant tokens; a mutated masking implementation fails the test.
- [ ] `early_stopping_patience` is emitted, not merely configured.
- [ ] `--continue-from` refuses a run-id and marks the manifest so the gate demands cross-type regression evidence.
- [ ] Checkpoints are discovered from the versioned output directory after training, and selection scores them with the gate's metric, not loss.
- [ ] Adapter lands on the staging volume; the RunManifest lands in **Blob** with `artifacts.status: "staged"`, LoB coverage, and seed recorded.
- [ ] `vit_gate` fires only when errors are perception errors, never on schema/reasoning errors alone.
- [ ] `sweep.py` refuses below production volume, writes one manifest per candidate with `is_sweep_run: true`, conditions each phase on the previous winner, excludes unscored candidates, and leaves phase 3 off by default.
