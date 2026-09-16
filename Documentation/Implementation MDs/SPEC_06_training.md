# SPEC 06 — Training (ms-swift LoRA: ONE Unified Adapter)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_05.
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §3 (ViT escalation gate — **LoRA, never full FT**), §4 (adapter strategy), §8b (tenancy per Fideon SPEC_12; de-identification per Fideon SPEC_11 — blocked, see SPEC_05), **§9.2 (bf16 base vs 4-bit — the decision and its open measurement)**, §9a (LoRA targets), §10 (**locked three-layer trainer stack**), §11 (full hyperparameter spec), §11a (**sweep methodology**), §12 (versioning + run registry).

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
Layer 1   PyTorch + Transformers + PEFT + bitsandbytes + Accelerate/DeepSpeed
```

`SFTTrainer` is **not skipped** — it sits underneath ms-swift. `train_foundation.py` and `train_adapter.py` are **thin wrappers that assemble the ms-swift config and launch it**, not custom training loops.

**Fallback clause:** dropping to ms-swift's `Seq2SeqTrainer` directly at Layer 3 is permitted **only** if ms-swift lacks a specific Qwen3-VL capability needed at implementation time. That is a contingency, not a parallel option — document it if invoked.

## Deliverables

> **Epochs are files, and ms-swift runs one pass over them** (arch v2.1 §6.1). `training/train.py` passes the first `num_train_epochs` of `corpus/{tenant}/v{n}/train/epoch_{1..4}.jsonl` as `dataset` and sets ms-swift's **`num_train_epochs: 1`**. Each file already holds every train document once in that epoch's modality draw, so the concatenation *is* the N-epoch run; also telling ms-swift N looped the N files N times — nine passes for a "3 epoch" run. The manifest records the logical count. Fewer epoch files than epochs is refused. The corpus is read for `--tenant`, never silently from the default tenant.

### 1. `training/train_foundation.py`
Entrypoint that:
- Pulls the base model and the corpus version from Blob (SPEC_02).
- Reads the corpus manifest's `deidentified` flag and records it. **The hard assertion is suspended while de-identification is blocked** (SPEC_00 §8, SPEC_05 §1); reinstate it here and in SPEC_02 once the image-redaction question is resolved.
- Loads `configs/training/foundation.yaml` + `configs/base_model.yaml`.
- Configures **base precision from `training/base_precision.py`** — one helper shared by both trainers, so a Foundation and the adapters stacked on it cannot hold the base differently. Default is a **bf16 frozen base** (`quantization_bit: 0`, passed explicitly rather than omitted, and the `bnb_4bit_*` arguments suppressed entirely so the run log never carries settings describing nothing the run did). Setting `load_in_4bit: true` switches to 4-bit NF4 with double quantization and bf16 compute. Then bf16 LoRA adapters (rank 64 / alpha 128 / dropout 0.05, `bias: none`), target modules = attention + MLP projections **+ the vision-language projector**, **ViT frozen** (`train_vit: false`), `flash_attention_2`.
- Applies the full arch §11 parameter set — AdamW paged 8-bit, cosine schedule with warmup, max grad norm 1.0, gradient checkpointing, effective batch via accumulation, resolution cap and `max_seq_len` from config.
- Launches **ms-swift** SFT (which runs ms-swift's `Seq2SeqTrainer` underneath) with DeepSpeed (ZeRO-2 default, ZeRO-3 when VRAM-constrained).
- Trains across **ALL doc types + ALL 3 modality regimes** — the mixed corpus is what makes the Foundation learn shared behavior (insurance terminology, table/checkbox reading, OCR-vs-image arbitration, JSON structural discipline).
- Saves the adapter to the **RunPod staging volume** at `/runpod-volume/staging/adapters/foundation/v{n}/` (master §12a). It is pushed to Blob later by `package` (SPEC_13 command 2), or immediately when `finetune --push-adapters` is set.
- Writes a `RunManifest` (SPEC_02) with full config, data stats, LoB coverage, seed, git commit, corpus version, MinerU version, schema/prompt template versions — at status **`training`**, because nothing has run yet. `launch_and_record` then flips it to `trained` when ms-swift returns, or **`failed`** when it raises, so the registry never claims weights a crashed run never wrote.
- Every value in the YAML's `evaluation:` block reaches ms-swift: `metric_for_best_model`, **`greater_is_better`** and `load_best_model_at_end` are passed into `swift_early_stopping_args`, not hardcoded beside it. `greater_is_better` was a literal `True` while all four YAMLs carried the key, so a config selecting on `eval_loss` would have restored the checkpoint with the **highest** loss — and that worst-of-run adapter is what gets staged, evaluated and offered to the gate. Where the helper is unpacked matters as much as what it returns: it goes **first**, so explicit keys win.

Flags: `--corpus vN`, `--out-version vN`, `--deepspeed zero2|zero3`, `--train-vit` (default false — the gated exception), `--from-base|--continue-from vN`, `--sweep-id`, `--sweep-phase`.

**Versioning rule (arch §12) — enforced in code, not just documented:**
- **Major corpus expansion** → retrain **from the original HF base model** on the full accumulated corpus. Continued training on top of an existing LoRA compounds drift across cycles; starting fresh is slower per-cycle but far more reproducible and debuggable.
- **Minor incremental patch** → continuing from the current Foundation checkpoint is acceptable, **but promotion requires a regression check against the frozen golden eval set for the *other* document types** (SPEC_08). `--continue-from` sets a manifest flag that the gate reads and refuses to promote without that regression evidence.
- **`--continue-from` takes a checkpoint DIRECTORY, never a registry run-id.** It reaches ms-swift as `resume_from_checkpoint`, which reads a path on disk; a run-id looks close enough to be passed by mistake and fails silently in the worst way — ms-swift finds no checkpoint, trains from base, and the manifest records a `continued_from` lineage that never happened, which the gate then reads as evidence. `assert_checkpoint_path` refuses the run-id shape, **including dotted versions (`foundation-v2.1`) and the `{doc_type}-adapter-v{n}` form `train_adapter` generates** — a guard that only matched undotted single-lineage ids missed every id this codebase actually produces.
- The Foundation is attached to a per-type run with ms-swift's **`adapters`** argument, **not `resume_from_checkpoint`**. Resume means *continue this run*: it restores optimizer state and the completed `global_step`, so a fresh 3-epoch adapter run resumes at the end of the Foundation's schedule and trains zero steps — and tries to load rank-64 weights into a rank-16 config on the way.

### 2. `training/train_adapter.py`
Per-doc-type (and per-tenant) adapter entrypoint:
- Loads the **current promoted Foundation** as the frozen base and trains a small LoRA (rank 16 / alpha 32 / dropout 0.05) on that doc type's slice of the corpus, at the per-type hyperparameters (LR 5e-5–1e-4, epochs 3–5, effective batch 16–32).
- **Always trains fresh from the Foundation — never continues from a previous adapter checkpoint** (arch §12). Per-type adapters are cheap and small, so there is no cost advantage to incremental training, and fresh training from a fixed, well-evaluated Foundation avoids stacking errors across cycles. This also gives every adapter version a clean dependency on exactly one Foundation version — critical when debugging "why did this document type regress".
- Records the `foundation_version` dependency in the manifest.
- Writes to the staging volume at `/runpod-volume/staging/adapters/{doc_type}/v{n}/`, pushed to Blob by `package`. A **per-tenant adapter lineage** (arch §8b) is **not built** — it is a real capability, but no broker requires one yet; add it when one does, following the same Foundation-dependency rules.
- **ACORD granularity (arch §4b):** **one `acord` adapter + per-form schemas.** With limited data per specific form, one adapter learning shared "ACORD-ness" generalizes better than several data-starved per-form adapters. Split to per-form adapters only once a form has 1000+ examples **and** evaluation shows the unified adapter underperforming. Per-form adapter configs are **not created** until then (SPEC_01); the per-form **schemas** do exist, because the classifier must still select the right schema.

Flags: `--doc-type acord|policy|lossrun`, `--acord-form 25|125|140` (optional, for the future split case), `--foundation vN`, `--corpus vN`, `--out-version vN`.

**Dependency-upgrade rule (arch §12):** a GRADUATED per-type adapter (§4.2) trains on the merged foundation weights, so when the unified run moves to a new major version every dependent adapter must be re-validated and likely retrained before it becomes production. Dormant until a type graduates — the default topology produces none. Treat it like a dependency upgrade, not an automatic cascade — `query_registry.adapters_depending_on()` (SPEC_02) produces the exact work list.

### 3. `training/data_collator.py`
**Override hook only.** ms-swift provides correct multimodal collation and `-100` label masking (system, image, and OCR tokens masked; **loss computed only on the assistant JSON**) by default — you don't hand-write it (arch §10). This file:
- Documents that the framework default is in use.
- Contains an **assertion/test that verifies masking is correct on a sample batch** — this is the single highest-value correctness check in the system, because broken masking trains the model to reproduce its own prompt and is invisible in loss curves.
- Implements custom masking only if a genuine need arises.

### 4. `training/callbacks/early_stopping.py`
- Early stopping on **validation loss + field-level F1, patience 2 evaluations** (arch §11). Wired into the ms-swift/Trainer callback system. `metric_for_best_model: field_f1`; best checkpoint retained.

### 5. `training/vit_gate.py` (decision helper — arch §3)
Given an eval report (SPEC_08) for a frozen-ViT Foundation, decide whether the ViT escalation fires:

```
1. Train Foundation with ViT frozen.
2. Evaluate specifically on: (a) the image-only eval subset, (b) the scanned-PDF eval subset.
3. IF image-only OR scanned accuracy is below target
   AND the errors are perception errors (misread characters, missed checkboxes)
   rather than schema/reasoning errors:
      → apply LoRA to the ViT and retrain Foundation.
4. ELSE keep it frozen (cheaper, faster, already sufficient).
```

- The **error-type distinction is the load-bearing part**: if the model reads a character correctly but puts it in the wrong JSON field, that is an LLM/projector problem and touching the ViT won't help. Implement a classifier over the eval error records that separates *perception* errors from *schema/reasoning* errors, and gate on that — not on the accuracy number alone.
- **When the ViT is trained, LoRA is applied — full fine-tuning of the vision encoder is never used** (arch §3). Full FT risks degrading the encoder's pretrained document/OCR capabilities, which are exactly what the image-only pathway depends on. The escalation is "add a ViT LoRA," not "unfreeze and train the encoder." `--train-vit` must therefore configure a ViT LoRA; assert that no code path enables full ViT fine-tuning.
- Emits a recommendation + rationale. **Does not auto-train** — it surfaces the decision.

### 6. `training/sweep.py` — built; **execution waits for production volume**

The protocol is implemented and tested against a stub trainer. What waits is
*running* it: a 9–12 run sweep against 25–30 documents per type mostly measures
noise — arch §8 calls pilot metrics "directional", and arch §11a scopes the sweep
to "before the first **production** run", not before the pilot.

That constraint is now **enforced in code rather than stated in a doc**:
`assert_enough_data` refuses below 200 documents per type. A sweep that ran at
pilot volume would produce a config that looks justified, carries a manifest
saying it won, and gets promoted as the production training config — all true,
and none of it meaning the hyperparameter was better.

It is deliberately **not a pipeline stage**. Nine to twelve training runs is an
operator decision, not something a build starts because it reached that point.

The bounded 3-phase design, so hyperparameter search doesn't become an unbounded one:

| Phase | Sweeps | Held fixed | Metric | Budget |
|---|---|---|---|---|
| **1 — Learning rate** (run first, highest impact) | Foundation `{5e-5, 1e-4, 2e-4}`; per-type `{2e-5, 5e-5, 1e-4}`; 1 epoch each | everything else | validation loss | 3 runs / adapter type |
| **2 — Epochs** | Foundation `{2, 3, 4}`; per-type `{3, 4, 5}`, early stopping patience 2 | best LR from Phase 1 | validation **field-level F1** | 3 runs / adapter type |
| **3 — LoRA rank** (**only if F1 plateaus** — secondary, not run by default) | Foundation rank `{32, 64, 128}` | best LR + epochs | validation field-level F1 | 3 runs |

- Managed via **Weights & Biases Sweeps or MLflow** hyperparameter tracking.
- **Every sweep run produces a full `run_manifest.json`** (SPEC_02) with `is_sweep_run: true` — sweep runs are first-class registry entries, not untracked side experiments. The manifest fields are reserved in SPEC_02 now so nothing needs changing when sweeps are turned on.
- The best configuration by validation field-level F1 is **promoted as the production training run** config.
- Total budget: **~9–12 training runs before the first production run**.

## Constraints
- ms-swift is the entrypoint (locked). The raw TRL path is a documented fallback only.
- **ViT frozen unless `--train-vit` is explicitly set — and then LoRA-on-ViT, never full fine-tune.**
- Every run writes a manifest — **no silent training, sweeps included**.
- Foundation runs record (not yet assert) de-identification status; every run asserts its corpus is single-tenant.
- Resolution cap and `max_seq_len` come from config and must match data prep and production inference.
- Seed is fixed, logged, and recorded in the manifest.

## Datasets, and the split that must stay split

The trainer receives **`--dataset` (train) and `--val_dataset` (val) as separate
arguments**. Passing both splits to `--dataset` makes ms-swift treat the
validation set as training data and then carve its own eval split out of the
union, so `metric_for_best_model` selects on documents the model memorised — and
the promotion gate reads that number. Nothing downstream can detect it: the loss
curve looks healthy and every metric improves.

**Argument rendering is part of this contract.** The ms-swift CLI is built from
a recorded config, and two rendering rules are load-bearing:

- A `False` value renders as `--flag false`, never as an omitted flag. Dropping
  it silently disabled every option whose correct value is `False` —
  `freeze_vit=False` meant `--train-vit` never reached the trainer while the run
  manifest recorded `vit_trainable: true`, claiming a ViT escalation that never
  happened.
- A list renders as one argv element **per item**. Joining them into a single
  space-separated string gives `HfArgumentParser`'s `nargs="+"` a one-element
  list containing every value as one token, so the corpus paths become one
  nonexistent filename and `lora_target_modules` matches no module at all.

## Acceptance checklist
- [ ] Foundation training runs end-to-end on a tiny sample corpus and produces a loadable LoRA adapter.
- [ ] Train and val are passed as **separate** datasets; no path puts the validation split into `--dataset`.
- [ ] The rendered ms-swift CLI carries `--freeze_vit false` when `--train-vit` is set, and one argv element per list item.
- [ ] The assistant span **includes** the end-of-turn token, so EOS is supervised — an unsupervised EOS means generation runs to `max_new_tokens`, which the runner reports as truncation and the row-completeness signal reports as dropped rows.
- [ ] Early stopping reaches the trainer: `early_stopping_patience` is emitted, not merely configured.
- [ ] **Label masking verified**: loss computed only on assistant tokens; a mutated masking implementation fails the test.
- [ ] A graduated per-type adapter (§4.2) trains on the MERGED foundation weights and records its `foundation_version` dependency. Never stacked on an unmerged adapter: vLLM applies one LoRA per request.
- [ ] `--continue-from` marks the manifest so the promotion gate demands cross-type regression evidence.
- [ ] Adapter lands on the staging volume; the RunManifest lands in **Blob** with `artifacts.status: "staged"`, LoB coverage, and seed recorded.
- [ ] `--train-vit` configures a **ViT LoRA**; no code path enables full ViT fine-tuning.
- [ ] `vit_gate` returns a correct fire/no-fire recommendation on seeded eval inputs, and does **not** fire when errors are schema/reasoning rather than perception.
- [ ] `sweep.py` runs Phase 1 on a stub trainer, writes one manifest per candidate with `is_sweep_run: true`, and ranks results.
- [ ] Phase 2 is conditioned on Phase 1's winner, and optimises **field F1**, not validation loss — loss can fall while extraction gets worse.
- [ ] A candidate that produced no score is **excluded from ranking**, not defaulted: a crashed run must not win a minimise-loss phase.
- [ ] The sweep **refuses to run** below production volume, and phase 3 is off by default.
