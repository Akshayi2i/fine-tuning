# SPEC 13 — Orchestration (Command Surface + RunPod Controller + Pipeline DAG)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: all prior specs (01–12).
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §13 (**end-to-end pipeline, 13 stages, idempotency, the gate**), §13c (**operator command surface**), §14 (**RunPod: ephemeral training pods vs. persistent serving endpoint**), §12 (every cycle recorded), §17 (extraction routine), §11a (sweep orchestration — deferred).

## Goal

Expose the whole lifecycle through **three operator commands plus one umbrella command**, so a cycle is run by a person who knows the version tag and nothing else about paths, pods, or stage wiring.

## 1. The command surface

Single entrypoint, `orchestration/run.py`, with four subcommands:

| # | Command | Covers arch §13 stages | Ends with |
|---|---|---|---|
| **1** | `finetune` | 1 → 7 (ingest, OCR, dataset build, train, evaluate, **gate**, merge) | Adapters + merged model on the **RunPod staging volume** |
| **2** | `package` | 8 → 11 (quantize, calibrate, **gate**, bundle) | A promoted **release bundle** in Azure Blob |
| **3** | `extract` | §17 extraction routine | Extracted JSON + confidence + metrics, locally |
| **—** | `all` | `finetune` then `package` | Same as command 2 |

**`all` never includes `extract`.** Extraction is a separate concern — you run it against a chosen model version whenever you want, including against models trained weeks earlier and against the untuned base. Folding it into the build would conflate "produce a model" with "use a model."

```bash
# 1 — ingest through merge. Artifacts staged on RunPod, not yet in Blob.
python -m orchestration.run finetune \
       --input ./intake \
       --doc-types acord policy lossrun \
       --corpus-version v2 \
       --out-version v2 \
       --gpu a100-80

# 2 — quantize, calibrate, gate, then publish the release bundle.
python -m orchestration.run package \
       --version v2 \
       --release-id release-2026.11.1 \
       --formats bf16 fp8

# 3 — extraction, model chosen by the operator.
python -m orchestration.run extract --model base --input testing/test_data/
python -m orchestration.run extract --model v2   --input testing/test_data/ \
       --mode ocr_plus_image --ground-truth testing/golden/

# all — commands 1 and 2 back to back. Extraction excluded by design.
python -m orchestration.run all \
       --input ./intake --out-version v2 --release-id release-2026.11.1 --formats bf16 fp8
```

Each subcommand is a thin argument-parsing shell over `pipeline_dag.py` stage functions (§4). No orchestration logic lives in the CLI layer.

## 2. `finetune` — command 1 (stages 1 → 7)

**`finetune` produces artifacts; `package` judges and publishes them.** The gate moved into `package` under arch v2.1 §13, because it scores the merged model in each serving format — and neither the merge nor the formats exist until `finetune` has finished.

Runs, in order: **ingest → OCR → labeling → dataset build → train (ONE unified adapter) → select checkpoint → merge.**

**The human-labeling precondition.** Stage 3 (labeling) is human work and cannot be inside an automated command. `finetune` therefore:

1. Ingests and OCRs **everything** in `--input`.
2. Builds the corpus from **only the source_ids that have a validated `golden.json`** (SPEC_04).
3. **Reports the unlabeled backlog** — a count and a `source_id` list written to the run report — and continues with what is labeled.
4. Aborts before training only if the labeled set is empty, or below `--min-labels-per-type` (default 25, matching the day-zero rule in SPEC_04).

This is the honest shape: you ingest 500 documents, 200 are labeled, you train on 200, and the command tells you 300 are waiting for a reviewer.

**No training fan-out.** One `finetune` run launches **one** training job. v1 launched 1 + N sequentially — a Foundation, then one adapter per document type stacked on it — and that topology is unservable: vLLM applies one LoRA per request, so the two could never both be active (arch v2.1 §4.1). A per-type adapter returns only through the §4.2 graduation gate, trained on the merged foundation and never stacked.

**Where the gate's numbers come from.** `finetune` reads the candidate's scores from the frozen-eval-set report (`eval-reports/{version}/summary.json`, SPEC_08) and the baseline from the currently promoted version's own report. Neither is optional and neither is fabricated: with no report the command fails rather than gating on nothing, and with no promoted version the baseline is genuinely `None` — a first version — rather than a lookup that quietly failed.

**The gate is a hard stop inside `package`.** It runs after merge, quantize and calibrate (arch v2.1 §13), so a regression stops the **release**, not the build: merging still happens, nothing is packaged, and the staged artifacts sit on a volume that will be reclaimed. It exits non-zero with the per-metric verdicts — floor, interval and basis, not a bare delta.

There is no `--force`. There **is** an override (§15.5), requiring a named person and a written reason, both recorded in the gate decision and the release bundle. v1 had none at all, on the reasoning that a waivable gate is a suggestion — which was right about the risk and wrong about the remedy, because the v1 gate demanded improvement on twelve metrics within 0.001 and could not be passed at pilot volume at all.

**Flags:** `--input`, `--doc-types`, `--corpus-version`, `--out-version`, `--gpu`, `--commit <sha>`, `--from-stage <name>` (resume), `--skip-ingest`, `--min-labels-per-type`, `--push-adapters` (see §3), `--tenant` (optional, defaults from env). `--foundation-only` is gone: one unified adapter is the only topology.

## 3. The RunPod staging volume

> **Terminology, to prevent a real bug.** In this spec set "**registry**" means the durable run-manifest registry in Azure Blob (SPEC_02). The RunPod side is the "**staging volume**" and is never called a registry. They have different lifetimes and different guarantees.

RunPod **pods are ephemeral** — a pod clones the repo, works, pushes, and terminates (arch §14). Persistence between commands 1 and 2 therefore comes from a **RunPod network volume**, mounted at `/runpod-volume` and attached to every pod the controller launches.

```
$RUNPOD_VOLUME_MOUNT/staging/            # default /runpod-volume; /workspace when mounted there
  adapters/foundation/v{n}/              # unified run: ms-swift output, checkpoint-* inside
  adapters/scope/{scope}/v{n}/           # a scoped run
  merged-models/{unified|scope/{scope}}/v{n}/
  quantized-models/{unified|scope/{scope}}/v{n}/vllm/{format}/
  train-data/{scope}/v{n}/               # the run's staged ms-swift JSONL
  train-images/{tenant}/{corpus}/        # page images, one cache per corpus version
  eval-reports/v{n}/
```

The layout deliberately mirrors the Blob layout (master §4) so `package` copies rather than translates.

**Why stage instead of pushing straight to Blob:** the merged model is ~16 GB. Quantization also runs on RunPod. Pushing the merged model to Azure at the end of command 1 and pulling it back at the start of command 2 costs a 32 GB round trip for no benefit.

**Durability rule — the one thing command 1 always sends to Blob.** A network volume is working storage, not an artifact of record. If command 2 never runs, or the volume is reclaimed, an untracked training run has happened. So `finetune` **always writes the run manifest to Blob** (SPEC_02) even though the weights stay staged, with `artifacts.status: "staged"` and `artifacts.staging_path` set. `package` later flips it to `"published"` and fills the real Blob paths. The manifest is a few KB; the lineage guarantee in arch §12 is worth that.

**Optional belt-and-braces:** `finetune --push-adapters` also pushes the LoRA adapters (tens of MB) to Blob immediately, leaving only the merged model staged. Off by default, per the staging design; recommended if a cycle's commands 1 and 2 may be separated by more than a day.

**Config:** `RUNPOD_VOLUME_ID` and `RUNPOD_VOLUME_MOUNT` (default `/runpod-volume`) in `.env`. The controller attaches the volume to every pod it launches; a launch without it is refused rather than silently writing to pod-local disk.

## 4. `package` — command 2 (stages 8 → 11)

Runs: **quantize → push adapters + merged model + quantized model(s) to Azure Blob → update the run manifest.**

- Reads `--version v{n}` from the staging volume. If that version is not staged, **fail loudly** naming the expected path and the remediation (re-run `finetune --from-stage merge`, or `--from-blob` if adapters were pushed with `--push-adapters`).
- Quantizes per `--formats` (default: **bf16 alone** — it is the merged model; FP8 only once the spike has verified it, per SPEC_10). GGUF, and its `mmproj`, is an on-request edge export, never a `package` output.
- Pushes **all three artifact classes** to their respective Blob locations (master §4):

| Artifact | Blob destination |
|---|---|
| Foundation + per-type LoRA adapters | `adapters/foundation/v{n}/`, `adapters/{doc_type}/v{n}/` |
| Merged fp16/bf16 model | `merged-models/{doc_type\|unified}/v{n}/` |
| Quantized GGUF, one subfolder per format | `quantized-models/{doc_type\|unified}/v{n}/gguf/{format}/` |
| Eval report | `eval-reports/v{n}/` |

- Updates the run manifest: `artifacts.status` → `"published"`, real Blob paths, `quantized_formats[]`.
- `--keep-staging` retains the volume copy (default: clear it after a verified push, so the volume doesn't fill).
- Quantization threshold validation (SPEC_10 / arch §13b) is **deferred this cycle** — the serving path is merged fp16/bf16 via vLLM. When it ships it becomes a gate inside `package`, between quantize and push.

**Flags:** `--version`, **`--release-id release-YYYY.M.N`** (required), `--formats`, `--skip-quantize` (push adapters + merged only), `--keep-staging`, `--from-blob`, `--dtype`, **`--foundation-only`**, **`--doc-types`**, **`--tenant`**.

**`--release-id` is checked before any stage runs** — under `all`, before training. Calibrators, gate decisions and the bundle are all addressed by it, and an invalid one used to surface only when calibrate tried to save what it had already fitted. It is named by the operator, never derived: a derived id would move between a failed run and its `--from-stage` resume and split one release across two ids. A missing or malformed id fails with the next free id for the month as the suggested fix; pass the same id when resuming.

**Where the package stages get their inputs.** Nothing is hand-supplied in a real cycle:

- **Checkpoints** — after training, `discover_checkpoints` reads the `checkpoint-*` directories ms-swift wrote under the staged adapter directory (the most recent `v*-<timestamp>/` run), and the best-loss one from `trainer_state.json`. Selection scores each as a decoder LoRA over `val/val.jsonl` through `evaluation/validation_generation.py`, whose metrics are the gate's own `build_report`.
- **Calibration samples** — when none are supplied, calibrate generates the validation split **with each staged serving format**, constrained as serving is, and labels every field's features correct or not against the golden label, split by the `val_half` the dataset build stamped. There is no shared or wildcard sample set.
- **Gate decision** — written under `releases/{tenant}/{release_id}/gate/bf16/`. The candidate metrics are the merged bf16 model's, so that is bf16's gate run and no other format's.

**The release bundle** (`releases/{tenant}/{release_id}/bundle.json`, plus a row in `release_index.json`) pins the adapter run, merged and per-format model paths, per-format calibrators and gate decisions, a hash of the prompt templates, schema versions, the OCR pin, and hashes of `configs/shared/vision.yaml`, `configs/inference/vllm_serving.yaml` and the dependency pins. Its status is **`promoted` only when every serving format has its own calibrator and its own gate run**; otherwise it is written as **`gated`** with the missing pieces listed in the stage result. Until a per-format gate run exists for quantized formats, an FP8 release is `gated`. The dependency hash reads `pyproject.toml` because the repo has no lockfile yet — declared ranges, not resolved versions.

The last three must match the `finetune` run that produced the version. `finetune` decides *which* models get built; `package` publishes their locations, so without them a standalone `package` fell back to `foundation_only=False` and all three doc types however finetune had actually run — writing three adapter prefixes and three merged-model prefixes into Blob for artifacts that were never built. An empty Blob prefix later reads as a published model. (Under `all` they come from the finetune flag set; adding them twice is an argparse conflict.)

**`stage_push` publishes only runs that finished.** Manifests are selected by version *and status*: a run left at `training` or marked `failed` has no weights, and flipping it to `published` advertises a Blob path that serving will fetch and find empty.

## 5. `extract` — command 3 (arch §17)

A wrapper over `testing/run_extraction.py` (SPEC_12), which itself wraps the serving pipeline (SPEC_11) so test == prod.

**Model selection is the point of this command.** `--model` accepts:

| Value | Resolves to |
|---|---|
| `base` | Untuned `Qwen/Qwen3-VL-8B-Instruct` at the pinned revision, **no adapter** |
| `v1`, `v2`, … | That version's merged model, plus a graduated per-type adapter where the routed type has one |
| `v2 --format fp8` | That version's FP8 serving weights |

`--model base` is not a curiosity — it is the **zero-shot baseline** of the pilot protocol (SPEC_15) and the day-zero pre-annotation path (SPEC_04). `pilot/zero_shot_baseline.py` is a thin wrapper over this command rather than a parallel implementation.

**Flags:** `--model`, `--input <file|dir>`, `--mode ocr_plus_image|image_only`, `--ground-truth <dir>`, `--format`, `--limit`, `--tenant`.

Outputs are unchanged from SPEC_12: `results/{version}/`, `metrics/{version}/`, `extraction_registry.json`.

## 6. `all` — commands 1 + 2

Runs `finetune`, then `package`, sharing `--out-version`. Halts if the gate fails; `package` is never reached on a failed gate. Accepts the union of both flag sets. **Does not run `extract`.**

## 7. `orchestration/runpod_controller.py`

**Training — ephemeral pods (arch §14):**
- **Deployment (2026-09-28): one RunPod H200 SXM (141 GB) pod runs every GPU stage in process** — OCR, training, checkpoint selection, merge, calibration and the gate share the card, each vLLM engine released before the next loads. Classes per stage stay configurable (`orchestration/config/pipeline.yaml`) for a future per-stage launcher; the original plan put OCR on a cheaper L4/L40S card.
- **Stages 2, 4 and 5 all want a GPU**, so `finetune` provisions **one pod for the whole command** rather than shuttling a corpus between machines. That is cheaper than three pod launches and removes two Blob round trips.
- **Training code lives in the private git repo; the pod clones it fresh at job start** at a pinned commit — code is never permanently resident on a pod.
- Every launched pod has the staging volume attached (§3).
- Pod job: pull base model + corpus from Blob → run → write to the staging volume → **terminate**.
- Poll status, stream logs (**PII-scrubbed**, master §8), handle failures and retries.

**Serving — persistent endpoint (arch §14):**
- Manage the **RunPod Serverless vLLM endpoint** (SPEC_11): deploy/update to a promoted version, health-check, roll back.

**Direct CLI** (the lower-level surface `run.py` calls; still available for single steps):
`train-foundation`, `train-adapter`, `evaluate`, `merge`, `quantize`, `push`, `deploy-endpoint`, `rollback-endpoint`.

## 8. `orchestration/pipeline_dag.py`

The 13 stages (arch v2.1 §13) as reusable, individually addressable stage functions — this is the implementation `run.py` composes, and the same functions an Airflow DAG or GitHub Actions workflow can schedule.

| # | Stage | Command | Writes to |
|---|---|---|---|
| 1 | **Ingestion** — checksum-deduped, immutable | `finetune` | `raw-documents/` (Blob) |
| 2 | **Preprocessing (GPU)** — MinerU OCR + page rendering at the resolution cap | `finetune` | `processed/` (Blob) |
| 3 | **Labeling** — human review | *(outside the CLI)* | `golden-labels/` (Blob) |
| 4 | **Dataset build** — compile JSONL, inject schema, modality split, train/val/test split | `finetune` | `corpus/v{n}/` (Blob) |
| 5 | **Training** — ONE unified LoRA on a bf16 base | `finetune` | staging volume |
| 6 | **Checkpoint eval** — vLLM generation on validation, selected by field F1 | `finetune` | staging volume |
| 7 | **Merge** — PEFT `merge_and_unload()`, one adapter | `finetune` | staging volume |
| 8 | **Quantize** — bf16 reference + FP8, vLLM-native | `package` | staging volume |
| 9 | **Calibrate** — per-field-type calibrators and risk-controlled thresholds, per serving format | `package` | release bundle |
| 10 | **Evaluation & gate** — frozen golden eval set, per serving format | `package` | gate decision |
| 11 | **Package** — release bundle pinning adapter, prompt, schema, calibrators, OCR pin and serving config | `package` | Blob |
| 12 | **Serving** — endpoint pulls the promoted release bundle | `deploy-endpoint` | endpoint |
| 13 | **Feedback loop** — low-confidence output, human corrections, and a random 5% of auto-accepted documents → labeling queue | `extract` + SPEC_04 | next corpus version |

**Pipeline properties that are requirements, not aspirations (arch §13):**
- Every stage reads from and writes to Azure Blob or the staging volume — never pod-local disk.
- Every training, evaluation, and quantization job produces a **run manifest** (SPEC_02).
- **Stages are idempotent and resumable.** `--from-stage` re-enters mid-pipeline; a completed stage re-run is a no-op, not a duplicate.
  - **Ingestion is the exception, and always runs.** There is no cheap correct completion check for it — comparing an input directory against ingested checksums costs the same as ingesting — and the approximate one that used to be here ("has anything ever been ingested?") was true from cycle two onward, so a second `finetune --input ./new_batch` skipped the stage and built the corpus from the previous batch. `ingest_directory` is checksum-deduped, so re-running it writes nothing and reports duplicates. An approximate guard on cheap idempotent work buys nothing and can be wrong in the direction that loses documents.
- **The gate (stage 10) is a hard stop.** Promotion requires every gating metric to pass. There is no `--force-deploy`; the only exception is the recorded written override (§2). `package` reads the recorded gate decision and refuses a version whose gate did not pass — running `package` after a blocked gate used to publish it.

## 9. `orchestration/config/`
- GPU class per stage, corpus/version tags, staging volume id and mount, schedule, notification hooks, retry policy.
- **Sweep orchestration (arch §11a) — deferred with SPEC_06 `sweep.py`:** when enabled, the 3-phase sweep runs as a scheduled job sequence (~9–12 runs before the first **production** run), each writing a manifest, winner promoted to the production training config. Not part of the first cycle.
- **Foundation-upgrade cascade (arch §12):** when a new unified major version passes its gate, generate the re-validation work list from `query_registry.adapters_depending_on()` (SPEC_02) and schedule retraining of every dependent GRADUATED adapter (§4.2) **before** it becomes production. Dormant until a type graduates — the default topology produces no dependents at all. The DAG produces the work list and waits for it to pass rather than silently promoting.

## Constraints
- Four subcommands, one entrypoint. `all` = `finetune` + `package`, never `extract`.
- Training pods **ephemeral**; the staging volume is the only thing that persists between them, and it is not an artifact of record.
- `finetune` always writes its run manifest to Blob even when weights stay staged.
- Orchestration runs on cheap CPU infra **outside** RunPod; only GPU-bound work runs on RunPod (arch §14).
- Secrets from env only. Logs PII-scrubbed before leaving the pod.
- No flag around the evaluation gate; only a recorded, attributed override (§2).

## Acceptance checklist
- [ ] `run finetune` executes stages 1→7 on a tiny fixture set and leaves adapters + merged model on the staging volume.
- [ ] `finetune` reports the unlabeled backlog and trains on the labeled subset; it aborts when labels are below `--min-labels-per-type`.
- [ ] `finetune` launches **one** training job (one unified adapter, or one scope).
- [ ] **A failed gate stops `package` before anything is published, exits non-zero, and prints per-metric deltas;** `package` run on its own refuses a version whose recorded gate did not pass.
- [ ] `finetune` writes a run manifest to Blob with `artifacts.status: "staged"` even though weights are not pushed.
- [ ] `run package --version v2` pushes the **real** selected adapter, the merged model and each non-bf16 serving format to their Blob paths, then flips the manifest to `"published"`.
- [ ] `package` fails loudly with remediation when the requested version is not on the staging volume.
- [ ] `run extract --model base` runs the untuned base model with no adapter; `--model v2` resolves that version's merged model, plus a graduated per-type adapter where the routed type has one.
- [ ] `run all` chains 1 and 2, **never runs extraction**, and does not reach `package` when the gate fails.
- [ ] `--from-stage` resumes mid-pipeline; re-running a completed stage is a no-op — **except ingestion**, which always runs and relies on checksum dedup.
- [ ] A completion check covers **every** target it claims to, not just the first: a quantize run that failed halfway must not read as complete on resume.
- [ ] The staging volume is cleared only after at least one manifest was actually published.
- [ ] A pod launched without the staging volume attached is refused.
- [ ] **Cross-type regression evidence carries both sides.** `StageContext` exposes `cross_type_evidence` as `{doc_type: {"current": {...}, "candidate": {...}}}` — the shape `promotion_gate` reads — and it is passed through verbatim. It is a **separate field** from `revalidation_evidence`, which holds per-doc-type booleans for the arch §12 cascade; reusing one field for both produced an entry with no `"current"` key, which the gate correctly rejected as empty. A continued Foundation could then not pass by any input: supply metrics and it read as empty evidence, supply the booleans and it read as no evidence at all.
- [ ] **A deterministic failure is not retried.** The retry policy exists for transient faults — a throttled Blob read, a pod that dropped. `PipelineError` and `PathError` mean the stage cannot run at all with these inputs (below the day-zero label floor, missing `--input`, a malformed `--out-version`), so a second attempt re-lists everything and fails with the identical message after a pointless backoff. They stop the run immediately, like a gate block.
- [ ] **A dry-run `rollback-endpoint` does not move the endpoint.** Popping the deployment history before checking `dry_run` meant a preview permanently rewrote the recorded live version: `health_check` reported the previous version while the current one was still serving, and the next real rollback refused with "nothing to roll back to".
- [ ] A pod leaves no persistent process after completion.
- [ ] `deploy-endpoint --model vN` updates the serving endpoint; `rollback-endpoint` restores the previous promoted version.
- [ ] *(deferred)* Quantization thresholds gate `package`; the sweep sequence runs Phase 1 → 2 and promotes the winning config.
- [ ] A Foundation version bump produces the dependent-adapter re-validation work list and blocks promotion until it passes.

---

## Current implementation (2026-09-27)

**Commands** (`orchestration/run.py`): `finetune`, `package`, `all`, `extract`, `deploy-endpoint`,
`rollback-endpoint`, and **`freeze-eval-set --corpus vN [--allow-small]`** (SPEC_08).

**Every long job survives a disconnect** (`orchestration/detach.py`, `scripts/pod_run.sh`):
- On the pod — `RUNPOD_POD_ID` set, or RunPod's `/etc/rp_environment`, or Linux with `/workspace` mounted
  and a GPU present (a laptop with a GPU and a `workspace` folder is not the pod) — every long entry point
  re-launches itself in a tmux session and returns at once with the job's name. Covered: every
  `orchestration.run` command except the endpoint switches, `training.train`, `training.sweep`,
  `evaluation.run_eval`, OCR and rendering, the three importers, pre-annotation, active learning, the
  Phase 0 spike and `setup_pod.sh`. The guard runs after argument parsing, so `--help` and bad arguments
  answer at once.
- Inside tmux, or inside a `pod_run.sh` run (`FIDEON_DETACHED=1`), a command runs in place — never a second
  copy. Off the pod nothing changes. `FIDEON_NO_DETACH=1` opts one command out.
- `pod_run.sh start|attach|status|tail|list|stop`: logs and exit codes in `/workspace/logs`; one run per
  name; nothing ends a run except `stop`, which asks for the run's name. It cannot survive the pod itself
  stopping — `status` then reports the run as interrupted and `--from-stage` resumes.
- `tests/test_detach.py` fails if a new entry point is neither guarded nor listed as quick.
- **A run keeps its caller's environment**: each run records the caller's `PATH` and `VIRTUAL_ENV`, because a
  tmux session otherwise takes them from the tmux server (started by whichever job came first); training finds
  `swift` next to the running Python before searching `PATH`. The bootstrap builds its venvs with Python >= 3.11
  (`FIDEON_PYTHON` overrides), since `python3` on Ubuntu 22.04 can be the system 3.10.

**Stage behaviour added since this spec**:
- **Deterministic failures are not retried** (`TrainingError`, `StagingError`, `LengthCheckError`,
  `CorpusViewError`, `CalledProcessError`); only transient faults are.
- Training raises when a real run leaves no checkpoint; `checkpoint_eval` rediscovers checkpoints on a
  resumed run; merge reads the selection from Blob when resuming, and refuses a real run with none.
- Engines are released between stages (checkpoint selection → merge → calibration per format → golden eval),
  so each finds the card free.
- `package` requires a passed recorded gate, finds calibrators in Blob on resume, and uploads real weights
  through `artifact_registry/transfer.py` (SPEC_10).
- `--push-adapters` uploads the real best-loss adapter after training (non-dry runs), not a placeholder.
- The dataset build excludes frozen eval documents and their families once the set is frozen (SPEC_05).

**Pod setup** (`scripts/setup_pod.sh ocr|train|serve|quantize|dev`): installs the role's requirements file,
builds flash-attn against the installed torch, installs tmux, checks packages and CUDA.

**Pod bootstrap** (`scripts/pod_bootstrap.sh [--pdf <policy.pdf>]`): one command for a fresh H200 SXM pod,
detached in tmux and safe to re-run. In order: GPU (warns if not an H200), `/workspace` mounted, the repo on
the volume, 150 GB free (`FIDEON_MIN_FREE_GB`); `.env` keys `RUNPOD_VOLUME_MOUNT=/workspace`, `HF_HOME`,
`MINERU_TOOLS_CONFIG_JSON=/workspace/magic-pdf.json`, filled only when unset, blank or the template's
default, and a refusal if the connection string is unquoted; `/workspace/venv` (`setup_pod.sh train`) and
`/workspace/venv-ocr` (`setup_pod.sh ocr`); MinerU's config onto the volume and to cuda; the base model at
the pinned revision into `/workspace/models`; the spike in the training environment (`--skip-mineru`) and,
with `--pdf`, its MinerU checks in the OCR environment (`--only-mineru`). It ends with what is still to do.

**OCR before `finetune`**: `stage_preprocessing` refuses with a `PipelineError` naming the OCR environment's
command when documents still need OCR and MinerU is not importable (the training environment). With nothing
pending it needs no MinerU.

**RunPod backend**: `RunPodBackend` (launching pods through the API) is still unimplemented;
`LocalBackend` runs the DAG in-process on whatever machine starts it — on the pod, inside tmux.

**Post-OCR check and smoke run (2026-09-29).** `data_pipeline/ocr_check.py` checks every labelled value against
its page's OCR text in `processed/` (the audit's matching), reporting `fideon:filled` fields apart from the
generator's own, real apart from synthetic and per line; it exits non-zero when added fields are found on their
page more than `--max-gap` (default 0.05) less often than original ones. `orchestration/smoke_run.py` runs a few
whole source families (4 train, 1 val, 1 test by default) through import, OCR (in `/workspace/venv-ocr`), the
post-OCR check and `finetune`, under tenant `smoke` and version `v0`, never freezing or packaging (the frozen set
is not tenant-scoped). Both detach into tmux on the pod.
