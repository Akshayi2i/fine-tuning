# Insurance Extraction Fine-Tuning (Fideon L3)

Fine-tunes **Qwen3-VL-8B-Instruct** to extract structured JSON from insurance
PDFs, with a calibrated confidence score on every field.

This repo is the **L3 VLM layer** of the Fideon pipeline. L0 to L2 handle intake,
carrier lookup, and structural inference; L3 is the fallback for the documents
they cannot handle - scanned input, unknown carrier, structure failure.

Its output schema is **owned by Fideon SPEC_00**, not by this repo. The schemas
in `schemas/` are those canonical models serialised to JSON Schema.

## Documents

| Where | What |
|---|---|
| `Documentation/finetuning-architecture-v2.1.docx` | The design and its rationale - the *why* (v2.1, with the v2.2 implementation update). v1 is kept for history only |
| `Documentation/Implementation MDs/SPEC_00` .. `SPEC_15` | Module specs and acceptance criteria - the *how*; SPEC_00 §13 indexes what the code does now |
| `Documentation/Implementation MDs/SPEC_ALL_COMBINED.md` | All specs in one file |

The two are in sync. **If they disagree, that is a bug in one of them** - fix the
disagreement rather than picking a winner.

## Two production modes, one model

1. `ocr_plus_image` - MinerU OCR text plus the page image.
2. `image_only` - the page image alone, no OCR.

The second is why a rule-based label mapper cannot serve this system: on a
scanned page there is no text for a rule to read.

## Operator commands (SPEC_13)

```bash
# 1  ingest -> OCR -> corpus -> train -> select checkpoint -> merge  (staged on the pod)
python -m orchestration.run finetune --input ./intake --corpus-version v1 --out-version v2

# 2  quantize -> calibrate -> GATE -> push real weights + release bundle to Azure Blob
python -m orchestration.run package --version v2 --release-id release-2026.11.1 --formats bf16

# 3  extraction, model chosen by the operator
python -m orchestration.run extract --model base|v1|v2 --input testing/test_data/

# all = 1 + 2. Never includes extraction.
python -m orchestration.run all --input ./intake --out-version v2 --release-id release-2026.11.1

# once, from the first real corpus build: the gate's frozen eval set
python -m orchestration.run freeze-eval-set --corpus v1
```

## Pilot protocol (SPEC_15)

Three experiments in increasing order of investment, run **before** committing to
full annotation. Each one attributes the next one's failures, so a missing
experiment blocks the summary rather than passing by default.

```bash
# A  zero-shot baseline - the untuned base model, no annotation cost
python -m orchestration.run extract --model base        --input pilot/baseline_docs/ --ground-truth pilot/baseline_golden/

# C  ... then the go/no-go across all three
python -m pilot.pilot_report --write
```

The deferred hyperparameter sweep (SPEC_06) runs **after** this passes and
**before** production-scale training: sweeping against 25-30 documents per type
measures noise.

## Setup

```bash
cp .env.example .env              # fill in Azure + RunPod credentials
pip install -r requirements.txt   # data pipeline, eval, test tooling - CPU only
pytest -q                         # fixture-driven, no GPU or live Azure needed
```

### The H200 pod: one command

Clone the repo onto the network volume and run the bootstrap. It runs in tmux,
so the laptop can close while it installs and downloads, and it is safe to
re-run: after a pod stop it only checks what is already there.

```bash
cd /workspace && git clone https://github.com/Akshayi2i/fine-tuning.git && cd fine-tuning
bash scripts/pod_bootstrap.sh --pdf /workspace/sample_policy.pdf   # --pdf is optional
```

It checks the GPU, the `/workspace` mount and 150 GB free; fills `.env`
(`RUNPOD_VOLUME_MOUNT=/workspace`, `HF_HOME`, `MINERU_TOOLS_CONFIG_JSON`) without
overwriting a value you set; builds **two environments** on the volume; moves
MinerU's config onto the volume and sets it to cuda; downloads the base model at
the pinned revision into `/workspace/models`; and runs the Phase 0 spike. It ends
with a list of anything still to do (the Azure connection string, MinerU's
weights).

| Environment | For | Why separate |
|---|---|---|
| `/workspace/venv` | training, `finetune`, `package`, serving (torch 2.8, ms-swift, vLLM 0.11, flash-attn) | |
| `/workspace/venv-ocr` | OCR only (MinerU 1.x) | MinerU 1.x cannot share an environment with vLLM 0.11 / torch 2.8 |

Everything is on `/workspace` because the container disk is wiped when the pod
stops. Keep the Azure connection string in **double quotes** in `.env`: it is
sourced by bash, which would otherwise cut it at the first `;`.

**OCR runs first, in its own environment**, then the pipeline:

```bash
/workspace/venv-ocr/bin/python -m data_pipeline.ocr.run_mineru --doc-type policy --all-unprocessed
source /workspace/venv/bin/activate
python -m orchestration.run finetune --scope personal_lines --skip-ingest --corpus-version v1 --out-version v1
```

`finetune` finds every document already OCR'd and skips that stage. If documents
still need OCR it stops with the command above rather than failing on an import.

### One role at a time

On a RunPod pod, one command per role installs its dependencies in the order
that works and checks CUDA afterwards (the bootstrap calls it for `train` and
`ocr`):

| Pod | Command | Installs |
|---|---|---|
| OCR (stage 2) | `bash scripts/setup_pod.sh ocr` | `requirements-ocr.txt`: MinerU 1.x (`magic-pdf[full]`); after downloading its weights, `python -m data_pipeline.ocr.mineru_config --cuda` (MinerU defaults to CPU) |
| Training | `bash scripts/setup_pod.sh train` | `requirements-train.txt`: ms-swift, torch 2.8, **and vLLM** (checkpoint selection and calibration generate with it), then flash-attn built against that torch |
| Serving | `bash scripts/setup_pod.sh serve` | `requirements-serve.txt`: vLLM 0.11.0, the same build the training pod calibrates with |
| Quantization (only once FP8 is verified) | `bash scripts/setup_pod.sh quantize` | `requirements-quantize.txt`: llmcompressor, in its own environment (its `datasets`/`transformers` ranges conflict with ms-swift and vLLM) |

Versions are declared once, in `pyproject.toml`; the requirements files only
choose dependency groups. `tests/test_dependencies.py` fails if the code imports
a package no group installs.

## Running on the pod: closing the laptop does not stop it

On the pod, every long job runs inside tmux automatically - the pipeline
(`python -m orchestration.run ...`), training, sweeps, evaluation, OCR,
ingestion, pre-annotation, the Phase 0 spike and `setup_pod.sh`. Type the
command as usual; it starts itself in a tmux session and returns at once with
the session's name. The job belongs to the pod, not to your SSH connection:
shutting the laptop, letting it sleep or losing Wi-Fi does not stop it.

```bash
python -m orchestration.run finetune --corpus-version v1 --out-version v1
#   -> "This one is running as 'finetune-20261003-141502'."

bash scripts/pod_run.sh list                  # every job on this pod
bash scripts/pod_run.sh status <name>         # running / finished + exit code + last log lines
bash scripts/pod_run.sh attach <name>         # watch live; Ctrl-b then d to leave it running
bash scripts/pod_run.sh tail <name>           # follow the log on the volume (/workspace/logs)
```

How it decides: on the pod (`RUNPOD_POD_ID` set, or RunPod's `/etc/rp_environment`,
or Linux with `/workspace` mounted and a GPU present) a command not already
inside tmux re-launches itself through `scripts/pod_run.sh`; inside tmux, or on a
laptop or CI, it runs in the foreground as before. `tests/test_detach.py` fails
if a new long-running entry point is added without this guard.

Nothing stops a run except `pod_run.sh stop <name>`, which asks you to type the
name to confirm. The session stays open after the run ends so its output can be
read. What a run does not survive is the pod itself stopping or restarting;
then `status` says so, and `--from-stage` resumes from the last completed stage.

## GPU only

Every step that runs a model refuses to start without a CUDA device rather than
fall back to the CPU (`common/gpu.py`): training, the merge, quantization, vLLM
generation (checkpoint selection, calibration, the golden eval, serving), the
Hugging Face backend and MinerU OCR. Models are placed with `device_map="cuda"`,
never `"auto"`, which would quietly offload layers to the CPU when VRAM runs
short. The error names the cause - a CPU-only torch build, or no visible GPU.
What stays on the CPU has no GPU form: JSON, tokenizer counts, downloads.

## The eval set: freeze it once

Train and val come from each corpus build. The promotion gate does NOT score the
build's test split — that is re-drawn on every rebuild — but a frozen golden eval
set at `golden-eval-set/` in Blob, the same documents for every model version.
Create it once, from the first real corpus build:

```bash
python -m orchestration.run freeze-eval-set --corpus v1
```

It refuses fewer than 150 test documents per type (arch §15.4) unless `--allow-small`. It copies that corpus's test split (labels, metadata, page images, OCR) into the
frozen set and records `manifest.json`. From then on every corpus build leaves the
frozen documents and their families out, and splits new documents into train
and val only. Freezing again is refused; replacing the set is a deliberate delete
followed by a new freeze, after which older scores are not comparable.

## Check the data before training

Before uploading a training folder, audit it on the laptop (read-only):

```bash
python -m data_pipeline.audit                      # default: data/training data, scope personal_lines
```

It reports blockers (one PDF + `golden.json` + `metadata.json` per folder, PDF readable, `lob` valid and in
scope, `page_ref` inside the PDF), checks each labelled value against the text on its page, checks formats and
totals, and draws a 5% spot-check sample per line. Fix and re-run until it exits 0. The report and CSVs go to
`data/audit_report/`, which is git-ignored: they quote label values.

## Getting the data to the pod: through Blob, not git

The training folder (~20 GB) never goes through git. Stage it in Blob from the
laptop with [azcopy](https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10),
then pull and import it on the pod:

```powershell
# laptop, after the audit passes. <raw container> is AZURE_RAW_CONTAINER; the SAS needs
# read, write and list on it. Re-run the same command to resume or to push corrections.
azcopy sync "D:\Fine-Tuning\data\training data" "https://<account>.blob.core.windows.net/<raw container>/intake/personal-v1?<SAS>" --recursive
```

```bash
# pod (runs in tmux; re-run to resume)
python -m data_pipeline.ingestion.pull_intake --batch personal-v1        # -> /workspace/intake/personal-v1
python -m data_pipeline.audit --input /workspace/intake/personal-v1      # optional second check
python -m data_pipeline.ingestion.import_labeled_pdfs --input /workspace/intake/personal-v1 --doc-type policy --validate-only
python -m data_pipeline.ingestion.import_labeled_pdfs --input /workspace/intake/personal-v1 --doc-type policy
# or pull and import in one: import_labeled_pdfs --from-blob personal-v1 --doc-type policy
```

`intake/` sits in the **raw container**, under the raw layer's access rule: it is
the delivered PDFs with their PII, readable only by ingestion. The pull runs 16
downloads in parallel, writes each file as `.part` and renames it when complete,
skips PDFs already on the volume, and always re-fetches labels, so a label fixed
and re-synced after an audit is never stale on the pod. The upload itself runs on
the laptop, which has to stay on for it; azcopy is parallel and resumable, and
everything after it runs on the pod.

## Scopes

A training run covers a **scope** (`configs/scopes.yaml`): `unified` (every type), `policy`, `lossrun`, and
`personal_lines` — policies of homeowners, personal auto, dwelling fire, ocean marine, classic auto,
motorcycle, recreational vehicle, personal umbrella and flood. A line-scoped release answers only for its
lines; a policy request must carry its `lob` to reach it.

```bash
python -m orchestration.run finetune --scope personal_lines --corpus-version v1 --out-version v1
```

## Build status

All ten phases are built. `pytest -q` runs the whole suite on CPU with no live
Azure account.

| Phase | Scope | State |
|---|---|---|
| 0 | Infra + dependency spike | **pending (yours)** |
| 1 | Contracts, configs, fixtures | done |
| 2 | Blob I/O + run registry | done |
| 3 | Ingest + MinerU OCR (GPU) | done |
| 4 | Inference core | done |
| 5 | Labeling + corpus builder | done |
| 6 | Training | done |
| 7 | Evaluation + calibration | done |
| 8 | Merge/quantize, serving, testing harness | done |
| 9 | Orchestration + CI | done |
| 10 | Pilot validation protocol (SPEC_15) | done; **runs when the corpus arrives** |

**Phases 1-10 prove the plumbing. They do not prove the model works.** No
accuracy claim is possible until the pilot runs on a real corpus - and a green
test suite is not evidence about extraction quality.

## Rules that are not style preferences

- **The repo never stores weights, corpora, or PDFs.** Everything data- or
  artifact-related moves through `artifact_registry/` to Azure Blob at runtime.
- **Training and inference prompts must render byte-identically.** Divergence
  degrades a fine-tuned model and is invisible in training metrics.
- **The image resolution cap is identical in corpus prep and serving.** A
  mismatch means serving a distribution the model never trained on.
- **Loss is computed only on assistant tokens.** System, image, and OCR tokens
  are masked to `-100`; break it and the model trains on its own prompt while
  the loss curve looks normal.
- **The alias registry is never used at inference.** It is labeling and
  evaluation material. The model does the semantic mapping.
- **The promotion gate has no override flag.** The one exception is a recorded
  written override (named approver, reason, waived gates), stored with the release.

## What is deliberately not wired

Every one of these is a documented boundary, not an oversight: each raises with
the reason and what unblocks it. They are the whole of what Phase 0 gates.

| Where | Waiting on |
|---|---|
| `data_pipeline/ocr/run_mineru.py` (`MinerUEngine`, written against MinerU 1.x) | the Phase 0 spike running it on the GPU with MinerU's model weights |
| `postprocessing/quantize.py` (FP8/AWQ export, GGUF export) | llm-compressor in its own environment; FP8 verified by the spike (bf16 needs no export) |
| `artifact_registry/transfer.py` (`pull_base_model` from the Hub) | not needed on the pod: the base is read from `/workspace/models` |
| `orchestration/runpod_controller.py` (`RunPodBackend`, endpoint deploy) | `RUNPOD_API_KEY` and the network volume; until then jobs run on the pod in tmux |
| `inference_core/model_runner.py` (vLLM and HF backends, written; merge in `training/merge.py`, written) | a GPU to run on - they refuse without CUDA |
| `testing/run_extraction.py`, `pilot/zero_shot_baseline.py` CLIs | a live model backend, i.e. the row above it |
| `training/data_collator.py` custom hook | nothing - ms-swift collates and masks; the hook exists only for a genuine override |
| `data_pipeline/labeling/pre_annotate.py` external backend | a compliance decision **and** a zero-retention endpoint; refuses without both |
| `evaluation/golden_eval.py`, `data_pipeline/labeling/active_learning.py` inference loops | a live model backend on the GPU; the golden eval runs through the serving pipeline |

`LocalBackend`, `EchoBackend` and `InMemoryBackend` are **real implementations**,
not stubs: they run the whole DAG, the whole pipeline and the whole registry in
process, which is why CI covers this much without a pod.

## Open items

- **Presidio de-identification is BLOCKED** (SPEC_05 section 1). Text-only
  de-identification corrupts the training signal - resolve before production.
- **Field glosses need SME review.** They are prompt text; changing one after the
  first corpus build forces a rebuild and retrain.
- **The dependency spike has not run.** ms-swift with Qwen3-VL, flash-attn,
  vLLM multi-LoRA, MinerU on GPU and its determinism. `scripts/pod_bootstrap.sh`
  runs it; the base model is pinned to `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
  (the Hub commit of 2025-10-15).
- **The pilot protocol is unrun.** `python -m pilot.pilot_report` reports
  `INCOMPLETE` until Experiments A, B and C have each written a report, and a
  missing experiment blocks rather than passing.
