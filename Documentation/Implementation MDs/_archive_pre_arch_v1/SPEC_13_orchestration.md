# SPEC 13 — Orchestration (RunPod Controller + Pipeline DAG)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: all prior specs (01–12).

## Goal

Automate the full lifecycle: spin up ephemeral RunPod training pods, run the pipeline stages, and manage the persistent serving endpoint — with business/orchestration logic living **outside** RunPod. (Arch §12, §13.)

## Deliverables

### 1. `orchestration/runpod_controller.py`
- Via the RunPod API:
  - Launch an **ephemeral GPU pod** for a training/eval/quantize job (instance configurable: A100 80GB for Foundation, smaller for per-type adapters).
  - Pod clones the repo at a given commit, installs, pulls only needed artifacts from Blob (SPEC_02), runs the job, pushes results + manifest, then **terminates**.
  - Poll status, collect logs, handle failures + retries.
- Manage the **persistent Serverless vLLM endpoint** (SPEC_11): deploy/update to a promoted version, health-check.
- CLI: `train-foundation --corpus vN --commit <sha> --gpu a100-80`, `train-adapter ...`, `quantize ...`, `deploy-endpoint --model vN`.

### 2. `orchestration/pipeline_dag.py`
- The end-to-end pipeline (arch §12 stages 1–11) as an ordered DAG (Airflow OR GitHub Actions; keep stage functions reusable):
  1. ingest → 2. OCR → 3. label → 4. dataset build → 5. train (foundation/adapters) → 6. evaluate + gate → 7. merge → 8. quantize + validate → 9. push artifacts → 10. deploy endpoint → 11. feedback loop (active-learning queue).
- Idempotent, resumable stages; a failed stage doesn't corrupt state.
- **Gating (SPEC_08) is a hard stop**: no promote/deploy unless the candidate beats current.

### 3. `orchestration/config/`
- Which GPU per stage, corpus/version tags, schedule (e.g., weekday retrains), notification hooks.

## Constraints
- Training pods **ephemeral** — nothing persistent resident; code cloned fresh.
- Orchestration/business logic on cheap CPU infra outside RunPod; only GPU-bound work on RunPod (arch §13).
- Every training/eval/quantize job writes a RunManifest (SPEC_02).
- Secrets from env only.

## Acceptance checklist
- [ ] Controller launches a pod, runs a trivial job, pushes output, tears down.
- [ ] `deploy-endpoint --model vN` updates the serving endpoint.
- [ ] DAG runs stages in dependency order; a failed stage halts promotion.
- [ ] Gating blocks deploy on regression.
- [ ] No persistent process left on a training pod after completion.
