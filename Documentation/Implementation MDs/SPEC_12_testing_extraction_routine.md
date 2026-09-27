# SPEC 12 — Testing / Extraction Routine (CLI Harness)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_03 (OCR), SPEC_07 (inference core), SPEC_08 (metrics), SPEC_09 (calibration), SPEC_11 (serving pipeline).
>
> **Naming hazard (master §0.1):** the architecture's "**Fideon SPEC_12**" is the external multi-tenant deployment spec. This file is our testing harness. Different documents.
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §17 (**extraction/testing routine**), §19 (folder placement), §13a (quantized-format validation consumer), §5 (confidence without ground truth).

## Goal

A CLI harness to test any fine-tuned model version on real PDFs post-training, producing structured JSON + per-field/overall confidence + quality metrics, with version-organized outputs and provenance. **It mirrors the exact production inference path (MinerU → model → JSON) by reusing the serving pipeline (SPEC_11), so what you test is what you ship.**

## Deliverables

### 1. Folder layout under `testing/` (create dirs with `.gitkeep`)
```
testing/
├── run_extraction.py            # CLI entrypoint
├── test_data/                   # INPUT: drop test PDFs here (scanned or digital, mixed)
├── prompts/                     # one prompt file per doc type
│   ├── acord.prompt.txt
│   ├── policy.prompt.txt
│   └── lossrun.prompt.txt
├── ocr_cache/                   # cached MinerU output per doc: {doc_stem}/page_*.md, page_*.png
├── results/{version}/           # OUTPUT: extracted JSON, organized BY MODEL VERSION
│   └── v2/abcLossRun.json
├── metrics/{version}/           # OUTPUT: per-doc metrics + _run_summary.json
└── extraction_registry.json     # provenance: which doc → which model version, when
```

### 2. `testing/prompts/*.prompt.txt`
One per active doc type. Each contains the role framing, the **injected JSON schema for that type pulled from the SPEC_01 registry** (so it stays in sync), the extraction rules (null-handling, date formats, repeating table rows), and the modality-mode instruction line.

**Why one file per type (arch §17):** each document type has a different schema, different field semantics (a Loss Run's "valuation date" vs. an ACORD's "policy effective date"), and different edge cases. Separate files mean the doc-type detection step maps directly to a prompt file, and one type's prompt can be tuned without risk to the others.

**These are the same prompts serving uses.** They must resolve to the identical rendered prompt as `common.prompts` — testing and production must not drift. Assert this equality in a test rather than maintaining it by discipline.

**Canonical field mapping needs no change here** (master §1.4): the semantic glosses are schema `description` fields, so they propagate automatically through the existing schema injection. The one requirement is negative — **the extraction path must never import `common.aliases`**. Mapping surface labels to canonical keys is the model's job; a runtime lookup would cap the system at a hand-written list. `test_no_runtime_aliases` (SPEC_14) enforces this.

### 3. `testing/run_extraction.py` — the CLI

```bash
# User decides which model to run — base, v1, v2, v3, ...
python testing/run_extraction.py --model v2   --input testing/test_data/
python testing/run_extraction.py --model base --input testing/test_data/   # zero-shot baseline

# Or through the operator command surface (SPEC_13), which wraps this:
python -m orchestration.run extract --model v2 --input testing/test_data/

# Single file, image-only mode (no OCR), to test the no-OCR pathway
python testing/run_extraction.py --model v2 --input testing/test_data/abcLossRun.pdf --mode image_only

# Full batch with OCR (default), with ground truth for metric scoring
python testing/run_extraction.py --model v3 --input testing/test_data/ \
       --mode ocr_plus_image --ground-truth testing/golden/
```

- `--model` resolves via SPEC_02 `resolve_model_version` — **the user never has to know paths, just the tag.** Accepted values:

  | Value | Resolves to |
  |---|---|
  | `base` | Untuned `Qwen/Qwen3-VL-8B-Instruct` at the pinned revision, **no adapter** |
  | `v1`, `v2`, … | The merged model, plus a graduated per-type adapter where the routed type has one |
  | `v2` + `--format fp8` | That version's FP8 serving weights. A GGUF format here is a category error: llama.cpp loads those, and the endpoint runs vLLM |

  **`base` is not a curiosity** — it is the zero-shot baseline of the pilot protocol (SPEC_15) and the day-zero pre-annotation path (SPEC_04). Supporting it here means those two have one implementation, not three.
- `--mode ocr_plus_image` (default) | `image_only` — both production pathways testable from one harness.
- `--ground-truth` is **optional**: if supplied, metrics are computed against golden JSON; if omitted, the routine still runs and produces JSON + confidence scores.
- Additional: `--format` (to test a specific quantized GGUF — the SPEC_10 `validate_quant` entry point when that ships), `--limit`, `--tenant` (optional, defaults from env).

**What it does, step by step (arch §17):**
```
For each PDF in --input:
  1. OCR              → MinerU produces markdown + rendered page image(s) → ocr_cache/
                        (skipped if --mode image_only; the image is still rendered)
  2. Doc-type detect  → classify as acord | policy | lossrun (+ ACORD form)
  3. Prompt select    → load the matching prompts/{doc_type}.prompt.txt
  4. Assemble input   → system = prompt (schema + instructions)
                        user   = [image] (+ OCR markdown, unless image_only)
  5. Model inference  → selected --model version generates JSON + token logprobs
  6. Confidence       → per-field confidence from logprobs, calibration applied
  7. Validate         → JSON parses + conforms to that doc type's schema
  8. Write result     → results/{version}/{doc_stem}.json
  9. Metrics          → if ground truth available → metrics/{version}/{doc_stem}.metrics.json
 10. Register         → append to extraction_registry.json
```

**Steps 2–7 are executed by calling `serving/pipeline.py` (SPEC_11)** — the routine is a thin CLI wrapper over the serving pipeline plus OCR and metrics. It must **NOT re-implement inference**; doing so breaks test == prod, which is the entire point of the harness.

### 4. Metrics output

**Per-doc `metrics/{version}/{doc}.metrics.json`** (arch §17):
```json
{
  "document": "abcLossRun.pdf",
  "model_version": "v2",
  "doc_type": "loss_run",
  "schema_valid": true,
  "overall_confidence": 0.88,
  "line_of_business": {"value": "workers_comp", "confidence": 0.92, "correct": true},
  "fields": {
    "carrier":          {"value": "...", "confidence": 0.97, "correct": true},
    "policy_number":    {"value": "...", "confidence": 0.94, "correct": true},
    "valuation_date":   {"value": "...", "confidence": 0.71, "correct": false},
    "claims[0].amount": {"value": 12000, "confidence": 0.63, "correct": true}
  },
  "field_exact_match_rate": 0.91,
  "list_field_f1": 0.88,
  "list_field_recall": 0.83,
  "row_completeness_flag": false,
  "pages_used": [1, 2]
}
```
All correctness uses SPEC_08 metrics and **`common.normalize`** — the same definition of "correct" as the promotion gate.

**Batch summary `metrics/{version}/_run_summary.json`** — aggregates across all test docs for that model version: mean field-exact-match, mean list-field F1, **list-field recall**, schema-validity rate, ECE, **LoB detection accuracy per value**, per-doc-type breakdown, per-mode breakdown, and mean latency/document. **This is the file compared across model versions to answer "did v3 beat v2?"** — the testing-time mirror of the SPEC_08 promotion gate.

**On the confidence-without-ground-truth case (arch §17):** even with no golden JSON — a brand-new unknown document — the routine still emits `overall_confidence` and per-field calibrated confidence. That is the whole payoff of the calibration work: confidence is available at inference time on documents you have never labeled, which is exactly what routes low-confidence extractions to human review in production.

### 5. `testing/extraction_registry.json`
One entry per extraction, so an output JSON is never ambiguous about its origin:
```json
{
  "extractions": [
    {
      "document": "abcLossRun.pdf",
      "doc_type": "loss_run",
      "model_version": "v2",
      "quant_format": "fp8",
      "mode": "ocr_plus_image",
      "result_path": "results/v2/abcLossRun.json",
      "metrics_path": "metrics/v2/abcLossRun.metrics.json",
      "overall_confidence": 0.88,
      "schema_valid": true,
      "extracted_at": "2026-02-15T11:03:00Z"
    }
  ]
}
```
Combined with the `results/{version}/` layout, anyone can see at a glance that `results/v2/abcLossRun.json` came from model v2 — from the path *and* the registry. `model_version` here is the same tag that resolves to a `run_manifest.json` (SPEC_02), so an extracted result traces all the way back to the corpus version and git commit that produced the model that generated it.

## Constraints
- Confidence emitted even without ground truth.
- **Same prompts and same pipeline as serving — reuse, never reimplement.**
- MinerU version parity with the training corpus asserted (SPEC_03).
- No PII in logs; `results/`, `metrics/`, and `ocr_cache/` are gitignored.

## Acceptance checklist
- [ ] `run_extraction --model v2 --input test_data/` writes `results/v2/*.json`.
- [ ] `--model base` runs the untuned base model with no adapter and writes `results/base/*.json`.
- [ ] `orchestration.run extract` produces byte-identical output to calling `run_extraction.py` directly.
- [ ] `--mode image_only` runs with no OCR and still extracts.
- [ ] Per-field + overall confidence present **with and without** ground truth.
- [ ] `line_of_business` present in every result, with confidence, `null` when undetermined.
- [ ] With `--ground-truth`, per-doc + summary metrics computed, including list-field recall and per-value LoB accuracy.
- [ ] `--format q4_k_m` runs a quantized GGUF (feeds SPEC_10 `validate_quant` once that is built).
- [ ] `extraction_registry.json` links result → model version → run manifest.
- [ ] `testing/prompts/*.prompt.txt` render identically to `common.prompts` for the same inputs (asserted, not assumed).
- [ ] **Output equals the serving pipeline's output on the same input** (parity test).

---

## Current implementation (2026-09-27)

- `testing/run_extraction.py` validates its arguments and points to the operator command, which is the
  supported entry: `python -m orchestration.run extract --model base|vN --input ... [--mode image_only]`.
  Extraction goes through `serving.pipeline.extract` — windows and merge for canonical policies, calibrated
  confidence, dates in `MM/DD/YYYY` — so test output is production output.
- A policy request should carry its line of business (`lob`); without it the fallback canonical schema is
  used.
- On the pod `extract` runs detached in tmux like every long job; it survives a closed laptop.
- The gate's numbers no longer come from this harness: they come from the frozen golden eval set
  (SPEC_08 `golden_eval.evaluate_version`), through the same serving pipeline.
