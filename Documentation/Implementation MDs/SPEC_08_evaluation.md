# SPEC 08 — Evaluation (Metrics + Promotion Gating)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_06, SPEC_07. Uses the shared inference core (SPEC_07) for all model runs — no forward dependency on serving or testing.
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §15 (**metric definitions + LoB metric**), §0b (LoB gating), §3 (ViT gate inputs), §5 (list-field recall), §13 step 6 (the gate is a hard stop), §12 (gate result recorded in the manifest).

## Goal

Evaluate any model/adapter version against the **frozen golden eval set** and gate promotion so no regression ships.

## The gate is a hard stop (arch §13)

A candidate is promoted only if it passes **every** gating metric for the relevant document type(s): the absolute floors, and paired-bootstrap non-inferiority against the current production version (arch v2.1 §15). **There is no `--force` flag.** The one exception path is a **recorded written override** (arch v2.1 §15.5): a named approver, a written reason and the gates it waives, stored in the gate decision and the release bundle. `package` reads the recorded decision and refuses a version whose gate did not pass.

## Deliverables

### 1. `common/normalize.py` (shared normalizer — **built in SPEC_01**, consumed here)
> **Moved to SPEC_01.** `common/normalize.py` is now built in SPEC_01 alongside the other `common/` helpers, because **SPEC_04 `derive_aliases` needs it too** — and SPEC_04 builds before SPEC_08. It is a pure, dependency-free string utility, so it belongs with the rest of `common/`. This spec **consumes** it; it no longer creates it.

Field-type normalizers make comparison reflect correctness rather than formatting (arch §15). Insurance fields specifically need this or you under-count correct extractions:
- **Dates**: `01/01/2026` == `2026-01-01`
- **Currency**: `$1,200.00` == `1200.0`
- **Entity names**: `Acme Mfg LLC` ≈ `ACME MANUFACTURING LLC`
- Policy/claim numbers: whitespace and separator normalization.

One definition of "matches", applied **consistently in the promotion gate, the testing routine (SPEC_12), and alias derivation (SPEC_04)** — divergence silently changes what "correct" means between them.

### 2. `evaluation/metrics/` — one module per metric, each pure and testable

| Where (`evaluation/…`) | Metric | What it catches (arch §15) |
|---|---|---|
| `metrics/field_accuracy.score_fields` | Exact + normalized match, per-field and aggregate | Core extraction accuracy |
| `metrics/field_accuracy.score_list_field` | Precision/recall/F1 for list fields (claims, schedule rows) | Precision/recall on repeating structures |
| `metrics/field_accuracy.score_list_field` (recall) | **Row completeness** — extracted row count vs ground truth; missed-row rate | Whole rows silently missed — a dropped Loss Run claim is invisible to per-value confidence (arch §5) |
| `metrics/coverage_metrics.score_schema_validity` | Parse + validate against the doc_type schema | Structural reliability; mirrors the Fideon SPEC_07 Stage 3 audit gate |
| `metrics/coverage_metrics.expected_calibration_error` | Expected Calibration Error on confidence vs correctness | Whether the confidence numbers are trustworthy, not just the extractions |
| `run_eval` — the `noisy_ocr` subset | Scored on the **deliberately-noisy-OCR subset** | Did the model correctly override bad OCR using the image? |
| `run_eval` — `image_only` / `scanned` subsets; `metrics/coverage_metrics.score_by_mode`; `training/vit_gate.classify_error` | Separate accuracy for the `image_only` and `scanned` subsets | Confirms the second production pathway works; **feeds the ViT escalation gate** (SPEC_06) |
| `doc_type_classifier_accuracy` — not applicable while the corpus builds no classify rows | Doc-type + ACORD-form classification accuracy | Whether the right adapter/prompt/schema is even selected — **a classifier at 92% caps the whole system at 92%** (arch §4a) |
| `metrics/coverage_metrics.score_lob` | **`line_of_business` detection accuracy, overall and per LoB value** | The VLM is the fallback LoB detector when L1/L2 miss (arch §0b) |
| `metrics/confusable.score_alias_accuracy` | Field accuracy **sliced by the observed surface label** | Whether the canonical mapping generalises across phrasings, or only works on the dominant one (master §1.4) |
| `metrics/confusable.score_misattribution` | Rate at which a **confusable entity's value is returned as the canonical field** | The failure that produces confident, well-formed, wrong extractions — a certificate holder returned as `insured_name` |
| `ExtractionResult.latency_ms` (serving) — not yet a report metric | Latency / token cost per document | Production feasibility, not just accuracy |
| `metrics/extraction_faults` — `score_false_nulls`, `score_hallucinations`, `score_page_selection` | False-null rate, hallucination rate, page-selection recall | A value on the page emitted as null; a value emitted that is on no page sent; pages with fields that routing dropped |

**Alias accuracy specifics (master §1.4).** Joins predictions to each eval document's `field_provenance` (SPEC_04) and reports accuracy per canonical field × surface label. This turns an unhelpful aggregate into an actionable one: *0.94 on "Named Insured", 0.61 on "Applicant"* tells you the mapping is not generalising and names the documents to go collect. **Reported, not gating** — rare aliases have too little support for a stable gate, and gating on them would block promotion on noise.

**`confusable.py` specifics.** For each canonical field, check whether the returned value matches the document's value for one of that field's registered **confusables** instead. Requires the eval golden labels to carry the confusable entities' values, so the frozen eval set must include the confusable co-occurrence documents from SPEC_05. **This is a gating metric.** Ordinary field accuracy already penalises a wrong value — but misattribution is worth isolating because it is systematic rather than random: it means the model has collapsed two distinct entities, it will keep doing so, and the output is fluent and confident enough to pass every structural check.

**LoB accuracy specifics (arch §15):** LoB accuracy is reported as **its own metric, measured per LoB value**, and is **never averaged into overall field accuracy** — a class that is rare in the corpus must not hide inside a healthy-looking aggregate. It is a gating metric.

**Mode accuracy specifics:** must also classify errors as **perception** (misread characters, missed checkboxes) vs **schema/reasoning** (right value, wrong field), because SPEC_06's `vit_gate` needs that distinction, not just the accuracy number.

### 3. `evaluation/run_eval.py`
- Given `--model vN` (resolved via SPEC_02) + the frozen golden eval set, run inference **through the serving pipeline** (`serving.pipeline.extract`, built on the SPEC_07 inference core — so windows, merge, date formatting and calibrated confidence are what is measured) across all eval docs and all three modes, compute every metric, and write `eval-reports/v{n}/{doc_type}/report.json` plus a top-level summary. Broken down **per doc_type and per modality mode**.
- Runs the eval subsets explicitly: `image_only`, `scanned`, `noisy_ocr`, `long_policy`, plus the full set.
- Records per-document error records (field, expected, got, error class) so `vit_gate` and failure-mode analysis have real material rather than aggregates.
- **The golden eval set is frozen and versioned separately, human-double-verified, and held constant across corpus versions** so model versions compare apples-to-apples over time (arch §8). Never train on it — assert no eval `source_id` appears in any corpus split.
- **Every gating metric must be emitted by `score_subset`.** `gating.GATING_METRICS` is checked with `require_all_measured=True`, so a metric the scorer never computes blocks *every* candidate, permanently, and the gate has no override by design. This is not hypothetical: `ece_confidence` and `confusable_misattribution_rate` were both gating metrics that `run_eval` never produced — `score_misattribution` had zero production callers and ECE had none at all — which made the gate unpassable under any input. The names must match exactly too: `list_field_recall` and `lob_detection_accuracy` are the gate's spellings, and emitting `list_recall` or `lob_accuracy` instead means the metric silently never arrives. **A test asserts the two sets agree** (`tests/test_module_seams.py`), because a hand-written metrics dict in a unit test proves nothing about what the scorer emits.
- ECE needs per-field confidences: `score_subset` reads them from each document's `metadata["field_confidence"]` and skips fields that carry none, rather than assuming a value. An unmeasured field is not evidence in either direction.
- Note: classifier accuracy uses the classifier from SPEC_11 once it exists; until then `run_eval` scores extraction with the true doc_type supplied, and classifier metrics fill in on a later pass. This keeps eval runnable before serving is built.

### 4. `evaluation/gating.py`
- `promotion_gate(candidate_report, current_report, doc_type)` → the candidate must match or beat the current production version on **every** gating metric. Returns per-metric deltas + an overall decision.

**Gating metric set:** field exact/normalized match, list-field F1, **list-field recall**, schema validity rate, ECE, OCR-arbitration accuracy, image-only accuracy, scanned accuracy, doc-type classifier accuracy, **LoB detection accuracy (overall and per value)**, **confusable misattribution rate**.

`alias_accuracy` is **reported alongside but not gated** — see the note above.

- **Additional gate for `--continue-from` Foundation runs (arch §12):** a minor incremental patch that continued from the previous Foundation checkpoint must additionally show **no regression on the *other* document types**, not just the type it was patching. The gate reads the manifest flag SPEC_06 sets and demands that cross-type evidence before passing.
- Writes the decision, per-metric deltas, and `gated_against` into the candidate's `RunManifest` (SPEC_02).
- **No override flag.** A failed gate is a failed gate unless a written, attributed override is recorded with it (above).

## Constraints
- Golden eval set is frozen + versioned separately; never train on it — assert it.
- Metrics deterministic + unit-testable on synthetic inputs.
- All model runs go through SPEC_07 — no bespoke inference here.
- `common/normalize.py` is the single definition of "correct" shared with SPEC_12.
- No PII in eval reports beyond what access control permits (master §8) — scrub or restrict.

## Acceptance checklist
- [ ] Each metric returns correct values on hand-constructed synthetic cases.
- [ ] Normalized match handles dates, currency, and entity names via `common.normalize`.
- [ ] `list_recall` flags a deliberately dropped claim row.
- [ ] `lob_accuracy` reports per-LoB-value accuracy and does not fold LoB into the aggregate field score.
- [ ] `alias_accuracy` slices field accuracy by observed surface label from `field_provenance`, and a field scoring well on its dominant alias but poorly on a rare one is visible in the report.
- [ ] `confusable_misattribution` scores a prediction that returns the certificate holder's value as `insured_name` as a failure, distinctly from an ordinary wrong value.
- [ ] A regression on confusable misattribution **alone** blocks promotion.
- [ ] `mode_accuracy` separates perception errors from schema/reasoning errors on seeded inputs.
- [ ] `run_eval` produces a per-doc-type report with mode breakdowns (`image_only`, `scanned`, `noisy_ocr`, `long_policy`), using SPEC_07.
- [ ] An eval `source_id` appearing in a corpus split causes a loud failure.
- [ ] `gating` blocks a candidate that regresses **any single** gate metric — including LoB accuracy alone.
- [ ] A `--continue-from` candidate without cross-type regression evidence is blocked.
- [ ] No code path allows overriding a failed gate.
- [ ] Promotion decision + per-metric deltas written to the RunManifest.

---

## Current implementation (2026-09-27)

**The frozen golden eval set** (`evaluation/freeze_eval_set.py`,
`python -m orchestration.run freeze-eval-set --corpus vN [--allow-small]`):
- copies one corpus build's **test split** — golden label, metadata, every page image and OCR page — into
  `golden-eval-set/{source_id}/`, with `golden-eval-set/manifest.json` (source corpus, commit, documents
  per type and per line of business, held-out carriers). `golden.json` is written last per document, so an
  interrupted freeze leaves nothing half-copied;
- **refused a second time** (the set is the yardstick every version is compared on), and **refused when
  any type would freeze fewer than 150 documents** (arch §15.4) unless `--allow-small`: the set cannot grow
  once frozen, and at 150 documents a rate near 80% is known to about ±3 points;
- after freezing, every corpus build excludes the frozen documents and their families and splits new
  documents into train/val only (SPEC_05).

Before this, nothing read the corpus test split and nothing populated `golden-eval-set/`, so the gate had
no documents.

**The golden eval** (`evaluation/golden_eval.py::evaluate_version`): asserts the set is disjoint from the
corpus, runs every document in all three modes through `serving.pipeline.extract` with the release's bf16
calibrators, scores failures as `{}` with the error recorded, and writes the report where the gate reads
it. The gate stage produces it when no report exists.

**Validation-based scoring** (checkpoint selection, calibration): `evaluation/validation_generation.py`.
A pass in which more than **10%** of rows failed to generate is **refused**, not scored — scoring failures
as empty answers turned a broken setup into a checkpoint "chosen" at 0.0 and a calibrator fitted on
nothing. Images are localised to the pod cache first (vLLM opens paths, the rows hold Blob keys).

**Scoring corrections** (`evaluation/metrics/`):
- a value the golden label does not state is **scored wrong** (canonical targets are sparse, so a loop over
  golden paths alone never saw an invented field); empty extra values cost nothing;
- **tables are found at any depth** (`auto.vehicles`) and rows compared on **values**, not envelopes;
- **misattribution** is checked on flattened nested paths;
- **LoB** is not scored for a label that carries none (canonical policies), and the LoB metric is not
  applicable to a policy-only scope;
- field accuracy is **pooled per document reading** (source × mode), then averaged over documents — a
  60-window policy counts once, not 60 times; a window with nothing to score adds nothing;
- the classifier metric is not applicable while the corpus builds no classify rows (`common.tasks.CORPUS_TASKS`).

**Line-scoped gate**: for a scope narrowed by line of business (`personal_lines`), `evaluate_version` scores
only the frozen documents whose line is in scope. Double annotation of the frozen documents (arch §15.4)
remains a manual step before the first production gate.
