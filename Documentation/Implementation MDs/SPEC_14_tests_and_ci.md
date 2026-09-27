# SPEC 14 — Unit Tests + CI Fixtures

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: all prior specs (01–13).
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §0a/§0b (schema + LoB contracts), §5 (list-field recall), §7 (prompt identity, split leakage), §8a/§8b (MinerU pin, tenancy), §10 (collator masking), §15 (gating).

## Goal

A test suite + CI that verifies the correctness-critical pieces without GPUs or live Azure, plus small fixtures. The tests exist to catch the failure modes the architecture calls out as silent — the ones that don't show up in a loss curve.

## Deliverables

### 1. `tests/fixtures/`
- 2–3 tiny sample PDFs per active doc type (or synthetic stand-ins), MinerU-style OCR markdown, and matching golden JSON — CI-small, **containing no real PII**.
- Fixtures covering the required corpus edge cases (arch §7): a multi-page document, a scanned/low-quality page, a document with a legitimately-absent field, and a Loss Run with a variable-length `claims` array.
- Golden JSON fixtures **all carry `line_of_business`**, including at least one `null` case and coverage of more than one LoB value.
- Golden JSON fixtures carry **`field_provenance`**, with the same canonical field appearing under **two different surface labels** across fixtures (e.g. `insured_name` as "Named Insured" in one, "Applicant" in another) so alias slicing has something to slice.
- At least one **confusable co-occurrence** fixture: a policy page carrying both a named insured and a certificate holder, with the golden JSON resolving them to different canonical keys.
- Seeded synthetic logprob/generation samples for calibration + confidence tests.
- An **in-memory mock Blob backend** (no live Azure).
- A seeded fake corpus manifest with `mineru_version`, `schema_version`, `prompt_template_version` for the pinning tests.

### 2. Core tests (mirror each spec's acceptance checks)

**Highest-value — these guard silent failures:**

| Test | Guards |
|---|---|
| `test_data_collator.py` | **Label masking**: loss only on assistant JSON tokens; system/image/OCR masked to `-100` (SPEC_06). Broken masking trains the model on its own prompt and is invisible in loss curves. **Mutation-check it** — a deliberately broken masking implementation must fail this test. |
| `test_split_leakage.py` | No `source_id` crosses train/val/test; modality expansion happens **after** the split (SPEC_05). Leakage inflates every eval number the project trusts. |
| `test_inference_core.py` | `build_messages` identical across eval/serving/testing contexts; `map_field_spans` correct for scalar, `null`, and list-row fields (SPEC_07). **Guards test == prod at the primitive level.** |
| `test_test_prod_parity.py` | Testing output == serving pipeline output on the same fixture (SPEC_11/12). Since testing reuses `serving/pipeline.py`, this asserts they don't diverge. |
| `test_prompt_parity.py` | The rendered prompt at corpus-build time (SPEC_05) is **byte-identical** to the one at inference time (SPEC_07) and to `testing/prompts/*.prompt.txt` (SPEC_12), for every doc_type × modality mode. Training/inference prompt drift is the most common cause of post-fine-tuning degradation and shows up in no training metric. |

**Contract and correctness:**

| Test | Guards |
|---|---|
| `test_schema_validity.py` | Schemas load; valid examples pass; malformed/wrong-type fail (SPEC_01/08) |
| `test_lob_contract.py` | Every schema requires `line_of_business`; out-of-enum values rejected; `null` accepted; a golden label missing the key is rejected by SPEC_04; per-value LoB accuracy computed correctly by SPEC_08 |
| `test_modality_mix.py` | 50/20/30 holds; `image_only` rows omit OCR; `noisy_ocr_image` renders the same prompt as `ocr_plus_image` (SPEC_05) |
| `test_corpus_tenancy.py` | No corpus file mixes tenants — the one live tenancy rule, because corpus composition is training data (SPEC_05) |
| `test_mineru_pin.py` | A MinerU version mismatch between corpus manifest and runtime raises in dataset build, serving, and testing (SPEC_03) |
| `test_metrics.py` | Normalized match (dates/currency/names), list F1, and **list recall** correct (SPEC_08) |
| `test_gating.py` | A regression on **any single** metric blocks promotion — including LoB accuracy alone; no override path exists; a `--continue-from` candidate lacking cross-type regression evidence is blocked (SPEC_08) |
| `test_calibration.py` | Temperature + isotonic reduce ECE; params persist/reload; `apply_calibration` **raises** on missing params; list-completeness flags dropped rows even when per-value confidence is high (SPEC_09) |
| `test_run_registry.py` | `RunManifest` validates; `adapters_depending_on` + `resolve_model_version` + `diff_manifests` correct (SPEC_02) |
| `test_router.py` | Low classifier confidence → Foundation-only fallback + review flag; ACORD returns doc_type **and** form; deterministic selection (SPEC_11) |
| `test_page_router.py` | Documents over the page threshold route and record `pages_used`; short docs skip; the declarations-page conflict rule resolves duplicates (SPEC_11) |
| `test_vit_gate.py` | Fires on perception-type errors below target; **does not** fire on schema/reasoning errors; no code path enables full ViT fine-tuning (SPEC_06) |
| `test_source_id.py` | id build/parse/validate round-trips (SPEC_01) |
| `test_alias_registry.py` | Registry loads for every doc type; **every schema field has a non-empty `description`**; no string appears in both `aliases` and `confusables` for one field; the rendered prompt contains descriptions and **contains no alias strings** (SPEC_01) |
| `test_field_provenance.py` | A label whose `field_provenance` names a registered confusable is **rejected** by `export_golden_labels`; provenance round-trips through review-tool export (SPEC_04) |
| `test_confusable_metric.py` | Misattribution scored correctly on synthetic cases and distinguished from an ordinary wrong value; a regression on it alone blocks promotion (SPEC_08) |
| `test_no_runtime_aliases.py` | **No module under `serving/`, `inference_core/`, or `testing/` imports `common.aliases`** — the master §1.4 anti-pattern, enforced by an import check rather than by discipline |
| `test_command_surface.py` | `all` = `finetune` + `package` and **never** invokes extraction; a failed gate stops `finetune` before merge and `all` before `package`; `--from-stage` resumes and a completed stage re-runs as a no-op (SPEC_13) |
| `test_model_resolution.py` | `resolve_model_version("base")` returns the pinned base with no adapter; `"v2"` returns Foundation + per-type; a `staged` manifest resolves to volume paths and a `published` one to Blob paths (SPEC_02/13) |
| `test_staging_contract.py` | `finetune` writes a Blob manifest with `status: "staged"` even though weights stay on the volume; `package` flips it to `"published"` with real paths and fails loudly when the version is not staged (SPEC_13) |

### 3. CI
- `.github/workflows/ci.yml`: install `[dev]` extras, run `ruff`, `mypy`, `pytest` on every push. Mark GPU + live-Azure tests to skip so CI is **CPU-only and fast**.
- **Coverage gate on the correctness-critical modules**: collator masking, split, inference core, prompt rendering, metrics, calibration, gating, registry.
- A **secret/PII scan** over fixtures so no real document data enters the repo.

## Constraints
- Runs without GPU or live Azure (mock/stub heavy deps).
- Fixtures contain no real PII.
- Deterministic (seeded) everywhere.

## Acceptance checklist
- [ ] `pytest` passes green on CPU-only CI.
- [ ] Collator masking test **fails** when masking is deliberately broken (mutation-checked).
- [ ] Split-leakage test **fails** if expansion is moved before the split.
- [ ] `test_prompt_parity` fails when the corpus-build prompt and the inference prompt diverge by a single character.
- [ ] `test_inference_core` + `test_test_prod_parity` pass on fixtures.
- [ ] `test_lob_contract` fails when a schema drops `line_of_business`.
- [ ] `test_corpus_tenancy` fails when a corpus mixes tenants.
- [ ] `test_gating` fails if any override path is introduced.
- [ ] `test_command_surface` fails if `all` ever reaches `package` on a failed gate, or if it invokes extraction.
- [ ] `test_staging_contract` fails if `finetune` can complete without writing a manifest to Blob.
- [ ] `test_alias_registry` fails when any schema field is missing a `description`, or when an alias string leaks into a rendered prompt.
- [ ] `test_no_runtime_aliases` fails the moment a serving or inference module imports `common.aliases`.
- [ ] `tests/test_module_seams.py` passes — **the seam suite**.
- [ ] `ruff` + `mypy` clean.

**Seam tests: run both sides, never hand-write one of them.** A full-tree review found 34 defects while the suite was entirely green, and every one lived at a boundary between two modules that had each been tested against a hand-written fixture rather than against what its neighbour actually produces. Three tests passed *because* they encoded the bug they were meant to guard: one asserted a review flag string nothing emits, one asserted a dry run moves the endpoint, one asserted a dry-run pipeline can publish.

A unit test proves a module is self-consistent. Only a test that runs both sides proves they agree. The suite covers: every `GATING_METRICS` name is one `score_subset` emits (the defect that made the promotion gate unpassable); the review-queue consumer recognises the flags `list_completeness` emits; every `vllm_serving.yaml` threshold reaches a parameter `extract` accepts; both trainers honour their YAML over helper defaults; the double-annotation sample is not a subset of `train`; and the gate's verdict key differs from the eval report's.

**Fix a pair, test the pair.** The trainer tests are parameterised over *both* trainers because the same class of bug — the validation-split leak, then the argument-ordering override — was twice fixed in one of two and reported as fixed in both.

---

## Current implementation (2026-09-27)

~1,460 tests, CPU-only, no live Azure (`pytest -q`). Test files added since this spec, each guarding a
failure that was real:

| Test | Guards |
|---|---|
| `test_eval_integrity.py` | invented fields scored wrong; nested tables; per-document pooling; failure-rate refusal; image localisation; calibration features on canonical paths; engine release; carrier normalisation; digit-aware estimate; per-window noise spread |
| `test_serving_parity.py` | `lob` on the request; page/image pairing; one blank-page text; one pixel budget; **every date field in every LOB schema** formatted; no serving page threshold; prompt hash over every prompt input, refused at cold start |
| `test_freeze_eval_set.py` | freeze layout the gate reads; refused twice; refused below 100/type without `--allow-small`; a rebuild after freezing trains on none of the eval set or its families |
| `test_split_by_line.py` | band per type, placement per line; small lines train whole; measured lines reach val and test; bands only raise edges; crossing a band never moves a trained family into evaluation |
| `test_release_weights.py` | merge refusals; scoped push paths; off-pod refusal; the base found under `/workspace/models` in all three layouts |
| `test_dependencies.py` | every third-party import installed by some group; requirements files name real groups; the training pod has vLLM |
| `test_detach.py`, `test_pod_run.py` | on the pod every long entry point detaches; already-safe jobs do not re-detach; every entry point classified; the launcher never stops the pod or a run unasked; LF line endings |
| `test_gpu_only.py` | model loaders require CUDA; no `device_map="cpu"`/`"auto"`; OCR refuses CPU; pod detection without `RUNPOD_POD_ID`, and look-alikes rejected |

Contract tests also check that `.env.example` documents every variable the code reads (and nothing it
does not), and that `SPEC_ALL_COMBINED.md` is regenerated, never hand-edited
(`python -m scripts.combine_specs --write`).

**Not testable off the pod**, and left to the Phase 0 spike: real tmux across a disconnect, a real merge, the
masking encoder on ms-swift's template, vLLM 0.11 loading Qwen3-VL with the pixel budget, and a real upload.
