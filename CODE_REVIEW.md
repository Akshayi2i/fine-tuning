# Code review — findings and status

Two reviews:

- **Review 1** — the full tree, 13 packages, 78 files.
- **Review 2** — the eight files touched by the page-pairing change.

**All 44 findings are fixed.** 661 tests passing, ruff and mypy clean across 79
source files. Every status here was verified by running the code, not by reading
it.

The suite was green before any of this work started. Every one of these findings
lived in a path no test covered — which is the whole lesson: a green suite was
never evidence of correctness, only evidence that the covered paths still behave.

---

## What was wrong, by theme

### Absence read as success — 11 findings

The largest cluster by far, and the one worth carrying forward: a guard written
to reject one specific falsy value rather than to require the affirmative one.
Every docstring stated the policy correctly; every implementation inverted it.

| Was | Now |
|---|---|
| `promotion_gate({}, None)` **passed** — `unmeasured` required a baseline, so a candidate with zero measured metrics sailed through, on v1 of all versions | An unmeasured metric is unmeasured, baseline or not |
| `cross_type_evidence={"acord": {}}` satisfied the mandatory cross-type requirement | An empty entry is the absence of evidence, and blocks |
| `mark_promoted` rejected only `is False`, so a never-gated manifest (`None`) promoted cleanly | Requires the affirmative gate result |
| `stated_count or …` read a stated count of **0** as absent; a document stating zero claims against four hallucinated rows scored **confidence 1.0** | 0 is a value; a wholly unsupported list scores 0.0 |
| `min(1.0, extracted/reference)` reported over-extraction as perfect | Symmetric — a surplus is as wrong as a shortfall |
| `if thin and len(labels) > 1` skipped the thinnest coverage possible: a field seen under exactly one label | Any thin label warns |
| `assert_modality_mix` imported `MODALITY_MIX`, took a `tolerance`, used **neither** | Compares the train split; warns when the corpus is too small for the tolerance to be reachable |
| `is_balanced` was mathematically unreachable — 5 LoB values against a 0.20 floor is exactly the uniform share | Floor capped at ¾ of the uniform share; 21/20/20/20/19 passes, 90/5/3/1/1 does not |
| `pre_annotate_batch` caught only `PreAnnotationError`; one bad document aborted the batch | Catches per-document failure, re-raises the compliance refusal |
| `vllm_entrypoint.handler` caught only `ServingError`, so the two most likely failures escaped as opaque platform errors | Returns a structured error for anything |
| An empty-dict pilot report counted as a completed experiment, so a truncated file could yield `go` | An empty report is a missing one |

### Completion checks cheaper than the work they guard — 3 findings

- `_is_ingested` returned `True` if *any* document had ever been ingested, so a second `finetune --input newdir` trained on the previous batch. **Fixed by deleting the check** — ingestion is checksum-deduped and idempotent, so always running it cannot be wrong in the direction that loses 500 documents.
- `_is_quantized` checked `doc_types[0]` alone; a half-finished quantize read as complete and `stage_push` published GGUFs that were never produced.
- `stage_push` cleared the staging volume even when **zero** manifests were published — deleting the only copy of the weights and reporting success.

### Nothing had ever run — the 4 blockers

Not subtle bugs. First-ten-minutes-of-execution bugs, which survived because
every GPU-boundary module was verified against its *interface* rather than its
behaviour.

- **`to_cli` dropped every `False` and space-joined every list.** `--train-vit` never reached ms-swift while the manifest recorded `vit_trainable=True`; six corpus paths became one nonexistent filename; `lora_target_modules` matched nothing, so PEFT would have attached zero adapters.
- **The validation split was passed as training data.** `corpus_paths` included both `train` and `val` and assigned both to `--dataset`; there was no `val_dataset` anywhere in the repo. `metric_for_best_model` then selected on memorised documents, and the gate read that number. **Fixed in `train_foundation.py` only — see the correction below.**
- **`run.py` never supplied `metrics_provider`**, so every real `finetune` trained a Foundation and three adapters on an A100 and then aborted at the gate. It now defaults to reading the frozen-eval-set report, with the baseline resolved from the promoted version's own report.
- **`cold_start` never set `calibration`** — the endpoint reported "warm" and then rejected every request — and it read `resolved["adapters"]`, a key `ResolvedModel` does not define, so every document served Foundation-only with a review flag.

### Silent data corruption — 8 findings

- `SPLIT_RATIOS_BY_VOLUME`'s mid band summed to **1.05**, so test got 10% while the manifest recorded 15%. Now sums to 1.0, **asserted at import**.
- `_find_run` matched version tags by substring, so `v1` matched `v10` — returning a staging path for a model published in Blob. Now matches the tag exactly.
- A render-only `ocr_meta.json` satisfied the OCR idempotence check, so a rendered document was never OCR'd and the CLI reported it processed.
- `count_table_rows` discounted one header per **page**, over-reporting any multi-table page and raising false row-completeness flags that force full manual review.
- Per-page OCR corruption used `str.replace`, damaging the first matching substring anywhere — including inside a longer token — while the recorded detail named a change that happened elsewhere.
- `_find_value_positions` used `line.find`, so a value repeated across cells resolved to the leftmost column and derived the neighbouring field's label as an alias.
- `download_dir("corpus/…/v1")` also pulled `v10` into a mangled path.
- Bare `"raw-documents"` was classified non-raw, bypassing the access guard.

### Metrics that measured the wrong thing — 6 findings

- The confusable filter was **computed and never applied**, so every coincidental value collision entered a gating metric.
- One document-wide row count was compared against **every** list field, guaranteeing a flag on any document with two — a perfect policy was flagged twice on every request.
- List rows were indexed into a dict, collapsing duplicates: two claim rows sharing a number scored **recall 0.5** on a byte-perfect extraction, and all-null keys collapsed 8 rows to 1.
- `_row_key` raised `AttributeError` on a list of scalars, destroying every other document's score in the run.
- `run_eval` emitted `list_field_f1`; the gate and the manifest both read `field_f1_list_fields`.
- The zero-shot baseline overwrote its measured schema-validity rate with a crash rate.

### Resource and lifecycle — 5 findings

- `collect_logs` ran **before** `terminate` in the pod teardown `finally`, so a log-fetch failure leaked a billing A100 — the exact failure the `finally` existed to prevent.
- `deploy_endpoint` recorded history before the dry-run check, so a preview corrupted the rollback target.
- `retry.backoff_seconds` was read by nothing; retries fired inside the same throttle window. `pod.image` and `staging_mount` were dead config too.
- `update_registry_index` was an unguarded read-modify-write; concurrent runs dropped each other's rows from the index `adapters_depending_on` reads.
- The per-type adapter dropped `load_best_model_at_end` and `save_total_limit`, so the best checkpoint was found and then discarded.

### Crashes on realistic input — 5 findings

- An ACORD with no form number produced the `(acord, None)` pair `schema_key` refuses — the whole extraction crashed instead of falling back.
- A page-routed document with one empty-OCR page aborted the **entire** document, and the always-include-first-page rule made a poorly-scanned page 1 the most likely one selected.
- A bare JSON array parsed and span-mapped cleanly, then died several layers down.
- `find_assistant_span` excluded the end-of-turn token, so EOS was never supervised — generation runs to `max_new_tokens`, reported downstream as dropped rows — and `verify_batch` rejected a correct ms-swift batch.
- A span past a truncated label row raised a bare `IndexError` instead of the `MaskingError` the module exists to produce.

### Reachability — 2 findings

- `rn`→`m` and `cl`→`d`, the two most realistic print confusions, were dead entries: the scanner only tested single characters.
- `swift_early_stopping_args` had no callers, so configured early stopping never reached the trainer.

---

## What this cost, and what to keep

Three fixes required changing a **test** that encoded the wrong behaviour:

- `test_package_pushes_all_three_artifact_classes_and_publishes` asserted the Foundation manifest owns `quantized_formats` — the exact defect.
- `test_rollback_needs_somewhere_to_roll_back_to` relied on dry runs recording deployment history.
- The cascade helper promoted manifests that were never gated.

A test that asserts the bug is worse than no test, because it defends it.

Two fixes needed a **size guard** rather than a threshold, because the check as
specified was arithmetically unsatisfiable: the modality mix below 50 rows, and
the LoB floor at five enum values. In both cases the honest move was to say so in
a warning rather than silently skip the check.

**The pattern to apply from here: assert the affirmative, never the specific
negative.** Eleven of these 44 were that one mistake.

---

## Correction — the val leak was fixed in one of two trainers

The entry above claimed the validation-split leak was closed. It was closed in
`train_foundation.py`. `train_adapter.py` still read:

```python
"dataset": [
    paths.corpus_split(corpus_version, doc_type, split) for split in ("train", "val")
],
```

so every per-type adapter trained on its own validation split, selected its best
checkpoint on memorised documents, and handed that score to the promotion gate.
A later review of the training stack found it. Both trainers now emit `dataset`
and `val_dataset` separately, and
`test_neither_trainer_puts_the_validation_split_in_the_training_set` asserts it
of **both**, because fixing one and reporting both is the mistake that let this
stand.

## The training-stack review — 13 further findings

Run against `train_foundation`, `train_adapter`, `data_collator`, `vit_gate`,
`sweep`, `early_stopping`, and the training YAMLs. `data_collator.py` and both
YAMLs came out clean.

**The adapter's Foundation was attached with the wrong argument.**
`resume_from_checkpoint` means *continue this run*: ms-swift restores optimizer
state and the completed `global_step`, so a fresh 3-epoch adapter run resumes at
the end of the Foundation's schedule and trains zero steps — and tries to load
rank-64 weights into a rank-16 config on the way. Now `adapters`.

**The ViT gate decided from nothing, and called it `hold`.** Three defects that
compounded:

- `evaluate_from_report` read `eval_metrics` / `subset_counts` / `error_records`.
  `EvalReport.as_dict()` emits `gate_metrics` / `by_doc_type`. Every well-formed
  report therefore produced None accuracies and zero document counts.
- A None accuracy skipped the volume check, fell through to the accuracy check,
  and was skipped again by `value is not None` — returning `hold` with a
  rationale stating both subsets met the 0.80 target when neither had been
  measured. Unmeasured is now `insufficient_data`, as everywhere else.
- The perception share was computed over **every** error while accuracy was
  gated on the image-only and scanned subsets. An `ocr_plus_image` majority —
  the largest subset, and where perception is least implicated — drowned a real
  image-only failure under the 60% threshold. `modality_mode` and `is_scanned`
  were recorded on every `ErrorRecord` and used by nothing.

Also in the gate: the misplacement check read only top-level fields, so a loss
run's `claims[]` confusables — an amount lifted from the wrong row, the canonical
misplacement error — reached the near-miss check and were classified as misread
characters. That is the one classification that decides whether a Foundation
retrain is justified.

The rest:

- Both trainers wrote a manifest with the default `status="trained"` **before**
  launching ms-swift, so a pod that OOM'd at step 40 left a registry entry
  asserting a trained adapter over a staging path holding nothing. `RunStatus`
  gains `training`; `launch_and_record` flips it to `trained` or `failed`.
- `_foundation_checkpoint` swallowed a registry lookup failure and returned a
  constructed staging path, so an explicit `--foundation` with no manifest
  trained against a directory nobody had checked. Now refused.
- `continue_from` was documented as a checkpoint path and looked like a run-id.
  A run-id there finds no checkpoint, trains from base, and records a lineage
  that never happened — which is then read as evidence when deciding how much
  regression testing a promotion needs. `assert_checkpoint_path` refuses it.
- `assert_enough_data({})` passed. An empty count is a caller that did not count.
- `phase3_rank.yaml` was the only sweep phase with no `per_type` grid, so
  `--adapter-type per_type` raised at phase 3 after six runs were spent.
- `**swift_early_stopping_args(...)` was unpacked *after* the explicit keys, so
  the helper's defaults silently overrode the YAML's `metric_for_best_model`.
  Unpacked first now, so explicit wins.
- `eval_loss or 0.0` recorded a loss of zero — a perfect model — for an
  evaluation that reported none.

Same pattern as the first 44, in a new place: **the gate's `hold` and the
manifest's `trained` were both affirmative claims made from absent evidence.**

---

## Still true, and unchanged by any of this

The code is now internally consistent and defensible. It has still never run
against a GPU, a real corpus, or a live Azure account. Phase 0's dependency spike
and the pilot protocol are what produce the first evidence about *quality*; this
work only removed the defects that would have corrupted that evidence.
