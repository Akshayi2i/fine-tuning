# SPEC 04 — Labeling, Golden JSON, and the Day-Zero Bootstrap

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01 (schemas, alias registry, `common.normalize`, `common.aliases`), SPEC_02, SPEC_03 (MinerU output — `derive_aliases` reads it).
>
> **Architecture refs:** `finetuning-architecture-v2.1.docx` §0a (schema contract), §0b (LoB in every label), §4c (**day-zero bootstrap**), §7 "Golden JSON Generation Mechanism", §13 step 11 (feedback loop).

## Goal

Turn OCR'd documents into human-verified **golden JSON** labels via a bootstrap pre-annotation + human-review workflow, with provenance tracking — and enforce the day-zero rule that no machine draft is accepted without correction until the corpus reaches a defined size.

## Deliverables

### 1. `data_pipeline/labeling/pre_annotate.py`
Generates a first-pass ("silver") JSON draft for a document using a configurable annotator backend. **Don't hand-type JSON from scratch** — a corrected draft is both faster and more consistent than starting blank (arch §7).

| Backend | When | Guard |
|---|---|---|
| `base_qwen3vl` (**default**) | Day zero, before any fine-tuned Foundation exists | none — self-hosted |
| `own_finetuned` | Once v1 is promoted; drafts improve every cycle, so labeling cost falls instead of staying flat | resolves the promoted version via SPEC_02 |
| `external_frontier` | One-time bootstrap only, if base Qwen3-VL zero-shot is too rough to be a useful starting point | **disabled by default** — requires both `--allow-external` AND the `ALLOW_EXTERNAL_PREANNOTATION` env flag, plus a configured zero-retention/enterprise endpoint |

**PII caveat (arch §7, master §8):** insurance documents contain names, TINs/SSNs, addresses, and financials. Sending them to a third-party API may violate the compliance posture or data-processing agreements. The external backend must **refuse to run** rather than warn, unless both permissions are present. If external pre-annotation isn't permissible, use the self-hosted base model and accept rougher first-pass drafts.

The draft is **never trusted as-is** — it is only a starting point for review.

### 2. Day-zero bootstrap rule *(arch §4c — implemented inside `export_golden_labels.py`, not as its own module)*
The day-zero path exists because on day zero **no fine-tuned Foundation adapter exists yet**:

1. Use the **base** `Qwen3-VL-8B-Instruct` with the zero-shot classification prompt (SPEC_01) to identify document type **and ACORD form number**.
2. Run extraction with the **base** model using the schema-injected prompts from SPEC_01.
3. **All pre-annotation output from the base model receives 100% human review** — no draft is accepted without correction until at least **25 labeled examples per document type** exist.

Implement this as an enforced rule, not a guideline — it is a counter and a threshold, so it lives in `export_golden_labels.py` rather than a dedicated module:
- `review_requirement(doc_type)` → `"full"` while `count(golden labels for doc_type) < 25`, otherwise `"confidence_routed"` (§5 active learning below).
- `export_golden_labels` refuses to write a label marked as accepted-without-review while the requirement is `"full"`.
- The gate is **explicitly temporary**: once Foundation v1.0 is trained and promoted, it becomes the zero-shot classifier and the base model is retired from the production inference path. Record the transition in the label metadata.

### 3. `data_pipeline/labeling/derive_aliases.py` — build the registry from labeled documents

The alias registry is **derived from existing (document, golden JSON) pairs**, not hand-written. Given documents you have already labeled canonically, the surface label for each field is recoverable by alignment: the golden JSON supplies the **value**, the OCR supplies the **text**, and the label is what introduces that value on the page.

```bash
python -m data_pipeline.labeling.derive_aliases --doc-type policy \
       --source-ids all --out schemas/aliases/policy.aliases.json --propose
```

**Algorithm — value-anchored label discovery.** For each `(canonical_field, value)` in each golden JSON:

1. **Locate the value in the OCR text.** Not exact match — the golden JSON is canonical (`2026-03-31`, `12400.00`) while the page is not (`03/31/2026`, `12,400.00`). Match through **`common.normalize`** (SPEC_01), the same normalizer evaluation uses, so "found in the document" means the same thing here as "correct" does at eval time.
2. **Skip low-entropy values.** A value of `"CA"` or `"3"` matches everywhere and yields noise. Require a minimum length / uniqueness threshold; record skipped fields rather than guessing.
3. **Read backward from the match** for a label, applying layout patterns in priority order against MinerU's markdown:

   | Pattern | Example | Priority |
   |---|---|---|
   | Inline key-value | `**Named Insured:** Rivera Fabrication LLC` | highest |
   | Bold label, adjacent value | `**Policy Number** WC-8842317-01` | high |
   | Table row label | `\| Applicant \| Rivera Fabrication LLC \|` | high |
   | Table column header | header cell above the value's column | medium |
   | Stacked form layout | label on one line, value on the next | medium |
   | Dotted leaders | `Applicant .......... Rivera Fabrication LLC` | medium |

4. **Normalize the candidate label** — strip punctuation and leaders, collapse whitespace, title-case — so `NAMED INSURED:` and `Named insured` converge.
5. **Aggregate across the corpus** — cluster candidates per canonical field and count supporting documents.

**Deriving confusables, which is the part you cannot hand-write reliably.** While walking the document, also collect **every** label→value pair found, including ones matching no canonical field. Then for each canonical field, a label is a **confusable candidate** when it is unmapped, its value is type-compatible with the canonical field's value (both organisation names, both dates, both currency), and it co-occurs in the same document. That is precisely how *Certificate Holder* surfaces as a confusable for `insured_name` — same shape of value, same page, different entity. Evidence-derived confusables beat guessed ones, because they are the distractors your real documents actually contain.

**Output — a proposal with evidence, never a silent overwrite:**

```json
{
  "insured_name": {
    "aliases": [
      {"label": "Named Insured", "documents": 41, "confidence": 0.98,
       "pattern": "inline_kv", "examples": ["policy_0003", "policy_0007"]},
      {"label": "Applicant", "documents": 5, "confidence": 0.91,
       "pattern": "stacked_form", "examples": ["policy_0044"]}
    ],
    "confusables": [
      {"label": "Certificate Holder", "documents": 23, "basis": "unmapped_type_compatible"}
    ],
    "unresolved": [
      {"source_id": "policy_0061", "reason": "value_not_found_in_ocr"}
    ]
  }
}
```

`--propose` writes to `*.aliases.proposed.json` for review; promoting to the live registry is an explicit confirm step. Treat this as **high-quality candidate generation, not ground truth** — a human confirms once, which is minutes of work against a blank-page alternative measured in hours.

**Three things this gives you beyond the registry itself:**

- **Retroactive `field_provenance`.** The same alignment fills in the provenance map (§5) for documents already labeled — so per-alias eval slicing works on your existing corpus without re-annotating anything.
- **`alias_coverage` before training.** You learn that 41 documents say *Named Insured* and 5 say *Applicant* while there is still time to go collect more *Applicant* documents, rather than discovering it from a disappointing eval.
- **A free QA pass over the golden labels.** Every `unresolved` entry means the value is nowhere in the document — so either the label is wrong or OCR failed on that page. Both are worth knowing, and neither is visible any other way.

**Honest limits.** Values appearing in several places (an insured name in both header and signature block) produce ambiguous anchors — the tool reports the ambiguity rather than picking. Checkbox and derived fields have no text anchor and are skipped. Heavily OCR-mangled values will not match and land in `unresolved`. None of these are silent: every skip and ambiguity is in the report.

### 4. `data_pipeline/labeling/review_tool/`
Integration config for an external labeling tool (Label Studio or Argilla) OR a lightweight custom review UI:
- Presents the page image(s) + OCR + draft JSON side-by-side. A reviewer with insurance domain knowledge corrects every field.
- **Surfaces each field's semantic gloss** (the schema `description`, SPEC_01) next to the field being reviewed, and the alias registry behind it. This is the labeling rulebook: without it, three reviewers will disagree about whether a document's *Applicant* is `insured_name`, and that inconsistency trains directly into the model as noise.
- **Records the observed surface label** for each canonical field it fills — see `field_provenance` below. This is one extra click per field and it is what makes per-alias diagnosis possible later.
- **`line_of_business` is a mandatory review field on every document type** (arch §0b) — the reviewer must set it to a valid LOB enum value or explicitly to `null`. It cannot be left unset.
- Support **double-annotation** on a configurable sample (default 10–20%): a second reviewer labels independently, disagreements are adjudicated, and an **inter-annotator agreement** score is produced. This measures how much noise/ambiguity exists in the labeling process itself — useful context when eval scores plateau below 100%, since some of that ceiling is human disagreement, not model error.
- Provide a task-export/import adapter so labels round-trip cleanly.

### 5. `data_pipeline/labeling/export_golden_labels.py`
- Writes verified labels to `golden-labels/{tenant_id}/{doc_type}/{source_id}/golden.json` and `label_metadata.json`.
- **`label_metadata.json` records provenance, not just the label** (arch §7 step 5): reviewer id, review date, which annotator backend produced the draft, base-model-vs-finetuned draft source, double-annotated flag, agreement score, review requirement in force at time of labeling. This matters later for debugging — if a document type's accuracy regresses, you want to check whether the *labels* for that batch were lower quality rather than only suspecting the model.
- **`field_provenance` — the observed surface label per canonical field** (master §1.4). Golden JSON keys are always canonical; this records what the document actually said:

  ```json
  "field_provenance": {
    "insured_name":  "Applicant",
    "policy_number": "Policy No.",
    "carrier":       "Insurer"
  }
  ```

  This is what lets SPEC_08 slice accuracy **per surface variant** — the difference between "field accuracy is 0.87" and "we score 0.94 on *Named Insured* and 0.61 on *Applicant*, go find more Applicant documents."

  For documents labeled **before** provenance capture existed, `derive_aliases.py` (§3) backfills this map by alignment. New labels capture it directly from the review tool; historic ones are recovered. Either way no document needs re-annotating.

- **Validates every `golden.json` against its Fideon SPEC_00 schema (SPEC_01) before writing** — reject invalid labels (arch §0a). Validation includes:
  - schema conformance (types, required keys, null representation),
  - **presence of `line_of_business`** with a valid enum value or explicit `null`,
  - `acord_form` set for ACORD documents (arch §4b two-level requirement),
  - **`field_provenance` does not name a registered confusable** for that field — a label claiming `insured_name` was found as "Certificate Holder" is rejected outright. This is the single highest-value annotation check in the system, because it catches the exact mistake that trains the model to conflate distinct entities.

**New surface forms.** When a reviewer meets a phrasing not in the registry, they append it to `schemas/aliases/{doc_type}.aliases.json`. That is a **registry edit, not a schema edit** — it triggers no corpus rebuild (master §7). Only editing a field's `description` does.
- Idempotent; re-export overwrites only with a new `label_metadata` version.

### 6. `data_pipeline/labeling/active_learning.py`
Once a model version exists (arch §7 step 6, §13 step 11):
- Run the promoted model over new unlabeled docs, and use the **calibrated confidence** (SPEC_09) to route: high-confidence extractions get a light spot-check, low-confidence fields get full manual review.
- Emits a prioritized review queue ordered by ascending confidence, with per-field flags rather than whole-document flags where possible.
- Also surfaces **list-field row-completeness flags** (SPEC_09) — a Loss Run with a row-count mismatch goes to full review regardless of per-value confidence.
- This is what makes labeling cost drop over successive corpus versions instead of staying flat.
- **Disabled while `review_requirement` is `"full"`** — confidence routing is meaningless before there are 25 labels/type.

## Constraints
- External pre-annotation OFF unless explicitly permitted by flag **and** env.
- Golden labels must be schema-valid and must carry `line_of_business`.
- Day-zero: 100% human review until ≥25 labeled examples per doc type.
- Store provenance for every label.
- Labels are tenant-scoped; never mix tenants in a review batch.
- No PII in logs.

## Acceptance checklist
- [ ] `pre_annotate` produces a draft with the self-hosted backend; the external backend **refuses** without both the flag and the env permission.
- [ ] `review_requirement` returns `"full"` below 25 labels/type and `"confidence_routed"` at/above it.
- [ ] `export_golden_labels` rejects a label missing `line_of_business`, one with an out-of-enum LoB value, and one failing schema validation.
- [ ] An ACORD label without `acord_form` is rejected.
- [ ] A label whose `field_provenance` names a registered confusable (`insured_name` found as "Certificate Holder") is **rejected**.
- [ ] `field_provenance` is captured for every filled field and round-trips through the review tool export.
- [ ] Appending a new surface form to the alias registry does not invalidate an existing corpus version.
- [ ] `derive_aliases` recovers `Named Insured` and `Applicant` as aliases of `insured_name` from a fixture set where each appears under a different layout pattern.
- [ ] It matches through `common.normalize`, so a golden `2026-03-31` anchors to a page reading `03/31/2026`.
- [ ] It proposes `Certificate Holder` as a confusable from a document where that label carries a type-compatible value mapping to no canonical field.
- [ ] It skips low-entropy values rather than guessing, and lists them in the report.
- [ ] A golden value absent from the OCR lands in `unresolved` — never silently dropped, never guessed.
- [ ] `--propose` writes `*.aliases.proposed.json` and never overwrites the live registry.
- [ ] Running it over already-labeled documents backfills `field_provenance` without re-annotation.
- [ ] Review workflow exports a corrected golden JSON that passes schema validation.
- [ ] Double-annotation path produces an agreement score on the sampled subset.
- [ ] `label_metadata.json` captures full provenance including draft backend and the review requirement in force.
- [ ] `active_learning` orders a queue by ascending confidence, surfaces row-completeness flags, and refuses to run while `review_requirement` is `"full"`.
