# SPEC 05 — Dataset Builder (JSONL, Modality, Split)

> Read `SPEC_00_MASTER_CONTEXT.md` first. Dependencies: SPEC_01, SPEC_02, SPEC_04.
>
> **Architecture refs:** `finetuning-architecture-v1.md` §6 (modality dropout), §7 (dataset format, schema registry, prompt template versioning, long documents), §8 (corpus versioning, split strategy), §8a (MinerU pin in the manifest), §8b (tenant isolation per Fideon SPEC_12; **de-identification per Fideon SPEC_11 — BLOCKED, see §1**), §0b (LoB coverage).

## Goal

Compile processed docs + golden labels into versioned, chat-format JSONL training corpora — applying the 3-regime modality split and leakage-safe train/val/test splitting, with a manifest that pins everything the corpus depends on.

## Deliverables

### 1. De-identification — **BLOCKED. Do not implement.**

Arch §8b requires the Foundation LoRA to train only on Presidio de-identified data (Fideon SPEC_11). **Do not build `data_pipeline/deidentify/` until the question below is answered**, because implementing it as specified would make the fine-tuned model measurably worse.

**Why this is a training bug, not a compliance gap.** Arch §8b specifies de-identification of **text**. It says nothing about the **page images** the vision encoder reads. If the text is de-identified and the image is not:

| Regime | Share of Foundation corpus | What the model is taught |
|---|---|---|
| `image_only` | **30%** | Image shows "John Smith"; target says `PERSON_1`. **The target is not derivable from the input.** These are unlearnable examples — 30% of the corpus becomes pure hallucination pressure. |
| `ocr_plus_image` | **50%** | OCR says `PERSON_1`, image says "John Smith", target says `PERSON_1` → teaches **trust-OCR-over-image**, the exact inverse of what the projector LoRA and the 20% `noisy_ocr_image` regime exist to teach (arch §3, §6). |

**Half-de-identifying is worse than either extreme.** Two coherent resolutions, both acceptable, to be chosen by the compliance owner:

1. **Redact page images consistently with the text** — input and target then agree, and the arbitration signal survives. Costs an image-redaction pipeline (bounding-box detection + overlay) that the architecture does not currently specify.
2. **De-identify nothing; rely on tenancy plus access control** — training data keeps real values and the input/target contract stays coherent. Requires explicit sign-off that PII in `corpus/` carries the same posture as `raw-documents/`.

Until resolved: the corpus manifest records `deidentified: false` and `image_redaction: "unresolved"`, the registry-side enforcement in SPEC_02 stays suspended, and **the limitation is stated to the compliance owner rather than left implicit**. If resolution 1 is chosen this section becomes the Presidio + image-redaction module; if resolution 2, it is deleted.

**Whichever is chosen, one rule holds:** de-identification must be **consistent across the OCR text, the page image, and the target JSON**. Any inconsistency between input and target teaches the model to hallucinate.

### 2. `data_pipeline/dataset_builder/split_train_val_test.py`
- **Split at `source_id` level BEFORE any modality expansion** (arch §8, master §10). Because each source PDF generates 3 JSONL rows, splitting *after* expansion would put the same document in both `train` and `val`/`test` under a different modality mode — data leakage that inflates eval metrics without reflecting real generalization.
- Ratio scales with per-type volume (arch §8), configurable, seeded, logged:

| Volume per doc type | Split | Note |
|---|---|---|
| Pilot (~25–30/type) | ~70 / 18 / 12 | ≈17–21 train / 4–5 val / 3–4 test. **Treat metrics as directional** — the batch's job is proving the pipeline works, not measuring model quality. |
| 200–1000/type | 75/15/10 or 80/10/10 | Val/test now large enough for stable metrics |

**Every triple must sum to 1.0**, and `common.constants` asserts it at import. The splitter assigns by hash threshold, so a triple summing to 1.05 gives the test split 10% while `ratios_by_doc_type` — copied verbatim into the corpus manifest — records 15%. Every downstream reader then believes the eval population is half again as large as it is.
| 1000+/type | 80/10/10 | Target state |

- Output: a deterministic `source_id → split` assignment, saved for reproducibility.
- **Splitting is tenant-scoped** — never build a split spanning tenants.

### 3. Modality expansion — `data_pipeline/dataset_builder/build_jsonl.py`
*(`expand_document` and `sample_to_target_mix`, not a separate `modality_dropout.py`: expansion and JSONL assembly share the split assignment and the row builder, and splitting them across two modules would mean passing that state between them for no gain.)*
- For each `source_id`, generate the 3 modality variants at the mix ratios **50 / 20 / 30** (arch §6). Each variant becomes one JSONL row **in the split its `source_id` already landed in**.
- `image_only` rows omit the OCR text block entirely and use the image-only system prompt — the explicit "no OCR provided" declaration, not a silently missing field (arch §6).
- `noisy_ocr_image` rows use the **same instruction as `ocr_plus_image`** — the noise lives in the data, not the prompt.

### 4. `data_pipeline/dataset_builder/noisy_ocr_augment.py`
- For `noisy_ocr_image` rows: take the real MinerU OCR and inject **realistic** corruptions — digit↔letter confusions (O/0, l/1, S/5), merged/split table cells, dropped headers, reading-order scrambles — while keeping the **golden JSON as the correct target**. This teaches image-over-OCR arbitration (arch §5, §6).
- Corruptions parameterized and seeded. Mirror MinerU's real failure modes; **random noise teaches the wrong lesson**.
- Tag each corrupted row with the corruption types applied, so SPEC_08's `ocr_arbitration_accuracy` can score specifically on documents where OCR and image genuinely disagree.

### 5. `data_pipeline/dataset_builder/build_jsonl.py`
Orchestrates: for each tenant, split, and doc_type, assemble chat-format rows using the exact data contract (master §9):
- `system` = rendered prompt (schema + modality instruction) via **`common.prompts`** — the same renderer serving and testing use. **Never hand-type the schema per example**; inject it from the registry at build time so a schema change is a one-file edit that regenerates the whole corpus consistently (arch §7).
- `user` = image block(s) (+ OCR text unless `image_only`). **Multi-page documents produce multiple ordered `image` blocks + concatenated OCR text in one example.**
- `assistant` = golden JSON string, including `line_of_business`.
- Carry `doc_type`, `acord_form`, `modality_mode`, `source_id`, `tenant_id`, `split`, `deidentified` on every row.
- Write `corpus/{tenant_id}/v{n}/{doc_type}/{train,val,test}.jsonl` to Blob.
- Reference images/OCR by Blob URI or a training-time-resolvable path (document the choice; it must resolve identically inside a RunPod pod).

**Required edge cases the corpus must actually contain (arch §7)** — assert coverage and report counts in the manifest:
- Multi-page documents (multiple image blocks in one example)
- Rotated / skewed scans and low-quality faxes
- Handwritten annotations on otherwise digital forms
- Fields legitimately absent — must produce `null`, not a hallucination
- Repeating table rows of variable length (Loss Run claims, ACORD schedule lines)
- **Long multi-page policies**, so the page-selection strategy (SPEC_11 `page_router`) is tested rather than assumed
- **Confusable co-occurrence** *(master §1.4)* — documents where a canonical field and at least one of its registered confusables both appear on the page (a named insured **and** a certificate holder; an insurer **and** a producer), with the golden JSON resolving them to different canonical keys. Without these the model learns the mapping but never the **boundary**, and will happily return the certificate holder as `insured_name` with high confidence. Track the count in the manifest and treat a zero count as a corpus defect, not an acceptable state.

### 6. `data_pipeline/corpus_manifest.py`
Writes `manifest.json` per corpus version — this is what training and eval read to know exactly what a corpus version contains, and what pins it:
- Example counts per `doc_type` × `split` × `modality_mode`; `source_id` lists per split.
- Split seed, ratios, builder git commit.
- **`mineru_version`**, **`ocr_device`**, and per-document `preprocessing_date` (arch §8a). Device is pinned alongside version because the model learns MinerU's output conventions, and GPU and CPU MinerU may not produce identical markdown.
- **`schema_version`** and **`prompt_template_version`** (arch §7) — a change to either forces a corpus rebuild and a new training cycle.
- `tenant_ids` contributing; `deidentified: false` and `image_redaction: "unresolved"` while §1 is blocked.
- **`lob_coverage`** (arch §0b) — computed here, not as a separate module: counts `line_of_business` per doc_type and split, compares against the **≥20%-per-value target**, and writes the actual distribution. Under-coverage emits a **loud warning naming the under-represented values**; it does not block the build, because the remedy is targeted document collection, not a build-time fix.
- **`alias_coverage`** (master §1.4) — same pattern, same module: counts examples per canonical field × observed surface label, read from each label's `field_provenance` (SPEC_04). **Alias coverage matters more than document count** — 95 documents saying *Named Insured* and 5 saying *Applicant* produce a model that is shaky on *Applicant* no matter how large the corpus is. Warn loudly naming variants below a configurable floor (default: fewer than 3 source documents). Non-blocking, for the same reason as LoB.
- **`confusable_example_count`** — how many source documents satisfy the confusable co-occurrence edge case, per canonical field. **Zero is a corpus defect** and warns accordingly.
- Edge-case coverage counts (multi-page, scanned, handwritten, null-field, long-policy).

**Corpus versions are immutable once used for a training run — never edit in place.** New labeled data appends into the next corpus version.

## Constraints
- Zero split leakage — assert no `source_id` appears in more than one split.
- Zero cross-tenant mixing — assert every row in a corpus shares its `tenant_id` prefix. This is the one live tenancy rule, because corpus composition is training data.
- Every row schema-checkable (the assistant JSON validates against the doc_type schema, including `line_of_business`).
- Deterministic + seeded; re-running with the same inputs yields byte-identical corpora.
- MinerU version at build time must match the version recorded on the processed documents (SPEC_03 `assert_version_matches`).

## Page pairing and the noise budget

- `SourceDocument.ocr_pages` holds **one markdown string per page**, never a joined blob. The corpus row's user turn is interleaved per page (master §9), so a training row and a serving request — including a page-routed one — have the same structure.
- `noisy_ocr_image` corruption is budgeted **per document, not per page** (`corrupt_ocr_pages`). Calling the single-page corrupter once per page multiplies the noise by the page count: a 20-page policy would take up to 40 corruptions instead of 2, and the "nothing changed" fallback would fire on every page. That is not a louder version of the same signal — a corpus where every page is garbage teaches the model to ignore OCR entirely, degrading the 50% `ocr_plus_image` regime that never sees noise.
- Corruption details record **which page** the noise landed on, so SPEC_08's `ocr_arbitration_accuracy` can match a corruption to the field it affected.

## Acceptance checklist
- [ ] No `source_id` crosses splits (automated assertion).
- [ ] No corpus file contains rows from more than one `tenant_id`.
- [ ] Each source produces exactly 3 rows with the correct modality distribution across the dataset (50/20/30).
- [ ] `noisy_ocr_image` rows have corrupted OCR but **correct** golden JSON targets, tagged with corruption types.
- [ ] `image_only` rows contain no OCR text and use the image-only prompt; `noisy_ocr_image` renders the same prompt as `ocr_plus_image`.
- [ ] Corpus rows interleave each page image with that page's own markdown; no two pages' markdown are joined.
- [ ] The noise budget is per document: a 20-page document takes no more corruptions than a 1-page one, and no noisy row is identical to its clean twin.
- [ ] Every assistant JSON validates against its schema and contains `line_of_business`.
- [ ] `corpus_manifest` warns loudly and names under-represented LoB values below the 20% target.
- [ ] `corpus_manifest` reports `alias_coverage` per canonical field × surface label and warns on variants below the floor.
- [ ] `confusable_example_count` is reported per canonical field, and a zero count warns.
- [ ] `manifest.json` counts match actual JSONL row counts, and record `mineru_version`, `schema_version`, `prompt_template_version`, tenant list, de-id status, and edge-case coverage.
- [ ] Rebuild with the same seed produces byte-identical output.
