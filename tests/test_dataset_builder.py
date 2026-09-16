"""SPEC_05 — splitting, modality expansion, corruption, and the corpus manifest.

The split-leakage test here is one of the four that guard silent failures: if
expansion moved before the split, every eval number in the project would be
inflated and nothing else would notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.constants import MODALITY_MODES
from data_pipeline.corpus_manifest import (
    build_manifest,
    compute_alias_coverage,
    compute_lob_coverage,
    count_confusable_examples,
)
from data_pipeline.dataset_builder.build_jsonl import (
    SourceDocument,
    build_corpus,
    expand_document,
    sample_to_target_mix,
    write_jsonl,
)
from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr
from data_pipeline.dataset_builder.split_train_val_test import (
    SplitError,
    assert_no_leakage,
    assert_single_tenant,
    assign_splits,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _documents(n: int = 12) -> list[SourceDocument]:
    golden = json.loads((FIXTURES / "golden/policy_0001.golden.json").read_text(encoding="utf-8"))
    ocr = (FIXTURES / "ocr/policy_0001_page_1.md").read_text(encoding="utf-8")
    return [
        SourceDocument(
            source_id=f"policy_{i:04d}", doc_type="policy",
            golden_label=golden, ocr_pages=[ocr],
            image_paths=[f"processed/default/policy/policy_{i:04d}/page_1.png"],
            tenant_id="default",
        )
        for i in range(1, n + 1)
    ]


# --------------------------------------------------------------------------
# Splitting — the ordering rule
# --------------------------------------------------------------------------

def test_split_covers_every_document_exactly_once():
    docs = _documents(12)
    assignment = assign_splits({"policy": [d.source_id for d in docs]})
    assert len(assignment.assignment) == 12
    assert set(assignment.assignment.values()) == {"train", "val", "test"}


def test_split_is_deterministic_under_the_same_seed():
    ids = [f"policy_{i:04d}" for i in range(1, 21)]
    assert assign_splits({"policy": ids}, seed=42).assignment == assign_splits({"policy": ids}, seed=42).assignment


def test_growth_does_not_reshuffle_documents_at_a_fixed_ratio():
    """Threshold assignment: a document's hash is fixed, so with fixed edges the
    corpus can grow without migrating anything already assigned.

    Slicing would fail this — every boundary shifts as the corpus grows, moving
    documents near it between splits.
    """
    from common.constants import SplitRatio

    ratio = SplitRatio(0.80, 0.10, 0.10)
    first = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 101)]}, ratios=ratio)
    second = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 501)]}, ratios=ratio)
    for source_id, split in first.assignment.items():
        assert second.assignment[source_id] == split


def test_crossing_a_volume_band_does_reassign_some_documents():
    """The honest limit of stability.

    Ratios scale with volume by design (arch §8), so crossing a band moves the
    thresholds and reassigns a minority of documents. Cross-version comparability
    comes from the **frozen golden eval set**, which is versioned separately and
    held constant — not from the corpus test split.
    """
    pilot = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 101)]})     # 70/18/12
    at_scale = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 301)]})  # 75/15/15

    assert pilot.ratios_by_doc_type["policy"] != at_scale.ratios_by_doc_type["policy"]
    moved = [sid for sid, split in pilot.assignment.items() if at_scale.assignment[sid] != split]
    assert moved, "crossing a band is expected to reassign some documents"
    assert len(moved) < len(pilot.assignment) * 0.25, "but only a minority of them"


def test_empty_split_is_repaired_at_small_volume():
    """At very small N a split can come up empty by chance. An empty test split
    cannot be evaluated at all, so one document is moved to fill it."""
    assignment = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 21)]})
    assert all(assignment.counts_by_doc_type["policy"][s] >= 1 for s in ("train", "val", "test"))


def test_split_ratio_scales_with_volume():
    """Pilot volume gets ~70/18/12; at scale 80/10/10 (arch §8)."""
    pilot = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 31)]})
    assert pilot.ratios_by_doc_type["policy"]["train"] == pytest.approx(0.70)

    at_scale = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 1501)]})
    assert at_scale.ratios_by_doc_type["policy"]["train"] == pytest.approx(0.80)


def test_every_split_gets_at_least_one_document():
    """A corpus with an empty test split cannot be evaluated at all."""
    assignment = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 6)]})
    counts = assignment.counts_by_doc_type["policy"]
    assert counts["train"] >= 1 and counts["val"] >= 1 and counts["test"] >= 1


def test_types_are_split_independently():
    """A global split on an unbalanced corpus can leave a type with no test
    documents, and its accuracy then silently unmeasured."""
    assignment = assign_splits({
        "policy": [f"policy_{i:04d}" for i in range(1, 21)],
        "lossrun": [f"lossrun_{i:04d}" for i in range(1, 6)],
    })
    for doc_type in ("policy", "lossrun"):
        assert all(assignment.counts_by_doc_type[doc_type][s] >= 1 for s in ("train", "val", "test"))


# --------------------------------------------------------------------------
# Expansion, and the leakage guard
# --------------------------------------------------------------------------

def test_one_document_expands_to_three_modality_rows():
    rows, _details = expand_document(_documents(1)[0], "train")
    assert len(rows) == 3
    assert {r["modality_mode"] for r in rows} == set(MODALITY_MODES)
    assert all(r["split"] == "train" for r in rows)


def test_image_only_row_carries_no_ocr_text():
    """The guarantee is "no OCR", not "no text at all": each page still carries
    its `<page N of M>` marker, which is document metadata rather than content
    read off the page."""
    rows, _ = expand_document(_documents(1)[0], "train")
    image_only = next(r for r in rows if r["modality_mode"] == "image_only")
    clean = next(r for r in rows if r["modality_mode"] == "ocr_plus_image")

    texts = [b["text"] for b in image_only["messages"][1]["content"] if b["type"] == "text"]
    assert all(t.startswith("<page ") and "\n" not in t for t in texts)

    ocr_body = "".join(
        b["text"] for b in clean["messages"][1]["content"] if b["type"] == "text"
    ).replace("<page 1 of 1>", "").strip()
    assert ocr_body and ocr_body not in "".join(texts)


def test_noisy_row_keeps_the_correct_golden_target():
    """The mismatch IS the training signal: the model can only get it right by
    reading the page (arch §5, §6)."""
    rows, _ = expand_document(_documents(1)[0], "train")
    clean = next(r for r in rows if r["modality_mode"] == "ocr_plus_image")
    noisy = next(r for r in rows if r["modality_mode"] == "noisy_ocr_image")

    assert noisy["messages"][2]["content"] == clean["messages"][2]["content"]   # same target
    clean_text = clean["messages"][1]["content"][-1]["text"]
    noisy_text = noisy["messages"][1]["content"][-1]["text"]
    assert noisy_text != clean_text                                            # different input


def test_noisy_and_clean_rows_share_one_system_prompt():
    """The noise lives in the data, not the instruction — at inference nothing
    announces that OCR is bad."""
    rows, _ = expand_document(_documents(1)[0], "train")
    clean = next(r for r in rows if r["modality_mode"] == "ocr_plus_image")
    noisy = next(r for r in rows if r["modality_mode"] == "noisy_ocr_image")
    assert clean["messages"][0] == noisy["messages"][0]


def test_no_source_id_crosses_splits():
    """THE leakage test. If expansion moved before the split, this fails."""
    docs = _documents(12)
    assignment = assign_splits({"policy": [d.source_id for d in docs]})
    result = build_corpus(docs, assignment)
    assert_no_leakage(assignment, result.all_rows)

    by_source: dict[str, set[str]] = {}
    for row in result.all_rows:
        by_source.setdefault(row["source_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_source.values())


def test_leakage_is_detected_when_it_is_injected():
    """Mutation check — the guard must actually fire."""
    docs = _documents(6)
    assignment = assign_splits({"policy": [d.source_id for d in docs]})
    result = build_corpus(docs, assignment)

    rows = list(result.all_rows)
    rows[0] = {**rows[0], "split": "test" if rows[0]["split"] != "test" else "train"}
    with pytest.raises(SplitError, match="LEAKAGE"):
        assert_no_leakage(assignment, rows)


def test_cross_tenant_mixing_is_refused():
    """The one live tenancy rule: corpus composition IS training data."""
    rows = [
        {"source_id": "policy_0001", "split": "train", "tenant_id": "broker_a"},
        {"source_id": "policy_0002", "split": "train", "tenant_id": "broker_b"},
    ]
    with pytest.raises(SplitError, match="multiple tenants"):
        assert_single_tenant(rows)


def test_rebuild_with_the_same_seed_is_byte_identical():
    """A corpus that changes between builds cannot be compared across model
    versions."""
    docs = _documents(8)
    assignment = assign_splits({"policy": [d.source_id for d in docs]})
    first = write_jsonl(build_corpus(docs, assignment, seed=42).all_rows)
    second = write_jsonl(build_corpus(docs, assignment, seed=42).all_rows)
    assert first == second


def test_sampling_moves_the_mix_toward_the_target():
    docs = _documents(30)
    assignment = assign_splits({"policy": [d.source_id for d in docs]})
    sampled = sample_to_target_mix(build_corpus(docs, assignment))

    train = [r for r in sampled.rows_by_split["train"]]
    share = {
        mode: sum(1 for r in train if r["modality_mode"] == mode) / len(train)
        for mode in MODALITY_MODES
    }
    assert share["ocr_plus_image"] > share["image_only"] > share["noisy_ocr_image"]


def test_val_and_test_keep_all_three_variants():
    """So image-only and noisy-OCR accuracy are measured on the full eval
    population, not a sample of it."""
    docs = _documents(30)
    assignment = assign_splits({"policy": [d.source_id for d in docs]})
    sampled = sample_to_target_mix(build_corpus(docs, assignment))

    for split in ("val", "test"):
        rows = sampled.rows_by_split[split]
        per_source: dict[str, set[str]] = {}
        for row in rows:
            per_source.setdefault(row["source_id"], set()).add(row["modality_mode"])
        assert all(modes == set(MODALITY_MODES) for modes in per_source.values())


# --------------------------------------------------------------------------
# Corruption
# --------------------------------------------------------------------------

def test_corruption_is_seeded_and_always_does_something():
    """A 'noisy' row identical to the clean one teaches nothing, and would
    silently dilute the arbitration signal below its intended share."""
    text = (FIXTURES / "ocr/lossrun_0001_page_1.md").read_text(encoding="utf-8")
    a = corrupt_ocr(text, "lossrun_0001", seed=42)
    b = corrupt_ocr(text, "lossrun_0001", seed=42)
    c = corrupt_ocr(text, "lossrun_0001", seed=99)

    assert a.text == b.text
    assert a.text != c.text
    assert a.was_corrupted and c.was_corrupted
    assert a.text != text


def test_corruptions_are_recorded_for_eval_slicing():
    """So ocr_arbitration_accuracy scores on documents where OCR and image
    genuinely disagree, rather than on the whole corpus."""
    text = (FIXTURES / "ocr/lossrun_0001_page_1.md").read_text(encoding="utf-8")
    result = corrupt_ocr(text, "lossrun_0001", seed=42)
    assert result.corruptions and result.details


# --------------------------------------------------------------------------
# Manifest and coverage
# --------------------------------------------------------------------------

def test_lob_under_coverage_names_the_thin_values():
    labels = [{"line_of_business": ["workers_comp"]}] * 19 + [{"line_of_business": ["property"]}]
    shares, warnings = compute_lob_coverage(labels)
    assert shares["workers_comp"] > 0.9
    assert warnings and "property" in warnings[0]


def test_alias_under_coverage_names_the_thin_labels():
    """95 documents saying 'Named Insured' and 5 saying 'Applicant' yield a model
    shaky on 'Applicant' however large the corpus is."""
    provenance = {f"policy_{i:04d}": {"insured_name": "Named Insured"} for i in range(1, 20)}
    provenance["policy_0020"] = {"insured_name": "Applicant"}
    counts, warnings = compute_alias_coverage(provenance)

    assert counts["insured_name"]["Named Insured"] == 19
    assert counts["insured_name"]["Applicant"] == 1
    assert warnings and "Applicant" in warnings[0]


def test_alias_coverage_is_quiet_when_balanced():
    provenance = {
        f"policy_{i:04d}": {"insured_name": "Applicant" if i % 2 else "Named Insured"}
        for i in range(1, 21)
    }
    _counts, warnings = compute_alias_coverage(provenance)
    assert not warnings


def test_zero_confusable_examples_is_reported_as_a_defect():
    """Without them the model learns the mapping but never the boundary."""
    labels = {"policy_0001": {"insured_name": "Rivera Fabrication LLC"}}
    count, warnings = count_confusable_examples(labels, "policy")
    assert count == 0
    assert warnings and "corpus defect" in warnings[0]


def test_confusable_co_occurrence_is_counted():
    labels = {
        "policy_0001": {
            "insured_name": "Rivera Fabrication LLC",
            "producer": "Hanover Risk Partners",
        }
    }
    count, warnings = count_confusable_examples(labels, "policy")
    assert count == 1
    assert not warnings


def test_manifest_records_every_pin_and_measurement():
    docs = _documents(12)
    assignment = assign_splits({"policy": [d.source_id for d in docs]})
    result = build_corpus(docs, assignment)

    manifest, report = build_manifest(
        corpus_version="v1",
        tenant_id="default",
        rows_by_split=result.rows_by_split,
        golden_labels_by_source={d.source_id: d.golden_label for d in docs},
        provenance_by_source={d.source_id: {"insured_name": "Applicant"} for d in docs},
        split_assignment=assignment.as_dict(),
        ocr_environment={"mineru_version": "1.4.2", "ocr_device": "cuda"},
        doc_types=["policy"],
        git_commit="deadbee",
    )

    # Reproducibility pins — a change to any forces a rebuild and retrain.
    assert manifest["mineru_version"] == "1.4.2"
    assert manifest["ocr_device"] == "cuda"
    assert manifest["prompt_template_version"]
    assert manifest["schema_versions"]["policy"]
    assert manifest["builder_git_commit"] == "deadbee"
    assert manifest["seed"] == 42

    # Composition and coverage.
    assert manifest["total_rows"] == 36
    assert set(manifest["modality_mix"]) == set(MODALITY_MODES)
    assert manifest["lob_coverage_target"] == 0.20
    assert "alias_coverage" in manifest

    # De-identification is blocked — recorded honestly, not omitted (SPEC_05 §1).
    assert manifest["deidentified"] is False
    assert manifest["image_redaction"] == "unresolved"
    assert isinstance(report.warnings, list)


def test_the_corruption_budget_is_per_document_not_per_page():
    """`corrupt_ocr` spends its budget per call, so calling it once per page
    would multiply the noise by the page count. A 20-page policy with every page
    mangled teaches "OCR is always garbage" rather than arbitration — and would
    degrade the 50% ocr_plus_image regime, which never sees noise at all."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages

    pages = [f"Policy ABC{i}234 issued 2026-01-0{i} amount 1{i},500.00" for i in range(1, 21)]
    corrupted, details = corrupt_ocr_pages(pages, "policy_0001", max_corruptions=2)

    assert len(corrupted) == len(pages)
    changed = sum(a != b for a, b in zip(pages, corrupted, strict=True))
    assert 1 <= changed <= 2, f"{changed} of 20 pages corrupted — the budget is not being honoured"
    assert len(details) <= 2


def test_a_noisy_document_is_never_identical_to_the_clean_one():
    """A noisy row that matches its clean twin teaches nothing and silently
    dilutes the 20% arbitration signal."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages

    for source_id in (f"policy_{i:04d}" for i in range(1, 15)):
        pages = ["Insured Rivera Fabrication LLC", "Policy WC-8842317-01 premium 47250.00"]
        corrupted, details = corrupt_ocr_pages(pages, source_id)
        assert corrupted != pages, f"{source_id} produced an uncorrupted noisy row"
        assert details


def test_page_corruption_is_seeded_and_reproducible():
    """A corpus that changes between builds cannot be compared across versions."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages

    pages = [f"page {i} Policy ABC1234 amount 12,500.00" for i in range(1, 6)]
    a, da = corrupt_ocr_pages(pages, "policy_0001", seed=42)
    b, db = corrupt_ocr_pages(pages, "policy_0001", seed=42)
    c, _dc = corrupt_ocr_pages(pages, "policy_0002", seed=42)

    assert a == b and da == db
    assert a != c or da != _dc, "different documents must corrupt differently"


def test_the_details_name_the_page_the_noise_landed_on():
    """SPEC_08's ocr_arbitration_accuracy slices on these; a detail that does not
    say where the change happened cannot be matched to a field."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages

    pages = [f"Policy ABC{i}234 amount 1{i},500.00" for i in range(1, 9)]
    _corrupted, details = corrupt_ocr_pages(pages, "policy_0003")
    assert all(d.startswith("page ") for d in details), details


def test_multi_character_confusions_are_reachable():
    """`rn`->`m` and `cl`->`d` are the two most realistic confusions in print,
    and scanning one character at a time left both as dead table entries — so
    the noise the corpus taught arbitration on was narrower than it looked."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import (
        CHAR_CONFUSIONS,
        _confusable_positions,
    )

    multi = [k for k in CHAR_CONFUSIONS if len(k) > 1]
    assert multi, "the table declares no multi-character confusions"
    for source in multi:
        token = f"AB{source}CD"
        assert (2, source) in _confusable_positions(token), f"{source!r} is unreachable"


def test_a_longer_confusion_wins_over_an_overlapping_shorter_one():
    """Applying a single-character rule inside `rn` would consume half of it and
    leave the other half stranded."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import _confusable_positions

    positions = _confusable_positions("Warner")
    assert (2, "rn") in positions
    starts = [i for i, _s in positions]
    assert len(starts) == len(set(starts)), "two rules claim the same position"


def test_every_confusion_entry_is_applied_at_its_full_width():
    """A multi-character source must replace all of its characters, not one."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import CHAR_CONFUSIONS

    token = "Warner"
    index, source = 2, "rn"
    corrupted = token[:index] + CHAR_CONFUSIONS[source] + token[index + len(source):]
    assert corrupted == "Wamer", corrupted
