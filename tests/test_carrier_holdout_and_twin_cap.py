"""Carrier and twin index on rows, one carrier held out per line, at most 20 twins per seed
(Fideon SPEC_09 amendment items 4, 5 and 8; handoff item 2)."""

from __future__ import annotations

from types import SimpleNamespace

from data_pipeline.dataset_builder.split_groups import (
    MAX_TWINS_PER_SEED,
    GroupRecord,
    GroupSplitAssignment,
    assign_delivered_splits,
    cap_twins,
    held_out_carrier_per_line,
)


def _record(group, carrier, line="homeowners", size=11):
    return GroupRecord(group_id=group, doc_type="policy", source_ids=[f"{group}-{i}" for i in range(size)],
                       carrier=carrier, line=line, synthetic=True)


# --------------------------------------------------------------------------
# Carrier hold-out per line
# --------------------------------------------------------------------------

def test_the_smallest_carrier_of_each_line_is_held_out_and_single_carrier_lines_listed():
    records = [_record("a", "big co", size=30), _record("b", "small co", size=5), _record("c", "big co"),
               _record("d", "only co", line="motorcycle")]
    held, single = held_out_carrier_per_line(records, seed=42)
    assert held == {"homeowners": "small co"} and single == ["motorcycle"]


def test_a_held_out_carrier_goes_to_test_whatever_split_it_was_delivered_in():
    splits = {"a": "train", "b": "val", "c": "val", "d": "test", "e": "train"}
    groups = {"policy": [_record("a", "acme"), _record("b", "acme"), _record("c", "zenith", size=40),
                         _record("d", "zenith", size=40), _record("e", "solo", line="motorcycle")]}
    result = assign_delivered_splits(groups, splits)
    assert result.held_out_carriers_by_line["policy"] == {"homeowners": "acme"}
    assert result.assignment["a"] == "test" and result.assignment["b"] == "test"     # moved out of train and val
    assert result.assignment["e"] == "train"                                         # single-carrier line: untouched
    assert result.single_carrier_lines["policy"] == ["motorcycle"]
    assert result.moved_to_test["policy"] == 2
    assert set(result.held_out_source_ids) == {f"a-{i}" for i in range(11)} | {f"b-{i}" for i in range(11)}
    recorded = result.as_dict()
    assert recorded["held_out_carriers_by_line"] == {"policy": {"homeowners": "acme"}}


def test_the_hold_out_can_be_turned_off_and_is_skipped_once_the_eval_set_is_frozen():
    splits = {"a": "train", "b": "val", "c": "test"}
    groups = {"policy": [_record("a", "acme"), _record("b", "zenith", size=40), _record("c", "zenith", size=40)]}
    assert assign_delivered_splits(groups, splits, hold_out_carriers=False).assignment == splits
    frozen = {"a": "train", "b": "val"}
    result = assign_delivered_splits({"policy": groups["policy"][:2]}, frozen, with_test=False)
    assert result.assignment == frozen and not result.held_out_carriers_by_line


# --------------------------------------------------------------------------
# Twin cap
# --------------------------------------------------------------------------

def _doc(source_id, family, synthetic=True, mode=None):
    return SimpleNamespace(source_id=source_id, family=family, synthetic=synthetic, render_mode=mode)


def test_at_most_twenty_twins_of_one_seed_train_per_render_mode():
    assignment = GroupSplitAssignment(assignment={"f": "train", "v": "val"})
    docs = ([_doc("f-orig", "f", synthetic=False)]
            + [_doc(f"f-{i}", "f") for i in range(25)]
            + [_doc(f"f-s{i}", "f", mode="scanned") for i in range(22)]
            + [_doc(f"v-{i}", "v") for i in range(30)])
    kept = cap_twins(docs, assignment, seed=42)
    ids = {d.source_id for d in kept}
    assert MAX_TWINS_PER_SEED == 20
    assert "f-orig" in ids                                                   # real documents always stay
    assert sum(1 for i in range(25) if f"f-{i}" in ids) == 20                # digital twins capped
    assert sum(1 for i in range(22) if f"f-s{i}" in ids) == 20               # scanned twins capped apart
    assert all(f"v-{i}" in ids for i in range(30))                           # validation never capped
    assert assignment.twins_dropped == {"f": 7} and assignment.twin_cap == 20
    again = cap_twins(docs, GroupSplitAssignment(assignment={"f": "train", "v": "val"}), seed=42)
    assert {d.source_id for d in again} == ids                               # the same twins every build


# --------------------------------------------------------------------------
# Carrier and twin index travel from the bundle to the row
# --------------------------------------------------------------------------

def test_the_import_records_carrier_and_twin_index():
    from data_pipeline.ingestion.import_labeled_pdfs import twin_index

    assert twin_index({"synthetic": True, "sample": "7"}) == 7
    assert twin_index({"synthetic": True, "twin_index": 3}) == 3
    assert twin_index({"synthetic": False, "sample": "7"}) is None


def test_the_delivered_carrier_outranks_the_one_the_label_names():
    from orchestration.pipeline_dag import _document_carrier

    label = {"carrier": {"company_name": {"raw": "Label Carrier Inc", "parsed": "Label Carrier Inc", "page_ref": [1]}}}
    assert _document_carrier({"carrier": "Delivered Carrier Co"}, label) != _document_carrier({}, label)
    assert _document_carrier({}, label)


def test_every_row_carries_carrier_and_twin_index():
    from data_pipeline.dataset_builder.build_jsonl import build_corpus
    from tests.test_dataset_builder import _documents

    docs = _documents(6)
    for index, doc in enumerate(docs):
        doc.carrier, doc.twin_index = f"carrier-{index % 2}", None
    assignment = GroupSplitAssignment(assignment={d.family: "train" for d in docs[:4]}
                                      | {docs[4].family: "val", docs[5].family: "test"})
    rows = [row for split_rows in build_corpus(docs, assignment).rows_by_split.values() for row in split_rows]
    assert rows and all("carrier" in row and "twin_index" in row for row in rows)


def test_the_run_manifest_records_the_split_policy():
    from orchestration.pipeline_dag import split_policy
    from registry_utils.models import DataStats

    policy = split_policy({"held_out_carriers_by_line": {"policy": {"homeowners": "acme"}},
                           "single_carrier_lines": {"policy": ["motorcycle"]}, "twin_cap": 20,
                           "twins_dropped": {"f": 5, "g": 2}})
    stats = DataStats(train_examples=1, val_examples=1, test_examples=1, split_policy=policy)
    assert stats.split_policy["held_out_carriers_by_line"]["policy"]["homeowners"] == "acme"
    assert stats.split_policy["twins_dropped"] == 7


# --------------------------------------------------------------------------
# Evaluation reports held-out carriers apart
# --------------------------------------------------------------------------

def test_held_out_carrier_documents_are_a_reported_subset_with_their_own_match():
    from evaluation.gating import GATING_METRICS
    from evaluation.run_eval import build_report, subset_of

    assert "held_out_carrier" in subset_of({"doc_type": "lossrun", "held_out_carrier": True})
    golden = {"insured_name": "A", "policy_number": "P-1"}
    seen = {"source_id": "s1", "doc_type": "lossrun", "modality_mode": "ocr_plus_image", "page_count": 1}
    unseen = {**seen, "source_id": "s2", "held_out_carrier": True}
    metrics = build_report("v", [(golden, golden, seen), (golden, {"insured_name": "A"}, unseen)]).gate_metrics()
    assert metrics["held_out_carrier_match"] == 0.5
    assert "held_out_carrier_match" not in GATING_METRICS                    # reported, never gated
