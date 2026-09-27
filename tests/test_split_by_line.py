"""The volume-banded split, applied per line of business.

The band (70/18/12, 75/15/10, 80/10/10) is chosen on the document type's
volume; each line is placed by the same hash rule, too-small lines train whole,
measured lines always reach val and test, and a band crossing never moves a
trained document into evaluation.
"""

from __future__ import annotations

import pytest

from common.constants import SPLIT_RATIOS_BY_VOLUME, SplitRatio
from data_pipeline.dataset_builder.split_groups import (
    MIN_DOCS_TO_MEASURE_LINE,
    GroupRecord,
    assert_bands_only_grow_train,
    assign_group_splits,
    line_of,
)

#: A personal-lines-shaped corpus: 1920 documents, one family each, uneven lines.
PERSONAL_LINES = {
    "homeowners": 620, "personal_auto": 480, "dwelling_fire": 300, "flood": 200,
    "recreational_vehicle": 140, "personal_umbrella": 90, "motorcycle": 60,
    "classic_auto": 27, "personal_watercraft": 3,
}


def _policies(lines: dict[str, int], prefix: str = "") -> dict[str, list[GroupRecord]]:
    return {"policy": [
        GroupRecord(group_id=f"{prefix}{line}-{i}", doc_type="policy",
                    source_ids=[f"{prefix}{line}-{i}"], line=line)
        for line, n in lines.items() for i in range(n)
    ]}


def test_the_band_is_chosen_on_the_types_volume():
    assignment = assign_group_splits(_policies(PERSONAL_LINES), hold_out_carriers=False)
    assert assignment.ratios_by_doc_type["policy"] == {"train": 0.8, "val": 0.1, "test": 0.1}


def test_every_measured_line_lands_near_the_ratio():
    assignment = assign_group_splits(_policies(PERSONAL_LINES), hold_out_carriers=False)
    for line, counts in assignment.counts_by_line["policy"].items():
        total = sum(counts.values())
        if total < 200:
            continue   # small lines are dominated by sampling noise
        assert abs(counts["train"] / total - 0.8) < 0.06, (line, counts)
        assert abs(counts["test"] / total - 0.1) < 0.05, (line, counts)


def test_every_measured_line_reaches_val_and_test():
    assignment = assign_group_splits(_policies(PERSONAL_LINES), hold_out_carriers=False)
    for line, counts in assignment.counts_by_line["policy"].items():
        if PERSONAL_LINES[line] >= MIN_DOCS_TO_MEASURE_LINE:
            assert counts["val"] >= 1 and counts["test"] >= 1, (line, counts)


def test_a_line_too_small_to_measure_trains_on_everything_it_has():
    assignment = assign_group_splits(_policies(PERSONAL_LINES), hold_out_carriers=False)
    assert assignment.counts_by_line["policy"]["personal_watercraft"] == {
        "train": 3, "val": 0, "test": 0,
    }
    assert assignment.train_only_lines["policy"] == ["personal_watercraft"]


def test_types_without_a_line_split_as_before():
    groups = {"acord": [
        GroupRecord(group_id=f"a{i}", doc_type="acord", source_ids=[f"a{i}"]) for i in range(300)
    ]}
    counts = assign_group_splits(groups, hold_out_carriers=False).counts_by_doc_type["acord"]
    assert counts["train"] and counts["val"] and counts["test"]


def test_the_bands_only_ever_move_edges_up():
    assert_bands_only_grow_train()
    edges = [(r.train, r.train + r.val) for _t, r in SPLIT_RATIOS_BY_VOLUME]
    assert edges == sorted(edges)


def test_a_band_that_shrank_train_would_be_refused(monkeypatch):
    from common import constants

    monkeypatch.setattr(constants, "SPLIT_RATIOS_BY_VOLUME", (
        (200, SplitRatio(0.80, 0.10, 0.10)), (10**9, SplitRatio(0.70, 0.18, 0.12)),
    ))
    with pytest.raises(ValueError, match="DOWN"):
        assert_bands_only_grow_train()


def test_crossing_a_band_never_moves_a_trained_document_into_evaluation():
    """150 policies (70/18/12) grow to 1500 (80/10/10). Every family that trained
    at 150 still trains at 1500; movement is only ever toward train."""
    small = assign_group_splits(_policies({"homeowners": 150}), hold_out_carriers=False)
    grown = assign_group_splits(
        _policies({"homeowners": 1500}), hold_out_carriers=False,
    )
    order = {"test": 0, "val": 1, "train": 2}
    for group_id, before in small.assignment.items():
        after = grown.assignment[group_id]
        assert order[after] >= order[before], (group_id, before, after)


@pytest.mark.parametrize("lob,expected", [
    ("Homeowners", "homeowners"), (["gl", "property"], "gl+property"),
    (["property", "gl"], "gl+property"), (None, None), ([], None),
])
def test_line_keys(lob, expected):
    assert line_of(lob) == expected
