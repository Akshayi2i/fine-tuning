"""Splitting one OCR batch across GPUs (data_pipeline.ocr.run_mineru.shard_of)."""

import pytest

from data_pipeline.ocr.run_mineru import shard_of

IDS = [f"policy_{i:04d}" for i in range(1, 1703)]


def test_shards_cover_every_document_exactly_once():
    shards = [shard_of(IDS, f"{i}/4") for i in range(4)]
    covered = [sid for shard in shards for sid in shard]
    assert sorted(covered) == sorted(IDS) and len(covered) == len(set(covered))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_every_process_cuts_the_same_list_whatever_its_order():
    assert shard_of(list(reversed(IDS)), "2/4") == shard_of(IDS, "2/4")


@pytest.mark.parametrize("bad", ["4/4", "-1/4", "1-4", "x/4"])
def test_a_bad_shard_is_refused(bad):
    with pytest.raises(ValueError):
        shard_of(IDS, bad)
