"""training.stage_data — the corpus, where ms-swift can read it, in its format.

The failures this guards were invisible to every other test, because nothing
here ever launches the trainer: the corpus rows could not be loaded by the
reader the trainer uses, and the paths the trainer was given did not exist on
the machine it runs on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.dataset_builder.build_jsonl import (
    SourceDocument,
    build_corpus,
    train_rows_by_epoch,
    write_jsonl,
)
from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_group_splits
from training.stage_data import SWIFT_IMAGE_TAG, StagingError, stage_training_data, to_swift_row

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _golden(name):
    return json.loads((FIXTURES / f"golden/{name}.golden.json").read_text(encoding="utf-8"))


def _ocr(name):
    return (FIXTURES / f"ocr/{name}_page_1.md").read_text(encoding="utf-8")


def _documents():
    """Both row shapes the corpus holds: a flat Loss Run and a windowed policy,
    with `lob` written the two ways real labels arrive — a string and a list."""
    docs = []
    for i in range(1, 11):
        docs.append(SourceDocument(
            source_id=f"lossrun_{i:04d}", doc_type="lossrun", golden_label=_golden("lossrun_0001"),
            ocr_pages=[_ocr("lossrun_0001")],
            image_paths=[f"processed/default/lossrun/lossrun_{i:04d}/page_1.png"],
            lob=["workers_comp"], tenant_id="default",
        ))
        docs.append(SourceDocument(
            source_id=f"policy_{i:04d}", doc_type="policy", golden_label=_golden("policy_0001"),
            ocr_pages=[_ocr("policy_0001")],
            image_paths=[f"processed/default/policy/policy_{i:04d}/page_1.png"],
            lob="workers_comp", tenant_id="default",
        ))
    return docs


def _seed(client):
    docs = _documents()
    groups: dict[str, list[GroupRecord]] = {}
    for d in docs:
        groups.setdefault(d.doc_type, []).append(
            GroupRecord(group_id=d.source_id, doc_type=d.doc_type, source_ids=[d.source_id])
        )
    built = build_corpus(docs, assign_group_splits(groups, seed=42))
    for epoch, rows in train_rows_by_epoch(built).items():
        client.write_text(paths.corpus_epoch_file("v1", epoch), write_jsonl(rows))
    client.write_text(paths.corpus_eval_split("v1", "val"),
                      write_jsonl(built.rows_by_split["val"]))
    for d in docs:
        for key in d.image_paths:
            client.write_bytes(key, PNG)
    return built


EPOCHS = [paths.corpus_epoch_file("v1", e) for e in (1, 2, 3, 4)]


# --------------------------------------------------------------------------
# The row format
# --------------------------------------------------------------------------

def _row():
    return {"source_id": "x", "messages": [
        {"role": "system", "content": "extract"},
        {"role": "user", "content": [
            {"type": "image", "image": "k/page_1.png"},
            {"type": "text", "text": "<page 1 of 2>\n\none"},
            {"type": "image", "image": "k/page_2.png"},
            {"type": "text", "text": "<page 2 of 2>\n\ntwo"},
        ]},
        {"role": "assistant", "content": "{}"},
    ], "lob": ["workers_comp"], "window_pages": [1, 2]}


def test_a_swift_row_is_string_content_with_one_placeholder_per_image_in_order():
    row = to_swift_row(_row(), lambda key: f"/local/{key}")
    assert set(row) == {"messages", "images"}
    assert all(isinstance(m["content"], str) for m in row["messages"])
    user = row["messages"][1]["content"]
    # Joined with nothing between: that is how the Qwen template concatenates a
    # content list, so the rendered prompt does not change.
    assert user == "<image><page 1 of 2>\n\none<image><page 2 of 2>\n\ntwo"
    assert row["images"] == ["/local/k/page_1.png", "/local/k/page_2.png"]
    assert user.count(SWIFT_IMAGE_TAG) == len(row["images"])


def test_text_containing_the_placeholder_is_refused():
    """It would be read as an image and pair every later page with the wrong text."""
    bad = _row()
    bad["messages"][1]["content"][1]["text"] = "see <image> below"
    with pytest.raises(StagingError, match="placeholder"):
        to_swift_row(bad, lambda key: key)


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------

def test_the_corpus_as_written_cannot_be_loaded_by_the_trainers_reader(client, tmp_path):
    """The regression this module exists for, pinned so it cannot quietly return."""
    pyarrow_json = pytest.importorskip("pyarrow.json")
    import pyarrow as pa

    _seed(client)
    local = tmp_path / "raw.jsonl"
    local.write_text(client.read_text(EPOCHS[0]), encoding="utf-8")
    with pytest.raises(pa.ArrowInvalid):
        pyarrow_json.read_json(local)


def test_staged_files_load_with_the_trainers_reader(client, tmp_path):
    pyarrow_json = pytest.importorskip("pyarrow.json")

    _seed(client)
    staged = stage_training_data(EPOCHS, paths.corpus_eval_split("v1", "val"), client, tmp_path)

    for local in [*staged.epoch_files, staged.val_path]:
        table = pyarrow_json.read_json(local)
        assert table.num_rows > 0
        assert set(table.column_names) == {"messages", "images"}


def test_every_image_is_a_local_file_that_exists(client, tmp_path):
    built = _seed(client)
    staged = stage_training_data(EPOCHS, None, client, tmp_path)
    # One page per document; only train documents are in the epoch files.
    assert staged.images == len({r["source_id"] for r in built.rows_by_split["train"]})
    for local in staged.epoch_files:
        for line in Path(local).read_text(encoding="utf-8").splitlines():
            for image in json.loads(line)["images"]:
                assert Path(image).is_absolute() and Path(image).read_bytes() == PNG


def test_a_missing_page_image_is_refused(client, tmp_path):
    built = _seed(client)
    trained = next(r["source_id"] for r in built.rows_by_split["train"] if r["doc_type"] == "policy")
    client.delete(f"processed/default/policy/{trained}/page_1.png")
    with pytest.raises(StagingError, match="not in Blob"):
        stage_training_data(EPOCHS, None, client, tmp_path)


def test_every_row_stores_lob_as_a_list(client):
    """Strings from the review tool, lists from the importers: one JSON column
    holding both is refused by pyarrow."""
    built = _seed(client)
    assert {type(r["lob"]) for r in built.all_rows} == {list}


# --------------------------------------------------------------------------
# train() hands ms-swift the staged files
# --------------------------------------------------------------------------

def test_a_real_run_points_ms_swift_at_local_staged_files(client, tmp_path, monkeypatch):
    from registry_utils.models import DataStats
    from training import train as T

    _seed(client)
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path))
    monkeypatch.setattr(T, "validate_all", lambda **_kw: None)
    launched = []
    monkeypatch.setattr(T, "launch", lambda config: launched.append(config))

    T.train(corpus_version="v1", out_version="v9", client=client, corpus_manifest={},
            data_stats=DataStats(train_examples=0, val_examples=0, test_examples=0),
            dry_run=False)

    (config,) = launched
    datasets = config.args["dataset"]
    assert datasets and all(Path(p).is_file() for p in datasets), datasets
    assert all(str(tmp_path) in p for p in datasets)
    assert Path(config.args["val_dataset"][0]).is_file()
