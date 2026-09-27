"""The test split becomes the frozen golden eval set, once, and stays out of training.

Before this nothing read the test split and nothing populated the eval set the
gate scores, so the gate had no documents and the test documents were held out
for nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_group_splits
from evaluation.freeze_eval_set import FreezeError, freeze_eval_set, is_frozen, manifest_key
from evaluation.golden_eval import load_golden_set
from evaluation.run_eval import eval_set_source_ids
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import make_context, seed_corpus


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


def _built_corpus(client, controller):
    from orchestration.pipeline_dag import run_stages, stages_from

    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_from("dataset_build")[:1])
    assert report.ok, report.render()
    return ctx


def _test_ids(client, ctx):
    import json

    text = client.read_text(paths.corpus_eval_split(ctx.corpus, "test", None))
    return {json.loads(line)["source_id"] for line in text.splitlines() if line.strip()}


def test_freezing_copies_the_test_split_in_the_layout_the_gate_reads(client, controller):
    ctx = _built_corpus(client, controller)
    test_ids = _test_ids(client, ctx)
    if not test_ids:
        pytest.skip("the fixture corpus drew no test document")

    manifest = freeze_eval_set(client, ctx.corpus, allow_small=True)
    assert set(manifest["source_ids"]) == test_ids
    assert eval_set_source_ids(client) == test_ids
    documents = load_golden_set(client)
    assert {d.source_id for d in documents} == test_ids
    assert all(d.image_keys and d.golden for d in documents)
    assert client.exists(manifest_key())


def test_freezing_twice_is_refused(client, controller):
    ctx = _built_corpus(client, controller)
    if not _test_ids(client, ctx):
        pytest.skip("the fixture corpus drew no test document")
    freeze_eval_set(client, ctx.corpus, allow_small=True)
    with pytest.raises(FreezeError, match="already frozen"):
        freeze_eval_set(client, ctx.corpus, allow_small=True)


def test_a_corpus_without_a_test_split_cannot_be_frozen(client):
    with pytest.raises(FreezeError, match="no test split"):
        freeze_eval_set(client, "v404", allow_small=True)
    assert not is_frozen(client)


def test_a_frozen_document_and_its_family_stay_out_of_the_corpus(client):
    from orchestration.pipeline_dag import exclude_eval_families

    client.write_json(f"{paths.golden_eval_set_dir()}/eval-1/golden.json", {})

    def doc(source_id, family):
        return SimpleNamespace(source_id=source_id, family=family, carrier=None)

    kept, excluded = exclude_eval_families(client, [
        doc("eval-1", "fam-a"), doc("renewal-of-eval-1", "fam-a"), doc("other", "fam-b"),
    ])
    assert [d.source_id for d in kept] == ["other"]
    assert excluded == ["eval-1", "renewal-of-eval-1"]


def test_once_frozen_new_documents_split_into_train_and_val_only():
    groups = {"acord": [
        GroupRecord(group_id=f"g{i}", doc_type="acord", source_ids=[f"s{i}"],
                    carrier=f"carrier-{i % 5}")
        for i in range(60)
    ]}
    assignment = assign_group_splits(groups, with_test=False)
    counts = assignment.counts_by_doc_type["acord"]
    assert counts["test"] == 0
    assert counts["train"] > 0 and counts["val"] > 0
    assert not assignment.held_out_carriers
    ratio = assignment.ratios_by_doc_type["acord"]
    assert ratio["test"] == 0.0
    assert abs(ratio["train"] + ratio["val"] - 1.0) < 1e-9


def test_a_rebuild_after_freezing_trains_on_none_of_the_eval_set(client, controller):
    import json

    from orchestration.pipeline_dag import stage_dataset_build

    ctx = _built_corpus(client, controller)
    if not _test_ids(client, ctx):
        pytest.skip("the fixture corpus drew no test document")
    frozen = set(freeze_eval_set(client, ctx.corpus, allow_small=True)["source_ids"])

    rebuilt = make_context(client, controller, corpus_version="v2")
    stage_dataset_build(rebuilt)
    in_corpus = set()
    for epoch in (1, 2, 3, 4):
        key = paths.corpus_epoch_file("v2", epoch, None)
        if client.exists(key):
            in_corpus |= {json.loads(ln)["source_id"]
                          for ln in client.read_text(key).splitlines() if ln.strip()}
    val = client.read_text(paths.corpus_eval_split("v2", "val", None))
    in_corpus |= {json.loads(ln)["source_id"] for ln in val.splitlines() if ln.strip()}
    assert in_corpus and not in_corpus & frozen
    assert not client.read_text(paths.corpus_eval_split("v2", "test", None)).strip()


def test_a_pilot_sized_set_is_not_frozen_by_accident(client, controller):
    ctx = _built_corpus(client, controller)
    if not _test_ids(client, ctx):
        pytest.skip("the fixture corpus drew no test document")
    with pytest.raises(FreezeError, match="allow-small"):
        freeze_eval_set(client, ctx.corpus)
    assert not is_frozen(client)
