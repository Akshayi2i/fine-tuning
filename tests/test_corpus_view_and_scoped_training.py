"""A scoped training run: its corpus view, its budget, its manifest, its merge.

Phase 4's acceptance criterion is that the unified path is untouched — the
existing training tests assert that and are not modified. What is new here is
everything a NARROWER run must get right:

* it reads a filtered **view** of the one corpus, not a corpus of its own;
* it spends only the sequence budget its own tasks and types need;
* it records what it covers, under its own run id and staging path;
* its merge folds ITS adapter, not whichever one is staged at that version.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.scopes import get_scope
from registry_utils.models import DataStats
from training.corpus_view import CorpusViewError, materialize
from training.train import build_manifest, build_training_config, corpus_max_length


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


EPOCH_PATHS = [f"corpus/default/v1/train/epoch_{i}.jsonl" for i in (1, 2, 3, 4)]


def _row(source_id: str, doc_type: str, epoch: int | None = None) -> dict:
    row = {"source_id": source_id, "doc_type": doc_type, "modality_mode": "ocr_plus_image"}
    if epoch is not None:
        row["epoch"] = epoch
    return row


def seed_corpus(client: BlobClient, version: str = "v1") -> None:
    """A corpus holding all three types, as the dataset build leaves it."""
    for epoch in (1, 2, 3, 4):
        rows = [
            _row("policy_0001", "policy", epoch),
            _row("policy_0002", "policy", epoch),
            _row("lossrun_0001", "lossrun", epoch),
            _row("acord_0001", "acord", epoch),
        ]
        client.write_text(
            paths.corpus_epoch_file(version, epoch),
            "".join(json.dumps(r) + "\n" for r in rows),
        )
    client.write_text(
        paths.corpus_eval_split(version, "val"),
        "".join(json.dumps(r) + "\n" for r in
                [_row("policy_0003", "policy"), _row("lossrun_0002", "lossrun")]),
    )


# --------------------------------------------------------------------------
# The corpus view
# --------------------------------------------------------------------------


def test_the_unified_scope_reads_the_corpus_itself(client):
    """Nothing is copied for unified: it covers every type, so the corpus files
    already ARE its view."""
    seed_corpus(client)
    view = materialize(get_scope("unified"), "v1", client)

    assert view.epoch_files == EPOCH_PATHS
    assert view.val_path == paths.corpus_eval_split("v1", "val")
    assert not client.exists(paths.corpus_scope_epoch_file("v1", 1, "unified") + ".copy")


def test_a_narrower_scope_reads_a_filtered_view_of_the_same_corpus(client):
    """Filtered, never rebuilt. A per-scope BUILD would draw its own split, so a
    document could be train in one scope and test in another — and the policy
    model's training documents could sit in the unified model's eval set."""
    seed_corpus(client)
    view = materialize(get_scope("policy"), "v1", client)

    assert view.epoch_files[0] == paths.corpus_scope_epoch_file("v1", 1, "policy")
    assert view.rows_by_epoch == {1: 2, 2: 2, 3: 2, 4: 2}   # two policies per epoch
    assert view.dropped_rows == 8                            # one lossrun + one acord x 4
    assert view.val_rows == 1

    written = [json.loads(line) for line in client.read_text(view.epoch_files[0]).splitlines()]
    assert {r["doc_type"] for r in written} == {"policy"}


def test_the_view_keeps_every_epoch_so_three_epochs_still_means_three_passes(client):
    seed_corpus(client)
    view = materialize(get_scope("policy"), "v1", client)

    per_epoch = [
        sorted(json.loads(line)["source_id"] for line in client.read_text(path).splitlines())
        for path in view.epoch_files
    ]
    assert len(per_epoch) == 4
    assert all(epoch == per_epoch[0] for epoch in per_epoch), "an epoch lost a document"


def test_a_scope_with_no_training_rows_is_refused(client):
    """Training on nothing produces an adapter that reports success and learned
    nothing, and every downstream metric would describe the base model."""
    for epoch in (1, 2, 3, 4):
        client.write_text(
            paths.corpus_epoch_file("v1", epoch),
            json.dumps(_row("acord_0001", "acord", epoch)) + "\n",
        )
    client.write_text(paths.corpus_eval_split("v1", "val"), "")

    with pytest.raises(CorpusViewError, match="no training rows"):
        materialize(get_scope("policy"), "v1", client)


def test_a_scope_cannot_invent_a_corpus_it_was_never_given(client):
    with pytest.raises(CorpusViewError, match="does not build one of its own"):
        materialize(get_scope("policy"), "v9", client)


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_a_scope_spends_only_the_sequence_budget_its_own_tasks_need():
    """The cap decides which card a run fits on. Charging a lossrun-only run the
    32768 policy-extraction override reserves activation memory for a document
    type it never sees."""
    assert corpus_max_length(get_scope("unified")) == 32768
    assert corpus_max_length(get_scope("policy")) == 32768     # its own override
    assert corpus_max_length(get_scope("lossrun")) == 24576     # the policy cap is not its own


def test_the_unscoped_call_still_covers_everything():
    """`corpus_max_length()` with no scope keeps its old meaning, so any caller
    that has not been given a scope yet is unchanged."""
    assert corpus_max_length() == 32768


def test_the_trainer_config_takes_its_cap_from_the_scope():
    lossrun, _ = build_training_config(
        corpus_paths=EPOCH_PATHS, output_dir="/tmp/out", scope=get_scope("lossrun")
    )
    unified, _ = build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")

    assert lossrun.args["max_length"] == 24576
    assert unified.args["max_length"] == 32768


# --------------------------------------------------------------------------
# What the run records
# --------------------------------------------------------------------------


def _manifest(scope_name: str):
    scope = get_scope(scope_name)
    _swift, recorded = build_training_config(
        corpus_paths=EPOCH_PATHS, output_dir="/tmp/out", scope=scope
    )
    return build_manifest(
        run_id=scope.run_id("v2"),
        corpus_version="v2",
        corpus_manifest={},
        training_cfg=recorded,
        data_stats=DataStats(train_examples=10, val_examples=2, test_examples=2),
        staging_path=paths.scoped_staging_adapter_dir(scope.name, "v2"),
        scope=scope,
    )


def test_a_unified_run_records_exactly_what_it_always_did():
    """The compatibility pin at the manifest level: same run id, same run_type,
    and no scope fields, so it is indistinguishable from a v2.1 manifest."""
    manifest = _manifest("unified")

    assert manifest.run_id == "extractor-v2"
    assert manifest.run_type == "unified"
    assert manifest.scope is None and manifest.doc_types == []
    assert manifest.artifacts.staging_path.endswith("adapters/foundation/v2")


def test_a_scoped_run_records_its_scope_and_its_coverage():
    manifest = _manifest("policy")

    assert manifest.run_id == "policy-v2"
    assert manifest.run_type == "scoped"
    assert manifest.scope == "policy" and manifest.doc_types == ["policy"]
    assert manifest.artifacts.staging_path.endswith("adapters/scope/policy/v2")


def test_two_scoped_runs_at_one_version_stage_to_different_directories():
    """Same version tag, different adapters. One staging path would mean the
    second run overwrites the first's weights."""
    assert _manifest("policy").artifacts.staging_path != _manifest("lossrun").artifacts.staging_path


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


def test_the_merge_folds_the_adapter_of_its_own_scope():
    """Two scoped runs share a version, so an unscoped merge would fold whichever
    adapter happened to be staged at that tag."""
    from training.merge import plan_merge

    policy = plan_merge(base_model="base@rev", version="v2", scope="policy")
    unified = plan_merge(base_model="base@rev", version="v2")

    assert policy.adapter.endswith("adapters/scope/policy/v2")
    assert policy.output_dir.endswith("merged-models/scope/policy/v2")
    assert unified.adapter.endswith("adapters/foundation/v2")
    assert unified.output_dir.endswith("merged-models/unified/v2")


def test_the_selected_checkpoint_still_wins_over_the_scope_directory():
    """§11.2 selection picks a checkpoint inside the staged run; the scope only
    decides which run that is."""
    from training.merge import plan_merge

    plan = plan_merge(
        base_model="base@rev", version="v2", scope="policy",
        selected_checkpoint="/staging/adapters/scope/policy/v2/v0-x/checkpoint-40",
    )
    assert plan.adapter.endswith("checkpoint-40")
    assert plan.output_dir.endswith("merged-models/scope/policy/v2")
