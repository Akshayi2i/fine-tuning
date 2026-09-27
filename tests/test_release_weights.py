"""The weights that ship are the weights that were measured.

Merge folds the selected checkpoint into a pinned base; package uploads the real
directories rather than a placeholder JSON; training launches only where its
outputs can be found.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from training.merge import MergeError, MergePlan, merge


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _plan(tmp_path: Path, *, adapter: Path, base: str = "Qwen/Qwen3-VL-8B-Instruct@abc123"):
    return MergePlan(base_model=base, adapter=str(adapter), output_dir=str(tmp_path / "out"),
                     dtype="bf16")


def test_a_merge_needs_an_adapter_to_merge(tmp_path):
    with pytest.raises(MergeError, match="adapter_config.json"):
        merge(_plan(tmp_path, adapter=tmp_path / "root"))


def test_a_merge_refuses_a_floating_base_revision(tmp_path):
    adapter = tmp_path / "checkpoint-100"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(MergeError, match="revision"):
        merge(_plan(tmp_path, adapter=adapter, base="Qwen/Qwen3-VL-8B-Instruct@PIN_ME"))


def test_a_scoped_push_lands_where_the_manifest_points(client, tmp_path):
    from artifact_registry.transfer import push_merged_model, push_scoped_adapter

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    prefix = push_scoped_adapter(adapter, "policy", "v2", client=client)
    assert prefix == paths.scoped_adapter_dir("policy", "v2")
    assert client.read_bytes(f"{prefix}/adapter_model.safetensors") == b"weights"

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")
    assert push_merged_model(merged, "v2", client=client, scope="policy") == (
        paths.merged_model_dir("v2", scope="policy")
    )


def test_pushing_an_absent_directory_is_refused(client, tmp_path):
    from artifact_registry.transfer import TransferError, push_scoped_adapter

    with pytest.raises(TransferError):
        push_scoped_adapter(tmp_path / "missing", "policy", "v2", client=client)


def test_training_refuses_to_launch_off_the_pod(monkeypatch):
    from training import train

    monkeypatch.delenv(train.OFF_POD_ENV, raising=False)
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", "/no/such/mount")
    with pytest.raises(train.TrainingError, match="refusing to launch"):
        train.assert_on_pod()


def test_the_off_pod_override_is_explicit(monkeypatch):
    from training import train

    monkeypatch.setenv(train.OFF_POD_ENV, "1")
    train.assert_on_pod()
