"""SPEC_02 — Blob layout, access rules, and the run registry.

Runs entirely against :class:`InMemoryBackend`: no Azure account, no network, no
GPU. The in-memory backend enforces the same write-once and container-isolation
rules as the real one, so passing here means passing the actual contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import (
    AccessDeniedError,
    BlobClient,
    ImmutableBlobError,
    InMemoryBackend,
)
from registry_utils import query_registry as Q
from registry_utils import write_run_manifest as W
from registry_utils.models import (
    Artifacts,
    DataStats,
    Dependencies,
    Promotion,
    RunManifest,
    TrainingConfig,
)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def ingestion_client(client: BlobClient) -> BlobClient:
    return BlobClient(
        backend=client._backend, container="main", raw_container="raw", context="ingestion"
    )


def _deps(**over) -> Dependencies:
    base = dict(
        base_model="qwen3-vl-8b-instruct@a1b2c3d",
        corpus_version="corpus/v3",
        code_git_commit="deadbee",
        mineru_version="1.4.2",
        schema_version="1.0.0",
        prompt_template_version="1.0.0",
    )
    base.update(over)
    return Dependencies(**base)


def _tc(**over) -> TrainingConfig:
    base = dict(
        lora_rank=16, lora_alpha=32, learning_rate=7e-5, epochs=4,
        gradient_accumulation_steps=16, effective_batch_size=16,
        target_modules=["q_proj"], resolution_cap_px=1792, max_seq_len=8192, seed=42,
    )
    base.update(over)
    return TrainingConfig(**base)


_DS = DataStats(train_examples=100, val_examples=20, test_examples=20)


def _adapter(run_id: str, doc_type: str, foundation: str) -> RunManifest:
    return RunManifest(
        run_id=run_id, run_type="per_type_adapter", doc_type=doc_type,
        dependencies=_deps(foundation_version=foundation),
        training_config=_tc(), data_stats=_DS,
    )


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def test_tenant_prefix_applied_only_to_document_layers():
    assert paths.is_tenant_scoped(paths.corpus_split("v2", "policy", "train"))
    assert paths.is_tenant_scoped(paths.raw_pdf("acord", "acord_0001"))
    # Shared artifacts carry no tenant data and must never be prefixed.
    assert not paths.is_tenant_scoped(paths.adapter_dir("foundation", "v2"))
    assert not paths.is_tenant_scoped(paths.registry_index())


def test_tenant_defaults_but_is_still_extractable():
    """Single-tenant build: reserved prefix, not a required argument."""
    assert paths.tenant_of(paths.corpus_split("v2", "policy", "train", "broker_a")) == "broker_a"
    assert paths.tenant_of(paths.adapter_dir("foundation", "v2")) is None


def test_staging_paths_are_absolute():
    """They are a real mount point, not blob keys. A relative path would write to
    the working directory and silently miss the volume."""
    assert paths.staging_adapter_dir("foundation", "v2").startswith("/")
    assert paths.staging_merged_model_dir("v2").startswith("/")


def test_staging_mirrors_the_blob_layout():
    """So ``package`` copies rather than translates."""
    assert paths.staging_adapter_dir("foundation", "v2").endswith(paths.adapter_dir("foundation", "v2"))


def test_invalid_inputs_are_rejected_early():
    with pytest.raises(paths.PathError, match="invalid version"):
        paths.adapter_dir("foundation", "2")           # missing the v
    with pytest.raises(paths.PathError, match="unknown doc_type"):
        paths.corpus_split("v1", "invoice", "train")
    with pytest.raises(paths.PathError, match="unknown quantization format"):
        paths.quantized_model_dir("v1", "q3_k_s")
    with pytest.raises(paths.PathError, match="1-based"):
        paths.processed_page("policy", "policy_0001", 0, "png")
    with pytest.raises(paths.PathError, match="not per-doc_type"):
        paths.adapter_dir("foundation", "v1", "policy")


# --------------------------------------------------------------------------
# Access rules
# --------------------------------------------------------------------------

def test_raw_documents_unreachable_from_training_context(client, ingestion_client):
    """raw-documents/ is the only unredacted-PII layer (arch §18a)."""
    key = paths.raw_pdf("acord", "acord_0001")
    ingestion_client.write_bytes(key, b"%PDF fake")
    with pytest.raises(AccessDeniedError, match="may not access"):
        client.read_bytes(key)


def test_raw_documents_are_write_once(ingestion_client):
    """A corrected document becomes a NEW source_id, so historical runs stay
    reproducible against the bytes they actually trained on."""
    key = paths.raw_pdf("acord", "acord_0001")
    ingestion_client.write_bytes(key, b"%PDF original")
    with pytest.raises(ImmutableBlobError, match="write-once"):
        ingestion_client.write_bytes(key, b"%PDF corrected")


def test_raw_documents_cannot_be_casually_deleted(ingestion_client):
    key = paths.raw_pdf("acord", "acord_0001")
    ingestion_client.write_bytes(key, b"%PDF original")
    with pytest.raises(ImmutableBlobError, match="compliance action"):
        ingestion_client.delete(key)


def test_raw_and_main_containers_are_actually_separate(client, ingestion_client):
    key = paths.raw_pdf("acord", "acord_0001")
    ingestion_client.write_bytes(key, b"x")
    backend = client._backend
    assert backend.exists("raw", key)
    assert not backend.exists("main", key)


def test_corpus_artifacts_round_trip(client):
    key = paths.corpus_manifest("v1")
    client.write_json(key, {"mineru_version": "1.4.2", "schema_version": "1.0.0"})
    assert client.read_json(key)["mineru_version"] == "1.4.2"


# --------------------------------------------------------------------------
# Manifest validation
# --------------------------------------------------------------------------

def test_manifest_requires_a_pinned_base_revision():
    with pytest.raises(ValueError, match="pin a revision"):
        _deps(base_model="qwen3-vl-8b-instruct")


def test_per_type_adapter_must_record_its_foundation():
    """Otherwise the Foundation-bump work list becomes a manual audit (arch §12)."""
    with pytest.raises(ValueError, match="foundation_version"):
        RunManifest(
            run_id="lossrun-adapter-v1", run_type="per_type_adapter", doc_type="lossrun",
            dependencies=_deps(), training_config=_tc(), data_stats=_DS,
        )


def test_vit_training_must_be_lora_never_full_fine_tune():
    """arch §3 — full FT risks the pretrained OCR ability image-only depends on."""
    with pytest.raises(ValueError, match="add a ViT LoRA"):
        _tc(vit_trainable=True, vit_method="frozen")


def test_published_artifacts_must_point_somewhere():
    with pytest.raises(ValueError, match="points nowhere|no Blob path"):
        Artifacts(status="published")


def test_promoted_run_must_record_who_and_when():
    with pytest.raises(ValueError, match="promoted"):
        RunManifest(
            run_id="foundation-v1", run_type="foundation", status="promoted",
            dependencies=_deps(), training_config=_tc(), data_stats=_DS,
        )


# --------------------------------------------------------------------------
# Registry behaviour
# --------------------------------------------------------------------------

def test_manifest_written_to_blob_even_while_weights_are_staged(client):
    """The rule that stops a reclaimed volume erasing a training run (master §12a)."""
    manifest = _adapter("lossrun-adapter-v2", "lossrun", "foundation-v2")
    assert manifest.artifacts.status == "staged"
    key = W.write_manifest(manifest, client)
    assert client.exists(key)
    assert client.read_json(key)["artifacts"]["status"] == "staged"


def test_index_replaces_rather_than_duplicates_a_run(client):
    """A run's row changes as it moves trained -> evaluated -> promoted. Two rows
    would make the index ambiguous exactly when answering 'what is live?'."""
    manifest = _adapter("lossrun-adapter-v2", "lossrun", "foundation-v2")
    W.write_manifest(manifest, client)
    manifest.status = "evaluated"
    W.write_manifest(manifest, client)
    rows = [r for r in Q.list_runs(client) if r["run_id"] == "lossrun-adapter-v2"]
    assert len(rows) == 1
    assert rows[0]["status"] == "evaluated"


def test_adapters_depending_on_returns_the_revalidation_work_list(client):
    for dt in ("acord", "policy", "lossrun"):
        W.write_manifest(_adapter(f"{dt}-adapter-v2", dt, "foundation-v2"), client)
    W.write_manifest(_adapter("lossrun-adapter-v1", "lossrun", "foundation-v1"), client)

    assert Q.adapters_depending_on("foundation-v2", client) == [
        "acord-adapter-v2", "lossrun-adapter-v2", "policy-adapter-v2",
    ]
    assert Q.adapters_depending_on("foundation-v1", client) == ["lossrun-adapter-v1"]


def test_resolve_base_returns_the_pinned_model_with_no_adapter(client):
    """`base` is the shared path for the pilot baseline, day-zero pre-annotation,
    and `extract --model base` (SPEC_13)."""
    resolved = Q.resolve_model_version("base", client)
    assert resolved["kind"] == "base"
    assert resolved["foundation_adapter"] is None
    assert "@" in resolved["base_model"]


def test_staged_version_resolves_to_the_volume_not_blob(client):
    """So a model can be spot-checked with `extract` before `package` runs."""
    W.write_manifest(
        RunManifest(run_id="foundation-v2", run_type="foundation",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS),
        client,
    )
    resolved = Q.resolve_model_version("v2", client, doc_type="lossrun")
    assert resolved["from_staging"] is True
    assert resolved["foundation_adapter"].startswith("/")


def test_published_version_resolves_to_blob(client):
    manifest = RunManifest(
        run_id="foundation-v3", run_type="foundation",
        dependencies=_deps(), training_config=_tc(), data_stats=_DS,
        artifacts=Artifacts(status="published", adapter_weights="adapters/foundation/v3"),
    )
    W.write_manifest(manifest, client)
    resolved = Q.resolve_model_version("v3", client)
    assert resolved["from_staging"] is False
    assert not resolved["foundation_adapter"].startswith("/")


def test_quantized_artifact_refused_while_staged(client):
    W.write_manifest(
        RunManifest(run_id="foundation-v2", run_type="foundation",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS),
        client,
    )
    with pytest.raises(Q.RegistryQueryError, match="still staged"):
        Q.resolve_model_version("v2", client, quant_format="q5_k_m")


def test_unknown_version_fails_loudly(client):
    with pytest.raises(Q.RegistryQueryError, match="no Foundation run"):
        Q.resolve_model_version("v99", client)


def test_diff_surfaces_what_would_explain_a_regression(client):
    """arch §12 — did the corpus change? the Foundation? MinerU? a hyperparameter?"""
    W.write_manifest(
        RunManifest(run_id="lossrun-adapter-v1", run_type="per_type_adapter", doc_type="lossrun",
                    dependencies=_deps(foundation_version="foundation-v1",
                                       corpus_version="corpus/v2", mineru_version="1.3.0"),
                    training_config=_tc(epochs=3), data_stats=_DS),
        client,
    )
    W.write_manifest(_adapter("lossrun-adapter-v2", "lossrun", "foundation-v2"), client)

    diff = Q.diff_manifests("lossrun-adapter-v1", "lossrun-adapter-v2", client)
    assert diff["dependencies.corpus_version"] == {"a": "corpus/v2", "b": "corpus/v3"}
    assert diff["dependencies.foundation_version"]["b"] == "foundation-v2"
    assert diff["dependencies.mineru_version"]["b"] == "1.4.2"
    assert diff["training_config.epochs"] == {"a": 3, "b": 4}

    # Dependencies first: a corpus or Foundation change explains a regression far
    # more often than a metric delta does.
    assert Q.summarize_diff(diff)[0].startswith("dependencies.")


def test_a_failed_gate_cannot_be_promoted(client):
    """The gate has no override path by design (arch §13)."""
    manifest = _adapter("lossrun-adapter-v2", "lossrun", "foundation-v2")
    manifest.promotion = Promotion(beat_previous_on_all_gates=False, failed_gates=["list_field_recall"])
    W.write_manifest(manifest, client)
    with pytest.raises(W.RegistryError, match="no override path"):
        W.mark_promoted(manifest, client, promoted_by="someone")


def test_mark_published_flips_status_and_clears_staging(client):
    manifest = _adapter("lossrun-adapter-v2", "lossrun", "foundation-v2")
    manifest.artifacts.staging_path = "/runpod-volume/staging/adapters/lossrun/v2"
    W.write_manifest(manifest, client)

    published = W.mark_published(manifest, client, adapter_weights="adapters/lossrun/v2")
    assert published.artifacts.status == "published"
    assert published.artifacts.staging_path is None


def test_sweep_runs_are_recorded_but_excluded_from_default_listings(client):
    """Recorded so they are first-class (arch §11a); hidden so they do not swamp
    a listing. Sweeps are deferred until after the pilot."""
    manifest = RunManifest(
        run_id="sweep-lr-01", run_type="per_type_adapter", doc_type="lossrun",
        dependencies=_deps(foundation_version="foundation-v2"),
        training_config=_tc(), data_stats=_DS,
        is_sweep_run=True, sweep_id="phase1_lr", sweep_phase="lr",
    )
    W.write_manifest(manifest, client)

    assert "sweep-lr-01" not in [r["run_id"] for r in Q.list_runs(client)]
    assert "sweep-lr-01" in [r["run_id"] for r in Q.list_runs(client, include_sweeps=True)]


def test_latest_promoted_finds_the_serving_version(client):
    W.write_manifest(
        RunManifest(run_id="foundation-v2", run_type="foundation", status="promoted",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS,
                    promotion=Promotion(promoted_by="ci", promoted_at=datetime.now(UTC))),
        client,
    )
    assert Q.latest_promoted(client, "foundation") == "foundation-v2"
