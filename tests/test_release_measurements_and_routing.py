"""The release step records latency, GPU memory and a routing table, and fails
when routing does (Fideon SPEC_09 handoff item 6)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.config import lob_to_layout_family
from common.scopes import get_scope
from evaluation import release_measurements
from evaluation.release_measurements import BASE_MODEL, latency_by_adapter, measurements, p95
from serving.release_router import build_serving_plan
from serving.routing_check import RoutingCheckError, plan_with_candidate, routing_table
from tests.test_personal_lines_scope import _promote_personal
from tests.test_registry import _bundle

PERSONAL = get_scope("personal_lines")


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


# --------------------------------------------------------------------------
# Latency and memory
# --------------------------------------------------------------------------

def test_p95_is_the_nearest_rank():
    assert p95(list(range(1, 101))) == 95 and p95([7.0]) == 7.0 and p95([]) is None


def test_latency_is_kept_per_adapter_one_entry_per_call():
    scored = [
        ({}, {}, {"adapter": "personal_lines-v3", "window_latencies_ms": [100, 200, 300]}),
        ({}, {}, {"adapter": "personal_lines-v3", "window_latencies_ms": [400]}),
        ({}, {}, {"adapter": None, "window_latencies_ms": [50]}),
        ({}, {}, {"adapter": "x", "error": "failed"}),
    ]
    assert latency_by_adapter(scored) == {
        BASE_MODEL: {"p95_ms": 50.0, "calls": 1},
        "personal_lines-v3": {"p95_ms": 400.0, "calls": 4},
    }


def test_measurements_say_what_they_measured(monkeypatch):
    monkeypatch.setattr(release_measurements, "peak_gpu_memory_mb", lambda: 71234.0)
    body = measurements([({}, {}, {"adapter": "a", "window_latencies_ms": [10]})])
    assert body["peak_gpu_memory_mb"] == 71234.0 and "golden eval" in body["measured_with"]


def test_no_gpu_is_no_reading(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(release_measurements.subprocess, "run", missing)
    assert release_measurements.peak_gpu_memory_mb() is None


def test_extract_records_one_latency_per_window(client):
    from tests.test_lossrun_windows_and_table_f1 import _serve, _Windows

    assert len(_serve(client, _Windows()).window_latencies_ms) == 3          # three windows
    assert len(_serve(client, _Windows(), pages=2).window_latencies_ms) == 1  # one call


def test_the_golden_eval_keeps_the_adapter_and_each_calls_latency(client, monkeypatch):
    from evaluation import golden_eval
    from evaluation.golden_eval import GoldenDocument
    from inference_core.model_runner import load_model
    from tests.test_lossrun_windows_and_table_f1 import GOLD, PAGES, _page_text, _Windows

    monkeypatch.setattr("training.stage_data.localize_keys", lambda client, keys, root: {k: k for k in keys})
    doc = GoldenDocument(source_id="lossrun_0001", doc_type="lossrun", golden=GOLD,
                         image_keys=[f"page_{p}.png" for p in range(1, PAGES + 1)],
                         page_texts={p: _page_text(p) for p in range(1, PAGES + 1)})
    model = load_model("base", client, backend_impl=_Windows())
    (_, _, metadata), = golden_eval.evaluate([doc], model, client, "/tmp", modes=("ocr_plus_image",))
    assert "adapter" in metadata and len(metadata["window_latencies_ms"]) == 3


# --------------------------------------------------------------------------
# The routing check
# --------------------------------------------------------------------------

def _candidate(release_id="release-2026.11.1", **over):
    bundle = _bundle(release_id=release_id, scope="personal_lines", doc_types=["policy"],
                     lines=sorted(PERSONAL.lines), created_at="2026-11-01T00:00:00+00:00",
                     adapter="personal_lines-v3", **over)
    return bundle


def test_every_line_is_routed_and_the_releases_own_lines_reach_it(client):
    bundle = _candidate()
    plan = plan_with_candidate(build_serving_plan(client), bundle.model_dump(mode="json"))
    rows = routing_table(plan, bundle.release_id, bundle.lines)
    assert [row["lob"] for row in rows] == sorted(lob_to_layout_family())
    by_line = {row["lob"]: row for row in rows}
    assert by_line["personal_auto"]["release_id"] == bundle.release_id
    assert by_line["personal_auto"]["adapter"] == "personal_lines-v3"
    assert by_line["agriculture_farm"]["lob_fallback_used"] and by_line["agriculture_farm"]["release_id"] is None


def test_a_line_the_release_was_trained_for_that_would_not_reach_it_fails_the_check(client):
    # An already promoted release narrowed to homeowners alone takes homeowners
    # first (narrowest lines win), so the candidate would never answer for it.
    _promote_personal(client, "release-2026.10.9")
    narrow = client.read_json(paths.release_bundle("release-2026.10.9"))
    narrow["lines"] = ["homeowners"]
    client.write_json(paths.release_bundle("release-2026.10.9"), narrow)
    bundle = _candidate()
    plan = plan_with_candidate(build_serving_plan(client), bundle.model_dump(mode="json"))
    with pytest.raises(RoutingCheckError, match="homeowners is a line release-2026.11.1 was trained for"):
        routing_table(plan, bundle.release_id, bundle.lines)


def test_the_release_step_writes_the_measurements_and_the_table_or_stops(client):
    from orchestration.pipeline_dag import PipelineError, record_release_measurements

    client.write_json(paths.eval_report("v3", scope="personal_lines"), {"release_measurements": {
        "latency_p95_ms_by_adapter": {"personal_lines-v3": {"p95_ms": 812.5, "calls": 40}},
        "peak_gpu_memory_mb": 70100.0, "measured_with": "the golden eval"}})
    ctx = SimpleNamespace(client=client, out_version="v3", scope=PERSONAL, tenant_id=None)
    bundle = _candidate()
    record_release_measurements(ctx, bundle)
    assert bundle.latency_p95_ms_by_adapter == {"personal_lines-v3": 812.5}
    assert bundle.peak_gpu_memory_mb == 70100.0 and len(bundle.routing) == len(lob_to_layout_family())

    _promote_personal(client, "release-2026.10.9")
    narrow = client.read_json(paths.release_bundle("release-2026.10.9"))
    narrow["lines"] = ["homeowners"]
    client.write_json(paths.release_bundle("release-2026.10.9"), narrow)
    with pytest.raises(PipelineError, match="is not released: routing check failed"):
        record_release_measurements(ctx, _candidate())
