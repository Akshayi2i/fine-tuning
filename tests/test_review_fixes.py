"""Fixes from the end-to-end code review, one test (or more) per finding."""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from serving.release_router import build_serving_plan
from tests.test_release_router import promote


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _calibration_file(client, release_id):
    from calibration.feature_calibrator import CalibratorSet
    from calibration.thresholds import ThresholdSet

    key = paths.release_calibrators(release_id, "bf16")
    client.write_json(key, {
        "calibrators": CalibratorSet(release_id=release_id, serving_format="bf16").as_dict(),
        "thresholds": ThresholdSet(release_id=release_id, serving_format="bf16").as_dict(),
    })
    bundle = client.read_json(paths.release_bundle(release_id))
    bundle["calibrators"] = {"bf16": key}
    client.write_json(paths.release_bundle(release_id), bundle)


def _adapter(client, scope, version):
    prefix = paths.scoped_adapter_dir(None if scope == "unified" else scope, version)
    client.write_json(f"{prefix}/adapter_config.json", {"r": 64})
    client.write_bytes(f"{prefix}/adapter_model.safetensors", b"weights")


# --------------------------------------------------------------------------
# #2 — serving goes THROUGH the release the plan chose
# --------------------------------------------------------------------------


def test_one_release_is_served_on_the_engine_weights_with_its_calibrators(client, tmp_path):
    from serving.vllm_entrypoint import load_release_runtimes

    promote(client, "release-2026.10.1", scope="policy", doc_types=["policy"], adapter="policy-v1")
    _calibration_file(client, "release-2026.10.1")
    runtimes = load_release_runtimes(build_serving_plan(client), client, adapter_root=tmp_path)
    runtime = runtimes["release-2026.10.1"]
    assert runtime.adapter is None                     # its merged model IS the engine
    assert runtime.calibrators is not None and runtime.thresholds is not None


def test_several_releases_are_each_served_as_their_own_lora(client, tmp_path):
    from serving.vllm_entrypoint import load_release_runtimes

    promote(client, "release-2026.10.1", adapter="extractor-v1")
    promote(client, "release-2026.10.2", scope="policy", doc_types=["policy"], adapter="policy-v1")
    _adapter(client, "unified", "v1")
    _adapter(client, "policy", "v1")
    runtimes = load_release_runtimes(build_serving_plan(client), client, adapter_root=tmp_path)
    assert (tmp_path / "release-2026.10.2" / "adapter_config.json").is_file()
    assert runtimes["release-2026.10.2"].adapter == str(tmp_path / "release-2026.10.2")
    assert runtimes["release-2026.10.1"].adapter == str(tmp_path / "release-2026.10.1")


def test_a_release_with_no_adapter_in_blob_fails_the_cold_start(client, tmp_path):
    from serving.vllm_entrypoint import ColdStartError, load_release_runtimes

    promote(client, "release-2026.10.1", adapter="extractor-v1")
    promote(client, "release-2026.10.2", scope="policy", doc_types=["policy"], adapter="policy-v1")
    _adapter(client, "unified", "v1")
    with pytest.raises(ColdStartError, match="no adapter"):
        load_release_runtimes(build_serving_plan(client), client, adapter_root=tmp_path)


def test_extract_generates_with_the_chosen_releases_adapter_and_calibrators(client, tmp_path):
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract
    from serving.vllm_entrypoint import load_release_runtimes

    promote(client, "release-2026.10.1", adapter="extractor-v1")
    promote(client, "release-2026.10.2", scope="policy", doc_types=["policy"], adapter="policy-v1")
    _adapter(client, "unified", "v1")
    _adapter(client, "policy", "v1")
    _calibration_file(client, "release-2026.10.1")
    plan = build_serving_plan(client)
    runtimes = load_release_runtimes(plan, client, adapter_root=tmp_path)

    backend = EchoBackend(json.dumps({"insured_name": "Rivera"}))
    model = load_model("base", client, backend_impl=backend)
    request = ExtractionRequest(
        source_id="lossrun_0001", image_paths=["x/page_1.png"], ocr_text="claims",
        known_doc_type="lossrun",
    )
    result = extract(request, model, StaticClassifier("lossrun"), None, plan=plan,
                     release_runtimes=runtimes, strict_schema=False)
    # A Loss Run: only the unified release serves it, so its LoRA is applied —
    # and its calibrators are used, so no v1 calibration was needed (None above).
    assert backend.calls[-1]["adapter"] == str(tmp_path / "release-2026.10.1")
    assert result.source_id == "lossrun_0001"


# --------------------------------------------------------------------------
# #3 — a rollback pin wins over a line-scoped release
# --------------------------------------------------------------------------


def test_a_rollback_pin_outranks_a_line_scoped_release(client):
    from common.scopes import get_scope

    promote(client, "release-2026.10.1")
    promote(client, "release-2026.10.2", scope="personal_lines", doc_types=["policy"])
    bundle = client.read_json(paths.release_bundle("release-2026.10.2"))
    bundle["lines"] = sorted(get_scope("personal_lines").lines)
    client.write_json(paths.release_bundle("release-2026.10.2"), bundle)

    unpinned = build_serving_plan(client)
    assert unpinned.release_for("policy", "homeowners").release_id == "release-2026.10.2"
    pinned = build_serving_plan(client, pins={"policy": "release-2026.10.1"})
    assert pinned.release_for("policy", "homeowners").release_id == "release-2026.10.1"
    assert "by line" in pinned.describe()          # #10: line releases are listed


# --------------------------------------------------------------------------
# #5 — a type with no test documents cannot slip past the freeze guard
# --------------------------------------------------------------------------


def test_a_type_with_no_test_documents_fails_the_freeze_guard(client):
    from evaluation.freeze_eval_set import FreezeError, freeze_eval_set

    rows = [{"source_id": f"policy_{i:04d}", "doc_type": "policy"} for i in range(200)]
    client.write_text(paths.corpus_eval_split("v1", "test"), "".join(json.dumps(r) + "\n" for r in rows))
    client.write_json(paths.corpus_manifest("v1"), {"doc_types": ["lossrun", "policy"]})
    with pytest.raises(FreezeError, match="'lossrun': 0"):
        freeze_eval_set(client, "v1")


# --------------------------------------------------------------------------
# #6 — a blank page in a text PDF does not make the document a scan
# --------------------------------------------------------------------------


@pytest.mark.parametrize("meta,expected", [
    ({"is_scanned": False, "failed_pages": [3]}, False),   # MinerU's classification wins
    ({"is_scanned": True, "failed_pages": []}, True),
    ({"failed_pages": [3]}, True),                          # older meta: the old rule
    ({"failed_pages": []}, False),
])
def test_scanned_comes_from_the_ocr_classification(meta, expected):
    from evaluation.freeze_eval_set import is_scanned

    assert is_scanned(meta) is expected


# --------------------------------------------------------------------------
# #7 — row-header and nested HTML tables
# --------------------------------------------------------------------------


def test_row_header_cells_do_not_make_body_rows_headers():
    from data_pipeline.ocr.run_mineru import count_table_rows

    table = ("<table><tr><th>Vehicle</th><th>VIN</th></tr>"
             + "".join(f"<tr><th>Vehicle {i}</th><td>VIN{i}</td></tr>" for i in range(1, 9))
             + "</table>")
    assert count_table_rows(table) == 8


def test_a_nested_table_does_not_cut_the_outer_one_short():
    from data_pipeline.ocr.run_mineru import count_table_rows

    nested = ("<table><tr><th>Loc</th><th>Detail</th></tr>"
              "<tr><td>1</td><td><table><tr><td>a</td></tr><tr><td>b</td></tr></table></td></tr>"
              "<tr><td>2</td><td>x</td></tr><tr><td>3</td><td>y</td></tr></table>")
    # outer: 3 body rows; inner: 2 rows, first treated as its header -> 1
    assert count_table_rows(nested) == 4


# --------------------------------------------------------------------------
# #8 — an explicit target table can leave a field type unenforced
# --------------------------------------------------------------------------


def test_a_type_left_out_of_an_explicit_target_table_is_reviewed():
    from calibration.thresholds import fit_thresholds

    scored = {t: [(0.99, True)] * 400 for t in ("money", "identifier")}
    result = fit_thresholds(scored, release_id="r", serving_format="bf16", targets={"money": 0.01})
    assert not result.thresholds["identifier"].enforced
    assert result.needs_review("identifier", 0.999)


def test_thresholds_round_trip_for_serving():
    from calibration.thresholds import ThresholdSet, fit_thresholds

    scored = {"money": [(0.99, True)] * 400}
    fitted = fit_thresholds(scored, release_id="r", serving_format="bf16")
    loaded = ThresholdSet.from_dict(fitted.as_dict())
    # What routing reads survives the round trip exactly (the stored error bound
    # is rounded to 4 places for display, which is not a routing input).
    for name, original in fitted.thresholds.items():
        assert loaded.thresholds[name].threshold == original.threshold
        assert loaded.thresholds[name].enforced == original.enforced
    assert loaded.needs_review("money", 0.99) == fitted.needs_review("money", 0.99)


# --------------------------------------------------------------------------
# #9 — every scope keeps classify; #11 — scoped pulls
# --------------------------------------------------------------------------


def test_every_scope_keeps_the_classify_task():
    from common.scopes import load_scopes
    from common.tasks import Task

    assert all(Task.CLASSIFY in s.tasks for s in load_scopes().values())


def test_a_scoped_merged_model_is_pulled_from_its_own_path(client, tmp_path):
    from artifact_registry.transfer import pull_merged_model, push_merged_model

    source = tmp_path / "merged"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    push_merged_model(source, "v1", client=client, scope="personal_lines")
    pulled = pull_merged_model("v1", tmp_path / "out", client=client, scope="personal_lines")
    assert (pulled / "config.json").is_file()
