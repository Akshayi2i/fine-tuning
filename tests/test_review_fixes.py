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


# ==========================================================================
# Second end-to-end review
# ==========================================================================


# #1 — MinerU's own device setting, not just a GPU being present
def test_mineru_config_must_say_cuda(tmp_path, monkeypatch):
    from data_pipeline.ocr.mineru_config import MinerUConfigError, assert_on_cuda, set_cuda

    config = tmp_path / "magic-pdf.json"
    monkeypatch.setenv("MINERU_TOOLS_CONFIG_JSON", str(config))
    with pytest.raises(MinerUConfigError, match="no MinerU config"):
        assert_on_cuda()
    config.write_text(json.dumps({"device-mode": "cpu", "models-dir": "/m"}), encoding="utf-8")
    with pytest.raises(MinerUConfigError, match="device-mode 'cpu'"):
        assert_on_cuda()
    set_cuda()
    assert_on_cuda()
    assert json.loads(config.read_text(encoding="utf-8")) == {
        "device-mode": "cuda", "models-dir": "/m", "formula-config": {"enable": False}}


def test_mineru_formula_recognition_must_be_off(tmp_path, monkeypatch):
    """Policies carry no equations, and MinerU 1.3's UniMERNet fails under
    transformers 4.57 on every page; the whole corpus is OCR'd with it off."""
    from data_pipeline.ocr.mineru_config import MinerUConfigError, assert_on_cuda, set_cuda

    config = tmp_path / "magic-pdf.json"
    monkeypatch.setenv("MINERU_TOOLS_CONFIG_JSON", str(config))
    template = {"device-mode": "cuda", "formula-config": {"mfd_model": "yolo_v8_mfd",
                                                          "mfr_model": "unimernet_small", "enable": True}}
    config.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(MinerUConfigError, match="formula recognition on"):
        assert_on_cuda()
    set_cuda()
    assert_on_cuda()
    kept = json.loads(config.read_text(encoding="utf-8"))["formula-config"]
    assert kept == {"mfd_model": "yolo_v8_mfd", "mfr_model": "unimernet_small", "enable": False}


def test_the_engine_refuses_a_mineru_configured_for_cpu(tmp_path, monkeypatch):
    from data_pipeline.ocr.run_mineru import MinerUEngine, OcrError

    config = tmp_path / "magic-pdf.json"
    config.write_text(json.dumps({"device-mode": "cpu"}), encoding="utf-8")
    monkeypatch.setenv("MINERU_TOOLS_CONFIG_JSON", str(config))
    monkeypatch.setattr("common.gpu.require_cuda", lambda what: "NVIDIA H100")
    with pytest.raises(OcrError, match="run on the CPU"):
        MinerUEngine().process(b"%PDF", device="cuda", max_long_side_px=800)


# #7 — a mixed PDF is OCR'd whole
def test_a_page_without_a_text_layer_sends_the_document_through_ocr(monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    from data_pipeline.ocr.run_mineru import MinerUEngine
    from tests.test_mineru_engine import CONTENT, _fake_mineru

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "DECLARATIONS page with a real text layer on it, typed")
    # A scanned endorsement: a page image and no text.
    scan = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 80))
    scan.clear_with(200)
    doc.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), pixmap=scan)
    doc.new_page().insert_text((72, 72), "SCHEDULE page with a real text layer on it, typed")
    calls = _fake_mineru(monkeypatch, CONTENT, scanned=False)   # MinerU alone would say "text"
    pages = MinerUEngine().process(doc.tobytes(), device="cuda", max_long_side_px=800)
    assert calls["mode"] == "ocr"
    assert [p.scanned for p in pages] == [False, True, False]


def test_a_blank_page_does_not_make_a_digital_pdf_a_scan():
    """Nothing is on it to OCR. One blank page sent a whole typed policy through
    OCR mode and into the scanned eval subset."""
    pymupdf = pytest.importorskip("pymupdf")
    from data_pipeline.ocr.run_mineru import text_layer_pages

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "DECLARATIONS page with a real text layer on it, typed")
    doc.new_page()                                              # blank: no text, image or drawing
    assert text_layer_pages(doc.tobytes()) == [True, True]


# #6 — a serving pod without MinerU does not fail the OCR pin
def test_a_serving_pod_without_mineru_starts(monkeypatch):
    from data_pipeline.ocr import mineru_version
    from serving.vllm_entrypoint import assert_ocr_pin

    monkeypatch.setattr(mineru_version, "get_mineru_version", lambda: mineru_version.UNKNOWN)
    assert_ocr_pin({"mineru_version": "1.3.12", "ocr_device": "cuda"})      # does not raise


def test_a_mineru_version_mismatch_is_a_cold_start_error(monkeypatch):
    from data_pipeline.ocr import mineru_version
    from serving.vllm_entrypoint import ColdStartError, assert_ocr_pin

    monkeypatch.setattr(mineru_version, "get_mineru_version", lambda: "1.2.0")
    monkeypatch.setattr(mineru_version, "assert_version_matches",
                        lambda m: (_ for _ in ()).throw(mineru_version.MinerUVersionError("1.2.0 != 1.3.12")))
    with pytest.raises(ColdStartError, match="1.2.0"):
        assert_ocr_pin({"mineru_version": "1.3.12", "ocr_device": "cuda"})


# #2 — only policies count as policy lines; ACORD lines stay in the enum coverage
def test_policy_line_counts_hold_policies_only(client):
    from orchestration.runpod_controller import LocalBackend, RunPodController
    from tests.test_orchestration import make_context, seed_corpus

    seed_corpus(client)
    for sid, doc_type in (("acord_0001", "acord"), ("lossrun_0001", "lossrun")):
        key = paths.label_metadata(doc_type, sid)
        client.write_json(key, {**client.read_json(key), "lob": ["general_liability"]})
    controller = RunPodController(backend=LocalBackend(), volume_id="v", git_commit="abc1234")
    from orchestration.pipeline_dag import stage_dataset_build

    stage_dataset_build(make_context(client, controller))
    manifest = client.read_json(paths.corpus_manifest("v1"))
    assert "general_liability" not in manifest["policy_line_counts"]
    assert set(manifest["policy_line_counts"]) <= {"wc", "commercial_auto"}


# #5 — an interrupted freeze is resumable, not permanent
def test_an_interrupted_freeze_is_not_frozen_and_can_be_resumed(client):
    from evaluation.freeze_eval_set import FreezeError, freeze_eval_set, is_frozen, partial_freeze

    root = paths.golden_eval_set_dir()
    client.write_json(f"{root}/policy_0001/metadata.json", {"frozen_from_corpus": "v1"})
    client.write_json(f"{root}/policy_0001/golden.json", {})
    assert not is_frozen(client)
    assert partial_freeze(client) == {"policy_0001": "v1"}
    with pytest.raises(FreezeError, match=r"interrupted freeze .* from corpus \['v1'\], not v2"):
        freeze_eval_set(client, "v2", allow_small=True)


def test_the_gate_refuses_a_partially_frozen_set(client):
    from evaluation.golden_eval import GoldenEvalError, evaluate_version

    client.write_json(f"{paths.golden_eval_set_dir()}/policy_0001/golden.json", {})
    from common.scopes import get_scope

    with pytest.raises(GoldenEvalError, match="interrupted"):
        evaluate_version(client, object(), version="v1", corpus_version="v1", scope=get_scope("policy"))


# #8 — narrowing keeps the line restriction
def test_narrowing_a_line_scope_keeps_its_lines():
    from common.scopes import get_scope, narrow

    narrowed = narrow(get_scope("personal_lines"), ["policy"])
    assert narrowed.lines == get_scope("personal_lines").lines
    assert not narrowed.covers_lob("gl")


# #3 #4 #9 — audit: the importer's layout, malformed page_ref, unreadable PDF
def test_the_audit_flags_layouts_the_importer_cannot_read(tmp_path):
    pytest.importorskip("pymupdf")
    from data_pipeline.audit import audit_folder
    from tests.test_data_audit import _document

    _document(tmp_path / "homeowners", "doc1")           # nested one level too deep
    (tmp_path / "loose.pdf").write_bytes(b"%PDF-1.4")    # loose in the input folder
    report = audit_folder(tmp_path, scope=None)
    details = " ".join(f.detail for f in report.blockers)
    assert "in subfolders below homeowners/" in details
    assert "directly in the input folder" in details


def test_a_scalar_page_ref_is_one_blocker_not_a_crash(tmp_path):
    pytest.importorskip("pymupdf")
    import copy

    from data_pipeline.audit import audit_folder
    from tests.test_data_audit import GOLDEN, _document

    golden = copy.deepcopy(GOLDEN)
    golden["policy"]["policy_number"]["page_ref"] = 1
    _document(tmp_path, "doc", golden=golden)
    report = audit_folder(tmp_path, scope=None)       # does not raise
    assert any("page_ref must be a list" in f.detail for f in report.blockers)


def test_an_unreadable_pdf_is_one_cause_not_a_row_per_value(tmp_path):
    from data_pipeline.audit import audit_folder

    folder = tmp_path / "doc"
    folder.mkdir()
    (folder / "policy.pdf").write_bytes(b"not a pdf at all")
    (folder / "golden.json").write_text(json.dumps({"policy": {"policy_number": {
        "raw": "WC-1", "parsed": "WC-1", "page_ref": [1]}}}), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"lob": "homeowners"}), encoding="utf-8")
    report = audit_folder(tmp_path, scope=None)
    assert any("cannot be opened" in f.detail for f in report.blockers)
    assert report.values == []
