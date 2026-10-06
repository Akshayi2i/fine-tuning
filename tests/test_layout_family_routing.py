"""Routing by layout family, with the base-model fallback (Fideon SPEC_06 §9a,
SPEC_09 §2.1a; handoff item 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.config import CONFIG_DIR, ROOT, UnknownLineError, layout_family_of, lob_to_layout_family
from serving.release_router import UnservedDocType, build_serving_plan
from tests.test_personal_lines_scope import _promote_personal
from tests.test_release_router import promote

LOB_SCHEMAS = CONFIG_DIR / "canonical schema" / "LOB Schema"


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


# --------------------------------------------------------------------------
# One table
# --------------------------------------------------------------------------

def test_every_line_this_repository_has_a_schema_for_has_one_family_entry():
    from common.schemas import _CANONICAL_NOT_REGISTERED

    # classic_auto.json is on disk but read as personal auto, so it is no line.
    lines = {p.stem for p in LOB_SCHEMAS.glob("*.json")
             if not p.stem.startswith("_") and p.stem not in _CANONICAL_NOT_REGISTERED}
    assert set(lob_to_layout_family()) == lines


def test_the_keys_equal_the_l1_codes_once_the_registry_is_here():
    registry = ROOT / "config" / "lob_registry.yaml"
    if not registry.exists():
        pytest.skip("config/lob_registry.yaml (the 48 L1 codes, Fideon SPEC_06 §9a) is not in this "
                    "repository; layout_families.yaml is keyed by its own line names until it is")
    import yaml

    codes = set((yaml.safe_load(registry.read_text(encoding="utf-8")) or {}).get("lobs") or [])
    assert set(lob_to_layout_family()) == codes


def test_the_router_and_lob_schema_map_agree_once_it_carries_families():
    schema_map = ROOT / "config" / "lob_schema_map.yaml"
    if not schema_map.exists():
        pytest.skip("lob_schema_map.yaml (layout_family column) is not in this repository")
    import yaml

    rows = yaml.safe_load(schema_map.read_text(encoding="utf-8")) or {}
    for line, row in rows.items():
        if isinstance(row, dict) and "layout_family" in row:
            assert layout_family_of(line) == row["layout_family"], line


@pytest.mark.parametrize("line,family", [
    ("personal_auto", "personal_lines"),
    ("homeowners", "personal_lines"),
    ("agriculture_farm", "property_pkg"),
    ("wc", "casualty_fleet"),
    ("directors_officers", "exec_specialty"),
    ("group_accident_sickness", "benefits_misc"),
])
def test_lines_map_to_their_family(line, family):
    assert layout_family_of(line) == family


def test_an_unknown_line_is_never_given_a_family():
    with pytest.raises(UnknownLineError, match="never routed silently"):
        layout_family_of("title")


# --------------------------------------------------------------------------
# The router
# --------------------------------------------------------------------------

def test_personal_auto_goes_to_the_personal_lines_adapter(client):
    _promote_personal(client)
    routed = build_serving_plan(client).route("policy", "personal_auto")
    assert routed.release.scope == "personal_lines" and routed.layout_family == "personal_lines"
    assert not routed.lob_fallback_used


def test_a_family_without_a_trained_adapter_goes_to_the_base_model(client):
    _promote_personal(client)
    routed = build_serving_plan(client).route("policy", "agriculture_farm")
    assert routed.release is None and routed.lob_fallback_used and routed.layout_family == "property_pkg"


def test_an_unknown_or_missing_line_raises_unless_the_fallback_is_chosen(client):
    _promote_personal(client)
    plan = build_serving_plan(client)
    for lob in ("title", None):
        with pytest.raises(UnservedDocType, match="allow_lob_fallback"):
            plan.route("policy", lob)
        routed = plan.route("policy", lob, allow_fallback=True)
        assert routed.release is None and routed.lob_fallback_used


def test_a_package_across_families_goes_to_the_base_model(client):
    _promote_personal(client)
    routed = build_serving_plan(client).route("policy", ["homeowners", "gl"])
    assert routed.release is None and routed.lob_fallback_used and routed.layout_family is None


def test_until_it_is_retired_an_unrestricted_release_answers_before_the_base_model(client):
    promote(client, "release-2026.9.1", scope="unified")
    _promote_personal(client)
    plan = build_serving_plan(client)
    assert plan.route("policy", "gl").release.scope == "unified"
    assert plan.route("policy", "homeowners").release.scope == "personal_lines"
    assert plan.route("acord").release.scope == "unified"            # other types as before


def test_serving_reads_a_fallback_policy_with_the_base_model_against_the_fallback_schema(client):
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import LOB_FALLBACK_FLAG, ExtractionRequest, PipelineError, extract

    _promote_personal(client)
    plan = build_serving_plan(client)
    backend = EchoBackend(json.dumps({}))
    model = load_model("base", client, backend_impl=backend)

    def request(lob, allow=False):
        return ExtractionRequest(source_id="policy_0001",
                                 image_paths=["processed/default/policy/policy_0001/page_1.png"],
                                 page_texts={1: "Declarations"}, known_doc_type="policy", known_lob=lob,
                                 allow_lob_fallback=allow)

    result = extract(request("agriculture_farm"), model, StaticClassifier("policy"), None,
                     plan=plan, strict_schema=False)
    assert result.route_info["lob_fallback_used"] and result.route_info["adapter"] is None
    assert result.route_info["layout_family"] == "property_pkg" and LOB_FALLBACK_FLAG in result.review_flags
    assert all(call["adapter"] is None for call in backend.calls)
    with pytest.raises(PipelineError, match="allow_lob_fallback"):
        extract(request("title"), model, StaticClassifier("policy"), None, plan=plan)
    assert extract(request("title", allow=True), model, StaticClassifier("policy"), None,
                   plan=plan, strict_schema=False).route_info["lob_fallback_used"]


def test_the_endpoint_passes_the_callers_choice_through():
    from serving.vllm_entrypoint import build_request

    payload = {"source_id": "p1", "image_paths": ["page_1.png"], "doc_type": "policy", "allow_lob_fallback": True}
    assert build_request(payload).allow_lob_fallback is True
    assert build_request({**payload, "allow_lob_fallback": False}).allow_lob_fallback is False


def test_layout_families_lists_no_line_twice():
    import yaml

    config = yaml.safe_load(Path(CONFIG_DIR / "layout_families.yaml").read_text(encoding="utf-8"))
    listed = [line for entry in config["families"].values() for line in entry["lobs"]] + config["no_family"]
    assert len(listed) == len(set(listed))
