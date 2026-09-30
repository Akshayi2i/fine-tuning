"""The base-versus-checkpoint comparison in scripts/diagnose_generation.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.validation_generation import ValidationGeneration

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_generation.py"


def _script():
    spec = importlib.util.spec_from_file_location("diagnose_generation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_base_and_checkpoint_are_scored_on_the_same_rows(tmp_path, monkeypatch):
    """Whether fine-tuning beat the model it started from: the base runs with no
    adapter, the checkpoint with its own, on identical rows."""
    import evaluation.validation_generation as V
    import inference_core.model_runner as M

    adapters = []

    def generate(rows, model, adapter=None):
        adapters.append(adapter)
        return [ValidationGeneration(row=r, golden={"a": 1}, extraction={"a": 1 if adapter else 2},
                                     error=None if adapter else "JSONDecodeError: x",
                                     failure_kind=None if adapter else "output") for r in rows]

    monkeypatch.setattr(V, "generate_validation", generate)
    monkeypatch.setattr(V, "score_generations", lambda gens, model_version: {
        "field_normalized_match": sum(g.extraction.get("a") == 1 for g in gens) / len(gens)})
    monkeypatch.setattr(M, "load_model", lambda tag, client: object())
    monkeypatch.setattr(M, "release_model", lambda model: None)

    out = tmp_path / "report.json"
    rc = _script()._against_base([{"source_id": "p1"}, {"source_id": "p2"}], None,
                                 SimpleNamespace(checkpoint="/ckpt", out=str(out)))
    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0 and adapters == [None, "/ckpt"]
    assert report["base"]["metrics"]["field_normalized_match"] == 0.0
    assert report["checkpoint"]["metrics"]["field_normalized_match"] == 1.0
    assert report["base"]["unusable_json"] == 2 and report["checkpoint"]["unusable_json"] == 0


def test_a_saved_comparison_can_be_rescored_without_the_gpu(tmp_path, monkeypatch):
    """Scoring questions (list recall read 0.0 under the old row matcher) must
    not cost another pass over the validation set on the pod."""
    import evaluation.validation_generation as V
    import inference_core.model_runner as M

    golden = {"forms_and_endorsements": [
        {"form_number": {"raw": "HO 00 03", "parsed": "HO 00 03", "page_ref": [1]},
         "form_title": {"raw": "Homeowners 3", "parsed": "Homeowners 3", "page_ref": [1]}}]}
    got = {"forms_and_endorsements": [
        {"form_number": {"raw": "HO-0003", "parsed": "HO-0003", "page_ref": [2]},
         "form_title": {"raw": "Homeowners", "parsed": "Homeowners", "page_ref": [2]}}]}
    row = {"source_id": "p1", "doc_type": "policy", "lob": "homeowners", "sections": "arrays",
           "messages": [{"role": "user", "content": [{"type": "image", "image": "x"}]}]}

    monkeypatch.setattr(V, "generate_validation", lambda rows, model, adapter=None: [
        ValidationGeneration(row=r, golden=golden, extraction=got) for r in rows])
    monkeypatch.setattr(M, "load_model", lambda tag, client: object())
    monkeypatch.setattr(M, "release_model", lambda model: None)

    script = _script()
    out = tmp_path / "report.json"
    script._against_base([row], None, SimpleNamespace(checkpoint="/ckpt", out=str(out)))
    saved = json.loads(out.read_text(encoding="utf-8"))["checkpoint"]["generations"][0]
    assert saved["row"]["messages"][0]["content"] == [{"type": "image"}]   # images as a count

    rescored = script.rescore(str(out))
    assert rescored["checkpoint"]["list_field_recall"] == 1.0


def test_table_rows_can_be_compared_from_a_saved_report(tmp_path):
    script = _script()
    env = lambda v: {"raw": v, "parsed": v, "page_ref": [1]}  # noqa: E731
    report = {"checkpoint": {"generations": [{
        "row": {"source_id": "p1", "lob": "homeowners", "messages": []},
        "golden": {"forms_and_endorsements": [{"form_number": env("HO 00 03")}]},
        "extraction": {"forms_and_endorsements": [{"form_number": env("CA 00 01")}]},
    }]}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    [line] = script.list_details(str(path))
    assert "key=['form_number']" in line and "HO 00 03" in line and "CA 00 01" in line
