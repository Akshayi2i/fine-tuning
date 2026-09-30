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


def test_every_wrong_field_is_counted_by_type(tmp_path):
    script = _script()
    env = lambda v: {"raw": v, "parsed": v, "page_ref": [1]}  # noqa: E731
    golden = {"policy": {"policy_number": env("HO-1"), "effective_date": env("01/01/2026"),
                         "expiration_date": env("01/01/2027"), "insured": env("Jane Rivera"),
                         "premium": env("1200")},
              "forms_and_endorsements": [{"form_number": env("HO 00 03")},
                                         {"form_number": env("HO 04 90")}]}
    got = {"policy": {"policy_number": env("HO-1"),                 # correct
                      "effective_date": env("01/01/2027"),          # value from another field
                      "insured": env("Jane Riveia"),                # misread
                      "premium": env("9999"),                       # wrong value
                      "agent": env("Smith")},                       # invented
           # expiration_date missing -> left empty
           "forms_and_endorsements": [{"form_number": env("HO 00 03")}, {"form_title": env("x")}]}
    report = {"checkpoint": {"generations": [
        {"row": {"source_id": "p1"}, "golden": golden, "extraction": got, "error": None},
        {"row": {"source_id": "p2"}, "golden": golden, "extraction": {}, "error": "JSONDecodeError: x"},
    ]}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = script.error_breakdown(str(path))["checkpoint"]
    assert result["errors"] == {"left empty": 1, "wrong value": 1, "value from another field": 1,
                                "misread (near miss)": 1, "invented (not in the label)": 1}
    stats = result["stats"]
    assert stats["unusable answers"] == 1 and stats["fields correct"] == 1
    assert stats["table rows expected"] == 2 and stats["table rows found"] == 1
    assert stats["table rows written with no/unknown ID"] == 1
