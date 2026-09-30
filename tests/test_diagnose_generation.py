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
