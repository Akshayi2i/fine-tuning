"""The inputs checkpoint selection and calibration never received.

``ctx.checkpoints`` and ``ctx.calibration_samples`` were read by their stages and
filled in by nothing, so on every real run checkpoint selection skipped and every
release shipped uncalibrated. These run the producers against a stub backend and
real corpus rows, so what they emit is what the stages read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation.checkpoint_eval import discover_checkpoints, generation_scorer, select_best
from evaluation.validation_generation import (
    calibration_samples,
    generate_validation,
    score_generations,
    split_prompt,
)
from inference_core.input_builder import build_training_row
from inference_core.model_runner import EchoBackend, load_model

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "golden" / "policy_0001.golden.json").read_text(encoding="utf-8"))
OCR = (FIXTURES / "ocr" / "policy_0001_page_1.md").read_text(encoding="utf-8")


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _val_rows(n: int = 4) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        row = build_training_row(
            "policy", f"policy_{i:04d}", [f"processed/default/policy/policy_{i:04d}/page_1.png"],
            [OCR], "ocr_plus_image", json.dumps(GOLDEN), split="val",
        )
        row["val_half"] = "calibration" if i % 2 else "threshold"
        rows.append(row)
    return rows


def _model(client, response: dict | str = GOLDEN):
    text = response if isinstance(response, str) else json.dumps(response, separators=(",", ":"))
    return load_model("base", client, backend_impl=EchoBackend(text))


def test_the_prompt_is_the_row_without_its_golden_answer():
    messages, golden = split_prompt(_val_rows(1)[0])
    assert golden == GOLDEN
    assert messages[-1]["role"] != "assistant"


def test_calibration_samples_come_from_both_halves_and_are_labelled(client):
    halves = calibration_samples(generate_validation(_val_rows(), _model(client)))

    assert halves["calibration"] and halves["threshold"]
    features, correct = halves["calibration"][0]
    assert features.is_usable
    assert all(correct for _, correct in halves["calibration"]), "an exact echo scored wrong"


def test_a_wrong_answer_is_labelled_wrong(client):
    wrong = {**GOLDEN, "policy_number": "NOT-THE-NUMBER"}
    halves = calibration_samples(generate_validation(_val_rows(), _model(client, wrong)))
    labels = {f.field_path: c for f, c in halves["calibration"]}
    assert labels["policy_number"] is False


def test_a_failed_generation_is_scored_not_dropped(client):
    generations = generate_validation(_val_rows(), _model(client, "not json"))
    assert all(g.error for g in generations)
    metrics = score_generations(generations)
    assert metrics["field_normalized_match"] < 0.5


def test_checkpoint_selection_scores_with_the_gate_metric(client):
    """Each checkpoint is passed as the adapter; the score is the gate's own."""
    backend = EchoBackend(json.dumps(GOLDEN, separators=(",", ":")))
    model = load_model("base", client, backend_impl=backend)
    scorer = generation_scorer(_val_rows(), model)

    report = select_best(["/s/checkpoint-10", "/s/checkpoint-20"], scorer)

    assert report.selected == "/s/checkpoint-20"   # tie breaks to the later step
    assert {c["adapter"] for c in backend.calls} == {"/s/checkpoint-10", "/s/checkpoint-20"}
    assert report.scores[0].metrics["field_normalized_match"] > 0.9


def test_checkpoints_are_discovered_under_the_versioned_run_directory(tmp_path):
    run = tmp_path / "v0-20260917-101500"
    for step in (10, 20, 30):
        (run / f"checkpoint-{step}").mkdir(parents=True)
    (run / "checkpoint-30" / "trainer_state.json").write_text(
        json.dumps({"best_model_checkpoint": "/elsewhere/v0-20260917-101500/checkpoint-20"}),
        encoding="utf-8",
    )

    checkpoints, best = discover_checkpoints(str(tmp_path))

    assert [Path(c).name for c in checkpoints] == ["checkpoint-10", "checkpoint-20", "checkpoint-30"]
    assert best is not None and Path(best).name == "checkpoint-20"
    assert discover_checkpoints(str(tmp_path / "missing")) == ([], None)


def test_calibrate_collects_its_own_samples_per_format(client):
    """collect_calibration_samples generates with each staged format, so the
    stage has evidence without an operator hand-building feature vectors."""
    from orchestration.pipeline_dag import StageContext, collect_calibration_samples

    client.write_text(
        paths.corpus_eval_split("v1", "val"),
        "".join(json.dumps(r) + "\n" for r in _val_rows()),
    )
    loaded: list[str] = []

    def loader(_ctx, fmt):
        loaded.append(fmt)
        return _model(client)

    ctx = StageContext(
        client=client, controller=None, out_version="v1",  # type: ignore[arg-type]
        formats=["bf16"], serving_model_loader=loader,
    )
    samples = collect_calibration_samples(ctx)

    assert loaded == ["bf16"]
    assert samples["bf16"]["calibration"] and samples["bf16"]["threshold"]


def test_rows_go_to_the_backend_in_batches_not_one_by_one(client):
    """One row at a time left vLLM idle between answers: hours per checkpoint."""
    backend = EchoBackend(json.dumps(GOLDEN, separators=(",", ":")))
    calls = []
    real = backend.generate_batch
    backend.generate_batch = lambda msgs, configs, adapter=None: (calls.append(len(msgs)),
                                                                   real(msgs, configs, adapter))[1]
    model = load_model("base", client, backend_impl=backend)
    generations = generate_validation(_val_rows(5), model, batch_rows=2)
    assert calls == [2, 2, 1]
    assert all(g.error is None and g.extraction for g in generations)


def test_a_batch_that_fails_together_is_retried_row_by_row(client):
    """One unreadable page refuses the whole vLLM call; only its row may be lost."""
    backend = EchoBackend(json.dumps(GOLDEN, separators=(",", ":")))
    backend.generate_batch = lambda msgs, configs, adapter=None: [RuntimeError("bad page")] * len(msgs)
    real_generate = backend.generate
    rows = _val_rows(3)

    def generate(messages, config, adapter=None):
        if messages == split_prompt(rows[1])[0]:
            raise RuntimeError("bad page")
        return real_generate(messages, config, adapter)

    backend.generate = generate
    generations = generate_validation(rows, load_model("base", client, backend_impl=backend))
    assert [g.error is None for g in generations] == [True, False, True]
    assert "bad page" in generations[1].error and generations[1].extraction == {}
