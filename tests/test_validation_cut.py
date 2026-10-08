"""The validation cut: a fixed stratified sample for in-training checks and checkpoint
selection, and a near-tie broken on the full validation split."""

from __future__ import annotations

from evaluation.checkpoint_eval import CheckpointScore, SelectionReport, break_tie
from evaluation.validation_sample import validation_sample


def _rows():
    rows = []
    for lob, n in (("homeowners", 1000), ("personal_auto", 300), ("motorcycle", 30)):
        for i in range(n):
            for mode in ("ocr_plus_image", "noisy_ocr_image", "image_only"):
                rows.append({"source_id": f"{lob}-{i}", "lob": [lob], "modality_mode": mode,
                             "sections": "decl", "doc_type": "policy"})
    return rows


def test_the_sample_is_its_size_and_keeps_every_line_and_mode():
    sample = validation_sample(_rows(), 400)
    assert len(sample) == 400
    groups = {(r["lob"][0], r["modality_mode"]) for r in sample}
    assert len(groups) == 9                                   # 3 lines x 3 modes, none dropped
    homeowners = sum(r["lob"][0] == "homeowners" for r in sample)
    assert 280 <= homeowners <= 320                           # roughly its 75% share


def test_the_same_rows_are_chosen_whatever_their_order_or_format():
    """The training checks and checkpoint selection read rows in different files
    and processes; they must pick the same rows."""
    rows = _rows()
    first = validation_sample(rows, 400)
    reshuffled = validation_sample(list(reversed(rows)), 400)
    identity = lambda r: (r["source_id"], r["modality_mode"])  # noqa: E731
    assert sorted(map(identity, first)) == sorted(map(identity, reshuffled))


def test_a_small_split_is_used_whole():
    rows = _rows()[:120]
    assert validation_sample(rows, 400) == rows


def _report(f1s):
    report = SelectionReport(scores=[CheckpointScore(f"/ck/checkpoint-{s}", s, {"field_normalized_match": f})
                                     for s, f in f1s])
    report.selected = max(report.scores, key=lambda c: (c.field_f1, c.step)).checkpoint
    return report


def test_a_near_tie_is_decided_on_the_full_validation_split():
    report = _report([(280, 0.610), (560, 0.620), (600, 0.500)])      # margin 0.010 < 0.015
    full = {"/ck/checkpoint-280": 0.64, "/ck/checkpoint-560": 0.61}
    break_tie(report, lambda ck: {"field_normalized_match": full[ck]}, 0.015)
    assert report.selected == "/ck/checkpoint-280"
    assert report.tie_break["changed_choice"] is True
    assert set(report.tie_break["full_validation_field_f1"]) == set(full)   # only the top two


def test_a_clear_winner_on_the_sample_stands():
    report = _report([(280, 0.55), (560, 0.62)])                      # margin 0.07
    calls = []
    break_tie(report, lambda ck: calls.append(ck) or {}, 0.015)
    assert report.selected == "/ck/checkpoint-560" and calls == [] and report.tie_break is None


def test_staging_narrows_only_the_validation_rows(tmp_path):
    import json

    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from training.stage_data import stage_training_data

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "{}"}],
           "task": "extract", "doc_type": "policy"}
    client.write_text("corpus/e1.jsonl", "\n".join(json.dumps(row) for _ in range(5)))
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(row) for _ in range(5)))
    staged = stage_training_data(["corpus/e1.jsonl"], "corpus/val.jsonl", client, tmp_path,
                                 val_filter=lambda rows: rows[:2])
    assert staged.val_rows == 2
    assert len((tmp_path / "train" / "epoch_1.jsonl").read_text().splitlines()) == 5


def test_checkpoint_selection_ranks_on_the_sample_and_keeps_the_full_split_for_ties():
    import json

    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation import checkpoint_eval as C

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    rows = _rows()[:900]
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(r) for r in rows))
    seen = []
    original = C.generation_scorer

    def recording(rows_, model):
        seen.append(len(rows_))
        return original(rows_, model)

    C.generation_scorer = recording
    try:
        scorer = C.vllm_scorer(client=client, val_path="corpus/val.jsonl", model=object(), sample_rows=400)
    finally:
        C.generation_scorer = original
    assert seen == [400, 900] and callable(scorer.full)


def _document_metrics(accuracy, invented=0.10, empty=0.40):
    return {"field_normalized_match": accuracy, "document_field_normalized_match": accuracy,
            "document_hallucination_rate": invented, "document_false_null_rate": empty}


def test_the_finalists_are_judged_on_whole_documents():
    from evaluation.checkpoint_eval import choose_on_documents

    report = _report([(280, 0.62), (560, 0.60), (600, 0.55), (640, 0.40)])
    full = {"/ck/checkpoint-280": _document_metrics(0.50), "/ck/checkpoint-560": _document_metrics(0.58),
            "/ck/checkpoint-600": _document_metrics(0.52)}
    asked = []
    choose_on_documents(report, lambda ck: asked.append(ck) or full[ck], finalists=3, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-560" and report.document_choice["changed_choice"] is True
    assert sorted(asked) == sorted(full)                         # the top three only


def test_a_finalist_that_invents_more_does_not_win_on_accuracy():
    from evaluation.checkpoint_eval import choose_on_documents

    report = _report([(280, 0.62), (560, 0.60)])
    full = {"/ck/checkpoint-280": _document_metrics(0.55, invented=0.10),
            "/ck/checkpoint-560": _document_metrics(0.58, invented=0.16)}     # 6 points more invented
    choose_on_documents(report, lambda ck: full[ck], finalists=3, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-280"
    assert report.document_choice["excluded_by_guard"] == ["/ck/checkpoint-560"]


def test_a_sample_that_is_the_whole_split_is_judged_without_scoring_again():
    from evaluation.checkpoint_eval import choose_on_documents

    report = _report([(280, 0.62), (560, 0.60)])
    report.scores[0].metrics.update(_document_metrics(0.50))
    report.scores[1].metrics.update(_document_metrics(0.58))
    choose_on_documents(report, None, finalists=2, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-560"


def test_one_candidate_needs_no_document_choice():
    from evaluation.checkpoint_eval import choose_on_documents

    report = _report([(280, 0.62)])
    choose_on_documents(report, lambda ck: 1 / 0, finalists=3, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-280" and report.document_choice is None
