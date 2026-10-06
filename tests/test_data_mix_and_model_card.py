"""Per-line synthetic_fraction and scanned_share, the input-mode mix, and a model
card per run (Fideon SPEC_09 amendment items 6 and 7; handoff item 3)."""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.constants import MODALITY_MIX
from common.scopes import get_scope
from training import data_mix
from training.corpus_view import materialize
from training.data_mix import (
    DataMixError,
    MixSettings,
    assert_corpus_mix,
    configured_modality_mix,
    parse_settings,
    select_documents,
)

PERSONAL = get_scope("personal_lines")


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _rows(line, seeds, twins, *, scanned=True, twin_scanned=None, prefix=None):
    """``seeds`` real documents of ``line``, each with ``twins`` synthetic ones."""
    prefix = prefix or line[:3]
    rows = []
    for s in range(seeds):
        family = f"{prefix}-seed{s}"
        rows.append({"source_id": family, "group_id": family, "doc_type": "policy", "lob": line,
                     "synthetic": False, "is_scanned": scanned, "modality_mode": "ocr_plus_image"})
        for t in range(twins):
            flag = scanned if twin_scanned is None else twin_scanned(t)
            rows.append({"source_id": f"{family}-t{t}", "group_id": family, "doc_type": "policy",
                         "lob": line, "synthetic": True, "is_scanned": flag,
                         "modality_mode": "ocr_plus_image"})
    return rows


def _line(row):
    return str(row["lob"])


def _within_one(kept_synthetic, documents, share):
    return abs(kept_synthetic - share * documents) <= 1


# --------------------------------------------------------------------------
# synthetic_fraction
# --------------------------------------------------------------------------

def test_the_built_set_matches_the_synthetic_fraction_within_one_example_per_line():
    rows = _rows("homeowners", 5, 10) + _rows("motorcycle", 3, 10)
    settings = MixSettings(synthetic_fraction=0.8, lines={"motorcycle": {"synthetic_fraction": 0.5}})
    keep, mix = select_documents(rows, settings, seed=42, line_of=_line)
    for line, share in (("homeowners", 0.8), ("motorcycle", 0.5)):
        assert _within_one(mix[line].synthetic_kept, mix[line].documents, share)
    assert mix["homeowners"].synthetic_kept == 20 and mix["motorcycle"].synthetic_kept == 3
    assert {r["source_id"] for r in rows if not r["synthetic"]} <= keep          # real golds always train


def test_the_synthetic_budget_spreads_over_every_seed_and_is_reproducible():
    rows = _rows("homeowners", 5, 10)
    keep, _ = select_documents(rows, MixSettings(synthetic_fraction=0.8), seed=42, line_of=_line)
    per_seed = {f"hom-seed{s}": sum(1 for k in keep if k.startswith(f"hom-seed{s}-")) for s in range(5)}
    assert set(per_seed.values()) == {4}                                          # 20 twins: 4 of each seed
    again, _ = select_documents(rows, MixSettings(synthetic_fraction=0.8), seed=42, line_of=_line)
    assert again == keep


def test_nothing_steered_keeps_every_document():
    rows = _rows("homeowners", 2, 10)
    keep, mix = select_documents(rows, MixSettings(), seed=42, line_of=_line)
    assert keep == {r["source_id"] for r in rows} and mix["homeowners"].synthetic_kept == 20


# --------------------------------------------------------------------------
# scanned_share
# --------------------------------------------------------------------------

def test_the_built_set_matches_the_scanned_share_within_one_example():
    rows = _rows("homeowners", 4, 10, scanned=False, twin_scanned=lambda t: t % 2 == 0)
    settings = MixSettings(synthetic_fraction=0.8, scanned_share=0.25)
    _, mix = select_documents(rows, settings, seed=42, line_of=_line)
    line = mix["homeowners"]
    assert _within_one(line.synthetic_kept, line.documents, 0.8)
    assert _within_one(line.scanned_kept, line.documents, 0.25)


def test_when_a_pool_runs_short_the_scanned_share_wins_and_says_so():
    # 4 real (scanned), 40 scanned twins and 4 digital ones; 50% scanned asked.
    rows = _rows("homeowners", 4, 11, scanned=True, twin_scanned=lambda t: t != 0)
    _, mix = select_documents(rows, MixSettings(scanned_share=0.5), seed=42, line_of=_line)
    line = mix["homeowners"]
    assert _within_one(line.scanned_kept, line.documents, 0.5)
    assert line.synthetic_kept < line.synthetic_available and "fewer" in line.note


def test_an_unreachable_scanned_share_keeps_the_budget_and_says_so():
    """Today's data: every original is a scan and so is every twin."""
    rows = _rows("homeowners", 4, 10, scanned=True)
    _, mix = select_documents(rows, MixSettings(scanned_share=0.4), seed=42, line_of=_line)
    assert mix["homeowners"].synthetic_kept == 40 and "out of reach" in mix["homeowners"].note


def test_a_scanned_share_on_rows_that_do_not_say_is_refused():
    rows = [{k: v for k, v in r.items() if k != "is_scanned"} for r in _rows("homeowners", 2, 3)]
    with pytest.raises(DataMixError, match="is_scanned"):
        select_documents(rows, MixSettings(scanned_share=0.5), seed=42, line_of=_line)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

def test_a_misspelt_line_or_an_impossible_share_is_refused():
    with pytest.raises(DataMixError, match="does not cover"):
        parse_settings({"lines": {"homeowner": {"scanned_share": 0.3}}}, lines=PERSONAL.lines)
    with pytest.raises(DataMixError, match="between 0 and 1"):
        parse_settings({"synthetic_fraction": 1.5})
    with pytest.raises(DataMixError, match="unknown key"):
        parse_settings({"synthetic_share": 0.5})


def test_personal_lines_reads_its_own_config_over_unified():
    from common.config import training_config

    assert PERSONAL.training_config == "personal_lines"
    own, base = training_config("personal_lines"), training_config("unified")
    assert own["lora"] == base["lora"] and own["optimization"] == base["optimization"]
    assert "extends" not in own and "data_mix" in own and "data_mix" not in base
    settings = data_mix.mix_settings(PERSONAL)
    assert set(settings.lines) == set(PERSONAL.lines)
    assert not settings.steers                                   # null until the shares are known


def test_a_config_that_extends_itself_is_refused(tmp_path, monkeypatch):
    from common import config

    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "a.yaml").write_text("extends: b\n")
    (tmp_path / "training" / "b.yaml").write_text("extends: a\n")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    with pytest.raises(config.ConfigError, match="loop"):
        config._training_config("a", ())


# --------------------------------------------------------------------------
# The input-mode mix
# --------------------------------------------------------------------------

def test_the_mode_mix_is_the_global_default_unless_the_scope_sets_one(monkeypatch):
    from common import config

    assert configured_modality_mix(PERSONAL) == MODALITY_MIX
    own = {"ocr_plus_image": 0.4, "noisy_ocr_image": 0.2, "image_only": 0.4}
    monkeypatch.setattr(config, "training_config", lambda name: {"modality_mix": own})
    assert configured_modality_mix(PERSONAL) == own


def test_a_run_on_a_corpus_drawn_with_another_mix_is_refused(monkeypatch):
    other = {"ocr_plus_image": 0.4, "noisy_ocr_image": 0.2, "image_only": 0.4}
    with pytest.raises(DataMixError, match="rebuild it"):
        assert_corpus_mix(PERSONAL, {"modality_mix_target": other})
    assert_corpus_mix(PERSONAL, {"modality_mix_target": dict(MODALITY_MIX)})
    assert_corpus_mix(PERSONAL, {})                                  # a corpus from before the record


# --------------------------------------------------------------------------
# The scope's view
# --------------------------------------------------------------------------

def _seed(client, train, val):
    for epoch in (1, 2, 3, 4):
        client.write_text(paths.corpus_epoch_file("v1", epoch), "".join(json.dumps(r) + "\n" for r in train))
    client.write_text(paths.corpus_eval_split("v1", "val"), "".join(json.dumps(r) + "\n" for r in val))


def test_the_view_keeps_the_same_documents_every_epoch_and_never_samples_validation(client, monkeypatch):
    settings = MixSettings(synthetic_fraction=0.5)
    monkeypatch.setattr(data_mix, "mix_settings", lambda scope: settings)
    val = _rows("homeowners", 1, 5, prefix="val")
    _seed(client, _rows("homeowners", 4, 10) + _rows("motorcycle", 2, 10), val)
    view = materialize(PERSONAL, "v1", client)
    epochs = [{json.loads(line)["source_id"] for line in client.read_text(f).splitlines() if line.strip()}
              for f in view.epoch_files]
    assert all(e == epochs[0] for e in epochs)
    assert view.data_mix["homeowners"]["synthetic_kept"] == 4 and view.data_mix["motorcycle"]["synthetic_kept"] == 2
    assert view.rested_documents == 54
    assert view.val_rows == len(val)
    assert view.examples_by_line["val"]["homeowners"] == {"documents": 6, "real": 1, "synthetic": 5, "rows": 6}
    assert view.examples_by_line["train"]["motorcycle"]["documents"] == 4


def test_every_row_says_whether_its_document_is_scanned():
    from data_pipeline.dataset_builder.build_jsonl import build_corpus
    from data_pipeline.dataset_builder.split_groups import GroupSplitAssignment
    from tests.test_dataset_builder import _documents

    docs = _documents(6)
    for index, doc in enumerate(docs):
        doc.is_scanned = index % 2 == 0
    assignment = GroupSplitAssignment(assignment={d.family: "train" for d in docs[:4]}
                                      | {docs[4].family: "val", docs[5].family: "test"})
    rows = [row for split_rows in build_corpus(docs, assignment).rows_by_split.values() for row in split_rows]
    assert rows and all(isinstance(row.get("is_scanned"), bool) for row in rows)


# --------------------------------------------------------------------------
# The model card
# --------------------------------------------------------------------------

def _scoped_manifest(client, monkeypatch):
    from registry_utils.models import DataStats
    from training.train import build_manifest, build_training_config

    settings = MixSettings(synthetic_fraction=0.8, scanned_share=0.35,
                           lines={"motorcycle": {"synthetic_fraction": 0.6, "scanned_share": 0.9}})
    monkeypatch.setattr(data_mix, "mix_settings", lambda scope: settings)
    _seed(client, _rows("homeowners", 4, 10, scanned=False, twin_scanned=lambda t: t < 4)
          + _rows("motorcycle", 2, 10), _rows("homeowners", 1, 2, prefix="val"))
    view = materialize(PERSONAL, "v1", client)
    _swift, recorded = build_training_config(corpus_paths=view.epoch_files, output_dir="/tmp/out", scope=PERSONAL)
    stats = DataStats(
        train_examples=view.train_rows, val_examples=view.val_rows, test_examples=view.test_rows,
        modality_mix={"ocr_plus_image": 0.49, "noisy_ocr_image": 0.21, "image_only": 0.3},
        modality_mix_target=dict(MODALITY_MIX),
        split_policy={"held_out_carriers_by_line": {"policy": {"homeowners": "Zenith Mutual"}},
                      "single_carrier_lines": {"policy": ["ocean_marine"]}, "moved_to_test": {"policy": 2},
                      "twin_cap": 20, "twins_dropped": 3},
        data_mix={"settings": view.mix_settings, "lines": view.data_mix, "rested_documents": view.rested_documents},
        examples_by_line=view.examples_by_line,
    )
    return build_manifest(run_id=PERSONAL.run_id("v9"), corpus_version="v1", corpus_manifest={},
                          training_cfg=recorded, data_stats=stats,
                          staging_path=paths.scoped_staging_adapter_dir(PERSONAL.name, "v9"), scope=PERSONAL)


def test_the_model_card_lists_every_configured_value(client, monkeypatch):
    from registry_utils.model_card import render_model_card

    manifest = _scoped_manifest(client, monkeypatch)
    card = render_model_card(manifest)
    for configured in ("synthetic_fraction 0.8", "scanned_share 0.35", "| 0.6 ->", "| 0.9 ->"):
        assert configured in card, configured
    for line, entry in manifest.data_stats.data_mix["lines"].items():
        assert f"| {line} | {entry['real']} | {entry['synthetic_kept']} of {entry['synthetic_available']} |" in card
    for mode, share in MODALITY_MIX.items():
        assert f"| {mode} | {share:g} |" in card
    assert "Zenith Mutual" in card and "ocean_marine" in card and "Twin cap per seed and render mode: 20" in card
    assert f"| lora_rank | {manifest.training_config.lora_rank} |" in card
    assert f"| learning_rate | {manifest.training_config.learning_rate:g} |" in card
    assert "Not gated yet." in card


def test_the_gate_result_and_tier_reach_the_card_written_beside_the_manifest(client, monkeypatch):
    from evaluation.gating import GATING_METRICS, apply_to_manifest, promotion_gate
    from registry_utils.write_run_manifest import write_manifest

    manifest = _scoped_manifest(client, monkeypatch)
    metrics = {name: 0.96 for name in GATING_METRICS}
    metrics.update(schema_validity_rate=1.0, ece_confidence=0.04, confusable_misattribution_rate=0.02,
                   false_null_rate=0.03, auto_accept_error_rate=0.01,
                   field_normalized_match=0.89, field_exact_match=0.89)
    apply_to_manifest(promotion_gate(metrics, None), manifest, metrics=metrics)
    assert manifest.promotion.tier == "interim" and manifest.eval_metrics.field_normalized_match == 0.89
    key = write_manifest(manifest, client)
    card = client.read_text(key.replace("run_manifest.json", "model_card.md"))
    assert key.endswith("/run_manifest.json") and "registry/scope/personal_lines/" in key
    assert "Verdict: passed, interim release." in card and "| field_normalized_match | 0.89 |" in card


def test_a_lines_null_takes_the_scope_default():
    settings = parse_settings({"synthetic_fraction": 0.8, "scanned_share": 0.5,
                               "lines": {"homeowners": {"synthetic_fraction": None, "scanned_share": None},
                                         "motorcycle": {"synthetic_fraction": 0.6}}}, lines=PERSONAL.lines)
    assert settings.for_line("homeowners") == (0.8, 0.5)
    assert settings.for_line("motorcycle") == (0.6, 0.5)
    assert settings.steers


def test_a_package_line_matches_in_any_order():
    settings = parse_settings({"lines": {"personal_auto,homeowners": {"synthetic_fraction": 0.7}}},
                              lines=PERSONAL.lines)
    assert settings.for_line("homeowners,personal_auto") == (0.7, None)
