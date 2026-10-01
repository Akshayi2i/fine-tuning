"""Data-parallel training on several GPUs, with the one-GPU run's effective batch."""

from __future__ import annotations

import pytest

from training import train as T

EPOCH_PATHS = [f"corpus/default/v1/train/epoch_{i}.jsonl" for i in (1, 2, 3, 4)]


def test_gpus_are_read_from_config_or_detected():
    assert T.training_gpus({"gpus": 4}) == 4
    assert T.training_gpus({}) == 1
    assert T.training_gpus({"gpus": "auto"}) >= 1          # 1 on a machine with no GPU


def test_accumulation_keeps_the_effective_batch():
    batch = {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "effective_batch_size": 8}
    assert T._accumulation_steps(batch, 1) == 8
    assert T._accumulation_steps(batch, 4) == 2             # 4 GPUs x 1 row x 2 = 8
    assert T._accumulation_steps(batch, 8) == 1
    with pytest.raises(T.TrainingError, match="cannot be split"):
        T._accumulation_steps(batch, 3)


def test_four_gpus_train_like_one_but_launch_four_processes(monkeypatch):
    one, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/o", train_rows=20_000)
    monkeypatch.setattr(T, "training_gpus", lambda distributed: 4)
    four, recorded = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/o", train_rows=20_000)
    assert four.env["NPROC_PER_NODE"] == "4" and "NPROC_PER_NODE" not in one.env
    assert four.args["gradient_accumulation_steps"] * 4 == one.args["gradient_accumulation_steps"]
    # Same ~10 checks; each lands on its own cadence's save, so a step or two apart.
    assert abs(four.args["eval_steps"] - one.args["eval_steps"]) <= 5
    assert four.args["save_steps"] > one.args["save_steps"]           # ~hourly at 4x the step rate
    assert four.args["ddp_find_unused_parameters"] is False
    assert recorded.effective_batch_size == 8


def test_checkpoint_cadence_scales_with_gpus():
    one = T._checkpoint_cadence(2_500, {"checkpoint_every_steps": 70, "save_total_limit": 4})
    four = T._checkpoint_cadence(2_500, {"checkpoint_every_steps": 70, "save_total_limit": 4}, gpus=4)
    assert one["save_steps"] <= 70 and 140 < four["save_steps"] <= 280
    assert four["eval_steps"] % four["save_steps"] == 0
