"""Model compute runs on the GPU or not at all; the pod is recognised without
depending on one environment variable."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from common import gpu
from orchestration import detach

ROOT = Path(__file__).resolve().parent.parent


def _fake_torch(monkeypatch, *, available: bool, cuda_build: str | None = "12.8"):
    """A stand-in torch module, so the guard is exercised without torch installed."""
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available, get_device_name=lambda _i: "NVIDIA H100 80GB HBM3",
    )
    torch.version = types.SimpleNamespace(cuda=cuda_build)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_no_cuda_is_refused_not_run_slowly(monkeypatch):
    _fake_torch(monkeypatch, available=False)
    with pytest.raises(gpu.GPUError, match="GPU only.*no CUDA device is visible"):
        gpu.require_cuda("the merge")


def test_a_cpu_only_torch_build_is_named_as_the_cause(monkeypatch):
    _fake_torch(monkeypatch, available=False, cuda_build=None)
    with pytest.raises(gpu.GPUError, match="CPU-only build"):
        gpu.require_cuda("training")


def test_a_visible_gpu_is_accepted(monkeypatch):
    _fake_torch(monkeypatch, available=True)
    assert gpu.require_cuda("training") == "NVIDIA H100 80GB HBM3"


def test_training_refuses_to_launch_without_a_gpu(monkeypatch):
    from training import train

    _fake_torch(monkeypatch, available=False)
    monkeypatch.delenv(train.OFF_POD_ENV, raising=False)
    with pytest.raises(train.TrainingError, match="GPU only"):
        train.assert_on_pod()


#: Where a model is loaded. Each must refuse to run without CUDA.
MODEL_LOADERS = {
    "inference_core/model_runner.py": 2,   # vLLM engine and the HF backend
    "training/merge.py": 1,
    "postprocessing/quantize.py": 1,
    "training/train.py": 1,                # through assert_on_pod, before launch
}


@pytest.mark.parametrize("rel,expected", sorted(MODEL_LOADERS.items()))
def test_every_model_loader_requires_cuda(rel, expected):
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert text.count("require_cuda(") >= expected, rel


def test_no_model_is_placed_on_the_cpu_or_left_to_auto_offload():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if "tests" in path.parts or any(p.startswith(".") for p in path.parts):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r'device_map\s*=\s*["\'](cpu|auto)["\']', line):
                offenders.append(f"{path.relative_to(ROOT)}: {line.strip()}")
    assert not offenders, offenders


def test_ocr_refuses_the_cpu():
    from data_pipeline.ocr.mineru_version import resolve_device

    with pytest.raises(Exception, match="GPU only"):
        resolve_device("cpu")


@pytest.fixture
def clean_env(monkeypatch):
    for var in (detach.POD_ENV, "TMUX", detach.DETACHED_ENV, detach.NO_DETACH_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(detach, "RUNPOD_ENV_FILE", Path("/nonexistent/rp_environment"))
    monkeypatch.setattr(detach, "WORKSPACE", Path("/nonexistent/workspace"))


def test_a_laptop_is_not_the_pod(clean_env):
    assert not detach.on_pod()


def _pod_like(monkeypatch, *, linux=True, mounted=True, gpu_there=True):
    monkeypatch.setattr(detach, "_is_linux", lambda: linux)
    monkeypatch.setattr(detach.os.path, "ismount", lambda p: mounted and p == detach.WORKSPACE)
    monkeypatch.setattr(gpu, "gpu_present", lambda: gpu_there)


def test_the_pod_is_recognised_without_its_variable(clean_env, monkeypatch):
    """An SSH session that did not inherit RUNPOD_POD_ID still detaches."""
    _pod_like(monkeypatch)
    assert detach.on_pod()


@pytest.mark.parametrize("linux,mounted,gpu_there", [
    (False, True, True),    # a Windows laptop with a GPU and a D:\workspace folder
    (True, False, True),    # a Linux box whose /workspace is just a folder
    (True, True, False),    # a mounted /workspace but no GPU
])
def test_lookalikes_are_not_the_pod(clean_env, monkeypatch, linux, mounted, gpu_there):
    _pod_like(monkeypatch, linux=linux, mounted=mounted, gpu_there=gpu_there)
    assert not detach.on_pod()


def test_runpods_environment_file_marks_the_pod(clean_env, monkeypatch, tmp_path):
    marker = tmp_path / "rp_environment"
    marker.write_text("export RUNPOD_POD_ID=abc\n", encoding="utf-8")
    monkeypatch.setattr(detach, "RUNPOD_ENV_FILE", marker)
    assert detach.on_pod()


def test_shell_scripts_have_unix_line_endings():
    for script in (ROOT / "scripts").glob("*.sh"):
        assert b"\r" not in script.read_bytes(), script.name
