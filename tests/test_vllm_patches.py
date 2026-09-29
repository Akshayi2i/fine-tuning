"""The Qwen3-VL LoRA fix for vLLM 0.11.0, against a stand-in vLLM."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import pytest

from inference_core import vllm_patches


@dataclass
class _Keys:
    language_model: list = field(default_factory=list)
    connector: list = field(default_factory=list)
    tower_model: list = field(default_factory=list)

    @staticmethod
    def from_string_field(language_model=None, connector=None, tower_model=None):
        as_list = lambda v: [v] if isinstance(v, str) else list(v or [])  # noqa: E731
        return _Keys(as_list(language_model), as_list(connector), as_list(tower_model))


def _fake_vllm(monkeypatch, tower="model.visual."):
    class Qwen3VL:
        def get_mm_mapping(self):
            return _Keys.from_string_field("language_model", tower + "merger", tower)

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.models": types.ModuleType("vllm.model_executor.models"),
        "vllm.model_executor.models.qwen3_vl": types.SimpleNamespace(
            Qwen3VLForConditionalGeneration=Qwen3VL),
        "vllm.model_executor.models.module_mapping": types.SimpleNamespace(MultiModelKeys=_Keys),
    }
    modules["vllm.model_executor.models"].qwen3_vl = modules["vllm.model_executor.models.qwen3_vl"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv(vllm_patches.IN_PROCESS_ENV, raising=False)
    return Qwen3VL


def test_the_vision_tower_is_excluded_by_the_engines_own_module_names(monkeypatch):
    """0.11.0 names the tower `model.visual.`; the engine's modules are `visual.`,
    so nothing was excluded and an image request failed in _lora_shrink."""
    cls = _fake_vllm(monkeypatch)
    assert vllm_patches.patch_qwen3_vl_lora("0.11.0") is True
    keys = cls().get_mm_mapping()
    assert keys.tower_model == ["visual."] and keys.connector == ["visual.merger"]
    assert keys.language_model == ["language_model"]


def test_the_engine_core_runs_in_process_so_the_fix_reaches_it(monkeypatch):
    import os

    _fake_vllm(monkeypatch)
    vllm_patches.patch_qwen3_vl_lora("0.11.0")
    assert os.environ[vllm_patches.IN_PROCESS_ENV] == "0"


def test_applying_it_twice_is_harmless(monkeypatch):
    cls = _fake_vllm(monkeypatch)
    vllm_patches.patch_qwen3_vl_lora("0.11.0")
    assert vllm_patches.patch_qwen3_vl_lora("0.11.0") is True
    assert cls().get_mm_mapping().tower_model == ["visual."]


def test_a_vllm_without_the_bug_is_left_alone(monkeypatch):
    import os

    cls = _fake_vllm(monkeypatch, tower="visual.")
    assert vllm_patches.patch_qwen3_vl_lora("0.11.1") is False
    assert vllm_patches.IN_PROCESS_ENV not in os.environ
    assert cls.get_mm_mapping is not vllm_patches._fixed_qwen3_vl_mapping


def test_unexpected_code_under_the_pinned_version_is_refused(monkeypatch):
    _fake_vllm(monkeypatch, tower="something.else.")
    with pytest.raises(vllm_patches.VllmPatchError, match="re-check"):
        vllm_patches.patch_qwen3_vl_lora("0.11.0")


def test_the_pinned_vllm_is_the_one_the_fix_targets():
    """Moving the pin off 0.11.0 means re-checking whether this fix is still needed."""
    from pathlib import Path

    lock = Path(__file__).resolve().parents[1] / "requirements-train.lock"
    pinned = [line.split("==")[1].strip() for line in lock.read_text().splitlines()
              if line.startswith("vllm==")]
    assert pinned and pinned[0] in vllm_patches._QWEN3_VL_BROKEN
