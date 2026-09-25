"""The Phase 0 spike runner's own contract.

The spike itself needs a GPU, so what is testable here is the property that
matters most: **it reports every check, and a failure or an exception in one
never stops the others.** A spike that aborts at the first problem costs a
pod-hour to learn one fact instead of eight.
"""

from __future__ import annotations

import json

from scripts import phase0_spike as spike


def test_a_raising_check_is_reported_not_propagated():
    def boom(_r):
        raise RuntimeError("CUDA driver missing")

    result = spike.run("cuda", "everything downstream", boom)
    assert result.ok is False
    assert "CUDA driver missing" in result.detail
    assert "traceback" in result.data


def test_undetermined_is_not_a_pass():
    """A check that could not run has not passed — the same rule the promotion
    gate applies to an unmeasured metric."""
    def cannot_tell(r):
        r.ok = None
        r.detail = "llama.cpp not present"

    result = spike.run("mmproj", "whether GGUF can see", cannot_tell)
    assert result.status == "UNKNOWN"
    assert result.ok is not True


def test_the_report_separates_failures_from_undetermined():
    results = [
        spike.Result("a", ok=True, detail="fine"),
        spike.Result("b", ok=False, detail="broken", decides="fall back to sdpa"),
        spike.Result("c", ok=None, detail="not installed"),
    ]
    report = spike.render(results)

    assert "1 passed, 1 failed, 1 undetermined." in report
    assert "fall back to sdpa" in report
    assert "Undetermined is not a pass" in report


def test_every_failure_names_what_it_decides():
    """A spike result that does not say what it changes is a fact nobody acts on."""
    results = [spike.Result("b", ok=False, detail="x", decides="serve merged models instead")]
    assert "serve merged models instead" in spike.render(results)


#: Every check ``main`` runs without a --pdf. Named here so adding a check to the
#: spike without deciding what it decides fails this file rather than passing
#: silently — the spike costs a pod-hour, so an unowned check is wasted budget.
GPU_FREE_CHECKS = (
    "check_flash_attn", "check_ms_swift", "check_model_loads",
    "check_interleaved_content", "check_swift_row_format", "check_swift_image_budget",
    "check_vllm_multi_lora",
    "check_llama_cpp_mmproj",
    "check_merger_module_names", "check_visual_token_geometry", "check_peak_vram_per_cap",
    "check_sequence_parallel", "check_structured_outputs", "check_raw_logprobs",
    "check_fp8_export",
)


def test_the_run_exits_non_zero_only_on_a_real_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(spike, "check_gpu", lambda r: (setattr(r, "ok", True), setattr(r, "detail", "ok"))[0])
    for name in GPU_FREE_CHECKS:
        monkeypatch.setattr(spike, name, lambda r: setattr(r, "ok", None))

    out = tmp_path / "report.json"
    code = spike.main(["--out", str(out), "--skip-mineru"])

    assert code == 0, "undetermined checks must not fail the run"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 1 + len(GPU_FREE_CHECKS)
    assert {r["status"] for r in payload["results"]} <= {"PASS", "UNKNOWN"}


def test_every_check_main_runs_exists_on_the_module():
    """Catches a wiring typo, which would otherwise surface as an AttributeError
    on the pod rather than here."""
    for name in ("check_gpu", *GPU_FREE_CHECKS, "check_mineru_gpu", "check_mineru_determinism"):
        assert hasattr(spike, name), f"{name} is wired into main but not defined"


def test_the_v2_1_measurement_checks_are_all_present():
    """arch v2.1 §16.0 lists twelve items and says nothing is annotated at scale
    until they are confirmed. Three of them set numbers the code consumes —
    merger module names (§9a), visual token geometry (§7a) and peak VRAM (§9.3) —
    so a spike missing one of those produces a corpus built on a guess."""
    import inspect

    source = inspect.getsource(spike.main)
    for check in ("merger_module_names", "visual_token_geometry", "peak_vram_per_cap",
                  "sequence_parallel_and_memory_flags", "vllm_structured_outputs",
                  "raw_logprobs_under_constraint", "fp8_export"):
        assert check in source, f"{check} is defined but never run by main()"

def test_every_check_names_what_it_decides():
    """`decides` is not documentation — the report prints it for each failure, and
    a failure with no consequence attached is a fact nobody acts on."""
    import inspect

    source = inspect.getsource(spike.main)
    # Each wiring reads run("<name>", "<decides>", check_x). Split on the call
    # rather than matching across it, so a reformat does not silently pass.
    for chunk in source.split("run(")[1:]:
        name = chunk.split('"')[1]
        decides = chunk.split('"')[3] if chunk.count('"') >= 4 else ""
        assert decides.strip(), f"{name} does not say what it decides"
