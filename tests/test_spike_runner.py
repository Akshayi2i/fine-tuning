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


def test_the_run_exits_non_zero_only_on_a_real_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(spike, "check_gpu", lambda r: (setattr(r, "ok", True), setattr(r, "detail", "ok"))[0])
    for name in ("check_flash_attn", "check_ms_swift", "check_model_loads",
                 "check_interleaved_content", "check_vllm_multi_lora", "check_llama_cpp_mmproj"):
        monkeypatch.setattr(spike, name, lambda r: setattr(r, "ok", None))

    out = tmp_path / "report.json"
    code = spike.main(["--out", str(out), "--skip-mineru"])

    assert code == 0, "undetermined checks must not fail the run"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 7
    assert {r["status"] for r in payload["results"]} <= {"PASS", "UNKNOWN"}
