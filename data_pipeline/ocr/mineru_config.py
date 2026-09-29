"""MinerU's own device setting, which decides where it actually runs.

MinerU 1.x reads its inference device from ``device-mode`` in its config file
(``~/magic-pdf.json``, or the name in ``MINERU_TOOLS_CONFIG_JSON`` under the home
directory) — not from whether a GPU exists. Its shipped template says ``cpu``. A
CUDA check alone passes on a GPU pod while MinerU quietly runs its CPU model
variants, and every ``ocr_meta.json`` would still record ``cuda``: CPU-formatted
markdown under a GPU pin, the silent shift the GPU-only rule exists to prevent.

    python -m data_pipeline.ocr.mineru_config --cuda     # set device-mode to cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


class MinerUConfigError(RuntimeError):
    """Raised when MinerU is not configured to run on the GPU."""


def config_path() -> Path:
    """Where MinerU 1.x reads its config: the home directory, file name overridable."""
    name = os.environ.get("MINERU_TOOLS_CONFIG_JSON") or "magic-pdf.json"
    return Path.home() / name


def assert_on_cuda(path: Path | None = None) -> None:
    """Refuse unless MinerU's config sends it to the GPU."""
    path = path or config_path()
    if not path.is_file():
        raise MinerUConfigError(
            f"no MinerU config at {path}. Download MinerU's model weights (download_models_hf.py), "
            "which writes it, then run: python -m data_pipeline.ocr.mineru_config --cuda"
        )
    try:
        mode = str(json.loads(path.read_text(encoding="utf-8")).get("device-mode", "")).lower()
    except ValueError as exc:
        raise MinerUConfigError(f"{path} is not valid JSON: {exc}") from exc
    if mode != "cuda":
        raise MinerUConfigError(
            f"{path} sets device-mode {mode or '(unset)'!r}, so MinerU would run on the CPU while the "
            "corpus records cuda. Fix: python -m data_pipeline.ocr.mineru_config --cuda"
        )
    if formula_enabled(json.loads(path.read_text(encoding="utf-8"))):
        raise MinerUConfigError(
            f"{path} has formula recognition on. It is off for this corpus: policies carry no "
            "equations, and MinerU 1.3's UniMERNet fails under transformers 4.57 ('cache_position'). "
            "Every document is OCR'd with it off, so training and serving read pages alike. "
            "Fix: python -m data_pipeline.ocr.mineru_config --cuda"
        )


def formula_enabled(body: dict) -> bool:
    """MinerU's formula recognition, which is on unless its config says otherwise."""
    return bool((body.get("formula-config") or {}).get("enable", True))


def set_cuda(path: Path | None = None) -> Path:
    """Set ``device-mode`` to ``cuda`` in MinerU's config, keeping every other key."""
    path = path or config_path()
    if not path.is_file():
        raise MinerUConfigError(f"no MinerU config at {path}; download MinerU's model weights first")
    body = json.loads(path.read_text(encoding="utf-8"))
    body["device-mode"] = "cuda"
    # Formula recognition (UniMERNet) off: policies carry no equations, and under
    # the transformers this stack pins it fails on every page ('cache_position').
    # Layout, OCR and table recognition are unaffected.
    formula = body.get("formula-config")
    body["formula-config"] = {**(formula if isinstance(formula, dict) else {}), "enable": False}
    path.write_text(json.dumps(body, indent=4), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or set MinerU's device (GPU only)")
    parser.add_argument("--cuda", action="store_true",
                        help="set device-mode to cuda and formula recognition off")
    args = parser.parse_args(argv)
    try:
        if args.cuda:
            print(f"device-mode set to cuda, formula recognition off, in {set_cuda()}")
        assert_on_cuda()
    except MinerUConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"MinerU runs on cuda, formula recognition off ({config_path()})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
