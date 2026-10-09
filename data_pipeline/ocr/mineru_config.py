"""MinerU's own settings that decide where it runs and which model weights it reads.

MinerU 3.x takes its device from ``MINERU_DEVICE_MODE`` (else whatever torch
finds) and its pipeline model weights from ``models-dir.pipeline`` in its config
file (``~/mineru.json``, or ``MINERU_TOOLS_CONFIG_JSON``: a name under the home
directory or an absolute path) when ``MINERU_MODEL_SOURCE`` is ``local``. Left
to itself it downloads the latest weights from Hugging Face on first use: a pod
set up next month would read pages with other weights than this one, the same
silent shift the version pin exists to prevent. So OCR runs only on the weights
the download script pinned, on the GPU.

    python -m data_pipeline.ocr.mineru_config            # check, exit 1 if not ready
    python -m data_pipeline.ocr.mineru_config --write DIR  # point the config at DIR's weights
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: What the run environment is set to before MinerU is imported
#: (:func:`apply_run_environment`): the GPU, and the pinned local weights.
RUN_ENVIRONMENT = {"MINERU_DEVICE_MODE": "cuda", "MINERU_MODEL_SOURCE": "local"}


class MinerUConfigError(RuntimeError):
    """Raised when MinerU is not set to read the pinned weights on the GPU."""


def config_path() -> Path:
    """Where MinerU 3.x reads its config: the home directory, or the name or
    absolute path in ``MINERU_TOOLS_CONFIG_JSON``."""
    name = os.environ.get("MINERU_TOOLS_CONFIG_JSON") or "mineru.json"
    path = Path(name)
    return path if path.is_absolute() else Path.home() / name


def apply_run_environment() -> None:
    """Send MinerU to the GPU and the local weights unless already set; called
    before MinerU is imported, since it reads both at import and model load."""
    for key, value in RUN_ENVIRONMENT.items():
        os.environ.setdefault(key, value)


def assert_on_cuda(path: Path | None = None) -> None:
    """Refuse unless MinerU would run on the GPU on the pinned local weights."""
    device = os.environ.get("MINERU_DEVICE_MODE")
    if device is not None and not device.lower().startswith("cuda"):
        raise MinerUConfigError(
            f"MINERU_DEVICE_MODE is {device!r}, so MinerU would run off the GPU while the corpus "
            "records cuda. Unset it, or set it to cuda."
        )
    source = os.environ.get("MINERU_MODEL_SOURCE")
    if source is not None and source != "local":
        raise MinerUConfigError(
            f"MINERU_MODEL_SOURCE is {source!r}: MinerU would fetch the latest weights rather than "
            "the pinned ones. Unset it, or set it to local."
        )
    path = path or config_path()
    if not path.is_file():
        raise MinerUConfigError(
            f"no MinerU config at {path}. Download the pinned model weights, which writes it: "
            "bash scripts/download_mineru_models.sh"
        )
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise MinerUConfigError(f"{path} is not valid JSON: {exc}") from exc
    if "device-mode" in body and "models-dir" not in body:
        raise MinerUConfigError(
            f"{path} is a MinerU 1.x config (magic-pdf). MinerU 3.x reads models-dir from "
            "mineru.json: bash scripts/download_mineru_models.sh"
        )
    weights = (body.get("models-dir") or {}).get("pipeline") if isinstance(body.get("models-dir"), dict) else None
    if not weights or not Path(weights).is_dir():
        raise MinerUConfigError(
            f"{path} names no pipeline model weights that exist (models-dir.pipeline = {weights!r}). "
            "Download them: bash scripts/download_mineru_models.sh"
        )


def write_config(weights: Path, path: Path | None = None) -> Path:
    """Point MinerU's config at the pipeline weights in ``weights``, keeping every
    other key. Reading them, rather than fetching, is ``MINERU_MODEL_SOURCE=local``
    (:data:`RUN_ENVIRONMENT`): MinerU 3.4's config file takes only huggingface or
    modelscope as its ``model-source`` (``local`` is ignored with a warning), and
    either would send a run without the variable to fetch the latest - so the key
    is removed."""
    path = path or config_path()
    body = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if "device-mode" in body and "models-dir" not in body:
        body = {}                                   # a MinerU 1.x config: nothing in it applies
    models = body.get("models-dir") if isinstance(body.get("models-dir"), dict) else {}
    body["models-dir"] = {**models, "pipeline": str(weights)}
    body.pop("model-source", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=4), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check (or write) MinerU's GPU and model-weight settings")
    parser.add_argument("--write", type=Path, metavar="DIR",
                        help="point the config at the pipeline model weights in DIR")
    args = parser.parse_args(argv)
    try:
        if args.write:
            print(f"models-dir.pipeline = {args.write} in {write_config(args.write)}")
        apply_run_environment()
        assert_on_cuda()
    except MinerUConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"MinerU reads the pinned weights on cuda ({config_path()})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
