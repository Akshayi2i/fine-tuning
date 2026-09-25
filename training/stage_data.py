"""Put a run's training data where ms-swift can read it, in the shape it reads.

Two gaps between the corpus and the trainer, closed in one place:

**Location.** The corpus — its epoch files, its validation file and every page
image a row points at — lives in Blob, addressed by keys. ``swift sft`` runs on
the pod and reads local files. Handed a Blob key it finds nothing; worse, an
unfamiliar dataset string can be taken for a hub dataset id. So the files a run
trains on are copied onto the staging volume first.

**Shape.** A corpus row stores its user turn as a list — ``{"type": "image"}``,
``{"type": "text"}``, interleaved — beside string system and assistant turns.
That is the shape vLLM serves from, so the corpus keeps it. It is not a shape
the trainer's loader can read: a JSON column that is a string in one place and a
list in another is refused by pyarrow, and so by HF ``datasets``, on the first
row. ms-swift's documented multimodal format is string content with an
``<image>`` placeholder per image and a top-level ``images`` list, in order.

**The conversion must not change what the model sees.** Qwen's chat template
renders a content list by concatenating, with no separator, a vision block for
each image and the text of each text block; ms-swift substitutes the same vision
block for each ``<image>``. So each image becomes ``<image>`` and the parts are
joined with nothing between them. That equivalence is asserted on the real
template by the Phase 0 spike (``check_swift_row_format``), not assumed here —
prompt parity between training and serving is the property the whole corpus
design rests on.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry.blob_client import BlobClient
from common.constants import TRAINER_SPECIAL_TAGS

log = logging.getLogger(__name__)

#: ms-swift's image placeholder, one per entry in the row's ``images`` list.
SWIFT_IMAGE_TAG = "<image>"

#: Concurrent image downloads. Staging runs on the paid pod before launch, and
#: 20,000 pages fetched one at a time is minutes of idle GPU.
DOWNLOAD_WORKERS = 16


class StagingError(RuntimeError):
    """Raised when a run's training data cannot be staged faithfully."""


@dataclass
class StagedData:
    """The local files a run reads, and what staging did."""

    epoch_files: list[str] = field(default_factory=list)
    val_path: str | None = None
    rows: int = 0
    val_rows: int = 0
    images: int = 0
    #: Images actually downloaded; the rest came from the version's shared cache.
    fetched: int = 0


def to_swift_row(row: dict[str, Any], local_image: Callable[[str], str]) -> dict[str, Any]:
    """One corpus row in ms-swift's format: string content, ``<image>``, ``images``.

    Only ``messages`` and ``images`` are kept. Everything else on a corpus row is
    bookkeeping the trainer does not read, and every extra column is one more
    whose type has to agree across every row before the file will load at all.
    """
    images: list[str] = []
    messages: list[dict[str, str]] = []
    for message in row.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            _refuse_placeholder(content, row)
            messages.append({"role": message["role"], "content": content})
            continue
        parts: list[str] = []
        for block in content or []:
            kind = block.get("type")
            if kind == "image":
                image = block.get("image")
                if not isinstance(image, str):
                    raise StagingError(
                        f"{row.get('source_id')}: an image block holds {type(image).__name__}, "
                        "not a path. A training row must reference its page by Blob key."
                    )
                parts.append(SWIFT_IMAGE_TAG)
                images.append(local_image(image))
            elif kind == "text":
                _refuse_placeholder(block.get("text", ""), row)
                parts.append(block.get("text", ""))
            else:
                raise StagingError(f"{row.get('source_id')}: unknown content block {kind!r}")
        messages.append({"role": message["role"], "content": "".join(parts)})
    return {"messages": messages, "images": images}


def _refuse_placeholder(text: str, row: dict[str, Any]) -> None:
    """Text containing one of ms-swift's tags would be parsed as one: a literal
    ``<image>`` pairs the next page with the wrong text, ``<bbox>`` or
    ``<video>`` is rewritten or fails to encode. Refused rather than escaped:
    there is no escape the template honours.

    The corpus build sets such documents aside first (``build_jsonl``), so this
    firing means a corpus built before that check. It is the backstop.
    """
    tag = next((t for t in TRAINER_SPECIAL_TAGS if t in text), None)
    if tag:
        raise StagingError(
            f"{row.get('source_id')}: the text contains a literal {tag!r}, which ms-swift "
            "parses as a special tag. Rebuild the corpus; the build now sets such documents aside."
        )


def stage_training_data(
    epoch_files: Sequence[str],
    val_path: str | None,
    client: BlobClient,
    local_root: str | Path,
    *,
    images_root: str | Path | None = None,
    workers: int = DOWNLOAD_WORKERS,
) -> StagedData:
    """Copy and convert a run's data onto local disk. Returns the local paths.

    Every row is read first, so the page images can be fetched concurrently and
    each exactly once. ``images_root`` is the per-corpus-version cache every run
    on that version shares (``paths.staging_train_images_dir``); without one the
    images go under ``local_root``.

    An image already in the cache is not fetched again. That is only safe
    because a download is written to a temporary file and renamed into place: a
    pod preempted mid-download leaves a ``.part`` file, never a truncated image
    under the real name that a later run would trust.
    """
    root = Path(local_root)
    cache = Path(images_root) if images_root is not None else root / "images"
    staged = StagedData()

    files: list[tuple[str, list[dict[str, Any]]]] = []
    for index, key in enumerate(epoch_files, start=1):
        files.append((f"train/epoch_{index}.jsonl", _read_rows(client, key)))
    if val_path:
        files.append(("val/val.jsonl", _read_rows(client, val_path)))

    keys = sorted({
        block["image"]
        for _name, rows in files
        for row in rows
        for message in row.get("messages") or []
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "image" and isinstance(block.get("image"), str)
    })
    local_of = {key: (cache / key).resolve() for key in keys}
    missing = [key for key in keys if not local_of[key].exists()]
    if missing:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(lambda key: _download(client, key, local_of[key]), missing))

    for name, rows in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(
                json.dumps(to_swift_row(row, lambda key: str(local_of[key])), ensure_ascii=False)
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        staged.rows += len(rows)
        if name.startswith("val/"):
            staged.val_path, staged.val_rows = str(target.resolve()), len(rows)
        else:
            staged.epoch_files.append(str(target.resolve()))
    staged.images = len(keys)
    staged.fetched = len(missing)

    log.info(
        "staged %d row(s) under %s; %d page image(s), %d fetched, cache %s",
        staged.rows, root, len(keys), len(missing), cache,
    )
    return staged


def _read_rows(client: BlobClient, key: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in client.read_text(key).splitlines() if line.strip()]


def _download(client: BlobClient, key: str, target: Path) -> None:
    """Fetch one image atomically: to ``<name>.part``, then renamed into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        client.download_file(key, partial)
    except Exception as exc:  # noqa: BLE001 - reported with the one question that matters
        partial.unlink(missing_ok=True)
        if not client.exists(key):
            raise StagingError(
                f"page image {key} is referenced by the corpus but is not in Blob. "
                "Training on a row without its page would teach the model to answer "
                "from text it was told came with an image."
            ) from exc
        raise StagingError(f"could not fetch page image {key}: {exc}") from exc
    os.replace(partial, target)
