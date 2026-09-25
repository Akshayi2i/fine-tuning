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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)

#: ms-swift's image placeholder, one per entry in the row's ``images`` list.
SWIFT_IMAGE_TAG = "<image>"


class StagingError(RuntimeError):
    """Raised when a run's training data cannot be staged faithfully."""


@dataclass
class StagedData:
    """The local files a run reads, and what staging did."""

    epoch_files: list[str] = field(default_factory=list)
    val_path: str | None = None
    rows: int = 0
    images: int = 0


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
    """OCR text that literally contains ``<image>`` would be read as a placeholder,
    pairing the next page image with the wrong text and shifting every image
    after it. Refused rather than escaped: there is no escape the template honours."""
    if SWIFT_IMAGE_TAG in text:
        raise StagingError(
            f"{row.get('source_id')}: the text contains a literal {SWIFT_IMAGE_TAG!r}, which "
            "ms-swift would read as an image placeholder and pair with the wrong page."
        )


def stage_training_data(
    epoch_files: Sequence[str],
    val_path: str | None,
    client: BlobClient,
    local_root: str | Path,
) -> StagedData:
    """Copy and convert a run's data onto local disk. Returns the local paths.

    Idempotent: an image already on the volume is not downloaded again, so a
    resumed run reuses what the first attempt staged.
    """
    root = Path(local_root)
    images_root = root / "images"
    staged = StagedData()
    local_of: dict[str, str] = {}

    def local_image(key: str) -> str:
        if key not in local_of:
            target = images_root / key
            if not target.exists():
                if not client.exists(key):
                    raise StagingError(
                        f"page image {key} is referenced by the corpus but is not in Blob. "
                        "Training on a row without its page would teach the model to answer "
                        "from text it was told came with an image."
                    )
                client.download_file(key, target)
            local_of[key] = str(target.resolve())
        return local_of[key]

    def stage(key: str, name: str) -> str:
        rows = [json.loads(line) for line in client.read_text(key).splitlines() if line.strip()]
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(
                json.dumps(to_swift_row(row, local_image), ensure_ascii=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        staged.rows += len(rows)
        return str(target.resolve())

    for index, key in enumerate(epoch_files, start=1):
        staged.epoch_files.append(stage(key, f"train/epoch_{index}.jsonl"))
    if val_path:
        staged.val_path = stage(val_path, "val/val.jsonl")
    staged.images = len(local_of)

    log.info(
        "staged %d row(s) and %d page image(s) under %s", staged.rows, staged.images, root
    )
    return staged
