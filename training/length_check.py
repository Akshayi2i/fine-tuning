"""Every staged row's real length, measured on the pod before launch.

The corpus build sizes rows with a character-based estimate (``cap_check``),
deliberately pessimistic because it runs with no model. That is the right tool
there and the wrong last word: a row the estimate under-counts reaches the
trainer, and under ``strict`` encoding — which the run uses, so a bad row cannot
be silently swapped for another — it stops the run mid-training, hours in.

So the count is taken again on the pod, where the tokenizer exists, before
``swift sft`` starts:

* **Text** is counted with the model's own tokenizer — exact, and fast.
* **Images** are counted from each page's dimensions through Qwen's resize rule —
  the same arithmetic the image processor runs, without decoding a pixel.
* **Template overhead** is a fixed margin per message and per image, on the
  pessimistic side.

Running every row through the full processor would be exact to the token, and
would idle a paid GPU for most of an hour on image preprocessing alone. This is
the difference between those two, spent on the part that is cheap to get right.
The resize factor and margins are confirmed by the Phase 0 spike
(``visual_token_geometry``, ``swift_image_budget``).
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: Pixels per side of one visual token: 16px patches merged 2x2. The same
#: geometry ``cap_check.PIXELS_PER_VISUAL_TOKEN`` assumes (32 * 32).
RESIZE_FACTOR = 32

#: Chat-template tokens around each message (role tags, separators) and around
#: each image (vision start/end). Rounded up: over-counting refuses a row an
#: operator can look at, under-counting stops a paid run.
TOKENS_PER_MESSAGE = 8
TOKENS_PER_IMAGE_WRAPPER = 4

#: Counts tokens for a batch of strings. The model's tokenizer on the pod; a
#: stub in tests.
TokenCounter = Callable[[list[str]], list[int]]


class LengthCheckError(RuntimeError):
    """Raised when a staged row cannot be measured."""


@dataclass
class LengthReport:
    """What was measured, and what does not fit."""

    rows: int = 0
    longest: int = 0
    over: list[dict[str, object]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.over


def png_size(path: str | Path) -> tuple[int, int]:
    """``(width, height)`` from a PNG's header, without decoding the image."""
    with open(path, "rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise LengthCheckError(f"{path} is not a PNG with a readable header")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def image_tokens(
    width: int, height: int, *, min_pixels: int, max_pixels: int, factor: int = RESIZE_FACTOR
) -> int:
    """Visual tokens for one page, by Qwen's ``smart_resize`` rule.

    Each side is rounded to the factor; an area above ``max_pixels`` is scaled
    down, one below ``min_pixels`` scaled up, keeping the aspect ratio.
    """
    if width <= 0 or height <= 0:
        raise LengthCheckError(f"image has no area ({width}x{height})")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return (h_bar * w_bar) // (factor * factor)


def measure(
    files: Sequence[str | Path],
    *,
    max_length: int,
    count_tokens: TokenCounter,
    min_pixels: int,
    max_pixels: int,
    batch: int = 256,
    output_caps: dict[str, list[int]] | None = None,
) -> LengthReport:
    """Measure every row of the staged (ms-swift format) files.

    Against ``max_length``, and — when ``output_caps`` gives each row's task
    reservation — the target (assistant turn) against its own cap too. A target
    past its reservation is the case that clips an answer, and the corpus
    estimate under-counts exactly that text (digits tokenize one apiece).
    """
    report = LengthReport()
    caps_by_file = output_caps or {}
    # One header read per image. Every epoch file names the same pages, so an
    # uncached pass read each page's header once per epoch — thousands of
    # small reads off a network volume before a launch.
    sizes: dict[str, tuple[int, int]] = {}

    def size_of(image: str) -> tuple[int, int]:
        if image not in sizes:
            sizes[image] = png_size(image)
        return sizes[image]

    for path in files:
        caps = caps_by_file.get(str(path)) or caps_by_file.get(str(Path(path).resolve()))
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            texts = [
                "".join(m["content"].replace("<image>", "") for m in row["messages"])
                for row in chunk
            ]
            targets = [
                row["messages"][-1]["content"] if row["messages"][-1]["role"] == "assistant" else ""
                for row in chunk
            ]
            target_tokens = count_tokens(targets) if caps else [0] * len(chunk)
            for index, (row, text_tokens, output_tokens) in enumerate(
                zip(chunk, count_tokens(texts), target_tokens, strict=True)
            ):
                visual = sum(
                    image_tokens(*size_of(image), min_pixels=min_pixels, max_pixels=max_pixels)
                    + TOKENS_PER_IMAGE_WRAPPER
                    for image in row["images"]
                )
                total = text_tokens + visual + TOKENS_PER_MESSAGE * len(row["messages"])
                report.rows += 1
                report.longest = max(report.longest, total)
                cap = caps[start + index] if caps else None
                if cap is not None and output_tokens > cap:
                    report.over.append({
                        "file": str(path), "row": start + index, "output_tokens": output_tokens,
                        "output_cap": cap, "tokens": total,
                    })
                elif total > max_length:
                    report.over.append({
                        "file": str(path), "row": start + index, "tokens": total,
                        "text_tokens": text_tokens, "visual_tokens": visual,
                        "images": len(row["images"]),
                    })
    return report


def tokenizer_counter(model_id: str, revision: str | None = None) -> TokenCounter:
    """The model's own tokenizer as a :data:`TokenCounter`. Loaded on the pod."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )

    def count(texts: list[str]) -> list[int]:
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]

    return count
