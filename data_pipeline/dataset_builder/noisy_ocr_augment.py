"""Deliberate OCR corruption for the ``noisy_ocr_image`` regime (arch §5, §6).

20% of the Foundation corpus pairs **corrupted OCR text** with the **correct**
golden JSON. That mismatch is the entire training signal for image-over-OCR
arbitration: the model can only produce the right answer by reading the page, so
it learns to trust the image when the two disagree.

Corruptions must **mirror MinerU's real failure modes**. Random noise teaches the
wrong lesson — the model would learn to distrust OCR that looks nothing like the
OCR it will actually receive, and would still be fooled by a plausible
`0`-for-`O` substitution in production.

Every corruption is seeded and recorded, so ``ocr_arbitration_accuracy``
(SPEC_08) can score specifically on documents where the OCR and the image
genuinely disagree, rather than on the whole corpus.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

CorruptionType = Literal[
    "char_confusion", "merged_cells", "split_cell",
    "dropped_header", "reading_order", "dropped_separator",
]

#: Character pairs OCR genuinely confuses, in both directions. These are the
#: substitutions that survive into production and quietly corrupt a policy
#: number or a claim amount.
CHAR_CONFUSIONS: dict[str, str] = {
    "O": "0", "0": "O", "l": "1", "1": "l", "I": "1",
    "S": "5", "5": "S", "B": "8", "8": "B", "Z": "2", "2": "Z",
    "rn": "m", "cl": "d",
}


@dataclass
class CorruptionResult:
    """Corrupted text plus what was done to it."""

    text: str
    corruptions: list[CorruptionType] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def was_corrupted(self) -> bool:
        return bool(self.corruptions)


def _corrupt_characters(text: str, rng: random.Random, rate: float = 0.02) -> tuple[str, str | None]:
    """Substitute confusable characters at a low rate.

    Applied only inside alphanumeric runs of 3+ characters — policy numbers,
    claim numbers, amounts — because that is where a substitution changes the
    *value* rather than just making prose look odd.
    """
    # Spans, not bare strings. `str.replace(token, ..., 1)` rewrote the first
    # occurrence of that substring anywhere — including inside a longer token, so
    # corrupting "ABC12" could damage "ABC1234" instead and the recorded detail
    # then named a change that happened somewhere else. SPEC_08's
    # ocr_arbitration_accuracy reads those details to decide which field
    # genuinely disagreed.
    spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z0-9]{3,}", text)]
    if not spans:
        return text, None

    changed: list[str] = []
    edits: list[tuple[int, int, str]] = []
    # Visited in a random order, not reading order. Walking top to bottom and
    # stopping at the third hit put nearly every corruption in the page header —
    # the carrier name and the form title — and almost never in the schedule
    # values below, which are what arbitration has to learn to check.
    for start, end, token in rng.sample(spans, len(spans)):
        if rng.random() > rate * 10:      # most tokens untouched
            continue
        candidates = _confusable_positions(token)
        if not candidates:
            continue
        index, source = rng.choice(candidates)
        corrupted = token[:index] + CHAR_CONFUSIONS[source] + token[index + len(source):]
        edits.append((start, end, corrupted))
        changed.append(f"{token} -> {corrupted}")
        if len(changed) >= 3:
            break

    result = text
    for start, end, corrupted in sorted(edits, reverse=True):   # right to left, so offsets hold
        result = result[:start] + corrupted + result[end:]
    return result, ("; ".join(changed) if changed else None)


def _confusable_positions(token: str) -> list[tuple[int, str]]:
    """Every ``(index, source)`` in ``token`` a confusion can be applied at.

    Multi-character sources are matched too. Scanning one character at a time
    made the two most realistic print confusions — the ligature-like ``rn``/``m``
    and ``cl``/``d`` pairs — unreachable dead entries in the table, so the noise
    the corpus taught arbitration on was narrower than the table claimed.
    Longest source first, so ``rn`` wins over a single-character rule at the
    same position.
    """
    found: list[tuple[int, str]] = []
    for source in sorted(CHAR_CONFUSIONS, key=len, reverse=True):
        start = 0
        while (index := token.find(source, start)) >= 0:
            if not any(i <= index < i + len(s) for i, s in found):
                found.append((index, source))
            start = index + 1
    return sorted(found)


def _merge_table_cells(text: str, rng: random.Random) -> tuple[str, str | None]:
    """Drop the pipes from one table row, running its cells together.

    A common MinerU failure on tight table layouts, and a good stress test: the
    row's values are still present but no longer column-aligned.
    """
    lines = text.splitlines()
    candidates = [
        i for i, line in enumerate(lines)
        if line.strip().startswith("|") and line.count("|") >= 3
        and not all(set(c.strip()) <= set("-: ") for c in line.strip("|").split("|") if c.strip())
    ]
    if len(candidates) < 2:               # keep the header row intact
        return text, None

    index = rng.choice(candidates[1:])
    merged = " ".join(c.strip() for c in lines[index].strip().strip("|").split("|") if c.strip())
    lines[index] = merged
    return "\n".join(lines), f"row {index} cells merged"


def _drop_header(text: str, rng: random.Random) -> tuple[str, str | None]:
    """Remove a bold or hash heading — MinerU sometimes loses them entirely."""
    lines = text.splitlines()
    candidates = [
        i for i, line in enumerate(lines)
        if line.strip().startswith("#") or re.fullmatch(r"\*\*[^*]+\*\*", line.strip())
    ]
    if not candidates:
        return text, None
    index = rng.choice(candidates)
    dropped = lines.pop(index).strip()
    return "\n".join(lines), f"dropped heading {dropped[:40]!r}"


def _scramble_reading_order(text: str, rng: random.Random) -> tuple[str, str | None]:
    """Swap two adjacent non-table lines — a multi-column reading-order slip."""
    lines = text.splitlines()
    candidates = [
        i for i in range(len(lines) - 1)
        if lines[i].strip() and lines[i + 1].strip()
        and not lines[i].strip().startswith("|") and not lines[i + 1].strip().startswith("|")
    ]
    if not candidates:
        return text, None
    index = rng.choice(candidates)
    lines[index], lines[index + 1] = lines[index + 1], lines[index]
    return "\n".join(lines), f"swapped lines {index} and {index + 1}"


def _drop_table_separator(text: str, rng: random.Random) -> tuple[str, str | None]:
    """Remove the ``|---|`` separator, so the table stops parsing as a table."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|") if c.strip()]
        if line.strip().startswith("|") and cells and all(set(c) <= set("-: ") for c in cells):
            lines.pop(i)
            return "\n".join(lines), "dropped table separator row"
    return text, None


_CORRUPTORS: dict[CorruptionType, Callable[[str, random.Random], tuple[str, str | None]]] = {
    "char_confusion": lambda t, r: _corrupt_characters(t, r),
    "merged_cells": _merge_table_cells,
    "dropped_header": _drop_header,
    "reading_order": _scramble_reading_order,
    "dropped_separator": _drop_table_separator,
}

#: Weighted so character confusion dominates — it is by far the most common real
#: failure, and the one whose consequences are worst, because it silently changes
#: a value rather than obviously mangling the layout.
_WEIGHTS: dict[CorruptionType, float] = {
    "char_confusion": 0.45,
    "merged_cells": 0.20,
    "dropped_header": 0.15,
    "reading_order": 0.10,
    "dropped_separator": 0.10,
}


def corrupt_ocr(
    text: str,
    source_id: str,
    *,
    seed: int = 42,
    max_corruptions: int = 2,
) -> CorruptionResult:
    """Apply realistic OCR corruptions to a document's text.

    Seeded per ``source_id``, so rebuilding a corpus with the same seed produces
    byte-identical output — a corpus that changes between builds cannot be
    compared across model versions.
    """
    rng = random.Random(f"{seed}:{source_id}")
    result = CorruptionResult(text=text)

    kinds = list(_WEIGHTS)
    weights = [_WEIGHTS[k] for k in kinds]
    attempts = rng.choices(kinds, weights=weights, k=max_corruptions)

    for kind in attempts:
        corrupted, detail = _CORRUPTORS[kind](result.text, rng)
        if detail:
            result.text = corrupted
            result.corruptions.append(kind)
            result.details.append(f"{kind}: {detail}")

    if not result.was_corrupted:
        # Fall back to character confusion at a higher rate. A "noisy" row that
        # is identical to the clean one teaches nothing, and would silently
        # dilute the 20% arbitration signal to less than 20%.
        corrupted, detail = _corrupt_characters(result.text, rng, rate=0.08)
        if detail:
            result.text = corrupted
            result.corruptions.append("char_confusion")
            result.details.append(f"char_confusion: {detail}")

    return result


def corrupt_ocr_pages(
    pages: Sequence[str],
    source_id: str,
    *,
    seed: int = 42,
    max_corruptions: int = 2,
) -> tuple[list[str], list[str]]:
    """Corrupt a document's pages under **one document-level budget**.

    The budget is the point. ``corrupt_ocr`` applies up to ``max_corruptions``
    per call, so calling it once per page multiplies the noise by the page count:
    a 20-page policy would get up to 40 corruptions instead of 2, and the
    "nothing changed" fallback would fire on every page, guaranteeing that every
    page is mangled.

    That is not a louder version of the same signal — it is a different lesson.
    ``noisy_ocr_image`` is 20% of the corpus and exists to teach *arbitration*:
    trust the image where the OCR is wrong **and the OCR is usually right**. A
    corpus where every page is garbage teaches the model to ignore OCR entirely,
    which would degrade the 50% ``ocr_plus_image`` regime it never sees noise in.

    Returns ``(pages, details)`` with the same page count and order.
    """
    if not pages:
        return [], []

    rng = random.Random(f"{seed}:{source_id}:pages")
    corrupted = list(pages)
    details: list[str] = []

    # Spend the document's budget on randomly chosen pages, one corruption each.
    # Pages may repeat: two corruptions landing on one page is a realistic scan.
    kinds = list(_WEIGHTS)
    weights = [_WEIGHTS[k] for k in kinds]
    for kind in rng.choices(kinds, weights=weights, k=max_corruptions):
        index = rng.randrange(len(corrupted))
        text, detail = _CORRUPTORS[kind](corrupted[index], rng)
        if detail:
            corrupted[index] = text
            details.append(f"page {index + 1} {kind}: {detail}")

    if not details:
        # Same fallback as the single-page path, applied once to one page rather
        # than to every page: a noisy row identical to the clean one teaches
        # nothing and silently dilutes the arbitration signal below 20%.
        index = rng.randrange(len(corrupted))
        text, detail = _corrupt_characters(corrupted[index], rng, rate=0.08)
        if detail:
            corrupted[index] = text
            details.append(f"page {index + 1} char_confusion: {detail}")

    return corrupted, details
