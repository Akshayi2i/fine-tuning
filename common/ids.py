"""``source_id`` — the traceability spine (master §3).

One identifier appears at every layer: the raw PDF, the OCR output, the golden
label, every compiled JSONL row, and every extraction result. It is what lets a
bad prediction be walked back to the page that produced it, and it is what makes
the split-leakage rule enforceable at all — you cannot assert that a document
does not cross splits without a stable name for the document.

Format: ``{doc_type}_{zero_padded_index}``, e.g. ``acord_0001``. The ACORD form
number is tracked separately as ``acord_form``, never baked into the id, because
a form is a property of the document and not part of its identity.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from common.constants import ACTIVE_DOC_TYPES, KNOWN_DOC_TYPES, UNCLASSIFIED

#: Width of the zero-padded index. Four digits carries 9,999 documents per type;
#: the parser accepts more, so passing this is not a silent failure.
INDEX_WIDTH = 4

_SOURCE_ID_RE = re.compile(r"^(?P<doc_type>[a-z]+)_(?P<index>\d+)$")


class SourceIdError(ValueError):
    """Raised on a malformed or unknown-type ``source_id``."""


class ParsedSourceId(NamedTuple):
    doc_type: str
    # Shadows ``tuple.index``. Kept because ``parsed.index`` is what a reader of
    # a source_id expects to write; nothing calls the tuple method on it.
    index: int  # type: ignore[assignment]
    raw: str


def build_source_id(doc_type: str, index: int, *, width: int = INDEX_WIDTH) -> str:
    """``("acord", 1)`` -> ``"acord_0001"``."""
    doc_type = doc_type.lower().strip()
    # UNCLASSIFIED is a real bucket, not a typo. Documents whose type is not yet
    # known park there rather than having a guess baked into the source_id —
    # the key everything downstream joins on. Rejecting it here made the
    # documented `--doc-type unclassified` mode fail one document at a time
    # with a message that read like bad input.
    if doc_type not in ACTIVE_DOC_TYPES and doc_type != UNCLASSIFIED:
        raise SourceIdError(
            f"unknown doc_type {doc_type!r}; active types: {ACTIVE_DOC_TYPES} "
            f"(or {UNCLASSIFIED!r} while the type is still unknown)"
        )
    if index < 0:
        raise SourceIdError(f"index must be non-negative, got {index}")
    return f"{doc_type}_{index:0{width}d}"


def parse_source_id(source_id: str) -> ParsedSourceId:
    """``"acord_0001"`` -> ``ParsedSourceId("acord", 1, "acord_0001")``."""
    if not isinstance(source_id, str):
        raise SourceIdError(f"source_id must be a string, got {type(source_id).__name__}")
    match = _SOURCE_ID_RE.match(source_id.strip())
    if not match:
        raise SourceIdError(
            f"malformed source_id {source_id!r}; expected {{doc_type}}_{{index}} such as 'acord_0001'"
        )
    doc_type = match.group("doc_type")
    # KNOWN, not ACTIVE: reading is not minting. A `lossrun_0001` written while
    # Loss Runs were active must stay parseable after the type is paused, or
    # every path, corpus row and golden label naming it becomes unreadable —
    # pausing a type would strand its artifacts instead of just stopping new
    # ones. `build_source_id` still refuses to MINT an inactive type.
    if doc_type not in KNOWN_DOC_TYPES and doc_type != UNCLASSIFIED:
        raise SourceIdError(
            f"source_id {source_id!r} names unknown doc_type {doc_type!r}; "
            f"known types: {KNOWN_DOC_TYPES}"
        )
    return ParsedSourceId(doc_type, int(match.group("index")), source_id.strip())


def is_valid_source_id(source_id: str) -> bool:
    """Non-raising form of :func:`parse_source_id`."""
    try:
        parse_source_id(source_id)
    except SourceIdError:
        return False
    return True


def doc_type_of(source_id: str) -> str:
    """The document type encoded in a ``source_id``."""
    return parse_source_id(source_id).doc_type


def next_source_id(existing: list[str], doc_type: str) -> str:
    """The next unused id for a type.

    Takes ``max(index) + 1`` rather than ``len(existing) + 1`` so that a gap —
    from a purged or skipped document — never causes a collision that would
    silently overwrite an existing raw PDF.
    """
    indices = [
        parsed.index
        for sid in existing
        if is_valid_source_id(sid) and (parsed := parse_source_id(sid)).doc_type == doc_type
    ]
    return build_source_id(doc_type, (max(indices) + 1) if indices else 1)
