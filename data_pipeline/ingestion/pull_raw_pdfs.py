"""Ingest raw PDFs into the immutable ``raw-documents/`` layer (arch §18a).

The front of the whole pipeline, and the only component that writes this layer.
Three properties matter more than throughput:

**Immutable.** ``original.pdf`` is write-once. A corrected document is ingested
as a *new* ``source_id`` rather than replacing the old one, so every historical
training run stays reproducible against the exact bytes it trained on. The guard
lives in :mod:`artifact_registry.blob_client`; this module never tries to bypass it.

**Deduplicated by content.** Loss Runs and renewal policies are re-submitted
constantly with minor changes. Checksumming on arrival prevents paying to label
the same document twice and prevents corpus bloat from near-identical copies.

**Digital vs scanned recorded at ingest.** Detected here because it is cheap
here and expensive later, and because it defines the scanned eval subset that
feeds the ViT escalation gate (arch §3, §15).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, ImmutableBlobError, for_ingestion
from common.constants import ACTIVE_DOC_TYPES, UNCLASSIFIED
from common.ids import build_source_id, is_valid_source_id, parse_source_id

log = logging.getLogger(__name__)

#: Bucket for documents whose type is not yet known. The classifier resolves
#: them later; forcing a guess at ingest would bake a mistake into the source_id,
#: which is the one identifier everything downstream joins on.
#: Re-exported for the CLI and callers; defined in common.constants so
#: common/ids.py and artifact_registry/paths.py can honour it too — they
#: sit below data_pipeline and cannot import from it.
UNCLASSIFIED = UNCLASSIFIED

CHUNK = 1 << 20


class IngestionError(RuntimeError):
    """Raised on an unreadable PDF or an unresolvable document type."""


@dataclass
class IngestResult:
    """What one ingestion run did — reported, never inferred from logs."""

    ingested: list[str] = field(default_factory=list)
    skipped_duplicates: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{len(self.ingested)} ingested"]
        if self.skipped_duplicates:
            parts.append(f"{len(self.skipped_duplicates)} duplicates skipped")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pdf(path: Path) -> dict[str, Any]:
    """Page count and whether the PDF carries an embedded text layer.

    The digital/scanned distinction drives the scanned eval subset (arch §15) and
    is one of the two inputs to the ViT escalation gate. A PDF with no extractable
    text is scanned; a mostly-empty text layer means a scan with a thin OCR layer
    already applied, which is treated as scanned because the model still has to
    read the pixels.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional extra
        raise IngestionError(
            'pypdf is not installed. Install the [data] extra: pip install -e ".[data]"'
        ) from exc

    try:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        sampled = reader.pages[: min(3, pages)]
        text = "".join((p.extract_text() or "") for p in sampled)
    except Exception as exc:
        raise IngestionError(f"could not read {path.name}: {exc}") from exc

    chars_per_page = len(text.strip()) / max(1, len(sampled))
    return {
        "page_count": pages,
        "is_scanned": chars_per_page < 50,
        "text_chars_per_sampled_page": round(chars_per_page, 1),
    }


def _existing_checksums(client: BlobClient, doc_type: str, tenant_id: str | None) -> dict[str, str]:
    """Map checksum -> source_id for what is already ingested for this type."""
    prefix = f"raw-documents/{paths._tenant(tenant_id)}/{doc_type}/"
    seen: dict[str, str] = {}
    for key in client.list(prefix):
        if not key.endswith("metadata.json"):
            continue
        try:
            meta = client.read_json(key)
        except Exception:
            continue
        if checksum := meta.get("checksum_sha256"):
            seen[checksum] = meta.get("source_id", key)
    return seen


def _existing_source_ids(client: BlobClient, doc_type: str, tenant_id: str | None) -> list[str]:
    prefix = f"raw-documents/{paths._tenant(tenant_id)}/{doc_type}/"
    out = set()
    for key in client.list(prefix):
        parts = key.split("/")
        if len(parts) >= 4 and is_valid_source_id(parts[3]):
            out.add(parts[3])
    return sorted(out)


def ingest_pdf(
    pdf_path: Path,
    doc_type: str,
    client: BlobClient,
    *,
    tenant_id: str | None = None,
    source_id: str | None = None,
    known_checksums: dict[str, str] | None = None,
    source_system: str = "manual",
) -> tuple[str | None, str]:
    """Ingest one PDF. Returns ``(source_id, status)``.

    ``status`` is ``"ingested"``, ``"duplicate"``, or ``"failed:<reason>"`` — the
    caller reports what happened rather than inferring it from log output.
    """
    checksum = sha256_file(pdf_path)

    if known_checksums is not None and checksum in known_checksums:
        original = known_checksums[checksum]
        log.info("skipping %s: identical content already ingested as %s", pdf_path.name, original)
        return original, "duplicate"

    if source_id is None:
        existing = _existing_source_ids(client, doc_type, tenant_id)
        indices = [parse_source_id(s).index for s in existing]
        source_id = build_source_id(doc_type, (max(indices) + 1) if indices else 1)

    inspected = inspect_pdf(pdf_path)
    metadata = {
        "source_id": source_id,
        "doc_type": doc_type,
        "tenant_id": paths._tenant(tenant_id),
        "ingested_at": datetime.now(UTC).isoformat(),
        "source_system": source_system,
        "original_filename": pdf_path.name,
        "checksum_sha256": checksum,
        "size_bytes": pdf_path.stat().st_size,
        **inspected,
        # Populated by the de-identification step if and when it is unblocked
        # (SPEC_05 §1). Recorded as unknown rather than omitted, so nothing
        # downstream can mistake absence for "no PII".
        "pii_flags": {"status": "not_assessed"},
        "retention_class": "standard",
    }

    try:
        client.upload_file(pdf_path, paths.raw_pdf(doc_type, source_id, tenant_id))
    except ImmutableBlobError:
        return source_id, "failed:already_exists_write_once"

    client.write_json(paths.raw_metadata(doc_type, source_id, tenant_id), metadata)
    if known_checksums is not None:
        known_checksums[checksum] = source_id
    log.info("ingested %s as %s (%d pages, %s)", pdf_path.name, source_id,
             inspected["page_count"], "scanned" if inspected["is_scanned"] else "digital")
    return source_id, "ingested"


def ingest_directory(
    input_dir: Path,
    doc_type: str,
    client: BlobClient,
    *,
    tenant_id: str | None = None,
    source_system: str = "manual",
) -> IngestResult:
    """Ingest every PDF in a directory. Idempotent and safe to re-run."""
    if doc_type != UNCLASSIFIED and doc_type not in ACTIVE_DOC_TYPES:
        raise IngestionError(
            f"unknown doc_type {doc_type!r}; expected one of {list(ACTIVE_DOC_TYPES)} "
            f"or {UNCLASSIFIED!r} when the type is not yet known"
        )

    result = IngestResult()
    known = _existing_checksums(client, doc_type, tenant_id)

    for pdf_path in sorted(input_dir.rglob("*.pdf")):
        try:
            source_id, status = ingest_pdf(
                pdf_path, doc_type, client, tenant_id=tenant_id,
                known_checksums=known, source_system=source_system,
            )
        except (IngestionError, Exception) as exc:  # noqa: BLE001 - one bad PDF must not stop a batch
            result.failed.append((pdf_path.name, str(exc)))
            log.warning("failed ingesting %s: %s", pdf_path.name, exc)
            continue

        if status == "ingested":
            result.ingested.append(source_id or pdf_path.name)
        elif status == "duplicate":
            result.skipped_duplicates.append((pdf_path.name, source_id or ""))
        else:
            result.failed.append((pdf_path.name, status))

    log.info("ingestion complete: %s", result.summary())
    return result


def purge_plan(source_id: str, doc_type: str, tenant_id: str | None = None) -> dict[str, list[str]]:
    """The full downstream cascade for a purge request (arch §18a).

    Returns the plan; it never deletes. Retaining derived data from a document
    that no longer legally exists is a real audit gap, so a purge has to reach
    ``processed/``, ``golden-labels/`` and any corpus entries built from it — but
    corpus files are shared across documents, so removing a row from one is a
    rebuild, not a delete. That is why this reports rather than acts.
    """
    return {
        "raw": [paths.raw_document_dir(doc_type, source_id, tenant_id)],
        "processed": [paths.processed_dir(doc_type, source_id, tenant_id)],
        "golden_labels": [paths.golden_label_dir(doc_type, source_id, tenant_id)],
        "corpus_action": [
            f"rebuild any corpus version whose manifest lists source_id {source_id!r} — "
            "corpus files are shared, so the row cannot simply be deleted"
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Ingest raw PDFs into raw-documents/")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--doc-type", required=True,
                        choices=[*ACTIVE_DOC_TYPES, UNCLASSIFIED])
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--source-system", default="manual")
    parser.add_argument("--purge-plan", metavar="SOURCE_ID",
                        help="print the downstream cascade for a purge request and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.purge_plan:
        print(json.dumps(purge_plan(args.purge_plan, args.doc_type, args.tenant), indent=2))
        return 0

    result = ingest_directory(
        args.input, args.doc_type, for_ingestion(),
        tenant_id=args.tenant, source_system=args.source_system,
    )
    print(result.summary())
    for name, reason in result.failed:
        print(f"  FAILED {name}: {reason}")
    return 1 if result.failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
