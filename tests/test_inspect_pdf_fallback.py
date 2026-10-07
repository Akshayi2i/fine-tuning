"""A PDF pypdf gives up on is read with PyMuPDF (data_pipeline.ingestion.pull_raw_pdfs.inspect_pdf)."""

from __future__ import annotations

import pytest

pypdf = pytest.importorskip("pypdf")
pymupdf = pytest.importorskip("pymupdf")


def _pdf(path, pages=2, text="Declarations page of a homeowners policy, typed and readable."):
    document = pymupdf.open()
    for _ in range(pages):
        document.new_page().insert_text((72, 72), text)
    document.save(path)
    return path


def _refusing_reader(*_args, **_kwargs):
    raise pypdf.errors.PdfReadError("Unexpected end of stream.")


def test_a_pdf_pypdf_cannot_read_is_read_with_pymupdf(tmp_path, monkeypatch):
    from data_pipeline.ingestion import pull_raw_pdfs

    monkeypatch.setattr(pypdf, "PdfReader", _refusing_reader)
    inspected = pull_raw_pdfs.inspect_pdf(_pdf(tmp_path / "document.pdf"))
    assert inspected["page_count"] == 2 and inspected["is_scanned"] is False


def test_a_pdf_neither_reader_can_open_is_refused(tmp_path, monkeypatch):
    from data_pipeline.ingestion import pull_raw_pdfs

    monkeypatch.setattr(pypdf, "PdfReader", _refusing_reader)
    broken = tmp_path / "document.pdf"
    broken.write_bytes(b"%PDF-1.7 not a pdf")
    with pytest.raises(pull_raw_pdfs.IngestionError, match="Unexpected end of stream"):
        pull_raw_pdfs.inspect_pdf(broken)
