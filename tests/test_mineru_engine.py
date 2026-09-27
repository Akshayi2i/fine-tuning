"""The MinerU engine: one markdown string per page, rows counted in HTML tables,
GPU only. MinerU itself is a stand-in here; the Phase 0 spike runs the real one."""

from __future__ import annotations

import sys
import types

import pytest

from data_pipeline.ocr.run_mineru import (
    MinerUEngine,
    OcrError,
    count_table_rows,
    pages_from_content_list,
)

# --------------------------------------------------------------------------
# Row counting — MinerU writes HTML tables
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,rows", [
    ("<table><tr><th>Claim</th><th>Paid</th></tr><tr><td>1</td><td>5</td></tr>"
     "<tr><td>2</td><td>7</td></tr></table>", 2),
    ("<table><tr><td>Claim</td></tr><tr><td>1</td></tr></table>", 1),       # first row is the header
    ("<table><tr><th>a</th></tr></table><p/><table><tr><th>b</th></tr><tr><td>1</td></tr></table>", 1),
    ("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |", 2),                          # pipe tables still count
    ("no tables here", 0),
])
def test_table_rows_are_counted_in_html_and_pipe_tables(text, rows):
    assert count_table_rows(text) == rows


# --------------------------------------------------------------------------
# One markdown string per page
# --------------------------------------------------------------------------


CONTENT = [
    {"type": "text", "text": "COMMON POLICY DECLARATIONS", "text_level": 1, "page_idx": 0},
    {"type": "text", "text": "Named Insured: Rivera Fabrication LLC", "page_idx": 0},
    {"type": "image", "img_path": "images/x.jpg", "img_caption": ["Carrier logo"], "page_idx": 0},
    {"type": "table", "table_body": "<table><tr><th>Vehicle</th></tr><tr><td>VIN 1</td></tr></table>",
     "table_caption": ["Schedule of Vehicles"], "table_footnote": ["* garaged in NY"], "page_idx": 2},
    {"type": "equation", "text": "$x=1$", "page_idx": 2},
    {"type": "text", "text": "ignored: out of range", "page_idx": 9},
]


def test_blocks_become_one_markdown_string_per_page():
    pages = pages_from_content_list(CONTENT, 3)
    assert pages[0] == ("# COMMON POLICY DECLARATIONS\n\nNamed Insured: Rivera Fabrication LLC"
                        "\n\nCarrier logo")
    assert pages[1] == ""                          # MinerU found nothing on page 2
    assert pages[2].startswith("Schedule of Vehicles\n\n<table>")
    assert "* garaged in NY" in pages[2] and "$x=1$" in pages[2]
    assert "images/x.jpg" not in pages[0]         # the picture is in the page image


# --------------------------------------------------------------------------
# The engine, with MinerU stood in
# --------------------------------------------------------------------------


def _pdf(pages: int) -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page(width=612, height=792).insert_text((72, 72), f"page {i + 1}")
    return doc.tobytes()


def _fake_mineru(monkeypatch, content, *, scanned=False):
    calls = {}

    class Method:
        OCR = "ocr"
        TXT = "txt"

    class Piped:
        def get_content_list(self, image_dir):
            return content

    class Inferred:
        def pipe_ocr_mode(self, writer):
            calls["mode"] = "ocr"
            return Piped()

        def pipe_txt_mode(self, writer):
            calls["mode"] = "txt"
            return Piped()

    class Dataset:
        def __init__(self, pdf_bytes):
            calls["bytes"] = len(pdf_bytes)

        def classify(self):
            return Method.OCR if scanned else Method.TXT

        def apply(self, fn, ocr):
            calls["ocr"] = ocr
            return Inferred()

    modules = {
        "magic_pdf": types.ModuleType("magic_pdf"),
        "magic_pdf.config": types.ModuleType("magic_pdf.config"),
        "magic_pdf.config.enums": types.SimpleNamespace(SupportedPdfParseMethod=Method),
        "magic_pdf.data": types.ModuleType("magic_pdf.data"),
        "magic_pdf.data.data_reader_writer": types.SimpleNamespace(FileBasedDataWriter=lambda d: d),
        "magic_pdf.data.dataset": types.SimpleNamespace(PymuDocDataset=Dataset),
        "magic_pdf.model": types.ModuleType("magic_pdf.model"),
        "magic_pdf.model.doc_analyze_by_custom_model": types.SimpleNamespace(doc_analyze=object()),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("common.gpu.require_cuda", lambda what: "NVIDIA H100")
    return calls


@pytest.mark.parametrize("scanned,mode", [(True, "ocr"), (False, "txt")])
def test_the_engine_returns_a_page_per_page_with_its_image(monkeypatch, scanned, mode):
    calls = _fake_mineru(monkeypatch, CONTENT, scanned=scanned)
    pages = MinerUEngine().process(_pdf(3), device="cuda", max_long_side_px=800)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert calls["mode"] == mode and calls["ocr"] is scanned
    assert pages[0].markdown.startswith("# COMMON POLICY DECLARATIONS")
    # An empty page is an OCR failure only on a scan; on a text layer it is blank.
    assert pages[1].ocr_failed is scanned and not pages[0].ocr_failed
    assert all(p.scanned is scanned for p in pages)
    assert pages[2].table_row_count == 1
    assert all(p.image_bytes.startswith(b"\x89PNG") for p in pages)


def test_a_content_list_returned_as_json_text_is_read(monkeypatch):
    import json

    _fake_mineru(monkeypatch, json.dumps(CONTENT))
    pages = MinerUEngine().process(_pdf(3), device="cuda", max_long_side_px=800)
    assert pages[0].markdown.startswith("# COMMON")


def test_the_engine_refuses_the_cpu():
    with pytest.raises(OcrError, match="GPU only"):
        MinerUEngine().process(b"%PDF", device="cpu", max_long_side_px=800)  # type: ignore[arg-type]


def test_the_engine_refuses_without_cuda(monkeypatch):
    from common.gpu import GPUError

    def no_gpu(what):
        raise GPUError(f"{what} runs on the GPU only")

    monkeypatch.setattr("common.gpu.require_cuda", no_gpu)
    with pytest.raises(OcrError, match="GPU only"):
        MinerUEngine().process(b"%PDF", device="cuda", max_long_side_px=800)
