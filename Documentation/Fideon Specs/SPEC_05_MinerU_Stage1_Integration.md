# SPEC_05 — Stage 1: MinerU Integration (Adaptive Physical Parser)

**Owner:** ML Engineering  
**Depends on:** SPEC_00 (MinerUOutput type), SPEC_04 (page_index consumed by chunker)  
**Language:** Python 3.11 · magic-pdf (MinerU) · PyMuPDF  
**Files to create:** `fideon/stage1/`  
**Performance target:** < 30 s for 50-page document · CPU for native PDF · GPU optional for scanned  
**Called by:** L3 service orchestrator (SPEC_01 § 8)  

---

## 1. Purpose

Stage 1 converts raw PDF bytes into structured markdown plus a page-to-markdown index.
Two code paths share the same output contract:

- **Native PDF path** (vector text): layout analysis → structured markdown. Fast, no OCR.
- **Scanned PDF path** (image-based): 300 DPI raster → OCR → markdown. Slower; GPU improves throughput but is not required.

The page index produced here is critical input for the Section Boundary Chunker (SPEC_04)
and for assigning page images to chunks.

---

## 2. Module layout

```
fideon/
  stage1/
    __init__.py
    mineru_runner.py        # async entry point: run_mineru()
    native_path.py          # native PDF → markdown via MinerU layout mode
    scanned_path.py         # scanned PDF → OCR → markdown
    page_index_builder.py   # constructs page_index dict from MinerU output
    output_normaliser.py    # clean and normalise MinerU markdown
    models.py               # MinerUOutput dataclass (re-exported from SPEC_04)
    tests/
      test_native_path.py
      test_scanned_path.py
      fixtures/
        native_loss_run.pdf      # known native PDF with expected markdown
        scanned_loss_run.pdf     # known scanned PDF with expected text
```

---

## 3. MinerUOutput contract (shared with SPEC_04)

```python
@dataclass
class MinerUOutput:
    markdown: str
    page_index: dict[str, list[int]]   # heading_text → [page_numbers]
    total_pages: int
    path_used: Literal["native", "scanned"]
    duration_ms: int
    errors: list[str] = field(default_factory=list)
```

---

## 4. Main async entry point  (`mineru_runner.py`)

```python
import asyncio
import time
from fideon.stage1.native_path import run_native
from fideon.stage1.scanned_path import run_scanned
from fideon.stage1.models import MinerUOutput

NATIVE_TIMEOUT_S  = 30
SCANNED_TIMEOUT_S = 90   # OCR is slower; GPU cuts this to ~20s

async def run_mineru(pdf_bytes: bytes, modality: str,
                     force_ocr: bool = False) -> MinerUOutput:
    """
    Select processing path based on modality, run MinerU, return normalised output.
    Raises MinerUError on hard failure.

    `force_ocr` takes the scanned path for a document that IS native. See
    "The third case" below: a native PDF whose figures are drawn rather than
    typed must not be sent down the native path, because that path re-reads
    the very text layer that does not carry its numbers.
    """
    t0 = time.monotonic()
    try:
        if modality == "native_pdf" and not force_ocr:
            result = await asyncio.wait_for(
                asyncio.to_thread(run_native, pdf_bytes),
                timeout=NATIVE_TIMEOUT_S,
            )
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(run_scanned, pdf_bytes),
                timeout=SCANNED_TIMEOUT_S,
            )
    except asyncio.TimeoutError:
        raise MinerUError(f"MinerU timed out after {NATIVE_TIMEOUT_S}s")

    result.duration_ms = int((time.monotonic() - t0) * 1000)
    return result


class MinerUError(RuntimeError):
    pass
```

### 4.1 The third case — native, but the data is drawn

Modality has two values and this path has three cases. SPEC_01 §7.2 describes
the third: a PDF that carries a real text layer, is correctly classified
`native_pdf`, and **draws its figures as filled vector outlines** instead of
typing them. NYCM's homeowner summary prints `LOCATION COVERAGES` as text and
draws Coverage A 360,000 through F 1,000 beneath it; one coverage band alone
carries 37 readable text spans against 380 glyph-sized vector paths.

Such a document is escalated by L1's plausibility gate, which reports that the
figures are drawn and that "this needs OCR or the VLM, not a better field map".
It then arrives here with `modality == "native_pdf"` — because it genuinely is
native, and nothing rewrites that — and the selector sends it to `run_native`,
which extracts the same text layer L1 already found wanting.

**The escalation lands on the one path that cannot fix it.** The document goes
to the GPU, costs a VLM call, and comes back missing the same numbers.

The caller is therefore responsible for passing `force_ocr=True` when
`fideon.routing.step0.data_is_drawn()` is true for the document:

```python
from fideon.routing.step0 import data_is_drawn

mineru = await run_mineru(pdf_bytes, meta.modality,
                          force_ocr=meta.data_is_drawn)
```

**WHY a flag rather than reclassifying the modality as `scanned_pdf`.** The
modality is consumed by more than this selector. SPEC_04 renders page images
only for `scanned_pdf`; SPEC_02 and SPEC_03 both skip their rungs on it; SPEC_18
carries it on every miss event and SPEC_14 logs it as a metric dimension.
Relabelling a native document would suppress the registry-miss event for a
carrier we genuinely failed to read, and would record the wrong modality in the
training corpus (SPEC_16). The document *is* native. What differs is that OCR
is the right reader for it, and that is what the flag says.

`force_ocr` therefore takes `SCANNED_TIMEOUT_S`, not `NATIVE_TIMEOUT_S` — it is
doing the scanned path's work and needs the scanned path's budget.

---

## 5. Native PDF path  (`native_path.py`)

```python
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.make_content_config import DropMode, MakeMode

import tempfile, os

def run_native(pdf_bytes: bytes) -> MinerUOutput:
    """
    MinerU 'auto' mode for native PDFs.
    Uses layout analysis + bounding-box text extraction (no OCR).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "input.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        writer = FileBasedDataWriter(tmpdir)
        dataset = PymuDocDataset(pdf_bytes)
        
        # MinerU pipeline: analyze → extract → write markdown
        infer_result = dataset.apply(doc_analyze, is_debug=False)
        pipe_result = infer_result.pipe_txt_mode(writer, debug_mode=False)

        # Read generated markdown
        md_path = os.path.join(tmpdir, "input.md")
        with open(md_path, "r", encoding="utf-8") as f:
            raw_markdown = f.read()

        total_pages = dataset.get_data_bits()   # page count
        page_index = build_page_index(raw_markdown, total_pages)
        markdown = normalise_markdown(raw_markdown)

    return MinerUOutput(
        markdown=markdown,
        page_index=page_index,
        total_pages=total_pages,
        path_used="native",
        duration_ms=0,   # filled in by run_mineru
    )
```

---

## 6. Scanned PDF path  (`scanned_path.py`)

```python
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

import fitz

OCR_DPI = 300

def run_scanned(pdf_bytes: bytes) -> MinerUOutput:
    """
    MinerU OCR mode for scanned PDFs.
    Rasters each page at 300 DPI, runs OCR, assembles markdown.

    GPU note: MinerU uses docling/PaddleOCR under the hood.
    If CUDA is available, OCR runs on GPU automatically.
    If not, falls back to CPU (slower but correct).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "input.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Raster at 300 DPI for OCR quality
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        raster_paths = []
        for i, page in enumerate(doc):
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_path = os.path.join(tmpdir, f"page_{i+1:04d}.png")
            pix.save(img_path)
            raster_paths.append(img_path)

        # MinerU OCR pipeline
        writer = FileBasedDataWriter(tmpdir)
        dataset = PymuDocDataset(pdf_bytes)
        infer_result = dataset.apply(doc_analyze, is_debug=False)
        pipe_result = infer_result.pipe_ocr_mode(writer, debug_mode=False)

        md_path = os.path.join(tmpdir, "input.md")
        with open(md_path, "r", encoding="utf-8") as f:
            raw_markdown = f.read()

        total_pages = len(doc)
        page_index = build_page_index(raw_markdown, total_pages)
        markdown = normalise_markdown(raw_markdown)

    return MinerUOutput(
        markdown=markdown,
        page_index=page_index,
        total_pages=total_pages,
        path_used="scanned",
        duration_ms=0,
    )
```

---

## 7. Page index builder  (`page_index_builder.py`)

```python
import re

# MinerU annotates its output with page markers in the form:
#   <!-- page N -->   or   \n\n---\nPage N\n---\n
# The exact format depends on MinerU version — detect both.

PAGE_MARKER_PATTERNS = [
    re.compile(r"<!--\s*page\s+(\d+)\s*-->", re.IGNORECASE),
    re.compile(r"^\s*Page\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE),
]

HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

def build_page_index(markdown: str, total_pages: int) -> dict[str, list[int]]:
    """
    Parse MinerU page markers to build a mapping:
      section_heading → [page_numbers]

    Algorithm:
    1. Find all page markers and their char offsets → {offset: page_num}
    2. Find all headings and their char offsets
    3. For each heading, find the nearest page marker before it → assign page_num
    4. Headings spanning multiple pages: assign both page numbers
    """
    # Step 1: collect page marker positions
    page_positions: list[tuple[int, int]] = []   # (char_offset, page_num)
    for pattern in PAGE_MARKER_PATTERNS:
        for m in pattern.finditer(markdown):
            page_positions.append((m.start(), int(m.group(1))))
    page_positions.sort(key=lambda x: x[0])

    if not page_positions:
        # MinerU did not emit page markers — fallback: divide equally
        return _fallback_index(markdown, total_pages)

    # Step 2: collect headings
    index: dict[str, list[int]] = {}
    for m in HEADING_PATTERN.finditer(markdown):
        heading_offset = m.start()
        heading_text = m.group(2).strip()
        # Find last page marker before this heading
        pages_before = [pn for (po, pn) in page_positions if po <= heading_offset]
        page_num = pages_before[-1] if pages_before else 1
        if heading_text not in index:
            index[heading_text] = []
        if page_num not in index[heading_text]:
            index[heading_text].append(page_num)

    return index


def _fallback_index(markdown: str, total_pages: int) -> dict[str, list[int]]:
    """
    When MinerU emits no page markers, estimate page from char position.
    Assumes roughly equal chars per page.
    """
    chars_per_page = max(len(markdown) / total_pages, 1)
    index: dict[str, list[int]] = {}
    for m in HEADING_PATTERN.finditer(markdown):
        heading_text = m.group(2).strip()
        estimated_page = int(m.start() / chars_per_page) + 1
        index[heading_text] = [min(estimated_page, total_pages)]
    return index
```

---

## 8. Markdown normaliser  (`output_normaliser.py`)

```python
import re

def normalise_markdown(raw: str) -> str:
    """
    Clean MinerU markdown for VLM consumption:
    - Strip internal page markers (the VLM does not need them)
    - Collapse excessive blank lines (> 2 consecutive → 2)
    - Normalise table formatting (ensure | delimiters are consistent)
    - Remove MinerU debug comments
    - Preserve all content (never delete text)
    """
    # Remove page markers
    for pattern in PAGE_MARKER_PATTERNS:
        raw = pattern.sub("", raw)

    # Remove HTML comments
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)

    # Collapse > 2 blank lines
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    # Normalise table cells: strip extra spaces inside |...|
    lines = []
    for line in raw.splitlines():
        if "|" in line:
            cells = line.split("|")
            line = "|".join(c.strip() for c in cells)
        lines.append(line)
    raw = "\n".join(lines)

    return raw.strip()
```

---

## 9. Error handling

| Failure mode | Behaviour |
|---|---|
| MinerU subprocess crashes | Raise `MinerUError`; Pipeline API returns `ErrorCode.MINERU_FAILED`; document goes to HITL |
| OCR produces empty text (< 10 chars) | Log warning, raise `MinerUError`; HITL |
| Timeout (native > 30s, scanned > 90s) | Raise `asyncio.TimeoutError` caught by caller; HITL |
| Page index is empty (no markers) | Use `_fallback_index()`; log warning; continue (don't fail) |
| MinerU version mismatch | Check `magic_pdf.__version__` at startup; log error if not in SUPPORTED_VERSIONS |

```python
SUPPORTED_MINERU_VERSIONS = {"0.8.x", "0.9.x"}   # update as tested
```

---

## 10. Acceptance criteria

- [ ] Native 50-page loss run produces valid markdown in < 30 s on CPU
- [ ] Scanned 50-page loss run produces OCR markdown with > 90% character accuracy vs ground truth
- [ ] `build_page_index()` correctly assigns section headings to page numbers in 5 test fixtures
- [ ] `normalise_markdown()` strips all page markers and HTML comments
- [ ] Empty/corrupt PDF raises `MinerUError` (not an unhandled exception)
- [ ] `run_mineru()` is cancellable via `asyncio.CancelledError` (does not orphan subprocesses)
- [ ] Native path and scanned path both produce the same output shape (`MinerUOutput` dataclass)
- [ ] `run_mineru(..., force_ocr=True)` on a NATIVE pdf takes the scanned path and
      the scanned timeout — the drawn-figure case of §4.1
- [ ] The NYCM homeowner summary (native, figures drawn) yields its coverage limits
      through Stage 1 when `force_ocr=True`, and does not when it is False; a
      document escalated for drawn data and then read on the native path is the
      defect this criterion exists to catch
