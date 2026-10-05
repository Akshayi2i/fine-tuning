# SPEC_04 — Section Boundary Chunker

**Owner:** Backend / ML Engineering  
**Depends on:** SPEC_00 (Chunk type, schemas), SPEC_05 (MinerU page index)  
**Language:** Python 3.11 · sentence-transformers · tiktoken  
**Files to create:** `fideon/chunker/`  
**Performance target:** < 200 ms total chunking time (CPU) · ≤ 2 s total VLM time for ≤ 150-page docs  
**Triggered when:** MinerU markdown token count > 28,000 tokens (~50 pages)  

---

## 1. Purpose

For documents longer than ~50 pages, sending the full MinerU markdown to the VLM in a single
call would exceed the context window and degrade extraction quality. The Section Boundary
Chunker splits the markdown at semantically meaningful boundaries (claim group headers,
policy year separators) before handing chunks to the VLM. It then merges VLM outputs by
primary key (Claim # / Transaction ID) to reconstruct the full document result.

The chunker runs on CPU in the Pipeline API Service — no GPU required.

---

## 2. Module layout

```
fideon/
  chunker/
    __init__.py
    chunker.py              # main entry point: chunk_document()
    token_counter.py        # estimate_tokens() using tiktoken
    boundary_detector.py    # detect_boundaries() — heuristic + encoder
    sentence_encoder.py     # DomainSentenceEncoder — load, encode, cosine sim
    image_partitioner.py    # assign page images to chunks (scanned PDFs)
    merger.py               # merge_chunks() — primary key reconciliation
    models.py               # Chunk dataclass
    tests/
      test_chunker.py       # uses 3 multi-year loss run fixtures
      test_merger.py
      fixtures/
        loss_run_3yr.md     # 3-year loss run, 90 pages → 3 chunks expected
        loss_run_1yr.md     # 1-year loss run, 20 pages → no chunking expected
```

---

## 3. Data models  (`models.py`)

```python
from dataclasses import dataclass, field

@dataclass
class Chunk:
    index: int                      # 0-based chunk sequence number
    markdown: str                   # markdown text for this chunk
    page_numbers: list[int]         # 1-based source page numbers covered by this chunk
    page_images: list[bytes] = field(default_factory=list)
    # Populated after boundary detection
    boundary_type: str = ""         # "policy_year" | "claim_group" | "page_limit"
    boundary_confidence: float = 0.0

@dataclass
class MinerUOutput:
    markdown: str
    page_index: dict[str, list[int]]   # section_heading → [page_nos]  (from MinerU)
    total_pages: int
```

---

## 4. Main entry point  (`chunker.py`)

```python
from fideon.chunker.token_counter import estimate_tokens
from fideon.chunker.boundary_detector import detect_boundaries
from fideon.chunker.image_partitioner import assign_images

TOKEN_THRESHOLD = 28_000   # ~50 pages of insurance docs
MAX_CHUNK_SIZE  = 30_000   # hard cap per chunk (tokens)

async def chunk_document(
    markdown: str,
    page_index: dict[str, list[int]],
    pdf_bytes: bytes,
    modality: str,            # "native_pdf" or "scanned_pdf"
    max_chunks: int = 10,
) -> list[Chunk]:
    """
    Split `markdown` into semantically coherent chunks for sequential VLM calls.
    Returns list of Chunk objects in document order.
    """
    total_tokens = estimate_tokens(markdown)
    if total_tokens <= TOKEN_THRESHOLD:
        # Should not be called, but defensive: return as single chunk
        return [Chunk(index=0, markdown=markdown,
                      page_numbers=list(range(1, _max_page(page_index) + 1)))]

    # 1. Detect candidate boundaries
    boundaries = detect_boundaries(markdown, page_index)
    # boundaries: list of (char_offset, boundary_type, confidence)

    # 2. Build chunks respecting MAX_CHUNK_SIZE
    chunks = build_chunks(markdown, boundaries, max_chunks)

    # 3. Assign page numbers from page_index
    chunks = assign_page_numbers(chunks, page_index)

    # 4. Assign page images (scanned PDFs only)
    if modality == "scanned_pdf":
        chunks = assign_images(chunks, pdf_bytes)

    return chunks


def build_chunks(markdown: str, boundaries: list[tuple], max_chunks: int) -> list[Chunk]:
    """
    Given candidate boundary offsets, build chunks such that:
    - No chunk exceeds MAX_CHUNK_SIZE tokens
    - Prefer splitting at high-confidence boundaries
    - Split mid-section only if necessary to respect MAX_CHUNK_SIZE
    """
    # Sort boundaries by char offset
    boundaries = sorted(boundaries, key=lambda b: b[0])

    # Greedy: advance through markdown, commit a split when remaining would overflow
    split_offsets = [0]
    current_start = 0
    for offset, btype, conf in boundaries:
        segment = markdown[current_start:offset]
        if estimate_tokens(segment) >= MAX_CHUNK_SIZE:
            # Force split at previous boundary
            split_offsets.append(offset)
            current_start = offset
    split_offsets.append(len(markdown))

    # Safety: if max_chunks exceeded, merge smallest adjacent pairs
    while len(split_offsets) - 1 > max_chunks:
        # Find smallest chunk and merge with its neighbour
        sizes = [estimate_tokens(markdown[split_offsets[i]:split_offsets[i+1]])
                 for i in range(len(split_offsets)-1)]
        smallest = sizes.index(min(sizes))
        split_offsets.pop(smallest + 1)

    chunks = []
    for i in range(len(split_offsets) - 1):
        text = markdown[split_offsets[i]:split_offsets[i+1]]
        chunks.append(Chunk(index=i, markdown=text, page_numbers=[]))

    return chunks
```

---

## 5. Token counter  (`token_counter.py`)

```python
import tiktoken

# Use cl100k_base (same family as GPT-4; good approximation for Qwen3-VL)
_enc = tiktoken.get_encoding("cl100k_base")

def estimate_tokens(text: str) -> int:
    """
    Returns approximate token count for `text`.
    Actual Qwen3-VL tokeniser differs slightly — this approximation errs
    on the conservative side (~5% high), which is intentional.
    """
    return len(_enc.encode(text))
```

---

## 6. Boundary detector  (`boundary_detector.py`)

### 6.1 — Heuristic patterns (fast pass, runs first)

```python
import re

# Patterns that indicate a natural split point in insurance docs
BOUNDARY_PATTERNS = [
    # Policy year separators
    (r"^#{1,3}\s*(Policy\s+Year|Coverage\s+Period)\s*[:\-–]\s*\d{4}", "policy_year", 0.95),
    (r"^#{1,3}\s*\d{4}\s*[-–]\s*\d{4}", "policy_year", 0.90),
    # Claim group headers
    (r"^#{1,3}\s*Claims?\s+(Summary|Detail|List|Report)", "claim_group", 0.88),
    (r"^#{1,3}\s*Open\s+Claims?", "claim_group", 0.85),
    (r"^#{1,3}\s*Closed\s+Claims?", "claim_group", 0.85),
    # Generic section separators in markdown (level-2 headings)
    (r"^##\s+\S", "section_h2", 0.60),
    # Horizontal rules (MinerU emits these between major sections)
    (r"^---\s*$", "horizontal_rule", 0.55),
]

def detect_heuristic_boundaries(markdown: str) -> list[tuple[int, str, float]]:
    """
    Returns list of (char_offset, boundary_type, confidence).
    Only returns HIGH-confidence boundaries from heuristics (≥ 0.80).
    Low-confidence boundaries are passed to the sentence encoder for confirmation.
    """
    results = []
    for i, line in enumerate(markdown.splitlines(keepends=True)):
        offset = sum(len(l) for l in markdown.splitlines(keepends=True)[:i])
        for pattern, btype, conf in BOUNDARY_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                results.append((offset, btype, conf))
                break
    return results
```

### 6.2 — Sentence encoder confirmation (runs on low-confidence candidates)

```python
from fideon.chunker.sentence_encoder import DomainSentenceEncoder
import numpy as np

_encoder: DomainSentenceEncoder = None

def get_encoder() -> DomainSentenceEncoder:
    global _encoder
    if _encoder is None:
        _encoder = DomainSentenceEncoder()
    return _encoder

COSINE_THRESHOLD = 0.35   # below this → genuine semantic boundary

def confirm_boundary_with_encoder(
    markdown: str,
    candidate_offset: int,
    context_chars: int = 300,
) -> float:
    """
    Compute cosine similarity between text immediately before and after candidate_offset.
    Low similarity → confirmed boundary.
    Returns adjusted confidence (0..1).
    """
    before = markdown[max(0, candidate_offset - context_chars):candidate_offset]
    after  = markdown[candidate_offset:candidate_offset + context_chars]
    if not before.strip() or not after.strip():
        return 0.5   # insufficient context

    enc = get_encoder()
    emb_before = enc.encode(before)
    emb_after  = enc.encode(after)

    cosine = float(np.dot(emb_before, emb_after) /
                   (np.linalg.norm(emb_before) * np.linalg.norm(emb_after) + 1e-8))

    # Low cosine → high confidence it's a real boundary
    boundary_confidence = 1.0 - cosine
    return round(boundary_confidence, 4)


def detect_boundaries(markdown: str,
                      page_index: dict[str, list[int]]) -> list[tuple[int, str, float]]:
    """
    Full boundary detection pipeline:
    1. Fast heuristic scan
    2. Encoder confirmation for medium-confidence heuristic candidates
    3. Inject page_index anchors as high-confidence fallback boundaries
    Returns list of (char_offset, boundary_type, confidence) sorted by offset.
    """
    # Step 1: heuristic
    heuristic = detect_heuristic_boundaries(markdown)
    confirmed = [b for b in heuristic if b[2] >= 0.80]   # high conf → accept as-is
    candidates = [b for b in heuristic if b[2] < 0.80]   # low conf → encoder check

    # Step 2: encoder confirmation
    for offset, btype, _ in candidates:
        conf = confirm_boundary_with_encoder(markdown, offset)
        if conf >= COSINE_THRESHOLD:
            confirmed.append((offset, btype, conf))

    # Step 3: page_index anchors (every N pages, inject a fallback boundary)
    PAGE_ANCHOR_INTERVAL = 50   # every 50 pages, force a boundary
    # ... map page numbers to markdown char offsets via page_index

    return sorted(confirmed, key=lambda b: b[0])
```

---

## 7. Domain Sentence Encoder  (`sentence_encoder.py`)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINE_TUNED_MODEL_PATH = "models/domain_sentence_encoder"   # fine-tuned on P&C insurance corpora

class DomainSentenceEncoder:
    """
    Lightweight embedding model used only for chunk boundary detection.
    NOT the DAPT BPE tokeniser — a separate, serving-time component.
    """

    def __init__(self):
        # Load fine-tuned version if available; fall back to base
        import os
        model_path = FINE_TUNED_MODEL_PATH if os.path.exists(FINE_TUNED_MODEL_PATH) else BASE_MODEL
        self._model = SentenceTransformer(model_path)

    def encode(self, text: str) -> np.ndarray:
        """Returns L2-normalised embedding vector."""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding

    # ── Fine-tuning spec (done in Phase 3, not at serving time) ──────────────
    #
    # Dataset: pairs of (anchor_text, positive_text) where anchor and positive
    #          are consecutive paragraphs from the SAME policy-year section,
    #          and (anchor_text, negative_text) are from DIFFERENT policy-year sections.
    # Loss:    MultipleNegativesRankingLoss
    # Epochs:  3-5 on ~5,000 pairs drawn from de-identified archive
    # Output:  saved to models/domain_sentence_encoder/
    #
    # Training is offline (Phase 3 of pre-production lifecycle).
    # The serving model is frozen — never updated mid-serving.
```

---

## 8. Image partitioner  (`image_partitioner.py`)

```python
import fitz

def assign_images(chunks: list[Chunk], pdf_bytes: bytes) -> list[Chunk]:
    """
    For scanned PDFs: render each page as a JPEG and assign to the correct chunk
    based on the chunk's page_numbers list (populated by assign_page_numbers()).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Render all pages once, store in a dict
    page_images: dict[int, bytes] = {}
    for page_num in range(1, len(doc) + 1):
        page = doc[page_num - 1]
        mat = fitz.Matrix(2.0, 2.0)   # 2x zoom → ~144 DPI effective
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        page_images[page_num] = pix.tobytes("jpeg")

    for chunk in chunks:
        chunk.page_images = [page_images[p] for p in chunk.page_numbers if p in page_images]

    return chunks
```

---

## 9. Primary-key merger  (`merger.py`)

```python
def merge_chunks(raw_outputs: list[dict], document_type: str) -> dict:
    """
    Merge VLM outputs from N sequential chunks into a single document dict.
    Strategy depends on document_type.
    """
    if document_type == "loss_run":
        return _merge_loss_run(raw_outputs)
    elif document_type == "policy_check":
        return _merge_policy_check(raw_outputs)
    else:
        # Default: take first chunk's top-level fields, concatenate list fields
        return _merge_generic(raw_outputs)


def _merge_loss_run(outputs: list[dict]) -> dict:
    """
    Primary key: claim_number (unique per claim across the whole document).
    Strategy:
    - Top-level fields (insured_name, run_date) from first chunk (or highest-confidence)
    - periods: merge by policy_number + policy_year; claims within each period merged by claim_number
    - Aggregate totals recomputed from merged claims
    """
    base = outputs[0].copy()
    seen_claims: dict[str, dict] = {}     # claim_number.raw → claim dict

    for output in outputs:
        for period in output.get("periods", []):
            for claim in period.get("claims", []):
                key = _extract_key(claim, "claim_number")
                if key is None:
                    continue
                if key not in seen_claims:
                    seen_claims[key] = claim
                else:
                    # Conflict resolution: take higher-confidence version
                    existing_conf = _avg_confidence(seen_claims[key])
                    new_conf = _avg_confidence(claim)
                    if new_conf > existing_conf:
                        seen_claims[key] = claim

    # Reassemble periods
    # ... group seen_claims back into periods by policy_year
    # Recompute aggregate totals
    return _reassemble(base, seen_claims)


def _extract_key(record: dict, key_field: str) -> str | None:
    fv = record.get(key_field, {})
    return fv.get("raw") or fv.get("parsed")


PRIMARY_KEYS = {
    "loss_run":      "claim_number",
    "policy_check":  None,              # no primary key; take union of coverages
    "quote_gen":     None,
    "acord_mapping": "field_id",
}
```

---

## 10. Page number assignment

```python
def assign_page_numbers(chunks: list[Chunk],
                        page_index: dict[str, list[int]]) -> list[Chunk]:
    """
    Use MinerU's page_index to map each chunk's markdown content to source page ranges.
    page_index: {section_heading_text → [page_numbers]}
    """
    for chunk in chunks:
        chunk_pages = set()
        # Find all section headings that appear in this chunk's markdown
        for heading, pages in page_index.items():
            if heading in chunk.markdown:
                chunk_pages.update(pages)
        chunk.page_numbers = sorted(chunk_pages)
        # Fallback: if no headings found in chunk, estimate from position
        if not chunk.page_numbers:
            chunk.page_numbers = _estimate_pages_from_position(chunk, chunks, page_index)
    return chunks
```

---

## 11. Acceptance criteria

- [ ] A 90-page 3-year loss run produces exactly 3 chunks (one per policy year)
- [ ] A 20-page single-year loss run is NOT chunked (token count below threshold)
- [ ] All chunks together contain 100% of the claims in the source document (no lost rows)
- [ ] `merge_chunks()` with 3 chunks containing overlapping claim numbers deduplicates correctly
- [ ] Page images are assigned to chunks such that each chunk contains only its own pages
- [ ] `DomainSentenceEncoder.encode()` returns a unit-norm vector
- [ ] Chunking completes in < 200 ms for a 90-page document
- [ ] Boundary detection finds policy-year headings in the 3-year fixture with confidence ≥ 0.85
- [ ] Merger preserves the highest-confidence version when the same claim_number appears in two chunks
