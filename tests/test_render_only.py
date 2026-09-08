"""SPEC_03 — the ``image_only`` rendering path.

``image_only`` is 30% of the Foundation corpus and a real production mode, so
these pages have to be as traceable as OCR'd ones: same resolution cap, same
version pinning, same provenance skeleton. A rendering path that quietly used a
different cap would teach the model that the two modes look different — a
distinction that does not exist at inference.
"""

from __future__ import annotations

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.ocr import render_only
from data_pipeline.ocr.render_only import RenderError, render_document


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(
        backend=InMemoryBackend(), container="main", raw_container="raw", context="ocr"
    )


@pytest.fixture
def seeded(client) -> BlobClient:
    client.write_bytes(paths.raw_pdf("policy", "policy_0001"), b"%PDF-1.7 fixture")
    client.write_json(paths.raw_metadata("policy", "policy_0001"),
                      {"source_id": "policy_0001", "checksum_sha256": "sha-abc"})
    return client


@pytest.fixture
def two_pages(monkeypatch):
    """Stand in for PyMuPDF, which is an optional extra and needs a real PDF."""
    monkeypatch.setattr(render_only, "render_pdf_pages",
                        lambda pdf_bytes, cap: [b"\x89PNG-1", b"\x89PNG-2"])


def test_rendering_writes_a_page_image_per_page(seeded, two_pages):
    meta = render_document("policy", "policy_0001", seeded)

    assert meta["page_count"] == 2
    for page in (1, 2):
        assert seeded.exists(paths.processed_page("policy", "policy_0001", page, "png"))


def test_no_markdown_is_written_for_an_image_only_document(seeded, two_pages):
    """The whole point of the mode: pixels, no text."""
    render_document("policy", "policy_0001", seeded)
    assert not seeded.exists(paths.processed_page("policy", "policy_0001", 1, "md"))


def test_the_metadata_marks_the_document_as_render_only(seeded, two_pages):
    meta = render_document("policy", "policy_0001", seeded)
    assert meta["render_only"] is True
    assert meta["table_row_counts"] == {}, "row counts cannot exist without OCR"


def test_an_image_only_document_is_as_traceable_as_an_ocr_one(seeded, two_pages):
    """Same version pin, same resolution record, same provenance — otherwise
    image_only documents become the untracked third of the corpus."""
    meta = render_document("policy", "policy_0001", seeded)
    for key in ("mineru_version", "ocr_device", "resolution_cap_px",
                "preprocessing_date", "source_checksum"):
        assert meta.get(key) is not None, f"{key} missing from an image_only document"
    assert meta["source_checksum"] == "sha-abc"


def test_the_resolution_cap_is_the_shared_one(seeded, two_pages):
    from common.config import resolution_cap_px

    meta = render_document("policy", "policy_0001", seeded)
    assert meta["resolution_cap_px"] == resolution_cap_px()


def test_rendering_is_idempotent_at_the_same_cap(seeded, monkeypatch):
    calls = {"n": 0}

    def counted(_pdf_bytes, _cap):
        calls["n"] += 1
        return [b"\x89PNG"]

    monkeypatch.setattr(render_only, "render_pdf_pages", counted)
    render_document("policy", "policy_0001", seeded)
    render_document("policy", "policy_0001", seeded)
    assert calls["n"] == 1, "a re-run re-rendered pages that were already correct"


def test_a_cap_change_forces_re_rendering(seeded, monkeypatch):
    """A different cap is a different input distribution, not a cosmetic change."""
    calls = {"n": 0}

    def counted(_pdf_bytes, _cap):
        calls["n"] += 1
        return [b"\x89PNG"]

    monkeypatch.setattr(render_only, "render_pdf_pages", counted)
    render_document("policy", "policy_0001", seeded)

    monkeypatch.setattr(render_only, "resolution_cap_px", lambda: 1536)
    render_document("policy", "policy_0001", seeded)
    assert calls["n"] == 2


def test_force_re_renders_even_when_nothing_changed(seeded, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        render_only, "render_pdf_pages",
        lambda *_a: (calls.__setitem__("n", calls["n"] + 1), [b"\x89PNG"])[1],
    )
    render_document("policy", "policy_0001", seeded)
    render_document("policy", "policy_0001", seeded, force=True)
    assert calls["n"] == 2


def test_a_document_that_renders_no_pages_is_an_error(seeded, monkeypatch):
    """Zero pages must not be recorded as a successfully rendered document."""
    monkeypatch.setattr(render_only, "render_pdf_pages",
                        lambda *_a: (_ for _ in ()).throw(RenderError("PDF produced no pages")))
    with pytest.raises(RenderError, match="no pages"):
        render_document("policy", "policy_0001", seeded)
    assert not seeded.exists(paths.ocr_meta("policy", "policy_0001"))


def test_an_ocr_meta_from_the_ocr_path_is_not_mistaken_for_a_render(seeded, two_pages):
    """An OCR'd document lacks `render_only`, so the skip check must re-render
    rather than assume the images are already there."""
    from common.config import resolution_cap_px

    seeded.write_json(paths.ocr_meta("policy", "policy_0001"), {
        "source_id": "policy_0001", "page_count": 2,
        "resolution_cap_px": resolution_cap_px(),
    })
    meta = render_document("policy", "policy_0001", seeded)
    assert meta["render_only"] is True
    assert seeded.exists(paths.processed_page("policy", "policy_0001", 1, "png"))


@pytest.fixture(autouse=True)
def _on_a_gpu_pod(monkeypatch):
    """OCR and rendering are GPU-only, so CI must simulate a GPU pod.

    Autouse rather than opt-in: every test in this module exercises a path that
    now refuses to run without CUDA, and that refusal is the point — a silent
    CPU fallback would write markdown from a different distribution.
    """
    from data_pipeline.ocr import mineru_version

    monkeypatch.setattr(mineru_version, "cuda_available", lambda: (True, "NVIDIA L4"))



# --------------------------------------------------------------------------
# The scale cap itself
# --------------------------------------------------------------------------


class _FakeRect:
    def __init__(self, width: float, height: float) -> None:
        self.width, self.height = width, height


def test_the_scale_puts_the_long_side_on_the_cap():
    """`page.rect` is in points, so the factor is cap/longest — anything else
    renders at a resolution nobody chose."""
    letter = _FakeRect(612, 792)          # US Letter, in points
    longest = max(letter.width, letter.height)
    scale = 1792 / longest

    assert round(longest * scale) == 1792
    assert round(letter.width * scale) == 1385   # aspect ratio preserved
    assert scale == pytest.approx(2.2626, abs=1e-4)


def test_a_small_page_is_still_capped_not_clamped_to_seventy_two_dpi():
    """A guard of cap/72 would silently switch small pages to a different rule."""
    tiny = _FakeRect(144, 216)            # 2x3 inches
    scale = 1792 / max(tiny.width, tiny.height)
    assert round(216 * scale) == 1792


def test_the_renderer_names_the_extra_that_actually_provides_it(monkeypatch):
    """The remediation must point at an extra that installs PyMuPDF — an error
    message naming the wrong one costs an afternoon."""
    import builtins
    import tomllib
    from pathlib import Path

    real_import = builtins.__import__

    def no_fitz(name, *args, **kwargs):
        if name == "fitz":
            raise ImportError("no module named fitz")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_fitz)
    with pytest.raises(RenderError, match=r"\[data\] extra"):
        render_only.render_pdf_pages(b"%PDF", 1792)

    monkeypatch.undo()
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    extra = " ".join(pyproject["project"]["optional-dependencies"]["data"]).lower()
    assert "pymupdf" in extra
