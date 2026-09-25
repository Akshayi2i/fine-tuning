"""Importing documents that arrive already prepared (SPEC_03, SPEC_04).

Page images, per-page MinerU markdown and a golden JSON written against the
canonical schema — placed where the pipeline already reads them, so the dataset
build picks them up with no other change.

The two guards are what these cover: an import with no MinerU version makes the
serving OCR-pin check unverifiable, and a golden label that reaches
``golden-labels/`` unvalidated is a training target nobody checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.ingestion.import_prepared import (
    ImportError_,
    import_batch,
    import_document,
    read_prepared,
)

#: A canonical policy label — every policy is written in the client's canonical
#: schema. Its line travels in metadata (`lob`), since that schema has no
#: top-level home for one.
GOLDEN = json.loads(
    (Path(__file__).resolve().parent / "fixtures/golden/policy_0001.golden.json")
    .read_text(encoding="utf-8")
)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def prepared(tmp_path, name="policy_batch_001", *, pages=2, golden=None, metadata=None,
             markdown=True):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    for page in range(1, pages + 1):
        (directory / f"page_{page}.png").write_bytes(b"\x89PNG" + str(page).encode())
        if markdown:
            (directory / f"page_{page}.md").write_text(f"# page {page}", encoding="utf-8")
    (directory / "golden.json").write_text(
        json.dumps(golden if golden is not None else GOLDEN), encoding="utf-8"
    )
    if metadata is not None:
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


# --------------------------------------------------------------------------
# Reading a bundle
# --------------------------------------------------------------------------


def test_pages_are_read_in_page_order_not_lexical_order(tmp_path):
    """page_10 sorts before page_2 lexically. Pairing an image with another
    page's text teaches the model to read one page while looking at another."""
    directory = prepared(tmp_path, pages=11)
    doc = read_prepared(directory)

    assert [p.stem for p in doc.images[:3]] == ["page_1", "page_2", "page_3"]
    assert doc.images[-1].stem == "page_11"
    assert [p.stem for p in doc.markdown] == [f"page_{i}" for i in range(1, 12)]


def test_a_page_count_mismatch_is_refused(tmp_path):
    directory = prepared(tmp_path, pages=3)
    (directory / "page_3.md").unlink()

    with pytest.raises(ImportError_, match="paired per page"):
        read_prepared(directory)


def test_an_unnumbered_page_file_is_refused(tmp_path):
    directory = prepared(tmp_path, pages=1)
    (directory / "cover.png").write_bytes(b"\x89PNG")

    with pytest.raises(ImportError_, match="does not name its page"):
        read_prepared(directory)


def test_a_bundle_with_no_golden_label_is_refused(tmp_path):
    directory = prepared(tmp_path)
    (directory / "golden.json").unlink()

    with pytest.raises(ImportError_, match="no golden.json"):
        read_prepared(directory)


# --------------------------------------------------------------------------
# The two guards
# --------------------------------------------------------------------------


def test_an_import_without_the_mineru_version_is_refused(tmp_path, client):
    """The model learns how MinerU formats its output, and serving refuses to
    start when its OCR version does not match the corpus pin. A blank pin makes
    that check unverifiable rather than satisfied."""
    doc = read_prepared(prepared(tmp_path))

    with pytest.raises(ImportError_, match="MinerU version"):
        import_document(doc, "policy", client, mineru_version="")


def test_a_schema_invalid_golden_never_reaches_the_label_store(tmp_path, client):
    """The first symptom of an unchecked target is a model that learned the
    wrong shape, by which point it is in the weights."""
    doc = read_prepared(prepared(tmp_path, golden={"insured_name": 42}))

    with pytest.raises(ImportError_, match="does not satisfy the policy schema"):
        import_document(doc, "policy", client, mineru_version="2.0.0")

    assert not client.list("golden-labels/")


# --------------------------------------------------------------------------
# What an import leaves behind
# --------------------------------------------------------------------------


def test_an_imported_document_lands_where_the_pipeline_already_reads(tmp_path, client):
    """A second layout would mean a second set of readers, and one of them would
    eventually be missed."""
    doc = read_prepared(prepared(tmp_path, metadata={"field_provenance": {"named_insured.primary_name": 1}, "lob": ["workers_comp"]}))
    source_id = import_document(doc, "policy", client, mineru_version="2.0.0")

    assert source_id == "policy_0001"
    assert client.exists(paths.processed_page("policy", source_id, 1, "png"))
    assert client.exists(paths.processed_page("policy", source_id, 2, "md"))
    assert client.read_json(paths.golden_label("policy", source_id)) == GOLDEN

    meta = client.read_json(paths.ocr_meta("policy", source_id))
    assert meta["page_count"] == 2 and meta["mineru_version"] == "2.0.0"
    assert meta["source_checksum"] and meta["imported"] is True
    assert meta["render_only"] is False

    label_meta = client.read_json(paths.label_metadata("policy", source_id))
    assert label_meta["field_provenance"] == {"named_insured.primary_name": 1}
    assert label_meta["lob"] == ["workers_comp"]


def test_nothing_is_written_to_the_raw_document_layer(tmp_path, client):
    """There is no PDF, so the importer never touches the unredacted-PII layer.

    Asserted at the backend, because a general-context client is not even allowed
    to look: the importer runs without the ingestion context precisely because it
    has no business there."""
    import_document(read_prepared(prepared(tmp_path)), "policy", client, mineru_version="2.0.0")

    assert not list(client._backend.list("raw", "raw-documents/"))


def test_a_bundle_with_no_markdown_is_recorded_as_image_only(tmp_path, client):
    doc = read_prepared(prepared(tmp_path, markdown=False))
    source_id = import_document(doc, "policy", client, mineru_version="2.0.0")

    assert client.read_json(paths.ocr_meta("policy", source_id))["render_only"] is True


def test_the_checksum_is_stable_across_imports_of_the_same_pages(tmp_path, client):
    """Deduplication and the OCR-skip check both key on it."""
    first = import_document(read_prepared(prepared(tmp_path, "a")), "policy", client,
                            mineru_version="2.0.0")
    second = import_document(read_prepared(prepared(tmp_path, "b")), "policy", client,
                             mineru_version="2.0.0")

    assert first != second
    assert (client.read_json(paths.ocr_meta("policy", first))["source_checksum"]
            == client.read_json(paths.ocr_meta("policy", second))["source_checksum"])


def test_ids_continue_the_counter_labeling_uses(tmp_path, client):
    """An imported document and a labeled one must never collide on a source_id —
    it is the key every later stage joins on."""
    ids = [
        import_document(read_prepared(prepared(tmp_path, name)), "policy", client,
                        mineru_version="2.0.0")
        for name in ("first", "second", "third")
    ]
    assert ids == ["policy_0001", "policy_0002", "policy_0003"]


def test_one_bad_bundle_does_not_lose_the_batch(tmp_path, client):
    """A hundred-document import should not be lost to one malformed label."""
    root = tmp_path / "batch"
    root.mkdir()
    prepared(root, "good_a")
    prepared(root, "bad", golden={"insured_name": 42})
    prepared(root, "good_b")

    report = import_batch(root, "policy", client, mineru_version="2.0.0")

    assert len(report.imported) == 2
    assert [name for name, _ in report.skipped] == ["bad"]
    assert "bad" in report.describe()


def test_pages_are_stored_as_png_whatever_arrives(tmp_path, client):
    """`processed/.../page_N.png` is what every reader asks for by name, so a
    stored .jpg leaves a document that looks complete and whose pages nothing can
    find."""
    from PIL import Image

    directory = tmp_path / "jpg_bundle"
    directory.mkdir()
    for page in (1, 2):
        Image.new("RGB", (8, 8), (page * 10, 0, 0)).save(directory / f"page_{page}.jpg")
        (directory / f"page_{page}.md").write_text(f"# page {page}", encoding="utf-8")
    (directory / "golden.json").write_text(json.dumps(GOLDEN), encoding="utf-8")

    source_id = import_document(
        read_prepared(directory), "policy", client, mineru_version="2.0.0"
    )

    for page in (1, 2):
        stored = paths.processed_page("policy", source_id, page, "png")
        assert client.exists(stored)
        assert client.read_bytes(stored).startswith(b"\x89PNG")
    assert not client.exists(paths.processed_page("policy", source_id, 1, "jpg"))


def test_a_mixed_extension_bundle_keeps_its_page_order(tmp_path, client):
    """Sorting each extension separately and concatenating put page_1.jpg after
    page_9.png, so images were paired with another page's markdown."""
    from PIL import Image

    directory = tmp_path / "mixed"
    directory.mkdir()
    for page in range(1, 11):
        if page == 1:
            Image.new("RGB", (8, 8)).save(directory / "page_1.jpg")
        else:
            (directory / f"page_{page}.png").write_bytes(b"\x89PNG" + str(page).encode())
        (directory / f"page_{page}.md").write_text(f"# page {page}", encoding="utf-8")
    (directory / "golden.json").write_text(json.dumps(GOLDEN), encoding="utf-8")

    doc = read_prepared(directory)

    assert [p.stem for p in doc.images] == [f"page_{i}" for i in range(1, 11)]
    assert [p.stem for p in doc.markdown] == [f"page_{i}" for i in range(1, 11)]
