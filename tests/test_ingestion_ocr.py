"""IMPL-03 — ingestion, MinerU version/device pinning, and OCR output.

Runs against :class:`InMemoryBackend` and a stub OCR engine, so no Azure, no
MinerU install, and no CUDA are needed. The stub implements the same
:class:`~data_pipeline.ocr.run_mineru.OcrEngine` protocol the real engine does,
which is the point of that interface existing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.ingestion import pull_raw_pdfs as ingest
from data_pipeline.ocr import mineru_version as mv
from data_pipeline.ocr import run_mineru
from data_pipeline.ocr.run_mineru import PageOutput

FIXTURE_OCR = Path(__file__).resolve().parent / "fixtures" / "ocr"


class StubEngine:
    """Deterministic stand-in for MinerU, driven by the OCR fixtures."""

    def __init__(self, markdown: str = "# STUB\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n"):
        self.markdown = markdown
        self.calls: list[dict] = []

    def process(self, pdf_bytes: bytes, *, device: str, max_long_side_px: int) -> list[PageOutput]:
        self.calls.append({"device": device, "cap": max_long_side_px})
        return [
            PageOutput(
                page_number=1,
                markdown=self.markdown,
                image_bytes=b"\x89PNG stub",
                table_row_count=run_mineru.count_table_rows(self.markdown),
            )
        ]


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def ocr_client(backend) -> BlobClient:
    return BlobClient(backend=backend, container="main", raw_container="raw", context="ocr")


@pytest.fixture
def ingestion_client(backend) -> BlobClient:
    return BlobClient(backend=backend, container="main", raw_container="raw", context="ingestion")


def _seed_raw_document(client: BlobClient, doc_type="policy", source_id="policy_0001", checksum="abc123"):
    client.write_bytes(paths.raw_pdf(doc_type, source_id), b"%PDF-1.7 stub")
    client.write_json(paths.raw_metadata(doc_type, source_id), {
        "source_id": source_id, "doc_type": doc_type,
        "checksum_sha256": checksum, "page_count": 1, "is_scanned": False,
    })


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
# Version and device pinning (arch §8a, §14)
# --------------------------------------------------------------------------

def test_version_mismatch_is_a_regression_trigger():
    """The model learns HOW MinerU formats output, so this is distribution shift."""
    env = mv.OcrEnvironment(mineru_version="1.4.2", device="cuda")
    with pytest.raises(mv.MinerUVersionError, match="distribution shift"):
        mv.assert_version_matches({"mineru_version": "1.3.0", "ocr_device": "cuda"}, env)


def test_device_mismatch_is_caught_too():
    env = mv.OcrEnvironment(mineru_version="1.4.2", device="cpu")
    with pytest.raises(mv.MinerUVersionError, match="device mismatch"):
        mv.assert_version_matches({"mineru_version": "1.4.2", "ocr_device": "cuda"}, env)


def test_device_check_relaxes_when_output_proves_identical():
    """If the Phase 0 spike shows GPU and CPU output match, the check can relax."""
    env = mv.OcrEnvironment(mineru_version="1.4.2", device="cpu")
    mv.assert_version_matches(
        {"mineru_version": "1.4.2", "ocr_device": "cuda"}, env, check_device=False
    )


def test_matching_environment_passes():
    env = mv.OcrEnvironment(mineru_version="1.4.2", device="cuda")
    mv.assert_version_matches({"mineru_version": "1.4.2", "ocr_device": "cuda"}, env)


def test_no_gpu_fails_rather_than_falling_back(monkeypatch):
    """Silent fallback is the failure worth preventing: the job would finish,
    write markdown from a different distribution than the corpus was built on,
    and report success. Failing here is the only way that becomes visible."""
    monkeypatch.setattr(mv, "cuda_available", lambda: (False, None))
    with pytest.raises(mv.MinerUVersionError, match="no CUDA device is visible"):
        mv.resolve_device("cuda", strict=True)


def test_gpu_is_the_only_device(monkeypatch):
    """Not a default with an opt-out — the only device there is.

    MinerU's CPU path uses lighter model variants and produces *different
    markdown* from the same PDF: different table splits, different cell
    boundaries. The model learns how MinerU formats its output (arch §8a), so a
    corpus built across both devices is built from two distributions. That is a
    correctness question, not a speed one, which is why there is no opt-out.
    """
    monkeypatch.setattr(mv, "cuda_available", lambda: (True, "NVIDIA L4"))
    assert mv.resolve_device(None) == "cuda"
    assert mv.resolve_device("cuda") == "cuda"

    with pytest.raises(mv.MinerUVersionError, match="GPU only"):
        mv.resolve_device("cpu")


def test_a_cpu_corpus_cannot_be_extended(monkeypatch):
    """Readable, not extendable: adding GPU documents to a CPU-built corpus puts
    two markdown formats in one training set."""
    with pytest.raises(mv.MinerUVersionError, match="GPU-only"):
        mv.assert_gpu_only("cpu")
    mv.assert_gpu_only("cuda")      # does not raise


def test_there_is_no_cpu_option_on_the_ocr_cli():
    """An option that accepts one value is a way to be surprised later.

    Asserted against argparse's real option list rather than the source text,
    which would also match the comment explaining why the flag is gone.
    """
    import pytest as _pytest

    from data_pipeline.ocr import run_mineru

    with _pytest.raises(SystemExit):        # argparse rejects an unknown option
        run_mineru.main(["--doc-type", "policy", "--source-ids", "x", "--device", "cpu"])


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------

def test_ingest_writes_pdf_and_metadata(tmp_path, ingestion_client, monkeypatch):
    """pypdf is an optional extra, so page inspection is stubbed — the ingestion
    contract being tested is the checksum, paths, and metadata, not PDF parsing."""
    monkeypatch.setattr(
        ingest, "inspect_pdf",
        lambda path: {"page_count": 2, "is_scanned": False, "text_chars_per_sampled_page": 900.0},
    )
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7 sample")

    source_id, status = ingest.ingest_pdf(pdf, "policy", ingestion_client, known_checksums={})
    assert status == "ingested"
    assert source_id == "policy_0001"

    assert ingestion_client.exists(paths.raw_pdf("policy", source_id))
    meta = ingestion_client.read_json(paths.raw_metadata("policy", source_id))
    assert meta["checksum_sha256"] == ingest.sha256_file(pdf)
    assert meta["original_filename"] == "sample.pdf"
    assert meta["page_count"] == 2
    assert meta["is_scanned"] is False
    # Never omitted: absence must not be mistakable for "no PII".
    assert meta["pii_flags"]["status"] == "not_assessed"


def test_identical_content_is_deduplicated(tmp_path, ingestion_client, monkeypatch):
    """Loss Runs and renewal policies are re-submitted constantly. De-duping
    prevents paying to label the same document twice."""
    monkeypatch.setattr(
        ingest, "inspect_pdf",
        lambda path: {"page_count": 1, "is_scanned": False, "text_chars_per_sampled_page": 900.0},
    )
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    first.write_bytes(b"%PDF identical bytes")
    second.write_bytes(b"%PDF identical bytes")

    known: dict[str, str] = {}
    sid_a, status_a = ingest.ingest_pdf(first, "policy", ingestion_client, known_checksums=known)
    sid_b, status_b = ingest.ingest_pdf(second, "policy", ingestion_client, known_checksums=known)

    assert status_a == "ingested"
    assert status_b == "duplicate"
    assert sid_b == sid_a, "the duplicate must resolve to the already-ingested source_id"


def test_a_corrected_document_becomes_a_new_source_id(tmp_path, ingestion_client, monkeypatch):
    """Raw documents are write-once, so a correction is a new source_id — that is
    what keeps historical runs reproducible against the bytes they trained on."""
    monkeypatch.setattr(
        ingest, "inspect_pdf",
        lambda path: {"page_count": 1, "is_scanned": False, "text_chars_per_sampled_page": 900.0},
    )
    original, corrected = tmp_path / "v1.pdf", tmp_path / "v2.pdf"
    original.write_bytes(b"%PDF original")
    corrected.write_bytes(b"%PDF corrected")

    known: dict[str, str] = {}
    sid_1, _ = ingest.ingest_pdf(original, "policy", ingestion_client, known_checksums=known)
    sid_2, status = ingest.ingest_pdf(corrected, "policy", ingestion_client, known_checksums=known)

    assert status == "ingested"
    assert sid_2 != sid_1
    assert ingestion_client.exists(paths.raw_pdf("policy", sid_1))
    assert ingestion_client.exists(paths.raw_pdf("policy", sid_2))


def test_scanned_flag_drives_the_eval_subset(tmp_path, ingestion_client, monkeypatch):
    """Detected at ingest because it is cheap here and expensive later, and it
    defines the scanned eval subset feeding the ViT gate (arch §3, §15)."""
    monkeypatch.setattr(
        ingest, "inspect_pdf",
        lambda path: {"page_count": 1, "is_scanned": True, "text_chars_per_sampled_page": 3.0},
    )
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF scanned")
    source_id, _ = ingest.ingest_pdf(pdf, "policy", ingestion_client, known_checksums={})
    assert ingestion_client.read_json(paths.raw_metadata("policy", source_id))["is_scanned"] is True


def test_checksum_is_stable_and_content_addressed(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"%PDF identical")
    b.write_bytes(b"%PDF identical")
    assert ingest.sha256_file(a) == ingest.sha256_file(b)

    b.write_bytes(b"%PDF different")
    assert ingest.sha256_file(a) != ingest.sha256_file(b)


def test_purge_plan_reports_the_full_cascade_without_deleting():
    """Retaining derived data from a purged document is a real audit gap, but
    corpus files are shared, so a purge is a rebuild rather than a delete."""
    plan = ingest.purge_plan("policy_0001", "policy")
    assert plan["raw"] and plan["processed"] and plan["golden_labels"]
    assert "rebuild" in plan["corpus_action"][0]
    assert "policy_0001" in plan["corpus_action"][0]


def test_unknown_doc_type_is_refused(tmp_path, ingestion_client):
    with pytest.raises(ingest.IngestionError, match="unknown doc_type"):
        ingest.ingest_directory(tmp_path, "invoice", ingestion_client)


def test_unclassified_bucket_is_allowed(tmp_path, ingestion_client):
    """Forcing a type guess at ingest would bake a mistake into the source_id,
    which is the identifier everything downstream joins on."""
    result = ingest.ingest_directory(tmp_path, ingest.UNCLASSIFIED, ingestion_client)
    assert result.ingested == []


# --------------------------------------------------------------------------
# OCR output
# --------------------------------------------------------------------------

def test_table_row_count_discounts_header_and_separator():
    """Feeds the row-completeness cross-check: the model extracting 6 rows from a
    page MinerU saw 8 on is a recall failure per-field confidence cannot see."""
    markdown = (
        "| Claim | Paid |\n| --- | --- |\n"
        "| A | 1 |\n| B | 2 |\n| C | 3 |\n"
    )
    assert run_mineru.count_table_rows(markdown) == 3


def test_table_row_count_on_a_real_fixture():
    markdown = (FIXTURE_OCR / "lossrun_0001_page_1.md").read_text(encoding="utf-8")
    # The fixture Loss Run has exactly three claim rows.
    assert run_mineru.count_table_rows(markdown) == 3


def test_table_row_count_is_zero_without_a_table():
    assert run_mineru.count_table_rows("# Heading\n\nSome prose.\n") == 0


def test_process_document_records_version_device_and_cap(ocr_client, monkeypatch):
    monkeypatch.setattr(
        run_mineru, "current_environment",
        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda", "NVIDIA L4"),
    )
    _seed_raw_document(ocr_client)
    meta = run_mineru.process_document("policy", "policy_0001", ocr_client, StubEngine())

    assert meta["mineru_version"] == "1.4.2"
    assert meta["ocr_device"] == "cuda"
    assert meta["resolution_cap_px"] == 1792
    assert meta["page_count"] == 1
    assert meta["table_row_counts"] == {"1": 2}
    assert meta["source_checksum"] == "abc123"


def test_process_document_writes_page_image_and_markdown(ocr_client, monkeypatch):
    monkeypatch.setattr(
        run_mineru, "current_environment",
        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"),
    )
    _seed_raw_document(ocr_client)
    run_mineru.process_document("policy", "policy_0001", ocr_client, StubEngine())

    assert ocr_client.exists(paths.processed_page("policy", "policy_0001", 1, "png"))
    assert ocr_client.exists(paths.processed_page("policy", "policy_0001", 1, "md"))


def test_reprocessing_is_skipped_when_nothing_changed(ocr_client, monkeypatch):
    monkeypatch.setattr(
        run_mineru, "current_environment",
        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"),
    )
    _seed_raw_document(ocr_client)
    engine = StubEngine()
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 1, "second run should have been skipped"


def test_a_mineru_version_change_forces_reprocessing(ocr_client, monkeypatch):
    """Each of version, device and checksum changes the input distribution the
    model was trained to arbitrate against."""
    _seed_raw_document(ocr_client)
    engine = StubEngine()

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.5.0", "cuda"))
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 2


def test_a_device_change_forces_reprocessing(ocr_client, monkeypatch):
    _seed_raw_document(ocr_client)
    engine = StubEngine()

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cpu"))
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 2


def test_engine_receives_the_configured_resolution_cap(ocr_client, monkeypatch):
    """The cap must be identical here and at inference, or the model is served a
    distribution it never trained on (arch §11)."""
    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    _seed_raw_document(ocr_client)
    engine = StubEngine()
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert engine.calls[0]["cap"] == 1792


def test_find_unprocessed_lists_only_documents_without_ocr_meta(ocr_client, monkeypatch):
    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    _seed_raw_document(ocr_client, source_id="policy_0001")
    _seed_raw_document(ocr_client, source_id="policy_0002")
    assert run_mineru.find_unprocessed(ocr_client, "policy") == ["policy_0001", "policy_0002"]

    run_mineru.process_document("policy", "policy_0001", ocr_client, StubEngine())
    assert run_mineru.find_unprocessed(ocr_client, "policy") == ["policy_0002"]


def test_ocr_engine_protocol_is_satisfied_by_the_stub():
    """The swappable interface is what lets the pipeline be verified without
    MinerU, and what lets the Phase 0 spike replace the engine outright."""
    engine: run_mineru.OcrEngine = StubEngine()
    pages = engine.process(b"%PDF", device="cpu", max_long_side_px=1792)
    assert pages and pages[0].page_number == 1


# --------------------------------------------------------------------------
# Near-duplicate detection and grouping (arch v2.1 §8.2)
# --------------------------------------------------------------------------

def _fp(source_id, **over):
    from data_pipeline.ingestion.dedup_and_group import DocumentFingerprint

    base = dict(
        source_id=source_id,
        doc_type="policy",
        content_sha256=f"sha-{source_id}",
    )
    base.update(over)
    return DocumentFingerprint(**base)


def test_the_same_file_twice_is_grouped_not_dropped():
    """A carrier that issues four hundred near-identical certificates is a real
    part of the distribution. Removing them trains on a corpus that does not look
    like production; what must not happen is those four hundred spanning a split."""
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([_fp("a", content_sha256="same"), _fp("b", content_sha256="same")])

    assert report.group_of["a"] == report.group_of["b"]
    assert report.exact_duplicates == {"b": "a"}
    assert set(report.group_of) == {"a", "b"}, "neither document was dropped"


def test_a_renewal_groups_with_its_prior_year():
    """Same carrier, same template, same account — every value differs and the
    family does not. This is the leak the source-document split could not see."""
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([
        _fp("2025", carrier="Acme Ins. Co.", template_id="ACORD25", account="Rivera Fabrication"),
        _fp("2026", carrier="ACME INS CO", template_id="acord25", account="rivera fabrication"),
    ])
    assert report.group_of["2025"] == report.group_of["2026"], (
        "carrier and account names differing only in case and punctuation are one family"
    )


def test_an_identical_page_one_layout_is_one_template():
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([
        _fp("a", layout_phash="ffff0000"),
        _fp("b", layout_phash="ffff0000"),
        _fp("c", layout_phash="0000ffff"),
    ])
    assert report.group_of["a"] == report.group_of["b"]
    assert report.group_of["c"] != report.group_of["a"]


def test_near_duplicate_text_merges_groups():
    from data_pipeline.ingestion.dedup_and_group import assign_groups, minhash

    shared = " ".join(f"policy clause number {i} applies to the named insured" for i in range(40))
    report = assign_groups([
        _fp("a", minhash=minhash(shared)),
        _fp("b", minhash=minhash(shared + " one extra trailing clause")),
        _fp("c", minhash=minhash("completely unrelated loss run claim table headings only")),
    ])
    assert report.group_of["a"] == report.group_of["b"]
    assert report.group_of["c"] != report.group_of["a"]
    assert report.near_duplicate_pairs


def test_grouping_is_transitive():
    """A chain of renewals is one account. Letting the ends of the chain split
    leaks exactly what grouping prevents, even where the ends are not themselves
    similar enough to merge directly."""
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([
        _fp("a", layout_phash="1111"),
        _fp("b", layout_phash="1111", account="shared"),
        _fp("c", account="shared"),
    ])
    assert report.group_of["a"] == report.group_of["c"]
    assert report.group_count == 1


def test_documents_naming_no_carrier_are_reported():
    """They can still be grouped by layout and text, but the held-out-carrier
    slice cannot use them — so the count is worth surfacing."""
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([_fp("a"), _fp("b", carrier="Acme")])
    assert report.unattributed == ["a"]


def test_an_unrelated_document_is_its_own_family():
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([_fp("a"), _fp("b"), _fp("c")])
    assert report.group_count == 3


def test_word_shingles_survive_a_single_ocr_error():
    """Character shingles break on every shingle overlapping a corrupted
    character; a word shingle loses only those containing that word."""
    from data_pipeline.ingestion.dedup_and_group import estimated_jaccard, minhash

    clean = " ".join(f"the named insured shown in declarations item {i}" for i in range(30))
    ocr_error = clean.replace("insured", "1nsured", 1)
    assert estimated_jaccard(minhash(clean), minhash(ocr_error)) > 0.85


def test_grouping_is_deterministic():
    """Corpus composition must be reproducible from the manifest alone."""
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    docs = [_fp(f"d{i}", layout_phash=f"h{i % 3}") for i in range(9)]
    assert assign_groups(docs).group_of == assign_groups(list(reversed(docs))).group_of
