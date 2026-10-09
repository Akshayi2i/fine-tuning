"""Real documents of one line with no twins and no split - the CGL source data -
turned into bundles, and the casualty_fleet scope they train
(data_pipeline.ingestion.prepare_bundles.prepare_original_bundles)."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from data_pipeline.ingestion.prepare_bundles import (
    BundleError,
    corrected_gold,
    prepare_original_bundles,
    read_recodes,
)

pymupdf = pytest.importorskip("pymupdf")

EXAMPLE = Path("configs/canonical schema/common schema/examples/homeowners_minimal.json")


def _gl_gold(insured: str = "TAYLOR SAMPLE", parts: tuple[str, ...] = ("gl",), code: str = "GL_BI_PD") -> dict:
    """The homeowners example's shared blocks, as a one-coverage general liability gold."""
    example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    gold = {key: copy.deepcopy(example[key]) for key in ("document", "carrier", "named_insured", "policy")}
    gold["named_insured"]["primary_name"].update(raw=insured, parsed=insured)
    gold["lob_parts"] = [{**copy.deepcopy(example["lob_parts"][0]), "part_id": f"part_{n}", "lob": lob}
                         for n, lob in enumerate(parts, start=1)]
    coverage = copy.deepcopy(example["coverages"][0])
    for key in ("applies_to", "included"):
        coverage.pop(key, None)
    gold["coverages"] = [{**coverage, "coverage_code": code}]
    return gold


def _source(tmp_path: Path, docs: dict[tuple[str, str], dict | None]) -> Path:
    """``docs``: (carrier folder, name) -> gold, or None for a PDF with no gold."""
    root = tmp_path / "CGL source data"
    for (carrier, name), gold in docs.items():
        pdf = root / "PDFs" / carrier / f"{name}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open()
        doc.new_page(width=612, height=792).insert_text((72, 72), "Commercial General Liability Declarations")
        doc.save(pdf)
        if gold is not None:
            path = root / "Gold JSON" / carrier / f"{name}.gold.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(gold), encoding="utf-8")
    return root


def test_a_renamed_coverage_code_is_recoded_as_the_line_s_code_file_says():
    """gl 3.0.0 moved three codes to the shared X_ codes and says so in its code
    file (replaces_codes): a gold drafted before the move is re-coded, not
    refused at the corpus build."""
    recodes = read_recodes()
    assert recodes["gl"]["GL_ADDITIONAL_INSURED"] == "X_ADDITIONAL_INSURED"
    assert recodes["gl"]["GL_WAIVER_OF_SUBROGATION"] == "X_WAIVER_OF_SUBROGATION"

    fixed, notes, problem = corrected_gold(_gl_gold(code="GL_ADDITIONAL_INSURED"), "gl", carrier="",
                                           text_pdf=Path("none.pdf"), recodes=recodes)
    assert problem is None and fixed["coverages"][0]["coverage_code"] == "X_ADDITIONAL_INSURED"
    assert ("coverage code", "coverages[0] GL_ADDITIONAL_INSURED -> X_ADDITIONAL_INSURED") in notes


def test_the_repository_s_recode_table_wins_over_the_code_file(tmp_path):
    table = tmp_path / "recodes.yaml"
    table.write_text("gl:\n  GL_ADDITIONAL_INSURED: X_PRIMARY_NONCONTRIBUTORY\n", encoding="utf-8")
    assert read_recodes(table)["gl"]["GL_ADDITIONAL_INSURED"] == "X_PRIMARY_NONCONTRIBUTORY"


def test_a_package_policy_filed_under_one_common_model_line_is_left_out():
    """A general liability part and an inland marine part: the corpus build refuses
    such a target and stops on it, so it never reaches a bundle."""
    _, _, problem = corrected_gold(_gl_gold(parts=("gl", "inland_marine")), "gl", carrier="",
                                   text_pdf=Path("none.pdf"), recodes={})
    assert problem is not None and problem.startswith("package policy: lob_parts lists 2 part(s)")
    _, _, problem = corrected_gold(_gl_gold(parts=("gl", "gl")), "gl", carrier="",
                                   text_pdf=Path("none.pdf"), recodes={})
    assert problem is not None and problem.startswith("package policy")


def test_each_pair_becomes_a_real_document_with_its_split_delivered(tmp_path):
    root = _source(tmp_path, {
        ("Johnson & Johnson", "dec_01"): _gl_gold("RIVERA FABRICATION LLC"),
        ("Johnson & Johnson", "Renewal_01"): _gl_gold("Rivera Fabrication, LLC"),
        ("Utica First", "dec_01"): _gl_gold("ANOTHER INSURED"),
    })
    out = tmp_path / "bundles"

    report = prepare_original_bundles(root, out, lob="gl")

    assert report.documents == 3
    folder = out / "gl__johnson_johnson__renewal_01"
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert (folder / "document.pdf").is_file() and (folder / "golden.json").is_file()
    assert meta["lob"] == "gl" and meta["synthetic"] is False and meta["source_system"] == "original"
    assert meta["split"] in ("train", "val", "test")
    # A folder is not always one carrier (a broker's holds several): the gold's own.
    assert meta["carrier"] is None
    # One insured, however it is spelt: one family, so one side of the split.
    twin = json.loads((out / "gl__johnson_johnson__dec_01" / "metadata.json").read_text(encoding="utf-8"))
    assert twin["template_id"] == meta["template_id"] and twin["split"] == meta["split"]
    other = json.loads((out / "gl__utica_first__dec_01" / "metadata.json").read_text(encoding="utf-8"))
    assert other["template_id"] != meta["template_id"]
    # The family names no one.
    assert "rivera" not in meta["template_id"].lower()

    with (out / "split.csv").open(encoding="utf-8", newline="") as fh:
        listed = {row["document"]: row["split"] for row in csv.DictReader(fh)}
    assert listed["gl__johnson_johnson__renewal_01"] == meta["split"] and len(listed) == 3


def test_the_same_source_splits_the_same_way_every_time(tmp_path):
    docs = {(f"Carrier {n % 7}", f"policy_{n:02d}"): _gl_gold(f"INSURED {n}") for n in range(30)}
    root = _source(tmp_path, docs)
    first = prepare_original_bundles(root, tmp_path / "a", lob="gl")
    second = prepare_original_bundles(root, tmp_path / "b", lob="gl")

    assert first.written == second.written
    assert {s for s, _ in first.written} == {"train", "val", "test"}
    assert (tmp_path / "a" / "split.csv").read_text() == (tmp_path / "b" / "split.csv").read_text()


def test_what_cannot_be_trained_is_left_out_and_said_so(tmp_path):
    root = _source(tmp_path, {
        ("Philadelphia", "renewal_01"): _gl_gold(parts=("gl", "inland_marine")),
        ("Philadelphia", "renewal_02"): _gl_gold(),
        ("Philadelphia", "orphan"): None,
    })
    out = tmp_path / "bundles"
    (out / "gl__philadelphia__renewal_01").mkdir(parents=True)           # an earlier run's folder

    report = prepare_original_bundles(root, out, lob="gl")

    assert report.documents == 1
    assert report.skipped["gold left out: package policy"] == 1
    assert report.skipped["a PDF without its gold"] == 1
    assert not (out / "gl__philadelphia__renewal_01").exists() and report.removed == 1
    corrections = (out / "corrections.csv").read_text(encoding="utf-8")
    assert 'gl__philadelphia__renewal_01,left out,"package policy' in corrections


def test_two_documents_that_would_share_a_folder_are_refused(tmp_path):
    root = _source(tmp_path, {("Great American", "dec_01"): _gl_gold(),
                              ("Great-American", "dec_01"): _gl_gold()})
    with pytest.raises(BundleError, match="share the folder name"):
        prepare_original_bundles(root, tmp_path / "bundles", lob="gl")


def test_a_line_outside_the_scope_is_refused(tmp_path):
    root = _source(tmp_path, {("Utica First", "dec_01"): _gl_gold()})
    from common.scopes import get_scope

    with pytest.raises(BundleError, match="outside the scope"):
        prepare_original_bundles(root, tmp_path / "bundles", lob="gl", lines=get_scope("personal_lines").lines)


def test_the_command_line_needs_the_line_of_real_documents_alone(tmp_path, capsys):
    from data_pipeline.ingestion.prepare_bundles import main

    root = _source(tmp_path, {("Utica First", "dec_01"): _gl_gold()})
    assert main(["--input", str(root), "--out", str(tmp_path / "bundles"), "--scope", "casualty_fleet"]) == 1
    assert "pass --lob" in capsys.readouterr().err
    assert main(["--input", str(root), "--out", str(tmp_path / "bundles"), "--scope", "casualty_fleet",
                 "--lob", "gl"]) == 0
    assert (tmp_path / "bundles" / "gl__utica_first__dec_01" / "metadata.json").is_file()


# --------------------------------------------------------------------------
# The casualty_fleet scope
# --------------------------------------------------------------------------


def test_the_casualty_fleet_adapter_covers_only_the_line_it_trains_on():
    """Serving routes by a release's lines: listed before they are trained,
    auto, wc and umbrella policies would reach an adapter that never saw them."""
    from common.scopes import assert_one_output_shape, get_scope

    scope = get_scope("casualty_fleet")
    assert scope.lines == frozenset({"gl"})
    assert scope.covers_lob("gl") and scope.covers_lob("general_liability") and scope.covers_lob("cgl")
    assert not scope.covers_lob("auto") and not scope.covers_lob("wc") and not scope.covers_lob("umbrella")
    # Its corpus holds general liability alone, whatever else the tenant holds.
    assert_one_output_shape(scope, ["gl", "auto"])
    assert scope.run_id("v1") == "casualty_fleet-v1"


def test_the_cgl_code_reads_the_general_liability_schema():
    from common.schemas import schema_key

    assert schema_key("policy", None, "cgl") == schema_key("policy", None, "gl") == "policy:gl"


def test_the_smoke_run_trains_the_scope_it_is_given(tmp_path):
    from orchestration.smoke_run import commands, parse_args

    args, _ = parse_args(["--scope", "casualty_fleet", "--batch", "cgl-v1", "--tenant", "smoke-cgl"])
    steps = commands(batch_dir=tmp_path, subset_dir=tmp_path, tenant=args.tenant, version="v0",
                     check_out=tmp_path, scope=args.scope)
    for step in ("preflight", "finetune"):
        assert steps[step][steps[step].index("--scope") + 1] == "casualty_fleet"
    default = commands(batch_dir=tmp_path, subset_dir=tmp_path, tenant="smoke", version="v0", check_out=tmp_path)
    assert default["finetune"][default["finetune"].index("--scope") + 1] == "personal_lines"
    with pytest.raises(SystemExit):
        parse_args(["--scope", "no_such_scope"])
