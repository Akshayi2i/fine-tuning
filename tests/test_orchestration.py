"""SPEC_13 + SPEC_14 — the command surface, the DAG, and the pod contract.

Four things here guard failures nothing else would notice:

* ``all`` reaching ``package`` after a blocked gate would ship a regression.
* ``all`` invoking extraction would make every build wait on a test run.
* A completed stage re-running would silently duplicate a corpus or a training run.
* A pod launched with no staging volume would write to disk that ceases to exist,
  with every write succeeding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import run as cli
from orchestration.pipeline_dag import (
    FINETUNE_STAGES,
    PACKAGE_STAGES,
    STAGES,
    GateBlocked,
    PipelineError,
    StageContext,
    run_stages,
    stages_for,
    stages_from,
)
from orchestration.runpod_controller import (
    GPU_CLASS_BY_STAGE,
    EndpointError,
    LocalBackend,
    PodLaunchError,
    PodSpec,
    RunPodController,
    scrub,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: A candidate that measures every gating metric and beats nothing — the shape a
#: real eval report has, without pretending to be a real score.
PASSING_METRICS = {
    "field_exact_match": 0.90,
    "field_normalized_match": 0.93,
    "list_field_recall": 0.88,
    "field_f1_list_fields": 0.86,
    "schema_validity_rate": 1.0,
    "ocr_arbitration_accuracy": 0.85,
    "image_only_accuracy": 0.78,
    "scanned_accuracy": 0.80,
    "doc_type_classifier_accuracy": 0.97,
    "lob_detection_accuracy": 0.92,
    "ece_confidence": 0.04,
    "confusable_misattribution_rate": 0.02,
}


# --------------------------------------------------------------------------
# Fixtures — a corpus seeded straight into the mock Blob backend
# --------------------------------------------------------------------------


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def client(backend) -> BlobClient:
    return BlobClient(backend=backend, container="main", raw_container="raw")


@pytest.fixture
def raw_client(backend) -> BlobClient:
    """Ingestion-context client — the only one permitted to touch raw-documents/."""
    return BlobClient(backend=backend, container="main", raw_container="raw", context="ingestion")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


def ingestion_client(client: BlobClient) -> BlobClient:
    """The same store, seen through the one context allowed near the originals."""
    return BlobClient(
        backend=client._backend, container=client.container,
        raw_container=client.raw_container, context="ingestion",
    )


def seed_corpus(client: BlobClient) -> list[str]:
    """Ingest + OCR + label the fixture set, as stages 1-3 would leave it."""
    raw = ingestion_client(client)
    seeded = []
    for golden_path in sorted(FIXTURES.glob("golden/*.golden.json")):
        source_id = golden_path.name.replace(".golden.json", "")
        doc_type = source_id.rsplit("_", 1)[0]
        metadata = json.loads(
            (FIXTURES / "golden" / f"{source_id}.label_metadata.json").read_text(encoding="utf-8")
        )
        ocr_path = FIXTURES / "ocr" / f"{source_id}_page_1.md"

        raw.write_bytes(paths.raw_pdf(doc_type, source_id), b"%PDF-1.7 fixture")
        raw.write_json(paths.raw_metadata(doc_type, source_id), {
            "source_id": source_id, "doc_type": doc_type, "checksum_sha256": f"sha-{source_id}",
        })
        client.write_text(paths.processed_page(doc_type, source_id, 1, "md"),
                          ocr_path.read_text(encoding="utf-8"))
        client.write_bytes(paths.processed_page(doc_type, source_id, 1, "png"), b"\x89PNG fixture")
        client.write_json(paths.ocr_meta(doc_type, source_id), {
            "source_id": source_id, "doc_type": doc_type, "page_count": 1,
            "mineru_version": "2.0.0", "ocr_device": "cuda", "resolution_cap_px": 1792,
            "source_checksum": f"sha-{source_id}", "table_row_counts": {"1": 0}, "failed_pages": [],
        })
        client.write_json(paths.golden_label(doc_type, source_id),
                          json.loads(golden_path.read_text(encoding="utf-8")))
        client.write_json(paths.label_metadata(doc_type, source_id), metadata)
        seeded.append(source_id)
    return seeded


def make_context(client: BlobClient, controller: RunPodController, **over) -> StageContext:
    defaults = dict(
        client=client,
        raw_client=ingestion_client(client),
        controller=controller,
        out_version="v1",
        tenant_id=None,
        dry_run=True,
        skip_ingest=True,
        min_labels_per_type=1,
        git_commit="abc1234",
        metrics_provider=lambda _ctx: dict(PASSING_METRICS),
    )
    defaults.update(over)
    return StageContext(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The command surface
# --------------------------------------------------------------------------


def test_all_is_finetune_then_package():
    assert stages_for("all") == FINETUNE_STAGES + PACKAGE_STAGES


def test_all_never_includes_extraction():
    """Producing a model and using one are different concerns. Folding extraction
    into the build would make every build wait on a test run (SPEC_13 §1)."""
    names = [s.name for s in stages_for("all")]
    assert "feedback_loop" not in names
    assert not any(s.command == "extract" for s in stages_for("all"))


def test_extract_does_not_map_to_build_stages():
    with pytest.raises(PipelineError, match="extraction routine"):
        stages_for("extract")


def test_every_stage_is_owned_by_exactly_one_command():
    """A stage in two commands would run twice under `all`."""
    finetune = {s.name for s in FINETUNE_STAGES}
    package = {s.name for s in PACKAGE_STAGES}
    assert not finetune & package
    assert len(STAGES) == 11


def test_stage_numbers_match_the_architecture_order():
    assert [s.number for s in STAGES] == list(range(1, 12))


def test_from_stage_resumes_mid_pipeline():
    resumed = [s.name for s in stages_from("merge", "finetune")]
    assert resumed == ["merge"]
    assert [s.name for s in stages_from("training", "finetune")][0] == "training"


def test_from_stage_on_all_carries_through_into_package():
    """Resuming at merge must still package afterwards — stopping at the end of
    finetune would leave the artifacts staged and the operator none the wiser."""
    resumed = [s.name for s in stages_from("merge", "all")]
    assert resumed == ["merge", "quantize", "push"]


def test_unknown_stage_names_the_valid_ones():
    with pytest.raises(PipelineError, match="unknown stage"):
        stages_from("trian", "finetune")


# --------------------------------------------------------------------------
# finetune, end to end on fixtures
# --------------------------------------------------------------------------


def test_finetune_runs_ingest_through_merge(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    assert [r.name for r in report.results] == [s.name for s in FINETUNE_STAGES]
    assert ctx.volume.exists(paths.staging_adapter_dir("foundation", "v1"))
    assert ctx.volume.exists(paths.staging_merged_model_dir("v1", "policy"))


def test_finetune_writes_a_blob_manifest_while_weights_stay_staged(client, controller):
    """The volume is working storage with no durability guarantee. Without this
    rule a reclaimed volume means a training run that left no trace (SPEC_13 §3)."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    from registry_utils.query_registry import get, list_runs

    rows = list_runs(client)
    assert rows, "finetune completed without recording a single run manifest"
    foundation = get("foundation-v1", client)
    assert foundation.artifacts.status == "staged"
    assert foundation.artifacts.staging_path


def test_finetune_reports_the_unlabeled_backlog(client, controller):
    """You ingest 500, 200 are labeled, you train on 200 — and the command says
    300 are waiting for a reviewer (SPEC_13 §2)."""
    seed_corpus(client)
    client.write_json(paths.ocr_meta("policy", "policy_0099"),
                      {"source_id": "policy_0099", "page_count": 1})

    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert "policy_0099" in report.unlabeled_backlog["policy"]
    assert "awaiting review" in report.render()


def test_finetune_aborts_below_the_day_zero_floor(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller, min_labels_per_type=25)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert not report.ok
    assert report.failed_at == "labeling"
    assert report.exit_code == 1


def test_finetune_with_no_labels_at_all_stops_before_training(client, controller):
    ctx = make_context(client, controller, doc_types=["policy"])
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.failed_at == "labeling"
    assert "human work" in report.results[-1].detail


def test_foundation_trains_before_any_adapter(client, controller):
    """An adapter is trained on top of the Foundation's weights, so a parallel
    fan-out would build adapters on a model that does not exist yet (arch §12)."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    trained = ctx.results["training"].data["trained"]
    assert trained[0] == "foundation"
    assert set(trained[1:]) == set(ctx.doc_types)


def test_foundation_only_skips_the_per_type_fan_out(client, controller):
    """At pilot volume a per-type adapter trains on documents the Foundation
    already saw, so it may add nothing (arch §4)."""
    seed_corpus(client)
    ctx = make_context(client, controller, foundation_only=True)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    assert ctx.results["training"].data["trained"] == ["foundation"]
    assert ctx.volume.exists(paths.staging_merged_model_dir("v1", None))


# --------------------------------------------------------------------------
# The gate — a hard stop with no way round it
# --------------------------------------------------------------------------


def test_a_regression_stops_finetune_before_merge(client, controller):
    seed_corpus(client)
    regressed = {**PASSING_METRICS, "list_field_recall": 0.60}
    ctx = make_context(
        client, controller,
        baseline_metrics=dict(PASSING_METRICS),
        metrics_provider=lambda _ctx: regressed,
    )
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.blocked_at == "evaluation_gate"
    assert report.exit_code == 1
    assert "merge" not in ctx.results
    assert not ctx.volume.exists(paths.staging_merged_model_dir("v1", "policy"))


def test_the_block_names_the_metric_and_its_delta(client, controller):
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        baseline_metrics=dict(PASSING_METRICS),
        metrics_provider=lambda _ctx: {**PASSING_METRICS, "lob_detection_accuracy": 0.55},
    )
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    blocked = report.results[-1]

    assert "lob_detection_accuracy" in blocked.detail
    assert "0.9200 ↓ 0.5500" in blocked.detail
    assert blocked.data["failed_gates"] == ["lob_detection_accuracy"]


def test_all_never_reaches_package_on_a_failed_gate(client, controller):
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        baseline_metrics=dict(PASSING_METRICS),
        metrics_provider=lambda _ctx: {**PASSING_METRICS, "field_exact_match": 0.10},
    )
    report = run_stages(ctx, stages_for("all"), command="all")

    ran = {r.name for r in report.results}
    assert "quantize" not in ran and "push" not in ran
    assert report.blocked_at == "evaluation_gate"


def test_an_unmeasured_metric_blocks_rather_than_passing(client, controller):
    """A metric that was not measured has not passed."""
    seed_corpus(client)
    partial = {k: v for k, v in PASSING_METRICS.items() if k != "scanned_accuracy"}
    ctx = make_context(client, controller, baseline_metrics=dict(PASSING_METRICS),
                       metrics_provider=lambda _ctx: partial)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.blocked_at == "evaluation_gate"


def _all_option_strings(parser: argparse.ArgumentParser) -> set[str]:
    options = {opt for action in parser._actions for opt in action.option_strings}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                options |= _all_option_strings(sub)
    return options


def test_no_cli_flag_can_override_the_gate():
    """A flag that exists gets used on the afternoon someone is in a hurry."""
    options = _all_option_strings(cli.build_parser())
    forbidden = {"--force", "--force-deploy", "--skip-gate", "--no-gate", "--ignore-gate"}
    assert not (options & forbidden), f"override flag(s) present: {sorted(options & forbidden)}"


def test_no_stage_function_accepts_an_override_parameter():
    import inspect

    from orchestration import pipeline_dag

    for stage in STAGES:
        params = set(inspect.signature(stage.run).parameters)
        assert params == {"ctx"}, f"{stage.name} takes {params} — a stage takes only the context"

    gate_params = set(inspect.signature(pipeline_dag.stage_evaluation_gate).parameters)
    assert not any("force" in p or "skip" in p for p in gate_params)


def test_the_gate_needs_a_metrics_provider_rather_than_assuming_a_pass(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller, metrics_provider=None)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.failed_at == "evaluation_gate"
    assert "metrics nobody measured" in report.results[-1].detail


def test_the_gate_reads_an_existing_eval_report_when_no_provider_is_given(client, controller):
    """The CLI supplies no provider, so without this default every real
    `finetune` trained a Foundation and three adapters on an A100 and then
    aborted at the gate."""
    from artifact_registry import paths

    seed_corpus(client)
    client.write_json(paths.eval_report("v1"), {"gate_metrics": dict(PASSING_METRICS)})

    ctx = make_context(client, controller, metrics_provider=None)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.ok, report.render()


def test_the_baseline_comes_from_the_promoted_version_s_own_report(client, controller):
    """A regression is only detectable against a real baseline, and the CLI
    never supplied one."""
    from artifact_registry import paths
    from orchestration.pipeline_dag import default_baseline_metrics
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import mark_promoted

    seed_corpus(client)
    first = make_context(client, controller, out_version="v1")
    run_stages(first, stages_for("finetune"), command="finetune")
    mark_promoted(get("foundation-v1", client), client, promoted_by="test")
    client.write_json(paths.eval_report("v1"), {"gate_metrics": dict(PASSING_METRICS)})

    probe = make_context(client, controller, out_version="v1.1", baseline_metrics=None)
    baseline = default_baseline_metrics(probe)
    assert baseline == PASSING_METRICS


def test_no_promoted_version_means_no_baseline_not_a_lookup_failure(client, controller):
    from orchestration.pipeline_dag import default_baseline_metrics

    assert default_baseline_metrics(make_context(client, controller)) is None


# --------------------------------------------------------------------------
# Idempotency and resumption
# --------------------------------------------------------------------------


def test_a_completed_stage_re_runs_as_a_no_op(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    first = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert first.ok, first.render()

    again = make_context(client, controller)
    second = run_stages(again, stages_for("finetune"), command="finetune")

    by_name = {r.name: r for r in second.results}
    assert by_name["dataset_build"].status == "skipped"
    assert by_name["training"].status == "skipped"
    assert by_name["merge"].status == "skipped"
    assert "no-op" in by_name["training"].detail


def test_the_gate_is_never_skipped_as_already_complete():
    """An 'already gated' shortcut would let a re-run inherit a pass it did not
    earn. Stage 6 has no completion check, on purpose."""
    gate = next(s for s in STAGES if s.name == "evaluation_gate")
    assert gate.is_complete is None


def test_resuming_at_merge_does_not_retrain(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    resumed = make_context(client, controller)
    report = run_stages(resumed, stages_from("merge", "finetune"), command="finetune")
    assert [r.name for r in report.results] == ["merge"]
    assert "training" not in resumed.results


# --------------------------------------------------------------------------
# package
# --------------------------------------------------------------------------


def test_package_pushes_all_three_artifact_classes_and_publishes(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    package_ctx = make_context(client, controller)
    report = run_stages(package_ctx, stages_for("package"), command="package")
    assert report.ok, report.render()

    pushed = package_ctx.results["push"].data["pushed"]
    assert any(k.startswith("adapter:") for k in pushed)
    assert any(k.startswith("merged:") for k in pushed)
    assert any(k.startswith("quantized:") for k in pushed)

    from registry_utils.query_registry import get

    foundation = get("foundation-v1", client)
    assert foundation.artifacts.status == "published"
    assert foundation.artifacts.staging_path is None
    assert foundation.artifacts.adapter_weights

    # A Foundation run owns its adapter and nothing else. In a per-type build
    # the merged and quantized models are produced per doc type, and the
    # doc_type=None "unified" paths exist only under --foundation-only, so
    # claiming them here would point the manifest at artifacts never built.
    assert foundation.artifacts.merged_model is None
    assert foundation.artifacts.quantized_formats == []

    policy = get("policy-adapter-v1", client)
    assert policy.artifacts.merged_model
    assert policy.artifacts.quantized_formats == ["fp16", "q5_k_m"]


def test_a_foundation_only_build_publishes_the_unified_model(client, controller):
    """With no per-type adapters the unified paths ARE what was built, so the
    Foundation run legitimately owns them."""
    seed_corpus(client)
    ctx = make_context(client, controller, foundation_only=True)
    run_stages(ctx, stages_for("finetune"), command="finetune")
    run_stages(ctx, stages_for("package"), command="package")

    from registry_utils.query_registry import get

    foundation = get("foundation-v1", client)
    assert foundation.artifacts.merged_model
    assert foundation.artifacts.quantized_formats == ["fp16", "q5_k_m"]


def test_package_clears_staging_after_a_verified_push(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")
    run_stages(ctx, stages_for("package"), command="package")

    assert ctx.volume.list(paths.staging_root()) == []


def test_keep_staging_retains_the_volume_copy(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller, keep_staging=True)
    run_stages(ctx, stages_for("finetune"), command="finetune")
    run_stages(ctx, stages_for("package"), command="package")

    assert ctx.volume.list(paths.staging_root())


def test_package_fails_loudly_with_remediation_when_nothing_is_staged(client, controller):
    ctx = make_context(client, controller, skip_quantize=True)
    report = run_stages(ctx, stages_for("package"), command="package")

    assert report.failed_at == "push"
    detail = report.results[-1].detail
    assert "not on the staging volume" in detail
    assert "--from-stage merge" in detail and "--push-adapters" in detail


def test_push_adapters_leaves_only_the_merged_model_staged(client, controller):
    """Recommended when commands 1 and 2 may be separated by more than a day —
    the adapters are tens of MB, the merged model is ~16 GB (SPEC_13 §3)."""
    seed_corpus(client)
    ctx = make_context(client, controller, push_adapters=True)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    assert client.exists(f"{paths.adapter_dir('foundation', 'v1')}/adapter_placeholder.json")


# --------------------------------------------------------------------------
# The pod contract
# --------------------------------------------------------------------------


def test_a_pod_without_the_staging_volume_is_refused():
    """A pod with no volume writes to pod-local disk, terminates, and the work is
    gone — silently, because every write succeeded."""
    controller = RunPodController(backend=LocalBackend(), volume_id=None, git_commit="abc1234")
    with pytest.raises(PodLaunchError, match="no staging volume attached"):
        controller.launch(controller.spec_for("train_foundation"))


def test_a_pod_leaves_no_process_behind_even_when_the_job_raises(controller):
    with pytest.raises(RuntimeError, match="boom"), controller.pod(controller.spec_for("merge")):
        raise RuntimeError("boom")
    assert controller.active_pods == []


def test_run_job_reports_a_failure_instead_of_leaking_a_pod(controller):
    result = controller.run_job(controller.spec_for("quantize"), lambda _h: 1 / 0)
    assert not result.ok and "division" in (result.error or "")
    assert controller.active_pods == []


def test_ocr_does_not_get_the_a100(controller):
    """MinerU saturates a much cheaper card; reserving the A100 for Foundation
    training is what keeps preprocessing inexpensive (SPEC_13 §7)."""
    assert "A100" not in GPU_CLASS_BY_STAGE["preprocessing"]
    assert GPU_CLASS_BY_STAGE["train_foundation"].startswith("A100")


def test_finetune_uses_one_pod_for_the_whole_training_fan_out(client, controller):
    """Three launches would mean three cold starts and two Blob round trips of
    the same corpus."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    assert controller.backend._counter == 1  # type: ignore[attr-defined]
    assert controller.active_pods == []


def test_a_pod_will_not_clone_a_moving_branch():
    spec = PodSpec(name="p", stage="merge", gpu_class="L40S", volume_id="vol", git_commit=None)
    with pytest.raises(PodLaunchError, match="no resolvable git commit"):
        spec.clone_command("git@example.com:repo.git")


def test_an_unresolved_commit_is_not_treated_as_pinned():
    """`capture_git_commit` returns 'unknown' off a git tree; a pod cannot clone it."""
    spec = PodSpec(name="p", stage="merge", gpu_class="L40S", volume_id="vol", git_commit="unknown")
    with pytest.raises(PodLaunchError):
        spec.clone_command("git@example.com:repo.git")


def test_logs_are_scrubbed_before_they_leave_the_pod():
    scrubbed = scrub(
        "loss 0.42 for Policy No. WC-8842317-01 reviewer adjuster@carrier.com ssn 123-45-6789"
    )
    assert "WC-8842317-01" not in scrubbed
    assert "adjuster@carrier.com" not in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "loss 0.42" in scrubbed


def test_a_dry_run_does_not_claim_the_endpoint_moved(controller):
    """History records what was DEPLOYED. Recording a preview made health_check
    report a version that was never live, and the next rollback then "restored"
    the version already running."""
    controller.deploy_endpoint("v2", dry_run=True)
    assert controller.health_check()["deployed_version"] is None
    assert controller.health_check()["history"] == []


def test_rollback_needs_somewhere_to_roll_back_to(controller):
    with pytest.raises(EndpointError, match="nothing to roll back to"):
        controller.rollback_endpoint(dry_run=True)

    # Two real deployments. `deploy_endpoint` refuses to run for real until the
    # RunPod API is wired, so the history is seeded the way that call would.
    controller._endpoint_versions.extend(["v1", "v2"])
    assert controller.health_check()["deployed_version"] == "v2"

    assert controller.rollback_endpoint(dry_run=True) == "v1"
    assert controller.health_check()["deployed_version"] == "v1"


# --------------------------------------------------------------------------
# The CLI shell
# --------------------------------------------------------------------------


def test_finetune_requires_an_out_version():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["finetune", "--input", "./intake"])


def test_all_accepts_the_union_of_both_flag_sets():
    args = cli.build_parser().parse_args([
        "all", "--input", "./intake", "--out-version", "v2",
        "--formats", "fp16", "q5_k_m", "--keep-staging",
    ])
    assert args.out_version == "v2" and args.formats == ["fp16", "q5_k_m"] and args.keep_staging


def test_extract_takes_base_as_a_first_class_model():
    """`base` is the pilot's zero-shot baseline and the day-zero pre-annotation
    path, not a curiosity (SPEC_13 §5)."""
    args = cli.build_parser().parse_args(["extract", "--model", "base", "--input", "docs/"])
    assert args.model == "base" and args.command == "extract"


def test_the_cli_holds_no_stage_logic():
    """Stage order, idempotency and the gate live in pipeline_dag, so the CLI
    cannot develop its own opinion about them."""
    import inspect

    source = inspect.getsource(cli)
    assert "promotion_gate" not in source
    assert "train_foundation" not in source


def test_build_context_carries_the_flags_through(client, controller):
    args = cli.build_parser().parse_args([
        "finetune", "--input", "./intake", "--out-version", "v3",
        "--foundation-only", "--push-adapters", "--dry-run", "--min-labels-per-type", "5",
    ])
    ctx = cli.build_context(args, client=client, controller=controller,
                            raw_client=ingestion_client(client))
    assert ctx.out_version == "v3" and ctx.foundation_only and ctx.push_adapters
    assert ctx.dry_run and ctx.min_labels_per_type == 5
    assert ctx.corpus == "v3"  # corpus defaults to the output version


def test_run_command_dispatches_all_through_both_commands(client, controller):
    seed_corpus(client)
    args = cli.build_parser().parse_args(["all", "--input", "./intake", "--out-version", "v1"])
    ctx = cli.build_context(
        args, client=client, controller=controller, raw_client=ingestion_client(client),
        skip_ingest=True, dry_run=True, min_labels_per_type=1,
        metrics_provider=lambda _c: dict(PASSING_METRICS),
    )
    report = cli.run_command("all", ctx)

    assert report.ok, report.render()
    assert [r.name for r in report.results] == [s.name for s in stages_for("all")]
    assert report.as_dict()["command"] == "all"


def test_gate_blocked_is_distinguishable_from_a_failure():
    """Both exit non-zero, but only one prints per-metric deltas."""
    assert issubclass(GateBlocked, RuntimeError)
    assert not issubclass(GateBlocked, PipelineError)


# --------------------------------------------------------------------------
# The Foundation-upgrade cascade (arch §12)
# --------------------------------------------------------------------------


def _promote_foundation(client: BlobClient, controller: RunPodController, version: str) -> None:
    """Run a cycle and promote it, so the next one has something to bump from."""
    from evaluation.gating import apply_to_manifest, promotion_gate
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import mark_promoted

    ctx = make_context(client, controller, out_version=version)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.ok, report.render()

    passed = promotion_gate(dict(PASSING_METRICS), None)
    for run_id in (f"foundation-{version}", *(f"{dt}-adapter-{version}" for dt in ctx.doc_types)):
        # Gate first, then promote — the order a real cycle uses. `mark_promoted`
        # now requires the affirmative gate result, so a manifest that was never
        # evaluated cannot be promoted at all.
        manifest = apply_to_manifest(passed, get(run_id, client))
        mark_promoted(manifest, client, promoted_by="test")


def test_a_minor_bump_does_not_trigger_the_cascade(client, controller):
    seed_corpus(client)
    _promote_foundation(client, controller, "v1")

    ctx = make_context(client, controller, out_version="v1.1")
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.ok, report.render()


def test_a_major_bump_blocks_until_dependent_adapters_are_revalidated(client, controller):
    """Every per-type adapter was trained on the previous Foundation's weights,
    so promoting a new one without re-validating them ships three models that
    were never evaluated against the base they now sit on."""
    seed_corpus(client)
    _promote_foundation(client, controller, "v1")

    ctx = make_context(client, controller, out_version="v2")
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.blocked_at == "evaluation_gate"
    detail = report.results[-1].detail
    assert "dependent adapter" in detail
    assert "policy-adapter-v1" in detail


def test_the_cascade_clears_once_every_dependent_has_passed(client, controller):
    seed_corpus(client)
    _promote_foundation(client, controller, "v1")

    from orchestration.pipeline_dag import foundation_upgrade_work_list

    probe = make_context(client, controller, out_version="v2")
    dependents = foundation_upgrade_work_list(probe).dependents
    assert dependents, "the seeded adapters should depend on foundation-v1"

    ctx = make_context(client, controller, out_version="v2",
                       revalidation_evidence=dict.fromkeys(dependents, True))
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.ok, report.render()


def test_partial_revalidation_still_blocks(client, controller):
    seed_corpus(client)
    _promote_foundation(client, controller, "v1")

    from orchestration.pipeline_dag import foundation_upgrade_work_list

    probe = make_context(client, controller, out_version="v2")
    dependents = foundation_upgrade_work_list(probe).dependents
    partial = dict.fromkeys(dependents, True)
    partial[dependents[0]] = False

    ctx = make_context(client, controller, out_version="v2", revalidation_evidence=partial)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.blocked_at == "evaluation_gate"
    assert dependents[0] in report.results[-1].detail


@pytest.mark.parametrize(
    ("previous", "candidate", "expected"),
    [
        ("foundation-v1", "v2", True),
        ("foundation-v1", "v1.1", False),
        ("foundation-v2.3", "v3", True),
        (None, "v1", False),
    ],
)
def test_major_bump_detection(previous, candidate, expected):
    from orchestration.pipeline_dag import is_major_bump

    assert is_major_bump(previous, candidate) is expected


# --------------------------------------------------------------------------
# Retry policy and configuration
# --------------------------------------------------------------------------


def test_a_transient_stage_failure_is_retried(client, controller):
    from orchestration.pipeline_dag import Stage, StageResult

    calls = {"n": 0}

    def flaky(_ctx):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient object-store error")
        return StageResult("flaky", "completed", "second attempt")

    ctx = make_context(client, controller, max_attempts=2)
    report = run_stages(ctx, [Stage(1, "flaky", "finetune", False, flaky)], command="finetune")
    assert report.ok and calls["n"] == 2


def test_a_blocked_gate_is_never_retried(client, controller):
    """Retrying a gate block would be an override path with extra steps."""
    from orchestration.pipeline_dag import Stage

    calls = {"n": 0}

    def blocking(_ctx):
        calls["n"] += 1
        raise GateBlocked("regressed")

    ctx = make_context(client, controller, max_attempts=5)
    report = run_stages(ctx, [Stage(1, "gate", "finetune", False, blocking)], command="finetune")
    assert report.blocked_at == "gate" and calls["n"] == 1


def test_the_config_file_is_actually_read():
    """A config file nothing reads documents a policy the system does not follow."""
    from orchestration import settings

    assert settings.gpu_class_for("train_foundation").startswith("A100")
    assert "A100" not in settings.gpu_class_for("preprocessing")
    assert settings.retry_policy()["retry_on_gate_block"] is False
    assert settings.defaults()["formats"] == ["fp16", "q5_k_m"]


def test_cli_defaults_come_from_the_config_file():
    from orchestration import settings

    args = cli.build_parser().parse_args(["package", "--version", "v2"])
    assert args.formats == settings.defaults()["formats"]
    assert args.dtype == settings.defaults()["dtype"]


def test_the_gpu_flag_actually_reaches_the_training_pod(client, controller):
    """A flag that parses and changes nothing is worse than no flag."""
    seed_corpus(client)
    args = cli.build_parser().parse_args([
        "finetune", "--out-version", "v1", "--gpu", "H100-80G", "--skip-ingest",
    ])
    ctx = cli.build_context(args, client=client, controller=controller,
                            raw_client=ingestion_client(client),
                            min_labels_per_type=1, dry_run=True,
                            metrics_provider=lambda _c: dict(PASSING_METRICS))
    assert ctx.gpu_class == "H100-80G"

    launched: list[str] = []
    original = controller.launch
    controller.launch = lambda spec: (launched.append(spec.gpu_class), original(spec))[1]  # type: ignore[method-assign]
    run_stages(ctx, stages_for("finetune"), command="finetune")
    assert launched == ["H100-80G"]
