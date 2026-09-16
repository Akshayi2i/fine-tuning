"""SPEC_06 — masking, the ViT gate, early stopping, and training configuration.

The masking tests here are the highest-value in the suite. Broken masking trains
the model to reproduce its own prompt, converges to a normal-looking loss curve,
and shows no symptom at all until evaluation — by which point a training run has
been spent. So the guard is **mutation-checked**: a deliberately broken masking
implementation must fail it.
"""

from __future__ import annotations

import pytest

from registry_utils.models import DataStats
from training import train_adapter as TA
from training import train_foundation as TF
from training.callbacks.early_stopping import EarlyStoppingState
from training.data_collator import (
    IGNORE_INDEX,
    MaskingError,
    assert_masking_correct,
    find_assistant_span,
    reference_labels,
    verify_batch,
)
from training.vit_gate import (
    ErrorRecord,
    classify_error,
    evaluate_gate,
    summarise_error_mix,
)

# A toy sequence: 10 prompt tokens, then 6 assistant tokens.
INPUT_IDS = list(range(100, 116))
ASSISTANT_START, ASSISTANT_END = 10, 16


# --------------------------------------------------------------------------
# Label masking — the critical guard
# --------------------------------------------------------------------------

def test_correct_masking_passes():
    labels = reference_labels(INPUT_IDS, ASSISTANT_START, ASSISTANT_END)
    report = assert_masking_correct(labels, ASSISTANT_START, ASSISTANT_END)
    assert report.is_correct
    assert report.supervised_tokens == 6
    assert report.masked_tokens == 10


def test_supervising_prompt_tokens_is_caught():
    """MUTATION CHECK. This is the failure the module exists to prevent: the
    model trains on reproducing its own system prompt, and the loss curve looks
    entirely normal while it happens."""
    labels = reference_labels(INPUT_IDS, ASSISTANT_START, ASSISTANT_END)
    labels[3] = INPUT_IDS[3]                       # leak one prompt token
    with pytest.raises(MaskingError, match="PROMPT token"):
        assert_masking_correct(labels, ASSISTANT_START, ASSISTANT_END)


def test_masking_out_assistant_tokens_is_caught():
    """The opposite failure: part of the JSON the model must produce is not
    trained on."""
    labels = reference_labels(INPUT_IDS, ASSISTANT_START, ASSISTANT_END)
    labels[12] = IGNORE_INDEX
    with pytest.raises(MaskingError, match="ASSISTANT token"):
        assert_masking_correct(labels, ASSISTANT_START, ASSISTANT_END)


def test_fully_masked_batch_is_caught():
    """Training would complete successfully and produce an adapter identical to
    its initialisation."""
    labels = [IGNORE_INDEX] * len(INPUT_IDS)
    with pytest.raises(MaskingError, match="NO tokens are supervised"):
        assert_masking_correct(labels, ASSISTANT_START, ASSISTANT_END)


def test_masking_everything_is_caught():
    """No masking at all — the whole sequence supervised."""
    with pytest.raises(MaskingError, match="PROMPT token"):
        assert_masking_correct(list(INPUT_IDS), ASSISTANT_START, ASSISTANT_END)


def test_batch_verification_across_rows():
    rows = [
        reference_labels(INPUT_IDS, ASSISTANT_START, ASSISTANT_END),
        reference_labels(INPUT_IDS, 8, 16),
    ]
    report = verify_batch({"labels": rows}, [(ASSISTANT_START, ASSISTANT_END), (8, 16)])
    assert report.supervised_tokens == 6 + 8


def test_batch_without_labels_is_caught():
    """Without labels nothing is supervised and training silently does nothing."""
    with pytest.raises(MaskingError, match="no `labels`"):
        verify_batch({"input_ids": INPUT_IDS}, [(0, 1)])


def test_one_bad_row_fails_the_whole_batch():
    good = reference_labels(INPUT_IDS, ASSISTANT_START, ASSISTANT_END)
    bad = list(INPUT_IDS)
    with pytest.raises(MaskingError, match=r"row 1"):
        verify_batch({"labels": [good, bad]}, [(ASSISTANT_START, ASSISTANT_END), (ASSISTANT_START, ASSISTANT_END)])


def test_assistant_span_is_located_from_header_tokens():
    ids = [1, 2, 3, 77, 78, 40, 41, 42, 99]
    start, end = find_assistant_span(ids, assistant_header_ids=[77, 78], end_token_id=99)
    assert (start, end) == (5, 9)
    assert ids[start:end] == [40, 41, 42, 99]


def test_the_span_includes_the_end_of_turn_token():
    """The model has to be taught to stop. Excluding the end token left EOS
    unsupervised, so generation runs to max_new_tokens — which the runner
    reports as truncation, i.e. as silently dropped rows."""
    ids = [1, 2, 77, 78, 40, 99]
    start, end = find_assistant_span(ids, assistant_header_ids=[77, 78], end_token_id=99)
    assert ids[end - 1] == 99

    labels = reference_labels(ids, start, end)
    assert labels[-1] == 99, "the end-of-turn token is masked out of the loss"


def test_a_span_past_a_truncated_row_is_a_masking_error():
    """Spans computed before truncation do not survive it. A bare IndexError
    gives none of the diagnosis this module exists to provide."""
    with pytest.raises(MaskingError, match="does not fit a row"):
        verify_batch({"labels": [[IGNORE_INDEX] * 8]}, [(4, 20)])


def test_a_missing_assistant_turn_is_caught():
    """Either the chat template changed or the header ids are wrong — both would
    mask the wrong span."""
    with pytest.raises(MaskingError, match="assistant turn was not found"):
        find_assistant_span([1, 2, 3], assistant_header_ids=[77, 78])


# --------------------------------------------------------------------------
# The ViT gate (arch §3, §16c)
# --------------------------------------------------------------------------

def _perception_errors(n: int = 10, modality: str = "image_only") -> list[ErrorRecord]:
    """Errors from the image-only subset by default: accuracy is gated on
    image-only and scanned documents, so the mix that decides comes from there."""
    return [
        ErrorRecord(f"policy_{i:04d}", "policy_number", "WC-8842317-01", "WC-8842317-0l",
                    error_class="perception", modality_mode=modality)
        for i in range(n)
    ]


def _reasoning_errors(n: int = 10, modality: str = "image_only") -> list[ErrorRecord]:
    return [
        ErrorRecord(f"policy_{i:04d}", "insured_name", "Rivera Fabrication LLC",
                    "Meridian Property Group LLC", error_class="schema_reasoning",
                    modality_mode=modality)
        for i in range(n)
    ]


def test_gate_refuses_to_decide_on_pilot_volume():
    """arch §16c — at pilot volume low accuracy is a data artifact, and thin data
    produces exactly the perception-shaped errors the gate keys on. Firing here
    would buy a retrain for a problem more documents would solve."""
    decision = evaluate_gate(
        image_only_accuracy=0.55, scanned_accuracy=0.60,
        image_only_documents=9, scanned_documents=7,     # pilot volume
        errors=_perception_errors(),
    )
    assert decision.decision == "insufficient_data"
    assert not decision.should_train_vit
    assert "data-volume artifact" in decision.rationale
    assert "resolution cap" in decision.recommendation   # the cheap fix comes first


def test_gate_holds_when_accuracy_meets_target():
    decision = evaluate_gate(
        image_only_accuracy=0.88, scanned_accuracy=0.85,
        image_only_documents=50, scanned_documents=40,
        errors=_perception_errors(3),
    )
    assert decision.decision == "hold"
    assert "train_vit: false" in decision.recommendation


def test_gate_holds_when_errors_are_reasoning_not_perception():
    """THE load-bearing distinction. If the model reads the character correctly
    and puts it in the wrong field, unfreezing the ViT will not help."""
    decision = evaluate_gate(
        image_only_accuracy=0.55, scanned_accuracy=0.60,
        image_only_documents=50, scanned_documents=40,
        errors=_reasoning_errors(),
    )
    assert decision.decision == "hold"
    assert "schema_reasoning" in decision.rationale
    assert not decision.should_train_vit


def test_gate_fires_only_on_low_accuracy_and_perception_errors():
    decision = evaluate_gate(
        image_only_accuracy=0.55, scanned_accuracy=0.60,
        image_only_documents=50, scanned_documents=40,
        errors=_perception_errors(),
    )
    assert decision.decision == "fire"
    assert decision.should_train_vit
    assert "Never a full fine-tune" in decision.recommendation


def test_even_when_firing_the_cheap_fixes_come_first():
    """Resolution cap and modality rebalance need no new labeling; a ViT LoRA
    costs a full Foundation retrain."""
    decision = evaluate_gate(
        image_only_accuracy=0.50, scanned_accuracy=0.55,
        image_only_documents=60, scanned_documents=60,
        errors=_perception_errors(20),
    )
    assert decision.decision == "fire"
    assert "resolution cap" in decision.recommendation
    assert "modality mix" in decision.recommendation


def test_a_subset_nobody_scored_is_not_a_subset_that_passed():
    """Skipping the count check on a None fell through to the accuracy check,
    where `value is not None` skipped it again — so the gate returned `hold`
    with a rationale asserting a target was met that nobody had measured."""
    decision = evaluate_gate(
        image_only_accuracy=None, scanned_accuracy=None,
        image_only_documents=0, scanned_documents=0,
        errors=[],
    )
    assert decision.decision == "insufficient_data"
    assert "never scored" in decision.rationale


def test_one_scored_subset_does_not_carry_an_unscored_one():
    decision = evaluate_gate(
        image_only_accuracy=0.92, scanned_accuracy=None,
        image_only_documents=60, scanned_documents=0,
        errors=[],
    )
    assert decision.decision == "insufficient_data"


def test_the_deciding_error_mix_comes_from_the_gated_subsets_only():
    """Accuracy is read from image-only and scanned documents, so the mix has to
    be too. Over every error, an ocr_plus_image majority — the largest subset,
    and where perception is least implicated — drowns a genuine image-only
    perception failure under the 60% threshold."""
    errors = (
        _perception_errors(9, modality="image_only")
        + _reasoning_errors(30, modality="ocr_plus_image")
    )
    decision = evaluate_gate(
        image_only_accuracy=0.55, scanned_accuracy=0.60,
        image_only_documents=50, scanned_documents=40,
        errors=errors,
    )
    assert decision.decision == "fire", "the ocr_plus_image majority outvoted the gated subsets"
    assert decision.error_mix["perception"] == pytest.approx(1.0)
    assert decision.overall_error_mix["perception"] < 0.60, "the overall mix is still reported"


def test_a_scanned_error_counts_even_when_its_modality_is_ocr_plus_image():
    """Scanned is a gated subset in its own right; a scanned document normally
    still carries OCR."""
    from training.vit_gate import in_gated_subsets

    scanned = ErrorRecord("policy_0001", "policy_number", "A", "4",
                          error_class="perception", modality_mode="ocr_plus_image",
                          is_scanned=True)
    plain = ErrorRecord("policy_0002", "policy_number", "A", "4",
                        error_class="perception", modality_mode="ocr_plus_image")
    assert in_gated_subsets([scanned, plain]) == [scanned]


def test_the_gate_reads_the_keys_an_eval_report_actually_emits():
    """It read `eval_metrics`/`subset_counts`/`error_records`, none of which
    EvalReport.as_dict emits — so every well-formed report produced None
    accuracies and zero documents, and the gate decided from nothing."""
    from evaluation.run_eval import EvalReport, SubsetReport
    from training.vit_gate import evaluate_from_report

    report = EvalReport(model_version="foundation-v1", corpus_version="v1", subsets=[
        SubsetReport("policy", "full", documents=90,
                     metrics={"field_normalized_match": 0.91}),
        SubsetReport("policy", "image_only", documents=50,
                     metrics={"field_normalized_match": 0.55},
                     error_records=[{"source_id": "policy_0001", "field_path": "policy_number",
                                     "expected": "WC-8842317-01", "got": "WC-8842317-0l",
                                     "error_class": "perception"}] * 10),
        SubsetReport("policy", "scanned", documents=40,
                     metrics={"field_normalized_match": 0.60}),
    ]).as_dict()

    decision = evaluate_from_report(report)

    assert decision.image_only_accuracy == pytest.approx(0.55)
    assert decision.scanned_accuracy == pytest.approx(0.60)
    assert decision.image_only_documents == 50 and decision.scanned_documents == 40
    assert decision.error_mix["perception"] == pytest.approx(1.0)
    assert decision.decision == "fire"


def test_a_payload_that_is_not_an_eval_report_is_refused():
    """Guessing at the shape is how the gate came to decide from all-None."""
    from training.vit_gate import evaluate_from_report

    with pytest.raises(ValueError, match="not an EvalReport payload"):
        evaluate_from_report({"eval_metrics": {"image_only_accuracy": 0.5}})


def test_error_classification_separates_misreading_from_misplacing():
    # A near-miss on the expected string: misread character.
    assert classify_error("WC-8842317-01", "WC-8842317-0l") == "perception"
    # A different field's value: read correctly, placed wrongly.
    assert classify_error(
        "Rivera Fabrication LLC", "Meridian Property Group LLC",
        all_expected={"insured_name": "Rivera Fabrication LLC",
                      "certificate_holder": "Meridian Property Group LLC"},
    ) == "schema_reasoning"
    # Nothing produced.
    assert classify_error("Rivera Fabrication LLC", None) == "omission"


def test_error_mix_is_summarised_as_shares():
    mix = summarise_error_mix(_perception_errors(6) + _reasoning_errors(4))
    assert mix["perception"] == pytest.approx(0.6)
    assert mix["schema_reasoning"] == pytest.approx(0.4)


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------

def test_early_stopping_tracks_field_f1_not_loss():
    """Loss can improve while extraction gets worse — loss rewards fluent JSON,
    and a confidently wrong value is fluent."""
    state = EarlyStoppingState(patience=2)
    assert not state.update(100, field_f1=0.70, eval_loss=0.45)
    assert not state.update(200, field_f1=0.78, eval_loss=0.38)
    assert not state.update(300, field_f1=0.77, eval_loss=0.35)
    assert state.update(400, field_f1=0.76, eval_loss=0.31)      # stop, despite falling loss
    assert state.best_score == 0.78 and state.best_step == 200


def test_improvement_resets_patience():
    state = EarlyStoppingState(patience=2)
    state.update(100, 0.70)
    state.update(200, 0.69)
    assert not state.update(300, 0.75)
    assert state.evaluations_without_improvement == 0


# --------------------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------------------

def test_foundation_config_is_bf16_lora_with_a_frozen_vit():
    """The arch §9 default: LoRA on a bf16 base, not QLoRA. Both serving paths
    hold the base in bf16/fp16, so training in bf16 means the adapter is applied
    to exactly the weights it trained against."""
    swift, recorded = TF.build_swift_config(
        corpus_paths=["corpus/default/v1/policy/train.jsonl"], output_dir="/tmp/out"
    )
    assert swift.args["quantization_bit"] == 0
    # Emitted only under 4-bit: a bnb setting in the rendered command line of a
    # bf16 run would describe nothing the run did.
    assert "bnb_4bit_quant_type" not in swift.args
    assert "bnb_4bit_use_double_quant" not in swift.args
    assert swift.args["freeze_vit"] is True
    assert recorded.technique == "LoRA"
    assert recorded.base_quantization == "bf16_frozen_base"
    assert recorded.vit_trainable is False
    assert recorded.vit_method == "frozen"
    assert recorded.lora_rank == 64 and recorded.lora_alpha == 128


def test_four_bit_is_a_live_flag_not_a_hardcoded_constant(monkeypatch):
    """`load_in_4bit` sat in the YAML being read by nothing while both trainers
    hardcoded `quantization_bit: 4`, so the QLoRA-vs-LoRA decision could not be
    A/B'd without a code change — and the manifest recorded a value that had no
    effect on the run."""
    base = TF.base_model_config()
    quantized = {**base, "quantization": {**base["quantization"], "load_in_4bit": True}}
    monkeypatch.setattr(TF, "base_model_config", lambda: quantized)

    swift, recorded = TF.build_swift_config(corpus_paths=["x"], output_dir="/tmp/out")

    assert swift.args["quantization_bit"] == 4
    assert swift.args["bnb_4bit_quant_type"] == "nf4"
    assert swift.args["bnb_4bit_use_double_quant"] is True
    assert recorded.technique == "QLoRA"
    assert recorded.base_quantization == "nf4_double_quant_bfloat16_compute"


def test_the_manifest_cannot_default_its_way_into_claiming_qlora():
    """These three carried "QLoRA" / NF4 / paged-8-bit as model defaults, so a
    bf16 run that did not pass them recorded a technique it never used. The
    record is the whole basis for attributing a regression, and base precision is
    exactly the kind of change that causes one."""
    import pytest

    from registry_utils.models import TrainingConfig

    with pytest.raises(ValueError):
        TrainingConfig(
            lora_rank=64, lora_alpha=128, learning_rate=1e-4, epochs=3,
            gradient_accumulation_steps=32, effective_batch_size=32,
            target_modules=["q_proj"], resolution_cap_px=1792, max_seq_len=8192, seed=42,
        )


def test_the_adapter_holds_the_base_the_same_way_as_its_foundation():
    """A rank-16 adapter stacked on a Foundation trained against a differently
    held base is a mismatch the promotion gate has no way to see."""
    found, _ = TF.build_swift_config(corpus_paths=["x"], output_dir="/tmp/out")
    adapter, _ = TA.build_adapter_config(
        "acord", corpus_version="v1",
        foundation_adapter_path="/runpod-volume/staging/adapters/foundation/v1",
        output_dir="/runpod-volume/staging/adapters/acord/v1",
    )
    assert adapter.args["quantization_bit"] == found.args["quantization_bit"]


def test_the_vision_projector_is_a_lora_target():
    """Where image evidence fuses with language — the locus of OCR-versus-image
    arbitration (arch §9a)."""
    swift, recorded = TF.build_swift_config(corpus_paths=["x"], output_dir="/tmp/out")
    assert "merger" in swift.args["lora_target_modules"]
    assert "merger" in recorded.target_modules


def test_training_the_vit_is_recorded_as_lora_never_full_fine_tune():
    """The §3 escalation is 'add a ViT LoRA'. The manifest model rejects any
    other combination outright."""
    swift, recorded = TF.build_swift_config(corpus_paths=["x"], output_dir="/tmp/o", train_vit=True)
    assert swift.args["freeze_vit"] is False
    assert recorded.vit_trainable is True
    assert recorded.vit_method == "lora"


def test_foundation_trains_across_every_doc_type():
    """The mixed corpus is what makes the Foundation learn shared behaviour."""
    from common.constants import ACTIVE_DOC_TYPES

    swift, _ = TF.build_swift_config(
        corpus_paths=[f"corpus/default/v1/{dt}/train.jsonl" for dt in ACTIVE_DOC_TYPES],
        output_dir="/tmp/out",
    )
    for doc_type in ACTIVE_DOC_TYPES:
        assert any(doc_type in path for path in swift.args["dataset"])


def test_per_type_adapter_is_rank_16_on_a_frozen_vit():
    swift, recorded = TA.build_adapter_config(
        "lossrun", corpus_version="v1",
        foundation_adapter_path="/runpod-volume/staging/adapters/foundation/v1",
        output_dir="/tmp/out",
    )
    assert recorded.lora_rank == 16 and recorded.lora_alpha == 32
    assert swift.args["freeze_vit"] is True          # never escalated at this layer
    assert "foundation" in swift.args["adapters"][0]


def test_per_type_adapter_trains_on_one_type_only():
    swift, _ = TA.build_adapter_config(
        "lossrun", corpus_version="v1", foundation_adapter_path="/f", output_dir="/tmp/out"
    )
    assert all("lossrun" in path for path in swift.args["dataset"])


def test_adapter_without_a_promoted_foundation_is_refused():
    """A per-type adapter is a specialisation of a Foundation, not a standalone
    model."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    with pytest.raises(TA.TrainingError, match="no promoted Foundation"):
        TA.resolve_foundation(client)


def test_unknown_doc_type_is_refused():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    with pytest.raises(TA.TrainingError, match="unknown doc_type"):
        TA.train_adapter(
            "invoice", corpus_version="v1", out_version="v1", client=client,
            corpus_manifest={}, data_stats=DataStats(train_examples=1, val_examples=1, test_examples=1),
            dry_run=True,
        )


def test_swift_cli_renders_flags_correctly():
    swift, _ = TF.build_swift_config(corpus_paths=["a.jsonl", "b.jsonl"], output_dir="/tmp/out")
    argv = swift.to_cli()
    assert argv[:2] == ["swift", "sft"]
    assert "--lora_rank" in argv and "64" in argv
    assert "--freeze_vit" in argv                    # boolean True renders as a bare flag


def test_no_code_path_can_enable_full_vit_fine_tuning():
    """arch §3 — the escalation is LoRA-on-ViT, never a full fine-tune, because a
    full one risks the pretrained document and OCR capability the image-only
    pathway depends on.

    Asserted structurally rather than by reading a config value: the type makes
    "full" unrepresentable, the manifest validator rejects the combination, and
    no builder emits a full-finetune flag. All three have to hold.
    """
    import inspect
    import typing

    from registry_utils.models import RunManifest
    from training import train_adapter, train_foundation

    allowed = typing.get_args(RunManifest.model_fields["training_config"].annotation
                              .model_fields["vit_method"].annotation)
    assert set(allowed) == {"frozen", "lora"}, "the type permits a method other than frozen/lora"

    for module in (train_foundation, train_adapter):
        source = inspect.getsource(module)
        for forbidden in ('train_type="full"', "full_finetune", "freeze_vit=False,  # full"):
            assert forbidden not in source, f"{module.__name__} has a full fine-tune path"


def test_a_trainable_vit_must_declare_lora():
    """The manifest refuses the combination outright, so a run cannot record a
    ViT escalation it did not perform by any method."""
    import pytest

    from registry_utils.models import TrainingConfig

    with pytest.raises(ValueError, match="vit_method"):
        TrainingConfig(
            technique="LoRA", base_quantization="bf16_frozen_base", optimizer="adamw_torch",
            lora_rank=64, lora_alpha=128, learning_rate=1e-4, epochs=3,
            gradient_accumulation_steps=32, effective_batch_size=32,
            target_modules=["q_proj"], resolution_cap_px=1792, max_seq_len=8192, seed=42,
            vit_trainable=True, vit_method="frozen",
        )


def test_the_adapter_run_attaches_the_foundation_rather_than_resuming_it():
    """`resume_from_checkpoint` means "continue THIS run": ms-swift restores the
    optimizer state and the completed global_step, so a fresh 3-epoch adapter run
    resumes at the end of the Foundation's schedule and trains zero steps. It
    would also load rank-64 Foundation weights into a rank-16 LoRA config."""
    swift, _recorded = TA.build_adapter_config(
        "acord", corpus_version="v1",
        foundation_adapter_path="/runpod-volume/staging/adapters/foundation/v1",
        output_dir="/runpod-volume/staging/adapters/acord/v1",
    )
    assert swift.args["adapters"] == ["/runpod-volume/staging/adapters/foundation/v1"]
    assert "resume_from_checkpoint" not in swift.args


def test_neither_trainer_puts_the_validation_split_in_the_training_set():
    """Both entrypoints, because fixing one and reporting both was the mistake.
    ms-swift carves its own eval split out of `--dataset`, so a val split passed
    there is trained on — and `metric_for_best_model` then selects on memorised
    documents that the promotion gate reads."""
    pass  # build_swift_config via TF

    adapter, _a = TA.build_adapter_config(
        "acord", corpus_version="v1", foundation_adapter_path="/f", output_dir="/o",
    )
    foundation, _f = TF.build_swift_config(
        corpus_paths=["corpus/default/v1/policy/train.jsonl"],
        val_paths=["corpus/default/v1/policy/val.jsonl"],
        output_dir="/o",
    )

    for name, swift in (("adapter", adapter), ("foundation", foundation)):
        train = swift.args["dataset"]
        assert all("val" not in path for path in train), f"{name} trains on its val split"
        assert swift.args.get("val_dataset"), f"{name} passes no separate val_dataset"


def test_an_unregistered_foundation_is_refused_rather_than_guessed():
    """Constructing a staging path meant an explicit --foundation with no
    manifest trained against a directory nobody had checked."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from training.train_adapter import _foundation_checkpoint

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    with pytest.raises(TA.TrainingError, match="no run manifest"):
        _foundation_checkpoint(client, "foundation-v9", "v9")


def test_explicit_evaluation_settings_win_over_the_early_stopping_defaults():
    """Unpacking the helper last silently replaced this config's
    metric_for_best_model with the helper's own default."""
    from common.config import training_config

    swift, _recorded = TA.build_adapter_config(
        "acord", corpus_version="v1", foundation_adapter_path="/f", output_dir="/o",
    )
    expected = training_config("acord_adapter")["evaluation"]["metric_for_best_model"]
    assert swift.args["metric_for_best_model"] == expected
    assert swift.args["early_stopping_patience"]


# --------------------------------------------------------------------------
# A manifest must never claim more than actually happened
# --------------------------------------------------------------------------


def _blob():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _pending_manifest():
    from registry_utils.models import (
        Artifacts,
        DataStats,
        Dependencies,
        RunManifest,
        TrainingConfig,
    )

    return RunManifest(
        run_id="foundation-v1", run_type="foundation",
        dependencies=Dependencies(base_model="qwen3-vl-8b-instruct@abc1234",
                                  corpus_version="corpus/v1", code_git_commit="abc1234"),
        training_config=TrainingConfig(
            technique="LoRA", base_quantization="bf16_frozen_base", optimizer="adamw_torch",
            lora_rank=64, lora_alpha=128, learning_rate=1e-4, epochs=3,
            gradient_accumulation_steps=32, effective_batch_size=32,
            target_modules=["q_proj"], resolution_cap_px=1792, max_seq_len=8192, seed=42),
        data_stats=DataStats(train_examples=10, val_examples=2, test_examples=2),
        artifacts=Artifacts(status="staged", staging_path="/runpod-volume/staging/x"),
        status="training",
    )


def test_a_run_that_never_launched_is_not_recorded_as_trained(monkeypatch):
    """The manifest is written before ms-swift starts — the run_id has to be
    reserved and the config captured even for a run that dies. But a pod that
    OOMs at step 40 must not leave a registry entry asserting a trained adapter
    and a staging path holding nothing."""
    client, manifest = _blob(), _pending_manifest()
    monkeypatch.setattr(TF, "launch", lambda _c: None)

    TF.launch_and_record(TF.SwiftConfig(args={}), manifest, client)
    assert manifest.status == "trained"


def test_a_crashed_run_is_recorded_as_failed(monkeypatch):
    client, manifest = _blob(), _pending_manifest()

    def oom(_config):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(TF, "launch", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        TF.launch_and_record(TF.SwiftConfig(args={}), manifest, client)

    assert manifest.status == "failed"
    from registry_utils.query_registry import get
    assert get("foundation-v1", client).status == "failed"


def test_a_registry_run_id_is_refused_where_a_checkpoint_path_belongs():
    """`continue_from` reaches ms-swift as resume_from_checkpoint, which reads a
    directory. A run-id finds no checkpoint, trains from base, and the manifest
    records `continued_from` — a lineage that never happened, later read as
    evidence for how much regression testing a promotion needs."""
    with pytest.raises(TF.TrainingError, match="checkpoint DIRECTORY"):
        TF.assert_checkpoint_path("foundation-v3")
    with pytest.raises(TF.TrainingError, match="local checkpoint directory"):
        TF.assert_checkpoint_path("https://blob.core.windows.net/main/adapters/foundation/v3")

    TF.assert_checkpoint_path("/runpod-volume/staging/adapters/foundation/v3")   # fine


# --------------------------------------------------------------------------
# Classification of nested misplacement
# --------------------------------------------------------------------------


def test_a_value_lifted_from_the_wrong_list_row_is_not_a_misread():
    """A loss run's confusables live inside `claims[]`. Reading only top-level
    fields sent the canonical misplacement error to the near-miss check, where
    two similar amounts read as a misread character — and that classification is
    what decides whether a Foundation retrain is justified."""
    golden = {
        "insured_name": "Rivera Fabrication LLC",
        "claims": [
            {"claim_number": "CLM-00417", "amount": "12500.00"},
            {"claim_number": "CLM-00418", "amount": "12800.00"},
        ],
    }
    # Row 1's amount answered with row 2's: read correctly, placed wrongly.
    assert classify_error("12500.00", "12800.00", all_expected=golden) == "schema_reasoning"
    # A genuine misread of a value that appears nowhere else stays perception.
    assert classify_error("12500.00", "125OO.00", all_expected=golden) == "perception"


def test_the_rank_phase_has_a_grid_for_both_adapter_types():
    """Every other phase does. Sweeping rank for the Foundation only meant
    `--adapter-type per_type` raised at phase 3 — after six runs were spent."""
    from training.sweep import candidates_for, load_phase

    for phase in ("lr", "epochs", "rank"):
        for adapter_type in ("foundation", "per_type"):
            assert candidates_for(phase, adapter_type=adapter_type), \
                f"{phase} has no grid for {adapter_type}"

    per_type = [c.value for c in candidates_for("rank", adapter_type="per_type")]
    assert 16 in per_type, "the per-type grid should bracket the rank-16 default"
    assert max(per_type) < max(load_phase("rank")["grid"]["foundation"]["lora_rank"])


def test_an_empty_document_count_is_not_enough_data():
    """An empty dict is a caller that did not count, not a corpus that is large
    enough — the same absence-is-success mistake the pipeline refuses elsewhere."""
    from training.sweep import SweepError, assert_enough_data

    with pytest.raises(SweepError, match="no per-type document counts"):
        assert_enough_data({})


def test_an_evaluation_with_no_loss_records_no_loss():
    """`or 0.0` wrote a loss of zero — a perfect model — for an evaluation that
    reported none, and that 0.0 then read as the best value in the history."""
    state = EarlyStoppingState(patience=2)
    state.update(100, field_f1=0.70)
    state.update(200, field_f1=0.75, eval_loss=0.31)

    assert "eval_loss" not in state.history[0]
    assert state.history[1]["eval_loss"] == 0.31
