"""``RunManifest`` — the lineage record for every training run (arch §12).

One manifest per run, Foundation and per-type alike, no exceptions. It is the
single record linking *code version + data version + config + resulting metrics
+ artifact location*, and without it these questions have no answer:

* which corpus version produced ``lossrun-adapter-v2.1``?
* why did ACORD accuracy drop between v2 and v3?
* which adapters depend on ``foundation-v2.0`` and must be re-validated now that
  Foundation has moved?

The manifest is written **even when the weights are only staged** on the RunPod
volume, because a volume is working storage with no durability guarantee. Without
that rule, a reclaimed volume means a training run that happened and left no
trace (master §12a).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RunType = Literal["foundation", "per_type_adapter"]
#: ``training`` exists because the manifest is written BEFORE ms-swift is
#: launched — the run_id has to be reserved and the config recorded even if
#: the run dies. Without it the pre-launch manifest defaulted to "trained",
#: so a pod that OOM'd at step 40 left a registry entry asserting a trained
#: adapter and a staging path holding nothing.
RunStatus = Literal["training", "trained", "evaluated", "promoted", "archived", "failed"]
ArtifactStatus = Literal["staged", "published"]
SweepPhase = Literal["lr", "epochs", "rank"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Dependencies(_Base):
    """Everything needed to reproduce the run.

    Reproducibility is the point: the first time your best model turns out to be
    one you cannot re-create, this is what makes it recoverable.
    """

    base_model: str = Field(..., description="qwen3-vl-8b-instruct@<hf_revision_pin>")
    corpus_version: str = Field(..., description="e.g. corpus/v3")
    code_git_commit: str = Field(..., description="exact repo commit that ran this")

    #: Which Foundation this adapter sits on. None for a Foundation run. This is
    #: what makes the arch §12 dependency-upgrade rule a query rather than an audit.
    foundation_version: str | None = None

    #: The model learns HOW MinerU formats its output, so a version bump is
    #: distribution shift, not a dependency bump (arch §8a).
    mineru_version: str | None = None
    ocr_device: str | None = Field(None, description="cuda | cpu — part of the pin if output differs")

    #: A change to either forces a corpus rebuild and a new training cycle (arch §7).
    schema_version: str | None = None
    prompt_template_version: str | None = None

    @field_validator("base_model")
    @classmethod
    def _revision_must_be_pinned(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError(
                "base_model must pin a revision as 'model_id@revision'. An unpinned base means "
                "two runs can silently use different weights, which makes a regression unattributable."
            )
        return v


class TrainingConfig(_Base):
    """The configuration that actually ran — not what a YAML file says today."""

    #: Required, not defaulted. These three described the *technique* while
    #: silently defaulting to QLoRA / NF4 / paged-8-bit, so a run that held the
    #: base in bf16 and never passed them recorded itself as a 4-bit QLoRA run.
    #: A manifest that misreports the technique is worse than no manifest: the
    #: whole point is that "why did this regress" is answerable from the record,
    #: and base precision is exactly the kind of change that causes a regression.
    technique: Literal["LoRA", "QLoRA"]

    #: ``bf16_frozen_base``, or ``<quant_type>_<double|single>_quant_<dtype>_compute``
    #: under 4-bit. Built by ``training.base_precision.manifest_descriptor`` from
    #: the config that ran, never hand-written at the call site.
    base_quantization: str

    lora_rank: int
    lora_alpha: int
    lora_dropout: float = 0.05
    bias: str = "none"
    learning_rate: float
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    epochs: int
    optimizer: str
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int
    effective_batch_size: int
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    target_modules: list[str]

    #: When true it is LoRA-on-ViT, never a full fine-tune: full FT risks
    #: degrading the pretrained OCR ability the image-only pathway depends on.
    vit_trainable: bool = False
    vit_method: Literal["frozen", "lora"] = "frozen"

    resolution_cap_px: int
    max_seq_len: int
    seed: int

    @model_validator(mode="after")
    def _vit_never_fully_fine_tuned(self) -> TrainingConfig:
        if self.vit_trainable and self.vit_method != "lora":
            raise ValueError(
                "vit_trainable is set but vit_method is not 'lora'. The §3 escalation is "
                "'add a ViT LoRA', never 'unfreeze and train the encoder' — full fine-tuning "
                "risks the pretrained document/OCR capability the image-only path relies on."
            )
        return self


class DataStats(_Base):
    """What the run actually trained on."""

    train_examples: int
    val_examples: int
    test_examples: int
    modality_mix: dict[str, float] = Field(default_factory=dict)

    #: Per-LoB-value share (arch §0b). Recorded so a later regression can be
    #: checked against coverage before it is blamed on the model.
    lob_coverage: dict[str, float] = Field(default_factory=dict)

    #: Per canonical field -> observed surface label -> count (arch §0c).
    alias_coverage: dict[str, dict[str, int]] = Field(default_factory=dict)

    #: Documents where a field and one of its confusables co-occur. Zero is a
    #: corpus defect: the model learns the mapping but never the boundary.
    confusable_example_count: int = 0

    tenant_ids: list[str] = Field(default_factory=list)

    #: Recorded, not enforced, while de-identification is blocked (SPEC_05 §1).
    deidentified: bool = False
    image_redaction: str = "unresolved"


class EvalMetrics(_Base):
    """Scores against the frozen golden eval set (arch §15). All optional —
    a manifest exists from the moment training completes, before evaluation."""

    field_exact_match: float | None = None
    field_normalized_match: float | None = None
    field_f1_list_fields: float | None = None
    list_field_recall: float | None = None
    schema_validity_rate: float | None = None
    ece_confidence: float | None = None
    ocr_arbitration_accuracy: float | None = None
    image_only_accuracy: float | None = None
    scanned_accuracy: float | None = None
    doc_type_classifier_accuracy: float | None = None

    #: Own metric, per value, never folded into the aggregate (arch §0b).
    lob_detection_accuracy: float | None = None
    lob_accuracy_by_value: dict[str, float] = Field(default_factory=dict)

    #: Reported, not gating — rare aliases have too little support to gate on.
    alias_accuracy: dict[str, dict[str, float]] = Field(default_factory=dict)

    #: GATING. Systematic rather than random: it means the model has collapsed
    #: two distinct entities, and the output is fluent enough to pass every
    #: structural check (arch §0c).
    confusable_misattribution_rate: float | None = None

    latency_ms_per_doc: float | None = None


class Artifacts(_Base):
    """Where the outputs live, and whether they are durable yet."""

    #: ``staged`` = on the RunPod volume, no durability guarantee.
    #: ``published`` = in Blob. ``package`` flips this.
    status: ArtifactStatus = "staged"
    staging_path: str | None = None

    adapter_weights: str | None = None
    merged_model: str | None = None
    quantized_model: str | None = None
    quantized_formats: list[str] = Field(default_factory=list)
    eval_report: str | None = None
    calibration_params: str | None = None

    @model_validator(mode="after")
    def _published_artifacts_have_a_path(self) -> Artifacts:
        if self.status == "published" and not (self.adapter_weights or self.merged_model):
            raise ValueError(
                "artifacts.status is 'published' but no Blob path is recorded. A published "
                "manifest that points nowhere is worse than a staged one — it claims durability "
                "the artifact does not have."
            )
        return self


class Promotion(_Base):
    """The gate decision. A candidate is promoted only if it matches or beats the
    current production version on **every** gating metric — no override path."""

    gated_against: str | None = None
    beat_previous_on_all_gates: bool | None = None
    failed_gates: list[str] = Field(default_factory=list)
    promoted_by: str | None = None
    promoted_at: datetime | None = None


class RunManifest(_Base):
    """One training run, completely described."""

    run_id: str
    run_type: RunType
    doc_type: str | None = None
    tenant_id: str | None = None
    status: RunStatus = "trained"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    #: Sweep runs are first-class registry entries, not untracked side
    #: experiments (arch §11a). Sweeps are deferred until after the pilot; the
    #: fields exist now so nothing changes when they are turned on.
    is_sweep_run: bool = False
    sweep_id: str | None = None
    sweep_phase: SweepPhase | None = None

    #: Set when a Foundation run continued from a previous checkpoint rather than
    #: retraining from base. The gate then demands cross-type regression evidence
    #: before promotion, because continued training compounds drift (arch §12).
    continued_from: str | None = None

    dependencies: Dependencies
    training_config: TrainingConfig
    data_stats: DataStats
    eval_metrics: EvalMetrics = Field(default_factory=EvalMetrics)
    artifacts: Artifacts = Field(default_factory=Artifacts)
    promotion: Promotion = Field(default_factory=Promotion)

    @model_validator(mode="after")
    def _consistency(self) -> RunManifest:
        if self.run_type == "per_type_adapter":
            if not self.doc_type:
                raise ValueError("a per_type_adapter run must name its doc_type")
            if not self.dependencies.foundation_version:
                raise ValueError(
                    "a per_type_adapter must record the foundation_version it was trained on. "
                    "Without it, 'which adapters does this Foundation bump invalidate?' becomes "
                    "a manual audit instead of a query (arch §12)."
                )
        if self.run_type == "foundation" and self.doc_type:
            raise ValueError("a Foundation run spans all doc types and must not name one")
        if self.is_sweep_run and not self.sweep_id:
            raise ValueError("a sweep run must carry its sweep_id to be groupable")
        if self.status == "promoted" and not self.promotion.promoted_at:
            raise ValueError("a promoted run must record when, and by whom, it was promoted")
        return self

    def index_row(self) -> dict[str, Any]:
        """The flat row for ``registry_index.json`` — the whole history at a glance."""
        return {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "doc_type": self.doc_type,
            "status": self.status,
            "artifact_status": self.artifacts.status,
            "created_at": self.created_at.isoformat(),
            "corpus_version": self.dependencies.corpus_version,
            "foundation_version": self.dependencies.foundation_version,
            "code_git_commit": self.dependencies.code_git_commit,
            "is_sweep_run": self.is_sweep_run,
            "field_exact_match": self.eval_metrics.field_exact_match,
            "list_field_recall": self.eval_metrics.list_field_recall,
            "lob_detection_accuracy": self.eval_metrics.lob_detection_accuracy,
            "confusable_misattribution_rate": self.eval_metrics.confusable_misattribution_rate,
        }
