"""Experiment B — the 5-document smoke test (IMPL-15 §2, arch §16b).

Annotate exactly 5 documents per type and train to **intentional overfit**: no
train/val split, 5 epochs, learning rate at the top of the sweep range. The model
is supposed to memorise these documents. That is the design.

**This proves the code pipeline is correct — not that the architecture
generalises.** A model that reproduces its own training set has demonstrated that
the collator masked the right tokens, the adapter saved and loaded, and the
manifest was written. It has demonstrated nothing about unseen documents, and
reading it as a proof of concept is how a pipeline bug survives to Experiment C.

**A failure here is a code bug, not an architecture problem.** Fix it before
spending annotation budget.

Why five documents and not one: a single source expands into three modality
variants, which does not exercise multi-page handling, an OCR-failure page, or
the image-only path. Five per type covers those.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PILOT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PILOT_ROOT / "reports"

SMOKE_DOCS_PER_TYPE = 5

#: Overfit targets. Loss must fall below this *by* the deadline epoch — a model
#: that cannot memorise five documents in two epochs has something wrong with it
#: upstream of any architectural question.
MAX_LOSS = 0.05
LOSS_EPOCH_DEADLINE = 2
MIN_TRAIN_F1 = 0.95

#: Every component the run must exercise. Named individually because "training
#: finished" is not the same claim as "the adapter reloaded and served" — and the
#: second one is where the interesting failures live.
PIPELINE_COMPONENTS: tuple[str, ...] = (
    "ms_swift_training_loop",
    "data_collator",
    "lora_adapter_loading",
    "adapter_save",
    "adapter_push_to_blob",
    "vllm_multi_adapter_hot_swap",
    "run_manifest_generation",
)


class SmokeTestError(RuntimeError):
    """Raised when the smoke test cannot be run as specified."""


def overfit_config() -> dict[str, Any]:
    """The deliberately-wrong-for-production training config this experiment uses.

    Every value here would be a mistake in a real run. They are correct for a
    memorisation check, and naming them in one place keeps them from leaking into
    a production config by copy-paste.
    """
    return {
        "num_train_epochs": 5,
        # No validation split: with five documents there is nothing to hold out
        # that would mean anything, and early stopping on it would fight the
        # memorisation this experiment is trying to achieve.
        "val_split": 0.0,
        "early_stopping": False,
        # Top of the sweep range (configs/training/foundation.yaml: 1e-4 to 2e-4).
        "learning_rate": 2.0e-4,
        "warmup_ratio": 0.0,
        "seed": 42,
    }


@dataclass
class ComponentResult:
    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"component": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class SmokeReport:
    """Experiment B's output: a pass/fail per component, plus the two numbers."""

    documents_per_type: dict[str, int] = field(default_factory=dict)
    loss_curve: list[float] = field(default_factory=list)
    train_f1: float | None = None
    components: list[ComponentResult] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def loss_target_met(self) -> bool:
        """Loss below ``MAX_LOSS`` at or before the deadline epoch.

        A run that crosses the threshold in epoch one has met it — converging
        sooner than the deadline is the good case, not an incomplete measurement.
        A short curve that never crossed simply fails, which is the same answer
        as a long one that never crossed.
        """
        return any(loss < MAX_LOSS for loss in self.loss_curve[:LOSS_EPOCH_DEADLINE])

    @property
    def f1_target_met(self) -> bool:
        return self.train_f1 is not None and self.train_f1 > MIN_TRAIN_F1

    @property
    def components_passed(self) -> bool:
        covered = {c.name for c in self.components}
        missing = set(PIPELINE_COMPONENTS) - covered
        if missing:
            return False
        return all(c.passed for c in self.components)

    @property
    def untested_components(self) -> list[str]:
        """Components no result was recorded for.

        Absent is not passing. A component nobody exercised is exactly the one
        that breaks in Experiment C.
        """
        return sorted(set(PIPELINE_COMPONENTS) - {c.name for c in self.components})

    @property
    def passed(self) -> bool:
        return self.loss_target_met and self.f1_target_met and self.components_passed

    def failures(self) -> list[str]:
        reasons: list[str] = []
        if not self.loss_target_met:
            # Sliced to the window the message names. `min` over the whole curve
            # printed a value from epoch 4 as 'best in that window' — evidence
            # that contradicted the verdict it was supporting.
            window = self.loss_curve[:LOSS_EPOCH_DEADLINE]
            best = min(window) if window else None
            reasons.append(
                f"training loss did not fall below {MAX_LOSS} within {LOSS_EPOCH_DEADLINE} epoch(s) "
                f"(best in that window: {best}). The model cannot memorise five documents, so "
                "look at the collator masking and the learning rate before anything else."
            )
        if not self.f1_target_met:
            reasons.append(
                f"train-set field F1 {self.train_f1} is not above {MIN_TRAIN_F1}. The model is "
                "being evaluated on documents it trained on; anything short of near-perfect means "
                "the training signal is not reaching the assistant tokens."
            )
        for component in self.components:
            if not component.passed:
                reasons.append(f"{component.name}: {component.detail or 'failed'}")
        for name in self.untested_components:
            reasons.append(
                f"{name}: not exercised. A component with no result has not passed — it is the "
                "one that will break in Experiment C."
            )
        return reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": "B_smoke_test",
            "generated_at": self.generated_at,
            "documents_per_type": dict(sorted(self.documents_per_type.items())),
            "config": overfit_config(),
            "loss_curve": self.loss_curve,
            "loss_target_met": self.loss_target_met,
            "train_field_f1": self.train_f1,
            "f1_target_met": self.f1_target_met,
            "components": [c.as_dict() for c in self.components],
            "untested_components": self.untested_components,
            "passed": self.passed,
            "failures": self.failures(),
            "interpretation": (
                "This proves the code pipeline runs end to end. It says nothing about "
                "generalisation — the model was evaluated on the documents it trained on."
            ),
        }


def check_document_counts(counts: dict[str, int]) -> list[ComponentResult]:
    """Verify the experiment was run at the specified size."""
    results = []
    for doc_type, count in sorted(counts.items()):
        results.append(ComponentResult(
            name=f"corpus_size:{doc_type}",
            passed=count >= SMOKE_DOCS_PER_TYPE,
            detail=(
                f"{count} document(s); {SMOKE_DOCS_PER_TYPE} are needed to cover multi-page, "
                "OCR-failure and image-only cases"
            ) if count < SMOKE_DOCS_PER_TYPE else f"{count} documents",
        ))
    return results


def evaluate_smoke(
    *,
    documents_per_type: dict[str, int],
    loss_curve: Sequence[float],
    train_f1: float | None,
    components: Sequence[ComponentResult] = (),
) -> SmokeReport:
    """Assemble the report from a completed smoke run."""
    report = SmokeReport(
        documents_per_type=dict(documents_per_type),
        loss_curve=list(loss_curve),
        train_f1=train_f1,
        components=list(components) + check_document_counts(documents_per_type),
    )
    log.info("smoke test %s", "PASSED" if report.passed else "FAILED")
    for reason in report.failures():
        log.error("  %s", reason)
    return report


def write_report(report: SmokeReport, root: Path = REPORTS_DIR) -> Path:
    path = root / "smoke_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def render(report: SmokeReport) -> str:
    lines = [
        "Experiment B — smoke test (intentional overfit, 5 docs/type)",
        f"  loss curve: {[round(v, 4) for v in report.loss_curve]} "
        f"(target < {MAX_LOSS} by epoch {LOSS_EPOCH_DEADLINE}: "
        f"{'met' if report.loss_target_met else 'NOT met'})",
        f"  train F1:   {report.train_f1} (target > {MIN_TRAIN_F1}: "
        f"{'met' if report.f1_target_met else 'NOT met'})",
        "  components:",
    ]
    lines += [f"    [{'ok  ' if c.passed else 'FAIL'}] {c.name}" for c in report.components]
    lines += [f"    [none] {name} — not exercised" for name in report.untested_components]
    lines.append(f"  -> {'PASS' if report.passed else 'FAIL'}")
    if not report.passed:
        lines.append("     A failure here is a code bug, not an architecture problem. "
                     "Fix it before spending annotation budget.")
    return "\n".join(lines)
