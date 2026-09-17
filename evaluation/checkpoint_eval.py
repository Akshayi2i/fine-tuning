"""Checkpoint selection by generated field F1 (arch v2.1 §11.2).

**Validation loss does not select what ships.** It is logged, and it drives early
stopping, but it is a poor proxy for extraction accuracy: loss is averaged over
every token the model produces, so it is dominated by the easy copy tokens — the
schema keys, the JSON punctuation, the boilerplate — that the model gets right
from the first hundred steps. A checkpoint can improve on loss while getting
worse at the values, which is the only thing the gate reads.

Field F1 needs **generation**, and the training loop's own eval does not generate
efficiently for a VLM. So selection is a separate job, run after training:

    last 3 checkpoints + the best-loss checkpoint
        -> generate on the validation split through vLLM
        -> score field F1 with the same code the gate uses
        -> the winner is what gets merged

**Checkpoints are loaded as a decoder LoRA, not merged.** vLLM applies one LoRA
per request, which is exactly enough for this — a per-checkpoint merge would cost
~16GB of disk and several minutes each, to answer a question that does not need
the weights folded in.

The candidate set is small on purpose. Scoring every checkpoint would multiply
the cost by the epoch count for a decision that, in practice, sits between the
last few and the one early stopping liked.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

#: How many trailing checkpoints to score, alongside the best-loss one.
TRAILING_CANDIDATES = 3

_CHECKPOINT_STEP = re.compile(r"checkpoint-(\d+)$")


class CheckpointEvalError(RuntimeError):
    """Raised when a checkpoint cannot be selected safely."""


class Scorer(Protocol):
    """Generates on the validation split and returns that checkpoint's metrics.

    Injected rather than imported so this module is testable without a GPU, and
    so the scorer is the **same** one the gate uses — a selector scoring by a
    different definition of "correct" picks a checkpoint the gate then rejects.
    """

    def __call__(self, checkpoint: str) -> dict[str, float]: ...


@dataclass
class CheckpointScore:
    """One candidate's result."""

    checkpoint: str
    step: int
    metrics: dict[str, float] = field(default_factory=dict)
    is_best_loss: bool = False

    @property
    def field_f1(self) -> float:
        """The selection metric — the gate's primary (arch v2.1 §15.2)."""
        return float(self.metrics.get("field_normalized_match", 0.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "step": self.step,
            "field_f1": round(self.field_f1, 4),
            "is_best_loss": self.is_best_loss,
            "metrics": {k: round(v, 4) for k, v in sorted(self.metrics.items())},
        }


@dataclass
class SelectionReport:
    """Which checkpoint won, against what it was compared, and by how much."""

    selected: str | None = None
    scores: list[CheckpointScore] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def best_loss_checkpoint(self) -> str | None:
        return next((s.checkpoint for s in self.scores if s.is_best_loss), None)

    @property
    def margin(self) -> float:
        """Field F1 of the winner minus the runner-up.

        Worth recording rather than just the winner: a margin inside ordinary
        eval noise means the selection was close to arbitrary, and a later
        regression is better read against that than against a bare filename.
        """
        ranked = sorted((s.field_f1 for s in self.scores), reverse=True)
        return round(ranked[0] - ranked[1], 4) if len(ranked) > 1 else 0.0

    @property
    def loss_and_f1_disagreed(self) -> bool:
        """Whether the best-loss checkpoint is NOT the one with the best F1.

        The reason this job exists. When it fires, validation loss would have
        shipped a different model — so it is logged loudly rather than buried.
        """
        best = self.best_loss_checkpoint
        return bool(best and self.selected and best != self.selected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "selection_metric": "field_normalized_match",
            "margin_over_runner_up": self.margin,
            "best_loss_checkpoint": self.best_loss_checkpoint,
            "loss_and_f1_disagreed": self.loss_and_f1_disagreed,
            "candidates": [s.as_dict() for s in self.scores],
            "skipped": [{"checkpoint": c, "reason": r} for c, r in self.skipped],
        }


def checkpoint_step(path: str) -> int:
    """The global step from a ``checkpoint-{n}`` directory name.

    Raises rather than returning 0 on an unparseable name: ordering the
    candidates by step is what "the last three" means, and a silent 0 would put
    a real checkpoint at the front of the list.
    """
    match = _CHECKPOINT_STEP.search(str(path).rstrip("/"))
    if not match:
        raise CheckpointEvalError(
            f"cannot read a step number from {path!r}; expected a directory ending "
            "'checkpoint-{n}'. Ordering by step is what selects the last three."
        )
    return int(match.group(1))


def select_candidates(
    checkpoints: list[str],
    best_loss: str | None = None,
    *,
    trailing: int = TRAILING_CANDIDATES,
) -> list[CheckpointScore]:
    """The candidate set: the last ``trailing`` by step, plus the best-loss one.

    The best-loss checkpoint is included even when it is not among the last few,
    because that is precisely the interesting case — early stopping liked it and
    training continued past it.
    """
    if not checkpoints:
        raise CheckpointEvalError(
            "no checkpoints to select from. Training saves on `save_steps`; a run with none "
            "either crashed before its first save or had save_strategy disabled."
        )

    ordered = sorted({c.rstrip("/") for c in checkpoints}, key=checkpoint_step)
    chosen = ordered[-trailing:]
    if best_loss:
        best_loss = best_loss.rstrip("/")
        if best_loss not in chosen:
            chosen = [best_loss, *chosen]

    return [
        CheckpointScore(
            checkpoint=path,
            step=checkpoint_step(path),
            is_best_loss=bool(best_loss) and path == best_loss,
        )
        for path in sorted(set(chosen), key=checkpoint_step)
    ]


def select_best(
    checkpoints: list[str],
    scorer: Scorer,
    *,
    best_loss: str | None = None,
    trailing: int = TRAILING_CANDIDATES,
) -> SelectionReport:
    """Score every candidate and return the one with the best field F1.

    A checkpoint whose scoring raises is **skipped and recorded**, not treated as
    a zero: a scorer that fell over on one candidate says nothing about that
    candidate's quality, and scoring it zero would silently remove it from
    contention.
    """
    report = SelectionReport()
    for candidate in select_candidates(checkpoints, best_loss, trailing=trailing):
        try:
            candidate.metrics = dict(scorer(candidate.checkpoint))
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not lose the rest
            report.skipped.append((candidate.checkpoint, f"{type(exc).__name__}: {exc}"))
            log.warning("could not score %s: %s", candidate.checkpoint, exc)
            continue
        report.scores.append(candidate)

    if not report.scores:
        raise CheckpointEvalError(
            "no checkpoint could be scored, so none can be selected. Merging an arbitrary one "
            f"would ship a model nobody measured. Skipped: {report.skipped}"
        )

    # Ties break toward the LATER step: it has seen more data, and picking the
    # earlier one on a tie would quietly prefer an under-trained checkpoint
    # whenever the metric saturates.
    winner = max(report.scores, key=lambda s: (s.field_f1, s.step))
    report.selected = winner.checkpoint

    if report.loss_and_f1_disagreed:
        log.warning(
            "validation loss and field F1 disagree: loss preferred %s, field F1 selected %s "
            "(margin %.4f). This is why selection does not read loss — it is averaged over "
            "every token, so easy copy tokens dominate it (arch v2.1 §11.2).",
            report.best_loss_checkpoint, report.selected, report.margin,
        )
    if report.margin and report.margin < 0.005:
        log.warning(
            "the selected checkpoint beat the runner-up by only %.4f field F1, which is inside "
            "ordinary eval noise — treat this selection as close to arbitrary.", report.margin,
        )
    log.info("selected %s from %d candidates", report.selected, len(report.scores))
    return report


def generation_scorer(rows: list[dict[str, Any]], model: Any) -> Scorer:
    """Score a checkpoint by generating the validation rows with it as a LoRA.

    Scored through :func:`evaluation.validation_generation.score_generations`,
    which is the gate's own ``build_report``. A selector scoring by a different
    definition of "correct" picks a checkpoint the gate rejects.
    """
    from evaluation.validation_generation import generate_validation, score_generations

    if not rows:
        raise CheckpointEvalError(
            "the validation split is empty, so no checkpoint can be scored. Selecting by "
            "nothing would ship an arbitrary checkpoint."
        )

    def score(checkpoint: str) -> dict[str, float]:
        generations = generate_validation(rows, model, adapter=checkpoint)
        return {
            k: float(v)
            for k, v in score_generations(generations, model_version=checkpoint).items()
            if isinstance(v, (int, float))
        }

    return score


def vllm_scorer(*, client: Any, val_path: str, model: Any = None) -> Scorer:
    """A :func:`generation_scorer` over the stored validation split, on the base.

    The base is loaded in bf16 with LoRA enabled and each checkpoint applied as a
    decoder LoRA: one LoRA per request is exactly what vLLM supports, so no
    checkpoint needs merging to be scored.
    """
    from evaluation.validation_generation import read_rows

    if model is None:  # pragma: no cover - needs a GPU
        from inference_core.model_runner import load_model

        model = load_model("base", client)
    return generation_scorer(read_rows(client.read_text(val_path)), model)


def discover_checkpoints(output_dir: str) -> tuple[list[str], str | None]:
    """The checkpoints a finished run left on disk, and the best-loss one.

    ms-swift writes under a versioned subdirectory (``output_dir/v0-<timestamp>/``),
    so the search is recursive; when a directory holds several runs, the most
    recently written one is this run. The best-loss checkpoint comes from the HF
    Trainer's ``trainer_state.json``, which is where early stopping recorded it.

    Returns ``([], None)`` when the directory does not exist — a dry run, or a
    trainer that saved nothing, which checkpoint selection then reports.
    """
    import json
    from pathlib import Path

    root = Path(output_dir)
    if not root.is_dir():
        return [], None
    found = [p for p in root.rglob("checkpoint-*") if p.is_dir() and _CHECKPOINT_STEP.search(p.name)]
    if not found:
        return [], None

    latest_run = max({p.parent for p in found}, key=lambda d: d.stat().st_mtime)
    checkpoints = sorted(
        (str(p) for p in found if p.parent == latest_run), key=checkpoint_step
    )

    best_loss = None
    state = Path(checkpoints[-1]) / "trainer_state.json"
    if state.exists():
        recorded = json.loads(state.read_text(encoding="utf-8")).get("best_model_checkpoint")
        if recorded:
            best_loss = str(latest_run / Path(recorded).name)
    return checkpoints, best_loss
