"""Early stopping (arch v2.1 §11.1, §11.2).

**Inside the trainer, early stopping reads validation loss.** Field F1 needs
generation, which the training loop's eval does not do efficiently for a VLM, so
`swift_early_stopping_args` is configured from `unified.yaml` with
`metric_for_best_model: eval_loss`, `greater_is_better: false`. Patience is
counted in evaluations, not epochs: at pilot volume an epoch is a handful of
steps.

**Loss never selects what ships.** It can keep improving while field extraction
gets worse — it rewards fluent JSON, and a confidently wrong value is fluent. The
checkpoint that is merged is chosen afterwards by generated field F1
(`evaluation.checkpoint_eval`). `EarlyStoppingState` is the field-F1 tracker for
that offline setting, and warns when the two signals diverge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DEFAULT_PATIENCE = 3


@dataclass
class EarlyStoppingState:
    """Tracks the best score seen and how long since it improved."""

    patience: int = DEFAULT_PATIENCE
    min_delta: float = 0.0
    best_score: float | None = None
    best_step: int = 0
    evaluations_without_improvement: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def should_stop(self) -> bool:
        return self.evaluations_without_improvement >= self.patience

    def update(self, step: int, field_f1: float, eval_loss: float | None = None) -> bool:
        """Record one evaluation. Returns True when training should stop."""
        # `or 0.0` recorded a loss of zero — a perfect model — for an evaluation
        # that reported no loss at all, and that 0.0 then read as the best value
        # in the history anyone later plots. Absent stays absent.
        entry: dict[str, float] = {"step": step, "field_f1": field_f1}
        if eval_loss is not None:
            entry["eval_loss"] = eval_loss
        self.history.append(entry)

        improved = self.best_score is None or field_f1 > self.best_score + self.min_delta
        if improved:
            self.best_score, self.best_step = field_f1, step
            self.evaluations_without_improvement = 0
        else:
            self.evaluations_without_improvement += 1

        # Worth surfacing: it means the model is getting more fluent while
        # getting the values more wrong, which loss alone would call progress.
        if len(self.history) >= 2 and eval_loss is not None:
            previous = self.history[-2]
            # `.get`, because an evaluation that reported no loss now records
            # none rather than a fictitious 0.0 — and a missing previous loss
            # means there is no divergence to detect, not a divergence from zero.
            previous_loss = previous.get("eval_loss")
            if (previous_loss is not None
                    and eval_loss < previous_loss and field_f1 < previous["field_f1"]):
                log.warning(
                    "step %d: eval_loss improved (%.4f -> %.4f) while field F1 FELL "
                    "(%.4f -> %.4f). The model is becoming more fluent and less correct; "
                    "loss alone would read this as progress.",
                    step, previous["eval_loss"], eval_loss, previous["field_f1"], field_f1,
                )

        if self.should_stop:
            log.info(
                "early stopping at step %d: no field-F1 improvement in %d evaluations "
                "(best %.4f at step %d)",
                step, self.evaluations_without_improvement, self.best_score, self.best_step,
            )
        return self.should_stop


def swift_early_stopping_args(
    patience: int = DEFAULT_PATIENCE,
    *,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    load_best_model_at_end: bool = True,
) -> dict[str, object]:
    """Early-stopping arguments for the ms-swift invocation.

    ms-swift wires these into the TRL/HF callback system; this is configuration,
    not a reimplementation of the callback (arch §10).

    Every value is a parameter rather than a literal. `greater_is_better` was
    hardcoded True while all four training YAMLs carried the key and no Python
    read it, so a config selecting on `eval_loss` — where lower is better — made
    HF restore the checkpoint with the *highest* loss, and that worst-of-run
    adapter is what got staged, evaluated and offered to the promotion gate.
    """
    return {
        "early_stopping_patience": patience,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "load_best_model_at_end": load_best_model_at_end,
    }
