"""How well a policy's line of business is read when no caller supplied it.

Scored per line, with a confusion matrix, because the errors that matter are
specific: a homeowners policy read as dwelling fire still reaches the right
adapter (one family) with the wrong schema; a personal auto policy read as
commercial auto reaches the wrong adapter altogether. An overall accuracy hides
both behind the volume of the easy lines.

The gate's floor is ``lob_detection_accuracy`` (evaluation.gating, 0.85); a
line with enough documents is also held to the floor on its own
(:meth:`LobDetectionReport.below_floor`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

#: A line is held to the floor on its own once it has this many documents.
PER_LINE_MIN_SUPPORT = 20


@dataclass
class LobDetectionReport:
    #: (expected line, detected line or None) -> documents.
    confusion: Counter = field(default_factory=Counter)

    @property
    def scored(self) -> int:
        return sum(self.confusion.values())

    @property
    def overall(self) -> float | None:
        right = sum(n for (expected, got), n in self.confusion.items() if expected == got)
        return right / self.scored if self.scored else None

    def by_line(self) -> dict[str, dict[str, float | int]]:
        """Per expected line: documents, accuracy, and how many reached the
        right layout family even when the line was wrong."""
        from common.config import lob_to_layout_family

        family = lob_to_layout_family()
        out: dict[str, dict[str, float | int]] = {}
        for line in sorted({expected for expected, _ in self.confusion}):
            rows = {got: n for (expected, got), n in self.confusion.items() if expected == line}
            total = sum(rows.values())
            right = rows.get(line, 0)
            same_family = sum(n for got, n in rows.items()
                              if got is not None and family.get(got) == family.get(line))
            out[line] = {"documents": total, "accuracy": round(right / total, 4),
                         "family_accuracy": round(same_family / total, 4),
                         "undetected": rows.get(None, 0)}
        return out

    def below_floor(self, floor: float, *, min_support: int = PER_LINE_MIN_SUPPORT) -> list[str]:
        return [line for line, row in self.by_line().items()
                if row["documents"] >= min_support and row["accuracy"] < floor]


def score_lob_detection(pairs: Iterable[tuple[str, str | None]]) -> LobDetectionReport:
    """``pairs``: (the line the document is, the line the classifier read)."""
    from serving.doc_type_classifier import known_line

    report = LobDetectionReport()
    for expected, detected in pairs:
        truth = known_line(expected)
        if truth is None:
            continue
        report.confusion[(truth, known_line(detected) if detected else None)] += 1
    return report
