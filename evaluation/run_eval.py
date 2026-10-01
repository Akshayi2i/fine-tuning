"""The evaluation driver (SPEC_08 §3, arch §8, §15).

Runs a model version over the **frozen golden eval set** through the SPEC_07
inference core, computes every metric, and writes
``eval-reports/v{n}/{doc_type}/report.json`` plus a top-level summary — broken
down per doc type *and* per modality mode.

Four subsets are scored explicitly on top of the full set: ``image_only``,
``scanned``, ``noisy_ocr`` and ``long_policy``. An aggregate hides exactly the
regressions that matter — image-only accuracy can fall ten points while the
overall number moves two, because image-only is a third of the corpus.

**The eval set is frozen and versioned separately**, human double-verified, and
held constant across corpus versions so model versions compare like with like.
:func:`assert_eval_set_disjoint` enforces the part of that which code can:
**no eval ``source_id`` may appear in any corpus split.** Train on your eval set
and every number in the registry becomes a measurement of memorisation, with no
symptom anywhere — the loss curve looks fine and the gate passes.

Per-document **error records** are kept, not just aggregates, because
``vit_gate`` (SPEC_06) needs the perception-vs-reasoning split and failure-mode
analysis needs real material.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.constants import ACTIVE_DOC_TYPES

log = logging.getLogger(__name__)

#: Eval subsets scored separately from the full set (SPEC_08 §3).
EVAL_SUBSETS: tuple[str, ...] = ("image_only", "scanned", "noisy_ocr", "long_policy")


class EvalError(RuntimeError):
    """Raised when an evaluation cannot be trusted or cannot be run."""


class EvalSetLeakage(EvalError):
    """Raised when an eval document also appears in a corpus split.

    Separate from :class:`EvalError` because it invalidates results that already
    exist rather than blocking a run: every metric computed against a leaked eval
    set has to be discarded, not re-run.
    """


# --------------------------------------------------------------------------
# The freeze guarantee
# --------------------------------------------------------------------------


def corpus_source_ids(
    client: BlobClient, corpus_version: str, tenant_id: str | None = None,
    *, include_test: bool = False,
) -> set[str]:
    """Every ``source_id`` a model built from this corpus LEARNED from or was
    SELECTED on — its train and val splits (and their scope views).

    The test split is excluded unless ``include_test``: no model trains or
    selects on it, and it is exactly what the golden eval set is frozen FROM
    (``freeze-eval-set``). Counting it made the natural sequence — build v1,
    freeze v1's test split, gate the model trained on v1 — fail as "leakage"
    for every frozen document. The group split keeps a frozen document's
    family out of train and val, so this is not a way around the guarantee.
    """
    found: set[str] = set()
    prefix = paths.corpus_dir(corpus_version, tenant_id)
    for key in client.list(prefix):
        if not key.endswith(".jsonl"):
            continue
        parts = key[len(prefix):].strip("/").split("/")
        if not include_test and ("test" in parts[:-1] or parts[-1].startswith("test")):
            continue
        for line in client.read_text(key).splitlines():
            if not line.strip():
                continue
            try:
                found.add(json.loads(line)["source_id"])
            except (ValueError, KeyError):
                continue
    return found


def eval_set_source_ids(client: BlobClient) -> set[str]:
    """Every ``source_id`` in the frozen golden eval set."""
    prefix = paths.golden_eval_set_dir()
    return {
        key[len(prefix):].strip("/").split("/")[0]
        for key in client.list(prefix)
        if key.endswith("golden.json")
    }


def assert_eval_set_disjoint(
    client: BlobClient, corpus_version: str, tenant_id: str | None = None
) -> None:
    """Refuse to evaluate against a corpus the eval set overlaps.

    This is the assertion that makes "never train on the eval set" a fact rather
    than an intention. Without it the failure is completely silent: training
    succeeds, the loss curve looks healthy, every metric improves, the gate
    passes, and the numbers describe memorisation.
    """
    overlap = eval_set_source_ids(client) & corpus_source_ids(client, corpus_version, tenant_id)
    if overlap:
        raise EvalSetLeakage(
            f"{len(overlap)} document(s) are in BOTH the frozen golden eval set and corpus "
            f"{corpus_version}: {sorted(overlap)[:10]}. Every metric measured against this eval "
            "set is a measurement of memorisation, and nothing else in the pipeline would show it "
            "— the loss curve looks healthy and the gate passes. Remove them from the corpus (the "
            "eval set is the thing held constant across versions, so it is the corpus that "
            "changes) and rebuild (arch §8)."
        )
    log.info("eval set is disjoint from corpus %s", corpus_version)


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


@dataclass
class SubsetReport:
    """Metrics for one doc type × one subset."""

    doc_type: str
    subset: str
    documents: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error_records: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type,
            "subset": self.subset,
            "documents": self.documents,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.metrics.items()},
            "error_records": self.error_records,
        }


@dataclass
class EvalReport:
    """One model version's evaluation, per doc type and per subset."""

    model_version: str
    corpus_version: str = ""
    subsets: list[SubsetReport] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    #: Filled in on a later pass: classifier accuracy needs the SPEC_11
    #: classifier, and eval must stay runnable before serving is built.
    classifier_scored: bool = False

    def for_doc_type(self, doc_type: str) -> list[SubsetReport]:
        return [s for s in self.subsets if s.doc_type == doc_type]

    def full_set(self) -> list[SubsetReport]:
        return [s for s in self.subsets if s.subset == "full"]

    def gate_metrics(self) -> dict[str, Any]:
        """The flat metric dict the promotion gate reads.

        Document-weighted across doc types, so a type with three eval documents
        does not carry the same weight as one with thirty.
        """
        totals: dict[str, float] = {}
        weights: dict[str, int] = {}
        for report in self.full_set():
            for name, value in report.metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[name] = totals.get(name, 0.0) + float(value) * report.documents
                    weights[name] = weights.get(name, 0) + report.documents

        metrics = {
            name: totals[name] / weights[name]
            for name in totals
            if weights.get(name)
        }

        # Subset accuracies are gating metrics in their own right (arch §15):
        # an aggregate that averages them away is how an image-only regression
        # ships behind a two-point overall move.
        for subset, gate_name in (
            ("image_only", "image_only_accuracy"),
            ("scanned", "scanned_accuracy"),
            ("noisy_ocr", "ocr_arbitration_accuracy"),
        ):
            scores: list[tuple[float, int]] = [
                (value, s.documents)
                for s in self.subsets
                if s.subset == subset
                and isinstance(value := s.metrics.get("field_normalized_match"), float)
            ]
            total_docs = sum(n for _v, n in scores)
            if total_docs:
                metrics[gate_name] = sum(v * n for v, n in scores) / total_docs

        if not self.classifier_scored:
            # Deliberately absent rather than zero: the gate treats a missing
            # metric as "not passed", which is the right answer for one nobody
            # measured. A zero would read as a measured catastrophe.
            metrics.pop("doc_type_classifier_accuracy", None)
        return metrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "corpus_version": self.corpus_version,
            "generated_at": self.generated_at,
            "classifier_scored": self.classifier_scored,
            "subsets_scored": sorted({s.subset for s in self.subsets}),
            "by_doc_type": {
                doc_type: [s.as_dict() for s in self.for_doc_type(doc_type)]
                for doc_type in sorted({s.doc_type for s in self.subsets})
            },
            "gate_metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in sorted(self.gate_metrics().items())
            },
        }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def subset_of(document: dict[str, Any]) -> list[str]:
    """Which eval subsets a document belongs to. A document can be in several."""
    subsets = ["full"]
    if document.get("modality_mode") == "image_only":
        subsets.append("image_only")
    if document.get("modality_mode") == "noisy_ocr_image":
        subsets.append("noisy_ocr")
    if document.get("is_scanned"):
        subsets.append("scanned")
    if document.get("doc_type") == "policy" and int(document.get("page_count", 1)) > 5:
        subsets.append("long_policy")
    return subsets


def _reconciliation_rate(
    scored: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> float | None:
    """Share of VERIFIABLE Loss Runs whose claims reconciled (arch v2.1 §5.5).

    Read from the serving path's own reconciliation report rather than recomputed
    here, so the number the gate reads is the number production produced — a
    second implementation would eventually disagree with the first, and the gate
    would be scoring something serving does not do.

    ``None`` when no document carried one: a subset with no Loss Runs has nothing
    to say about reconciliation, and emitting 0.0 would report a total failure of
    a check that never ran.
    """
    reports = [
        metadata["reconciliation"] for _, _, metadata in scored
        if isinstance(metadata.get("reconciliation"), dict)
    ]
    if not reports:
        return None
    verifiable = [r for r in reports if r.get("status") != "unverifiable"]
    if not verifiable:
        # Every Loss Run printed no totals. Unverifiable is not failure, but it
        # is not evidence either — reported as None so the gate does not read an
        # absence of evidence as a score.
        log.warning(
            "%d Loss Run(s) scored and none printed totals, so row completeness is "
            "unverifiable across the whole subset (arch v2.1 §5.5).", len(reports),
        )
        return None
    return sum(1 for r in verifiable if r.get("status") == "reconciled") / len(verifiable)


def score_subset(
    doc_type: str,
    subset: str,
    scored: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> SubsetReport:
    """Score one doc type × subset from ``(expected, got, metadata)`` triples."""
    from evaluation.metrics.auto_accept import AutoAcceptTally, score_auto_accept
    from evaluation.metrics.confusable import aggregate_misattribution, score_misattribution
    from evaluation.metrics.coverage_metrics import (
        expected_calibration_error,
        score_lob,
        score_schema_validity,
    )
    from evaluation.metrics.extraction_faults import (
        _is_empty,
        score_false_nulls,
        score_hallucinations,
        score_page_selection,
    )
    from evaluation.metrics.field_accuracy import score_all_list_fields, score_fields
    from training.vit_gate import classify_error

    report = SubsetReport(doc_type=doc_type, subset=subset, documents=len(scored))
    if not scored:
        return report

    accuracies, exacts, recalls, f1s, list_precisions = [], [], [], [], []
    # Single-value fields as a retrieval task, counted over every field of the
    # subset: precision = of the values the model wrote, the share that were
    # right; recall = of the values the label holds, the share found. Accuracy
    # alone cannot tell a model that writes little but well from one that
    # writes everything and half of it wrong.
    written = expected_filled = right_and_written = 0
    by_lob: dict[str, list[int]] = {}
    by_field: dict[str, list[int]] = {}
    # Both of these are GATING_METRICS with require_all_measured=True, and
    # neither was ever computed here — so the gate blocked every candidate that
    # ever reached it, permanently, with no override. `score_misattribution` had
    # zero production callers; ECE had none at all. Computing them is the fix:
    # dropping them from the gate would have made it pass by no longer checking
    # the two things this project exists to get right.
    misattributions = []
    # Values delivered without a review flag, and how many were wrong: what
    # the review thresholds promise, measured on documents they never saw.
    auto_accept = AutoAcceptTally()
    confidences: list[float] = []
    correctness: list[bool] = []

    # Field accuracy is pooled per DOCUMENT reading (source x modality), then
    # averaged over documents. Averaged per row, a policy read as sixty windows
    # counted sixty times and a one-page certificate once, so the gate measured
    # long policies and little else. Pooling also means a window with nothing to
    # score — boilerplate pages answered with nothing — adds nothing, rather
    # than a 0.0.
    pooled: dict[Any, list[int]] = {}
    for index, (expected, got, metadata) in enumerate(scored):
        accuracy = score_fields(expected, got)
        key = (metadata.get("source_id") or f"#{index}", metadata.get("modality_mode"))
        tally = pooled.setdefault(key, [0, 0, 0])
        tally[0] += sum(r.correct for r in accuracy.results)
        tally[1] += sum(r.exact for r in accuracy.results)
        tally[2] += accuracy.total
        line = metadata.get("lob")
        line = ", ".join(line) if isinstance(line, list) else (line or "unknown")
        lob_tally = by_lob.setdefault(str(line), [0, 0])
        lob_tally[0] += sum(r.correct for r in accuracy.results)
        lob_tally[1] += accuracy.total
        for result in accuracy.results:
            has_value, wrote = not _is_empty(result.expected), not _is_empty(result.got)
            written += wrote
            expected_filled += has_value
            right_and_written += bool(result.correct and has_value and wrote)
            field_tally = by_field.setdefault(result.field_path, [0, 0])
            field_tally[0] += bool(result.correct)
            field_tally[1] += 1

        auto_accept.add(score_auto_accept(expected, got))
        misattributions.append(
            score_misattribution(
                expected, got, doc_type, source_id=metadata.get("source_id", "")
            )
        )

        # ECE pairs each field's stated confidence with whether it was right.
        # Fields carrying no confidence are skipped rather than assumed correct
        # or assumed zero — an unmeasured field is not evidence either way.
        stated = metadata.get("field_confidence") or {}
        for result in accuracy.results:
            score = stated.get(result.field_path)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                confidences.append(float(score))
                correctness.append(bool(result.correct))

        for result in accuracy.failures():
            report.error_records.append({
                "source_id": metadata.get("source_id", ""),
                "field_path": result.field_path,
                "expected": result.expected,
                "got": result.got,
                "error_class": classify_error(result.expected, result.got, all_expected=expected),
                "modality_mode": metadata.get("modality_mode", "ocr_plus_image"),
                "is_scanned": bool(metadata.get("is_scanned")),
            })

        for list_report in score_all_list_fields(expected, got).values():
            recalls.append(list_report.recall)
            f1s.append(list_report.f1)
            if list_report.got_rows:
                list_precisions.append(list_report.precision)

    for correct, exact, total in pooled.values():
        if total:
            accuracies.append(correct / total)
            exacts.append(exact / total)

    lob = score_lob([(e, g) for e, g, _m in scored])
    misattribution = aggregate_misattribution(misattributions)

    # An ACORD document with no recorded form has no selectable schema. Counting
    # it as invalid keeps the run going and does not flatter the result; letting
    # the SchemaError escape would discard every other document's score too.
    validatable, unselectable = [], []
    for _e, got, meta in scored:
        form = meta.get("acord_form")
        if doc_type == "acord" and not form:
            unselectable.append(meta.get("source_id", ""))
        else:
            validatable.append((
                meta.get("source_id", ""), got, doc_type, form,
                meta.get("lob"), meta.get("sections"),
            ))

    validity = score_schema_validity(validatable)
    validity_total = validity.total + len(unselectable)
    validity_rate = validity.valid / validity_total if validity_total else 0.0
    if unselectable:
        log.warning(
            "%d ACORD eval document(s) record no acord_form, so no schema could be selected and "
            "they are counted invalid: %s. That is a defect in the frozen eval set — the form is "
            "what selects the schema (arch §4b).",
            len(unselectable), sorted(unselectable)[:10],
        )

    report.metrics = {
        # None when no row had anything to score — not measured, not zero.
        "field_normalized_match": sum(accuracies) / len(accuracies) if accuracies else None,
        "field_exact_match": sum(exacts) / len(exacts) if exacts else None,
        "list_field_recall": sum(recalls) / len(recalls) if recalls else None,
        # The name the gate (GATING_METRICS) and RunManifest both use. Emitting
        # `list_field_f1` meant the gate never received it: an F1 collapse did
        # not block, and once any promoted manifest carried the real name every
        # later candidate would be blocked forever as "not measured".
        "field_f1_list_fields": sum(f1s) / len(f1s) if f1s else None,
        # Of the table rows the model wrote, the share that are real rows.
        "list_field_precision": (
            sum(list_precisions) / len(list_precisions) if list_precisions else None
        ),
        "field_precision": right_and_written / written if written else None,
        "field_recall": right_and_written / expected_filled if expected_filled else None,
        "field_f1": (
            2 * right_and_written / (written + expected_filled)
            if written + expected_filled else None
        ),
        # Reported, not gated: where accuracy is won and lost.
        "field_accuracy_by_lob": {
            line: round(correct / total, 4) for line, (correct, total) in sorted(by_lob.items())
            if total
        } or None,
        # The fields with at least 3 scorings, lowest accuracy first.
        "weakest_fields": [
            {"field": path, "accuracy": round(correct / total, 4), "scored": total}
            for path, (correct, total) in sorted(
                by_field.items(), key=lambda item: (item[1][0] / item[1][1], -item[1][1])
            )
            if total >= 3
        ][:15] or None,
        "schema_validity_rate": validity_rate,
        # None when no value carried a flag (a flat extraction): not measured.
        "auto_accept_error_rate": auto_accept.rate,
        # Absent, not 0.0, when no document in the set carries a line to score.
        "lob_detection_accuracy": lob.overall if lob.scored else None,
        "lob_accuracy_by_value": lob.accuracy_by_value(),
        # A gating metric in its own right (arch §15): returning the certificate
        # holder's name for insured_name is the failure the canonical mapping
        # exists to prevent, and it is invisible in an aggregate field score.
        "confusable_misattribution_rate": misattribution.rate,
        # Deliberately absent — not zero — when no field carried a confidence.
        # Zero is the best possible ECE, so defaulting to it would report
        # perfect calibration for a run where calibration was never measured.
        "ece_confidence": (
            expected_calibration_error(confidences, correctness) if confidences else None
        ),
        # --- arch v2.1 §15.2: faults an aggregate field score cannot see ------
        #
        # A false null produces NO tokens, so §5 confidence is blind to it —
        # only this metric sees a value that was on the page and came back empty.
        "false_null_rate": score_false_nulls(
            [(expected, got) for expected, got, _ in scored],
            source_ids=[m.get("source_id", "") for _, _, m in scored],
        ).rate,
        # Scored against the TEXT OF THE PAGES THAT WERE SENT, not against the
        # golden label: checking against the label alone would call every wrong
        # value a hallucination, including an honest misread of something
        # actually printed — and those have different remedies.
        #
        # Only rows that were SENT text: against an image-only row's empty text
        # every value the model read off the image would count as invented.
        "hallucination_rate": score_hallucinations([
            (expected, got, str(metadata["ocr_text"]))
            for expected, got, metadata in scored
            if metadata.get("ocr_text")
        ]).rate if any(m.get("ocr_text") for _, _, m in scored) else None,
        # Only meaningful where routing ran. A document that sent every page has
        # no selection to score, and counting it as perfect recall would dilute
        # the metric toward 1.0 with documents that never exercised it.
        "page_selection_recall": score_page_selection([
            (
                metadata.get("source_id", ""),
                metadata.get("provenance_pages") or [],
                metadata.get("selected_pages") or [],
            )
            for _, _, metadata in scored
            if metadata.get("provenance_pages")
        ]).recall if any(m.get("provenance_pages") for _, _, m in scored) else None,
        # Emitted only where reconciliation ran — Loss Runs. gate_metrics()
        # weights by the documents that produced each metric, so a metric that
        # only one doc type can produce still reaches the gate, weighted by that
        # type's documents rather than diluted by the others.
        "lossrun_totals_reconciliation_rate": _reconciliation_rate(scored),
    }
    report.metrics = {k: v for k, v in report.metrics.items() if v is not None}
    return report


def expected_subsets(scope: Any = None) -> set[str]:
    """Which eval subsets this run should contain documents for.

    ``long_policy`` only exists where policies do, so warning about its absence
    on a Loss Run scope reports a gap that cannot be filled. The modality subsets
    are NOT scope-dependent: §6 requires every corpus to cover all three regimes,
    so their absence is a real defect whatever the scope covers.
    """
    if scope is None:
        return set(EVAL_SUBSETS)
    covered = set(EVAL_SUBSETS)
    if "policy" not in getattr(scope, "doc_types", ()):
        covered.discard("long_policy")
    return covered


def build_report(
    model_version: str,
    documents: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    *,
    corpus_version: str = "",
    classifier_scored: bool = False,
    scope: Any = None,
) -> EvalReport:
    """Score every doc type × subset from one flat list of results.

    Args:
        documents: ``(expected, got, metadata)`` triples. ``metadata`` carries
            ``doc_type``, ``modality_mode``, ``is_scanned``, ``page_count`` and
            ``source_id`` — the fields :func:`subset_of` reads.
    """
    report = EvalReport(
        model_version=model_version,
        corpus_version=corpus_version,
        classifier_scored=classifier_scored,
    )

    from common.canonical import without_system_fields

    buckets: dict[tuple[str, str], list[Any]] = {}
    for expected, got, metadata in documents:
        # Not scored: the system supplies them, the model is never asked
        # (common.canonical.SYSTEM_SUPPLIED_FIELDS). Stripped from both sides, so a
        # gold label or frozen eval set that still carries them counts nothing.
        expected, got = without_system_fields(expected), without_system_fields(got)
        # A gold label written for another line's schema is scored in this
        # line's (configs/label_mappings.yaml), and only on what that schema can
        # hold - as its training target was built (common.canonical.schema_label).
        from common.canonical import schema_label

        # Both sides: a key outside the schema is not a field this report
        # measures. The constrained model cannot write one; an unconstrained
        # output that does fails schema_validity_rate, which is where it counts.
        doc_type = metadata.get("doc_type", "unknown")
        selectors = (doc_type, metadata.get("acord_form"), metadata.get("lob"))
        expected = schema_label(expected, *selectors)
        got = schema_label(got, *selectors)
        for subset in subset_of(metadata):
            buckets.setdefault((doc_type, subset), []).append((expected, got, metadata))

    for (doc_type, subset), rows in sorted(buckets.items()):
        report.subsets.append(score_subset(doc_type, subset, rows))

    missing = expected_subsets(scope) - {s.subset for s in report.subsets}
    if missing:
        # Reported rather than raised: an eval set with no scanned documents is
        # a coverage gap in the eval set, and pretending it scored 100% would be
        # far worse than saying it was not measured.
        log.warning(
            "eval set contains no %s document(s), so those subset metrics are absent. The "
            "promotion gate treats an absent metric as not-passed, which is correct — but the "
            "real fix is adding those documents to the frozen eval set (arch §8).",
            sorted(missing),
        )
    return report


def write_report(report: EvalReport, client: BlobClient) -> list[str]:
    """Write per-doc-type reports plus the top-level summary."""
    written = []
    for doc_type in sorted({s.doc_type for s in report.subsets}):
        key = paths.eval_report(report.model_version, doc_type)
        client.write_json(key, {
            "model_version": report.model_version,
            "doc_type": doc_type,
            "subsets": [s.as_dict() for s in report.for_doc_type(doc_type)],
        })
        written.append(key)

    summary_key = paths.eval_report(report.model_version)
    client.write_json(summary_key, report.as_dict())
    written.append(summary_key)
    return written


def render(report: EvalReport) -> str:
    lines = [f"eval {report.model_version} (corpus {report.corpus_version or 'unrecorded'})"]
    for subset in report.subsets:
        accuracy = subset.metrics.get("field_normalized_match")
        lines.append(
            f"  {subset.doc_type:8} {subset.subset:12} n={subset.documents:3} "
            + (f"field={accuracy:.3f}" if isinstance(accuracy, float) else "field=n/a")
        )
    if not report.classifier_scored:
        lines.append("  doc_type_classifier_accuracy: not scored this pass (SPEC_11 classifier)")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Evaluate a model version on the frozen eval set")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus-version", dest="corpus_version", required=True)
    parser.add_argument("--doc-types", nargs="+", default=list(ACTIVE_DOC_TYPES),
                        choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--scope", default=None, help="configs/scopes.yaml; defaults to unified")
    args = parser.parse_args(list(argv) if argv is not None else None)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('evaluation.run_eval', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = BlobClient()

    # Runs before anything is evaluated: a leaked eval set makes every number
    # below meaningless, so it is not worth computing them first.
    assert_eval_set_disjoint(client, args.corpus_version, args.tenant)

    # Through serving.pipeline, never a bespoke inference path (SPEC_08): the
    # report measures what production serves.
    from common.scopes import default_scope, get_scope
    from evaluation.golden_eval import dumps, evaluate_version
    from inference_core.model_runner import load_model

    scope = get_scope(args.scope) if args.scope else default_scope()
    body = evaluate_version(
        client, load_model(args.model, client),
        version=args.model, corpus_version=args.corpus_version, scope=scope,
        tenant_id=args.tenant,
    )
    print(dumps(body.get("gate_metrics") or {}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
