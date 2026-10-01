"""A scope's view of one corpus (arch v2.1 §4.1, §8.2).

**The corpus is built once and filtered, never built per scope.** A per-scope
build would draw its own group assignment, its own split and its own modality
regimes, so a document could be *train* in the unified corpus and *test* in the
policy one. Two releases built that way are not comparable, and the policy
model's training documents could sit in the unified model's eval set — the exact
leakage the group-aware split exists to prevent.

So a scope reads a **filtered view**: the same rows, minus the document types it
does not cover, written beside the corpus under ``train/scope/{name}/``. The
unified scope reads the corpus files directly, so nothing is copied for it.

Filtering rather than re-splitting also keeps the epoch structure: every epoch
file holds each of the scope's train documents exactly once, in that epoch's
sampled modality regime, which is what makes "3 epochs" mean three passes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.scopes import Scope

log = logging.getLogger(__name__)

#: Materialized epoch files, matching the corpus build (arch v2.1 §6.1).
EPOCH_FILES = 4


class CorpusViewError(RuntimeError):
    """Raised when a scope's view cannot be built from the corpus."""


@dataclass
class CorpusView:
    """Where a scope's training run reads from, and what it found."""

    scope: str
    epoch_files: list[str] = field(default_factory=list)
    val_path: str = ""
    rows_by_epoch: dict[int, int] = field(default_factory=dict)
    val_rows: int = 0
    test_rows: int = 0
    dropped_rows: int = 0
    #: Line of business -> times its rows appear in each epoch file (line_balance).
    line_repeats: dict[str, int] = field(default_factory=dict)

    @property
    def train_rows(self) -> int:
        return sum(self.rows_by_epoch.values())

    def describe(self) -> str:
        return (
            f"{self.scope}: {self.train_rows} train row(s) across {len(self.epoch_files)} epoch "
            f"file(s), {self.val_rows} val row(s), {self.dropped_rows} row(s) outside the scope"
        )


def _filter(text: str, scope: Scope) -> tuple[str, int, int]:
    """Keep the rows this scope covers — its document types and, for a scope
    narrowed by line of business, its lines. Returns ``(jsonl, kept, dropped)``."""
    kept: list[str] = []
    dropped = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise CorpusViewError(f"corpus row is not JSON: {exc}") from exc
        if row.get("doc_type") in scope.doc_types and scope.covers_lob(row.get("lob")):
            kept.append(line)
        else:
            dropped += 1
    return "\n".join(kept) + ("\n" if kept else ""), len(kept), dropped


def _line_of(row: dict) -> str:
    lob = row.get("lob")
    return ",".join(sorted(map(str, lob))) if isinstance(lob, list) else str(lob or "unknown")


def _balance_settings(scope: Scope) -> dict[str, int]:
    from common.config import training_config

    settings = training_config(scope.training_config).get("line_balance") or {}
    return {"min_documents": int(settings.get("min_documents") or 0),
            "max_repeat": int(settings.get("max_repeat") or 1)}


def line_repeats(rows: list[dict], *, min_documents: int, max_repeat: int) -> dict[str, int]:
    """How many times each line's rows go into an epoch file (``line_balance``).

    Counted in DOCUMENTS, not rows: a line of long policies has many rows per
    document and is not small for it.
    """
    documents: dict[str, set] = {}
    for row in rows:
        documents.setdefault(_line_of(row), set()).add(row.get("source_id"))
    if min_documents <= 0 or max_repeat <= 1 or not documents:
        return {line: 1 for line in documents}
    # Toward the largest line, never past it: when every line is small (a smoke
    # corpus), repeating all of them alike would multiply the run's length and
    # change nothing about the balance.
    target = min(min_documents, max(len(ids) for ids in documents.values()))
    return {
        line: 1 if len(ids) >= target else min(max_repeat, -(-target // len(ids)))
        for line, ids in documents.items()
    }


def _balance(body: str, repeats: dict[str, int]) -> tuple[str, int]:
    """``body`` with each row written as many times as its line's repeat count."""
    out: list[str] = []
    for line in body.splitlines():
        if line.strip():
            out.extend([line] * repeats.get(_line_of(json.loads(line)), 1))
    return "\n".join(out) + ("\n" if out else ""), len(out)


def materialize(
    scope: Scope,
    corpus_version: str,
    client: BlobClient,
    *,
    tenant_id: str | None = None,
) -> CorpusView:
    """Write this scope's filtered epoch and validation files, and report them.

    A no-op for the unified scope: it covers every type, so the corpus files are
    already its view and copying them would double the storage to say nothing.
    """
    view = CorpusView(scope=scope.name)

    if scope.is_unified:
        view.epoch_files = [
            paths.corpus_epoch_file(corpus_version, epoch, tenant_id)
            for epoch in range(1, EPOCH_FILES + 1)
        ]
        view.val_path = paths.corpus_eval_split(corpus_version, "val", tenant_id)
        return view

    for epoch in range(1, EPOCH_FILES + 1):
        source = paths.corpus_epoch_file(corpus_version, epoch, tenant_id)
        if not client.exists(source):
            raise CorpusViewError(
                f"corpus {corpus_version} has no {source}. A scope filters the corpus that was "
                "built; it does not build one of its own."
            )
        body, kept, dropped = _filter(client.read_text(source), scope)
        if epoch == 1:
            # One set of repeat counts for every epoch, from the documents of
            # the first: every epoch file holds every training document once.
            balance = _balance_settings(scope)
            view.line_repeats = line_repeats(
                [json.loads(line) for line in body.splitlines() if line.strip()], **balance)
            boosted = {line: n for line, n in view.line_repeats.items() if n > 1}
            if boosted:
                log.info("line balance: repeating %s in every epoch file", boosted)
        body, kept = _balance(body, view.line_repeats)
        target = paths.corpus_scope_epoch_file(corpus_version, epoch, scope.name, tenant_id)
        client.write_text(target, body)
        view.epoch_files.append(target)
        view.rows_by_epoch[epoch] = kept
        view.dropped_rows += dropped

    if not view.train_rows:
        raise CorpusViewError(
            f"scope {scope.name!r} covers {list(scope.doc_types)}, and corpus {corpus_version} "
            "holds no training rows for any of them. Training on nothing would produce an "
            "adapter that reports success and learned nothing."
        )

    source = paths.corpus_eval_split(corpus_version, "val", tenant_id)
    if not client.exists(source):
        raise CorpusViewError(f"corpus {corpus_version} has no validation split at {source}")
    body, kept, _ = _filter(client.read_text(source), scope)
    view.val_path = paths.corpus_scope_eval_split(corpus_version, "val", scope.name, tenant_id)
    client.write_text(view.val_path, body)
    view.val_rows = kept

    test = paths.corpus_eval_split(corpus_version, "test", tenant_id)
    if client.exists(test):
        _body, view.test_rows, _ = _filter(client.read_text(test), scope)

    if not view.val_rows:
        # Not fatal here — it is the checkpoint selector and the gate that need
        # validation rows, and they say so themselves. But a run that trains with
        # no validation has no early stopping and no checkpoint selection, and an
        # operator should learn that before the GPU bill, not after.
        log.warning(
            "scope %s has no validation rows in corpus %s: early stopping and checkpoint "
            "selection will have nothing to read.", scope.name, corpus_version,
        )

    log.info("corpus view %s", view.describe())
    return view
