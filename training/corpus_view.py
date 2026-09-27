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
