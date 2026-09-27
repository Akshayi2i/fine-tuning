"""Training scopes — what one run covers (arch v2.1 §4.1).

The scope is the addressing key: it becomes the run-id lineage and the artifact
path segment. Two properties have to hold or the whole scheme is unsafe:

* **unified is unchanged** — it still mints ``extractor-v{n}`` and still renders
  the ``unified`` path segment, so every artifact already in Blob is reachable;
* **a scope cannot widen anything** — not its document types, and not the set of
  metrics it is excused from. Otherwise "configure a scope" becomes "waive a
  gate", which is exactly what the §15.2 rules exist to prevent.
"""

from __future__ import annotations

import pytest

from common.constants import ACTIVE_DOC_TYPES
from common.scopes import (
    ScopeError,
    assert_scopes_are_coherent,
    default_scope,
    get_scope,
    load_scopes,
    narrow,
    parse_scopes,
    structural_not_applicable,
)
from common.tasks import Task

# --------------------------------------------------------------------------
# The compatibility pin
# --------------------------------------------------------------------------


def test_the_unified_scope_addresses_exactly_what_it_always_did():
    """THE backward-compatibility test. `extractor-v1` and `merged-models/unified/…`
    exist in Blob; if either string moves, those artifacts become unreachable by
    the name that addresses them."""
    unified = get_scope("unified")

    assert unified.run_id("v1") == "extractor-v1"
    assert unified.path_segment == "unified"
    assert unified.doc_types == tuple(ACTIVE_DOC_TYPES)
    assert unified.is_unified


def test_the_default_scope_is_unified():
    """A command with no --scope must keep training what it trained yesterday."""
    assert default_scope().name == "unified"
    assert parse_scopes(None) == (get_scope("unified"),)


def test_a_scoped_run_is_addressed_separately_from_a_graduated_adapter():
    """`adapters/policy/` already belongs to a §4.2 graduated per-type adapter.
    A policy-SCOPE adapter is a different artifact — trained on the base, not on
    the merged foundation — so it must not land on the same prefix."""
    policy = get_scope("policy")

    assert policy.run_id("v2") == "policy-v2"
    assert policy.path_segment == "scope/policy"
    assert policy.path_segment != "policy"


# --------------------------------------------------------------------------
# Narrowing, never widening
# --------------------------------------------------------------------------


def test_a_scope_trains_only_active_document_types():
    for scope in load_scopes().values():
        assert set(scope.doc_types) <= set(ACTIVE_DOC_TYPES), scope.name


def test_a_scope_never_serves_what_it_did_not_train():
    """A release serving a type its model never saw returns confident nonsense,
    and nothing downstream reports that as anything but a bad extraction."""
    for scope in load_scopes().values():
        assert set(scope.serves) <= set(scope.doc_types), scope.name


def test_doc_types_can_narrow_a_scope_but_not_widen_it():
    policy = get_scope("policy")
    unified = get_scope("unified")

    assert narrow(unified, ["policy"]).doc_types == ("policy",)
    assert narrow(unified, None) is unified

    with pytest.raises(ScopeError, match="outside scope"):
        narrow(policy, ["acord"])


def test_narrowing_keeps_the_name_and_the_lineage():
    """The artifacts are still addressed by the scope's name, so a narrowed run
    must still describe that scope — which is why widening is refused."""
    narrowed = narrow(get_scope("unified"), ["policy", "acord"])
    assert narrowed.name == "unified" and narrowed.run_id("v2") == "extractor-v2"
    assert "lossrun" not in narrowed.serves


# --------------------------------------------------------------------------
# Not-applicable is derived, not declared
# --------------------------------------------------------------------------


def test_every_declared_not_applicable_metric_is_structurally_absent():
    for scope in load_scopes().values():
        assert scope.not_applicable_metrics <= structural_not_applicable(scope), scope.name


def test_a_policy_scope_has_no_loss_run_reconciliation_to_do():
    """No Loss Run in the corpus means no printed totals to reconcile. That is a
    statement about the eval set, not a pass."""
    assert "lossrun_totals_reconciliation_rate" in structural_not_applicable(get_scope("policy"))
    assert "lossrun_totals_reconciliation_rate" not in structural_not_applicable(get_scope("unified"))


def test_a_single_type_scope_still_classifies_so_the_floor_keeps_meaning():
    """A policy-only release still has to recognise an ACORD and refuse it. If
    `classify` were dropped, classifier accuracy would become not-applicable and
    a real guarantee would quietly disappear."""
    policy = get_scope("policy")
    assert Task.CLASSIFY in policy.tasks


def test_the_classifier_metric_applies_exactly_when_classify_rows_are_trained(monkeypatch):
    """Declared is not trained. While the corpus builds no classify rows the
    classifier is never trained, so its accuracy is not a measurement of this
    model — and as a required gate it blocked every candidate. The moment the
    corpus emits classify rows, the floor applies again, for single-type scopes
    too."""
    from common import scopes as S
    from common.tasks import CORPUS_TASKS

    policy = get_scope("policy")
    assert Task.CLASSIFY not in CORPUS_TASKS
    assert "doc_type_classifier_accuracy" in structural_not_applicable(policy)

    monkeypatch.setattr(S, "CORPUS_TASKS", CORPUS_TASKS | {Task.CLASSIFY})
    assert "doc_type_classifier_accuracy" not in structural_not_applicable(policy)


def test_a_scope_cannot_excuse_itself_from_a_metric_it_can_produce():
    """The loophole this rule closes: declaring a metric not-applicable is a
    waiver unless the scope's own shape says nothing could have produced it."""
    from common.scopes import _build

    with pytest.raises(ScopeError, match="declares"):
        _build("sneaky", {
            "doc_types": ["lossrun"],
            "tasks": ["classify", "extract", "lossrun_totals"],
            # It trains Loss Runs, so this one IS producible here.
            "not_applicable_metrics": ["lossrun_totals_reconciliation_rate"],
        })


# --------------------------------------------------------------------------
# Config coherence
# --------------------------------------------------------------------------


def test_the_declared_scopes_are_coherent():
    """Runs the same validation `config.validate_all` runs at launch."""
    assert_scopes_are_coherent()


def test_an_unknown_scope_names_the_ones_that_exist():
    with pytest.raises(ScopeError, match="unknown scope"):
        get_scope("policyy")


def test_two_scopes_cannot_share_a_lineage():
    """Colliding lineages mean colliding run ids, and each scope would resolve to
    the other's artifacts."""
    lineages = [scope.lineage for scope in load_scopes().values()]
    assert len(lineages) == len(set(lineages))


def test_a_scope_cannot_train_an_inactive_document_type():
    from common.scopes import _build

    with pytest.raises(ScopeError, match="not active document types"):
        _build("quotes", {"doc_types": ["quote"], "tasks": ["classify", "extract"]})


def test_the_unified_scope_may_not_rename_its_lineage():
    from common.scopes import _build

    with pytest.raises(ScopeError, match="must keep the lineage"):
        _build("unified", {
            "lineage": "unified",
            "doc_types": list(ACTIVE_DOC_TYPES),
            "tasks": ["classify", "extract"],
        })


def test_a_scope_is_refused_rather_than_silently_emptied():
    from common.scopes import _build

    with pytest.raises(ScopeError, match="trains no document types"):
        _build("empty", {"doc_types": [], "tasks": ["extract"]})
    with pytest.raises(ScopeError, match="declares no tasks"):
        _build("taskless", {"doc_types": ["policy"], "tasks": []})


def test_a_duplicate_scope_on_the_command_line_is_refused():
    with pytest.raises(ScopeError, match="named twice"):
        parse_scopes(["policy", "policy"])


def test_several_scopes_resolve_in_order():
    """`--scope unified --scope policy` trains two independent adapters, in the
    order given."""
    assert [s.name for s in parse_scopes(["unified", "policy"])] == ["unified", "policy"]
