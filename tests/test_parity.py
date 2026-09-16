"""Training and serving must agree (arch v2.1 §10.2).

The three checks that make "train what you serve" a fact rather than an
intention. Every other test in this suite verifies one side of a boundary; these
run **both sides** and compare.

They matter because the failure they catch is silent. If ms-swift's chat template
and vLLM's processor tokenize the same example differently — one extra BOS token,
a different image-token count, a system prompt joined with a different separator
— nothing raises. The model simply performs worse in production than it did in
evaluation, and the gap looks like ordinary generalisation loss.

**Two of these need a GPU and the pinned stack**, so they are marked ``gpu`` and
skipped in CI. That is a real limitation, not a formality: until they run on a
pod, train/serve parity is asserted by construction rather than measured. The
third — prompt parity — is pure and runs everywhere, because the prompt renderer
is the one piece both sides share in this repo today.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Ten fixed examples, per §10.2. Fixed rather than sampled so a failure is
#: reproducible and a fix is verifiable against the same inputs.
PARITY_EXAMPLE_COUNT = 10

#: Field values must match on at least this many. Not all ten, because bf16
#: kernels differ slightly between HF and vLLM and one document occasionally
#: lands on the other side of a rounding boundary.
MIN_MATCHING_EXAMPLES = 9


# --------------------------------------------------------------------------
# Prompt parity — pure, runs everywhere
# --------------------------------------------------------------------------

def test_the_serving_prompt_is_the_corpus_prompt():
    """One renderer, one source of truth (arch v2.1 §7.3).

    v1 had three prompt texts and two prompt homes. A corpus built against one
    and served against another trains a model to expect instructions it never
    receives — and the symptom is a quieter model, not an error.
    """
    from common.prompts import render_system_prompt

    for doc_type, acord_form in (("policy", None), ("lossrun", None), ("acord", "25")):
        for mode in ("ocr_plus_image", "image_only"):
            rendered = render_system_prompt(doc_type, mode, acord_form)
            assert rendered.strip(), f"{doc_type}/{mode} rendered an empty prompt"
            # The schema is injected, not described: the model is given the
            # exact structure it must produce.
            assert "{" in rendered and "}" in rendered


def test_the_reference_prompts_match_the_renderer():
    """The snapshots under testing/prompts are generated, never hand-edited.

    A hand-edited snapshot is a prompt that exists in the repo, is read by a
    human reviewing behaviour, and is not the one the model ever saw.
    """
    from testing.render_prompts import drifted

    assert not drifted(), (
        "reference prompts no longer match common.prompts. Regenerate with "
        "`python -m testing.render_prompts --write` — never hand-edit them."
    )


def test_noisy_ocr_is_not_announced_in_the_prompt():
    """The noise lives in the data, not the instruction (arch §6).

    A distinct prompt for noisy OCR would teach the model that bad OCR is
    announced — and at inference nothing announces it.
    """
    from common.prompts import render_system_prompt

    assert (
        render_system_prompt("policy", "ocr_plus_image")
        == render_system_prompt("policy", "noisy_ocr_image")
    )


def test_the_prompt_version_is_recorded_so_drift_is_attributable():
    """A corpus manifest records the prompt version it was built with (§8.1). A
    release records the version it serves with (§12.3). Comparing them is what
    turns "the model got worse" into "the prompt changed on the 14th"."""
    from common.prompts import prompt_versions

    for doc_type, acord_form in (("policy", None), ("acord", "25")):
        versions = prompt_versions(doc_type, acord_form)
        assert versions["prompt_template_version"], f"{doc_type} records no prompt version"
        # Schema and prompt travel together: a schema change with an unchanged
        # prompt version is a corpus nobody can attribute a regression to.
        assert versions.get("schema_version")


# --------------------------------------------------------------------------
# Token parity — needs the pinned stack
# --------------------------------------------------------------------------

@pytest.mark.gpu
def test_template_token_parity():  # pragma: no cover - needs the pinned stack
    """ms-swift's template and vLLM's processor must produce IDENTICAL prompt
    token IDs and identical image-token counts for the same example.

    This is the check that makes train/serve parity a measurement. A single
    extra special token shifts every position the model learned; a different
    image-token count means the picture it sees at serving time is not the one it
    trained on. Neither raises.
    """
    pytest.skip(
        "Phase 0 spike items 1-2. Tokenize the same §7.1 example through "
        "ms-swift's Qwen3-VL template and through vLLM's processor, and assert "
        "the prompt token IDs and the image-token count are equal. Until this "
        "runs on a pod, train/serve parity is asserted by construction rather "
        "than measured."
    )


@pytest.mark.gpu
def test_serving_output_parity():  # pragma: no cover - needs a GPU
    """HF and vLLM must extract the same VALUES from the same ten documents.

    Parsed and normalized, not compared as text: bf16 kernels differ slightly
    between the two runtimes, so byte equality would fail on a correct pair. What
    must hold is that the extracted field values agree on at least nine of ten,
    and that no single example differs in more than one field — a broad, shallow
    disagreement is rounding, while a deep one on a single document is a bug.
    """
    pytest.skip(
        f"Phase 8 GPU milestone. Generate over {PARITY_EXAMPLE_COUNT} fixed examples "
        "through the HF backend and through vLLM on the merged model, normalize both "
        f"with common.normalize, and assert agreement on >= {MIN_MATCHING_EXAMPLES} "
        "with no example differing in more than one field (arch v2.1 §10.2)."
    )


# --------------------------------------------------------------------------
# The contract these tests are about, checked without a GPU
# --------------------------------------------------------------------------

def test_training_and_serving_read_the_same_vision_settings():
    """The settings that decide what the model sees come from ONE file.

    Two files holding a resolution cap is how a model gets trained at one
    resolution and served at another — and nothing downstream reports that as
    anything but degraded accuracy (arch v2.1 §7a).
    """
    from common.config import SHARED_VISION_CONFIG, vision_for_task
    from common.tasks import Task

    assert SHARED_VISION_CONFIG.exists()
    for task in Task:
        budget = vision_for_task(str(task))
        assert budget["max_pixels"] > 0
        assert budget["min_pixels"] < budget["max_pixels"], (
            f"{task}: min_pixels must be below max_pixels, or every page is upscaled"
        )


def test_training_and_serving_read_the_same_sequence_caps():
    from common.config import SHARED_SEQUENCE_CONFIG, sequence_for_task
    from common.tasks import Task

    assert SHARED_SEQUENCE_CONFIG.exists()
    for task in Task:
        budget = sequence_for_task(str(task))
        assert budget["max_output_tokens"] < budget["max_seq_len"]


def test_the_trainer_max_length_covers_every_task_the_corpus_holds():
    """ms-swift takes ONE max_length and the corpus interleaves every task, so
    anything below the largest cap truncates the longest one rather than
    rejecting it — and a clipped assistant span trains the model to stop early."""
    from common.config import seq_cap_for_task, shared_sequence_config
    from common.tasks import Task
    from training.train import corpus_max_length

    largest = corpus_max_length()
    declared = shared_sequence_config().get("tasks", {})
    for task in Task:
        name = str(task)
        assert seq_cap_for_task(name) <= largest
        for doc_type in (declared.get(name, {}).get("by_doc_type") or {}):
            assert seq_cap_for_task(name, doc_type) <= largest, (
                f"{name}/{doc_type} exceeds the trainer's max_length"
            )


def test_the_serving_config_and_the_shared_caps_agree():
    """A serving endpoint configured below a task's cap truncates exactly the
    documents the cap was raised for."""
    from common.config import serving_config
    from training.train import corpus_max_length

    serving_cap = int(serving_config()["sequence"]["max_seq_len"])
    # The trainer's own number, including per-doc-type overrides. Recomputing it
    # here would be a second definition that could disagree with the first.
    largest = corpus_max_length()
    assert serving_cap >= largest, (
        f"serving is capped at {serving_cap} while the largest task cap is {largest}; "
        "a routed policy would be truncated at inference only"
    )


def test_the_gpu_parity_tests_are_registered_rather_than_forgotten():
    """A skipped test that nobody remembers is an unchecked assumption with
    extra steps. This asserts they exist and are marked, so `pytest -m gpu` on a
    pod runs them rather than finding nothing."""
    import inspect

    source = inspect.getsource(__import__(__name__, fromlist=["_"]))
    for name in ("test_template_token_parity", "test_serving_output_parity"):
        assert f"def {name}" in source
        assert source.index("@pytest.mark.gpu") < source.index(f"def {name}")


def test_the_parity_fixtures_exist_for_the_gpu_run():
    """Ten fixed examples, so a parity failure is reproducible and a fix is
    verifiable against the same inputs."""
    golden = sorted((ROOT / "tests" / "fixtures" / "golden").glob("*.golden.json"))
    assert golden, "no golden fixtures to build parity examples from"
    for path in golden:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "line_of_business" in payload
