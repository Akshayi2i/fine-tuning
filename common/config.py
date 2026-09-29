"""Config loading — YAML plus environment, with the cross-file invariants enforced.

Config lives in ``configs/`` and secrets live in the environment; nothing is
hardcoded (master §7). Beyond loading, this module's real job is enforcing the
invariants that span *two* files and would otherwise be maintained by hand:

* **Resolution parity** — the image cap in ``base_model.yaml`` and
  ``inference/vllm_serving.yaml`` must be equal. They are read by different
  stages, so nothing else would notice them drifting apart, and the symptom
  would be a model served a distribution it never trained on.
* **Effective batch consistency** — the declared effective batch must equal
  ``per_device × grad_accum × n_gpu``, or the run manifest records a batch size
  that was never actually used, and the run stops being reproducible.
"""

from __future__ import annotations

import os
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import yaml

from common.constants import DEFAULT_RESOLUTION_CAP_PX, RESOLUTION_CAP_RANGE_PX

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

BASE_MODEL_CONFIG = CONFIG_DIR / "base_model.yaml"
SERVING_CONFIG = CONFIG_DIR / "inference" / "vllm_serving.yaml"

#: Read by dataset build, training and serving alike (arch v2.1 §7a). One file
#: per dimension rather than one block per stage, because the failure mode of two
#: stages disagreeing is a model served a distribution it never trained on — and
#: nothing downstream reports that as anything but degraded accuracy.
SHARED_VISION_CONFIG = CONFIG_DIR / "shared" / "vision.yaml"
SHARED_SEQUENCE_CONFIG = CONFIG_DIR / "shared" / "sequence.yaml"

#: WHAT a run covers — which document types and tasks (arch v2.1 §4.1). Read by
#: training, orchestration, the gate and serving alike, for the same reason the
#: shared files above are: a scope two stages disagree about is a model gated on
#: one coverage and served as another.
SCOPES_CONFIG = CONFIG_DIR / "scopes.yaml"

#: WHICH LOBs share a visual grammar (SPEC_09 §2.1). One table, two readers:
#: training uses it to decide which documents a family's adapter trains on, and
#: the schema registry uses it to decide which canonical schemas to register.
#: Two copies of this mapping would eventually disagree, and the symptom would be
#: a document extracted by an adapter that never saw its layout.
LAYOUT_FAMILIES_CONFIG = CONFIG_DIR / "layout_families.yaml"


class ConfigError(RuntimeError):
    """Raised on a missing config, a missing required env var, or a broken invariant."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, failing loudly rather than returning ``{}``."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} did not parse to a mapping")
    return data


@lru_cache(maxsize=1)
def base_model_config() -> dict[str, Any]:
    return load_yaml(BASE_MODEL_CONFIG)


#: Overrides ``model.local_dir`` in base_model.yaml.
BASE_MODEL_DIR_ENV = "FIDEON_BASE_MODEL_DIR"


def base_model_dir() -> Path | None:
    """The local directory holding the base weights, or ``None`` if there is none.

    Looked for under ``model.local_dir`` (or ``$FIDEON_BASE_MODEL_DIR``): the
    directory itself, a folder named after the model, or a Hugging Face cache
    snapshot — the pinned revision's when it is there. A directory counts only
    if it holds a ``config.json``.
    """
    model = base_model_config()["model"]
    root_setting = os.environ.get(BASE_MODEL_DIR_ENV) or model.get("local_dir")
    if not root_setting:
        return None
    root = Path(root_setting)
    model_id = str(model["model_id"])
    candidates = [root, root / model_id.rsplit("/", 1)[-1], root / model_id]
    snapshots = root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if snapshots.is_dir():
        pinned = snapshots / str(model.get("revision", ""))
        candidates += [pinned, *sorted(p for p in snapshots.iterdir() if p.is_dir())]
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    return None


def base_model_source() -> str:
    """What to LOAD the base from: the local directory, else the Hub id.

    The Hub id is the fallback for a machine without the weights (a laptop dry
    run, CI). A real launch never reaches it: ``training.train.assert_on_pod``
    refuses to start while ``local_dir`` is configured and holds no model.
    """
    local = base_model_dir()
    return str(local) if local is not None else str(base_model_config()["model"]["model_id"])


@lru_cache(maxsize=1)
def serving_config() -> dict[str, Any]:
    return load_yaml(SERVING_CONFIG)


@cache
def training_config(name: str) -> dict[str, Any]:
    """Load ``configs/training/{name}.yaml``. ``unified`` is the only one until a type graduates (§4.2)."""
    return load_yaml(CONFIG_DIR / "training" / f"{name}.yaml")


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Read an env var. Secrets come from the environment only, never from YAML."""
    value = os.environ.get(name, default)
    if required and not value:
        raise ConfigError(
            f"required environment variable {name} is not set — see .env.example for the full list"
        )
    return value


@lru_cache(maxsize=1)
def scopes_config() -> dict[str, Any]:
    """The raw scope declarations. Parsed and validated by :mod:`common.scopes`."""
    return load_yaml(SCOPES_CONFIG)


@lru_cache(maxsize=1)
def layout_families_config() -> dict[str, Any]:
    """The raw family declarations. See :func:`lobs_in_family`."""
    return load_yaml(LAYOUT_FAMILIES_CONFIG)


def lobs_in_family(family: str) -> tuple[str, ...]:
    """The lines of business belonging to one layout family, in declared order.

    Order is preserved because it is the order the LOBs are registered in and
    reported in; a set would make the schema registry's iteration order depend on
    hash seeding, and a corpus manifest that lists its schemas in a different
    order on every build cannot be diffed across model versions.
    """
    families = layout_families_config().get("families") or {}
    entry = families.get(family)
    if entry is None:
        raise ConfigError(
            f"no layout family {family!r} in {LAYOUT_FAMILIES_CONFIG.name}; "
            f"declared families: {sorted(families)}"
        )
    return tuple(entry.get("lobs") or ())


@lru_cache(maxsize=1)
def shared_vision_config() -> dict[str, Any]:
    return load_yaml(SHARED_VISION_CONFIG)


@lru_cache(maxsize=1)
def shared_sequence_config() -> dict[str, Any]:
    return load_yaml(SHARED_SEQUENCE_CONFIG)


def vision_for_task(task: str) -> dict[str, Any]:
    """The pixel budget for one task (arch v2.1 §7a).

    ``max_pixels`` rather than a long-side cap, so a landscape page and a
    portrait page of the same document get the same budget. Under v1's long-side
    cap a landscape Loss Run was quietly cheaper — and read worse — than the same
    table printed portrait.

    An unlisted task falls back to ``defaults``, which is the extraction budget:
    a new task is then expensive but correct, rather than silently unbudgeted.
    """
    from common.tasks import parse

    config = shared_vision_config()
    resolved = dict(config.get("defaults", {}))
    resolved.update(config.get("tasks", {}).get(str(parse(task)), {}))
    if "max_pixels" not in resolved:
        raise ConfigError(
            f"no max_pixels for task {task!r} and no defaults block in {SHARED_VISION_CONFIG.name}"
        )
    return resolved


def pixel_budget(tasks: Any = None) -> tuple[int, int]:
    """``(min_pixels, max_pixels)`` for the full-resolution tasks among ``tasks``.

    The ONE budget both sides resize pages to: the ms-swift processor env at
    training (``training.train._pixel_budget``) and the vLLM engine's
    ``mm_processor_kwargs`` at serving. Each side used to take its own — training
    the processor default until recently, serving the processor default still —
    so the day a budget changed, the model would be served pixels it never saw.

    ``tasks`` defaults to every task a corpus builds. Raises when they disagree:
    a trainer and an engine each take one budget.
    """
    from common.tasks import CORPUS_TASKS, FULL_RESOLUTION_TASKS, Task

    chosen = [t for t in (tasks if tasks is not None else CORPUS_TASKS)
              if t in FULL_RESOLUTION_TASKS] or [Task.EXTRACT]
    budgets = {
        (int(b["min_pixels"]), int(b["max_pixels"]))
        for b in (vision_for_task(str(task)) for task in chosen)
    }
    if len(budgets) != 1:
        raise ConfigError(
            f"the full-resolution tasks {sorted(map(str, chosen))} disagree on their pixel "
            f"budget {sorted(budgets)}; one model is trained and served at one budget."
        )
    (budget,) = budgets
    # Whole visual tokens only. The trainer is given the budget both as pixels
    # and as a token count (pixels // 1024); a budget that is not a multiple
    # makes those two forms disagree, and which one a processor reads decides
    # the resolution the model sees.
    from data_pipeline.dataset_builder.cap_check import PIXELS_PER_VISUAL_TOKEN

    uneven = [v for v in budget if v % PIXELS_PER_VISUAL_TOKEN]
    if uneven:
        raise ConfigError(
            f"pixel budget {budget} is not a whole number of {PIXELS_PER_VISUAL_TOKEN}-pixel "
            f"visual tokens ({uneven}); round it in {SHARED_VISION_CONFIG.name}."
        )
    return budget


def sequence_for_task(task: str, doc_type: str | None = None) -> dict[str, Any]:
    """The sequence cap and reserved output budget for one task (arch v2.1 §7a).

    ``max_output_tokens`` is reserved, never borrowed from. The assistant span is
    the one part of a sequence that must never be truncated: a clipped JSON
    target trains the model to stop early, and on a Loss Run that means training
    it to omit claim rows.

    ``doc_type`` selects a per-type override where one exists. Extraction is one
    task — the model is conditioned on the document type through the prompt, not
    routed to a different task — but a routed policy sends more pages against a
    larger schema than an ACORD does, so it gets a larger budget.
    """
    from common.tasks import parse

    config = shared_sequence_config()
    resolved = dict(config.get("defaults", {}))
    per_task = dict(config.get("tasks", {}).get(str(parse(task)), {}))
    overrides = per_task.pop("by_doc_type", {}) or {}
    resolved.update(per_task)
    if doc_type and doc_type in overrides:
        resolved.update(overrides[doc_type])
    if "max_seq_len" not in resolved:
        raise ConfigError(
            f"no max_seq_len for task {task!r} and no defaults block in {SHARED_SEQUENCE_CONFIG.name}"
        )
    return resolved


def seq_cap_for_task(task: str, doc_type: str | None = None) -> int:
    """Just the sequence cap, for callers that do not need the output budget."""
    return int(sequence_for_task(task, doc_type)["max_seq_len"])


def assert_task_budgets_are_coherent() -> None:
    """Every task must appear in both shared files, with output inside the cap.

    The two files are edited separately and read together. A task present in one
    and missing from the other resolves to a default that was chosen for a
    different shape of input — which is exactly the class of silent mismatch
    these files exist to prevent.
    """
    from common.tasks import Task

    vision_tasks = set(shared_vision_config().get("tasks", {}))
    sequence_tasks = set(shared_sequence_config().get("tasks", {}))

    for task in Task:
        name = str(task)
        missing = [
            f.name for f, present in (
                (SHARED_VISION_CONFIG, name in vision_tasks),
                (SHARED_SEQUENCE_CONFIG, name in sequence_tasks),
            ) if not present
        ]
        if missing:
            raise ConfigError(
                f"task {name!r} is not budgeted in {missing} — it would fall back to a default "
                "chosen for a different shape of input (arch v2.1 §7a)."
            )

    declared = shared_sequence_config().get("tasks", {})
    for task in Task:
        name = str(task)
        # The bare budget, then every per-doc-type override of it — an override
        # is a whole separate cap, and one that reserves more output than it
        # allows total would only surface as truncation at build time.
        variants = [(None, sequence_for_task(name))]
        variants += [
            (dt, sequence_for_task(name, dt))
            for dt in (declared.get(name, {}).get("by_doc_type") or {})
        ]
        for doc_type, budget in variants:
            cap, output = int(budget["max_seq_len"]), int(budget["max_output_tokens"])
            label = f"{name} ({doc_type})" if doc_type else name
            if output >= cap:
                raise ConfigError(
                    f"task {label!r} reserves {output} output tokens inside a {cap}-token cap, "
                    "leaving nothing for the schema, the page images or the OCR text."
                )
            serving = int(serving_config().get("generation", {}).get("max_new_tokens", 0))
            if serving and output > serving:
                raise ConfigError(
                    f"task {label!r} reserves {output} output tokens, but serving generates at "
                    f"most {serving} (vllm_serving.yaml generation.max_new_tokens): an answer "
                    "training teaches in full would be cut off at serving."
                )


def resolution_cap_px() -> int:
    """The image long-side cap, validated against the architecture's range.

    Image token count is a direct function of page resolution, making this the
    single biggest cost and latency lever (arch §11).

    **Superseded by** :func:`vision_for_task` (arch v2.1 §7a). Kept because the
    v1 OCR and inference paths still read a single long-side cap; they move to
    per-task pixel budgets in a later phase, and this goes with them.
    """
    cap = int(base_model_config().get("vision", {}).get("max_image_long_side_px", DEFAULT_RESOLUTION_CAP_PX))
    low, high = RESOLUTION_CAP_RANGE_PX
    if not low <= cap <= high:
        raise ConfigError(
            f"resolution cap {cap}px is outside the architecture's {low}-{high}px range (arch §11). "
            "Below it, small print and checkboxes are missed; above it, cost and latency blow out."
        )
    return cap


def assert_resolution_parity() -> None:
    """The cap must be identical in training prep and production inference.

    Not a tuning knob — a mismatch is a correctness bug. The model would be
    served images at a resolution it never saw, and nothing downstream would
    report it as anything but degraded accuracy.
    """
    train_cap = base_model_config().get("vision", {}).get("max_image_long_side_px")
    serve_cap = serving_config().get("vision", {}).get("max_image_long_side_px")
    if train_cap != serve_cap:
        raise ConfigError(
            f"resolution cap mismatch: base_model.yaml says {train_cap}px, "
            f"inference/vllm_serving.yaml says {serve_cap}px. These must be equal — the model is "
            "trained at one resolution and would be served at another (arch §11)."
        )


def assert_effective_batch(config: dict[str, Any], n_gpu: int = 1) -> None:
    """Declared effective batch must match what the settings actually produce."""
    batch = config.get("batch", {})
    declared = batch.get("effective_batch_size")
    if declared is None:
        return
    actual = int(batch.get("per_device_train_batch_size", 1)) * int(batch.get("gradient_accumulation_steps", 1)) * n_gpu
    if actual != declared:
        raise ConfigError(
            f"effective_batch_size is declared {declared} but per_device × grad_accum × n_gpu = {actual}. "
            "The run manifest would record a batch size that was never used."
        )


def assert_model_revision_pinned() -> None:
    """A floating base-model revision makes a regression unattributable."""
    revision = base_model_config().get("model", {}).get("revision")
    if not revision or revision == "PIN_ME":
        raise ConfigError(
            "base_model.yaml model.revision is unpinned. Two runs could silently use different "
            "base weights, which makes 'why did this regress' unanswerable. Pin it (Phase 0)."
        )


def validate_all(*, require_pinned_revision: bool = False) -> None:
    """Run every cross-file invariant. Called at the start of any real run.

    ``require_pinned_revision`` is off by default so the fixture-driven test
    suite runs before Phase 0's infrastructure exists; orchestration turns it on.
    """
    resolution_cap_px()
    assert_resolution_parity()
    assert_task_budgets_are_coherent()
    # Every scope's training config, not one literal name: a scope pointing at a
    # missing or incoherent config must fail at launch rather than when that
    # scope is first trained.
    from common.scopes import assert_scopes_are_coherent, load_scopes

    assert_scopes_are_coherent()
    for name in sorted({scope.training_config for scope in load_scopes().values()}):
        assert_effective_batch(training_config(name))
    if require_pinned_revision:
        assert_model_revision_pinned()


def generation_config() -> dict[str, Any]:
    """Generation settings, shared by eval, serving, and testing.

    Logprobs are mandatory: they are the entire basis of per-field confidence
    (arch §5), so a config that disables them is rejected here rather than
    producing confidence-free output somewhere downstream.
    """
    gen = dict(serving_config().get("generation", {}))
    if not gen.get("logprobs"):
        raise ConfigError(
            "generation.logprobs is disabled. Per-field confidence is derived from token "
            "logprobs (arch §5); without them the pipeline cannot route low-confidence "
            "extractions to review, which is the point of the confidence work."
        )
    return gen
