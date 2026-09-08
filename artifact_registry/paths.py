"""Blob and staging-volume layout — expressed once, here.

Every path in the system is built through this module. The layout appearing in
two places is how a training job ends up writing where a serving job will never
look, so nothing else constructs a path by string concatenation.

Two storage surfaces with **different guarantees**, and conflating them is the
bug this module's naming exists to prevent:

Azure Blob
    The artifact of record. Durable, versioned, and where the run registry lives.

RunPod staging volume
    Working storage between ``finetune`` (command 1) and ``package`` (command 2).
    Adapters and the ~16GB merged model sit here so quantization — which also
    runs on RunPod — does not pull them back out of Azure, a 32GB round trip for
    nothing. **No durability guarantee.** It is never called a registry
    (master §12a).

Tenancy (arch §8b): the ``tenant_id`` prefix is *reserved* on every path holding
tenant document data, so no migration is needed when a second broker arrives.
The build is single-tenant — ``tenant_id`` defaults from ``DEFAULT_TENANT_ID``
rather than being a required argument everywhere. The one rule that is live and
enforced is that a corpus file must never mix tenants, because corpus
composition is training data.
"""

from __future__ import annotations

import os
import re
from pathlib import PurePosixPath
from typing import Final, Literal

from common.constants import ACTIVE_DOC_TYPES

AdapterKind = Literal["foundation", "doc_type"]

#: Blob prefixes carrying tenant document data. These get the tenant prefix.
TENANT_SCOPED: Final[frozenset[str]] = frozenset(
    {"raw-documents", "processed", "golden-labels", "corpus"}
)

#: Blob prefixes holding no tenant data. These are never prefixed — the shared
#: Foundation adapter and the run registry are cross-tenant by design.
SHARED: Final[frozenset[str]] = frozenset(
    {"base-models", "adapters", "merged-models", "quantized-models",
     "registry", "eval-reports", "golden-eval-set", "calibration"}
)

_VERSION_RE = re.compile(r"^v\d+(\.\d+)*$")
_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

#: Used where an artifact is not per-document-type (a Foundation-only merge).
UNIFIED = "unified"


class PathError(ValueError):
    """Raised on an invalid tenant, version tag, document type, or format."""


# --------------------------------------------------------------------------
# Validation — cheap, and it turns a silent mis-write into an immediate error
# --------------------------------------------------------------------------

def default_tenant() -> str:
    return os.environ.get("DEFAULT_TENANT_ID", "default")


def _tenant(tenant_id: str | None) -> str:
    tenant = (tenant_id or default_tenant()).strip().lower()
    if not _TENANT_RE.match(tenant):
        raise PathError(
            f"invalid tenant_id {tenant!r}; expected lowercase alphanumeric with - or _ "
            "(it becomes a path segment, so it cannot contain slashes or spaces)"
        )
    return tenant


def _version(version: str) -> str:
    """Accept ``v1``, ``v2.1``. Reject a bare ``1`` or a floating tag."""
    tag = str(version).strip()
    if not _VERSION_RE.match(tag):
        raise PathError(
            f"invalid version {version!r}; expected a tag like 'v1' or 'v2.1'. "
            "Untagged artifacts cannot be traced back to the run that made them."
        )
    return tag


def _doc_type(doc_type: str, *, allow_unified: bool = False) -> str:
    dt = doc_type.strip().lower()
    if allow_unified and dt == UNIFIED:
        return dt
    if dt not in ACTIVE_DOC_TYPES:
        allowed = list(ACTIVE_DOC_TYPES) + ([UNIFIED] if allow_unified else [])
        raise PathError(f"unknown doc_type {doc_type!r}; expected one of {allowed}")
    return dt


def _join(*parts: str) -> str:
    return str(PurePosixPath(*[p.strip("/") for p in parts if p]))


# --------------------------------------------------------------------------
# Tenant-scoped: the PII-bearing document layers
# --------------------------------------------------------------------------

def raw_document_dir(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    """``raw-documents/{tenant}/{doc_type}/{source_id}/``

    The only layer holding unredacted PII. It lives in a separate container with
    tighter RBAC, and only ingestion writes it (arch §18a).
    """
    return _join("raw-documents", _tenant(tenant_id), _doc_type(doc_type), source_id)


def raw_pdf(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    """The immutable original. Write-once — a corrected document becomes a NEW
    ``source_id`` so historical training runs stay reproducible against the exact
    bytes they trained on (arch §18a)."""
    return _join(raw_document_dir(doc_type, source_id, tenant_id), "original.pdf")


def raw_metadata(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    return _join(raw_document_dir(doc_type, source_id, tenant_id), "metadata.json")


def processed_dir(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    """``processed/{tenant}/{doc_type}/{source_id}/`` — MinerU output + page renders."""
    return _join("processed", _tenant(tenant_id), _doc_type(doc_type), source_id)


def processed_page(doc_type: str, source_id: str, page: int, suffix: str,
                   tenant_id: str | None = None) -> str:
    """A rendered page or its OCR text. Pages are 1-based and ordered — multi-page
    ordering is positionally meaningful to the model (arch §3)."""
    if page < 1:
        raise PathError(f"page numbers are 1-based, got {page}")
    return _join(processed_dir(doc_type, source_id, tenant_id), f"page_{page}.{suffix.lstrip('.')}")


def ocr_meta(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    """Records ``mineru_version``, ``ocr_device``, resolution cap, and per-page
    table row counts (the last feeds the row-completeness check in SPEC_09)."""
    return _join(processed_dir(doc_type, source_id, tenant_id), "ocr_meta.json")


def golden_label_dir(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    return _join("golden-labels", _tenant(tenant_id), _doc_type(doc_type), source_id)


def golden_label(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    return _join(golden_label_dir(doc_type, source_id, tenant_id), "golden.json")


def label_metadata(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    """Provenance: reviewer, date, draft backend, agreement score, and
    ``field_provenance`` — the observed surface label per canonical field."""
    return _join(golden_label_dir(doc_type, source_id, tenant_id), "label_metadata.json")


def corpus_dir(version: str, tenant_id: str | None = None) -> str:
    return _join("corpus", _tenant(tenant_id), _version(version))


def corpus_split(version: str, doc_type: str, split: str, tenant_id: str | None = None) -> str:
    """``corpus/{tenant}/v{n}/{doc_type}/{split}.jsonl``"""
    if split not in ("train", "val", "test"):
        raise PathError(f"unknown split {split!r}; expected train, val or test")
    return _join(corpus_dir(version, tenant_id), _doc_type(doc_type), f"{split}.jsonl")


def corpus_manifest(version: str, tenant_id: str | None = None) -> str:
    """Pins everything the corpus depends on: MinerU version and device, schema
    and prompt-template versions, LoB and alias coverage, de-identification status."""
    return _join(corpus_dir(version, tenant_id), "manifest.json")


# --------------------------------------------------------------------------
# Shared: models, adapters, registry — no tenant data
# --------------------------------------------------------------------------

def base_model_dir() -> str:
    """Cached from Hugging Face at the pinned revision."""
    return _join("base-models", "qwen3-vl-8b-instruct")


def adapter_dir(kind: AdapterKind, version: str, doc_type: str | None = None) -> str:
    """``adapters/foundation/v{n}/`` or ``adapters/{doc_type}/v{n}/``

    Two lineages only. A per-tenant lineage is **not built** until a broker
    actually requires one (arch §8b).
    """
    tag = _version(version)
    if kind == "foundation":
        if doc_type is not None:
            raise PathError("the Foundation adapter is not per-doc_type — pass doc_type=None")
        return _join("adapters", "foundation", tag)
    if kind == "doc_type":
        if doc_type is None:
            raise PathError("a per-type adapter needs a doc_type")
        return _join("adapters", _doc_type(doc_type), tag)
    raise PathError(f"unknown adapter kind {kind!r}; expected 'foundation' or 'doc_type'")


def merged_model_dir(version: str, doc_type: str | None = None) -> str:
    """``merged-models/{doc_type|unified}/v{n}/`` — fp16/bf16, post ``merge_and_unload``."""
    scope = _doc_type(doc_type, allow_unified=True) if doc_type else UNIFIED
    return _join("merged-models", scope, _version(version))


def quantized_model_dir(version: str, fmt: str, doc_type: str | None = None) -> str:
    """``quantized-models/{scope}/v{n}/gguf/{format}/`` — one subfolder per format."""
    valid = {"fp16", "bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m"}
    fmt = fmt.strip().lower()
    if fmt not in valid:
        raise PathError(f"unknown quantization format {fmt!r}; expected one of {sorted(valid)}")
    scope = _doc_type(doc_type, allow_unified=True) if doc_type else UNIFIED
    return _join("quantized-models", scope, _version(version), "gguf", fmt)


def run_manifest(run_id: str, run_type: str, doc_type: str | None = None) -> str:
    """``registry/foundation/{run_id}/run_manifest.json`` or
    ``registry/adapters/{doc_type}/{run_id}/run_manifest.json``

    Written even when the weights are only staged, so a reclaimed volume never
    means a training run that happened and left no trace (master §12a).
    """
    if run_type == "foundation":
        return _join("registry", "foundation", run_id, "run_manifest.json")
    if run_type == "per_type_adapter":
        if doc_type is None:
            raise PathError("a per-type adapter manifest needs a doc_type")
        return _join("registry", "adapters", _doc_type(doc_type), run_id, "run_manifest.json")
    raise PathError(f"unknown run_type {run_type!r}; expected 'foundation' or 'per_type_adapter'")


def registry_index() -> str:
    """Flat table of every run — status and key metrics at a glance, without
    opening individual manifests."""
    return _join("registry", "registry_index.json")


def calibration_params(version: str, doc_type: str) -> str:
    """Fitted per model version AND doc type; never reused across versions."""
    return _join("calibration", _version(version), f"{_doc_type(doc_type)}.json")


def eval_report(version: str, doc_type: str | None = None) -> str:
    base = _join("eval-reports", _version(version))
    return _join(base, _doc_type(doc_type), "report.json") if doc_type else _join(base, "summary.json")


def golden_eval_set_dir() -> str:
    """Frozen and versioned separately, held constant across corpus versions so
    model versions stay comparable (arch §8). Never trained on."""
    return "golden-eval-set"


# --------------------------------------------------------------------------
# RunPod staging volume — working storage, NOT the registry
# --------------------------------------------------------------------------

def staging_root() -> str:
    """Absolute filesystem path to the staging area on the mounted volume.

    Absolute on purpose: ``_join`` strips leading slashes because blob keys have
    none, but this is a real mount point. A relative path here would write to
    whatever the working directory happens to be and silently miss the volume.
    """
    mount = os.environ.get("RUNPOD_VOLUME_MOUNT", "/runpod-volume")
    return "/" + _join(mount, "staging") if mount.startswith("/") else _join(mount, "staging")


def _under_staging(*parts: str) -> str:
    root = staging_root()
    tail = _join(*parts)
    return f"{root.rstrip('/')}/{tail}"


def staging_adapter_dir(kind: AdapterKind, version: str, doc_type: str | None = None) -> str:
    """Mirrors :func:`adapter_dir` so ``package`` copies rather than translates."""
    return _under_staging(adapter_dir(kind, version, doc_type))


def staging_merged_model_dir(version: str, doc_type: str | None = None) -> str:
    return _under_staging(merged_model_dir(version, doc_type))


def staging_quantized_model_dir(version: str, fmt: str, doc_type: str | None = None) -> str:
    return _under_staging(quantized_model_dir(version, fmt, doc_type))


def staging_eval_report(version: str, doc_type: str | None = None) -> str:
    return _under_staging(eval_report(version, doc_type))


def staging_run_manifest(run_id: str) -> str:
    """A working copy. The durable one always goes to Blob (master §12a)."""
    return _under_staging("run_manifests", f"{run_id}.json")


# --------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------

def is_tenant_scoped(blob_path: str) -> bool:
    """Whether a path's top-level prefix carries tenant data."""
    return str(blob_path).strip("/").split("/", 1)[0] in TENANT_SCOPED


def tenant_of(blob_path: str) -> str | None:
    """Extract the tenant from a tenant-scoped path, else ``None``.

    Used to assert a corpus file never mixes tenants — the one live tenancy rule.
    """
    parts = str(blob_path).strip("/").split("/")
    return parts[1] if len(parts) >= 2 and parts[0] in TENANT_SCOPED else None


def requires_raw_container(blob_path: str) -> bool:
    """Whether a path belongs in the separately-permissioned raw container.

    ``raw-documents/`` is the only layer with unredacted PII, so it is reachable
    only from ingestion and OCR — never from training or serving (arch §18a).
    """
    # The bare prefix counts. Stripping the trailing slash before testing for
    # one classified "raw-documents" itself as non-raw, so `list("raw-documents")`
    # from a training context bypassed the AccessDeniedError guard entirely.
    normalised = str(blob_path).strip("/")
    return normalised == "raw-documents" or normalised.startswith("raw-documents/")
