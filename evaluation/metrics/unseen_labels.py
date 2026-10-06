"""Accuracy on printed labels the training data never showed (common-model lines).

Aliases - the labels carriers print for a field - are never in the prompt
(master §1.4): the model is meant to learn what a field MEANS, from its
description and from documents that print it many ways. Whether it did is
measured here, by splitting field accuracy into values whose printed label the
training documents showed and values whose label they never did. A model that
learned the lookup table rather than the meaning does well on the first and
badly on the second.

Reported, not gated: an eval set's unseen labels are few, and which they are
depends on the carriers it happens to hold.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def common_model_aliases(lob: str) -> dict[str, list[str]]:
    """Field path (no list markers) -> the labels the SPEC_21 sources record.

    From the common model's per-field aliases, the overlay's own, and each
    coverage code's (as ``coverages.coverage_code=<CODE>``).
    """
    import json

    from common.config import load_yaml
    from common.schemas import CANONICAL_DIR, _common_model, load_schema, resolve_local

    bundle = load_schema("policy", None, lob)
    defs = bundle.get("$defs") or {}
    out: dict[str, list[str]] = {}

    def add(path: str, labels: Iterable[Any]) -> None:
        bucket = out.setdefault(path, [])
        for label in labels:
            if isinstance(label, str) and label not in bucket:
                bucket.append(label)

    def walk(node: Any, path: str, seen: frozenset[str]) -> None:
        for name, sub in ((node or {}).get("properties") or {}).items():
            here = f"{path}.{name}" if path else name
            if isinstance(sub, dict) and isinstance(sub.get("fideon:aliases"), list):
                add(here, sub["fideon:aliases"])
            ref = sub.get("$ref", "") if isinstance(sub, dict) else ""
            target = ref.rsplit("/", 1)[-1] if ref else ""
            if target in seen or target.endswith("Value"):
                continue
            resolved = resolve_local(sub, defs)
            if not isinstance(resolved, dict):
                continue
            child = resolve_local(resolved.get("items"), defs) if "items" in resolved else resolved
            if isinstance(child, dict) and child.get("properties"):
                walk(child, here, seen | ({target} if target else set()))

    walk(bundle, "", frozenset())
    overlay = json.loads((CANONICAL_DIR / f"{lob}.json").read_text(encoding="utf-8"))
    for path, labels in (overlay.get("fideon:aliases") or {}).items():
        add(re.sub(r"\[\d*\]", "", path), labels)
    code_file = overlay.get("fideon:coverage_codes_file")
    if code_file:
        for entry in load_yaml(CANONICAL_DIR / code_file).get("codes") or []:
            add(f"coverages.coverage_code={entry['code']}", entry.get("aliases") or [])
    shared = _common_model().get("fideon:shared_coverage_codes") or {}
    for code in overlay.get("fideon:coverage_codes") or []:
        add(f"coverages.coverage_code={code}", (shared.get(code) or {}).get("aliases") or [])
    return out


def normalise(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def printed(label: str, text: str) -> bool:
    """Whether ``label`` is printed in ``text`` (normalised), as a whole phrase."""
    needle = normalise(label)
    if len(needle) < 3:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text) is not None


def label_split(
    results: Iterable[Any], ocr_text: str, aliases: dict[str, list[str]], seen: set[str],
) -> dict[str, list[bool]]:
    """Field results split by whether their printed label was seen in training.

    A field counts only when the document prints at least one of its labels:
    ``seen`` when one of those was seen in training, ``unseen`` when none was.
    """
    text = normalise(ocr_text)
    seen_normalised = {normalise(label) for label in seen}
    out: dict[str, list[bool]] = {"seen": [], "unseen": []}
    for result in results:
        path = re.sub(r"\[\d*\]", "", getattr(result, "field_path", ""))
        on_page = [label for label in aliases.get(path, []) if printed(label, text)]
        if not on_page:
            continue
        bucket = "seen" if any(normalise(label) in seen_normalised for label in on_page) else "unseen"
        out[bucket].append(bool(getattr(result, "correct", False)))
    return out
