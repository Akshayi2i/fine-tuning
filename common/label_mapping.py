"""Labels written for one line's schema, used under another's (configs/label_mappings.yaml).

A line whose labels keep their data in another line's block is moved into its
own block, with fields renamed, before training targets are built and before
gold labels are scored. The stored labels are never changed. No mapping is
configured today: classic auto, the one line that needed it, is now read as
personal auto itself (common.lob.MERGED_LINES).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "label_mappings.yaml"


@cache
def _mappings() -> dict[str, Any]:
    if not CONFIG.exists():
        return {}
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}


def _lines(lob: Any) -> list[str]:
    if isinstance(lob, (list, tuple)):
        return [str(x) for x in lob]
    return [str(lob)] if lob else []


@cache
def _target_fields(lob: str, block: str, section: str) -> frozenset[str] | None:
    """The fields the target schema defines for ``block.section`` (rows' fields
    for a list); None when the schema does not say."""
    from common.schemas import resolved_schema

    schema = resolved_schema("policy", None, lob)
    defs = schema.get("$defs") or {}

    def resolve(node: Any) -> dict[str, Any]:
        while isinstance(node, dict) and "$ref" in node:
            node = defs.get(node["$ref"].rsplit("/", 1)[-1])
        return node if isinstance(node, dict) else {}

    node = resolve((schema.get("properties") or {}).get(block))
    node = resolve((node.get("properties") or {}).get(section))
    if node.get("type") == "array":
        node = resolve(node.get("items"))
    props = node.get("properties")
    return frozenset(props) if props else None


def _move(value: Any, rename: dict[str, str], keep: frozenset[str] | None) -> Any:
    """One section's value with its fields renamed and narrowed to ``keep``."""
    def one(row: Any) -> Any:
        if not isinstance(row, dict):
            return row
        moved = {rename.get(k, k): v for k, v in row.items()}
        return {k: v for k, v in moved.items() if keep is None or k in keep}

    if isinstance(value, list):
        return [one(row) for row in value]
    return one(value)


def map_label(label: Any, lob: Any) -> Any:
    """``label`` with any configured block moved into its line's own block.

    A copy when anything moves; ``label`` itself otherwise. Applies only when
    the label has the source block and not the target one.
    """
    if not isinstance(label, dict):
        return label
    for line in _lines(lob):
        mapping = _mappings().get(line)
        if not mapping:
            continue
        source, target = mapping["from"], mapping["to"]
        if not label.get(source) or label.get(target):
            continue
        moved: dict[str, Any] = {}
        for section, spec in (mapping.get("sections") or {}).items():
            value = label[source].get(section)
            if not value:
                continue
            to = spec.get("to", section)
            moved[to] = _move(value, spec.get("rename") or {}, _target_fields(line, target, to))
        mapped = {k: v for k, v in label.items() if k != source}
        if moved:
            mapped[target] = moved
        return mapped
    return label
