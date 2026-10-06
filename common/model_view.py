"""A common-model (SPEC_21) line schema as the extraction model sees it.

The bundle a line overlay loads as (``common.schemas._bundle``) is the client's
contract: every value an envelope with ``confidence`` and ``flagged``, rules the
decoder cannot enforce, fields the pipeline fills, notes for schema authors.
Validation judges a served answer against all of it. The model is shown, and
held to, something narrower:

* every value as ``raw`` / ``parsed`` / ``page_ref`` - confidence and the review
  flag are the pipeline's to fill, from the logprobs and the thresholds;
* no field the pipeline fills itself (the document's type, modality, page count
  and file name, whether the policy is a package, a form's page range, a
  coverage's id) and no full-text tier;
* the rules the decoder cannot enforce (``if``/``then``, ``dependentRequired``)
  rewritten as ``anyOf`` variants it can, and the keywords xgrammar refuses
  (``minProperties``, ``uniqueItems``, ...) dropped;
* the line's coverage codes, each with its meaning, as the only values a code
  field takes;
* descriptions written for the model where the client's are written for schema
  authors (``configs/model_view.yaml``).

Rendered into the prompt and handed to structured decoding alike, so the model is
both told and constrained the same thing.
"""

from __future__ import annotations

import copy
from functools import cache
from typing import Any

from common.config import CONFIG_DIR, load_yaml

#: Repo-owned overrides of the model view. The client's schema folder is theirs
#: and never edited, so a description written for schema authors is replaced
#: here, until the client's own text is rewritten for the model.
MODEL_VIEW_CONFIG = CONFIG_DIR / "model_view.yaml"

#: Fields the pipeline fills, never the model, per common-model definition.
#: Shown, they would teach the model to write values no page prints.
PIPELINE_FILLED: dict[str, tuple[str, ...]] = {
    "document": ("doc_type", "modality", "page_count", "source_file_name"),
    "policy": ("is_package",),
    "Form": ("page_range",),
    "Coverage": ("coverage_id",),
}

#: Fields that hold a coverage code, per definition. Each is held to the line's
#: list (``CoverageCode``).
CODE_FIELDS: dict[str, tuple[str, ...]] = {
    "Coverage": ("coverage_code",),
    "Limit": ("basis_coverage_code",),
    "PremiumItem": ("coverage_code",),
    "UnderlyingPolicy": ("coverage_code",),
}
CODE_LIST_FIELDS: dict[str, tuple[str, ...]] = {"Deductible": ("applies_to_coverages",)}

COVERAGE_CODE_DEF = "CoverageCode"

#: Keywords the decoder cannot use: xgrammar refuses the first four outright
#: (vLLM's has_xgrammar_unsupported_json_features), and the rest are either
#: rewritten as variants first or carry nothing the model acts on.
_DROPPED_KEYWORDS = frozenset({
    "minProperties", "maxProperties", "uniqueItems", "propertyNames", "patternProperties",
    "format", "default", "allOf", "if", "then", "else", "dependentRequired",
})


class ModelViewError(RuntimeError):
    """Raised when a bundle cannot be turned into the model's view of it."""


@cache
def model_view_config() -> dict[str, Any]:
    return load_yaml(MODEL_VIEW_CONFIG)


def model_view(bundle: dict[str, Any]) -> dict[str, Any]:
    """``bundle`` (whole or sliced) as the model is shown and constrained to it."""
    config = model_view_config()
    schema = copy.deepcopy(bundle)
    defs: dict[str, Any] = schema.setdefault("$defs", {})
    names: dict[str, str] = dict(schema.get("fideon:coverage_code_names") or {})
    lob = schema.get("fideon:lob")

    _drop_top_level(schema)
    for name, fields in PIPELINE_FILLED.items():
        _drop_fields(defs.get(name), fields)
    for name, defn in list(defs.items()):
        if _is_value_type(defn):
            defs[name] = _narrowed_value(defn)
    if names:
        defs[COVERAGE_CODE_DEF] = _coverage_code_def(names, config.get("coverage_code_style", "oneof"))
        _point_code_fields(defs)
    if lob and "LobPart" in defs and "lob" in defs["LobPart"].get("properties", {}):
        defs["LobPart"]["properties"]["lob"] = {"const": lob, "description": "Line of business of the part."}
    _apply_descriptions(defs, config.get("descriptions") or {})
    if lob:
        _apply_id_patterns(schema, defs, lob)
    for name, defn in list(defs.items()):
        variants = _as_variants(defn, defs)
        if variants is not None:
            defs[name] = variants
    if isinstance(schema.get("required"), list):
        # A required LIST would force an empty one into every window without it;
        # objects stay required and are written as {} when nothing in them is stated.
        schema["required"] = [r for r in schema["required"] if not _is_array_node(
            (schema.get("properties") or {}).get(r), defs)]
        if not schema["required"]:
            del schema["required"]
    _relax_required_lists(defs)

    cleaned = _without_keywords(schema)
    cleaned = {k: v for k, v in cleaned.items() if not k.startswith("fideon:")}
    return _strip_fideon(_pruned(cleaned))


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def _drop_top_level(schema: dict[str, Any]) -> None:
    """The file's own title and description, the full-text tier, and the
    ``fideon:`` properties (provenance): none is the model's to write."""
    schema.pop("title", None)
    schema.pop("description", None)
    properties = schema.get("properties") or {}
    for name in [n for n in properties if n == "text_sections" or n.startswith("fideon:")]:
        del properties[name]
    if isinstance(schema.get("required"), list):
        schema["required"] = [r for r in schema["required"] if r in properties]


def _drop_fields(defn: dict[str, Any] | None, fields: tuple[str, ...]) -> None:
    if not isinstance(defn, dict):
        return
    properties = defn.get("properties") or {}
    for field in fields:
        properties.pop(field, None)
    if isinstance(defn.get("required"), list):
        defn["required"] = [r for r in defn["required"] if r not in fields]


def _is_value_type(defn: Any) -> bool:
    return isinstance(defn, dict) and {"raw", "parsed", "page_ref"} <= set(defn.get("properties") or {})


def _narrowed_value(defn: dict[str, Any]) -> dict[str, Any]:
    """A value as the model writes it. ``raw`` is never null: a value the
    document does not state is omitted, not written empty."""
    out: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["raw", "parsed", "page_ref"],
        "properties": {
            "raw": {"type": "string"},
            "parsed": copy.deepcopy(defn["properties"]["parsed"]),
            "page_ref": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1},
        },
    }
    if defn.get("description"):
        out["description"] = defn["description"]
    return out


def _coverage_code_def(names: dict[str, str], style: str) -> dict[str, Any]:
    if style == "enum":
        return {
            "type": "string",
            "enum": list(names),
            "description": "; ".join(f"{code}: {name}" for code, name in names.items()),
        }
    if style != "oneof":
        raise ModelViewError(f"coverage_code_style must be 'oneof' or 'enum', not {style!r}")
    return {"type": "string", "oneOf": [{"const": c, "description": n} for c, n in names.items()]}


def _point_code_fields(defs: dict[str, Any]) -> None:
    ref = {"$ref": f"#/$defs/{COVERAGE_CODE_DEF}"}
    for name, fields in CODE_FIELDS.items():
        properties = (defs.get(name) or {}).get("properties") or {}
        for field in fields:
            if field in properties:
                properties[field] = {**ref, **_described(properties[field])}
    for name, fields in CODE_LIST_FIELDS.items():
        properties = (defs.get(name) or {}).get("properties") or {}
        for field in fields:
            if field in properties:
                properties[field] = {"type": "array", "items": dict(ref), **_described(properties[field])}


def _described(node: dict[str, Any]) -> dict[str, Any]:
    return {"description": node["description"]} if node.get("description") else {}


def _apply_descriptions(defs: dict[str, Any], overrides: dict[str, str | None]) -> None:
    """``Def`` or ``Def.field`` -> the description the model reads (None: none)."""
    for path, text in overrides.items():
        node = _description_target(defs, path)
        if node is None:
            continue  # this bundle (or slice) does not carry it
        if text is None:
            node.pop("description", None)
        else:
            node["description"] = text


def _description_target(defs: dict[str, Any], path: str) -> dict[str, Any] | None:
    name, _, field = path.partition(".")
    node = defs.get(name)
    if not isinstance(node, dict):
        return None
    if not field:
        return node
    target = (node.get("properties") or {}).get(field)
    return target if isinstance(target, dict) else None


def _apply_id_patterns(schema: dict[str, Any], defs: dict[str, Any], lob: str) -> None:
    """Each table's structural id held to its own prefix (``veh_2``, not ``loc_2``),
    from the section map's ``ids``. The common model allows any ``word_N``."""
    from common.schema_sections import structural_ids

    for table, spec in structural_ids(lob).items():
        block = _resolved((schema.get("properties") or {}).get(table), defs)
        row = _resolved(block.get("items"), defs) if block.get("type") == "array" else {}
        field = (row.get("properties") or {}).get(spec["field"])
        if isinstance(field, dict):
            field["pattern"] = f"^{spec['prefix']}_[0-9]+$"


def _as_variants(defn: Any, defs: dict[str, Any]) -> dict[str, Any] | None:
    """A definition whose rules the decoder cannot enforce, as ``anyOf`` variants.

    Two shapes occur in the common model, and both are rewritten:

    * ``if <field> == V then require R`` (Limit: a sublimit names what it limits;
      Deductible: a flat one has an amount, a percentage one a percentage) - one
      variant per value named, with that value as a ``const`` and R required, and
      one for every other value;
    * ``dependentRequired`` (Limit: a percentage names the coverage it is a
      percentage of) - each variant split in two, the property present with its
      dependants required, or absent altogether.

    Every object is closed (``additionalProperties: false``), so a property left
    out of a variant cannot be written in it. Descriptions are kept on the first
    variant only: the model reads them once.
    """
    if not isinstance(defn, dict) or "properties" not in defn:
        return None
    rules = [r for r in defn.get("allOf") or [] if isinstance(r, dict) and "if" in r]
    dependents: dict[str, list[str]] = dict(defn.get("dependentRequired") or {})
    if not rules and not dependents:
        return None

    properties: dict[str, Any] = defn["properties"]
    branches: list[tuple[dict[str, Any], list[str]]] = [({}, [])]
    if rules:
        cases = []
        for rule in rules:
            condition = (rule["if"].get("properties") or {})
            if len(condition) != 1:
                raise ModelViewError(f"cannot rewrite a rule on {sorted(condition)} as variants")
            ((field, spec),) = condition.items()
            if "const" not in spec:
                raise ModelViewError(f"cannot rewrite a rule on {field} that is not a const")
            cases.append((field, spec["const"], list((rule.get("then") or {}).get("required") or [])))
        fields = {field for field, _, _ in cases}
        if len(fields) != 1:
            raise ModelViewError(f"rules switch on several fields {sorted(fields)}")
        (field,) = fields
        values = _enum_values(properties[field], defs)
        description = _described(_resolved(properties[field], defs))
        branches = [({field: {"const": value, **description}}, extra) for _, value, extra in cases]
        rest = [v for v in values if v not in {value for _, value, _ in cases}]
        if rest:
            branches.append(({field: {"enum": rest, **description}}, []))
    for prop, needs in dependents.items():
        split: list[tuple[dict[str, Any], list[str]]] = []
        for override, extra in branches:
            split.append((dict(override), [*extra, prop, *needs]))
            split.append(({**override, prop: None}, list(extra)))
        branches = split

    base_required = list(defn.get("required") or [])
    variants = []
    for index, (override, extra) in enumerate(branches):
        shown: dict[str, Any] = {}
        for name, sub in properties.items():
            if name in override:
                if override[name] is None:
                    continue
                sub = override[name]
            shown[name] = sub if index == 0 else {k: v for k, v in sub.items() if k != "description"}
        required = list(dict.fromkeys(r for r in [*base_required, *extra] if r in shown))
        variants.append({"type": "object", "additionalProperties": False,
                         "properties": shown, "required": required})
    kept = {k: v for k, v in defn.items() if k in ("description",)}
    return {**kept, "anyOf": variants}


def _enum_values(node: Any, defs: dict[str, Any]) -> list[Any]:
    values = _resolved(node, defs).get("enum")
    if not isinstance(values, list):
        raise ModelViewError("a rule's field must take a fixed list of values")
    return values


def _resolved(node: Any, defs: dict[str, Any]) -> dict[str, Any]:
    seen = 0
    while isinstance(node, dict) and "$ref" in node and seen < 20:
        node = defs.get(str(node["$ref"]).rsplit("/", 1)[-1], {})
        seen += 1
    return node if isinstance(node, dict) else {}


def _is_array_node(node: Any, defs: dict[str, Any]) -> bool:
    return _resolved(node, defs).get("type") == "array" if isinstance(node, dict) else False


def _relax_required_lists(defs: dict[str, Any]) -> None:
    """A top-level block that is itself a list (``lob_parts``) loses its minimum:
    the pipeline supplies the part when the model reads none."""
    lob_parts = defs.get("lob_parts")
    if isinstance(lob_parts, dict):
        lob_parts.pop("minItems", None)


def _without_keywords(node: Any) -> Any:
    if isinstance(node, list):
        return [_without_keywords(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key in _DROPPED_KEYWORDS:
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            # Names here are field or definition names, never keywords: a field
            # called "default" or "format" must survive.
            out[key] = {name: _without_keywords(sub) for name, sub in value.items()}
        else:
            out[key] = _without_keywords(value)
    return out


def _strip_fideon(node: Any) -> Any:
    if isinstance(node, list):
        return [_strip_fideon(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {k: _strip_fideon(v) for k, v in node.items() if not str(k).startswith("fideon:")}


def _pruned(schema: dict[str, Any]) -> dict[str, Any]:
    """Only the definitions the schema's properties reach, transitively."""
    from common.schemas import _local_ref_names

    defs = schema.get("$defs") or {}
    reached = _local_ref_names(schema.get("properties") or {})
    pending = sorted(reached)
    while pending:
        for name in _local_ref_names(defs.get(pending.pop(), {})):
            if name not in reached:
                reached.add(name)
                pending.append(name)
    return {**schema, "$defs": {k: v for k, v in defs.items() if k in reached}}
