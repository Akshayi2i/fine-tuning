"""The master §1.4 anti-pattern, enforced by an import check rather than by discipline.

The alias registry is a **corpus-analysis artifact**. It slices evaluation ("did
the model get `insured_name` right when the document said 'Applicant'?") and it
tells you where coverage is thin. It never appears in a prompt and it never runs
at inference time.

The reason is the whole point of the canonical-mapping design: if a lookup table
maps "Applicant" to ``insured_name`` at runtime, the model never has to learn the
semantics, and the first unseen surface label — of which insurance documents have
an unbounded supply — produces a miss with no signal that anything went wrong.
The registry would be doing the model's job, badly, and hiding that it was.

Discipline is not a control. An import is, and it is checked here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: The runtime path: what runs when a document is extracted. ``data_pipeline``
#: and ``evaluation`` are deliberately absent — deriving and reporting on aliases
#: is exactly what those are for.
RUNTIME_PACKAGES = ("serving", "inference_core", "testing")

FORBIDDEN_MODULE = "common.aliases"


def imports_the_alias_registry(source: str, filename: str = "<memory>") -> list[str]:
    """Every import in ``source`` that reaches the alias registry.

    AST rather than a substring search, so a docstring explaining the rule (this
    file, for one) is not mistaken for a violation.
    """
    found: list[str] = []
    tree = ast.parse(source, filename=filename)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == FORBIDDEN_MODULE or alias.name.startswith(FORBIDDEN_MODULE + "."):
                    found.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == FORBIDDEN_MODULE:
                found.append(f"from {module} import {', '.join(a.name for a in node.names)}")
            elif module == "common" and any(a.name == "aliases" for a in node.names):
                found.append("from common import aliases")
    return found


def runtime_modules() -> list[Path]:
    return sorted(
        path
        for package in RUNTIME_PACKAGES
        for path in (ROOT / package).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_the_runtime_packages_exist():
    """A check that scans nothing passes trivially."""
    modules = runtime_modules()
    assert len(modules) >= 8, f"only found {len(modules)} runtime modules — the scan is not reaching them"


@pytest.mark.parametrize("module_path", runtime_modules(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_runtime_module_imports_the_alias_registry(module_path: Path):
    violations = imports_the_alias_registry(
        module_path.read_text(encoding="utf-8"), str(module_path)
    )
    assert not violations, (
        f"{module_path.relative_to(ROOT)} imports the alias registry: {violations}. "
        "The registry is a corpus-analysis artifact — using it at runtime means a lookup table "
        "is doing the semantic mapping the model was trained to do, and the first unseen surface "
        "label then misses silently (master §1.4)."
    )


def test_the_check_actually_detects_a_violation():
    """Without this the guard could be scanning nothing and still passing."""
    assert imports_the_alias_registry("from common.aliases import canonical_for")
    assert imports_the_alias_registry("from common import aliases")
    assert imports_the_alias_registry("import common.aliases")
    assert imports_the_alias_registry("import common.aliases as a")


def test_the_check_does_not_fire_on_prose_or_neighbouring_modules():
    assert not imports_the_alias_registry('"""We never import common.aliases here."""')
    assert not imports_the_alias_registry("from common import schemas, normalize")
    assert not imports_the_alias_registry("from data_pipeline.labeling import derive_aliases")
