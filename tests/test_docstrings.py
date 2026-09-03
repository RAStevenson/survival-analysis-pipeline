"""Every module, class, and function in the product code carries a docstring.

Project rule, 2026-09-02: a docstring on everything, however brief, because
a rule with a triviality exemption invites a judgement call every time and
is the one that gets neglected. Nested functions count. Tests are exempt,
since a test's name is its description.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRODUCT = sorted((REPO / "src" / "survival_analysis_pipeline").glob("*.py")) + sorted(
    (REPO / "scripts").glob("*.py")
)


def _bare(path: Path) -> list[str]:
    """Names of every definition in the file, the module included, with no docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    if ast.get_docstring(tree) is None:
        out.append("<module>")
    for node in ast.walk(tree):
        is_def = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if is_def and ast.get_docstring(node) is None:
            out.append(f"{node.name} (line {node.lineno})")
    return out


def test_every_definition_has_a_docstring() -> None:
    """Fail naming every bare definition, so the fix is a list and not a hunt."""
    assert PRODUCT, "no product files found"
    offenders = {p.name: _bare(p) for p in PRODUCT if _bare(p)}
    assert not offenders, "definitions without a docstring:\n" + "\n".join(
        f"  {f}: {', '.join(names)}" for f, names in offenders.items()
    )
