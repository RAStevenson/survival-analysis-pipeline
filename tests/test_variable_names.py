"""Every name bound in src/, scripts/, and tests/ says what it holds:
no one- or two-character variable, parameter, loop variable, exception
name, or instance attribute outside a closed list of conventions."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "scripts", "tests")
# X and y: a model's input matrix and target. ax: one matplotlib axes.
# i, j: an integer loop counter. Nothing else is allowed short.
ALLOWED_SHORT = {"X", "y", "ax", "i", "j"}


def _bound_names(tree):
    """Yield (line, name) for every name the tree binds: assignment and
    loop targets, function and lambda arguments, exception names, and
    attributes assigned on self."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.lineno, node.id
        elif isinstance(node, ast.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.ExceptHandler) and node.name:
            yield node.lineno, node.name
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            yield node.lineno, node.attr


def _short_names(path):
    """Return one 'file:line name' string per short name bound in the
    file at path that is not on the allowed list."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    relative_path = path.relative_to(ROOT).as_posix()
    flagged = set()
    for line, name in _bound_names(tree):
        stripped_name = name.strip("_")
        if stripped_name and len(stripped_name) <= 2 and stripped_name not in ALLOWED_SHORT:
            flagged.add((line, name))
    return [f"{relative_path}:{line} {name}" for line, name in sorted(flagged)]


def test_every_name_says_what_it_holds():
    flagged = [
        entry
        for folder in SOURCE_DIRS
        for path in sorted((ROOT / folder).rglob("*.py"))
        for entry in _short_names(path)
    ]
    assert not flagged, "short names (python-repo-layout, variable names):\n  " + "\n  ".join(
        flagged
    )
