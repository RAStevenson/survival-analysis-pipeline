"""Every script in scripts/ does its work inside main() and runs it only
behind the __main__ guard, so importing a script runs nothing, and main()
parses its arguments, so --help prints help instead of running the
script. The module level may hold only the docstring, the path
bootstrap, imports, ALL_CAPS constants, and function definitions."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_guard(node: ast.stmt) -> bool:
    """Whether node is `if __name__ == "__main__": main()` and nothing else."""
    return (
        isinstance(node, ast.If)
        and ast.unparse(node.test) == "__name__ == '__main__'"
        and len(node.body) == 1
        and ast.unparse(node.body[0]) == "main()"
        and not node.orelse
    )


def _is_allowed_module_statement(node: ast.stmt) -> bool:
    """Whether a module-level statement is one of the allowed kinds: the
    docstring, an import, the sys.path bootstrap, an ALL_CAPS constant,
    or a function definition."""
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return True
    if isinstance(node, ast.Expr) and ast.unparse(node.value).startswith("sys.path.insert("):
        return True
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return all(isinstance(target, ast.Name) and target.id.isupper() for target in targets)
    return False


def _calls_parse_args(function: ast.FunctionDef) -> bool:
    """Whether the function body calls something named parse_args, as code rather than text."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parse_args"
        for node in ast.walk(function)
    )


def _entry_problems(path: Path) -> list[str]:
    """Return one 'file: problem' string per way the script at path
    breaks the main() rule."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    relative_path = path.relative_to(ROOT).as_posix()
    problems = []
    main_functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    if not main_functions:
        problems.append(f"{relative_path}: no main() function")
    elif not _calls_parse_args(main_functions[0]):
        # Without a parser, --help and mistyped flags are ignored and the
        # script does its full work.
        problems.append(f"{relative_path}: main() parses no arguments")
    if not tree.body or not _is_guard(tree.body[-1]):
        problems.append(f"{relative_path}: does not end with the __main__ guard")
    for node in tree.body[:-1]:
        if not _is_allowed_module_statement(node):
            problems.append(
                f"{relative_path}:{node.lineno}: work at module level: "
                f"{ast.unparse(node).splitlines()[0][:70]}"
            )
    return problems


def test_every_script_runs_through_main() -> None:
    problems = [
        problem
        for path in sorted((ROOT / "scripts").glob("*.py"))
        for problem in _entry_problems(path)
    ]
    assert not problems, "scripts not run through main():\n  " + "\n  ".join(problems)
