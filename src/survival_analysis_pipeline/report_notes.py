"""Per-run authored notes, inserted into the generated report at fixed anchors.

The report template states measurements; interpretation, motivation, and any
dataset-specific claim live in a notes directory beside the run, one markdown
file per anchor. An absent directory or file inserts nothing. Notes may cite
metric values through @val{path} tokens resolved from the run's metrics at
build time, so authored prose can quote numbers with zero drift risk; an
unresolvable token fails the build, the same contract as an uncited figure.

Every anchor is named for the report section it lands in, so a stranger can
tell from the filename where the prose will appear:

  motivation.md      creates its own section after the summary
  data.md            appended to the data section
  interpretation.md  creates its own section after the attribution section
  limitations.md     appended to the limitations section

Token grammar: @val{pooled.c_xgb:.3f}. The path walks the metrics dict with
dots, integer segments index into lists (folds.0.c_xgb), and the part after
the colon is a Python format spec. Floats require an explicit format spec so
no note silently prints fifteen digits; everything else defaults to str().

Markdown support is deliberately minimal, because notes are prose: blank-line
paragraphs, `code`, and **bold**. Headings are refused, since section headings
belong to the template.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

SECTION_ANCHORS = ("motivation", "interpretation")
APPEND_ANCHORS = ("data", "limitations")
KNOWN_ANCHORS = frozenset(SECTION_ANCHORS + APPEND_ANCHORS)

_TOKEN_RE = re.compile(r"@val\{([^{}:]+)(?::([^{}]+))?\}")
_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


def _lookup(values: dict, path: str) -> object:
    """Walk a dotted path into the metrics, indexing lists by integer, and fail naming the segment
    that does not resolve.
    """
    node: object = values
    for part in path.split("."):
        if isinstance(node, list):
            if not part.isdigit() or int(part) >= len(node):
                raise ValueError(f"@val{{{path}}}: {part!r} does not index the list at that point")
            node = node[int(part)]
        elif isinstance(node, dict):
            if part not in node:
                raise ValueError(f"@val{{{path}}}: no key {part!r} in the metrics")
            node = node[part]
        else:
            raise ValueError(f"@val{{{path}}}: {part!r} descends into a scalar")
    return node


def resolve_tokens(text: str, values: dict) -> str:
    """Replace every @val token in a note with its value from the metrics, formatted as the token
    asks; floats without a format spec are refused.
    """

    def sub(match: re.Match[str]) -> str:
        """Resolve one token match."""
        path, fmt = match.group(1), match.group(2)
        value = _lookup(values, path)
        if fmt:
            return format(value, fmt)
        if isinstance(value, float):
            raise ValueError(
                f"@val{{{path}}} is a float and needs an explicit format, e.g. @val{{{path}:.3f}}"
            )
        return str(value)

    return _TOKEN_RE.sub(sub, text)


def _render_paragraphs(text: str) -> list[str]:
    """Split note text into paragraphs and render each as escaped HTML with code and bold marks;
    headings are refused.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    rendered = []
    for p in paragraphs:
        if p.lstrip().startswith("#"):
            raise ValueError("notes carry prose, not headings; the template owns the headings")
        p = escape(p, quote=False)
        p = _CODE_RE.sub(r"<code>\1</code>", p)
        p = _BOLD_RE.sub(r"<strong>\1</strong>", p)
        rendered.append(" ".join(p.split()))
    return rendered


def load_run_notes(notes_dir: Path | None, values: dict) -> dict[str, str]:
    """Read every anchor file in notes_dir, resolve tokens against values,
    and return anchor -> HTML. Each comes back as <p> blocks wrapped in HTML
    comment markers so the invariance test can strip them."""
    if notes_dir is None or not notes_dir.is_dir():
        return {}
    notes: dict[str, str] = {}
    for path in sorted(notes_dir.glob("*.md")):
        anchor = path.stem
        if anchor not in KNOWN_ANCHORS:
            raise ValueError(
                f"{path.name}: unknown notes anchor {anchor!r}; known anchors are "
                f"{', '.join(sorted(KNOWN_ANCHORS))}. Name the file for the report "
                "section it lands in."
            )
        paragraphs = _render_paragraphs(resolve_tokens(path.read_text(encoding="utf-8"), values))
        if not paragraphs:
            continue
        body = "\n\n".join(f"<p>{p}</p>" for p in paragraphs)
        notes[anchor] = f"<!--note:{anchor}-->\n{body}\n<!--/note:{anchor}-->"
    return notes
