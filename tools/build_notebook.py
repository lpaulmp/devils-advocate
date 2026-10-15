"""Build the shareable notebook from devils_advocate.py.

The .py keeps its comments and docstrings for review; the notebook drops them from the code cells and
explains itself through its markdown cells. With --execute it runs top to bottom (dry-run, temporary
HOME so nothing is written to the real home folder) and keeps the outputs as the example run.

    python tools/build_notebook.py [--execute]
"""
import argparse
import ast
import io
import os
import re
import sys
import tempfile
import tokenize
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "devils_advocate.py"
TARGET = ROOT / "devils_advocate.ipynb"
KERNELSPEC = {"name": "python3", "display_name": "Python 3", "language": "python"}


def split_cells(text: str) -> list[str]:
    """Split a jupytext percent file at its `# %%` markers (each chunk keeps its marker)."""
    return [c for c in re.split(r"(?m)^(?=# %%)", text) if c]


def strip_code(code: str) -> str:
    """Remove comments and docstrings from Python code; a body left empty gets `pass`."""
    lines = code.splitlines()
    drop: set[int] = set()  # 0-based line numbers
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            span = range(first.lineno - 1, first.end_lineno)
            if len(node.body) == 1:
                indent = re.match(r"\s*", lines[first.lineno - 1]).group()
                lines[first.lineno - 1] = indent + "pass"
                drop.update(span[1:])
            else:
                drop.update(span)
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start[0] - 1, tok.start[1]
            kept = lines[row][:col].rstrip()
            if kept:
                lines[row] = kept
            else:
                drop.add(row)
    out = "\n".join(line for i, line in enumerate(lines) if i not in drop)
    return re.sub(r"\n{3,}", "\n\n", out).strip("\n") + "\n"


def without_docstrings(code: str) -> str:
    """AST dump of the code with docstrings removed: what strip_code must preserve exactly."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def checked_strip(code: str) -> str:
    """strip_code, refusing any result that is not the same program minus its docstrings."""
    stripped = strip_code(code)
    if without_docstrings(stripped) != without_docstrings(code):
        raise ValueError("stripping comments changed the code; check the cell:\n" + code[:200])
    return stripped


def notebook_source(text: str) -> str:
    cells = []
    for cell in split_cells(text):
        marker, _, body = cell.partition("\n")
        if "[markdown]" in marker or not marker.startswith("# %%"):
            cells.append(cell.rstrip("\n") + "\n")
        else:
            cells.append(marker + "\n" + checked_strip(body))
    return "\n\n".join(c.rstrip("\n") for c in cells) + "\n"


def build(execute: bool = False, target: Path = TARGET) -> nbformat.NotebookNode:
    nb = jupytext.reads(notebook_source(SOURCE.read_text(encoding="utf-8")), fmt="py:percent")
    nb.metadata["kernelspec"] = KERNELSPEC
    nb.metadata.pop("jupytext", None)
    if execute:
        run(nb)
    nbformat.write(nb, target)
    return nb


def run(nb: nbformat.NotebookNode, timeout: int = 120):
    """Execute top to bottom in dry-run with a temporary HOME (no keys, nothing written to the real home)."""
    from nbclient import NotebookClient

    saved = {k: os.environ.get(k) for k in ("HOME", "ANTHROPIC_API_KEY", "XAI_API_KEY")}
    with tempfile.TemporaryDirectory() as home:
        os.environ["HOME"] = home
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("XAI_API_KEY", None)
        try:
            NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
            for cell in nb.cells:  # timestamps would make every rebuild a diff
                cell.metadata.pop("execution", None)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true", help="run it top to bottom (dry-run) and keep the outputs")
    args = parser.parse_args()
    build(execute=args.execute)
    print(f"Wrote {TARGET.relative_to(ROOT)}", file=sys.stderr)
