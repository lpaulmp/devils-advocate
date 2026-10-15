"""Notebook (stage 6): built from the .py without comments or docstrings, self-contained, in sync with the
source, and it runs top to bottom in dry-run without keys or writes to the real home folder."""
import ast
import os
import re
import sys

import nbformat
import pytest

import devils_advocate as da
from tools import build_notebook as bn


def test_strip_code_removes_comments_and_docstrings():
    code = '''"""Module doc."""
import os  # trailing comment
# full-line comment
URL = "http://x#not-a-comment"  # but this is
PROMPT = """keep # this
and this"""


def f():
    """Only a docstring."""


class C:
    """Doc."""
    x = 1  # comment
'''
    out = bn.strip_code(code)
    ast.parse(out)  # still valid Python
    assert "#" not in out.replace("http://x#not-a-comment", "").replace("keep # this", "")
    assert "Module doc" not in out and "Only a docstring" not in out and "Doc." not in out
    assert 'URL = "http://x#not-a-comment"' in out and 'PROMPT = """keep # this\nand this"""' in out
    assert "def f():\n    pass" in out and "x = 1\n" in out


@pytest.fixture(scope="module")
def notebook(tmp_path_factory):
    return bn.build(execute=False, target=tmp_path_factory.mktemp("nb") / "devils_advocate.ipynb")


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def output_text(cells) -> str:
    """Plain text of all outputs (stream or display), without terminal colour codes."""
    parts = []
    for c in cells:
        for o in c.get("outputs", []):
            parts.append(o.get("text") or o.get("data", {}).get("text/plain", ""))
    return ANSI.sub("", "".join(parts))


def code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def markdown(nb):
    return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")


def test_code_cells_compile_and_have_no_comments(notebook):
    for cell in code_cells(notebook):
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert ast.get_docstring(node) is None, node.name
        lines = cell.source.splitlines()
        in_template = "CONFIG_TEMPLATE" in cell.source  # TOML comments inside a string are data
        assert in_template or not any(l.lstrip().startswith("#") for l in lines)


def test_markdown_explains_what_the_spec_asks(notebook):
    text = markdown(notebook)
    # The notebook explains itself in Spanish (the owner's delivery language).
    for needle in ("**El problema.**", "## Roles", "## Flujo de información", "## Rondas y poda", "semana 3",
                   "beam search", "A*", "## Control de costos", "## Ejecutar"):
        assert needle in text, needle


def test_variables_cell_replaces_argparse(notebook):
    sources = [c.source for c in code_cells(notebook)]
    variables = next(s for s in sources if s.startswith("IDEA ="))
    namespace = {}
    exec(variables, namespace)
    assert {"IDEA", "ROUNDS", "BUDGET", "DRY_RUN"} <= namespace.keys()
    assert namespace["DRY_RUN"] is True  # sharing the notebook never spends money by default
    assert "notebook_run(IDEA, ROUNDS, BUDGET, DRY_RUN)" in sources[-1]


def test_notebook_is_self_contained(notebook):
    source = "\n".join(c.source for c in code_cells(notebook))
    assert "import devils_advocate" not in source and "from devils_advocate" not in source
    assert "def call_critic" in source and "def call_judge" in source


def test_committed_notebook_is_in_sync_with_the_source(notebook):
    committed = nbformat.read(bn.TARGET, as_version=4)
    assert [(c.cell_type, c.source) for c in committed.cells] == [(c.cell_type, c.source) for c in notebook.cells]


def test_committed_notebook_ran_without_errors_or_secrets():
    committed = nbformat.read(bn.TARGET, as_version=4)
    outputs = [o for c in code_cells(committed) for o in c.get("outputs", [])]
    assert outputs, "build it with --execute so the example run is included"
    assert not any(o.output_type == "error" for o in outputs)
    text = output_text(code_cells(committed))
    assert "Global verdict (simulated)" in text and "Report saved: ~/devils-advocate/history/" in text
    raw = nbformat.writes(committed)
    assert not re.search(r"/Users/|/home/|/var/folders|/private/|/tmp/", raw)  # no absolute paths in the outputs
    assert not any("execution" in c.metadata for c in committed.cells)  # no timestamps


def test_notebook_runs_top_to_bottom(notebook):
    """The real check: a kernel executes every cell in dry-run with no keys and a temporary HOME."""
    bn.run(notebook)
    assert not any(o.output_type == "error" for c in code_cells(notebook) for o in c.get("outputs", []))
    text = output_text(code_cells(notebook)[-1:])
    assert "Global verdict (simulated)" in text and "Report saved: ~/devils-advocate/history/" in text


def test_notebook_run_asks_for_missing_keys_without_storing_them(monkeypatch, tmp_path):
    asked, ran = [], []
    monkeypatch.setattr(da.keyring, "get_password", lambda service, user: None)
    for var in ("ANTHROPIC_API_KEY", "XAI_API_KEY"):
        monkeypatch.setenv(var, "")  # records the variable so monkeypatch removes it after the test
        monkeypatch.delenv(var)
    monkeypatch.setattr(da.getpass, "getpass", lambda prompt: asked.append(prompt) or "typed-key")
    monkeypatch.setattr(da, "run_with_config", lambda *args, **kwargs: ran.append((args, kwargs)))
    da.notebook_run("an idea", rounds=1, budget=0.05, dry_run=False)
    assert len(asked) == 2 and all("not stored" in p for p in asked)
    assert ran == [(("an idea", 1, 0.05, False), {"verbosity": 2})]
    assert os.environ["ANTHROPIC_API_KEY"] == "typed-key"  # only in this process, never written anywhere


def test_notebook_run_dry_run_never_asks_for_keys(monkeypatch):
    monkeypatch.setattr(da.getpass, "getpass", lambda prompt: pytest.fail("asked for a key in dry-run"))
    ran = []
    monkeypatch.setattr(da, "run_with_config", lambda *args, **kwargs: ran.append((args, kwargs)))
    da.notebook_run("an idea")
    assert ran == [(("an idea", 1, 0.1, True), {"verbosity": 2})]


def test_notebook_run_turns_exits_into_a_message(monkeypatch):
    import io
    from rich.console import Console

    buf = io.StringIO()
    monkeypatch.setattr(da, "console", Console(file=buf, width=200, color_system=None))

    def stop(*args, **kwargs):
        raise SystemExit(2)

    monkeypatch.setattr(da, "run_with_config", stop)
    da.notebook_run("an idea")  # does not raise in the notebook
    assert "Stopped (exit code 2)." in buf.getvalue()


@pytest.mark.parametrize("code", [
    'def f(): """doc"""\n',
    'def f():\n    """d"""; return 1\n',
    'P = """a\n\n\n\nb"""\n',
])
def test_stripper_never_changes_the_program(code):
    """Either the stripped code is the same program minus docstrings, or the build refuses."""
    try:
        out = bn.checked_strip(code)
    except (ValueError, SyntaxError):
        return
    assert bn.without_docstrings(out) == bn.without_docstrings(code)


def test_idea_starting_with_a_dash_is_not_an_option(monkeypatch, tmp_path):
    import io
    from rich.console import Console

    buf = io.StringIO()
    monkeypatch.setattr(da, "console", Console(file=buf, width=200, color_system=None))
    monkeypatch.setattr(da, "CONFIG_PATH", str(tmp_path / "config.toml"))
    da.notebook_run("-tabs-vs-spaces")
    assert "Global verdict (simulated)" in buf.getvalue()


@pytest.mark.parametrize("idea, rounds, budget", [("  ", 1, 0.1), ("x", 4, 0.1), ("x", 1, 0), ("x", 1, float("inf")), ("x", 1, "0.1")])
def test_notebook_run_rejects_bad_variables(monkeypatch, idea, rounds, budget):
    import io
    from rich.console import Console

    buf = io.StringIO()
    monkeypatch.setattr(da, "console", Console(file=buf, width=200, color_system=None))
    monkeypatch.setattr(da, "run", lambda *a, **k: pytest.fail("ran with bad variables"))
    da.notebook_run(idea, rounds, budget)
    assert "Stopped (exit code 2)." in buf.getvalue()


def test_cli_only_code_is_left_out_of_the_notebook(notebook):
    code = "\n".join(c.source for c in code_cells(notebook))
    for name in ("def main", "def parse_args", "def configure", "CONFIG_TEMPLATE", "def read_idea", "def positive_float"):
        assert name not in code
    assert "def notebook_run" in code and "def run_with_config" in code


def test_notebook_run_takes_keys_from_colab_secrets(monkeypatch):
    import types

    asked, ran = [], []
    monkeypatch.setattr(da.keyring, "get_password", lambda service, user: None)
    for var in ("ANTHROPIC_API_KEY", "XAI_API_KEY"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    secrets = {"ANTHROPIC_API_KEY": "from-colab"}  # the xAI one is missing: fall back to getpass
    userdata = types.SimpleNamespace(get=lambda name: secrets[name])  # KeyError like a missing secret
    colab = types.ModuleType("google.colab")
    colab.userdata = userdata
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(da.getpass, "getpass", lambda prompt: asked.append(prompt) or "typed-key")
    monkeypatch.setattr(da, "run_with_config", lambda *args, **kwargs: ran.append(args))
    da.notebook_run("an idea", dry_run=False)
    assert os.environ["ANTHROPIC_API_KEY"] == "from-colab" and os.environ["XAI_API_KEY"] == "typed-key"
    assert len(asked) == 1 and "xai" in asked[0] and ran


def test_colab_secret_is_none_outside_colab(monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    assert da.colab_secret("ANTHROPIC_API_KEY") is None


def test_install_cell_runs_only_in_colab(notebook):
    first_code = code_cells(notebook)[0].source
    assert 'if "google.colab" in sys.modules:' in first_code and "pip" in first_code
    assert "import argparse" not in "\n".join(c.source for c in code_cells(notebook))


def test_personal_problem_goes_only_to_the_delivery_copy(tmp_path):
    target = tmp_path / "entrega" / "devils_advocate.ipynb"
    nb = bn.build(target=target, problem="Mi texto personal.\n")
    intro = nb.cells[0].source
    assert "**El problema.** Mi texto personal." in intro and "es fácil ver los pros" not in intro
    assert "## Roles" in intro  # the rest of the intro is kept
    assert target.exists()


def test_delivery_folder_is_gitignored():
    import subprocess
    out = subprocess.run(["git", "check-ignore", "entrega/devils_advocate.ipynb", "entrega/problema.md"],
                         cwd=bn.ROOT, capture_output=True, text=True)
    assert out.stdout.split() == ["entrega/devils_advocate.ipynb", "entrega/problema.md"]


def test_cli_only_marker_must_precede_a_statement():
    with pytest.raises(ValueError, match="cli-only"):
        bn.drop_cli_only("x = 1\n    # cli-only\n")
    assert bn.drop_cli_only("a = 1\n# cli-only\ndef f():\n    return 2\nb = 3\n") == "a = 1\nb = 3\n"


def test_spec_command_notebook_runs_top_to_bottom():
    """The SPEC's literal command, `jupytext --to ipynb devils_advocate.py` (comments kept), also runs."""
    import jupytext

    nb = jupytext.read(bn.SOURCE)
    nb.metadata["kernelspec"] = bn.KERNELSPEC
    bn.run(nb)
    assert not any(o.output_type == "error" for c in code_cells(nb) for o in c.get("outputs", []))
    assert "Global verdict (simulated)" in output_text(code_cells(nb)[-1:])
