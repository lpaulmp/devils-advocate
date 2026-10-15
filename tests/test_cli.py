"""Dry-run needs no keys or network, only masked text reaches the roles, and CLI errors exit cleanly."""
import io
import socket

import pytest
from rich.console import Console

import devils_advocate as da

FAKE_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"
FAKE_TOKEN = "sk-ant-api03-" + "x" * 24


@pytest.fixture
def out(monkeypatch, tmp_path):
    """Capture console output, isolate config, and forbid keys and network."""
    buf = io.StringIO()
    monkeypatch.setattr(da, "console", Console(file=buf, width=200, color_system=None))
    monkeypatch.setattr(da, "CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    # Never touch the real keychain from tests.
    monkeypatch.setattr(da.keyring, "get_password", lambda service, user: None)

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    return buf


@pytest.mark.parametrize("extra", [[], ["--local"], ["--rounds", "1"], ["--rounds", "3"]])
def test_dry_run_without_keys_or_network(out, extra):
    da.main(["-vv", "--dry-run", *extra, "an idea"])
    assert "Veredicto global" in out.getvalue()


def test_dry_run_reports_call_cap(out):
    da.main(["-vv", "--dry-run", "--rounds", "3", "x"])
    assert "máx. llamadas=7" in out.getvalue()


def test_only_masked_text_reaches_roles(out, monkeypatch, tmp_path):
    (tmp_path / "config.toml").write_text('[mask]\nterms = ["Falcon"]\n', encoding="utf-8")
    seen = []
    original = da.simulate_critic
    monkeypatch.setattr(da, "simulate_critic", lambda idea, focus=(): seen.append(idea) or original(idea, focus))

    da.main(["-vv", "--dry-run", f"Falcon uses {FAKE_KEY} on 10.0.0.1"])
    assert seen and all(idea == "[TERM] uses [AWS_KEY] on [IP]" for idea in seen)
    assert "Enmascarado: 3 reemplazo(s)" in out.getvalue()


@pytest.mark.parametrize("level", [[], ["-v"], ["-vv"]])
def test_no_output_contains_a_secret(out, level):
    da.main([*level, "--dry-run", f"keys {FAKE_KEY} {FAKE_TOKEN} mail a@example.com"])
    text = out.getvalue()
    for secret in (FAKE_KEY, FAKE_TOKEN, "a@example.com"):
        assert secret not in text


@pytest.mark.parametrize("level", [[], ["-v"], ["-vv"]])
def test_model_text_is_not_parsed_as_markup(out, monkeypatch, level):
    def critic(idea, focus=()):
        return da.CriticResponse(objections=[
            da.Objection(id="O1", kind="premise", text="[simulated] a [/b] b :smile:", severity=1, probability=1),
        ])

    monkeypatch.setattr(da, "simulate_critic", critic)
    da.main([*level, "--dry-run", "x"])
    assert "[simulated] a [/b] b :smile:" in out.getvalue()


def test_live_run_without_key_exits_cleanly(out):
    with pytest.raises(SystemExit) as e:
        da.main(["-vv", "an idea"])
    assert e.value.code == 2
    assert "falta la clave de Anthropic" in out.getvalue()


def test_local_live_run_not_implemented(out):
    with pytest.raises(SystemExit) as e:
        da.main(["-vv", "--local", "an idea"])
    assert e.value.code == 2


@pytest.mark.parametrize("argv", [
    ["--rounds", "4", "x"],
    ["--budget", "-1", "x"],
    ["--budget", "0", "x"],
    ["--budget", "nan", "x"],
    ["--budget", "inf", "x"],
    ["-f", "idea.md", "x"],  # idea and file are mutually exclusive
])
def test_invalid_arguments_rejected(argv, capsys):
    with pytest.raises(SystemExit) as e:
        da.parse_args(argv)
    assert e.value.code == 2


@pytest.mark.parametrize("argv", [[], ["   "], ["-f", "missing.md"]])
def test_missing_idea_exits_cleanly(out, tmp_path, monkeypatch, argv):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        da.main(["-vv", "--dry-run", *argv])
    assert e.value.code == 2


def test_idea_from_file(out, tmp_path):
    idea = tmp_path / "idea.md"
    idea.write_text("an idea from a file", encoding="utf-8")
    da.main(["-vv", "--dry-run", "-f", str(idea)])
    assert "Veredicto global" in out.getvalue()


def test_invalid_config_aborts(out, tmp_path):
    (tmp_path / "config.toml").write_text("[mask\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        da.main(["-vv", "--dry-run", "x"])
    assert e.value.code == 2
