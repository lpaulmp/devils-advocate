"""Costs (stage 5): prices from config, estimate before each call, budget with the Judge's share, cost table,
per-model usage after a fallback, and --configure. No network, no real keychain."""
import io
import json
import os
import re
import socket
import stat
from types import SimpleNamespace

import pytest
from rich.console import Console

import devils_advocate as da
from tests.fakes import FakeClaude, FakeGrok, critic_response, judge_response

FAKE_KEY = "sk-ant-api03-" + "c" * 24  # fake API key; must never be printed
REAL_ESTIMATE = da.estimate_tokens  # captured before the live fixture replaces it
ROUND = {"objections": [{"id": "O1", "kind": "premise", "text": "Nobody asked for it.", "severity": 4, "probability": 3}]}
PRICES = {"claude-haiku-5-5": {"input": 100.0, "output": 100.0},
          "grok-4.20-0309-non-reasoning": {"input": 100.0, "output": 100.0},
          "claude-sonnet-5-5": {"input": 100.0, "output": 100.0}}


def prices_toml(prices):
    return "[prices]\n" + "".join(f'"{m}" = {{ input = {p["input"]}, output = {p["output"]} }}\n' for m, p in prices.items())


# --- config ------------------------------------------------------------------------------------------

def test_prices_from_config(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[prices]\n"claude-haiku-5-5" = { input = 0.1, output = 0.5 }\n"grok-x" = { input = 1, output = 2 }\n', encoding="utf-8")
    assert da.load_config(cfg)["prices"] == {"claude-haiku-5-5": {"input": 0.1, "output": 0.5}, "grok-x": {"input": 1, "output": 2}}


@pytest.mark.parametrize("entry", [
    '"m" = 1', '"m" = { input = 1 }', '"m" = { input = -1, output = 1 }', '"m" = { input = true, output = 1 }',
    '"m" = { input = inf, output = 1 }', '"m" = { input = "1", output = 1 }', '"m" = { input = 1, output = 1, cache = 1 }',
])
def test_invalid_prices_fail_closed(tmp_path, entry):
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"[prices]\n{entry}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prices"):
        da.load_config(cfg)


# --- Session -----------------------------------------------------------------------------------------

def test_cost_and_spent():
    s = da.Session(max_calls=5, prices={"a": {"input": 2.0, "output": 10.0}})
    s.record("judge", 1000, 500, "a")
    s.record("judge", 1000, 500, "a-20261001")  # dated id of a priced alias
    s.record("critic", 1000, 500, "unpriced")
    assert s.cost("a", 1000, 500) == pytest.approx(0.007)
    assert s.spent() == pytest.approx(0.014)
    assert s.usage["judge"] == {"input_tokens": 2000, "output_tokens": 1000}


def test_start_call_checks_budget_with_estimate_reserve_and_attempts():
    s = da.Session(max_calls=10, budget=0.099, prices={"a": {"input": 10.0, "output": 10.0}}, judge_reserve=0.03)
    s.start_call(reserve=1, model="a", input_tokens=4000, max_output=1000)  # 0.05 + 0.03 kept ≤ 0.099
    s.record("critic", 4000, 1000, "a")                                       # spent 0.05
    with pytest.raises(da.BudgetExceeded, match="kept for the Judge"):
        s.start_call(reserve=1, model="a", input_tokens=1000, max_output=1000)  # 0.05 + 0.02 + 0.03 > 0.099
    s.start_call(model="a", input_tokens=1000, max_output=1000)              # the Judge itself may use the share
    with pytest.raises(da.BudgetExceeded):
        s.start_call(model="a", input_tokens=1000, max_output=1000, attempts=3)  # 0.05 + 0.06 > 0.099
    assert s.calls == 2


def test_unpriced_models_are_not_blocked_but_listed():
    s = da.Session(max_calls=5, budget=0.0001)
    s.start_call(model="unknown", input_tokens=10**6, max_output=10**6)
    assert s.calls == 1 and s.unpriced == {"unknown"}


def test_estimate_is_conservative():
    assert da.estimate_tokens("x" * 300) == 150  # 300 / 3 + 50 overhead
    assert da.estimate_tokens("数据" * 50) == 250  # non-ASCII: 2 tokens per character + 50
    assert da.estimate_tokens("ñ", "a" * 3) == 53


def test_judge_budgets_two_attempts(monkeypatch):
    monkeypatch.setattr(da, "estimate_tokens", lambda *texts: 1000)
    # One attempt would cost (1000 + 700) × $100/M = $0.17; a fallback could bill two: $0.34 > $0.30.
    session = da.Session(max_calls=3, budget=0.30, prices={"claude-sonnet-5-5": {"input": 100.0, "output": 100.0}})
    claude = FakeClaude()
    with pytest.raises(da.BudgetExceeded):
        da.call_judge(claude, session, "idea", [], [])
    assert claude.judge.requests == []


# --- live runs -------------------------------------------------------------------------------------------

@pytest.fixture
def live(monkeypatch, tmp_path):
    buf = io.StringIO()
    monkeypatch.setattr(da, "console", Console(file=buf, width=200, color_system=None))
    monkeypatch.setattr(da, "CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.setattr(da.keyring, "get_password", lambda service, user: FAKE_KEY)
    monkeypatch.setattr(da, "estimate_tokens", lambda *texts: 1000)  # fixed estimate: easy budget arithmetic

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    state = SimpleNamespace(out=buf, tmp=tmp_path, claude=FakeClaude(critic_response(ROUND), critic_response(ROUND)), grok=FakeGrok())
    monkeypatch.setattr(da, "make_anthropic_client", lambda: state.claude)
    monkeypatch.setattr(da, "make_xai_client", lambda: state.grok)
    return state


# With prices of $100/M and estimate 1000 input tokens:
#   Critic estimate (1000 + 600) → $0.16, Auditor (1000 + 500) → $0.15,
#   Judge share kept during the rounds 2 × (1000 + 3000 + 700) → $0.94.
#   Real usage: Critic 150 tokens → $0.015, Auditor 120 → $0.012, Judge 500 → $0.05.

def test_budget_stops_the_rounds_but_keeps_the_judge(live, isolated_history):
    (live.tmp / "config.toml").write_text(prices_toml(PRICES), encoding="utf-8")
    da.main(["--budget", "1.12", "an idea"])
    out = live.out.getvalue()
    assert len(live.claude.requests) == 1 and len(live.grok.requests) == 1  # round 2's Critic would exceed
    assert "Round 2: Critic failed (budget would be exceeded" in out
    assert len(live.claude.judge.requests) == 1                            # the Judge still runs
    assert "$0.0770 of $1.12" in out                                       # 0.015 + 0.012 + 0.05
    [report] = sorted(isolated_history.glob("*.md"))
    text = report.read_text(encoding="utf-8")
    assert "## Verdict: ADJUST" in text and "budget would be exceeded" in text
    assert "| **total** | | | | **$0.0770** of $1.12 |" in text


def test_budget_too_small_for_anything_exits_cleanly(live, isolated_history):
    (live.tmp / "config.toml").write_text(prices_toml(PRICES), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        da.main(["--budget", "0.5", "an idea"])
    assert e.value.code == 2
    assert "Critic failed: budget would be exceeded" in live.out.getvalue()
    assert live.claude.requests == []


def test_cost_table_per_role_and_model(live):
    (live.tmp / "config.toml").write_text(prices_toml(PRICES), encoding="utf-8")
    da.main(["--budget", "5", "an idea"])
    out = live.out.getvalue()
    assert "Usage and cost" in out
    assert re.search(r"critic\W+claude-haiku-5-5\W+200\W+100\W+\$0\.0300", out)  # two Critic calls
    assert re.search(r"auditor\W+grok-4\.20-0309-non-reasoning\W+160\W+80\W+\$0\.0240", out)  # two rounds
    assert re.search(r"judge\W+claude-sonnet-5-5\W+300\W+200\W+\$0\.0500", out)


def test_fallback_usage_is_recorded_per_model(live, isolated_history):
    prices = {**PRICES, "claude-sonnet-5": {"input": 50.0, "output": 50.0}}
    (live.tmp / "config.toml").write_text(prices_toml(prices), encoding="utf-8")
    live.claude.judge.queue = [judge_response(fallback="claude-sonnet-5")]
    da.main(["--budget", "5", "an idea"])
    out = live.out.getvalue()
    assert re.search(r"judge\W+claude-sonnet-5-5\W+300\W+200\W+\$0\.0500", out)
    assert re.search(r"judge\W+claude-sonnet-5\W+300\W+200\W+\$0\.0250", out)


def test_missing_prices_warn_and_do_not_block(live, isolated_history):
    da.main(["--budget", "0.0001", "an idea"])
    out = live.out.getvalue()
    assert "no price in config.toml for claude-haiku-5-5, grok-4.20-0309-non-reasoning, claude-sonnet-5-5" in out
    assert len(live.claude.judge.requests) == 1
    text = sorted(isolated_history.glob("*.md"))[0].read_text(encoding="utf-8")
    assert "No price in config.toml for:" in text


# --- --configure -----------------------------------------------------------------------------------------

@pytest.fixture
def configure(monkeypatch, tmp_path):
    buf = io.StringIO()
    monkeypatch.setattr(da, "console", Console(file=buf, width=200, color_system=None))
    monkeypatch.setattr(da, "CONFIG_PATH", str(tmp_path / "cfg" / "config.toml"))
    stored = {}
    monkeypatch.setattr(da.keyring, "set_password", lambda service, user, key: stored.__setitem__((service, user), key))
    answers = []
    monkeypatch.setattr(da.getpass, "getpass", lambda prompt: answers.pop(0))
    return SimpleNamespace(out=buf, stored=stored, answers=answers, config=tmp_path / "cfg" / "config.toml")


def test_configure_stores_keys_and_creates_private_template(configure):
    configure.answers += [FAKE_KEY, ""]
    da.main(["--configure"])
    assert configure.stored == {("devils-advocate-anthropic", "api_key"): FAKE_KEY}
    out = configure.out.getvalue()
    assert FAKE_KEY not in out and "xai: unchanged" in out
    text = configure.config.read_text(encoding="utf-8")
    assert "[prices]" in text and not re.search(r"^\s*\"[^\"]+\"\s*=", text, re.M)  # no uncommented price
    assert da.load_config(configure.config) == {"models": {}, "mask": {"terms": []}, "prices": {}}
    assert stat.S_IMODE(os.stat(configure.config).st_mode) == 0o600


def test_configure_never_overwrites_config(configure):
    configure.config.parent.mkdir(parents=True)
    configure.config.write_text("# mine\n", encoding="utf-8")
    configure.answers += ["", ""]
    da.main(["--configure"])
    assert configure.config.read_text(encoding="utf-8") == "# mine\n"
    assert "already exists" in configure.out.getvalue()


def test_configure_keyring_error_exits_without_printing_the_key(configure, monkeypatch):
    def broken(service, user, key):
        raise da.KeyringError(f"cannot store {key}")

    monkeypatch.setattr(da.keyring, "set_password", broken)
    configure.answers += [FAKE_KEY]
    with pytest.raises(SystemExit) as e:
        da.main(["--configure"])
    assert e.value.code == 2
    assert FAKE_KEY not in configure.out.getvalue()


def test_each_role_checks_the_budget_with_its_own_model_and_estimate(live, monkeypatch):
    """Spy on start_call: every call passes its model, a real input estimate and its max_tokens."""
    monkeypatch.setattr(da, "estimate_tokens", REAL_ESTIMATE)
    calls = []
    real = da.Session.start_call

    def spy(self, reserve=0, model="", input_tokens=0, max_output=0, attempts=1):
        calls.append(dict(reserve=reserve, model=model, input_tokens=input_tokens, max_output=max_output, attempts=attempts))
        return real(self, reserve, model, input_tokens, max_output, attempts)

    monkeypatch.setattr(da.Session, "start_call", spy)
    da.main(["--rounds", "1", "an idea"])
    critic, auditor, judge = calls
    assert (critic["model"], critic["max_output"], critic["reserve"]) == ("claude-haiku-5-5", 600, 1)
    assert (auditor["model"], auditor["max_output"], auditor["reserve"]) == ("grok-4.20-0309-non-reasoning", 500, 1)
    assert (judge["model"], judge["max_output"], judge["reserve"], judge["attempts"]) == ("claude-sonnet-5-5", 700, 0, 2)
    # Each estimate covers everything the request sends: system prompt, content and output schema.
    c, a, j = live.claude.requests[0], live.grok.requests[0], live.claude.judge.requests[0]
    assert [critic["input_tokens"], auditor["input_tokens"], judge["input_tokens"]] == [
        REAL_ESTIMATE(c["system"], c["messages"][0]["content"], json.dumps(c["output_config"])),
        REAL_ESTIMATE(a["messages"][0]["content"], a["messages"][1]["content"], json.dumps(a["response_format"])),
        REAL_ESTIMATE(j["system"], j["messages"][0]["content"], json.dumps(j["output_config"])),
    ]


def test_budget_can_block_the_auditor(live, isolated_history):
    # Critic: 0 + 0.16 + 0.94 kept ≤ 1.102; Auditor: 0.015 + 0.15 + 0.94 = 1.105 > 1.102.
    (live.tmp / "config.toml").write_text(prices_toml(PRICES), encoding="utf-8")
    da.main(["--budget", "1.102", "--rounds", "1", "an idea"])
    out = live.out.getvalue()
    assert live.grok.requests == []
    assert "Round 1: Auditor failed (budget would be exceeded" in out
    assert len(live.claude.judge.requests) == 1


def test_fallback_to_an_unpriced_model_is_reported(live, isolated_history):
    (live.tmp / "config.toml").write_text(prices_toml(PRICES), encoding="utf-8")
    live.claude.judge.queue = [judge_response(fallback="claude-fable-5-1")]
    da.main(["--budget", "5", "an idea"])
    out = live.out.getvalue()
    assert "No price in config.toml for: claude-fable-5-1" in out
    text = sorted(isolated_history.glob("*.md"))[0].read_text(encoding="utf-8")
    assert "No price in config.toml for: `claude-fable-5-1`" in text


def test_configure_template_error_does_not_print_the_home_path(configure, monkeypatch):
    configure.answers += ["", ""]

    def broken(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(configure.config))

    monkeypatch.setattr(da.os, "open", broken)
    with pytest.raises(SystemExit) as e:
        da.main(["--configure"])
    assert e.value.code == 2
    out = configure.out.getvalue()
    assert "Permission denied" in out and str(configure.config.parent) not in out
