"""Rounds and beam-search pruning (stage 3), plus live runs wired to fake clients."""
import io
import socket
from types import SimpleNamespace

import pytest
from rich.console import Console

import devils_advocate as da
from tests.fakes import EMPTY_CRITIC, FakeClaude, FakeGrok, all_valid, auditor_response, critic_response

FAKE_KEY = "xai-" + "k" * 24  # fake API key; must never be printed
REAL_MAKE_XAI_CLIENT = da.make_xai_client  # captured before any test replaces it


def obj(i, severity, probability, prefix="O"):
    return da.Objection(id=f"{prefix}{i}", kind="failure", text=f"objection {prefix}{i}", severity=severity, probability=probability)


def entry(i, severity, probability, verdict="valid", prefix="O"):
    return da.Entry(obj(i, severity, probability, prefix), 1, "critic", verdict)


# --- prune -------------------------------------------------------------------------------------

def test_prune_keeps_top_k_valid_or_new():
    candidates = [entry(1, 2, 2), entry(2, 5, 5, "exaggerated"), entry(3, 4, 4), entry(4, 5, 4, "incorrect"),
                  entry(5, 3, 3, "new", "G"), entry(6, 1, 1)]
    beam = da.prune([], candidates)
    assert [e.objection.id for e in beam] == ["O3", "G5", "O1"]


def test_prune_merges_previous_beam():
    old = [entry(1, 5, 5), entry(2, 2, 2)]
    beam = da.prune(old, [entry(3, 4, 4), entry(4, 1, 1)])
    assert [e.objection.id for e in beam] == ["O1", "O3", "O2"]


def test_prune_ties_keep_earlier_entries():
    old = [entry(1, 3, 4)]  # score 12
    beam = da.prune(old, [entry(2, 4, 3), entry(3, 3, 4), entry(4, 2, 3)])  # 12, 12, 6
    assert [e.objection.id for e in beam] == ["O1", "O2", "O3"]


def test_unaudited_objections_never_enter_the_beam():
    assert da.prune([], [entry(1, 5, 5, None)]) == []


# --- run_rounds ---------------------------------------------------------------------------------

class Script:
    """Scripted role functions: each call pops the next answer (or exception) and records its input."""

    def __init__(self, critic_answers, auditor_answers=None):
        self.critic_answers, self.auditor_answers = list(critic_answers), list(auditor_answers or [])
        self.focus, self.audited, self.already_found = [], [], []

    def critic(self, idea, focus):
        self.focus.append([o.id for o in focus])
        answer = self.critic_answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return da.CriticResponse(objections=answer)

    def auditor(self, idea, objections, already_found):
        self.audited.append([o.id for o in objections])
        self.already_found.append([o.id for o in already_found])
        answer = self.auditor_answers.pop(0) if self.auditor_answers else None
        if isinstance(answer, Exception):
            raise answer
        new = answer or []
        return da.AuditorResponse(verdicts=[da.Verdict(id=o.id, verdict="valid", reason="r") for o in objections], new=new)


def test_ids_are_renumbered_for_the_whole_run_and_focus_is_the_beam():
    round1 = [obj(1, 4, 4), obj(2, 1, 1)]
    round2 = [obj(1, 5, 5)]  # the Critic restarts at O1; the run renumbers it O3
    script = Script([round1, round2], [[obj(1, 2, 2, "G")], [obj(1, 3, 3, "G")]])
    ledger, beam, note = da.run_rounds(script.critic, script.auditor, "idea", rounds=2)
    assert [e.objection.id for e in ledger] == ["O1", "O2", "G1", "O3", "G2"]
    assert script.focus == [[], ["O1", "G1", "O2"]]
    assert script.audited == [["O1", "O2"], ["O3"]]
    assert script.already_found == [[], ["O1", "G1", "O2"]]  # the Auditor sees the beam too
    assert [e.objection.id for e in beam] == ["O3", "O1", "G2"]
    assert note == ""


def test_converges_when_critic_finds_nothing_new():
    script = Script([[obj(1, 3, 3)], []])
    ledger, beam, note = da.run_rounds(script.critic, script.auditor, "idea", rounds=3)
    assert "Converged in round 2" in note
    assert script.audited == [["O1"]]  # no Auditor call for an empty round
    assert len(script.focus) == 2      # no third round


def test_round_one_critic_failure_propagates():
    script = Script([da.RoleError("boom")])
    with pytest.raises(da.RoleError):
        da.run_rounds(script.critic, script.auditor, "idea", rounds=2)


def test_later_critic_failure_keeps_what_exists():
    script = Script([[obj(1, 3, 3)], da.CallLimitReached("cap")])
    ledger, beam, note = da.run_rounds(script.critic, script.auditor, "idea", rounds=2)
    assert [e.objection.id for e in ledger] == ["O1"] and [e.objection.id for e in beam] == ["O1"]
    assert "Round 2: Critic failed (cap)" in note


def test_auditor_failure_leaves_objections_unaudited():
    script = Script([[obj(1, 3, 3)]], [da.RoleError("bad json")])
    ledger, beam, note = da.run_rounds(script.critic, script.auditor, "idea", rounds=2)
    assert ledger[0].verdict is None and beam == []
    assert "Auditor failed" in note


# --- live runs ----------------------------------------------------------------------------------

@pytest.fixture
def live(monkeypatch, tmp_path):
    buf = io.StringIO()
    monkeypatch.setattr(da, "console", Console(file=buf, width=200, color_system=None))
    monkeypatch.setattr(da, "CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.setattr(da.keyring, "get_password", lambda service, user: FAKE_KEY)

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    state = SimpleNamespace(out=buf, claude=FakeClaude(), grok=FakeGrok(), tmp=tmp_path)
    monkeypatch.setattr(da, "make_anthropic_client", lambda: state.claude)
    monkeypatch.setattr(da, "make_xai_client", lambda: state.grok)
    return state


ROUND = {"objections": [{"id": "O1", "kind": "premise", "text": "Nobody asked for it.", "severity": 4, "probability": 3}]}


def test_live_rounds_stop_at_rounds_and_reserve_the_judge(live):
    # Every round finds something new, so --rounds stops the run.
    live.claude.queue = [critic_response(ROUND)] * 3
    da.main(["--rounds", "2", "an idea"])
    out = live.out.getvalue()
    assert len(live.claude.requests) == 2 and len(live.grok.requests) == 2
    assert "calls: 5/5" in out  # the fifth call is the Judge's
    assert len(live.claude.judge.requests) == 1
    assert "Beam (top 3): O1, O2" in out


def test_live_rounds_cap_stops_the_run_before_rounds_do(live):
    # A Critic retry in each round: round 2's retry would take the Judge's call, so the cap stops it.
    live.claude.queue = [critic_response("nope"), critic_response(ROUND), critic_response("nope"), critic_response(ROUND)]
    da.main(["--rounds", "2", "an idea"])
    out = live.out.getvalue()
    assert "calls: 5/5" in out  # 4 for the rounds + the Judge's reserved call
    assert "Round 2: Critic failed (call cap reached" in out
    assert len(live.claude.judge.requests) == 1
    assert len(live.claude.requests) == 3 and len(live.grok.requests) == 1


def test_auditor_round_two_sees_earlier_edge_cases(live):
    live.claude.queue = [critic_response(ROUND), critic_response(ROUND)]
    edge = {"id": "G1", "kind": "edge_case", "text": "Empty input breaks it", "severity": 3, "probability": 4}
    live.grok.queue = [lambda r: all_valid(r, new=[edge])]
    da.main(["--rounds", "2", "an idea"])
    second = live.grok.requests[1]["messages"][1]["content"]
    found = second.split("<already_found>")[1]
    assert "Empty input breaks it" in found
    assert "<already_found>" not in live.grok.requests[0]["messages"][1]["content"]


def test_live_rounds_one_round_with_a_critic_retry_still_leaves_the_judge(live):
    live.claude.queue = [critic_response("nope"), critic_response(ROUND)]
    da.main(["--rounds", "1", "an idea"])
    out = live.out.getvalue()
    assert "calls: 3/3" in out  # Critic + retry + the Judge's reserved call
    assert "Auditor failed (call cap reached" in out  # cap 3 minus the Judge's reserve
    assert len(live.claude.judge.requests) == 1
    assert "unaudited" in out


def test_live_run_shows_verdicts_reasons_and_new_edge_cases(live):
    live.claude.queue = [critic_response(ROUND)]
    live.grok.queue = [lambda r: all_valid(r, new=[{"id": "G1", "kind": "edge_case", "text": "Empty input [x]", "severity": 2, "probability": 4}])]
    da.main(["an idea"])
    out = live.out.getvalue()
    assert "↳ Holds." in out and "Empty input [x]" in out and "G1" in out
    assert "Converged in round 2" in out
    assert "auditor: 80 input / 40 output tokens" in out


def test_auditor_receives_only_masked_text(live):
    live.claude.queue = [critic_response(ROUND)]
    secret = "AKIA" + "ABCDEFGHIJKLMNOP"
    da.main([f"Keep {secret} in git"])
    user = live.grok.requests[0]["messages"][1]["content"]
    assert secret not in user and "[AWS_KEY]" in user


def test_missing_xai_key_fails_before_any_call(live, monkeypatch):
    monkeypatch.setattr(da.keyring, "get_password", lambda service, user: None)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(da, "make_xai_client", REAL_MAKE_XAI_CLIENT)
    with pytest.raises(SystemExit) as e:
        da.main(["an idea"])
    assert e.value.code == 2
    assert "no xAI API key" in live.out.getvalue()
    assert live.claude.requests == []


@pytest.mark.parametrize("status", [401, 529])
def test_xai_errors_do_not_leak_the_key_or_retry(live, monkeypatch, status):
    """Real OpenAI SDK + mock transport: the key is never printed and each request is one counted call."""
    import httpx2 as httpx

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": f"bad key {FAKE_KEY}"}})

    real = da.openai.OpenAI

    def with_transport(**kwargs):
        return real(**kwargs, http_client=da.openai.DefaultHttpxClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(da.openai, "OpenAI", with_transport)
    monkeypatch.setattr(da, "make_xai_client", REAL_MAKE_XAI_CLIENT)
    live.claude.queue = [critic_response(ROUND)]
    da.main(["--rounds", "1", "an idea"])
    out = live.out.getvalue()
    assert FAKE_KEY not in out
    assert len(requests) == 1
    assert f"Auditor failed (API error {status}" in out


def test_dry_run_rounds_converge(monkeypatch, tmp_path):
    buf = io.StringIO()
    monkeypatch.setattr(da, "console", Console(file=buf, width=200, color_system=None))
    monkeypatch.setattr(da, "CONFIG_PATH", str(tmp_path / "config.toml"))
    da.main(["--dry-run", "--rounds", "3", "x"])
    out = buf.getvalue()
    assert "Beam (top 3): O1, O3, G1" in out
    assert "Converged in round 2" in out
