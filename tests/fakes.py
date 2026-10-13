"""Fake Anthropic and xAI (OpenAI-compatible) clients for tests. No network."""
import json
import re
from types import SimpleNamespace

VALID_CRITIC = {"objections": [
    {"id": "O1", "kind": "premise", "text": "Nobody asked for it.", "severity": 4, "probability": 3},
    {"id": "O2", "kind": "cost", "text": "Hosting grows with traffic.", "severity": 3, "probability": 4},
]}
EMPTY_CRITIC = {"objections": []}


def critic_response(payload, stop_reason="end_turn", input_tokens=100, output_tokens=50):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def auditor_response(payload, finish_reason="stop", refusal=None, prompt_tokens=80, completion_tokens=40):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, refusal=refusal), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def sent_ids(request) -> list[str]:
    """Objection ids in the <objections> block of an Auditor request."""
    return re.findall(r'"id": "(O\d+)"', request["messages"][1]["content"])


def all_valid(request, new=()):
    """An Auditor answer that marks every objection it was sent as valid."""
    return auditor_response({
        "verdicts": [{"id": i, "verdict": "valid", "reason": "Holds."} for i in sent_ids(request)],
        "new": list(new),
    })


class FakeClaude:
    """Returns queued Critic responses in order (then 'nothing new') and records every request."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        return self.queue.pop(0) if self.queue else critic_response(EMPTY_CRITIC)


class FakeGrok:
    """Returns queued Auditor responses (or builds one from the request) and records every request."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.requests.append(request)
        reply = self.queue.pop(0) if self.queue else all_valid
        return reply(request) if callable(reply) else reply
