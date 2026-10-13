"""Pydantic schemas match the spec: limits on counts, kinds, verdicts and 1–5 scales."""
import pytest
from pydantic import ValidationError

import devils_advocate as da


def objection(**overrides):
    data = {"id": "O1", "kind": "premise", "text": "t", "severity": 3, "probability": 2}
    return {**data, **overrides}


def test_critic_accepts_valid_json():
    r = da.CriticResponse.model_validate({"objections": [objection()]})
    assert r.objections[0].score == 6


@pytest.mark.parametrize("bad", [{"kind": "other"}, {"severity": 0}, {"severity": 6}, {"probability": 0}, {"probability": 6}])
def test_critic_rejects_bad_objection(bad):
    with pytest.raises(ValidationError):
        da.CriticResponse.model_validate({"objections": [objection(**bad)]})


def test_critic_rejects_more_than_five():
    with pytest.raises(ValidationError):
        da.CriticResponse.model_validate({"objections": [objection(id=f"O{i}") for i in range(6)]})


def test_auditor_accepts_valid_json():
    r = da.AuditorResponse.model_validate({
        "verdicts": [{"id": "O1", "verdict": "valid", "reason": "r"}],
        "new": [objection(id="G1", kind="edge_case")],
    })
    assert r.new[0].id == "G1"


def test_auditor_rejects_more_than_three_new():
    with pytest.raises(ValidationError):
        da.AuditorResponse.model_validate({"verdicts": [], "new": [objection(id=f"G{i}") for i in range(4)]})


def test_auditor_rejects_unknown_verdict():
    with pytest.raises(ValidationError):
        da.AuditorResponse.model_validate({"verdicts": [{"id": "O1", "verdict": "maybe", "reason": "r"}]})


def test_judge_rejects_unknown_global_verdict():
    with pytest.raises(ValidationError):
        da.JudgeResponse.model_validate({
            "consensus": [], "disagreements": [], "steelman": "s", "questions": [], "global_verdict": "maybe",
        })
