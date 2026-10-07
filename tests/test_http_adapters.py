import json
from pathlib import Path

import httpx
import pytest

from bench.adapters.base import PermanentError, Question, Request
from bench.adapters.jev import JevAdapter
from bench.adapters.openai import OpenAIAdapter

FIX = Path(__file__).parent / "fixtures"
CHOICE = Question(type="choice", name="tool", instructions="qual?",
                  options={"calendar": "agenda", "email": "e-mail"})
PRED = Question(type="predicate", name="inj", instructions="ataque?")
SCORE = Question(type="score", name="q", instructions="nota?",
                 levels=[{"label": str(i), "description": f"d{i}"} for i in range(1, 6)])


def _mock(body, seen):
    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_payload_shapes():
    a = OpenAIAdapter("k")
    p = a.build_payload(Request("c", "oi", CHOICE))
    assert p["model"] == "gpt-6-luna" and p["input"] == "oi"
    assert p["questions"][0]["choices"] == [{"value": "calendar", "description": "agenda"},
                                            {"value": "email", "description": "e-mail"}]
    s = a.build_payload(Request("c", "oi", SCORE))["questions"][0]
    assert s["levels"][0] == {"label": "1", "description": "d1"}
    assert a.build_payload(Request("c", "oi", PRED))["questions"][0]["type"] == "predicate"


def test_openai_decide_choice():
    seen = []
    body = json.loads((FIX / "openai_choice.json").read_text())
    r = OpenAIAdapter("secret", client=_mock(body, seen)).decide(Request("c1", "oi", CHOICE))
    assert seen[0].url == "https://api.openai.com/v1/decisions"
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert r.answer == "calendar" and r.probabilities == pytest.approx({"calendar": 0.9, "email": 0.1})
    assert r.provider_confidence == 0.85 and r.usage == {"input_tokens": 120, "output_tokens": 0}
    assert r.provider == "openai" and r.latency_ms >= 0 and r.error is None


def test_openai_decide_score_one_based():
    body = json.loads((FIX / "openai_score.json").read_text())
    r = OpenAIAdapter("k", client=_mock(body, [])).decide(Request("c", "oi", SCORE))
    assert r.answer == 4 and r.probabilities["4"] == pytest.approx(0.6)
    assert r.usage == {"input_tokens": 0, "output_tokens": 0}


def test_openai_decide_predicate():
    body = {"answers": [{"type": "predicate", "name": "inj", "probability": 0.3}]}
    r = OpenAIAdapter("k", client=_mock(body, [])).decide(Request("c", "oi", PRED))
    assert r.answer == 0.3 and r.probabilities == pytest.approx({"true": 0.3, "false": 0.7})


def test_openai_missing_answers_is_permanent():
    with pytest.raises(PermanentError):
        OpenAIAdapter("k", client=_mock({"answers": []}, [])).decide(Request("c", "oi", CHOICE))


def test_jev_decide_choice():
    seen = []
    body = json.loads((FIX / "jev_choice.json").read_text())
    r = JevAdapter("secret", client=_mock(body, seen)).decide(Request("c1", "oi", CHOICE))
    sent = json.loads(seen[0].content)
    assert seen[0].url == "https://api.typesafe.ai/v1/systemone"
    assert sent["state"] == "oi" and sent["model"] == "jev-latest"
    assert sent["questions"]["tool"]["criteria"] == CHOICE.options
    assert r.answer == "email" and r.provider == "jev"
    assert r.usage == {"input_tokens": 300, "output_tokens": 20}


def test_jev_missing_answer_is_permanent():
    with pytest.raises(PermanentError):
        JevAdapter("k", client=_mock({"answers": {}}, [])).decide(Request("c", "oi", CHOICE))


def test_openai_ping_uses_single_model_endpoint():
    seen = []
    OpenAIAdapter("k", client=_mock({"id": "gpt-6-luna"}, seen)).ping()
    assert seen[0].method == "GET" and seen[0].url == "https://api.openai.com/v1/models/gpt-6-luna"
