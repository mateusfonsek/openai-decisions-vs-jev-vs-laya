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



def _status(code, seen):
    def handler(req):
        seen.append(req)
        return httpx.Response(code, json={"error": "unauthorized"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_ping_hits_decisions_endpoint_without_key():
    seen = []
    ms = OpenAIAdapter("secret", client=_status(401, seen)).ping()
    assert ms >= 0
    assert seen[0].method == "POST" and seen[0].url == "https://api.openai.com/v1/decisions"
    assert "authorization" not in seen[0].headers


def test_jev_ping_hits_systemone_without_key():
    seen = []
    ms = JevAdapter("secret", client=_status(403, seen)).ping()
    assert ms >= 0
    assert seen[0].method == "POST" and seen[0].url == "https://api.typesafe.ai/v1/systemone"
    assert "authorization" not in seen[0].headers


def test_injection_question_is_identical_for_all_providers():
    from bench.dataset import load_suites

    q = load_suites(Path(__file__).parent.parent / "dataset" / "suites.yaml")["injection"]
    req = Request("c", "oi", q)
    openai_q = OpenAIAdapter("k").build_payload(req)["questions"][0]
    jev_q = JevAdapter("k").build_payload(req)["questions"][q.name]
    assert set(jev_q) == {"type", "instructions"}  # nenhum campo que a OpenAI não recebe
    assert jev_q["instructions"] == openai_q["instructions"]
    assert "apenas falam sobre ataques" in openai_q["instructions"]


def test_ping_rejects_unexpected_status():
    # 200/404/redirect não são a recusa de borda esperada: a medida não seria só de rede
    with pytest.raises(PermanentError):
        OpenAIAdapter("k", client=_status(200, [])).ping()
    with pytest.raises(PermanentError):
        JevAdapter("k", client=_status(404, [])).ping()
