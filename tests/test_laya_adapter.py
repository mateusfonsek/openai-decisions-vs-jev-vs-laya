import json
from pathlib import Path

import pytest

from bench.adapters.base import PermanentError, Question, Request
from bench.adapters.laya import LayaAdapter

CHOICE = Question(type="choice", name="tool", instructions="qual?",
                  options={"calendar": "agenda", "email": "e-mail"})
PRED = Question(type="predicate", name="inj", instructions="ataque?")


class FakeRouter:
    def __init__(self, out):
        self.out, self.calls = out, []

    def predict(self, state, questions, model=None):
        self.calls.append((state, questions, model))
        return self.out


def test_laya_forces_multilingual_and_parses_choice():
    router = FakeRouter({"answers": {"tool": {"choice": "email",
                                              "probabilities": {"calendar": 0.3, "email": 0.7},
                                              "confidence": 0.6}}})
    r = LayaAdapter(router=router).decide(Request("c1", "manda um email", CHOICE))
    state, questions, model = router.calls[0]
    assert model == "multilingual" and state == "manda um email"
    assert questions == {"tool": {"type": "choice", "instructions": "qual?", "criteria": CHOICE.options}}
    assert r.answer == "email" and r.provider == "laya" and r.latency_ms >= 0
    assert r.usage == {"input_tokens": 0, "output_tokens": 0}


def test_laya_predicate():
    r = LayaAdapter(router=FakeRouter({"answers": {"inj": {"noul": 0.2}}})).decide(Request("c", "oi", PRED))
    assert r.answer == 0.2


def test_laya_missing_answer_is_permanent():
    with pytest.raises(PermanentError):
        LayaAdapter(router=FakeRouter({"answers": {}})).decide(Request("c", "oi", CHOICE))


def test_laya_ping_is_none():
    assert LayaAdapter(router=FakeRouter({})).ping() is None


FIXTURE = Path(__file__).parent / "fixtures" / "laya_choice.json"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture real ainda não capturada")
def test_laya_real_fixture_parses():
    out = json.loads(FIXTURE.read_text(encoding="utf-8"))
    q = Question(type="choice", name="tool", instructions="Qual ferramenta?",
                 options={"calendar": "agenda e lembretes", "email": "e-mails", "no_tool": "conversa simples"})
    r = LayaAdapter(router=FakeRouter(out)).decide(Request("c", "marca uma reunião amanhã às 10h", q))
    assert r.answer in q.options and abs(sum(r.probabilities.values()) - 1) < 1e-6
