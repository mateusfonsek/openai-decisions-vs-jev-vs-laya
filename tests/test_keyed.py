import httpx
import pytest

from bench.adapters.base import PermanentError, Question, TransientError
from bench.adapters.http import post_json
from bench.adapters.keyed import parse_keyed_answer, to_keyed_question

CHOICE = Question(type="choice", name="tool", instructions="qual?",
                  options={"calendar": "agenda", "email": "e-mail", "no_tool": "nada"})
PRED = Question(type="predicate", name="inj", instructions="ataque?",
                criteria={"true": "sim", "false": "não"})
SCORE = Question(type="score", name="q", instructions="nota?",
                 levels=[{"label": str(i), "description": f"d{i}"} for i in range(1, 6)])


def test_to_keyed_question_shapes():
    assert to_keyed_question(CHOICE) == {"type": "choice", "instructions": "qual?",
                                         "criteria": CHOICE.options}
    assert to_keyed_question(PRED) == {"type": "noul", "instructions": "ataque?",
                                       "criteria": {"true": "sim", "false": "não"}}
    assert to_keyed_question(SCORE)["criteria"][0] == "1: d1"


def test_parse_keyed_fills_missing_and_normalizes():
    answer, probs, conf = parse_keyed_answer(
        CHOICE, {"choice": "calendar", "probabilities": {"calendar": 1.5, "email": 0.5}, "confidence": 0.8})
    assert answer == "calendar"
    assert probs == {"calendar": 0.75, "email": 0.25, "no_tool": 0.0}
    assert conf == 0.8


def test_parse_keyed_predicate():
    answer, probs, conf = parse_keyed_answer(PRED, {"noul": 0.9})
    assert answer == 0.9 and probs == {"true": 0.9, "false": pytest.approx(0.1)} and conf is None


def test_parse_keyed_score_shifts_to_one_based():
    answer, probs, _ = parse_keyed_answer(
        SCORE, {"score": 3.1, "probabilities": {"0": 0.0, "1": 0.1, "2": 0.6, "3": 0.3}})
    assert answer == 3
    assert probs["5"] == 0.0 and probs["3"] == pytest.approx(0.6)


def test_parse_keyed_choice_outside_options_is_permanent():
    with pytest.raises(PermanentError):
        parse_keyed_answer(CHOICE, {"choice": "phone", "probabilities": {"calendar": 1.0}})


def test_parse_keyed_missing_probabilities_is_permanent():
    with pytest.raises(PermanentError):
        parse_keyed_answer(CHOICE, {"choice": "calendar"})


def _client(status, body):
    return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(status, json=body)))


def test_post_json_classifies_errors():
    with pytest.raises(TransientError):
        post_json(_client(429, {}), "https://x/y", {}, {})
    with pytest.raises(TransientError):
        post_json(_client(529, {}), "https://x/y", {}, {})
    with pytest.raises(PermanentError):
        post_json(_client(422, {"detail": "bad"}), "https://x/y", {}, {})
    data, ms = post_json(_client(200, {"ok": True}), "https://x/y", {}, {})
    assert data == {"ok": True} and ms >= 0
