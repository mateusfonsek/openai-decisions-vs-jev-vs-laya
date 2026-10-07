import pytest

from bench.adapters.base import PermanentError, Question, Request, Response, argmax, normalize


def test_normalize_divides_by_sum():
    assert normalize({"a": 1.0, "b": 3.0}) == {"a": 0.25, "b": 0.75}


def test_normalize_zero_sum_is_permanent_error():
    with pytest.raises(PermanentError):
        normalize({"a": 0.0, "b": 0.0})


def test_argmax_tie_returns_first_in_order():
    assert argmax({"1": 0.4, "2": 0.4, "3": 0.2}) == "1"


def test_response_roundtrip_keeps_accents():
    r = Response(case_id="x-1", provider="jev", answer="calendário", probabilities={"calendário": 1.0})
    assert Response.from_dict(r.to_dict()) == r


def test_request_holds_question():
    q = Question(type="predicate", name="q", instructions="é ataque?")
    assert Request(case_id="c", input="oi", question=q).question.type == "predicate"
