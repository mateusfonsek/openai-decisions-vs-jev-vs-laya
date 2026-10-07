import pytest

from bench.adapters.base import Response
from bench.dataset import Case
from bench.metrics import (
    accuracy, auroc, binary_prf, confusion, ece, macro_f1, percentile, reliability, suite_metrics,
)


def test_accuracy_and_macro_f1():
    y, p = ["a", "a", "b", "b"], ["a", "b", "b", "b"]
    assert accuracy(y, p) == 0.75
    # a: P=1, R=0.5, F1=2/3 ; b: P=2/3, R=1, F1=0.8 -> média 0.7333
    assert macro_f1(y, p, ["a", "b"]) == pytest.approx((2 / 3 + 0.8) / 2)
    assert confusion(y, p, ["a", "b"]) == {"a": {"a": 1, "b": 1}, "b": {"a": 0, "b": 2}}


def test_binary_prf():
    assert binary_prf([True, True, False, False], [True, False, True, False]) == (0.5, 0.5, 0.5)
    assert binary_prf([False], [False]) == (0.0, 0.0, 0.0)


def test_auroc_perfect_random_and_ties():
    assert auroc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auroc([True, False], [0.5, 0.5]) == 0.5
    assert auroc([True, True], [0.9, 0.8]) is None


def test_ece_hand_computed():
    # bin [0.9,1.0]: conf média 0.9, acc 0.5 -> |0.4| * 2/4 ; bin [0.6,0.7): conf 0.6, acc 1.0 -> 0.4 * 2/4
    assert ece([0.9, 0.9, 0.6, 0.6], [True, False, True, True]) == pytest.approx(0.4)
    bins = reliability([0.9, 0.9, 0.6, 0.6], [True, False, True, True])
    assert [b["n"] for b in bins] == [2, 2]


def test_percentile_linear():
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([], 50) is None


def _c(i, expected, difficulty="easy"):
    return Case(id=f"x{i}", suite="s", difficulty=difficulty, source="synthetic", input="t", expected=expected)


def test_suite_metrics_predicate_with_failure_and_hard_negatives():
    pairs = [
        (_c(1, True), Response("x1", "p", answer=0.9, probabilities={"true": 0.9, "false": 0.1})),
        (_c(2, False, "adversarial"), Response("x2", "p", answer=0.7, probabilities={"true": 0.7, "false": 0.3})),
        (_c(3, False, "adversarial"), Response("x3", "p", answer=0.2, probabilities={"true": 0.2, "false": 0.8})),
        (_c(4, True), Response("x4", "p", error="TransientError: 429")),
    ]
    m = suite_metrics("predicate", pairs)
    assert m["n"] == 4 and m["failure_rate"] == 0.25
    assert m["fpr_hard_negatives"] == 0.5
    assert m["recall"] == 1.0 and m["precision"] == 0.5


def test_suite_metrics_score():
    pairs = [
        (_c(1, 3), Response("x1", "p", answer=3, probabilities={"1": 0, "2": 0, "3": 1, "4": 0, "5": 0})),
        (_c(2, 5), Response("x2", "p", answer=4, probabilities={"1": 0, "2": 0, "3": 0, "4": 1, "5": 0})),
        (_c(3, 1), Response("x3", "p", answer=4, probabilities={"1": 0, "2": 0, "3": 0, "4": 1, "5": 0})),
    ]
    m = suite_metrics("score", pairs)
    assert m["exact"] == pytest.approx(1 / 3) and m["within_1"] == pytest.approx(2 / 3)
    assert m["mae"] == pytest.approx(4 / 3)
