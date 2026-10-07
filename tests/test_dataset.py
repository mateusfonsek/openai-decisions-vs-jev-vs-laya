from pathlib import Path

from bench.dataset import (
    Case, check_balance, load_cases, load_suites, to_request, validate_cases, write_cases,
)

SUITES_YAML = Path(__file__).parent.parent / "dataset" / "suites.yaml"


def _case(i, suite="routing", expected="calendar", difficulty="easy", text="oi"):
    return Case(id=f"{suite}-{i:03d}", suite=suite, difficulty=difficulty, source="synthetic",
                input=text, expected=expected)


def test_load_suites_builds_questions():
    qs = load_suites(SUITES_YAML)
    assert set(qs) == {"routing", "injection", "judge"}
    assert len(qs["routing"].options) == 8
    assert qs["injection"].type == "predicate"
    assert len(qs["judge"].levels) == 5


def test_write_and_load_roundtrip_with_accents(tmp_path):
    cases = [_case(1, text="reunião às 15h com a Conceição")]
    p = tmp_path / "routing.jsonl"
    write_cases(p, cases)
    assert load_cases(p) == cases


def test_validate_catches_bad_expected_duplicate_and_length():
    q = load_suites(SUITES_YAML)["routing"]
    cases = [_case(1), _case(1), _case(2, expected="telefone"), _case(3, text="x" * 2801)]
    errors = validate_cases(cases, "routing", q)
    assert any("duplicado" in e for e in errors)
    assert any("telefone" in e for e in errors)
    assert any("2801" in e for e in errors)


def test_validate_predicate_and_score_types():
    qs = load_suites(SUITES_YAML)
    bad_pred = [_case(1, suite="injection", expected="sim")]
    bad_score = [_case(1, suite="judge", expected=6)]
    assert validate_cases(bad_pred, "injection", qs["injection"])
    assert validate_cases(bad_score, "judge", qs["judge"])
    ok = [_case(1, suite="judge", expected=5)]
    assert validate_cases(ok, "judge", qs["judge"]) == []


def test_check_balance_flags_skewed_options():
    q = load_suites(SUITES_YAML)["routing"]
    diffs = ["easy", "ambiguous", "adversarial"]
    cases = [_case(i, expected="calendar", difficulty=diffs[i % 3]) for i in range(150)]
    assert any("calendar" in e for e in check_balance(cases, q))


def test_to_request_uses_suite_question():
    q = load_suites(SUITES_YAML)["routing"]
    r = to_request(_case(7, text="marca reunião"), q)
    assert r.case_id == "routing-007" and r.input == "marca reunião" and r.question is q
