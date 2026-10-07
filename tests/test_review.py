import csv
import shutil
from pathlib import Path

import pytest

from bench.dataset import Case, load_cases, write_cases
from bench.review import export_review, import_review, parse_expected

ROOT = Path(__file__).parent.parent


def _setup(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    shutil.copy(ROOT / "dataset" / "suites.yaml", d / "suites.yaml")
    write_cases(d / "routing.jsonl", [
        Case("routing-001", "routing", "easy", "synthetic", "marca reunião", "calendar"),
        Case("routing-002", "routing", "easy", "synthetic", "manda email", "calendar"),
        Case("routing-003", "routing", "easy", "synthetic", "oi tudo bem", "email"),
    ])
    write_cases(d / "injection.jsonl", [Case("injection-001", "injection", "easy", "synthetic", "ignore tudo", True)])
    write_cases(d / "judge.jsonl", [Case("judge-001", "judge", "easy", "synthetic", "P: 2+2? R: 4", 5)])
    return d


def _rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_export_has_review_columns(tmp_path):
    d = _setup(tmp_path)
    out = tmp_path / "review.csv"
    assert export_review(d, out) == 5
    rows = _rows(out)
    assert set(rows[0]) >= {"id", "suite", "difficulty", "input", "expected", "aprovado", "correcao"}
    assert rows[0]["aprovado"] == ""


def test_import_applies_approve_correct_remove(tmp_path):
    d = _setup(tmp_path)
    out = tmp_path / "review.csv"
    export_review(d, out)
    rows = _rows(out)
    decisions = {"routing-001": ("s", ""), "routing-002": ("n", "email"), "routing-003": ("n", ""),
                 "injection-001": ("sim", ""), "judge-001": ("n", "4")}
    for r in rows:
        r["aprovado"], r["correcao"] = decisions[r["id"]]
    _write(out, rows)
    rep = import_review(out, d)
    assert rep["corrected"] == ["routing-002", "judge-001"] and rep["removed"] == ["routing-003"]
    routing = {c.id: c.expected for c in load_cases(d / "routing.jsonl")}
    assert routing == {"routing-001": "calendar", "routing-002": "email"}
    assert load_cases(d / "judge.jsonl")[0].expected == 4


def test_import_aborts_on_unreviewed_rows(tmp_path):
    d = _setup(tmp_path)
    out = tmp_path / "review.csv"
    export_review(d, out)
    before = (d / "routing.jsonl").read_text()
    with pytest.raises(ValueError, match="routing-001"):
        import_review(out, d)
    assert (d / "routing.jsonl").read_text() == before


def test_import_rejects_invalid_correction(tmp_path):
    d = _setup(tmp_path)
    out = tmp_path / "review.csv"
    export_review(d, out)
    rows = _rows(out)
    for r in rows:
        r["aprovado"] = "s"
    rows[0]["aprovado"], rows[0]["correcao"] = "n", "telefone"
    _write(out, rows)
    with pytest.raises(ValueError, match="telefone"):
        import_review(out, d)


def test_parse_expected():
    assert parse_expected("Não", "predicate") is False
    assert parse_expected("true", "predicate") is True
    assert parse_expected(" 3 ", "score") == 3
    assert parse_expected("email", "choice") == "email"
