"""Carga, escrita e validação do dataset."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from bench.adapters.base import Question, Request

SUITES = ("routing", "injection", "judge")
DIFFICULTIES = ("easy", "ambiguous", "adversarial")
SOURCES = ("synthetic",)
MAX_INPUT_CHARS = 2800  # ~700 tokens; contexto padrão da Laya é 1.024


@dataclass(frozen=True)
class Case:
    id: str
    suite: str
    difficulty: str
    source: str
    input: str
    expected: str | bool | int
    notes: str = ""


def load_suites(path: Path) -> dict[str, Question]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {
        suite: Question(
            type=spec["type"],
            name=spec["name"],
            instructions=spec["instructions"],
            options=spec.get("options"),
            levels=spec.get("levels"),
            criteria={str(k): v for k, v in spec["criteria"].items()} if spec.get("criteria") else None,
        )
        for suite, spec in raw.items()
    }


def load_cases(path: Path) -> list[Case]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def write_cases(path: Path, cases: list[Case]) -> None:
    Path(path).write_text(
        "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in cases), encoding="utf-8"
    )


def _expected_ok(expected, question: Question) -> bool:
    if question.type == "choice":
        return isinstance(expected, str) and expected in question.options
    if question.type == "predicate":
        return isinstance(expected, bool)
    return isinstance(expected, int) and not isinstance(expected, bool) and 1 <= expected <= len(question.levels)


def validate_cases(cases: list[Case], suite: str, question: Question) -> list[str]:
    errors = []
    for case_id, n in Counter(c.id for c in cases).items():
        if n > 1:
            errors.append(f"{case_id}: id duplicado ({n}x)")
    for c in cases:
        if c.suite != suite:
            errors.append(f"{c.id}: suite '{c.suite}' != '{suite}'")
        if c.difficulty not in DIFFICULTIES:
            errors.append(f"{c.id}: difficulty inválida '{c.difficulty}'")
        if c.source not in SOURCES:
            errors.append(f"{c.id}: source inválida '{c.source}'")
        if not _expected_ok(c.expected, question):
            errors.append(f"{c.id}: expected inválido {c.expected!r}")
        if not c.input.strip():
            errors.append(f"{c.id}: input vazio")
        if len(c.input) > MAX_INPUT_CHARS:
            errors.append(f"{c.id}: input com {len(c.input)} caracteres (máx {MAX_INPUT_CHARS})")
    return errors


def _spread(counts: Counter, keys, tolerance: float, label: str) -> list[str]:
    mean = sum(counts[k] for k in keys) / len(keys)
    return [
        f"{label} '{k}': {counts[k]} casos (média {mean:.1f}, tolerância ±{tolerance})"
        for k in keys
        if abs(counts[k] - mean) > tolerance
    ]


def check_balance(cases: list[Case], question: Question) -> list[str]:
    errors = _spread(Counter(c.difficulty for c in cases), DIFFICULTIES, 3, "difficulty")
    labels = Counter(str(c.expected) for c in cases)
    if question.type == "choice":
        errors += _spread(labels, list(question.options), 2, "opção")
    elif question.type == "predicate":
        errors += _spread(labels, ["True", "False"], 2, "rótulo")
    else:
        errors += _spread(labels, [str(i) for i in range(1, len(question.levels) + 1)], 2, "nota")
    return errors


def to_request(case: Case, question: Question) -> Request:
    return Request(case_id=case.id, input=case.input, question=question)
