"""Exporta o dataset para revisão humana em CSV e importa as decisões de volta."""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from bench.dataset import SUITES, load_cases, load_suites, validate_cases, write_cases

FIELDS = ["id", "suite", "difficulty", "input", "expected", "aprovado", "correcao", "notes"]
YES = {"s", "sim", "y", "yes", "true"}
NO = {"n", "não", "nao", "no", "false"}


def parse_expected(text: str, qtype: str):
    t = text.strip()
    if qtype == "predicate":
        if t.lower() in YES:
            return True
        if t.lower() in NO:
            return False
        raise ValueError(f"valor inválido para sim/não: {t!r}")
    if qtype == "score":
        return int(t)
    return t


def export_review(dataset_dir: Path, out_csv: Path) -> int:
    n = 0
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:  # BOM para Excel/Numbers
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for suite in SUITES:
            for c in load_cases(Path(dataset_dir) / f"{suite}.jsonl"):
                w.writerow({"id": c.id, "suite": c.suite, "difficulty": c.difficulty, "input": c.input,
                            "expected": json.dumps(c.expected, ensure_ascii=False).strip('"'),
                            "aprovado": "", "correcao": "", "notes": c.notes})
                n += 1
    return n


def import_review(csv_path: Path, dataset_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    marks = {i: (r.get("aprovado") or "").strip().lower() for i, r in rows.items()}
    unreviewed = [i for i, m in marks.items() if not m]
    if unreviewed:
        raise ValueError(f"{len(unreviewed)} linhas sem 'aprovado': {', '.join(unreviewed[:20])}")
    unknown = [i for i, m in marks.items() if m not in YES | NO]
    if unknown:
        raise ValueError(f"'aprovado' deve ser s ou n; valor não reconhecido em: {', '.join(unknown[:20])}")

    questions = load_suites(dataset_dir / "suites.yaml")
    report = {"kept": 0, "corrected": [], "removed": []}
    new_cases = {}
    for suite in SUITES:
        q = questions[suite]
        kept = []
        for c in load_cases(dataset_dir / f"{suite}.jsonl"):
            r = rows.get(c.id)
            if r is None or marks[c.id] in YES:
                kept.append(c)
            elif (r.get("correcao") or "").strip():
                kept.append(replace(c, expected=parse_expected(r["correcao"], q.type)))
                report["corrected"].append(c.id)
            else:
                report["removed"].append(c.id)
        errors = validate_cases(kept, suite, q)
        if errors:
            raise ValueError("correções inválidas: " + "; ".join(errors[:10]))
        new_cases[suite] = kept
    for suite, cases in new_cases.items():  # só grava depois de validar todas as suítes
        write_cases(dataset_dir / f"{suite}.jsonl", cases)
        report["kept"] += len(cases)
    return report
