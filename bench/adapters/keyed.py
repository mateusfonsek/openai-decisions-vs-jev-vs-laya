"""Formato com chaves usado por Jev (TypeSafe) e Laya: questions/answers como dict."""
from __future__ import annotations

from bench.adapters.base import PermanentError, Question, argmax, normalize


def to_keyed_question(q: Question) -> dict:
    if q.type == "choice":
        return {"type": "choice", "instructions": q.instructions, "criteria": dict(q.options)}
    if q.type == "predicate":
        d = {"type": "noul", "instructions": q.instructions}
        if q.criteria:
            d["criteria"] = dict(q.criteria)
        return d
    return {
        "type": "score",
        "instructions": q.instructions,
        "criteria": [f"{lv['label']}: {lv['description']}" for lv in q.levels],
    }


def parse_keyed_answer(q: Question, a: dict):
    if q.type == "predicate":
        if "noul" not in a:
            raise PermanentError("resposta sem 'noul'")
        p = float(a["noul"])
        return p, {"true": p, "false": 1.0 - p}, None

    raw = a.get("probabilities")
    if not raw:
        raise PermanentError("resposta sem 'probabilities'")
    raw = {str(k): float(v) for k, v in raw.items()}
    conf = a.get("confidence")
    conf = float(conf) if conf is not None else None

    if q.type == "choice":
        probs = normalize({opt: raw.get(opt, 0.0) for opt in q.options})
        choice = a.get("choice", argmax(probs))
        if choice not in q.options:
            raise PermanentError(f"choice fora das opções: {choice!r}")
        return choice, probs, conf

    probs = normalize({str(i + 1): raw.get(str(i), 0.0) for i in range(len(q.levels))})
    return int(argmax(probs)), probs, conf
