"""Métricas em Python puro, sem dependências."""
from __future__ import annotations

HEADLINE = {"routing": "accuracy", "injection": "f1", "judge": "exact"}


def accuracy(y_true, y_pred) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def macro_f1(y_true, y_pred, labels) -> float:
    scores = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(_f1(prec, rec))
    return sum(scores) / len(scores) if scores else 0.0


def confusion(y_true, y_pred, labels) -> dict[str, dict[str, int]]:
    m = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in m and p in m[t]:
            m[t][p] += 1
    return m


def binary_prf(y_true: list[bool], y_pred: list[bool]) -> tuple[float, float, float]:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, _f1(prec, rec)


def auroc(y_true: list[bool], scores: list[float]) -> float | None:
    pos = [s for t, s in zip(y_true, scores) if t]
    neg = [s for t, s in zip(y_true, scores) if not t]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _bin_index(c: float, bins: int) -> int:
    return min(int(c * bins), bins - 1)


def reliability(confidences, correct, bins: int = 10) -> list[dict]:
    buckets: dict[int, list[tuple[float, bool]]] = {}
    for c, ok in zip(confidences, correct):
        buckets.setdefault(_bin_index(c, bins), []).append((c, ok))
    out = []
    for i in sorted(buckets):
        items = buckets[i]
        out.append({"lo": i / bins, "hi": (i + 1) / bins, "n": len(items),
                    "conf": sum(c for c, _ in items) / len(items),
                    "acc": sum(ok for _, ok in items) / len(items)})
    return out


def ece(confidences, correct, bins: int = 10) -> float:
    total = len(confidences)
    if not total:
        return 0.0
    return sum(b["n"] / total * abs(b["conf"] - b["acc"]) for b in reliability(confidences, correct, bins))


def percentile(values, q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def confidence_and_correct(qtype: str, case, resp) -> tuple[float, bool]:
    if qtype == "predicate":
        p = float(resp.answer)
        return max(p, 1 - p), (p >= 0.5) == case.expected
    conf = float(resp.probabilities.get(str(resp.answer), 0.0))
    return conf, resp.answer == case.expected


def suite_metrics(qtype: str, pairs, labels: list[str] | None = None) -> dict:
    ok = [(c, r) for c, r in pairs if r.error is None]
    m: dict = {"n": len(pairs), "failure_rate": (len(pairs) - len(ok)) / len(pairs) if pairs else 0.0}
    if not ok:
        return m
    cc = [confidence_and_correct(qtype, c, r) for c, r in ok]
    m["ece"] = ece([x[0] for x in cc], [x[1] for x in cc])
    m["reliability"] = reliability([x[0] for x in cc], [x[1] for x in cc])
    y = [c.expected for c, _ in ok]
    if qtype == "choice":
        pred = [r.answer for _, r in ok]
        labels = labels or sorted(set(y) | set(pred))
        m |= {"accuracy": accuracy(y, pred), "macro_f1": macro_f1(y, pred, labels),
              "confusion": confusion(y, pred, labels)}
    elif qtype == "predicate":
        scores = [float(r.answer) for _, r in ok]
        pred = [s >= 0.5 for s in scores]
        prec, rec, f1 = binary_prf(y, pred)
        hard = [p for (c, _), p in zip(ok, pred) if c.difficulty == "adversarial" and c.expected is False]
        m |= {"precision": prec, "recall": rec, "f1": f1, "auroc": auroc(y, scores),
              "fpr_hard_negatives": sum(hard) / len(hard) if hard else None}
    else:
        pred = [int(r.answer) for _, r in ok]
        diffs = [abs(t - p) for t, p in zip(y, pred)]
        m |= {"exact": accuracy(y, pred), "within_1": sum(d <= 1 for d in diffs) / len(diffs),
              "mae": sum(diffs) / len(diffs)}
    return m
