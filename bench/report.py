"""Métricas por provedor × suíte; grava metrics.json, summary.md (uma tabela por cenário) e gráficos."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import yaml

from bench.charts import write_charts
from bench.dataset import DIFFICULTIES, SUITES, load_cases, load_suites
from bench.metrics import percentile, suite_metrics
from bench.runner import load_results
from bench.tables import markdown_tables


def _rtt_median(results_dir: Path, provider: str) -> float:
    p = results_dir / f"{provider}.rtt.json"
    if not p.exists():
        return 0.0
    data = json.loads(p.read_text())
    samples = data.get("before", []) + data.get("after", [])
    return median(samples) if samples else 0.0


def build_metrics(results_dir: Path, dataset_dir: Path, pricing: dict) -> dict:
    results_dir, dataset_dir = Path(results_dir), Path(dataset_dir)
    questions = load_suites(dataset_dir / "suites.yaml")
    out: dict = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        provider = path.stem
        responses = load_results(path)
        rtt = _rtt_median(results_dir, provider)
        price = pricing.get(provider, {"input_per_m": 0.0, "output_per_m": 0.0})
        out[provider] = {}
        for suite in SUITES:
            q = questions[suite]
            labels = list(q.options) if q.type == "choice" else None
            pairs = [(c, responses[c.id]) for c in load_cases(dataset_dir / f"{suite}.jsonl") if c.id in responses]
            if not pairs:
                continue
            m = suite_metrics(q.type, pairs, labels)
            m["by_difficulty"] = {
                d: suite_metrics(q.type, [p for p in pairs if p[0].difficulty == d], labels)
                for d in DIFFICULTIES if any(p[0].difficulty == d for p in pairs)
            }
            ok = [r for _, r in pairs if r.error is None]
            lat = [r.latency_ms for r in ok if r.latency_ms is not None]
            m["latency_p50"] = percentile(lat, 50)
            m["latency_p95"] = percentile(lat, 95)
            # ping mais lento que a decisão = endpoint de ping não representa a rede; não estimar
            m["latency_p50_net"] = m["latency_p50"] - rtt if lat and rtt < m["latency_p50"] else None
            m["rtt_median"] = rtt
            if ok:
                tin = sum(r.usage.get("input_tokens", 0) for r in ok) / len(ok)
                tout = sum(r.usage.get("output_tokens", 0) for r in ok) / len(ok)
                m["cost_per_1k"] = round((tin * price["input_per_m"] + tout * price["output_per_m"]) / 1e6 * 1000, 6)
            out[provider][suite] = m
    return out


def write_report(results_dir: Path, dataset_dir: Path, pricing_path: Path,
                 logos_dir: Path = Path("assets/logos")) -> Path:
    pricing = yaml.safe_load(Path(pricing_path).read_text(encoding="utf-8")) or {}
    metrics = build_metrics(results_dir, dataset_dir, pricing)
    out = Path(results_dir) / "report"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(markdown_tables(metrics), encoding="utf-8")
    write_charts(metrics, out, logos_dir)
    return out
