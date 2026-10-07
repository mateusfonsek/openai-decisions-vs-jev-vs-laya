"""Métricas por provedor × suíte, summary.md e gráficos."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from bench.dataset import DIFFICULTIES, SUITES, load_cases, load_suites  # noqa: E402
from bench.metrics import HEADLINE, percentile, suite_metrics  # noqa: E402
from bench.runner import load_results  # noqa: E402

COLORS = {"openai": "#10a37f", "jev": "#6b5bd6", "laya": "#e07a2f"}


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
            m["latency_p50_net"] = max(0.0, m["latency_p50"] - rtt) if lat else None
            m["rtt_median"] = rtt
            if ok:
                tin = sum(r.usage.get("input_tokens", 0) for r in ok) / len(ok)
                tout = sum(r.usage.get("output_tokens", 0) for r in ok) / len(ok)
                m["cost_per_1k"] = round((tin * price["input_per_m"] + tout * price["output_per_m"]) / 1e6 * 1000, 6)
            out[provider][suite] = m
    return out


def _fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v:.1f}"


def _summary_md(metrics: dict) -> str:
    lines = ["| Provedor | Suíte | Métrica principal | ECE | p50 (ms) | p50 sem rede (ms) | p95 (ms) | US$/1k | Falhas |",
             "|---|---|---|---|---|---|---|---|---|"]
    for provider, suites in metrics.items():
        for suite, m in suites.items():
            key = HEADLINE[suite]
            lines.append(
                f"| {provider} | {suite} | {key} {_fmt(m.get(key), True)} | {_fmt(m.get('ece'), True)} | "
                f"{_fmt(m.get('latency_p50'))} | {_fmt(m.get('latency_p50_net'))} | {_fmt(m.get('latency_p95'))} | "
                f"{m.get('cost_per_1k', 0):.4f} | {_fmt(m['failure_rate'], True)} |")
    return "\n".join(lines) + "\n"


def _hero(metrics: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    markers = {"routing": "o", "injection": "s", "judge": "^"}
    for provider, suites in metrics.items():
        for suite, m in suites.items():
            if m.get("latency_p50") is None:
                continue
            size = 80 + 4000 * m.get("cost_per_1k", 0)
            ax.scatter(m["latency_p50"], m.get(HEADLINE[suite], 0) * 100, s=size, marker=markers[suite],
                       color=COLORS.get(provider, "#888"), alpha=0.8, edgecolors="black", linewidths=0.5)
            ax.annotate(f"{provider} · {suite}", (m["latency_p50"], m.get(HEADLINE[suite], 0) * 100),
                        textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Latência p50 (ms, escala log)")
    ax.set_ylabel("Métrica principal da suíte (%)")
    ax.set_title("Decision models em PT-BR: qualidade × latência (tamanho = custo por 1k)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _by_difficulty(metrics: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(SUITES), figsize=(13, 4), sharey=True)
    providers = list(metrics)
    width = 0.8 / max(len(providers), 1)
    for ax, suite in zip(axes, SUITES):
        for i, provider in enumerate(providers):
            bd = metrics[provider].get(suite, {}).get("by_difficulty", {})
            vals = [(bd.get(d, {}).get(HEADLINE[suite]) or 0) * 100 for d in DIFFICULTIES]
            ax.bar([x + i * width for x in range(len(DIFFICULTIES))], vals, width,
                   label=provider, color=COLORS.get(provider, "#888"))
        ax.set_xticks([x + width * (len(providers) - 1) / 2 for x in range(len(DIFFICULTIES))], DIFFICULTIES)
        ax.set_title(f"{suite} ({HEADLINE[suite]})")
    axes[0].set_ylabel("%")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _reliability(metrics: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(SUITES), figsize=(13, 4), sharey=True)
    for ax, suite in zip(axes, SUITES):
        ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1)
        for provider, suites in metrics.items():
            bins = suites.get(suite, {}).get("reliability", [])
            ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], marker="o",
                    label=provider, color=COLORS.get(provider, "#888"))
        ax.set_title(suite)
        ax.set_xlabel("confiança")
    axes[0].set_ylabel("acerto observado")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(results_dir: Path, dataset_dir: Path, pricing_path: Path) -> Path:
    pricing = yaml.safe_load(Path(pricing_path).read_text(encoding="utf-8")) or {}
    metrics = build_metrics(results_dir, dataset_dir, pricing)
    out = Path(results_dir) / "report"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(_summary_md(metrics), encoding="utf-8")
    _hero(metrics, out / "hero.png")
    _by_difficulty(metrics, out / "by_difficulty.png")
    _reliability(metrics, out / "reliability.png")
    return out
