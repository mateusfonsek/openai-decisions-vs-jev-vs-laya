"""Gráficos do relatório, com o logo de cada provedor no lugar de cores (texto se faltar o logo)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402

from bench.dataset import DIFFICULTIES  # noqa: E402
from bench.metrics import HEADLINE  # noqa: E402
from bench.tables import NAMES, ORDER, SUITE_COLUMNS  # noqa: E402

LOGO_FILES = {"openai": "openai.png", "jev": "typesafe.png", "laya": "laya.png", "laya-rot": "laya.png"}
HEADLINE_LABEL = {"accuracy": "Acurácia (%)", "f1": "F1 (%)", "exact": "Nota exata (%)"}
DIFF_LABEL = {"easy": "fácil", "ambiguous": "ambíguo", "adversarial": "adversarial"}
DIFF_SHADES = {"easy": "#c9d6e3", "ambiguous": "#7f9cb8", "adversarial": "#2f4b66"}
LOGO_PX = 30
ARROW = {"arrowstyle": "-", "color": "#999", "linewidth": 0.8}


def spread_offsets(points: list[tuple[float, float]], min_dist: float) -> list[tuple[float, float]]:
    """Deslocamentos (mesma unidade dos pontos) para que nenhum par de logos fique a menos de
    min_dist; pontos já afastados não se movem."""
    steps = [(0, 0)] + [(dx * k, dy * k) for k in (1, 2, 3)
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))]
    placed: list[tuple[float, float]] = []
    offsets = []
    for x, y in points:
        for sx, sy in steps:
            cand = (x + sx * min_dist, y + sy * min_dist)
            if all(((cand[0] - px) ** 2 + (cand[1] - py) ** 2) ** 0.5 >= min_dist for px, py in placed):
                break
        placed.append(cand)
        offsets.append((cand[0] - x, cand[1] - y))
    return offsets


def logo_for(provider: str, logos_dir: Path) -> Path | None:
    p = Path(logos_dir) / LOGO_FILES.get(provider, f"{provider}.png")
    return p if p.exists() else None


def _providers(metrics: dict, suite: str) -> list[str]:
    return [p for p in ORDER if suite in metrics.get(p, {})] + \
           [p for p in metrics if p not in ORDER and suite in metrics[p]]


def _mark(ax, provider: str, xy, logos_dir: Path, xycoords="data", offset=(0, 0), arrow=False):
    """Logo do provedor centrado em xy; nome em texto se não houver arquivo de logo."""
    path = logo_for(provider, logos_dir)
    if path is None:
        ax.annotate(NAMES.get(provider, provider), xy, xycoords=xycoords, xytext=offset,
                    textcoords="offset points", ha="center", va="center", fontsize=9, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#888"},
                    arrowprops=ARROW if arrow else None)
        return
    img = plt.imread(path)
    box = AnnotationBbox(OffsetImage(img, zoom=LOGO_PX / max(img.shape[:2])), xy, xycoords=xycoords,
                         xybox=offset, boxcoords="offset points", frameon=False,
                         arrowprops=ARROW if arrow else None)
    ax.add_artist(box)
    if provider == "laya-rot":  # mesmo logo da Laya; identifica a variante
        ax.annotate("+ rotação", xy, xycoords=xycoords, xytext=(offset[0], offset[1] - LOGO_PX * 0.75),
                    textcoords="offset points", ha="center", va="top", fontsize=7)


def _hero(metrics: dict, suite: str, path: Path, logos_dir: Path) -> None:
    key = HEADLINE[suite]
    fig, ax = plt.subplots(figsize=(8, 5))
    pts = [(p, metrics[p][suite]) for p in _providers(metrics, suite)
           if metrics[p][suite].get("latency_p50") is not None]
    xs = [m["latency_p50"] for _, m in pts]
    ys = [(m.get(key) or 0) * 100 for _, m in pts]
    ax.scatter(xs, ys, s=12, color="#555", zorder=3)  # valor exato; o logo fica ao lado se colidir
    ax.set_xscale("log")
    ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    ax.set_ylim(max(0, min(ys) - 12), min(105, max(ys) + 10))
    to_pt = 72 / fig.dpi  # pixels de tela -> pontos (unidade dos offsets)
    pts_pt = [tuple(v * to_pt for v in ax.transData.transform((x, y))) for x, y in zip(xs, ys)]
    for (p, _), x, y, off in zip(pts, xs, ys, spread_offsets(pts_pt, min_dist=LOGO_PX * 1.4)):
        _mark(ax, p, (x, y), logos_dir, offset=off, arrow=off != (0, 0))
    ax.set_xlabel("Latência p50 (ms, escala log) — mais à esquerda é mais rápido")
    ax.set_ylabel(HEADLINE_LABEL[key] + " — mais alto é melhor")
    ax.set_title(f"{SUITE_COLUMNS[suite][0]}: qualidade × latência")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _by_difficulty(metrics: dict, suite: str, path: Path, logos_dir: Path) -> None:
    key = HEADLINE[suite]
    providers = _providers(metrics, suite)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.8 / len(DIFFICULTIES)
    for j, d in enumerate(DIFFICULTIES):
        vals = [((metrics[p][suite].get("by_difficulty", {}).get(d) or {}).get(key) or 0) * 100
                for p in providers]
        ax.bar([i + (j - 1) * width for i in range(len(providers))], vals, width,
               color=DIFF_SHADES[d], label=DIFF_LABEL[d])
    ax.set_xticks(range(len(providers)), [""] * len(providers))
    for i, p in enumerate(providers):
        _mark(ax, p, (i, 0), logos_dir, xycoords=("data", "axes fraction"), offset=(0, -22))
    ax.set_ylim(0, 105)
    ax.set_ylabel(HEADLINE_LABEL[key])
    ax.set_title(f"{SUITE_COLUMNS[suite][0]}: por dificuldade")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _reliability(metrics: dict, path: Path, logos_dir: Path) -> None:
    suites = list(SUITE_COLUMNS)
    fig, axes = plt.subplots(1, len(suites), figsize=(14, 4.6), sharey=True)
    markers = ["o", "s", "^", "D"]
    for ax, suite in zip(axes, suites):
        ax.plot([0, 1], [0, 1], "--", color="#bbb", linewidth=1)
        for k, p in enumerate(_providers(metrics, suite)):
            bins = metrics[p][suite].get("reliability", [])
            if not bins:
                continue
            xs, ys = [b["conf"] for b in bins], [b["acc"] for b in bins]
            ax.plot(xs, ys, marker=markers[k % len(markers)], color="#555", linewidth=1, markersize=4)
            _mark(ax, p, (xs[-1], ys[-1]), logos_dir, offset=(16, 0))
        ax.set_xlim(0, 1.15)
        ax.set_ylim(0, 1.05)
        ax.set_title(SUITE_COLUMNS[suite][0], fontsize=10)
        ax.set_xlabel("confiança declarada")
    axes[0].set_ylabel("acerto observado (diagonal = calibrado)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_charts(metrics: dict, out_dir: Path, logos_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suite in SUITE_COLUMNS:
        if any(suite in s for s in metrics.values()):
            _hero(metrics, suite, out_dir / f"hero_{suite}.png", logos_dir)
            _by_difficulty(metrics, suite, out_dir / f"by_difficulty_{suite}.png", logos_dir)
    _reliability(metrics, out_dir / "reliability.png", logos_dir)
