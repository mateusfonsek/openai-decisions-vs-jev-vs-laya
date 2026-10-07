"""Uma tabela por cenário: rich para o terminal, Markdown para o README."""
from __future__ import annotations

from rich import box
from rich.table import Table

NAMES = {"openai": "OpenAI", "jev": "Jev (TypeSafe)", "laya": "Laya", "laya-rot": "Laya + rotação"}
ORDER = ["openai", "jev", "laya", "laya-rot"]


def _pct(v):
    return f"{v * 100:.1f}%"


def _ms(v):
    return f"{v:.0f}"


def _usd(v):
    return f"US$ {v:.4f}"


def _num(v):
    return f"{v:.3f}"


def _diff(d, key):
    return lambda m: (m.get("by_difficulty", {}).get(d) or {}).get(key)


# (cabeçalho, extrator, formato, maior_é_melhor)
SUITE_COLUMNS = {
    "routing": ("Roteamento de agente", [
        ("Acurácia", lambda m: m.get("accuracy"), _pct, True),
        ("F1 macro", lambda m: m.get("macro_f1"), _pct, True),
        ("Fácil", _diff("easy", "accuracy"), _pct, True),
        ("Ambíguo", _diff("ambiguous", "accuracy"), _pct, True),
        ("Adversarial", _diff("adversarial", "accuracy"), _pct, True),
    ]),
    "injection": ("Detecção de prompt injection", [
        ("Precisão", lambda m: m.get("precision"), _pct, True),
        ("Recall", lambda m: m.get("recall"), _pct, True),
        ("F1", lambda m: m.get("f1"), _pct, True),
        ("AUROC", lambda m: m.get("auroc"), _num, True),
        ("Falso positivo em negativos difíceis", lambda m: m.get("fpr_hard_negatives"), _pct, False),
    ]),
    "judge": ("LLM-as-judge (nota de 1 a 5)", [
        ("Nota exata", lambda m: m.get("exact"), _pct, True),
        ("Erro de até 1 ponto", lambda m: m.get("within_1"), _pct, True),
        ("Erro médio", lambda m: m.get("mae"), lambda v: f"{v:.2f}", False),
    ]),
}
COMMON_COLUMNS = [
    ("ECE", lambda m: m.get("ece"), _pct, False),
    ("p50 (ms)", lambda m: m.get("latency_p50"), _ms, False),
    ("p50 sem rede (ms)", lambda m: m.get("latency_p50_net"), _ms, False),
    ("p95 (ms)", lambda m: m.get("latency_p95"), _ms, False),
    ("Custo por 1k", lambda m: m.get("cost_per_1k"), _usd, False),
]


def _rows(metrics: dict, suite: str):
    """[(nome, [(texto, é_melhor), ...])] para os provedores com dados na suíte."""
    title, cols = SUITE_COLUMNS[suite]
    cols = cols + COMMON_COLUMNS
    providers = [p for p in ORDER if suite in metrics.get(p, {})] + \
                [p for p in metrics if p not in ORDER and suite in metrics[p]]
    values = {p: [get(metrics[p][suite]) for _, get, _, _ in cols] for p in providers}
    best = []
    for i, (_, _, _, higher) in enumerate(cols):
        present = [values[p][i] for p in providers if values[p][i] is not None]
        best.append((max if higher else min)(present) if present else None)
    rows = [(NAMES.get(p, p), [("—" if v is None else fmt(v), v is not None and v == best[i])
                               for i, ((_, _, fmt, _), v) in enumerate(zip(cols, values[p]))])
            for p in providers]
    return title, [c[0] for c in cols], rows


def markdown_tables(metrics: dict) -> str:
    parts = []
    for suite in SUITE_COLUMNS:
        if not any(suite in s for s in metrics.values()):
            continue
        title, headers, rows = _rows(metrics, suite)
        lines = [f"### {title}", "", "| Provedor | " + " | ".join(headers) + " |",
                 "|---|" + "---|" * len(headers)]
        for name, cells in rows:
            lines.append(f"| {name} | " + " | ".join(f"**{t}**" if b else t for t, b in cells) + " |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"


def rich_tables(metrics: dict) -> list[Table]:
    tables = []
    for suite in SUITE_COLUMNS:
        if not any(suite in s for s in metrics.values()):
            continue
        title, headers, rows = _rows(metrics, suite)
        t = Table(title=title, box=box.ROUNDED, title_style="bold", header_style="bold cyan")
        t.add_column("Provedor", style="bold")
        for h in headers:
            t.add_column(h, justify="right")
        for name, cells in rows:
            t.add_row(name, *[f"[bold green]{tx}[/]" if b else tx for tx, b in cells])
        tables.append(t)
    return tables
