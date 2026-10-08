from rich.console import Console

from bench.tables import markdown_tables, rich_tables

M = {
    "openai": {"routing": {"accuracy": 0.8, "macro_f1": 0.79, "ece": 0.04, "latency_p50": 200.0,
                           "latency_p50_net": 70.0, "latency_p95": 900.0, "cost_per_1k": 0.0256,
                           "by_difficulty": {"easy": {"accuracy": 0.9}, "ambiguous": {"accuracy": 0.7},
                                             "adversarial": {"accuracy": 0.8}}}},
    "laya": {"routing": {"accuracy": 0.3, "macro_f1": 0.3, "ece": 0.4, "latency_p50": 24.0,
                         "latency_p50_net": 24.0, "latency_p95": 30.0, "cost_per_1k": 0.0,
                         "by_difficulty": {"easy": {"accuracy": 0.4}, "ambiguous": {"accuracy": 0.2},
                                           "adversarial": {"accuracy": 0.3}}}},
}


def test_markdown_has_one_table_per_suite_present_and_bolds_best():
    md = markdown_tables(M)
    assert md.count("### ") == 1 and "Roteamento de agente" in md
    assert "| OpenAI | **80,0%**" in md          # melhor acurácia em negrito
    assert "**24**" in md                          # menor latência em negrito
    assert "**US$ 0,0000**" in md                  # menor custo em negrito


def test_rich_tables_render_titles_and_display_names():
    console = Console(record=True, width=200)
    for t in rich_tables(M):
        console.print(t)
    out = console.export_text()
    assert "Roteamento de agente" in out and "OpenAI" in out and "Laya" in out


def test_values_equal_after_rounding_are_all_bold():
    m = {p: {"injection": {"auroc": v, "latency_p50": 23.0 + d}}
         for p, v, d in [("openai", 0.99271, 0.0), ("jev", 0.99324, 0.4), ("laya", 0.725, 0.1)]}
    md = markdown_tables(m)
    assert md.count("**0,993**") == 2           # 0,993 x 0,993: os dois em negrito
    assert md.count("**23**") == 3              # 23,0 / 23,4 / 23,1 aparecem como 23


def test_judge_within_one_column_name():
    m = {"jev": {"judge": {"exact": 0.8, "within_1": 0.98, "mae": 0.2}}}
    assert "Acerto ±1 ponto" in markdown_tables(m)
