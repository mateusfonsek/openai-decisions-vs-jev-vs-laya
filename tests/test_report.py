import json
import shutil
from pathlib import Path

from bench.adapters.base import Response
from bench.dataset import Case, write_cases
from bench.report import build_metrics, write_report

ROOT = Path(__file__).parent.parent


def _dataset(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    shutil.copy(ROOT / "dataset" / "suites.yaml", d / "suites.yaml")
    write_cases(d / "routing.jsonl", [Case("routing-001", "routing", "easy", "synthetic", "a", "calendar"),
                                      Case("routing-002", "routing", "adversarial", "synthetic", "b", "email")])
    write_cases(d / "injection.jsonl", [Case("injection-001", "injection", "easy", "synthetic", "c", True),
                                        Case("injection-002", "injection", "adversarial", "synthetic", "d", False)])
    write_cases(d / "judge.jsonl", [Case("judge-001", "judge", "easy", "synthetic", "e", 5)])
    return d


def _results(tmp_path):
    r = tmp_path / "results" / "2026-10-08"
    r.mkdir(parents=True)
    rows = [
        Response("routing-001", "jev", "calendar", {"calendar": 0.9, "email": 0.1}, None, 100.0,
                 {"input_tokens": 1000, "output_tokens": 0}),
        Response("routing-002", "jev", "calendar", {"calendar": 0.6, "email": 0.4}, None, 300.0,
                 {"input_tokens": 1000, "output_tokens": 0}),
        Response("injection-001", "jev", 0.8, {"true": 0.8, "false": 0.2}, None, 200.0,
                 {"input_tokens": 1000, "output_tokens": 0}),
        Response("injection-002", "jev", error="TransientError: 429"),
        Response("judge-001", "jev", 5, {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1.0}, None, 150.0,
                 {"input_tokens": 1000, "output_tokens": 0}),
    ]
    (r / "jev.jsonl").write_text("".join(json.dumps(x.to_dict()) + "\n" for x in rows))
    (r / "jev.rtt.json").write_text(json.dumps({"before": [50.0, 50.0], "after": [50.0]}))
    return r


PRICING = {"jev": {"input_per_m": 0.042, "output_per_m": 0.0}}


def test_build_metrics(tmp_path):
    m = build_metrics(_results(tmp_path), _dataset(tmp_path), PRICING)
    routing = m["jev"]["routing"]
    assert routing["accuracy"] == 0.5
    assert routing["latency_p50"] == 200.0 and routing["latency_p50_net"] == 150.0
    assert routing["cost_per_1k"] == 0.042  # 1000 tokens × 0,042/1M × 1000 chamadas
    assert routing["by_difficulty"]["adversarial"]["accuracy"] == 0.0
    assert m["jev"]["injection"]["failure_rate"] == 0.5


def test_write_report_creates_files(tmp_path):
    results = _results(tmp_path)
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text("jev: {input_per_m: 0.042, output_per_m: 0.0}\n")
    out = write_report(results, _dataset(tmp_path), pricing)
    for name in ["metrics.json", "summary.md", "hero.png", "by_difficulty.png", "reliability.png"]:
        assert (out / name).exists(), name
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "jev" in summary and "routing" in summary


def test_net_latency_is_none_when_ping_slower_than_decision(tmp_path):
    results = _results(tmp_path)
    (results / "jev.rtt.json").write_text(json.dumps({"before": [900.0], "after": [900.0]}))
    m = build_metrics(results, _dataset(tmp_path), PRICING)
    assert m["jev"]["routing"]["latency_p50_net"] is None
