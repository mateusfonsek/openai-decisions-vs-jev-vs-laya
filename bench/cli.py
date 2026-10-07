"""CLI: bench validate | run | report | review-export | review-import."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from bench.dataset import SUITES, check_balance, load_cases, load_suites, to_request, validate_cases

KEYS = {"openai": "OPENAI_API_KEY", "jev": "TYPESAFE_API_KEY"}


def _adapter(provider: str):
    if provider in KEYS:
        key = os.environ.get(KEYS[provider], "").strip()
        if not key:
            raise SystemExit(f"erro: defina {KEYS[provider]} no .env para rodar '{provider}'")
        if provider == "openai":
            from bench.adapters.openai import OpenAIAdapter
            return OpenAIAdapter(key)
        from bench.adapters.jev import JevAdapter
        return JevAdapter(key)
    from bench.adapters.laya import LayaAdapter
    return LayaAdapter()


def cmd_validate(args) -> int:
    d = Path(args.dataset)
    questions = load_suites(d / "suites.yaml")
    problems = 0
    for suite in SUITES:
        cases = load_cases(d / f"{suite}.jsonl")
        errors = validate_cases(cases, suite, questions[suite])
        if not args.no_balance:
            errors += check_balance(cases, questions[suite])
        print(f"{suite}: {len(cases)} casos, {len(errors)} problemas")
        for e in errors:
            print(f"  - {e}")
        problems += len(errors)
    return 1 if problems else 0


def cmd_run(args) -> int:
    from bench.runner import measure_rtt, run_requests

    try:
        adapter = _adapter(args.provider)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    d = Path(args.dataset)
    questions = load_suites(d / "suites.yaml")
    suites = SUITES if args.suite == "all" else (args.suite,)
    requests = []
    for suite in suites:
        cases = load_cases(d / f"{suite}.jsonl")
        if args.limit:
            cases = cases[: args.limit]
        requests += [to_request(c, questions[suite]) for c in cases]
    out_dir = Path(args.results) / (args.date or date.today().isoformat())
    out_dir.mkdir(parents=True, exist_ok=True)
    rtt_path = out_dir / f"{args.provider}.rtt.json"
    rtt = json.loads(rtt_path.read_text()) if rtt_path.exists() else {"before": [], "after": []}
    rtt["before"] += measure_rtt(adapter)
    stats = run_requests(adapter, requests, out_dir / f"{args.provider}.jsonl")
    rtt["after"] += measure_rtt(adapter)
    rtt_path.write_text(json.dumps(rtt))
    print(f"{args.provider}: {stats} → {out_dir}")
    return 0


def cmd_report(args) -> int:
    from bench.report import write_report

    out = write_report(Path(args.results_dir), Path(args.dataset), Path(args.pricing))
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"relatório em {out}")
    return 0


def cmd_review_export(args) -> int:
    from bench.review import export_review

    n = export_review(Path(args.dataset), Path(args.out))
    print(f"{n} casos exportados para {args.out}")
    return 0


def cmd_review_import(args) -> int:
    from bench.review import import_review

    try:
        rep = import_review(Path(args.csv), Path(args.dataset))
    except ValueError as e:
        print(f"erro: {e}", file=sys.stderr)
        return 2
    print(f"mantidos: {rep['kept']} | corrigidos: {len(rep['corrected'])} | removidos: {len(rep['removed'])}")
    for cid in rep["removed"]:
        print(f"  removido: {cid}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="bench")
    p.add_argument("--dataset", default="dataset")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="valida schema e balanceamento do dataset")
    v.add_argument("--dataset", default=argparse.SUPPRESS)
    v.add_argument("--no-balance", action="store_true")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="roda um provedor sobre o dataset")
    r.add_argument("--dataset", default=argparse.SUPPRESS)
    r.add_argument("--provider", choices=["openai", "jev", "laya"], required=True)
    r.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    r.add_argument("--limit", type=int, default=0, help="casos por suíte (0 = todos)")
    r.add_argument("--results", default="results")
    r.add_argument("--date", default=None)
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="gera métricas, summary.md e gráficos")
    rp.add_argument("results_dir")
    rp.add_argument("--dataset", default=argparse.SUPPRESS)
    rp.add_argument("--pricing", default="pricing.yaml")
    rp.set_defaults(func=cmd_report)

    e = sub.add_parser("review-export", help="exporta CSV para revisão humana")
    e.add_argument("--dataset", default=argparse.SUPPRESS)
    e.add_argument("--out", default="review.csv")
    e.set_defaults(func=cmd_review_export)

    i = sub.add_parser("review-import", help="aplica o CSV revisado ao dataset")
    i.add_argument("csv")
    i.add_argument("--dataset", default=argparse.SUPPRESS)
    i.set_defaults(func=cmd_review_import)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
