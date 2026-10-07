"""Execução sequencial, retomável, com retry e medição de rede."""
from __future__ import annotations

import json
import time
from pathlib import Path

from bench.adapters.base import Request, Response, TransientError


def load_results(path: Path) -> dict[str, Response]:
    results: dict[str, Response] = {}
    path = Path(path)
    if not path.exists():
        return results
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = Response.from_dict(json.loads(line))
        except (ValueError, TypeError):
            continue  # linha truncada por interrupção
        results[r.case_id] = r
    return results


def _decide_with_retry(adapter, req: Request, max_retries: int, backoff: float, sleep) -> Response:
    last = ""
    for attempt in range(max_retries):
        try:
            return adapter.decide(req)
        except TransientError as e:
            last = str(e)
            if attempt < max_retries - 1:
                sleep(backoff * 2**attempt)
        except Exception as e:  # PermanentError e qualquer bug do adapter viram falha registrada
            return Response(case_id=req.case_id, provider=adapter.name, error=f"{type(e).__name__}: {e}")
    return Response(case_id=req.case_id, provider=adapter.name, error=f"TransientError: {last}")


def run_requests(adapter, requests: list[Request], out_path: Path, warmup: int = 3,
                 max_retries: int = 4, backoff: float = 1.0, sleep=time.sleep) -> dict:
    out_path = Path(out_path)
    done = {cid for cid, r in load_results(out_path).items() if r.error is None}
    pending = [r for r in requests if r.case_id not in done]
    stats = {"ok": 0, "failed": 0, "skipped": len(requests) - len(pending)}
    if not pending:
        return stats
    for _ in range(warmup):
        try:
            adapter.decide(pending[0])
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size and not out_path.read_bytes().endswith(b"\n"):
        with out_path.open("a", encoding="utf-8") as f:
            f.write("\n")  # isola a linha truncada
    with out_path.open("a", encoding="utf-8") as f:
        for req in pending:
            resp = _decide_with_retry(adapter, req, max_retries, backoff, sleep)
            f.write(json.dumps(resp.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            stats["failed" if resp.error else "ok"] += 1
    return stats


def measure_rtt(adapter, n: int = 10) -> list[float]:
    samples = []
    for _ in range(n):
        try:
            ms = adapter.ping()
        except Exception:
            continue
        if ms is None:
            return []
        samples.append(ms)
    return samples
