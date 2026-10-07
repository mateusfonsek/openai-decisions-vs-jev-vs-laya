"""HTTP com latência medida e erros classificados para o retry do runner."""
from __future__ import annotations

import time

import httpx

from bench.adapters.base import PermanentError, TransientError


def _check(r: httpx.Response) -> None:
    if r.status_code == 429 or r.status_code >= 500:
        raise TransientError(f"HTTP {r.status_code}")
    if r.status_code >= 400:
        raise PermanentError(f"HTTP {r.status_code}: {r.text[:200]}")


def post_json(client: httpx.Client, url: str, headers: dict, payload: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    try:
        r = client.post(url, json=payload, headers=headers)
    except httpx.TransportError as e:
        raise TransientError(f"{type(e).__name__}: {e}") from e
    ms = (time.perf_counter() - t0) * 1000
    _check(r)
    try:
        return r.json(), ms
    except ValueError as e:
        raise PermanentError("resposta não é JSON") from e


def timed_get(client: httpx.Client, url: str, headers: dict) -> float:
    t0 = time.perf_counter()
    try:
        r = client.get(url, headers=headers)
    except httpx.TransportError as e:
        raise TransientError(f"{type(e).__name__}: {e}") from e
    ms = (time.perf_counter() - t0) * 1000
    _check(r)
    return ms
