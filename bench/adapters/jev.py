"""Jev, da TypeSafe (POST /v1/systemone)."""
from __future__ import annotations

import httpx

from bench.adapters.base import PermanentError, Request, Response
from bench.adapters.http import post_json, timed_get
from bench.adapters.keyed import parse_keyed_answer, to_keyed_question

URL = "https://api.typesafe.ai/v1/systemone"
PING_URL = "https://api.typesafe.ai/v1/models"


class JevAdapter:
    name = "jev"

    def __init__(self, api_key: str, model: str = "jev-latest", client: httpx.Client | None = None):
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.client = client or httpx.Client(timeout=30)

    def build_payload(self, req: Request) -> dict:
        q = req.question
        return {"state": req.input, "model": self.model, "questions": {q.name: to_keyed_question(q)}}

    def decide(self, req: Request) -> Response:
        data, ms = post_json(self.client, URL, self.headers, self.build_payload(req))
        a = (data.get("answers") or {}).get(req.question.name)
        if a is None:
            raise PermanentError("resposta sem answers para a pergunta")
        answer, probs, conf = parse_keyed_answer(req.question, a)
        u = data.get("usage") or {}
        usage = {"input_tokens": int(u.get("input_tokens", 0)), "output_tokens": int(u.get("output_tokens", 0))}
        return Response(case_id=req.case_id, provider=self.name, answer=answer, probabilities=probs,
                        provider_confidence=conf, latency_ms=ms, usage=usage, raw=data)

    def ping(self) -> float:
        return timed_get(self.client, PING_URL, self.headers)
