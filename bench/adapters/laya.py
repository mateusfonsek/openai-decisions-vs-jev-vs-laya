"""Laya (ConvAI Innovations), executada localmente."""
from __future__ import annotations

import time

from bench.adapters.base import PermanentError, Request, Response
from bench.adapters.keyed import parse_keyed_answer, to_keyed_question


class LayaAdapter:
    name = "laya"

    def __init__(self, router=None, model: str = "multilingual"):
        if router is None:
            from laya import Router  # extra opcional: uv sync --extra laya

            router = Router()
        self.router = router
        self.model = model  # explícito: o roteamento automático poderia escolher o checkpoint em inglês

    def decide(self, req: Request) -> Response:
        q = req.question
        t0 = time.perf_counter()
        out = self.router.predict(req.input, {q.name: to_keyed_question(q)}, model=self.model)
        ms = (time.perf_counter() - t0) * 1000
        a = (out.get("answers") or {}).get(q.name)
        if a is None:
            raise PermanentError("resposta sem answers para a pergunta")
        answer, probs, conf = parse_keyed_answer(q, a)
        return Response(case_id=req.case_id, provider=self.name, answer=answer, probabilities=probs,
                        provider_confidence=conf, latency_ms=ms,
                        usage={"input_tokens": 0, "output_tokens": 0}, raw=out)

    def ping(self) -> None:
        return None
