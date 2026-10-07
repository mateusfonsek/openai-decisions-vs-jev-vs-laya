"""Laya (ConvAI Innovations), executada localmente."""
from __future__ import annotations

import time

from bench.adapters.base import PermanentError, Request, Response, argmax, normalize
from bench.adapters.keyed import parse_keyed_answer, to_keyed_question


class LayaAdapter:
    def __init__(self, router=None, model: str = "multilingual", rotate: bool = False):
        if router is None:
            from laya import Router  # extra opcional: uv sync --extra laya

            router = Router()
        self.router = router
        self.model = model  # explícito: o roteamento automático poderia escolher o checkpoint em inglês
        # rotate: média das probabilidades sobre todas as rotações da ordem das opções (choice),
        # mitigação do viés de posição documentada pelo próprio projeto Laya
        self.rotate = rotate
        self.name = "laya-rot" if rotate else "laya"

    def _predict(self, req: Request, qd: dict):
        q = req.question
        out = self.router.predict(req.input, {q.name: qd}, model=self.model)
        a = (out.get("answers") or {}).get(q.name)
        if a is None:
            raise PermanentError("resposta sem answers para a pergunta")
        return out, parse_keyed_answer(q, a)

    def decide(self, req: Request) -> Response:
        q = req.question
        qd = to_keyed_question(q)
        t0 = time.perf_counter()
        if self.rotate and q.type == "choice":
            items = list(qd["criteria"].items())
            raws, total = [], {opt: 0.0 for opt in q.options}
            for i in range(len(items)):
                out, (_, probs, _) = self._predict(req, dict(qd, criteria=dict(items[i:] + items[:i])))
                raws.append(out)
                for opt, p in probs.items():
                    total[opt] += p
            probs = normalize(total)
            answer, conf, raw = argmax(probs), None, {"rotations": raws}
        else:
            raw, (answer, probs, conf) = self._predict(req, qd)
        ms = (time.perf_counter() - t0) * 1000
        return Response(case_id=req.case_id, provider=self.name, answer=answer, probabilities=probs,
                        provider_confidence=conf, latency_ms=ms,
                        usage={"input_tokens": 0, "output_tokens": 0}, raw=raw)

    def ping(self) -> None:
        return None
