"""OpenAI Decisions API (POST /v1/decisions)."""
from __future__ import annotations

import httpx

from bench.adapters.base import PermanentError, Request, Response, argmax, normalize
from bench.adapters.http import post_json, timed_unauth_post

URL = "https://api.openai.com/v1/decisions"


def _usage(data: dict) -> dict[str, int]:
    u = data.get("usage") or {}
    return {"input_tokens": int(u.get("input_tokens", 0)), "output_tokens": int(u.get("output_tokens", 0))}


class OpenAIAdapter:
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-6-luna", client: httpx.Client | None = None):
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.client = client or httpx.Client(timeout=30)

    def build_payload(self, req: Request) -> dict:
        q = req.question
        qd = {"type": q.type, "name": q.name, "instructions": q.instructions}
        if q.type == "choice":
            qd["choices"] = [{"value": k, "description": v} for k, v in q.options.items()]
        elif q.type == "score":
            qd["levels"] = [{"label": lv["label"], "description": lv["description"]} for lv in q.levels]
        return {"model": self.model, "input": req.input, "questions": [qd]}

    def _parse(self, req: Request, data: dict):
        q = req.question
        ans = next((a for a in data.get("answers") or [] if a.get("name") == q.name), None)
        if ans is None:
            raise PermanentError("resposta sem answers para a pergunta")
        if q.type == "predicate":
            p = float(ans["probability"])
            return p, {"true": p, "false": 1.0 - p}, None
        items = ans.get("probabilities") or []
        if not items:
            raise PermanentError("resposta sem 'probabilities'")
        conf = ans.get("confidence")
        conf = float(conf) if conf is not None else None
        if q.type == "choice":
            raw = {str(i["value"]): float(i["probability"]) for i in items}
            probs = normalize({opt: raw.get(opt, 0.0) for opt in q.options})
            choice = ans.get("choice", argmax(probs))
            if choice not in q.options:
                raise PermanentError(f"choice fora das opções: {choice!r}")
            return choice, probs, conf
        raw = {int(i["value"]): float(i["probability"]) for i in items}
        probs = normalize({str(i + 1): raw.get(i, 0.0) for i in range(len(q.levels))})
        return int(argmax(probs)), probs, conf

    def decide(self, req: Request) -> Response:
        data, ms = post_json(self.client, URL, self.headers, self.build_payload(req))
        answer, probs, conf = self._parse(req, data)
        return Response(case_id=req.case_id, provider=self.name, answer=answer, probabilities=probs,
                        provider_confidence=conf, latency_ms=ms, usage=_usage(data), raw=data)

    def ping(self) -> float:
        return timed_unauth_post(self.client, URL)
