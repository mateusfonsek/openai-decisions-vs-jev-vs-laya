"""Formato neutro compartilhado por todos os adapters."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

QType = Literal["choice", "predicate", "score"]


class TransientError(Exception):
    """Falha que vale tentar de novo (429, 5xx, rede)."""


class PermanentError(Exception):
    """Falha que não melhora com retry (4xx, resposta inválida)."""


@dataclass(frozen=True)
class Question:
    type: QType
    name: str
    instructions: str
    options: dict[str, str] | None = None  # choice: valor -> descrição
    levels: list[dict[str, str]] | None = None  # score: [{label, description}], índice 0 = nota 1
    criteria: dict[str, str] | None = None  # predicate: {"true": ..., "false": ...} opcional


@dataclass(frozen=True)
class Request:
    case_id: str
    input: str
    question: Question


@dataclass
class Response:
    case_id: str
    provider: str
    answer: Any = None  # choice: str | predicate: float P(true) | score: int 1..N
    probabilities: dict[str, float] = field(default_factory=dict)
    provider_confidence: float | None = None
    latency_ms: float | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Response":
        return cls(**d)


class Adapter(Protocol):
    name: str

    def decide(self, request: Request) -> Response: ...

    def ping(self) -> float | None:
        """RTT em ms de uma chamada leve; None para provedores locais."""
        ...


def normalize(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    if total <= 0:
        raise PermanentError("probabilidades somam zero")
    return {k: v / total for k, v in probs.items()}


def argmax(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)
