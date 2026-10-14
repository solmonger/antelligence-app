"""v4.2 C0: seeding helpers and a sleeping fake backend for throughput profiling (fakes only; no network, no keys).

``seed64`` / ``rng`` are the one seeding rule of ``docs/v4.2/INTERFACES.md`` section 1. The latency stream used here,
``c0_latency``, is C0-local: it only chooses how long a fake backend sleeps, never a decision, a metric or a world draw,
so it adds no row to the section 1 table that other workstreams code against.

``SleepingBackend`` wraps a ``v41_fake`` worker backend and awaits ``asyncio.sleep(d)`` before it answers, with d drawn
from a recorded per-model latency table (101 quantiles, written by ``scripts/v42_attribute.py``). The draw is a pure
function of (model, request hash), so a run is repeatable and two runs of the same case sleep the same way. The answer
itself is the wrapped fake's, so metrics do not depend on sleeping at all (a test pins that).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from antelligence.experiments.v41_fake import FAKE_IDENTITY, FakeWorkerBackend, build_fake_queen
from antelligence.experiments.v41_harness import BONSAI, D1, FW, Q4, Q9, Worker
from antelligence.kernel.decision import DecisionRequest, DecisionResponse

LATENCY_STREAM = "c0_latency"


def seed64(*parts: Any) -> int:
    """INTERFACES section 1: first 8 bytes of sha256 of the ``|``-joined parts, big endian."""
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big")


def rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(seed64(*parts))


class LatencyTable:
    """Per-model latency quantiles in seconds; ``draw`` maps a uniform u in [0, 1) to a latency by interpolation."""

    def __init__(self, models: Mapping[str, Sequence[float]], quantiles: Optional[Sequence[float]] = None) -> None:
        self.models = {m: np.asarray(q, dtype=np.float64) for m, q in models.items()}
        self.quantiles = np.asarray(quantiles if quantiles is not None else np.linspace(0, 1, 101), dtype=np.float64)
        for m, q in self.models.items():
            if len(q) != len(self.quantiles) or np.any(np.diff(q) < 0):
                raise ValueError(f"latency table for {m!r} must be non-decreasing with {len(self.quantiles)} points")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LatencyTable":
        doc = json.loads(Path(path).read_text())
        return cls({m: v["q"] for m, v in doc["models"].items()}, doc["quantiles"])

    def draw(self, model: str, u: float) -> float:
        # the look is a d1 call; the queen and unknown models fall back to the table's median model
        table = self.models.get(model)
        if table is None:
            table = next(iter(self.models.values()))
        return float(np.interp(u, self.quantiles, table))


class SleepingBackend:
    """A worker backend that sleeps a recorded-distribution latency (times ``time_scale``) then answers like the fake."""

    def __init__(self, inner: FakeWorkerBackend, table: LatencyTable, *, time_scale: float = 1.0,
                 model_key: Optional[str] = None) -> None:
        self.inner, self.table, self.time_scale = inner, table, float(time_scale)
        self.model_key = model_key or inner.name
        self.model, self.name = inner.model, inner.name
        self.slept_s = 0.0
        self.calls = 0

    def describe(self) -> Dict[str, Any]:
        return self.inner.describe()

    def latency_s(self, request: DecisionRequest) -> float:
        u = float(rng(LATENCY_STREAM, self.model_key, request.request_hash).random())
        return self.table.draw(self.model_key, u) * self.time_scale

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        d = self.latency_s(request)
        await asyncio.sleep(d)
        self.slept_s += d
        self.calls += 1
        resp = await self.inner.decide(request)
        self.inner.requests.clear()      # FakeBackend keeps every request forever; a long profile must not
        return _with_timing(resp, d)


def _with_timing(resp: DecisionResponse, d: float):
    import dataclasses
    return dataclasses.replace(resp, timing={"server_ms": 0, "network_ms": int(d * 1000), "queue_ms": 0},
                               elapsed_s=d)


def build_sleeping_workers(table: LatencyTable, time_scale: float = 1.0) -> Dict[str, Worker]:
    """The five v4.1 worker names, each a sleeping fake. FW and Bonsai borrow the table's d1/9B rows when absent."""
    alias = {FW: Q9, BONSAI: Q9}
    workers: Dict[str, Worker] = {}
    for name in (D1, Q4, Q9, BONSAI, FW):
        key = name if name in table.models else alias.get(name, Q9)
        workers[name] = Worker(SleepingBackend(FakeWorkerBackend(name), table, time_scale=time_scale, model_key=key))
    return workers


def build_queen():
    return build_fake_queen()


def dev_only_gate(cases: Sequence[str]):
    """A gate that admits only the named development cases and returns ``"dev"``: eval, reserve and unknown ids raise.

    The harness refuses fake backends on any non-dev split anyway; this makes the profile harness refuse them too
    before anything is loaded (C0 runs no evaluation case)."""
    allowed = frozenset(cases)

    def gate(case: str) -> str:
        if case not in allowed:
            raise PermissionError(f"C0 profiling runs development cases only; {case!r} is refused")
        return "dev"
    return gate


__all__ = ["FAKE_IDENTITY", "LatencyTable", "SleepingBackend", "build_queen", "build_sleeping_workers", "dev_only_gate",
           "rng", "seed64"]
