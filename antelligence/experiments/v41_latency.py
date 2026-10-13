"""Latency benchmark (goal 4.7), separate from evaluation.

At a fixed concurrency (1 and 8), 200 development decisions go to each backend; the report gives seconds per decision
as server compute plus network, and summed. Also: mode A against mode B board overhead, and anchor latency per outside
system. Latency logged during evaluation is a secondary number ("under load"); these are the headline numbers.

The decisions are real development states: a scripted S episode is walked on dev cases and every squad-round is rendered
as a typed request (no model is involved in producing them). Everything here is a function of injected backends, so the
tests run on fakes.
"""

from __future__ import annotations

import asyncio
import hashlib
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.anchoring.anchorer import MODES
from antelligence.kernel.board import Board, generate_run_keys
from antelligence.kernel.decision import DecisionBackend, DecisionRequest, DecisionResponse
from antelligence.kernel.scheduler import derive_seed
from antelligence.kernel.types import LocalView
from antelligence.worlds.tumor_mri import (ScriptS, TumorMRIWorld, WorldConfig, build_request, select_candidates)

CONCURRENCIES = (1, 8)
N_DECISIONS = 200


def harvest_requests(load_case: Callable[..., Tuple[Any, Any]], cases: Sequence[str], model: str, n: int = N_DECISIONS,
                     config: Optional[WorldConfig] = None, seed: int = 1) -> List[DecisionRequest]:
    """``n`` squad-round requests for ``model`` from scripted S episodes on ``cases`` (development cases only: the
    caller passes the dev list; this function never sees the split)."""
    cfg = config or WorldConfig()
    out: List[DecisionRequest] = []
    episodes = []
    for case in cases:
        crop, regions = load_case(case)
        w = TumorMRIWorld(crop, cfg, seed=seed, regions=regions)
        w.set_candidates(select_candidates(w.regions))
        episodes.append((case, w, ScriptS()))
    tick = 0
    while len(out) < n and tick < cfg.ticks:
        if tick % cfg.decision_every == 0:
            for case, w, policy in episodes:
                for a in w.agents():
                    obs = w.observe(a, tick)
                    if len(out) < n:
                        out.append(build_request(model, obs.data, seed=derive_seed(seed, f"{case}|{a}", tick)))
                    w.apply(a, policy.decide(LocalView(agent_id=a, tick=tick, revision=w.revision, seed=seed,
                                                       observation=obs.data)), tick)
        for _, w, _ in episodes:
            w.step_environment(tick)
        tick += 1
    if len(out) < n:
        raise ValueError(f"only {len(out)} states available for {n} decisions; add cases")
    return out


def _mean(xs: Sequence[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


async def bench_backend(backend: DecisionBackend, requests: Sequence[DecisionRequest], concurrency: int) -> Dict[str, Any]:
    """Send every request with at most ``concurrency`` in flight. Per decision: server, network and queue seconds as
    the backend reports them, the summed latency, and the wall-clock seconds per decision (throughput view)."""
    sem = asyncio.Semaphore(concurrency)
    replies: List[DecisionResponse] = []
    errors = 0

    async def one(req: DecisionRequest) -> None:
        nonlocal errors
        async with sem:
            try:
                replies.append(await backend.decide(req))
            except Exception:  # noqa: BLE001  (counted; a benchmark reports failures, it does not hide them)
                errors += 1

    t0 = time.perf_counter()
    await asyncio.gather(*(one(r) for r in requests))
    wall = time.perf_counter() - t0
    server = [r.timing.get("server_ms", 0) / 1000 for r in replies]
    network = [r.timing.get("network_ms", 0) / 1000 for r in replies]
    queue = [r.timing.get("queue_ms", 0) / 1000 for r in replies]
    total = [s + n + q for s, n, q in zip(server, network, queue)]
    elapsed = sorted(r.elapsed_s for r in replies)
    return {"concurrency": concurrency, "n": len(requests), "ok": len(replies), "errors": errors,
            "server_s_per_decision": _mean(server), "network_s_per_decision": _mean(network),
            "queue_s_per_decision": _mean(queue), "summed_s_per_decision": _mean(total),
            "elapsed_s_mean": _mean(elapsed), "elapsed_s_p95": elapsed[int(0.95 * (len(elapsed) - 1))] if elapsed else 0.0,
            "wall_s_per_decision": wall / max(1, len(requests)), "wall_s": wall,
            "cached_replies": sum(1 for r in replies if r.cached)}


async def bench_all(backends: Mapping[str, DecisionBackend], requests_by_backend: Mapping[str, Sequence[DecisionRequest]],
                    concurrencies: Sequence[int] = CONCURRENCIES) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, backend in backends.items():
        out[name] = {f"c{c}": await bench_backend(backend, requests_by_backend[name], c) for c in concurrencies}
    return out


def bench_board_modes(n_entries: int = 500, repeats: int = 3) -> Dict[str, Any]:
    """Cost of mode A (hash chain only) against mode B (hash chain plus an ed25519 signature on every entry): the same
    ``n_entries`` decision entries appended to a fresh board, best of ``repeats`` (milliseconds per entry)."""
    keys = generate_run_keys()
    header = {"study": "v4.1", "arm": "latency", "case": "synthetic", "seed": 0, "split": "dev",
              "world_cfg_sha256": hashlib.sha256(b"w").hexdigest(), "harness_cfg_sha256": hashlib.sha256(b"h").hexdigest(),
              "pubkeys": keys.pubkeys, "anchor_pubkey": "0" * 64, "code_sha": "0" * 40}

    def payload(i: int) -> Dict[str, Any]:
        return {"squad": i % 8, "round": i // 8, "target": "ABCDEFSX"[i % 8], "release": i % 2, "amount_level": 1 + i % 5,
                "p_target_ppm": 500_000, "p_release_ppm": 600_000, "p_amount_ppm": 400_000, "backend": "bench",
                "deferred": 0, "router_pick": "bench", "brief_ids": []}

    best: Dict[str, float] = {}
    for mode in ("A", "B"):
        times = []
        for _ in range(repeats):
            board = Board(header, mode=mode, keys=keys.seeds if mode == "B" else None)
            t0 = time.perf_counter()
            for i in range(n_entries):
                board.append("decision", f"squad-{i % 8}", payload(i), i // 8)
            times.append((time.perf_counter() - t0) / n_entries * 1000)
            board.close()
        best[mode] = min(times)
    return {"n_entries": n_entries, "A_ms_per_entry": best["A"], "B_ms_per_entry": best["B"],
            "B_minus_A_ms_per_entry": best["B"] - best["A"],
            "B_over_A": best["B"] / best["A"] if best["A"] else None}


def anchor_latency(run_logs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Seconds per anchor, per outside system (OTS, R, C), over the anchored runs' logs. Mode C is the cost of enqueueing
    (the transaction lands later); its confirmation latency is the chain client's, reported by the plumbing run."""
    out: Dict[str, Any] = {}
    for mode in MODES:
        xs = [x for log in run_logs for x in (log.get("anchor_latency_s") or {}).get(mode, [])]
        out[mode] = {"n": len(xs), "mean_s": _mean(xs), "max_s": max(xs) if xs else 0.0}
    return out


def bench_anchors(anchorer_factory: Callable[[Board, Any, Path], Any], n_entries: int = 150) -> Dict[str, Any]:
    """Anchor latency per system: anchor a synthetic signed board of ``n_entries`` at its anchor points through the given
    factory (the same one the harness uses) and read the anchorer's own timings."""
    from antelligence.anchoring import mocks
    from antelligence.kernel.board import anchor_sizes

    board, _, akey = mocks.synthetic_run(n_entries)
    with tempfile.TemporaryDirectory() as tmp:
        anchorer = anchorer_factory(board, akey, Path(tmp))
        for size in anchor_sizes(board.size()):
            anchorer.anchor(board, size)
        return anchor_latency([{"anchor_latency_s": anchorer.latency_s}])
