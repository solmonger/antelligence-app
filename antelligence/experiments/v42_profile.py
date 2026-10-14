"""v4.2 C0: engine profiling harness for the v4.1 SW-B episode path (fake backends only).

Everything here runs the real ``V41Harness`` on a development case with sleeping fake backends
(``v42_fake.SleepingBackend``), mock anchors (``anchoring.tamper.Lab``) and the in-memory ``KeyVault``. No network, no
model server, no keys (the vault derives throw-away test keys from the run label), no v4.1 cache directory, no
evaluation case (the gate admits dev cases only).

* :class:`PhaseTimer` / :func:`instrument` - exclusive wall timers per tick phase, patched in around the unmodified code.
* :class:`LagMonitor` - an asyncio task that records how late its own sleeps wake.
* :func:`run_episode`, :func:`measure_inprocess`, :func:`measure_pool` - one episode; c episodes in one process; N
  processes with one episode each.
* :func:`fingerprint` - the metrics a speed fix must leave identical.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from cryptography.hazmat.primitives.asymmetric import ec

from antelligence.anchoring.rekor import AnchorKey
from antelligence.experiments import v41_fake, v41_harness, v42_fake
from antelligence.experiments.v41_harness import ARM_SPECS, KeyVault, V41Harness
from antelligence.kernel.board import DEFAULT_AUTHORS, RunKeys, pubkey_hex

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CASES_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
LATENCY_REL = Path("docs/research/v4.2-20261014/throughput/latency_dist.json")
SPLIT_REL = Path("config/v4_1/split.json")
WORLD_REL = Path("config/v4_1/world.yaml")
TICKS = 120


def dev_cases(repo: Path = REPO) -> List[str]:
    doc = json.loads((repo / SPLIT_REL).read_text())
    return list(doc["dev_a"]) + list(doc["dev_b"])


def dev_a_cases(repo: Path = REPO) -> List[str]:
    return list(json.loads((repo / SPLIT_REL).read_text())["dev_a"])


# --------------------------------------------------------------------------------------------------
# Keys: deterministic throw-away test keys (profiling only; never real signing keys)
# --------------------------------------------------------------------------------------------------

class SeededKeyVault(KeyVault):
    """``KeyVault`` with no ledger directory whose keys are a function of the run label, so two runs of the same
    (arm, case, seed) build the same header, entry ids and request hashes. These keys sign nothing real."""

    def get(self, label: str):
        if label in self._mem:
            return self._mem[label]
        seeds = {a: hashlib.sha256(f"v42-c0-test-key|{label}|{a}".encode()).digest() for a in DEFAULT_AUTHORS}
        keys = RunKeys({a: pubkey_hex(s) for a, s in seeds.items()}, seeds)
        n = int.from_bytes(hashlib.sha256(f"v42-c0-test-anchor|{label}".encode()).digest(), "big")
        order = int("FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551", 16)
        akey = AnchorKey(ec.derive_private_key(n % (order - 1) + 1, ec.SECP256R1()))
        self._mem[label] = (keys, akey)
        return keys, akey


# --------------------------------------------------------------------------------------------------
# Event-loop lag
# --------------------------------------------------------------------------------------------------

class LagMonitor:
    """Sleeps ``interval`` repeatedly and records how late each wake-up was. CPU work in the same thread delays the wake."""

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.lags: List[float] = []
        self._task: Optional[asyncio.Task] = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            t = loop.time()
            await asyncio.sleep(self.interval)
            self.lags.append(max(0.0, loop.time() - t - self.interval))

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> Dict[str, Any]:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        v = sorted(self.lags)
        if not v:
            return {"samples": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] * 1000.0   # noqa: E731
        return {"samples": len(v), "mean_ms": statistics.mean(v) * 1000.0, "p50_ms": q(0.5), "p95_ms": q(0.95),
                "p99_ms": q(0.99), "max_ms": v[-1] * 1000.0, "interval_ms": self.interval * 1000.0}


# --------------------------------------------------------------------------------------------------
# Phase timers (exclusive: a phase does not include the phases it calls)
# --------------------------------------------------------------------------------------------------

class PhaseTimer:
    def __init__(self, ticks: int = TICKS) -> None:
        self.total: Dict[str, float] = defaultdict(float)
        self.calls: Counter = Counter()
        self.per_tick: Dict[str, List[float]] = defaultdict(lambda: [0.0] * (ticks + 1))
        self.tick = 0
        self._stack: List[List[Any]] = []
        self._ticks = ticks

    def _charge(self, phase: str, dt: float) -> None:
        self.total[phase] += dt
        self.per_tick[phase][min(self.tick, self._ticks)] += dt

    def enter(self, phase: str) -> None:
        now = time.perf_counter()
        if self._stack:
            top = self._stack[-1]
            self._charge(top[0], now - top[1])
        self._stack.append([phase, now])
        self.calls[phase] += 1

    def exit(self) -> None:
        now = time.perf_counter()
        phase, start = self._stack.pop()
        self._charge(phase, now - start)
        if self._stack:
            self._stack[-1][1] = now

    def wrap(self, fn: Callable, phase: str) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def aw(*a: Any, **k: Any):
                self.enter(phase)
                try:
                    return await fn(*a, **k)
                finally:
                    self.exit()
            return aw

        @functools.wraps(fn)
        def w(*a: Any, **k: Any):
            self.enter(phase)
            try:
                return fn(*a, **k)
            finally:
                self.exit()
        return w

    def wrap_step(self, fn: Callable, phase: str) -> Callable:
        """Wrap ``TumorMRIWorld.step_environment`` and move the tick marker around it."""
        inner = self.wrap(fn, phase)

        @functools.wraps(fn)
        def w(world: Any, tick: int):
            self.tick = int(tick)
            try:
                return inner(world, tick)
            finally:
                self.tick = int(tick) + 1
        return w


PHASES: Tuple[Tuple[str, str, str], ...] = (
    # (module-or-class path, attribute, phase)
    ("antelligence.worlds.tumor_mri.physics:DrugField", "step", "world_diffusion"),
    ("antelligence.worlds.tumor_mri.world:Scorer", "coverage", "world_coverage_score"),
    ("antelligence.worlds.tumor_mri.physics:Pathfinder", "path", "world_astar"),
    ("antelligence.worlds.tumor_mri.world:TumorMRIWorld", "step_environment", "world_step_other"),
    ("antelligence.worlds.tumor_mri.world:TumorMRIWorld", "apply", "world_apply_other"),
    ("antelligence.worlds.tumor_mri.world:TumorMRIWorld", "metrics", "world_metrics"),
    ("antelligence.worlds.tumor_mri.world:TumorMRIWorld", "observe", "observation"),
    ("antelligence.experiments.v41_harness", "local_patch", "look_patch_extract"),
    ("antelligence.experiments.v41_harness", "look_request", "look_png_render"),
    ("antelligence.kernel.board:Board", "append", "board_append"),
    ("antelligence.kernel.board:Board", "entries", "board_read"),
    ("antelligence.kernel.board:Board", "by_type", "board_read"),
    ("antelligence.kernel.briefs", "build_brief", "brief_build"),
    ("antelligence.kernel.briefs", "build_brief_board_off", "brief_build"),
    ("antelligence.experiments.v41_harness:V41Harness", "_compact", "brief_build"),
    ("antelligence.experiments.v41_router:KNNRouter", "predict", "conformal_router"),
    ("antelligence.experiments.v41_harness", "router_features", "conformal_router"),
    ("antelligence.experiments.v41_conformal:ConformalDeferral", "decide_answers", "conformal_router"),
    ("antelligence.experiments.v41_harness:V41Harness", "_apply", "decision_entries"),
    ("antelligence.experiments.v41_harness:V41Harness", "_queen_step", "queen_step"),
    ("antelligence.experiments.v41_harness:V41Harness", "_write", "bundle_write"),
    ("antelligence.experiments.v41_harness:V41Harness", "_bundle", "bundle_build"),
    ("antelligence.experiments.v41_fake", "varied_answers", "fake_backend_answers"),
)


def _resolve(path: str) -> Tuple[Any, str]:
    import importlib
    mod, _, cls = path.partition(":")
    obj = importlib.import_module(mod)
    return (getattr(obj, cls) if cls else obj), path


@contextlib.contextmanager
def instrument(timer: PhaseTimer, extra: Sequence[Tuple[str, str, str]] = ()) -> Iterator[PhaseTimer]:
    """Patch the phase functions in for the duration of the block, then restore them."""
    undo: List[Tuple[Any, str, Any]] = []
    try:
        for path, attr, phase in (*PHASES, *extra):
            owner, _ = _resolve(path)
            original = owner.__dict__[attr] if isinstance(owner, type) else getattr(owner, attr)
            if isinstance(original, staticmethod):
                wrapped: Any = staticmethod(timer.wrap(original.__func__, phase))
            elif attr == "step_environment":
                wrapped = timer.wrap_step(original, phase)
            else:
                wrapped = timer.wrap(original, phase)
            undo.append((owner, attr, original))
            setattr(owner, attr, wrapped)
        yield timer
    finally:
        for owner, attr, original in reversed(undo):
            setattr(owner, attr, original)


class TimedAnchorer:
    """Wraps a mock-lab ``Anchorer`` so its hashing, signing and mock OTS/Rekor/chain work lands in ``mock_anchor``."""

    def __init__(self, inner: Any, timer: Optional[PhaseTimer]) -> None:
        self._inner, self._timer = inner, timer

    def _timed(self, name: str, *a: Any, **k: Any):
        fn = getattr(self._inner, name)
        if self._timer is None:
            return fn(*a, **k)
        self._timer.enter("mock_anchor")
        try:
            return fn(*a, **k)
        finally:
            self._timer.exit()

    def maybe_anchor(self, board):
        return self._timed("maybe_anchor", board)

    def finalize(self, board):
        return self._timed("finalize", board)

    def refresh_chain(self):
        return self._timed("refresh_chain")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# --------------------------------------------------------------------------------------------------
# Building and running
# --------------------------------------------------------------------------------------------------

@dataclass
class ProfileConfig:
    repo_root: str = str(REPO)
    cases_root: str = str(DEFAULT_CASES_ROOT)
    latency_path: str = ""
    time_scale: float = 0.01
    arm: str = "SW-B"
    work_dir: str = ""           # run files (decisions.jsonl, bundle.json, board export) go here; "" = tempdir
    anchors: bool = True
    lag_interval_s: float = 0.02

    def latency(self) -> Path:
        return Path(self.latency_path) if self.latency_path else Path(self.repo_root) / LATENCY_REL


def build_harness(cfg: ProfileConfig, timer: Optional[PhaseTimer] = None, out_dir: Optional[Path] = None, *,
                  load_case: Optional[Callable] = None, world_config: Any = None) -> V41Harness:
    """The profiling harness. ``load_case`` / ``world_config`` are test seams (a synthetic phantom instead of the MRI store)."""
    from antelligence.anchoring.tamper import Lab
    from antelligence.worlds.tumor_mri import WorldConfig
    from antelligence.worlds.tumor_mri.store_adapter import make_harness_loader
    repo = Path(cfg.repo_root)
    spec = ARM_SPECS[cfg.arm]
    table = v42_fake.LatencyTable.load(cfg.latency())
    workers = v42_fake.build_sleeping_workers(table, cfg.time_scale)
    router, thresholds = v41_fake.fake_bootstrap(False)
    bonsai_router, bonsai_thresholds = v41_fake.fake_bootstrap(True)
    anchorer_factory = post_mode_factory = reputation_client = None
    allow_unanchored = True
    if cfg.anchors and spec.anchored:
        lab = Lab()
        anchorer_factory = lambda board, akey, run_dir: TimedAnchorer(lab.anchorer(board, akey, run_dir), timer)  # noqa: E731
        allow_unanchored = False
    if spec.board == "D":
        post_mode_factory = v41_fake.MockPostMode
    if spec.reputation == "chain":
        reputation_client = v41_fake.MockReputationClient()
    loader = load_case or make_harness_loader(root=cfg.cases_root, repo_root=repo)
    return V41Harness(
        load_case=loader, gate=v42_fake.dev_only_gate(dev_cases(repo)), workers=workers,
        queen_provider=v42_fake.build_queen() if spec.uses_queen else None,
        world_config=world_config or WorldConfig.from_yaml(str(repo / WORLD_REL)), router=router, bonsai_router=bonsai_router,
        thresholds=thresholds, bonsai_thresholds=bonsai_thresholds, anchorer_factory=anchorer_factory,
        allow_unanchored=allow_unanchored, post_mode_factory=post_mode_factory, reputation_client=reputation_client,
        keys=SeededKeyVault(), out_dir=out_dir, backends_kind="fake", code_sha="0" * 40,
        hardware={n: "fake" for n in ("d1", "qwen3.5-9b", "qwen3.5-4b", "bonsai-8b", "fw")})


def fingerprint(res: Any) -> Dict[str, Any]:
    """What a speed fix must leave identical: the cell without wall-clock fields, the trace hash, the board root and the
    anchor count, the decision-log hash, and a hash of the whole decision log and bundle body."""
    cell = {k: v for k, v in res.cell.items() if k not in ("wall_s", "seconds_to_50", "visibility_latency_ms")}
    extra = res.bundle.get("extra", {})
    return {"cell": cell, "trace_hash": res.trace_hash, "board_root": extra.get("board_root"),
            "board_head": extra.get("board_head"), "decision_log_sha256": extra.get("decision_log_sha256"),
            "anchor_sizes": extra.get("anchor_sizes"), "anchors": len(res.anchors),
            "ticks": res.bundle.get("metrics", {}).get("ticks", res.run_log.get("ticks")),
            "log_sha256": hashlib.sha256(json.dumps(_scrub_wall(res.decision_log), sort_keys=True,
                                                    default=str).encode()).hexdigest()}


def _scrub_wall(log: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in ("server_ms", "network_ms", "queue_ms", "calls", "counterfactual",
                                                     "audit")} for r in log]


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


def swap_usage() -> str:
    try:
        return subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


@contextlib.contextmanager
def work_dir(cfg: ProfileConfig) -> Iterator[Path]:
    if cfg.work_dir:
        p = Path(cfg.work_dir)
        p.mkdir(parents=True, exist_ok=True)
        yield p
        return
    p = Path(tempfile.mkdtemp(prefix="v42c0-"))
    try:
        yield p
    finally:
        shutil.rmtree(p, ignore_errors=True)


async def run_episode(cfg: ProfileConfig, case: str, seed: int, *, timer: Optional[PhaseTimer] = None,
                      monitor: bool = True) -> Dict[str, Any]:
    """One episode, timed. Returns wall, CPU, slept time, lag, the fingerprint and (with a timer) the phase table."""
    with work_dir(cfg) as out:
        h = build_harness(cfg, timer, out)
        mon = LagMonitor(cfg.lag_interval_s)
        slept0 = sum(w.backend.slept_s for w in h.workers.values())
        cpu0, t0 = time.process_time(), time.perf_counter()
        if monitor:
            mon.start()
        res = await h.run(cfg.arm, case, seed)
        wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
        lag = await mon.stop() if monitor else {}
        slept = sum(w.backend.slept_s for w in h.workers.values()) - slept0
        return {"case": case, "seed": seed, "arm": cfg.arm, "wall_s": wall, "cpu_s": cpu, "slept_s_total": slept,
                "lag": lag, "fingerprint": fingerprint(res), "board_size": res.cell["board_size"],
                "decisions": res.run_log["decisions"], "ticks": res.run_log["ticks"],
                "peak_rss_mb": peak_rss_mb()}


def timed_episode(cfg: ProfileConfig, case: str, seed: int) -> Dict[str, Any]:
    """Phase-timed episode (conc 1) with exclusive timers; ``other_glue_s`` = CPU not inside any timed phase."""
    timer = PhaseTimer()
    with instrument(timer):
        out = asyncio.run(run_episode(cfg, case, seed, timer=timer))
    phases = dict(timer.total)
    out["phases_s"] = phases
    out["phase_calls"] = dict(timer.calls)
    out["phase_per_tick_s"] = {k: [round(x, 6) for x in v] for k, v in timer.per_tick.items()}
    out["other_glue_s"] = out["cpu_s"] - sum(phases.values())
    return out


def cprofile_episode(cfg: ProfileConfig, case: str, seed: int, top: int = 30) -> Dict[str, Any]:
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    asyncio.run(run_episode(cfg, case, seed, monitor=False))
    pr.disable()
    st = pstats.Stats(pr)
    rows = []
    for (fn, line, name), (cc, nc, tt, ct, _callers) in st.stats.items():
        rows.append({"function": f"{Path(fn).name}:{line}({name})", "ncalls": nc, "tottime_s": tt, "cumtime_s": ct})
    rows.sort(key=lambda r: -r["tottime_s"])
    return {"total_s": st.total_tt, "top_by_tottime": rows[:top],
            "top_by_cumtime": sorted(rows, key=lambda r: -r["cumtime_s"])[:top]}


# --------------------------------------------------------------------------------------------------
# Concurrency: one process vs a process pool
# --------------------------------------------------------------------------------------------------

async def _inprocess(cfg: ProfileConfig, c: int, cases: Sequence[str], seed: int) -> Dict[str, Any]:
    with work_dir(cfg) as out:
        h = build_harness(cfg, None, out)
        mon = LagMonitor(cfg.lag_interval_s)
        cpu0, t0 = time.process_time(), time.perf_counter()
        mon.start()
        results = await h.run_many([(cfg.arm, cases[i % len(cases)], seed) for i in range(c)], concurrency=c)
        wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
        lag = await mon.stop()
    failed = [repr(r) for r in results if isinstance(r, BaseException)]
    ok = [r for r in results if not isinstance(r, BaseException)]
    return {"mode": "inprocess", "n": c, "wall_s": wall, "cpu_s": cpu, "episodes_ok": len(ok), "failed": failed,
            "episodes_per_hour": len(ok) / wall * 3600.0, "lag": lag, "peak_rss_mb": peak_rss_mb(),
            "fingerprints": {f"{r.cell['case']}:{r.cell['seed']}": fingerprint(r)["trace_hash"] for r in ok}}


def measure_inprocess(cfg: ProfileConfig, c: int, cases: Sequence[str], seed: int = 1) -> Dict[str, Any]:
    before = swap_usage()
    out = asyncio.run(_inprocess(cfg, c, cases, seed))
    out.update(time_scale=cfg.time_scale, swap_before=before, swap_after=swap_usage())
    return out


def _warm() -> int:
    return os.getpid()


def _pool_episode(cfg_dict: Dict[str, Any], case: str, seed: int) -> Dict[str, Any]:
    cfg = ProfileConfig(**cfg_dict)
    t_import = time.perf_counter()
    out = asyncio.run(run_episode(cfg, case, seed))
    out["pid"] = os.getpid()
    out["worker_wall_s"] = time.perf_counter() - t_import
    return {"case": case, "seed": seed, "wall_s": out["wall_s"], "cpu_s": out["cpu_s"], "lag": out["lag"],
            "peak_rss_mb": out["peak_rss_mb"], "trace_hash": out["fingerprint"]["trace_hash"], "pid": out["pid"]}


def measure_pool(cfg: ProfileConfig, n: int, cases: Sequence[str], seed: int = 1) -> Dict[str, Any]:
    """N spawned processes, one episode each. The pool is warmed first (imports done) so the wall is the episodes'."""
    import concurrent.futures
    import multiprocessing
    before = swap_usage()
    ctx = multiprocessing.get_context("spawn")
    t_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n, mp_context=ctx) as ex:
        warm_futs = [ex.submit(_warm) for _ in range(n)]
        pids = {f.result() for f in warm_futs}
        t_warm = time.perf_counter() - t_start
        t0 = time.perf_counter()
        futs = [ex.submit(_pool_episode, asdict(cfg), cases[i % len(cases)], seed) for i in range(n)]
        results = [f.result() for f in futs]
        wall = time.perf_counter() - t0
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    child_rss_mb = child_rss / (1024.0 * 1024.0) if sys.platform == "darwin" else child_rss / 1024.0
    lag_p95 = max(r["lag"]["p95_ms"] for r in results)
    lag_max = max(r["lag"]["max_ms"] for r in results)
    return {"mode": "pool", "n": n, "wall_s": wall, "startup_s": t_warm, "distinct_pids": len(pids),
            "cpu_s_sum": sum(r["cpu_s"] for r in results), "episodes_ok": len(results),
            "episodes_per_hour": len(results) / wall * 3600.0,
            "lag": {"p95_ms_worst_process": lag_p95, "max_ms_worst_process": lag_max,
                    "p50_ms_median_process": statistics.median(r["lag"]["p50_ms"] for r in results)},
            "peak_rss_mb_child_max": child_rss_mb, "time_scale": cfg.time_scale, "swap_before": before,
            "swap_after": swap_usage(),
            "fingerprints": {f"{r['case']}:{r['seed']}": r["trace_hash"] for r in results}}


# --------------------------------------------------------------------------------------------------
# Scripted arms S and S'
# --------------------------------------------------------------------------------------------------

def time_scripted(cfg: ProfileConfig, arm: str, cases: Sequence[str], seed: int = 1) -> List[Dict[str, Any]]:
    """S or S' per episode through the harness (no workers, no board)."""
    from dataclasses import replace
    c2 = replace(cfg, arm=arm, anchors=False)
    out = []
    for case in cases:
        r = asyncio.run(run_episode(c2, case, seed, monitor=False))
        out.append({"case": case, "arm": arm, "wall_s": r["wall_s"], "cpu_s": r["cpu_s"],
                    "coverage": r["fingerprint"]["cell"]["coverage"], "ticks": r["ticks"],
                    "trace_hash": r["fingerprint"]["trace_hash"], "fingerprint": r["fingerprint"]})
    return out
