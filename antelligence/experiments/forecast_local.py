"""The fair test's 101 real forecasts, run free on the Mac with local models. See docs/plans/2026-11-02-forecast-local.md.

Why: on exam questions the swarm never beat one good reader (shared-read, fresh questions, night run). Forecasting is
the one task where averaging many different minds is a known win (the "wisdom of crowds", and LLM crowds matching a
human crowd in Schoenegger et al., 2024). So this asks the open question there: can ants do something the reader
can't?

Same questions, prompts, lenses, combining and pass line as the frozen fair test (``fair_test.py``). Only the models
change: the same local reader and four ants as the night run, 4-bit MLX builds served one at a time by
``mlx_lm.server`` on this Mac, thinking off. No paid call is possible: every call goes to ``127.0.0.1``.

Two tests, each with its own verdict, both fixed before any call:

* **Test A, lenses** (the fair test's own primary, on the local reader): ``swarm5`` (the reader through 5 lenses,
  each alone, probabilities averaged) against ``single5`` (the reader's plain prompt sampled 5 times), on all 101.
* **Test B, a mixed crowd**: ``mix5`` (the reader's first plain answer plus one plain answer from each of the 4
  ants, averaged) against the same ``single5``, on the forecasts that resolved after the newest model came out
  (``LATE_FROM``), so no model in the crowd can have read the outcome.

Both use 5 calls per question, so neither gets more tries. Pass line (frozen in the fair test): Brier gain of at
least 0.005 with the 95% range above 0.

Everything here is pure except :class:`MlxServer`, which starts and stops the one ``mlx_lm.server`` this run owns.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import shutil
import socket
import statistics as st
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.experiments import fair_test as ft
from antelligence.providers.base import ChatRequest, ChatResponse

KIND = "forecast"
REPO = Path(__file__).resolve().parents[2]
MODELS_FILE = REPO / "docs/research/forecast-local-20261102/models.json"
LATE_FROM = "2026-08-25"      # the newest model in the crowd (Granite 4.2 3B) came out this day
FIRST_RESOLVED = "2026-07-03"  # the earliest forecast in the set resolved this day
MAX_TOKENS = 800              # thinking is off; a 60-word JSON answer is ~100 tokens, so 800 only cuts off loops
TEMPERATURE = ft.TEMPERATURE
K = ft.K
SINGLE = ft.SINGLE_SAMPLES
PASS_GAIN = ft.PASS["forecast_min_brier_gain"]
MIN_USABLE = ft.MIN_USABLE
PORT = 8611                   # nothing else on the Mac uses it (8090 chat shim, 8091 embeddings, 8094 RAG, 8520 d1)


# --------------------------------------------------------------------- models

@dataclass(frozen=True)
class Model:
    tag: str                                      # the Hugging Face name mlx_lm.server loads
    family: str
    gb: float                                     # rough size in memory, for the memory check
    released: str                                 # public release day, YYYY-MM-DD
    extra_body: Optional[Mapping[str, Any]] = None


def _model(d: Mapping[str, Any]) -> Model:
    unknown = set(d) - {"tag", "family", "gb", "released", "extra_body", "note", "source"}
    if unknown:
        raise ValueError(f"unknown model keys {sorted(unknown)} in {d.get('tag')!r}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(d.get("released", ""))):
        raise ValueError(f"{d.get('tag')!r} needs its release day as YYYY-MM-DD")
    return Model(str(d["tag"]), str(d["family"]), float(d.get("gb") or 0), str(d["released"]),
                 dict(d["extra_body"]) if d.get("extra_body") else None)


def load_models(path: Path = MODELS_FILE) -> Tuple[Model, Tuple[Model, ...]]:
    """(reader, ants). Ants come from different families, and ``LATE_FROM`` must be the newest release day."""
    cfg = json.loads(Path(path).read_text())
    unknown = set(cfg) - {"about", "reader", "ants"}
    if unknown:
        raise ValueError(f"unknown keys {sorted(unknown)} in the model file")
    reader, ants = _model(cfg["reader"]), tuple(_model(a) for a in cfg["ants"])
    if not ants:
        raise ValueError("no ants in the model file")
    if len({a.family for a in ants}) != len(ants):
        raise ValueError("each ant must come from a different family")
    if len({a.tag for a in ants} | {reader.tag}) != len(ants) + 1:
        raise ValueError("the reader and every ant must be different models")
    if reader.released >= FIRST_RESOLVED:
        raise ValueError("the reader came out after the first forecast resolved, so Test A could be scored on memory")
    newest = max(m.released for m in (reader, *ants))
    if newest != LATE_FROM:
        raise ValueError(f"LATE_FROM is {LATE_FROM} but the newest model came out {newest}: the late cut is wrong")
    return reader, ants


def cache_dir_name(tag: str) -> str:
    """Where the Hugging Face hub cache keeps a model: ``models--<org>--<name>``."""
    return "models--" + tag.replace("/", "--")


def missing_models(hub: Path, models: Sequence[Model]) -> List[str]:
    """Models with no downloaded snapshot under ``hub``."""
    out = []
    for m in models:
        snaps = hub / cache_dir_name(m.tag) / "snapshots"
        if not snaps.is_dir() or not any(p.is_dir() for p in snaps.iterdir()):
            out.append(m.tag)
    return out


def on_external_disk(path: Path) -> bool:
    """True when ``path`` (followed through symlinks) lives on a mounted drive, as macOS mounts them under /Volumes."""
    return str(Path(path).expanduser().resolve()).startswith("/Volumes/")


def dir_bytes(path: Path) -> int:
    """Bytes of every real file under ``path`` (a symlink to a file inside counts once, through its target)."""
    total, seen = 0, set()
    for p in Path(path).rglob("*"):
        try:
            if p.is_file():
                real = p.resolve()
                if real not in seen:
                    seen.add(real)
                    total += real.stat().st_size
        except OSError:
            continue
    return total


def snapshot_files(model_dir: Path) -> Dict[str, int]:
    """Every file a model's snapshots hold, by path under ``snapshots``, with its size (links followed)."""
    snaps = Path(model_dir) / "snapshots"
    return {str(p.relative_to(snaps)): p.stat().st_size for p in sorted(snaps.rglob("*")) if p.is_file()}


def copied_whole(src: Path, dst: Path) -> bool:
    """True when ``dst`` has every snapshot file of ``src`` at the same size, each one stored inside ``dst``."""
    if not (dst / "snapshots").is_dir():
        return False
    want, inside = snapshot_files(src), dst.resolve()
    have = snapshot_files(dst)
    if have != want:
        return False
    return all(inside in (dst / "snapshots" / rel).resolve().parents for rel in have)


def copy_models(src_hub: Path, dst_hub: Path, models: Sequence[Model], *, min_free_gb: float = 15.0,
                log: Callable[[str], None] = print) -> Dict[str, int]:
    """Copy each model's folder from one hub cache to another (copy, never move; the source is only read).

    Links inside a model folder (snapshots -> blobs) stay links when they point inside the folder, so the copy is
    the same size as the source; a link that points anywhere else (say, back to the old drive) is replaced by a copy
    of its file, so the new cache never reads the old disk. A model already copied whole is skipped. Stops before
    copying when the target disk would keep less than ``min_free_gb`` free. Returns bytes per model."""
    src_hub, dst_hub = Path(src_hub).expanduser(), Path(dst_hub).expanduser()
    gone = missing_models(src_hub, models)
    if gone:
        raise FileNotFoundError(f"not in {src_hub}: {', '.join(gone)}")
    dst_hub.mkdir(parents=True, exist_ok=True)
    sizes = {m.tag: dir_bytes(src_hub / cache_dir_name(m.tag)) for m in models}
    todo = [m for m in models if not copied_whole(src_hub / cache_dir_name(m.tag), dst_hub / cache_dir_name(m.tag))]
    need = sum(sizes[m.tag] for m in todo) / 1e9
    free = shutil.disk_usage(dst_hub).free / 1e9
    if free - need < min_free_gb:
        raise OSError(f"copying needs {need:.1f} GB and would leave {free - need:.1f} GB free on the target disk "
                      f"(at least {min_free_gb:.0f} GB must stay free)")
    for m in todo:
        src, dst = src_hub / cache_dir_name(m.tag), dst_hub / cache_dir_name(m.tag)
        if dst.exists():
            shutil.rmtree(dst)   # a half copy from an earlier try, ours alone
        log(f"copying {m.tag} ({sizes[m.tag] / 1e9:.1f} GB)")
        shutil.copytree(src, dst, symlinks=True)
        inside = dst.resolve()
        for p in sorted(dst.rglob("*")):
            if p.is_symlink():
                target = (p.parent / os.readlink(p)).resolve() if not os.path.isabs(os.readlink(p)) \
                    else Path(os.readlink(p)).resolve()
                if inside not in target.parents:          # points outside the copy: bring the file itself
                    real = (src / p.relative_to(dst)).resolve()
                    p.unlink()
                    shutil.copy2(real, p)
        if not copied_whole(src, dst):
            raise OSError(f"{m.tag}: the copy doesn't match the source; rerun `copy-models`")
    return sizes


# ------------------------------------------------------------------- requests

def request(model: Model, messages: Tuple[Dict[str, str], ...], qid: str, arm: str, index: int) -> ChatRequest:
    return ChatRequest(model=model.tag, messages=messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                       seed=ft.call_seed(qid, arm, index), extra_body=model.extra_body)


def first_requests(q: ft.Question, reader: Model) -> List[ChatRequest]:
    """The reader's plain prompt 10 times, then its 5 lenses (the fair test's ``single`` and ``swarm``)."""
    return ([request(reader, ft.plain_messages(q), q.id, "single", i) for i in range(SINGLE)]
            + [request(reader, ft.lens_messages(q, i), q.id, "swarm", i) for i in range(K)])


def seen_requests(q: ft.Question, reader: Model, swarm: Sequence[Tuple[Optional[float], str]]) -> List[ChatRequest]:
    """The 5 lenses again, each after reading the other four first answers (the fair test's ``seen``)."""
    out = []
    for i in range(K):
        peers = [(j, ft._show(KIND, swarm[j][0]), swarm[j][1]) for j in range(K) if j != i]
        out.append(request(reader, ft.seen_messages(q, i, peers), q.id, "seen", i))
    return out


def ant_request(q: ft.Question, ant: Model) -> ChatRequest:
    """One plain answer from an ant: the same prompt the reader gets in ``single``."""
    return request(ant, ft.plain_messages(q), q.id, f"ant:{ant.family}", 0)


def plan(n: int, ants: int = 4) -> Dict[str, int]:
    reader = n * (SINGLE + 2 * K)
    return {"questions": n, "reader_calls": reader, "ant_calls": n * ants, "calls": reader + n * ants}


# ------------------------------------------------------------------ the server

class MlxServer:
    """The one ``mlx_lm.server`` this run starts, one model at a time, on ``127.0.0.1:port``.

    It reads models only from ``hub`` (``HF_HUB_CACHE``) with downloads turned off (``HF_HUB_OFFLINE=1``), so a
    missing model fails at once instead of quietly downloading to the default cache. It never stops a process it
    didn't start."""

    def __init__(self, cmd: Sequence[str], hub: Path, *, port: int = PORT, log_dir: Optional[Path] = None,
                 start_timeout_s: float = 1800.0) -> None:
        self.cmd, self.hub, self.port = list(cmd), Path(hub), port
        self.log_dir, self.start_timeout_s = log_dir, start_timeout_s
        self.proc: Optional[subprocess.Popen] = None
        self.model: Optional[str] = None
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _answers(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def start(self, tag: str) -> None:
        if self.model == tag and self.proc is not None and self.proc.poll() is None:
            return
        self.stop()
        if not port_free(self.port):
            raise RuntimeError(f"port {self.port} is already in use by something this run didn't start")
        env = {**os.environ, "HF_HUB_CACHE": str(self.hub), "HF_HUB_OFFLINE": "1"}
        out = subprocess.DEVNULL
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log = (self.log_dir / f"mlx-{tag.rsplit('/', 1)[-1]}.log").open("a")
            out = self._log
        self.proc = subprocess.Popen([*self.cmd, "--model", tag, "--host", "127.0.0.1", "--port", str(self.port)],
                                     env=env, stdout=out, stderr=subprocess.STDOUT)
        self.model = tag
        end = time.monotonic() + self.start_timeout_s
        while time.monotonic() < end:
            code = self.proc.poll()
            if code is not None:
                self.proc, self.model = None, None
                raise RuntimeError(f"mlx_lm.server for {tag} exited with code {code} before answering"
                                   + (f" (see {self._log.name})" if self._log else ""))
            if self._answers():
                return
            time.sleep(1.0)
        self.stop()
        raise RuntimeError(f"mlx_lm.server for {tag} did not answer within {self.start_timeout_s:.0f}s")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        if self._log is not None:
            self._log.close()
            self._log = None
        self.proc, self.model = None, None


class OwnServer:
    """Wraps the provider pointed at our own server. ``mlx_lm.server`` may report the model as a local path, so the
    served name is taken as the model the server was started with, and a call for any other model is refused."""

    def __init__(self, provider: Any, server: Any) -> None:
        self.provider, self.server = provider, server

    async def complete(self, req: ChatRequest) -> ChatResponse:
        from antelligence.providers.base import ProviderError

        if req.model != self.server.model:
            raise ProviderError(f"asked for {req.model!r} while the server holds {self.server.model!r}")
        resp = await self.provider.complete(req)
        return dataclasses.replace(resp, model=req.model, extra={**(resp.extra or {}), "served_as": resp.model})


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) != 0


def free_gb() -> Optional[float]:
    """Memory the system can hand out now, in GB (macOS ``vm_stat``; Linux MemAvailable). None when unknown."""
    import platform

    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
            page = int(re.search(r"page size of (\d+) bytes", out).group(1))
            pages = 0
            for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                m = re.search(rf"{name}:\s+(\d+)", out)
                pages += int(m.group(1)) if m else 0
            return pages * page / 1e9
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        return None
    return None


class MemoryTight(ft.BrokenRun):
    """Not enough free memory to load the next model safely. Nothing is lost: free memory and rerun."""


# --------------------------------------------------------------------- running

class LocalRun:
    """Runs the reader, then each ant, one model in memory at a time. A model whose calls are all cached is never
    loaded, so a rerun picks up where a stopped run was.

    ``runner`` is a :class:`fair_test.Runner` (cache, stop after unusable replies in a row); ``server`` has
    ``start(tag)`` and ``stop()``; ``guard(model)`` raises :class:`MemoryTight` when there's no room for it."""

    def __init__(self, runner: ft.Runner, server: Any, reader: Model, ants: Sequence[Model], *,
                 concurrency: int = 2, guard: Optional[Callable[[Model], None]] = None,
                 log: Callable[[str], None] = print) -> None:
        self.runner, self.server, self.reader, self.ants = runner, server, reader, tuple(ants)
        self.concurrency, self.guard, self.log = concurrency, guard, log
        self.loaded: List[str] = []

    async def _ask_all(self, reqs: Sequence[ChatRequest]) -> Dict[str, Dict[str, Any]]:
        sem = asyncio.Semaphore(self.concurrency)

        async def one(r: ChatRequest) -> Tuple[str, Dict[str, Any]]:
            async with sem:
                return r.request_hash, await self.runner.ask(r, KIND)

        return dict(await asyncio.gather(*(one(r) for r in reqs)))

    def _load(self, model: Model, reqs: Sequence[ChatRequest]) -> None:
        if all(r.request_hash in self.runner.cache for r in reqs):
            return   # everything for this model is already answered
        if self.server.model != model.tag:
            self.server.stop()
            if self.guard:
                self.guard(model)
            self.log(f"loading {model.tag}")
            self.server.start(model.tag)
            self.loaded.append(model.tag)

    async def run(self, questions: Sequence[ft.Question]) -> List[Dict[str, Any]]:
        start = time.monotonic()
        try:
            first = {q.id: first_requests(q, self.reader) for q in questions}
            self._load(self.reader, [r for rs in first.values() for r in rs])
            got = await self._ask_all([r for rs in first.values() for r in rs])
            self.log(f"reader, first answers: {len(got)} calls ({time.monotonic() - start:.0f}s)")
            seen: Dict[str, List[ChatRequest]] = {}
            for q in questions:
                swarm = [ft._reply(KIND, got[r.request_hash]) for r in first[q.id][SINGLE:]]
                seen[q.id] = seen_requests(q, self.reader, swarm)
            self._load(self.reader, [r for rs in seen.values() for r in rs])
            got.update(await self._ask_all([r for rs in seen.values() for r in rs]))
            self.log(f"reader, second answers done ({time.monotonic() - start:.0f}s)")
            ant_reqs: Dict[str, Dict[str, ChatRequest]] = {q.id: {} for q in questions}
            for ant in self.ants:
                reqs = [ant_request(q, ant) for q in questions]
                self._load(ant, reqs)
                got.update(await self._ask_all(reqs))
                for q, r in zip(questions, reqs):
                    ant_reqs[q.id][ant.family] = r
                self.log(f"ant {ant.family} done ({time.monotonic() - start:.0f}s)")
        finally:
            self.server.stop()
        rows = []
        for q in questions:
            ans = [ft._reply(KIND, got[r.request_hash])[0] for r in first[q.id] + seen[q.id]]
            rows.append({"id": q.id, "kind": KIND, "group": q.group,
                         "single": ans[:SINGLE], "swarm": ans[SINGLE:SINGLE + K], "seen": ans[SINGLE + K:],
                         "ants": {fam: ft._reply(KIND, got[r.request_hash])[0] for fam, r in ant_reqs[q.id].items()}})
        return rows


# --------------------------------------------------------------------- scoring

def arm_probs(row: Mapping[str, Any]) -> Dict[str, float]:
    """Every arm's forecast for one question. ``mix5`` is the reader's first plain answer plus each ant's."""
    out = dict(ft.arm_answers(row))
    ants = list(row["ants"].values())
    out["ants4"] = ft.mean_prob(ants)
    out["mix5"] = ft.mean_prob([row["single"][0], *ants])
    return out


def brier(p: float, truth: str) -> float:
    return ft.score_item(KIND, p, truth)


def usable(answers: Sequence[Optional[float]]) -> float:
    return sum(a is not None for a in answers) / len(answers) if answers else 0.0


def _test(rows: Sequence[Mapping[str, Any]], key: Mapping[str, str], arm: str, base: str,
          answers: Callable[[Mapping[str, Any]], List[Optional[float]]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0, "verdict": "invalid run: no questions to score", "pass": False}
    a = [brier(arm_probs(r)[arm], key[r["id"]]) for r in rows]
    b = [brier(arm_probs(r)[base], key[r["id"]]) for r in rows]
    gain, lo, hi = ft.paired_boot([y - x for x, y in zip(a, b)])   # positive: ``arm`` has the lower (better) Brier
    share = usable([x for r in rows for x in answers(r)])
    passed = gain >= PASS_GAIN and lo > 0
    if share < MIN_USABLE:
        verdict = (f"invalid run: only {share:.0%} of these answers were usable (at least {MIN_USABLE:.0%} needed), "
                   "so nothing was measured")
    else:
        verdict = "supported" if passed else "not supported"
    return {"n": len(rows), arm: round(st.mean(a), 4), base: round(st.mean(b), 4), "gain": round(gain, 4),
            "ci95": [round(lo, 4), round(hi, 4)], "usable": round(share, 4), "pass": passed and share >= MIN_USABLE,
            "verdict": verdict}


def _vs(rows: Sequence[Mapping[str, Any]], key: Mapping[str, str], arm: str, base: str) -> Dict[str, Any]:
    a = [brier(arm_probs(r)[arm], key[r["id"]]) for r in rows]
    b = [brier(arm_probs(r)[base], key[r["id"]]) for r in rows]
    gain, lo, hi = ft.paired_boot([y - x for x, y in zip(a, b)])
    return {"n": len(rows), "gain": round(gain, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def score(rows: Sequence[Mapping[str, Any]], key: Mapping[str, str], late: Sequence[str],
          crowd: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    """Test A (lenses, all forecasts) and Test B (mixed crowd, forecasts resolved after ``LATE_FROM``), each with
    its own verdict, plus secondaries that carry no verdict."""
    rows = [r for r in rows if r["id"] in key]
    late_set = set(late)
    late_rows = [r for r in rows if r["id"] in late_set]
    out: Dict[str, Any] = {
        "test_a_lenses": _test(rows, key, "swarm5", "single5", lambda r: r["single"][:K] + r["swarm"]),
        "test_b_mixed_crowd": _test(late_rows, key, "mix5", "single5",
                                    lambda r: r["single"][:K] + list(r["ants"].values())),
        "late_from": LATE_FROM,
    }
    sec: Dict[str, Any] = {}
    if rows:
        means = {arm: round(st.mean(brier(arm_probs(r)[arm], key[r["id"]]) for r in rows), 4)
                 for arm in arm_probs(rows[0])}
        sec["brier_all"] = means
        sec["mix5_vs_single5_all"] = _vs(rows, key, "mix5", "single5")
        sec["seen10_vs_single10"] = _vs(rows, key, "seen10", "single10")
        if late_rows:
            sec["ants4_vs_single5_late"] = _vs(late_rows, key, "ants4", "single5")
            sec["brier_late"] = {arm: round(st.mean(brier(arm_probs(r)[arm], key[r["id"]]) for r in late_rows), 4)
                                 for arm in ("single1", "single5", "swarm5", "mix5", "ants4")}
        fams = sorted(rows[0]["ants"])
        sec["per_model"] = {"reader": round(st.mean(brier(ft.mean_prob(r["single"][:1]), key[r["id"]])
                                                    for r in rows), 4)}
        for fam in fams:
            sec["per_model"][fam] = round(st.mean(brier(ft.mean_prob([r["ants"][fam]]), key[r["id"]]) for r in rows), 4)
        sec["usable_by_model"] = {"reader": round(usable([x for r in rows for x in r["single"] + r["swarm"] + r["seen"]]), 4),
                                  **{fam: round(usable([r["ants"][fam] for r in rows]), 4) for fam in fams}}
        groups: Dict[str, List[Mapping[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r["group"], []).append(r)
        sec["by_field"] = {g: {"n": len(rs), **{arm: round(st.mean(brier(arm_probs(r)[arm], key[r["id"]]) for r in rs), 4)
                                                for arm in ("single5", "swarm5", "mix5")}}
                           for g, rs in sorted(groups.items())}
        if crowd:
            with_crowd = [r for r in rows if r["id"] in crowd]
            if with_crowd:
                sec["market_crowd"] = {
                    "n": len(with_crowd),
                    "market": round(st.mean(brier(crowd[r["id"]], key[r["id"]]) for r in with_crowd), 4),
                    **{arm: round(st.mean(brier(arm_probs(r)[arm], key[r["id"]]) for r in with_crowd), 4)
                       for arm in ("single5", "swarm5", "mix5")}}
    out["secondary"] = sec
    return out


def block(result: Mapping[str, Any]) -> str:
    """The short block Eshaan pastes back."""
    lines = ["FORECAST RESULT (paste this back to Claude in Duet)"]
    for name, label, arm in (("test_a_lenses", "Test A lenses", "swarm5"), ("test_b_mixed_crowd", "Test B mixed crowd", "mix5")):
        t = result[name]
        if not t.get("n"):
            lines.append(f"{label}: {t['verdict']}")
            continue
        lines.append(f"{label} (n {t['n']}): {arm} Brier {t[arm]} vs reader x5 {t['single5']}, gain {t['gain']} "
                     f"[{t['ci95'][0]}, {t['ci95'][1]}], usable {t['usable']:.0%} -> {t['verdict'].upper()}")
    sec = result.get("secondary", {})
    if "per_model" in sec:
        lines.append("per model Brier (one answer each): " + ", ".join(f"{k} {v}" for k, v in sec["per_model"].items()))
    if "usable_by_model" in sec:
        lines.append("usable: " + ", ".join(f"{k} {v:.0%}" for k, v in sec["usable_by_model"].items()))
    for k in ("mix5_vs_single5_all", "seen10_vs_single10", "ants4_vs_single5_late"):
        if k in sec:
            v = sec[k]
            lines.append(f"{k} (n {v['n']}): gain {v['gain']} [{v['ci95'][0]}, {v['ci95'][1]}]")
    if "market_crowd" in sec:
        m = sec["market_crowd"]
        lines.append(f"market crowd (n {m['n']}): market {m['market']}, reader x5 {m['single5']}, lenses {m['swarm5']}, "
                     f"mixed {m['mix5']}")
    return "\n".join(lines)
