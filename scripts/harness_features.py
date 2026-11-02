#!/usr/bin/env python3
"""Score the five borrowed parts as switches, alone and mixed. See docs/plans/2026-11-02-borrowed-parts.md.

Eshaan, 2026-11-02: "we can eval each of them as features and see what works best maybe?"

Commands::

    python scripts/harness_features.py free --night NIGHT --out OUT          # stop, router, cache on saved answers, $0
    python scripts/harness_features.py mac --night NIGHT --mlx-server PATH --out OUT   # every stage, one paste-back
    python scripts/harness_features.py mac --dry-run --out OUT               # the same with fake models, seconds, $0

``mac`` runs these stages in order (``--stages`` picks some; each one's result is written as soon as it is done):

1. ``free``: the night run's saved answers, learned on set 1 + set 2 (5 folds inside them), set 3 scored once, for
   today's harness and every mix of the stop rule, the router and the cache-aware count. No model calls.
2. ``probe``: four calls to the smallest ant, to see whether this ``mlx_lm.server`` sends answer odds and reuses a
   shared prompt start. A few seconds.
3. ``confsweep``: every ant alone on every saved question again, now with answer odds (the reader's answers are
   kept). Local models only: no cost, a few hours.
4. ``gate``: free again on the saved answers with the new ant answers in, now with the confidence gate as well.
5. ``promptopt``: a small GEPA-style search for shorter ant instructions on 30 training questions, then every ant on
   set 3 with the pick, at the ant level and inside the harness. Local models only, a few hours.

The final test set is not here: set 3 is the night run's own third set, and the hidden final test stays unseen.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import random
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import shared_read as sr  # noqa: E402
from antelligence.harness import borrowed as bw  # noqa: E402
from antelligence.harness import promptopt as po  # noqa: E402
from antelligence.harness.core import Step, Task  # noqa: E402
from antelligence.harness.importer import import_sets, set_of  # noqa: E402
from antelligence.harness.planner import sweep_batch  # noqa: E402
from antelligence.harness.recipes import cheap_first  # noqa: E402
from antelligence.harness.resilience import Patience  # noqa: E402
from antelligence.harness.tools import AntInfo, LiveBox, OddsRunner, ant_tool, parse_ant, trim_loads  # noqa: E402
from antelligence.providers import logprobs as lp  # noqa: E402


def _harness_cli():
    spec = importlib.util.spec_from_file_location("_harness_cli_for_features", REPO / "scripts/harness.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _harness_cli()
STAGES = ("free", "probe", "confsweep", "gate", "promptopt")
TRAIN, TEST = ("set1", "set2"), ("set3",)
TOP_LOGPROBS = 8              # mlx_lm.server takes up to 11; 8 holds every option letter's chance at the answer
HEADER = "BORROWED PARTS RESULT (paste this back to Claude in Duet)"
TEST_NOTE = ("Set 3 is the night run's third set, scored once after everything was picked on sets 1 and 2; the hidden "
             "final test set was not used.")


def log(msg: str) -> None:
    print(f"[features] {msg}", flush=True)


# ------------------------------------------------------------------ the saved answers

def fake_night(root: Path, cfg: sr.Config, per_task: int = 2, seeds: Sequence[int] = (101, 102, 103)) -> Path:
    """Three saved sets in the night run's own format, made with fake models (as ``harness.py selftest`` does), with a
    ``meta.json`` listing the ants in another order than the model file, as a real run's may. $0, seconds."""
    for name, seed in zip(("set1", "set2", "set3"), seeds):
        run = root / name / "run"
        if (run / "rows.jsonl").exists():
            continue
        tasks, key = H.make_dev(root / name / "dev", seed, per_task)
        run.mkdir(parents=True, exist_ok=True)
        runner = sr.Runner(H.fake_provider(tasks, key, cfg), run / "cache.jsonl", concurrency=4, check_names=False,
                           max_fail_streak=10 ** 6)
        box = LiveBox(runner, cfg.reader, cfg.ants, parser=cfg.parser, code=False)
        steps = asyncio.run(sweep_batch(box, tasks))
        order = list(reversed(box.roster))                     # the saved columns' order
        H.write_jsonl(run / "questions.jsonl", [H.task_row(t) for t in tasks])
        H.write_jsonl(run / "answers.jsonl", [{"id": i, "label": v} for i, v in key.items()])
        H.write_jsonl(run / "rows.jsonl", [{
            "id": t.id, "group": t.kind, "reader": steps[t.id]["reader"].answer,
            "reader_notes": steps[t.id]["notes"].payload["notes"], "leak": steps[t.id]["notes"].payload["leak"],
            **{arm: [steps[t.id][ant_tool(a.index, arm)].answer for a in order] for arm in ("alone", "notes", "both")}}
            for t in tasks])
        (run / "meta.json").write_text(json.dumps({"ants": [[a.tag, a.family, a.lens] for a in order]}) + "\n")
    return root


def limit_sets(tasks: Sequence[Task], limit: int) -> List[Task]:
    if not limit:
        return list(tasks)
    seen: Dict[str, int] = {}
    out = []
    for t in tasks:
        s = set_of(t.id)
        if seen.get(s, 0) < limit:
            out.append(t)
            seen[s] = seen.get(s, 0) + 1
    return out


def original(t: Task) -> Task:
    """The task under the id the night run asked it with (its calls' seeds are made from that id)."""
    return Task(t.id.split(":", 1)[1], t.passage, t.stem, t.options, t.kind)


@dataclasses.dataclass
class State:
    cfg: sr.Config
    sets: Dict[str, Path]
    tasks: List[Task]
    key: Dict[str, str]
    steps: Dict[str, Dict[str, Step]]
    roster: Tuple[AntInfo, ...]
    reports: List[Dict[str, Any]]
    conf: Dict[str, Dict[str, Step]] = dataclasses.field(default_factory=dict)
    free_rep: Optional[Dict[str, Any]] = None
    gate_rep: Optional[Dict[str, Any]] = None
    odds: bool = False

    @property
    def specs(self) -> Dict[str, sr.ModelSpec]:
        return {s.tag: s for s in self.cfg.ants}

    def ids(self, sets: Sequence[str], tasks: Optional[Sequence[Task]] = None) -> List[str]:
        return [t.id for t in (tasks if tasks is not None else self.tasks) if set_of(t.id) in sets]


def load_state(a: argparse.Namespace) -> State:
    cfg = sr.load_config(a.models)
    night = a.night
    if a.dry_run and night is None:
        night = fake_night(a.out / "fake-night", cfg, per_task=a.fake_per_task)
    if night is None:
        raise SystemExit("--night is needed (the folder holding set1/run, set2/run, set3/run)")
    sets = H.night_sets(night)
    if sorted(sets) != ["set1", "set2", "set3"]:
        raise SystemExit(f"need set1, set2 and set3 under {night}; found {sorted(sets) or 'none'}")
    tasks, key, steps, roster, reports = import_sets(sets, cfg)
    tasks = limit_sets(tasks, a.limit)
    keep = {t.id for t in tasks}
    return State(cfg, sets, tasks, {k: v for k, v in key.items() if k in keep},
                 {k: v for k, v in steps.items() if k in keep}, roster, reports)


# ------------------------------------------------------------------ 1 and 4: scoring the switches

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


def stage_free(st: State, out: Path) -> List[str]:
    rep = bw.evaluate_features(st.tasks, st.key, st.steps, st.roster, st.ids(TRAIN), st.ids(TEST),
                               reader=st.cfg.reader, specs=st.specs, features=("stop", "router", "cache"), log=log)
    st.free_rep = rep
    save_json(out / "free" / "features.json", bw.public(rep))
    imp = "; ".join(f"{r['set']} {r['tasks']}q {r['priced']:.0%} priced" for r in st.reports)
    return [f"saved answers: {imp}"] + bw.table_lines(rep, "FREE (saved night answers, no model calls)")


def stage_gate(st: State, out: Path) -> List[str]:
    if not st.conf:
        return ["GATE: skipped (no answer-odds sweep: " + ("the probe found no readable odds)" if not st.odds
                                                          else "run the confsweep stage first)")]
    tasks = [t for t in st.tasks if t.id in st.conf]
    steps = bw.overlay(st.steps, st.conf)
    rep = bw.evaluate_features(tasks, st.key, {t.id: steps[t.id] for t in tasks}, st.roster, st.ids(TRAIN, tasks),
                               st.ids(TEST, tasks), reader=st.cfg.reader, specs=st.specs, log=log)
    st.gate_rep = rep
    save_json(out / "gate" / "features.json", bw.public(rep))
    return bw.table_lines(rep, "GATE (saved answers with the new ant answers in, all four switches)")


# ------------------------------------------------------------------ the local server

def mlx_version(cmd: str) -> str:
    parts = shlex.split(cmd or "")
    if not parts:
        return "unknown"
    if Path(parts[0]).name in ("uvx", "uv"):
        pins = [p for p in parts if p.startswith("mlx-lm") or p.startswith("mlx_lm")]
        return pins[0] if pins else "unknown (uvx)"
    exe = shutil.which(parts[0]) or parts[0]
    for py in (Path(exe).parent / "python", Path(exe).parent / "python3"):
        if py.exists():
            try:
                got = subprocess.run([str(py), "-c", "import importlib.metadata as m; print(m.version('mlx-lm'))"],
                                     capture_output=True, text=True, timeout=60)
                if got.returncode == 0 and got.stdout.strip():
                    return got.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
    return "unknown"


class Server:
    """The MLX stand-in on its own ports, started for the live stages and always stopped after."""

    def __init__(self, a: argparse.Namespace, mlx_server: str, tag: str) -> None:
        self.a, self.cmd, self.tag = a, mlx_server, tag
        self.service = None

    def __enter__(self) -> str:
        nr = H._night_module()
        for port in (self.a.port, self.a.mlx_port):
            if not nr.port_free(port):
                raise SystemExit(f"something already listens on port {port}; pick others with --port/--mlx-port")
        logs = self.a.out / "logs"
        self.service = nr.Service(
            "MLX stand-in", [sys.executable, str(REPO / "scripts/mlx_standin.py"), "--models", str(self.a.models),
                             "--mlx-server", self.cmd, "--port", str(self.a.port), "--mlx-port", str(self.a.mlx_port),
                             "--log", str(logs / f"mlx-{self.tag}.log")],
            f"http://127.0.0.1:{self.a.port}/v1/models", logs / f"standin-{self.tag}.log")
        self.service.up(seconds=300)
        return f"http://127.0.0.1:{self.a.port}/v1"

    def __exit__(self, *exc: Any) -> None:
        if self.service:
            self.service.down()


def provider_for(a: argparse.Namespace, st: State, base: Optional[str]) -> Any:
    if base is None:
        from antelligence.harness.fake_odds import OddsFakeProvider

        tasks = [original(t) for t in st.tasks]
        key = {original_id(i): v for i, v in st.key.items()}
        skill = {st.cfg.reader.tag: 0.78, **{s.tag: 0.45 + 0.05 * i for i, s in enumerate(st.cfg.ants)}}
        return OddsFakeProvider(tasks, key, skill, {s.tag: s.gb for s in [st.cfg.reader, *st.cfg.ants]})
    a.base_url = base
    return H.openai_provider(a)


def original_id(tid: str) -> str:
    return tid.split(":", 1)[1]


# ------------------------------------------------------------------ 2: probe

async def _probe(provider: Any, st: State) -> Dict[str, Any]:
    box = LiveBox(sr.Runner(provider, Path("/dev/null"), check_names=False), st.cfg.reader, st.cfg.ants,
                  parser=st.cfg.parser, code=False, logprobs=TOP_LOGPROBS)
    first = cheap_first(box.roster)[0]
    tag = box.roster[first].tag
    other = next((a.index for a in box.roster if a.tag == tag and a.index != first), first)
    q1, q2 = original(st.tasks[0]), original(st.tasks[1 % len(st.tasks)])

    def req(i: int, t: Task) -> Any:
        spec, info = box.ant_spec(i), box.roster[i]
        item = sr.Item(t.id, t.kind, t.passage, t.stem, t.options)
        return sr.make_request(spec, sr.ant_messages(item, "alone", info.lens, spec), t.id, f"probe:{spec.tag}",
                               info.lens, sr.ANT_MAX_TOKENS)
    calls = []
    for name, i, t in (("A: question 1, way 1", first, q1), ("B: question 1, way 2", other, q1),
                       ("C: A again", first, q1), ("D: question 2, way 1", first, q2)):
        t0 = time.monotonic()
        try:
            resp = await provider.complete(req(i, t))
        except Exception as exc:                       # the probe reports; it never stops the run
            calls.append({"call": name, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        summary = (resp.extra or {}).get("logprobs")
        from antelligence.harness.tools import read_answer

        got = read_answer(resp.content or "", t, st.cfg.parser)
        calls.append({"call": name, "prompt_tokens": resp.prompt_tokens, "cached": (resp.extra or {}).get(
            "cached_tokens"), "seconds": round(time.monotonic() - t0, 2), "answer": got,
            "sure": lp.letter_sure(summary, got), "has_text": bool(summary and summary.get("text")),
            "slot": (summary or {}).get("slot")})
    ok = [c for c in calls if "error" not in c]
    return {"model": tag, "calls": calls, "odds": any(c["sure"] is not None for c in ok),
            "odds_any": any(c["has_text"] or c["slot"] for c in ok),
            "cache": any((c.get("cached") or 0) > 0 for c in ok[1:]), "cache_field": any(c.get("cached") is not None
                                                                                         for c in ok)}


def probe_lines(p: Mapping[str, Any], version: str) -> List[str]:
    lines = [f"PROBE: mlx-lm {version}, model {p['model']}; answer odds readable: {'yes' if p['odds'] else 'no'}; "
             f"prefix cache counted: {'yes' if p['cache'] else ('no' if p['cache_field'] else 'not reported')}"]
    for c in p["calls"]:
        if "error" in c:
            lines.append(f"  {c['call']}: failed ({c['error']})")
        else:
            lines.append(f"  {c['call']}: {c['prompt_tokens']} prompt tokens, {c['cached']} reused, {c['seconds']}s, "
                         f"answer {c['answer']}, sure {c['sure']}")
    return lines


# ------------------------------------------------------------------ 3: confsweep

def remap(live: Mapping[str, Mapping[str, Step]], live_roster: Sequence[AntInfo], night_roster: Sequence[AntInfo],
          prefix: str) -> Dict[str, Dict[str, Step]]:
    """Live steps under the saved sets' names: ant numbers in the saved column order, ids with their set prefix."""
    pos = {(a.tag, a.lens): a.index for a in night_roster}
    out: Dict[str, Dict[str, Step]] = {}
    for tid, per in live.items():
        new: Dict[str, Step] = {}
        for tool, s in per.items():
            a = parse_ant(tool)
            name = tool
            if a:
                info = live_roster[a[0]]
                j = pos.get((info.tag, info.lens))
                if j is None:
                    continue
                name = ant_tool(j, a[1])
            new[name] = dataclasses.replace(s, tool=name)
        out[f"{prefix}:{tid}"] = new
    return out


def live_box(a: argparse.Namespace, st: State, runner: sr.Runner, *, logprobs: int,
             ant_messages: Optional[Callable[..., Any]] = None) -> LiveBox:
    return LiveBox(runner, st.cfg.reader, st.cfg.ants, parser=st.cfg.parser, code=False, logprobs=logprobs,
                   ant_messages=ant_messages, patience=Patience(drive=a.drive))


def runner_for(a: argparse.Namespace, provider: Any, path: Path, seeds: Sequence[Path] = ()) -> sr.Runner:
    path.parent.mkdir(parents=True, exist_ok=True)
    return OddsRunner(provider, path, concurrency=a.concurrency, check_names=False, max_fail_streak=10 ** 6,
                     seed_paths=[p for p in seeds if Path(p).exists()])


async def sweep_ants(box: LiveBox, tasks: Sequence[Task], base: Optional[str], label: str, workers: int = 2
                     ) -> Dict[str, Dict[str, Step]]:
    tools = [ant_tool(i, "alone") for i in range(len(box.roster))]
    t0 = time.monotonic()
    n = [0]

    def on_group(g: List[str]) -> None:
        n[0] += 1
        log(f"{label}: model {n[0]} ({box.roster[parse_ant(g[0])[0]].tag}), {len(tasks)} questions "
            f"({time.monotonic() - t0:.0f}s)")
    return await sweep_batch(box, tasks, tools, between=H.unloader(base, box) if base else None, on_group=on_group,
                             order="question", workers=workers)


def cache_check(live: Mapping[str, Mapping[str, Step]], tasks: Sequence[Task], box: LiveBox, st: State
                ) -> Dict[str, Any]:
    """The server's own count of reused prompt tokens against this module's estimate, on the sweep's real order."""
    prompts = bw.Prompts(box.roster, st.cfg.reader, st.specs)
    rate = bw.token_rates(tasks, live, prompts)
    tot = {"first_in": 0, "first_cached": 0, "rest_in": 0, "rest_cached": 0, "rest_est": 0, "known": 0, "n": 0}
    by_tag: Dict[str, List[int]] = {}
    for a in box.roster:
        by_tag.setdefault(a.tag, []).append(a.index)
    for t in tasks:
        per = live.get(t.id, {})
        for tag, idx in by_tag.items():
            seq = [per[ant_tool(i, "alone")] for i in idx if ant_tool(i, "alone") in per]
            if not seq:
                continue
            est = bw.fresh_in(seq, t, prompts, rate)
            for k, s in enumerate(seq):
                c = s.payload.get("cached_in")
                tot["n"] += 1
                tot["known"] += isinstance(c, int)
                if k == 0:
                    tot["first_in"] += s.tokens_in
                    tot["first_cached"] += c or 0
                else:
                    tot["rest_in"] += s.tokens_in
                    tot["rest_cached"] += c or 0
                    tot["rest_est"] += s.tokens_in - est[k]
    share = lambda x, y: round(x / y, 4) if y else None  # noqa: E731
    return {"measured_known": share(tot["known"], tot["n"]), "first_reused": share(tot["first_cached"], tot["first_in"]),
            "second_reused": share(tot["rest_cached"], tot["rest_in"]),
            "second_estimated": share(tot["rest_est"], tot["rest_in"]), "raw": tot}


def calibration(steps: Mapping[str, Mapping[str, Step]], key: Mapping[str, str]) -> Dict[str, Any]:
    bins = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01))
    rows = {f"{lo:.2f}-{min(hi, 1.0):.2f}": [0, 0] for lo, hi in bins}
    known = total = 0
    for tid, per in steps.items():
        for tool, s in per.items():
            if not parse_ant(tool) or s.answer is None:
                continue
            total += 1
            if s.sure is None:
                continue
            known += 1
            for lo, hi in bins:
                if lo <= s.sure < hi:
                    r = rows[f"{lo:.2f}-{min(hi, 1.0):.2f}"]
                    r[0] += 1
                    r[1] += s.answer == key.get(tid)
    return {"known": round(known / total, 4) if total else None, "answers": total,
            "bins": {k: {"n": n, "right": round(r / n, 4) if n else None} for k, (n, r) in rows.items()}}


async def stage_confsweep(a: argparse.Namespace, st: State, provider: Any, base: Optional[str], out: Path
                          ) -> List[str]:
    t0 = time.monotonic()
    conf: Dict[str, Dict[str, Step]] = {}
    calls = loads = failed = 0
    checks = []
    for name, path in sorted(st.sets.items()):
        tasks = [original(t) for t in st.tasks if set_of(t.id) == name]
        runner = runner_for(a, provider, out / "confsweep" / name / "cache.jsonl", [path / "cache.jsonl"])
        box = live_box(a, st, runner, logprobs=TOP_LOGPROBS)
        raw = await sweep_ants(box, tasks, base, f"confsweep {name}", a.concurrency)
        live, n_loads = trim_loads(raw)
        checks.append(cache_check(live, tasks, box, st))
        conf.update(remap(live, box.roster, st.roster, name))
        calls, loads, failed = calls + runner.calls, loads + n_loads, failed + runner.failed
    st.conf = conf
    H.save_bundle(out / "confsweep", [t for t in st.tasks if t.id in conf], st.key, conf, st.roster,
                  {"source": "confsweep", "top_logprobs": TOP_LOGPROBS})
    cal = calibration(conf, st.key)
    raw = {k: sum(c["raw"][k] for c in checks) for k in checks[0]["raw"]} if checks else {}
    share = lambda x, y: f"{100 * x / y:.0f}%" if y else "n/a"  # noqa: E731
    save_json(out / "confsweep" / "stats.json", {"calibration": cal, "cache": checks, "calls": calls,
                                                 "loads": loads, "failed": failed})
    lines = [f"CONFSWEEP: {calls} new ant calls ({failed} unusable) in {(time.monotonic() - t0) / 3600:.1f} h; "
             f"{loads} load-time outliers counted at the median; answer odds on {pct(cal['known'])} of {cal['answers']} "
             f"ant answers",
             "  how sure vs how often right: " + "; ".join(
                 f"{k}: {v['n']} answers, " + ("n/a" if v["right"] is None else f"{100 * v['right']:.0f}% right")
                 for k, v in cal["bins"].items())]
    if raw:
        lines.append(f"  prompt start reused by the server: second way of looking at a question {share(raw['rest_cached'], raw['rest_in'])} "
                     f"(this code's estimate {share(raw['rest_est'], raw['rest_in'])}); first ask of a question "
                     f"{share(raw['first_cached'], raw['first_in'])} (start shared with the question before); server "
                     f"reported it on {share(raw['known'], raw['n'])} of calls")
    return lines


# ------------------------------------------------------------------ 5: promptopt

def train_sample(st: State, n: int, seed: int = 20261102) -> List[Task]:
    """``n`` training questions, spread over the kinds, only ones already swept with odds when there is a sweep (so
    the original prompt's score there costs nothing)."""
    pool = [t for t in st.tasks if set_of(t.id) in TRAIN and (not st.conf or t.id in st.conf)]
    by: Dict[str, List[Task]] = {}
    for t in pool:
        by.setdefault(t.kind or "", []).append(t)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    out: List[Task] = []
    seen_ids = set()
    while len(out) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(out) < n:
                t = by[k].pop()
                if original_id(t.id) not in seen_ids:      # two sets can reuse an id: keep the first only
                    seen_ids.add(original_id(t.id))
                    out.append(t)
    return out


def one_per_model(roster: Sequence[AntInfo]) -> List[int]:
    best: Dict[str, AntInfo] = {}
    for a in roster:
        if a.tag not in best or a.lens < best[a.tag].lens:
            best[a.tag] = a
    order = cheap_first(roster)
    return [i for i in order if best.get(roster[i].tag) is roster[i]]


def ant_level(steps: Mapping[str, Mapping[str, Step]], key: Mapping[str, str]) -> Dict[str, Any]:
    got = [(tid, s) for tid, per in steps.items() for tool, s in per.items() if (parse_ant(tool) or (0, ""))[1] == "alone"]
    n = len(got)
    return {"n": n, "accuracy": round(sum(s.answer == key.get(t) for t, s in got) / n, 4) if n else None,
            "tokens": round(sum(s.tokens for _, s in got) / n, 1) if n else None,
            "tokens_in": round(sum(s.tokens_in for _, s in got) / n, 1) if n else None,
            "seconds": round(sum(s.latency_s for _, s in got) / n, 2) if n else None}


async def stage_promptopt(a: argparse.Namespace, st: State, provider: Any, base: Optional[str], out: Path
                          ) -> List[str]:
    t0 = time.monotonic()
    logprobs = TOP_LOGPROBS if st.odds else 0
    seeds = [p / "cache.jsonl" for p in st.sets.values()] + [out / "confsweep" / n / "cache.jsonl" for n in st.sets]
    runner = runner_for(a, provider, out / "promptopt" / "cache.jsonl", seeds)
    train = train_sample(st, a.po_train)
    otrain = [original(t) for t in train]
    okey = {original_id(t.id): st.key[t.id] for t in train}
    ants = one_per_model(LiveBox(runner, st.cfg.reader, st.cfg.ants, code=False).roster)

    def make_box(msgs: Optional[Callable[..., Any]]) -> LiveBox:
        return live_box(a, st, runner, logprobs=logprobs, ant_messages=msgs)
    found = await po.search(make_box, runner, st.cfg.reader, otrain, okey, ants, rounds=a.po_rounds,
                            per_round=a.po_per_round, log=log)
    save_json(out / "promptopt" / "search.json", found)
    pick, orig = found["pick"], found["original"]
    lines = [f"PROMPT: tried {len(found['tried'])} ant instructions on {len(train)} training questions with "
             f"{len(ants)} ants (one per model): original {pct(orig['accuracy'])}, {orig['tokens']:.0f} tokens an "
             f"answer; picked [{pick['origin']}] {pct(pick['accuracy'])}, {pick['tokens']:.0f} tokens",
             f"  picked instructions: {pick['system']} | {pick['format']}"]
    if (pick["system"], pick["format"]) == po.ORIGINAL.key:
        lines.append("  nothing shorter held its accuracy on the training questions, so the set 3 check was skipped")
        return lines
    p = po.prompt_of(pick)
    test = [t for t in st.tasks if set_of(t.id) in TEST and (not st.conf or t.id in st.conf)]
    box = make_box(bw.custom_ants(p.system, p.fmt))
    raw = await sweep_ants(box, [original(t) for t in test], base, "promptopt set3", a.concurrency)
    live, _ = trim_loads(raw)
    new = remap(live, box.roster, st.roster, "set3")
    H.save_bundle(out / "promptopt" / "set3", test, st.key, new, st.roster, {"source": "promptopt", "prompt": pick})
    before = bw.overlay(st.steps, st.conf) if st.conf else st.steps
    was, now = ant_level({t.id: before[t.id] for t in test}, st.key), ant_level(new, st.key)
    lines.append(f"  set 3, all {len(st.roster)} ants alone: {pct(now['accuracy'])} vs {pct(was['accuracy'])} with "
                 f"the original; {now['tokens']:.0f} vs {was['tokens']:.0f} tokens an answer ({now['tokens_in']:.0f} "
                 f"vs {was['tokens_in']:.0f} read); {now['seconds']}s vs {was['seconds']}s")
    rep = st.gate_rep if st.conf else st.free_rep
    if rep is None:                                        # this stage run on its own: learn the routes again
        rep = await asyncio.to_thread(stage_gate if st.conf else stage_free, st, out)   # (its own replays: off
        rep = st.gate_rep if st.conf else st.free_rep                                   # this event loop)
    swapped = bw.overlay(before, new)
    res = {"ant_level": {"before": was, "after": now}, "harness": []}
    for label in dict.fromkeys(["harness (today)", rep.get("best_mix")]):
        if not label:
            continue
        # the same routes (learned on sets 1 and 2) replayed on set 3, once with the ants' answers as they were and
        # once with the short-instruction answers in their place, so the two differ only in the instructions
        o = await asyncio.to_thread(bw.swap_score, rep, label, before, st.roster, st.tasks, st.key)
        s = await asyncio.to_thread(bw.swap_score, rep, label, swapped, st.roster, st.tasks, st.key,
                                    bw.Prompts(st.roster, st.cfg.reader, st.specs, bw.custom_ants(p.system, p.fmt)))
        res["harness"].append({"before": o, "after": s})
        t_, o_ = s["test"], o["test"]
        lines.append(f"  {label} with it on set 3: {pct(t_['accuracy'])} (was {pct(o_['accuracy'])}), tokens vs reader "
                     f"x{t_['tokens_ratio']} (was x{o_['tokens_ratio']}), time x{t_['seconds_ratio']} "
                     f"(was x{o_['seconds_ratio']}), {t_['verdict']}")
    save_json(out / "promptopt" / "final.json", res)
    lines.append(f"  ({runner.calls} new calls, {(time.monotonic() - t0) / 3600:.1f} h)")
    return lines


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


# ------------------------------------------------------------------ the commands

class Result:
    """RESULT.txt, written again after every stage so a stopped run still leaves what it finished."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.blocks: List[str] = [HEADER]

    def add(self, lines: Sequence[str]) -> None:
        self.blocks += list(lines)
        self.path.write_text("\n".join(self.blocks + [TEST_NOTE]) + "\n")
        print("\n".join(lines), flush=True)

    def text(self) -> str:
        return "\n".join(self.blocks + [TEST_NOTE])


def run_stage(res: Result, name: str, fn: Callable[[], Any]) -> bool:
    log(f"stage {name}")
    try:
        got = fn()
        if asyncio.iscoroutine(got):
            got = asyncio.run(got)
        res.add(got)
        return True
    except SystemExit:
        raise
    except Exception as exc:                     # one stage failing still leaves the others' results
        res.add([f"{name.upper()}: failed: {type(exc).__name__}: {str(exc)[:300]}"])
        return False


def cmd_free(a: argparse.Namespace) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    st = load_state(a)
    res = Result(a.out / "RESULT.txt")
    ok = run_stage(res, "free", lambda: stage_free(st, a.out))
    print("\n" + res.text())
    return 0 if ok else 1


def cmd_mac(a: argparse.Namespace) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in (a.stages.split(",") if a.stages else STAGES) if s]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stages {bad}; pick from {', '.join(STAGES)}")
    st = load_state(a)
    res = Result(a.out / "RESULT.txt")
    if a.dry_run:
        res.add(["DRY RUN: fake models and fake saved sets, so these numbers mean nothing; it checks the plumbing"])
    if "free" in stages:
        run_stage(res, "free", lambda: stage_free(st, a.out))
    if (a.out / "confsweep" / "steps.json").exists() and "confsweep" not in stages:
        _, _, conf, _, _ = H.load_bundle(a.out / "confsweep")
        st.conf, st.odds = conf, True
    live = [s for s in ("probe", "confsweep", "promptopt") if s in stages]
    if live:
        servers = [a.mlx_server] + ([a.odds_server] if a.odds_server else [])
        for k, cmd in enumerate(servers):
            ctx: Any = _Null() if a.dry_run else Server(a, cmd, f"{k}")
            with ctx as base:
                provider = provider_for(a, st, base)
                version = "fake" if a.dry_run else mlx_version(cmd)
                p = asyncio.run(_probe(provider, st))
                save_json(a.out / f"probe-{k}.json", {**p, "version": version, "server": cmd})
                res.add(probe_lines(p, version))
                st.odds = p["odds"]
                if not p["odds"] and k + 1 < len(servers):
                    res.add([f"  no answer odds from this server; trying --odds-server ({a.odds_server})"])
                    continue
                if "confsweep" in stages:
                    if st.odds:
                        run_stage(res, "confsweep", lambda: stage_confsweep(a, st, provider, base, a.out))
                    else:
                        res.add(["CONFSWEEP: skipped (this server sends no answer odds the gate can read)"])
                if "gate" in stages:
                    run_stage(res, "gate", lambda: stage_gate(st, a.out))
                if "promptopt" in stages:
                    run_stage(res, "promptopt", lambda: stage_promptopt(a, st, provider, base, a.out))
            break
    elif "gate" in stages:
        run_stage(res, "gate", lambda: stage_gate(st, a.out))
    res.add([H.DEV_NOTE])
    print("\n" + res.text())
    return 0


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s: argparse.ArgumentParser) -> None:
        s.add_argument("--night", type=Path, default=None, help="the night run's folder (holds set1/run, ...)")
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--models", type=Path, default=H.MODELS)
        s.add_argument("--limit", type=int, default=0, help="at most this many questions a set (0: all)")
        s.add_argument("--dry-run", action="store_true", help="fake models and fake saved sets, $0, seconds")
        s.add_argument("--fake-per-task", type=int, default=2, help=argparse.SUPPRESS)

    s = sub.add_parser("free", help="stop rule, router and cache-aware count on the saved answers; no model calls")
    common(s)
    s = sub.add_parser("mac", help="every stage, one block to paste back")
    common(s)
    s.add_argument("--stages", default="", help=f"comma list from {','.join(STAGES)} (default: all, in order)")
    s.add_argument("--mlx-server", default="", help="the mlx_lm.server command (a path, or a command line)")
    s.add_argument("--odds-server", default="",
                   help="a second mlx_lm.server command to try if the first sends no answer odds")
    s.add_argument("--port", type=int, default=H.PORT)
    s.add_argument("--mlx-port", type=int, default=H.MLX_PORT)
    s.add_argument("--concurrency", type=int, default=2)
    s.add_argument("--timeout", type=float, default=600.0)
    s.add_argument("--drive", type=Path, default=None, help="a folder on the model drive, checked before retries")
    s.add_argument("--po-train", type=int, default=30, help="training questions for the prompt search")
    s.add_argument("--po-rounds", type=int, default=3)
    s.add_argument("--po-per-round", type=int, default=2)
    a = p.parse_args(argv)
    if a.cmd == "mac" and not a.dry_run and not a.mlx_server:
        live = [x for x in (a.stages.split(",") if a.stages else STAGES) if x in ("probe", "confsweep", "promptopt")]
        if live:
            raise SystemExit("mac needs --mlx-server for the live stages (or --stages free,gate, or --dry-run)")
    return {"free": cmd_free, "mac": cmd_mac}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
