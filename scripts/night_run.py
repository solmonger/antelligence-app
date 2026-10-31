"""The night run: about ten hours on the Mac, unattended, ending in one plain report. See docs/plans/2026-10-31-night-run.md.

What it tries, in this order (each stage saves its files when it ends, so a crash or a stop loses at most the stage
it was in, and a rerun picks up from the caches):

1. ``route``: a free re-score of the first fresh set's saved answers that sends each kind of question to the setup
   that did best on it (plus the same on the AGIEval run, for reference). The routes are frozen to ``routes.json``
   before any new question exists.
2. ``set2``: a brand-new fresh set (new seed): reader + 8 ants, then Jev and d1-3B, with the exact frozen code of the
   first set. ``route-check`` then scores the frozen routes on it: the real test of routing.
3. ``notebook``: field-notebook ants (each ant reads what every other model found working alone, once per model)
   on set 1, then set 2. Do they stop copying the reader's mistakes?
4. ``set3``, if time is left: one more fresh set, for tighter error ranges when the sets are pooled.
5. ``pool`` and ``report``: pooled numbers over every set, ``REPORT.md`` in plain words, and a short block to paste.

Commands::

    uv run python scripts/night_run.py all --hours 10 --mlx-server <path> --d1-python <path> [--agieval <dir>]
    uv run python scripts/night_run.py all --dry-run --night /tmp/night     # the whole night with fake models, $0

The single steps (``route``, ``set``, ``route-check``, ``notebook``, ``pool``, ``report``) can be run by hand too.

Safety, built in: one local model at a time; a memory watch every minute (pressure level, swap, free memory and the
``fseventsd`` file watcher's size) that stops the current step and the servers this script started when pressure is
critical or swap grows by more than 2 GB, waits for normal, and retries once; it only ever stops processes it started
itself; Jev under 5 cents for the whole night; the MLX server on port 8091 so the chat shim on 8090 is never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from antelligence.experiments import dm_addon as dm
from antelligence.experiments import fair_test as ft
from antelligence.experiments import notebook_exam as nbx
from antelligence.experiments import route as rt
from antelligence.experiments import shared_read as sr

REPO = Path(__file__).resolve().parents[1]
NIGHT = REPO / "docs/research/night-20261031"
FRESH = REPO / "docs/research/fresh-20261031"
AGIEVAL = REPO / "docs/research/fair-test-20261029"
MODELS = FRESH / "models.json"
STANDIN_PORT, MLX_PORT, D1_PORT = 11434, 8091, 8520
JEV_NIGHT_CAP = 0.05          # dollars, the whole night (each set's own frozen cap is $0.01)
SWAP_GROWTH_GB = 2.0
FSEVENTSD_WARN_GB = 8.0
# Minutes each kind of step took (or is expected to take) on the Mac, to decide whether set 3 still fits.
MINUTES = {"swarm": 130, "dm_jev": 5, "dm_d1": 20, "notebook": 90}
CODE = ("antelligence/experiments/route.py", "antelligence/experiments/notebook_exam.py", "scripts/night_run.py",
        "docs/research/fresh-20261031/prereg.json")
DMS = ("jev", "d1-3b")


# ---------------------------------------------------------------- small helpers

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    return dm.read_jsonl(p) if Path(p).exists() else []


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    tmp.replace(p)


def read_json(p: Path, default: Any = None) -> Any:
    return json.loads(p.read_text()) if p.exists() else default


def run_of(night: Path, name: str, dry: bool) -> Path:
    return night / name / ("dry-run" if dry else "run")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------- freezing

def fingerprint() -> Dict[str, Any]:
    return {"code": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in CODE},
            "route_lines": rt.LINES, "route_prefer": list(rt.PREFER), "notebook_lines": nbx.LINES,
            "jev_night_cap": JEV_NIGHT_CAP, "swap_growth_gb": SWAP_GROWTH_GB, "mlx_port": MLX_PORT}


def changed(night: Path) -> List[str]:
    p = NIGHT / "prereg.json"
    if not p.exists():
        return ["no prereg.json"]
    old, new = json.loads(p.read_text()), fingerprint()
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def need_frozen(night: Path) -> None:
    ch = changed(night)
    if ch:
        raise SystemExit(f"the night plan changed since it was frozen ({', '.join(ch)}). Don't re-freeze: tell Eshaan.")


def cmd_freeze(a: argparse.Namespace) -> int:
    write_json(NIGHT / "prereg.json", fingerprint())
    print(f"froze -> {NIGHT / 'prereg.json'}")
    return 0


# ---------------------------------------------------------------- set tables

def fresh_table(d: Path) -> rt.Table:
    key = {r["id"]: r["label"] for r in read_jsonl(d / "answers.jsonl")}
    swarm = dm.export_shared_read(read_jsonl(d / "rows.jsonl"))
    dms = {n: {r["id"]: r for r in read_jsonl(d / f"dm_{n}.jsonl")} for n in DMS if (d / f"dm_{n}.jsonl").exists()}
    return rt.table(swarm, dms, key)


def agieval_table(d: Path) -> rt.Table:
    """The AGIEval shared-read run as exported by the decision-model add-on (``swarm_shared-read.jsonl`` and its
    ``dm_<name>_shared-read.jsonl``), grouped by AGIEval task."""
    key = {r["id"]: r["label"] for r in read_jsonl(AGIEVAL / "exam_answers.jsonl")}
    swarm = [dm.SwarmRow.from_json(r) for r in read_jsonl(d / "swarm_shared-read.jsonl")]
    dms = {n: {r["id"]: r for r in read_jsonl(d / f"dm_{n}_shared-read.jsonl")} for n in DMS
           if (d / f"dm_{n}_shared-read.jsonl").exists()}
    return rt.table(swarm, dms, key)


# ------------------------------------------------------------------------ route

def cmd_route(a: argparse.Namespace) -> int:
    need_frozen(a.night)
    d = run_of(a.night, "set1", a.dry_run)
    t = fresh_table(d)
    if not t:
        raise SystemExit(f"no saved answers in {d}: copy the first fresh run there first")
    arms = rt.common_arms(t)
    res: Dict[str, Any] = {"set1": rt.study(t, arms)}
    if a.agieval:
        at = agieval_table(Path(a.agieval).expanduser())
        if at:
            res["agieval"] = {**rt.study(at, rt.common_arms(at)),
                              "caution": "AGIEval is public and Jev likely memorised it (fresh run 1), so Jev's arms "
                                         "here are flattered"}
    write_json(a.night / "route_set1.json", res)
    frozen = a.night / ("routes_dry.json" if a.dry_run else "routes.json")
    if frozen.exists():
        log(f"routes already frozen in {frozen.name}: kept, so set 2 is scored on the routes picked before it existed")
    else:
        write_json(frozen, {**res["set1"]["picked"], "picked_on": "set1", "picked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    s = res["set1"]
    log(f"route on set 1: reader {s['reader']:.1%}; in-sample routed {s['in_sample']['routed']:.1%} (optimistic); "
        f"split-half routed {s['split_half']['routed']:.1%}, single best {s['split_half']['single']:.1%}; "
        f"routes {json.dumps({g: v['arm'] for g, v in s['picked']['routes'].items()})}")
    return 0


def cmd_route_check(a: argparse.Namespace) -> int:
    need_frozen(a.night)
    frozen = read_json(a.night / ("routes_dry.json" if a.dry_run else "routes.json"))
    if not frozen:
        raise SystemExit("no frozen routes: run `route` first")
    t = fresh_table(run_of(a.night, a.set, a.dry_run))
    if not t:
        raise SystemExit(f"no answers for {a.set}")
    res = rt.held_out(frozen, t)
    write_json(a.night / f"route_check_{a.set}.json", res)
    log(f"routes from set 1 on {a.set}: routed {res['routed']:.1%} vs reader {res['reader']:.1%} "
        f"({res['routed_vs_reader']['gain']:+.1%} {res['routed_vs_reader']['ci95']}); {res['verdict']}")
    return 0


# ------------------------------------------------------------------ fresh sets

def prepare_set(night: Path, name: str) -> Path:
    """A set folder that runs under the first set's frozen plan, unchanged (same code, same caps)."""
    d = night / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FRESH / "prereg.json", d / "prereg.json")
    return d


# ---------------------------------------------------------------- notebook

def _ss():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_shared_read_cli_night", REPO / "scripts/shared_read.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def cmd_notebook(a: argparse.Namespace) -> int:
    need_frozen(a.night)
    d = run_of(a.night, a.set, a.dry_run)
    rows = read_jsonl(d / "rows.jsonl")
    if not rows:
        raise SystemExit(f"no first-round answers in {d}")
    meta = read_json(d / "meta.json", {})
    usable = sr.usable_share(rows)
    if usable < sr.MIN_USABLE:
        raise SystemExit(f"only {usable:.0%} of {a.set}'s first-round answers are usable: the notebook needs a sound "
                         "first round")
    ss = _ss()
    ns = argparse.Namespace(base_url=f"http://localhost:{STANDIN_PORT}/v1", models=str(MODELS), headroom_gb=6.0,
                            out=str(d), dry_run=a.dry_run, concurrency=2, timeout=600.0, loose_names=True, limit=0,
                            max_fail_streak=30, retry_unusable=False, seed_cache=[], parser=None, offline=False)
    cfg = ss.config(ns)
    roster = sr.ant_roster(cfg.ants)
    if meta.get("ants") and [m[0] for m in meta["ants"]] != [s.tag for s, _ in roster]:
        raise SystemExit("the first round's ants are not the ants in the model file: not the same swarm")
    items = [i for i in sr.load_items(d / "questions.jsonl")]
    by_id = {r["id"]: r for r in rows}
    cache = {}
    for line in (d / "cache.jsonl").read_text().splitlines() if (d / "cache.jsonl").exists() else []:
        if line.strip():
            row = json.loads(line)
            if not row.get("error"):
                cache[row["request_hash"]] = row
    first = nbx.first_round(items, by_id, cache, roster, cfg.parser)
    reasons = sum(bool(r) for v in first.values() for _, r in v)
    runner = sr.Runner(ss.provider(ns), d / "nb_cache.jsonl", concurrency=2, check_names=False, max_fail_streak=30)
    p = nbx.plan(len(first), len(roster))
    log(f"notebook on {a.set}: {p['items']} questions, {p['calls']} calls ({len(runner.cache)} already done); "
        f"{reasons} of {sum(len(v) for v in first.values())} first-round reasons found")

    def model(spec: sr.ModelSpec) -> None:
        ss.guard(ns, spec)
        log(f"  notebook: ant model {spec.tag}")

    t0 = time.monotonic()
    import asyncio

    try:
        nb = asyncio.run(nbx.run(runner, items, by_id, first, cfg.ants, on_model=model,
                                 after_model=lambda s: ss.unload(ns, s), parser=cfg.parser))
    except sr.BrokenRun as exc:
        log(f"STOPPED: {exc}")
        return 2
    merged = nbx.merge(rows, nb)
    dm.write_jsonl(d / "nb_rows.jsonl", merged)
    write_json(d / "nb_meta.json", {"new_calls": runner.calls, "failed_calls": runner.failed,
                                    "seconds": round(time.monotonic() - t0), "reasons_found": reasons})
    log(f"notebook on {a.set} done: {runner.calls} new calls ({runner.failed} unusable)")
    return 0


# ----------------------------------------------------------------------- pool

def _sets(night: Path, dry: bool) -> List[str]:
    return [n for n in ("set1", "set2", "set3") if (run_of(night, n, dry) / "rows.jsonl").exists()]


def _key(d: Path) -> Dict[str, str]:
    return {r["id"]: r["label"] for r in read_jsonl(d / "answers.jsonl")}


def _b(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Any]:
    g, lo, hi = ft.paired_boot([x - y for x, y in zip(xs, ys)])
    return {"gain": round(g, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(xs)}


def pool(night: Path, dry: bool) -> Dict[str, Any]:
    """Every set's main numbers, and the same pooled over all sets (ids prefixed with the set name)."""
    out: Dict[str, Any] = {"sets": {}, "pooled": {}}
    tables: Dict[str, rt.Table] = {}
    nb_rows: List[Dict[str, Any]] = []
    nb_key: Dict[str, str] = {}
    for name in _sets(night, dry):
        d = run_of(night, name, dry)
        t = fresh_table(d)
        tables[name] = t
        info = {"n": len(t), "seed": (read_json(d / "set.json", {}) or {}).get("seed"),
                "means": {arm: round(st.mean(v["right"][arm] for v in t.values()), 4) for arm in rt.common_arms(t)}}
        rows = read_jsonl(d / "nb_rows.jsonl")
        if rows:
            key = _key(d)
            info["notebook"] = nbx.score(rows, key)
            for r in rows:
                nb_rows.append({**r, "id": f"{name}:{r['id']}"})
            nb_key.update({f"{name}:{i}": v for i, v in key.items()})
        out["sets"][name] = info
    allt = {f"{n}:{i}": v for n, t in tables.items() for i, v in t.items()}
    if allt:
        arms = rt.common_arms(allt)
        col = {arm: [v["right"][arm] for v in allt.values()] for arm in arms}
        P: Dict[str, Any] = {"n": len(allt), "sets": list(tables),
                             "means": {arm: round(st.mean(c), 4) for arm, c in col.items()}}
        pairs = [("jev:dm1", "reader"), ("jev:read", "reader"), ("notes8", "alone8"), ("notes8", "reader"),
                 ("both8", "reader"), ("jev:vote", "notes8")]
        P["compare"] = {f"{x} vs {y}": _b(col[x], col[y]) for x, y in pairs if x in col and y in col}
        P["by_kind"] = {g: {arm: round(st.mean(v["right"][arm] for v in allt.values() if v["group"] == g), 4)
                            for arm in arms} for g in rt.groups(allt)}
        out["pooled"] = P
    if nb_rows:
        out["pooled"]["notebook"] = nbx.score(nb_rows, nb_key)
    for name in ("set2", "set3"):
        rc = read_json(night / f"route_check_{name}.json")
        if rc:
            out.setdefault("route_check", {})[name] = rc
    return out


def cmd_pool(a: argparse.Namespace) -> int:
    res = pool(a.night, a.dry_run)
    write_json(a.night / ("pool_dry.json" if a.dry_run else "pool.json"), res)
    log(f"pooled {res['pooled'].get('n', 0)} questions over {', '.join(res['pooled'].get('sets', []))}")
    return 0


# --------------------------------------------------------------------- report

def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def gap(b: Optional[Mapping[str, Any]]) -> str:
    if not b:
        return "n/a"
    return f"{b['gain']:+.1%} (range {b['ci95'][0]:+.1%} to {b['ci95'][1]:+.1%})"


def spread(b: Optional[Mapping[str, Any]]) -> str:
    if not b:
        return "n/a"
    return f"{b['gain']:+.1%} (moves {b['spread'][0]:+.1%} to {b['spread'][1]:+.1%} with the split)"


def report(night: Path, dry: bool) -> Dict[str, str]:
    """REPORT.md in plain words, and the short block to paste back. Numbers only from the saved files."""
    state = read_json(night / ("state_dry.json" if dry else "state.json"), {}) or {}
    route1 = read_json(night / "route_set1.json", {}) or {}
    P = pool(night, dry)
    pooled = P.get("pooled") or {}
    L: List[str] = ["# Night run, 2026-10-31: report", "",
                    "Written by `scripts/night_run.py report` from the saved files. Plan: "
                    "`docs/plans/2026-10-31-night-run.md`.", ""]
    B: List[str] = ["NIGHT RUN RESULT (paste this back to Claude in Duet)"]

    L += ["## What ran", ""]
    stages = state.get("stages") or {}
    names = [n for n in state.get("order", []) if n in stages] + sorted(set(stages) - set(state.get("order", [])))
    for name in names:
        s = stages[name]
        mins = f", {s['seconds'] / 60:.0f} min" if s.get("seconds") else ""
        L.append(f"- {name}: {s['status']}{mins}" + (f". {s['note']}" if s.get("note") else ""))
    stops = state.get("memory_stops", 0)
    L += ["", f"Memory: {stops} stop(s) for memory. Peak swap growth {state.get('peak_swap_growth_gb', 0):.1f} GB, "
              f"lowest free memory {state.get('min_free_pct', 'n/a')}%, largest fseventsd "
              f"{state.get('max_fseventsd_gb', 0):.1f} GB. Jev spent ${state.get('jev_usd', 0):.4f} tonight.", ""]
    B.append("stages: " + "; ".join(f"{k} {stages[k]['status']}" for k in names))

    # 1. routing
    L += ["## 1. Sending each kind of question to its best setup", ""]
    s1 = route1.get("set1")
    if s1:
        routes = {g: v["arm"] for g, v in s1["picked"]["routes"].items()}
        L += [f"Picked on set 1 ({s1['n']} questions): " + ", ".join(f"{g} to {a}" for g, a in routes.items()) +
              f". The single best setup on set 1 was {s1['picked']['single']['arm']}.", "",
              f"- On set 1 itself (optimistic, it picks and scores on the same questions): routed "
              f"{pct(s1['in_sample']['routed'])} against the reader's {pct(s1['reader'])}.",
              f"- Split-half check on set 1 (routes picked on half of each kind, scored on the other half, over "
              f"{s1['split_half']['reps']} random splits): routed {pct(s1['split_half']['routed'])}, "
              f"{spread(s1['split_half']['routed_vs_reader'])} against the reader, "
              f"{spread(s1['split_half']['routed_vs_single'])} against the single best setup."]
        B.append(f"route set1: reader {pct(s1['reader'])}; in-sample {pct(s1['in_sample']['routed'])}; "
                 f"split-half {pct(s1['split_half']['routed'])} vs reader "
                 f"{spread(s1['split_half']['routed_vs_reader'])}; routes {json.dumps(routes)}")
    for name, rc in (P.get("route_check") or {}).items():
        L.append(f"- **The real test, on {name}** (new questions, routes frozen before they existed): routed "
                 f"{pct(rc['routed'])} against the reader's {pct(rc['reader'])}, {gap(rc['routed_vs_reader'])}; "
                 f"against the single best setup ({rc['single']['arm']}, {pct(rc['single']['score'])}): "
                 f"{gap(rc['routed_vs_single'])}. Verdict: {rc['verdict']}.")
        L.append("  By kind: " + "; ".join(f"{g} ({v['arm']}) {pct(v['routed'])} vs reader {pct(v['reader'])}"
                                         for g, v in rc["by_kind"].items()))
        B.append(f"route check {name}: routed {pct(rc['routed'])} vs reader {pct(rc['reader'])} "
                 f"{gap(rc['routed_vs_reader'])}; vs single {gap(rc['routed_vs_single'])}; VERDICT {rc['verdict']}")
    ag = route1.get("agieval")
    if ag:
        L.append(f"- On the 240 AGIEval questions, for reference only (Jev likely memorised them): split-half routed "
                 f"{pct(ag['split_half']['routed'])} against the reader's {pct(ag['reader'])}, "
                 f"{spread(ag['split_half']['routed_vs_reader'])}.")
    L.append("")

    # 2. notebook
    L += ["## 2. Field-notebook ants", "",
          "The scripted rumour-world version is already rebuilt and recorded (draft PR 24: about +8 to +10 points). "
          "Tonight's question was whether the idea carries over to the exam swarm: each ant reads what every other "
          "model found working alone, one line per model, before it answers.", ""]
    nbp = pooled.get("notebook")
    if nbp:
        arms = nbp["arms"]
        L += [f"Pooled over {nbp['n']} questions (the reader was wrong on {nbp['reader_wrong_items']}):", "",
              "| setup | swarm right | average ant | ants picking a wrong reader's letter | mistakes alike |",
              "|---|---|---|---|---|"]
        for arm, label in (("alone", "ants alone"), ("notes", "ants with notes"), ("both", "question + notes"),
                           ("nb", "question + notebook"), ("nb_notes", "question + notes + notebook")):
            x = arms[arm]
            L.append(f"| {label} | {pct(x['vote'])} | {pct(x['ant_mean'])} | {pct(x['copies_wrong_reader'])} | "
                     f"{x['error_alikeness']:.2f} |")
        L += ["", f"- The reader alone: {pct(nbp['reader_alone'])}.",
              f"- Notebook on top of notes vs notes alone (question + notes): {gap(nbp['nb_notes_vs_both'])}.",
              f"- Notebook vs ants alone: {gap(nbp['nb_vs_alone'])}.",
              f"- Copying a wrong reader, drop from question + notes to + notebook: {gap(nbp['copy_drop_vs_both'])}.",
              f"- Best notebook setup ({nbp['best_arm']}) vs the reader alone: {gap(nbp['best_vs_reader'])}.",
              f"- Verdict: {json.dumps(nbp['verdict'])}.", ""]
        B.append(f"notebook pooled n {nbp['n']}: nb_notes {pct(arms['nb_notes']['vote'])} vs both "
                 f"{pct(arms['both']['vote'])} {gap(nbp['nb_notes_vs_both'])}; nb {pct(arms['nb']['vote'])} vs alone "
                 f"{pct(arms['alone']['vote'])}; reader {pct(nbp['reader_alone'])}; copy wrong reader both "
                 f"{pct(arms['both']['copies_wrong_reader'])} -> nb_notes {pct(arms['nb_notes']['copies_wrong_reader'])}"
                 f" {gap(nbp['copy_drop_vs_both'])}; VERDICT {json.dumps(nbp['verdict'])}")
    else:
        L += ["No notebook run finished.", ""]

    # 3. pooled sets
    L += ["## 3. All fresh sets pooled", ""]
    if pooled.get("means"):
        L.append(f"{pooled['n']} new questions over {', '.join(pooled['sets'])}. Share right:")
        L.append("")
        for arm, m in sorted(pooled["means"].items(), key=lambda kv: -kv[1]):
            L.append(f"- {arm}: {pct(m)}")
        L.append("")
        for k, v in pooled.get("compare", {}).items():
            L.append(f"- {k}: {gap(v)}")
        L.append("")
        L.append("By kind: " + "; ".join(f"{g}: reader {pct(v.get('reader'))}, notes8 {pct(v.get('notes8'))}, "
                                         f"jev:dm1 {pct(v.get('jev:dm1'))}" for g, v in pooled["by_kind"].items()))
        B.append(f"pooled n {pooled['n']} ({', '.join(pooled['sets'])}): " + "; ".join(
            f"{k} {gap(v)}" for k, v in pooled.get("compare", {}).items()))
    L += ["", "## What helped, what didn't, what next", "",
          "_Claude Code on the Mac fills this in from the numbers above, in plain words, and changes nothing above._", ""]
    md = "\n".join(L) + "\n"
    (night / ("REPORT_dry.md" if dry else "REPORT.md")).write_text(md)
    return {"report": md, "block": "\n".join(B)}


def cmd_report(a: argparse.Namespace) -> int:
    r = report(a.night, a.dry_run)
    print(r["block"])
    return 0


# ---------------------------------------------------------------- memory watch

def _sh(cmd: Sequence[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def memory_now() -> Dict[str, Optional[float]]:
    """macOS memory: pressure level (1 normal, 2 warning, 4 critical), swap used (GB), free share (%), and the
    fseventsd file watcher's resident size (GB). Every value is None where it can't be read (and on Linux)."""
    if sys.platform != "darwin":
        return {"level": None, "swap_gb": None, "free_pct": None, "fseventsd_gb": None}
    level = _sh(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip()
    swap = re.search(r"used = ([\d.]+)([MG])", _sh(["sysctl", "vm.swapusage"]))
    free = re.search(r"free percentage:\s*(\d+)%", _sh(["memory_pressure", "-Q"]))
    fse = 0.0
    found = False
    for line in _sh(["ps", "-axo", "rss=,comm="]).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[1].endswith("fseventsd"):
            fse += int(parts[0]) / 1e6
            found = True
    return {"level": float(level) if level.isdigit() else None,
            "swap_gb": (float(swap.group(1)) / (1000 if swap.group(2) == "M" else 1)) if swap else None,
            "free_pct": float(free.group(1)) if free else None, "fseventsd_gb": round(fse, 2) if found else None}


class Watch:
    """Checks memory every ``every`` seconds. ``tripped`` is set (with a reason) when pressure is critical or swap
    has grown more than ``SWAP_GROWTH_GB`` since the baseline; the night loop then stops the step it is in."""

    def __init__(self, path: Path, read: Callable[[], Dict[str, Optional[float]]] = memory_now, every: float = 60.0):
        self.path, self.read, self.every = path, read, every
        self.tripped: Optional[str] = None
        self.base_swap: Optional[float] = None
        self.peak_growth = 0.0
        self.min_free: Optional[float] = None
        self.max_fse = 0.0
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def check(self) -> Dict[str, Optional[float]]:
        m = self.read()
        if m.get("swap_gb") is not None:
            if self.base_swap is None:
                self.base_swap = m["swap_gb"]
            growth = m["swap_gb"] - self.base_swap
            self.peak_growth = max(self.peak_growth, growth)
            if growth > SWAP_GROWTH_GB and not self.tripped:
                self.tripped = f"swap grew {growth:.1f} GB"
        if m.get("level") is not None and m["level"] >= 4 and not self.tripped:
            self.tripped = "memory pressure critical"
        if m.get("free_pct") is not None:
            self.min_free = m["free_pct"] if self.min_free is None else min(self.min_free, m["free_pct"])
        if m.get("fseventsd_gb") is not None:
            self.max_fse = max(self.max_fse, m["fseventsd_gb"])
        with self.path.open("a") as fh:
            fh.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **m, "tripped": self.tripped}) + "\n")
        return m

    def normal(self) -> bool:
        m = self.read()
        return m.get("level") in (None, 1.0) and (m.get("free_pct") is None or m["free_pct"] >= 20)

    def rebase(self) -> None:
        m = self.read()
        self.base_swap = m.get("swap_gb")
        self.tripped = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def loop() -> None:
            while not self._stop.wait(self.every):
                self.check()

        self.check()
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()


# -------------------------------------------------------------------- services

def port_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_http(url: str, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(2)
    return False


class Service:
    """A server this script starts and stops itself (its own process group, so stopping it stops its children)."""

    def __init__(self, name: str, cmd: Sequence[str], health: str, log_path: Path, cwd: Path = REPO):
        self.name, self.cmd, self.health, self.log_path, self.cwd = name, list(cmd), health, log_path, cwd
        self.proc: Optional[subprocess.Popen] = None

    def up(self, seconds: float = 900) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        out = self.log_path.open("a")
        self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        log(f"started {self.name} (pid {self.proc.pid})")
        if not wait_http(self.health, seconds):
            self.down()
            raise RuntimeError(f"{self.name} did not answer at {self.health}")

    def down(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=30)
            except ProcessLookupError:
                pass
            log(f"stopped {self.name} (pid {self.proc.pid})")
        self.proc = None


# ------------------------------------------------------------------- the night

class Night:
    def __init__(self, a: argparse.Namespace, watch: Optional[Watch] = None):
        self.a, self.dry = a, a.dry_run
        self.night: Path = a.night
        self.logs = self.night / "logs"
        self.state_path = self.night / ("state_dry.json" if self.dry else "state.json")
        self.state = read_json(self.state_path, {}) or {}
        self.state.setdefault("stages", {})
        self.state.setdefault("memory_stops", 0)
        self.start = time.monotonic() - float(self.state.get("elapsed_s", 0))
        self.deadline_s = a.hours * 3600
        self.watch = watch or Watch(self.logs / "memory.jsonl", every=0.05 if self.dry else 60.0)
        self.services: Dict[str, Service] = {}
        if not self.dry:
            self.services["standin"] = Service(
                "MLX stand-in", [sys.executable, str(REPO / "scripts/mlx_standin.py"), "--models", str(MODELS),
                                 "--mlx-server", a.mlx_server, "--port", str(STANDIN_PORT), "--mlx-port", str(MLX_PORT),
                                 "--log", str(self.logs / "mlx.log")],
                f"http://127.0.0.1:{STANDIN_PORT}/v1/models", self.logs / "standin.log")
            if a.d1_python:
                self.services["d1"] = Service(
                    "d1-3B shim", [a.d1_python, str(REPO / "scripts/v41_d1_server.py"), "--host", "127.0.0.1",
                                   "--port", str(D1_PORT)], f"http://127.0.0.1:{D1_PORT}/health", self.logs / "d1.log")

    # bookkeeping
    def save(self) -> None:
        self.state["elapsed_s"] = round(time.monotonic() - self.start)
        self.state["memory_stops"] = self.state.get("memory_stops", 0)
        self.state["peak_swap_growth_gb"] = round(max(self.state.get("peak_swap_growth_gb", 0), self.watch.peak_growth), 2)
        if self.watch.min_free is not None:
            old = self.state.get("min_free_pct")
            self.state["min_free_pct"] = self.watch.min_free if old is None else min(old, self.watch.min_free)
        self.state["max_fseventsd_gb"] = round(max(self.state.get("max_fseventsd_gb", 0), self.watch.max_fse), 2)
        self.state["jev_usd"] = round(self.jev_spent(), 6)
        write_json(self.state_path, self.state)

    def left_min(self) -> float:
        return (self.deadline_s - (time.monotonic() - self.start)) / 60

    def jev_spent(self) -> float:
        total = 0.0
        for name in ("set2", "set3"):
            p = run_of(self.night, name, self.dry) / "cache_jev.jsonl"
            total += sum(r.get("cost_usd") or 0 for r in read_jsonl(p))
        return total

    def done(self, stage: str) -> bool:
        return self.state["stages"].get(stage, {}).get("status") == "done"

    def mark(self, stage: str, status: str, seconds: float = 0, note: str = "") -> None:
        self.state["stages"][stage] = {"status": status, "seconds": round(seconds), "note": note}
        order = self.state.setdefault("order", [])
        if stage not in order:
            order.append(stage)
        self.save()
        log(f"stage {stage}: {status}" + (f" ({note})" if note else ""))

    # running one step under the memory watch
    def _child(self, cmd: Sequence[str], log_name: str, timeout_s: float) -> str:
        """Run ``cmd``; return "ok", "failed: ..." or "memory: ...". Stops it (and its group) if the watch trips."""
        with (self.logs / f"{log_name}.log").open("a") as out:
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            end = time.monotonic() + timeout_s
            while proc.poll() is None:
                if self.watch.tripped:
                    reason = self.watch.tripped
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=60)
                    except (subprocess.TimeoutExpired, ProcessLookupError):
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    return f"memory: {reason}"
                if time.monotonic() > end:
                    os.killpg(proc.pid, signal.SIGTERM)
                    return f"failed: took longer than {timeout_s / 60:.0f} min"
                time.sleep(0.05 if self.dry else 5)
        return "ok" if proc.returncode == 0 else f"failed: exit {proc.returncode}, see logs/{log_name}.log"

    def recover(self) -> bool:
        """After a memory stop: stop every service this script started, wait for normal (up to 60 min), rebase swap."""
        self.state["memory_stops"] = self.state.get("memory_stops", 0) + 1
        for s in self.services.values():
            s.down()
        end = time.monotonic() + (0.5 if self.dry else 3600)
        calm = 0
        while time.monotonic() < end:
            calm = calm + 1 if self.watch.normal() else 0
            if calm >= 3:
                self.watch.rebase()
                self.save()
                return True
            time.sleep(0.05 if self.dry else 60)
        self.save()
        return False

    def step(self, stage: str, cmd: Sequence[str], needs: Sequence[str] = (), timeout_min: float = 240) -> bool:
        """One resumable step: done steps are skipped; a memory stop is retried once after memory is normal."""
        if self.done(stage):
            return True
        t0 = time.monotonic()
        for attempt in (1, 2):
            try:
                for name, s in self.services.items():   # one local model at a time: only what this step needs is up
                    if name not in needs:
                        s.down()
                for name in needs:
                    if name in self.services:
                        self.services[name].up()
            except RuntimeError as exc:
                self.mark(stage, "failed", time.monotonic() - t0, str(exc))
                return False
            log(f"stage {stage} (attempt {attempt}): {' '.join(Path(c).name if '/' in c else c for c in cmd[1:4])} ...")
            res = self._child(cmd, stage, timeout_min * 60)
            if res == "ok":
                self.mark(stage, "done", time.monotonic() - t0)
                return True
            if res.startswith("memory") and attempt == 1:
                log(f"stage {stage}: stopped for memory ({res}); waiting for memory to settle")
                if not self.recover():
                    self.mark(stage, "skipped", time.monotonic() - t0, "memory did not settle within an hour")
                    return False
                continue
            self.mark(stage, "failed" if not res.startswith("memory") else "skipped", time.monotonic() - t0, res)
            return False
        return False

    def py(self, script: str, *args: str) -> List[str]:
        return [sys.executable, str(REPO / "scripts" / script), *args, *(["--dry-run"] if self.dry else [])]

    def fresh_set(self, name: str, with_notebook: bool) -> bool:
        data = prepare_set(self.night, name)
        dflag = ["--data", str(data)]
        ok = self.step(f"{name}:make", self.py("fresh_test.py", "make", *dflag), timeout_min=10)
        ok = ok and self.step(f"{name}:swarm", self.py("fresh_test.py", "swarm", *dflag), needs=("standin",),
                              timeout_min=300)
        if not ok:
            return False
        if self.dry or self.jev_spent() + 0.01 <= JEV_NIGHT_CAP:
            self.step(f"{name}:jev", self.py("fresh_test.py", "dm", "--dm", "jev", *dflag,
                                             *([] if self.dry else ["--live"])), timeout_min=60)
        else:
            self.mark(f"{name}:jev", "skipped", note="the night's 5-cent Jev cap would be passed")
        if self.dry or "d1" in self.services:
            self.step(f"{name}:d1", self.py("fresh_test.py", "dm", "--dm", "d1-3b", *dflag,
                                            *([] if self.dry else ["--live"])), needs=("d1",), timeout_min=90)
        else:
            self.mark(f"{name}:d1", "skipped", note="no --d1-python given")
        self.step(f"{name}:score", self.py("fresh_test.py", "score", *dflag), timeout_min=10)
        if with_notebook:
            self.step(f"{name}:notebook", self.py("night_run.py", "notebook", "--set", name, "--night", str(self.night)),
                      needs=("standin",), timeout_min=240)
        return True

    def run(self) -> int:
        need_frozen(self.night)
        self.logs.mkdir(parents=True, exist_ok=True)
        if not (run_of(self.night, "set1", self.dry) / "rows.jsonl").exists():
            if not self.dry:
                raise SystemExit("set 1 is missing: copy the first fresh run's run/ folder to "
                                 f"{run_of(self.night, 'set1', False)} first")
            self.fresh_set("set1", with_notebook=False)   # dry: make a fake first set
        if not self.dry:
            busy = [p for p in (STANDIN_PORT, MLX_PORT, D1_PORT) if not port_free(p)]
            if busy:
                raise SystemExit(f"ports {busy} are in use: stop only what this run would start, never anything else")
        nflag = ["--night", str(self.night)]
        self.watch.start()
        try:
            agi =["--agieval", self.a.agieval] if self.a.agieval else []
            self.step("route", self.py("night_run.py", "route", *nflag, *agi), timeout_min=10)
            if self.fresh_set("set2", with_notebook=False):
                self.step("route-check:set2", self.py("night_run.py", "route-check", "--set", "set2", *nflag),
                          timeout_min=10)
            self.step("set1:notebook", self.py("night_run.py", "notebook", "--set", "set1", *nflag),
                      needs=("standin",), timeout_min=240)
            if self.done("set2:swarm"):
                self.step("set2:notebook", self.py("night_run.py", "notebook", "--set", "set2", *nflag),
                          needs=("standin",), timeout_min=240)
            need = MINUTES["swarm"] + MINUTES["dm_jev"] + MINUTES["dm_d1"]
            if self.done("set3:swarm") or self.left_min() >= need + 20 or self.dry:
                with_nb = self.dry or self.left_min() >= need + MINUTES["notebook"] + 20
                if self.fresh_set("set3", with_notebook=with_nb):
                    self.step("route-check:set3", self.py("night_run.py", "route-check", "--set", "set3", *nflag),
                              timeout_min=10)
            else:
                self.mark("set3", "skipped", note=f"only {self.left_min():.0f} min left, a set takes about {need}")
        finally:
            for s in self.services.values():
                s.down()
            self.watch.stop()
            self.save()
        self.step("pool", self.py("night_run.py", "pool", *nflag), timeout_min=10)
        self.state["stages"].pop("report", None)
        r = report(self.night, self.dry)
        self.mark("report", "done")
        r = report(self.night, self.dry)   # again, so the report lists itself as done
        print(r["block"], flush=True)
        return 0


def cmd_all(a: argparse.Namespace) -> int:
    if not a.dry_run and not a.mlx_server:
        raise SystemExit("--mlx-server is needed (the mlx_lm.server the first fresh run used)")
    return Night(a).run()


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "route", "route-check", "notebook", "pool", "report", "all"):
        p = sub.add_parser(name)
        p.add_argument("--night", type=Path, default=NIGHT)
        p.add_argument("--dry-run", action="store_true", help="fake models, $0")
    sub.choices["route"].add_argument("--agieval", default="", help="the dm-addon run folder (swarm_shared-read.jsonl)")
    for name in ("route-check", "notebook"):
        sub.choices[name].add_argument("--set", required=True, choices=("set1", "set2", "set3"))
    al = sub.choices["all"]
    al.add_argument("--hours", type=float, default=10.0)
    al.add_argument("--mlx-server", default="", help="the mlx_lm.server command")
    al.add_argument("--d1-python", default="", help="the Python of the d1-3B venv (for scripts/v41_d1_server.py)")
    al.add_argument("--agieval", default="", help="the dm-addon run folder, for routing on AGIEval too")
    a = ap.parse_args(argv)
    return {"freeze": cmd_freeze, "route": cmd_route, "route-check": cmd_route_check, "notebook": cmd_notebook,
            "pool": cmd_pool, "report": cmd_report, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
