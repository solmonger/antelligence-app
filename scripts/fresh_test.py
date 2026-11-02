"""The fresh-question test: is Jev's 81.7% on AGIEval reasoning or memory? See docs/plans/2026-10-31-fresh-questions.md.

New exam questions are made on the Mac when the run starts (new names and numbers from a seed drawn then), so no
model can have seen them. The same arms as run 1 then answer them: the local reader alone; eight small local ants
alone, with the reader's notes, and both; Jev and Liquid d1-3B alone, reading the ants' answers, and as one more
voter. A second, cheap check moves the answer letters on the 240 AGIEval questions Jev got 81.7% on.

Steps, in this order::

    uv run python scripts/fresh_test.py make              # 160 new questions (40 of each kind), checked by brute force
    uv run python scripts/fresh_test.py swarm             # reader + 8 ants, local and free (needs the MLX stand-in)
    uv run python scripts/fresh_test.py dm --dm jev --live           # Jev, capped at $0.01
    uv run python scripts/fresh_test.py dm --dm d1-3b --live         # free, local (start its server first)
    uv run python scripts/fresh_test.py leak --dm jev --live --dm1 <run 1's dm_jev_shared-read.jsonl>   # $0.015 cap
    uv run python scripts/fresh_test.py score             # one block to paste back

``--dry-run`` on swarm, dm and leak uses fake models, $0. Everything resumes from its caches.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import secrets
import statistics as st
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional, Sequence

from antelligence.experiments import dm_addon as dm
from antelligence.experiments import dm_leak as leak
from antelligence.experiments import fair_test as ft
from antelligence.experiments import fresh_questions as fq
from antelligence.experiments import shared_read as sr

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs/research/fresh-20261031"
MODELS = DATA / "models.json"
AGIEVAL = REPO / "docs/research/fair-test-20261029"
STANDIN = "http://localhost:11434/v1"
PER_TASK = 40
CAPS = {"dm": 0.01, "leak": 0.015}           # with run 1's $0.0223, Jev stays under 5 cents in all
# Run 1 on the 240 public AGIEval questions (docs/research/dm-addon-20261030/RESULTS.md), for side-by-side reading.
RUN1 = {"jev:dm1": 0.817, "jev:read": 0.775, "reader": 0.633, "notes8": 0.575, "jev:vote": 0.538,
        "d1-3b:vote": 0.488, "d1-3b:dm1": 0.471}
RUN1_LEAD = 0.183                             # Jev alone minus the reader alone, run 1
# Frozen reading of Jev's lead over the reader on the new questions (Jev alone minus reader alone).
LEAD = {"holds": 0.10, "gone": 0.10}
CODE = ("antelligence/experiments/fresh_questions.py", "antelligence/experiments/dm_leak.py",
        "antelligence/experiments/dm_addon.py", "antelligence/experiments/shared_read.py",
        "antelligence/kernel/decision.py", "scripts/fresh_test.py", "scripts/shared_read.py", "scripts/dm_addon.py",
        "scripts/mlx_standin.py", "docs/research/fresh-20261031/models.json")


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_{name}_cli", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    return dm.read_jsonl(p) if p.exists() else []


def run_dir(a: argparse.Namespace) -> Path:
    d = a.data / ("dry-run" if getattr(a, "dry_run", False) else "run")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------- freeze

def fingerprint() -> Dict[str, Any]:
    return {"code": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in CODE},
            "per_task": PER_TASK, "tasks": list(fq.TASKS), "caps": CAPS, "lead_lines": LEAD,
            "leak_lines": leak.LINES, "run1": RUN1, "dms": {n: dm.DMS[n].model for n in ("jev", "d1-3b")}}


def changed(a: argparse.Namespace) -> List[str]:
    path = a.data / "prereg.json"
    if not path.exists():
        return ["no prereg.json"]
    old, new = json.loads(path.read_text()), fingerprint()
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def need_frozen(a: argparse.Namespace) -> None:
    ch = changed(a)
    if ch:
        raise SystemExit(f"the plan changed since it was frozen ({', '.join(ch)}). Don't re-freeze: tell Eshaan.")


def cmd_freeze(a: argparse.Namespace) -> int:
    a.data.mkdir(parents=True, exist_ok=True)
    (a.data / "prereg.json").write_text(json.dumps(fingerprint(), indent=1, sort_keys=True) + "\n")
    print(f"froze -> {a.data / 'prereg.json'}")
    return 0


# ------------------------------------------------------------------------- make

def cmd_make(a: argparse.Namespace) -> int:
    need_frozen(a)
    d = run_dir(a)
    if (d / "questions.jsonl").exists():
        print(f"A question set is already in {d}: kept as it is, so a stopped run resumes on the same questions.")
        return 0
    seed = a.seed if a.seed is not None else secrets.randbits(63)
    start = time.monotonic()
    qs, ans, specs = fq.make_set(seed, a.per_task)
    bad = [q["id"] for q, s in zip(qs, specs) if not fq.check(q, s)]
    if bad:
        raise SystemExit(f"{len(bad)} questions failed the brute-force check ({bad[:5]}): nothing written")
    dm.write_jsonl(d / "questions.jsonl", qs)
    dm.write_jsonl(d / "answers.jsonl", ans)
    dm.write_jsonl(d / "specs.jsonl", specs)
    counts = {t: sum(q["task"] == t for q in qs) for t in fq.TASKS}
    (d / "set.json").write_text(json.dumps({"seed": seed, "per_task": a.per_task, "counts": counts,
                                            "made_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1) + "\n")
    print(f"{len(qs)} new questions ({', '.join(f'{t} {n}' for t, n in counts.items())}), every one checked by brute "
          f"force, in {time.monotonic() - start:.1f}s. Seed {seed}. -> {d}")
    return 0


def _questions(d: Path) -> List[ft.Question]:
    if not (d / "questions.jsonl").exists():
        raise SystemExit(f"no questions in {d}: run `make` first")
    return ft.load_questions(d / "questions.jsonl", "exam")


# ------------------------------------------------------------------------ swarm

def cmd_swarm(a: argparse.Namespace) -> int:
    need_frozen(a)
    d = run_dir(a)
    _questions(d)
    ss = _load("shared_read")
    ns = argparse.Namespace(base_url=a.base_url, models=str(a.models), headroom_gb=a.headroom_gb, out=str(d),
                            dry_run=a.dry_run, concurrency=a.concurrency, timeout=600.0, loose_names=True, limit=0,
                            max_fail_streak=a.max_fail_streak, retry_unusable=False, seed_cache=[], parser=None,
                            offline=False)
    cfg = ss.config(ns)
    items = sr.load_items(d / "questions.jsonl")
    runner = ss.make_runner(ns, d)
    p = sr.plan(len(items), cfg.ants)
    print(f"{p['items']} questions, {p['calls']} local calls ({len(runner.cache)} already done). Reader "
          f"{cfg.reader.tag}; ants {', '.join(s.tag for s in cfg.ants)}; parser {cfg.parser}", flush=True)
    t0 = time.monotonic()
    try:
        got = asyncio.run(ss._run(ns, items, runner, cfg.reader, cfg.ants))
    except sr.BrokenRun as exc:
        print(f"STOPPED: {exc}")
        return 2
    meta = {"reader": cfg.reader.tag, "ants": [[s.tag, s.family, lens] for s, lens in sr.ant_roster(cfg.ants)],
            "families": [s.family for s, _ in sr.ant_roster(cfg.ants)], "items": len(got), "parser": cfg.parser,
            "dry_run": bool(a.dry_run), "new_calls": runner.calls, "failed_calls": runner.failed,
            "reused_unusable": runner.reused, "seconds": round(time.monotonic() - t0)}
    dm.write_jsonl(d / "rows.jsonl", got)
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"Done: {len(got)} questions, {runner.calls} new calls ({runner.failed} unusable) in "
          f"{meta['seconds'] / 60:.0f} min.")
    return 0


def _swarm(d: Path) -> Dict[str, dm.SwarmRow]:
    rows = read_jsonl(d / "rows.jsonl")
    if not rows:
        raise SystemExit(f"no swarm answers in {d}: run `swarm` first")
    swarm = {r.id: r for r in dm.export_shared_read(rows)}
    answers = [a.answer for r in swarm.values() for a in r.ants]
    share = sum(x is not None for x in answers) / len(answers) if answers else 0.0
    if share < dm.MIN_USABLE:   # run 1's fair-test source went through with every ant empty; this one won't
        raise SystemExit(f"only {share:.0%} of the ants' answers are usable (at least {dm.MIN_USABLE:.0%} needed): "
                         "fix the swarm run before asking any decision model")
    return swarm


# --------------------------------------------------------------------------- dm

def _spec(a: argparse.Namespace) -> dm.DMSpec:
    s = dm.DMS[a.dm]
    return dm.DMSpec(s.name, a.model or s.model, a.url or s.url, s.price, s.hosted, a.concurrency or s.concurrency, s.about)


def _backend(a: argparse.Namespace, spec: dm.DMSpec) -> Any:
    if a.live == a.dry_run:
        raise SystemExit("say --live (the real model) or --dry-run (a fake one, $0)")
    return _load("dm_addon").backend(spec, a.live, spec.url, spec.model)


def cmd_dm(a: argparse.Namespace) -> int:
    need_frozen(a)
    spec = _spec(a)
    backend = _backend(a, spec)
    d = run_dir(a)
    qs = _questions(d)
    swarm = _swarm(d)
    opts = dm.load_options(d / "questions.jsonl")
    est = dm.estimate(qs, swarm, opts, spec)
    runner = dm.DMRunner(backend, d / f"cache_{spec.name}.jsonl", max_usd=a.max_usd or CAPS["dm"], price=spec.price)
    print(f"{spec.name}: {est['calls']} calls, about ${est['usd']:.4f} (run 1 billed about 1.75 times its estimate); "
          f"cap ${runner.max_usd:.3f}; already ${runner.spent:.4f} in the cache", flush=True)
    status = "complete"
    out: List[Dict[str, Any]] = []
    try:
        out = asyncio.run(runner.run(qs, swarm, spec.model, opts, concurrency=spec.concurrency))
    except ft.SpendCap as exc:
        status = f"stopped at the cap: {exc}"
    except dm.BrokenRun as exc:
        status = f"STOPPED, the model's replies are not usable: {exc}"
    if out:
        dm.write_jsonl(d / f"dm_{spec.name}.jsonl", out)
    print(f"{status}: {len(out)}/{len(qs)} questions, {runner.calls} new calls ({runner.failed} failed), "
          f"${runner.spent:.4f} spent in this cache")
    return 0 if status == "complete" else 2


# ------------------------------------------------------------------------- leak

def cmd_leak(a: argparse.Namespace) -> int:
    need_frozen(a)
    spec = _spec(a)
    backend = _backend(a, spec)
    d = run_dir(a)
    raws = read_jsonl(AGIEVAL / "exam_questions.jsonl")
    est = leak.estimate(raws, spec.model, spec.price)
    runner = dm.DMRunner(backend, d / f"cache_{spec.name}_leak.jsonl", max_usd=a.max_usd or CAPS["leak"],
                         price=spec.price)
    print(f"{spec.name} letter check on the {len(raws)} AGIEval questions: {est['calls']} calls, about "
          f"${est['usd']:.4f}; cap ${runner.max_usd:.3f}", flush=True)
    try:
        picks = leak.run(runner, raws, spec.model, concurrency=spec.concurrency)
    except (ft.SpendCap, dm.BrokenRun) as exc:
        print(f"STOPPED: {exc}")
        return 2
    dm.write_jsonl(d / f"leak_{spec.name}.jsonl",
                   [{"id": str(r["id"]), **{arm: picks[arm].get(str(r["id"])) for arm in leak.ARMS}} for r in raws])
    if a.dm1:
        src = Path(a.dm1).expanduser()
        if not src.exists():
            raise SystemExit(f"no {src}")
        dm.write_jsonl(d / f"leak_{spec.name}_dm1.jsonl", read_jsonl(src))
    print(f"done: {runner.calls} new calls ({runner.failed} failed), ${runner.spent:.4f} spent")
    return 0


# ------------------------------------------------------------------------ score

def _boot(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Any]:
    gain, lo, hi = ft.paired_boot([x - y for x, y in zip(xs, ys)])
    return {"gain": round(gain, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def lead_verdict(lead: Mapping[str, Any]) -> str:
    if lead["gain"] >= LEAD["holds"] and lead["ci95"][0] > 0:
        return "holds: Jev still beats the reader clearly on questions it can't have seen"
    if lead["ci95"][1] < LEAD["gone"]:
        return "gone: on new questions Jev is not clearly better than the reader, so run 1's lead was likely memory"
    return "unclear: Jev's lead shrank or is too uncertain to call"


def score_all(d: Path) -> Dict[str, Any]:
    key = {r["id"]: r["label"] for r in read_jsonl(d / "answers.jsonl")}
    rows = read_jsonl(d / "rows.jsonl")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    out: Dict[str, Any] = {"set": json.loads((d / "set.json").read_text()) if (d / "set.json").exists() else {}}
    if rows:
        s = sr.score(rows, key, meta["families"])
        out["swarm"] = {"n": s["n"], "usable_share": s["usable_share"], "reader_alone": s["all"]["reader_alone"],
                        "votes": {arm: s["all"]["arms"][arm]["vote"] for arm in sr.ARMS},
                        "average_ant": {arm: s["all"]["arms"][arm]["ant_mean"] for arm in sr.ARMS},
                        "notes_vs_alone": s["all"]["notes_vs_alone"], "verdict": s["verdict"],
                        "by_task": s["by_group"], "meta": meta}
        swarm = {r.id: r for r in dm.export_shared_read(rows)}
        dms = {n: {r["id"]: r for r in read_jsonl(d / f"dm_{n}.jsonl")} for n in ("jev", "d1-3b")
               if (d / f"dm_{n}.jsonl").exists()}
        if dms:
            res = dm.score("shared-read", list(swarm.values()), dms, key)
            ex = res["kinds"]["exam"]
            out["means"] = ex["means"]
            out["dms"] = {n: {k: v for k, v in info.items() if k.startswith(("vote_vs", "read_vs", "usable"))}
                          for n, info in ex["dms"].items()}
            ids = [r.id for r in swarm.values() if r.id in key]
            reader = [ft.score_item("exam", swarm[i].baselines.get("reader"), key[i]) for i in ids]
            for n, ans in dms.items():
                mine = [ft.score_item("exam", dm.dm_value("exam", ans.get(i, {}).get("dm1")), key[i]) for i in ids]
                lead = _boot(mine, reader)
                out["dms"][n]["lead_over_reader"] = lead
                if n == "jev":
                    out["verdict_jev_lead"] = lead_verdict(lead)
            by_task: Dict[str, Dict[str, float]] = {}
            for task in fq.TASKS:
                tids = [i for i in ids if swarm[i].group == task]
                if tids:
                    by_task[task] = {"n": len(tids), "reader": round(st.mean(
                        ft.score_item("exam", swarm[i].baselines.get("reader"), key[i]) for i in tids), 4)}
                    for n, ans in dms.items():
                        by_task[task][f"{n}:dm1"] = round(st.mean(ft.score_item(
                            "exam", dm.dm_value("exam", ans.get(i, {}).get("dm1")), key[i]) for i in tids), 4)
            out["dm_by_task"] = by_task
        out["cost"] = {n: _load("dm_addon").cost_of(d / f"cache_{n}.jsonl") for n in ("jev", "d1-3b")
                       if (d / f"cache_{n}.jsonl").exists()}
    picks_path = d / "leak_jev.jsonl"
    if picks_path.exists():
        prows = read_jsonl(picks_path)
        picks = {arm: {r["id"]: r.get(arm) for r in prows} for arm in leak.ARMS}
        akey = {r["id"]: r["label"] for r in read_jsonl(AGIEVAL / "exam_answers.jsonl")}
        dm1_rows = read_jsonl(d / "leak_jev_dm1.jsonl")
        dm1 = leak.read_dm1(dm1_rows) if dm1_rows else None
        out["leak"] = leak.score(read_jsonl(AGIEVAL / "exam_questions.jsonl"), picks, akey, dm1)
        out["leak"]["cost"] = _load("dm_addon").cost_of(d / "cache_jev_leak.jsonl")
    return out


def report(r: Mapping[str, Any]) -> str:
    L = ["FRESH TEST RESULT (paste this back to Claude in Duet)"]
    s = r.get("set") or {}
    if s:
        L.append(f"questions: {sum(s['counts'].values())} new ({', '.join(f'{k} {v}' for k, v in s['counts'].items())}), "
                 f"seed {s['seed']}")
    sw = r.get("swarm")
    if sw:
        L.append(f"swarm: usable {sw['usable_share']:.0%}; reader alone {sw['reader_alone']:.1%}; votes " +
                 ", ".join(f"{k} {v:.1%}" for k, v in sw["votes"].items()) +
                 f"; notes vs alone {sw['notes_vs_alone']['gain']:+.1%} {sw['notes_vs_alone']['ci95']}")
        L.append(f"  swarm verdict: {json.dumps(sw['verdict'])}")
        L.append("  by task (reader / alone / notes / both): " + "; ".join(
            f"{g} {v['reader_alone']:.0%}/{v['alone']:.0%}/{v['notes']:.0%}/{v['both']:.0%}"
            for g, v in sw["by_task"].items()))
        meta = sw.get("meta") or {}
        L.append(f"  run: {meta.get('new_calls')} new calls, {meta.get('failed_calls')} unusable, "
                 f"{(meta.get('seconds') or 0) / 60:.0f} min")
    if r.get("means"):
        L.append("arms on new questions (run 1 on AGIEval in brackets):")
        for arm, m in sorted(r["means"].items(), key=lambda kv: -kv[1]):
            old = RUN1.get(arm)
            L.append(f"  {arm:14s} {m:6.1%}" + (f"   [{old:.1%}]" if old is not None else ""))
        for n, info in r["dms"].items():
            lead = info["lead_over_reader"]
            L.append(f"  {n}: alone vs reader {lead['gain']:+.1%} {lead['ci95']}; " + "; ".join(
                f"{k.replace('_vs_', ' vs ')} {v['gain']:+.1%} {v['ci95']}" for k, v in info.items()
                if k.startswith(("vote_vs", "read_vs"))))
        L.append("  by task: " + "; ".join(f"{t} " + "/".join(f"{k} {v:.0%}" for k, v in row.items() if k != "n")
                                         for t, row in r["dm_by_task"].items()))
        if "verdict_jev_lead" in r:
            L.append(f"VERDICT, Jev's lead over the reader (run 1: {RUN1_LEAD:+.1%}): {r['verdict_jev_lead']}")
    for n, c in (r.get("cost") or {}).items():
        L.append(f"cost {n}: {c['calls']} calls ({c['failed']} failed), ${c['usd']:.4f}, median {c['median_s']}s")
    lk = r.get("leak")
    if lk:
        L.append(f"letter check on the 240 AGIEval questions: n {lk['n']}, chance {lk['chance']:.1%}; " +
                 ", ".join(f"{k} {v:.1%}" for k, v in lk["means"].items() if v is not None))
        if "relabel_drop" in lk:
            L.append(f"  moving the letters cost {lk['relabel_drop']['gain']:+.1%} {lk['relabel_drop']['ci95']}")
        if "blind_above_chance" in lk:
            L.append(f"  options alone, above chance: {lk['blind_above_chance']['gain']:+.1%} "
                     f"{lk['blind_above_chance']['ci95']}")
        L.append("  by group: " + "; ".join(f"{g} " + "/".join(f"{k} {v:.0%}" for k, v in row.items() if k != "n")
                                          for g, row in lk["by_group"].items()))
        L.append(f"  LETTER CHECK VERDICT: {lk['verdict']}; cost ${lk['cost']['usd']:.4f}")
    return "\n".join(L)


def cmd_score(a: argparse.Namespace) -> int:
    d = run_dir(a)
    ch = changed(a)
    r = score_all(d)
    r["frozen_plan_unchanged"] = not ch
    (d / "result.json").write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
    print(report(r))
    if ch:
        print(f"WARNING: the plan changed since it was frozen ({', '.join(ch)}): this is a new test")
    return 0


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "make", "swarm", "dm", "leak", "score"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=DATA)
        if name in ("make", "swarm", "dm", "leak", "score"):
            p.add_argument("--dry-run", action="store_true", help="fake models, $0, in a separate dry-run folder")
    sub.choices["make"].add_argument("--seed", type=int, default=None, help="default: drawn now, at random")
    sub.choices["make"].add_argument("--per-task", type=int, default=PER_TASK)
    sw = sub.choices["swarm"]
    sw.add_argument("--base-url", default=STANDIN)
    sw.add_argument("--models", type=Path, default=MODELS)
    sw.add_argument("--headroom-gb", type=float, default=6.0)
    sw.add_argument("--concurrency", type=int, default=2)
    sw.add_argument("--max-fail-streak", type=int, default=30)
    for name in ("dm", "leak"):
        p = sub.choices[name]
        p.add_argument("--dm", choices=("jev", "d1-3b"), default="jev")
        p.add_argument("--live", action="store_true")
        p.add_argument("--url", default="")
        p.add_argument("--model", default="")
        p.add_argument("--concurrency", type=int, default=0)
        p.add_argument("--max-usd", type=float, default=0.0, help="lower than the frozen cap only")
    sub.choices["leak"].add_argument("--dm1", default="", help="run 1's dm_jev_shared-read.jsonl, to compare with")
    a = ap.parse_args(argv)
    if getattr(a, "max_usd", 0) and a.max_usd > CAPS[a.cmd]:
        raise SystemExit(f"--max-usd can't go above the frozen cap ${CAPS[a.cmd]}")
    return {"freeze": cmd_freeze, "make": cmd_make, "swarm": cmd_swarm, "dm": cmd_dm, "leak": cmd_leak,
            "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
