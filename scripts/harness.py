#!/usr/bin/env python3
"""The harness, from the command line. See docs/plans/2026-11-02-harness-v0.md.

Commands (every one but ``sweep``, ``run`` and ``mac`` without ``--dry-run`` is free and offline)::

    uv run python scripts/harness.py selftest --out /tmp/h            # the whole loop with fake models, $0, seconds
    uv run python scripts/harness.py make-dev --out DIR --seed 7 --per-task 3     # new code-made dev puzzles
    uv run python scripts/harness.py import --set set1=RUN1 --set set2=RUN2 --out DATA   # saved work -> steps, $0
    uv run python scripts/harness.py learn --data DATA --train set1,set2 --test set3 --out OUT   # or --folds 5
    uv run python scripts/harness.py sweep --dev DIR --base-url URL --out DATA    # every tool on every question
    uv run python scripts/harness.py run --dev DIR --policy OUT/policy.json --base-url URL --out DATA
    uv run python scripts/harness.py mac --night NIGHT --mlx-server PATH --out OUT   # the Mac's one-shot smoke run

The test-set rule: the final test set is picked last and kept unseen. Everything here runs on code-made dev puzzles
and saved dev answers only, and every report says so.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import dm_addon as dm  # noqa: E402
from antelligence.experiments import fresh_questions as fq  # noqa: E402
from antelligence.experiments import shared_read as sr  # noqa: E402
from antelligence.providers.base import ChatRequest  # noqa: E402
from antelligence.harness import devsets  # noqa: E402
from antelligence.harness import learner as ln  # noqa: E402
from antelligence.harness import scorecard as sc  # noqa: E402
from antelligence.harness.core import Step, Task, bucket, load_key, load_tasks  # noqa: E402
from antelligence.harness.importer import import_sets, set_of  # noqa: E402
from antelligence.harness.planner import Planner, batch_order, sweep_batch, sweep_tools  # noqa: E402
from antelligence.harness.recipes import by_name, menu  # noqa: E402
from antelligence.harness.resilience import Patience  # noqa: E402
from antelligence.harness.tools import LiveBox, ReplayBox, load_steps, parse_ant, save_steps, trim_loads  # noqa: E402

MODELS = REPO / "docs/research/fresh-20261031/models.json"   # the models the saved sets were run with
FRESH_NOTE = ("STILL DEV NUMBERS: brand-new code-made puzzles, so nothing here was tuned on them, but they are not the "
              "final test set either.")
DEV_NOTE = "DEV NUMBERS ONLY: code-made puzzles and saved dev answers, not the final test. They prove nothing yet."
PORT, MLX_PORT = 11534, 8097          # not 11434/8091 (the night run's) and not 8090 (Eshaan's chat shim)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def task_row(t: Task) -> Dict[str, Any]:
    return {"id": t.id, "passage": t.passage, "question": t.stem, "options": list(t.options), "task": t.kind}


def save_bundle(d: Path, tasks: Sequence[Task], key: Mapping[str, str], steps, roster, meta) -> None:
    d.mkdir(parents=True, exist_ok=True)
    write_jsonl(d / "tasks.jsonl", [task_row(t) for t in tasks])
    write_jsonl(d / "key.jsonl", [{"id": i, "label": v} for i, v in key.items()])
    save_steps(d / "steps.json", steps, roster, meta)


def load_bundle(d: Path):
    tasks = load_tasks(d / "tasks.jsonl")
    key = load_key(d / "key.jsonl")
    steps, roster, meta = load_steps(d / "steps.json")
    return tasks, key, steps, roster, meta


# --------------------------------------------------------------------- make-dev

def make_dev(out: Path, seed: int, per_task: int) -> Tuple[List[Task], Dict[str, str]]:
    qs, ans, specs = devsets.make_set(seed, per_task)
    bad = [q["id"] for q, s in zip(qs, specs) if not fq.check(q, s)]
    if bad:
        raise SystemExit(f"puzzles failed their own brute-force check: {bad}")
    prefix = f"dev{seed}"
    for q in qs:
        q["id"] = f"{prefix}-{q['id']}"
    for a in ans:
        a["id"] = f"{prefix}-{a['id']}"
    write_jsonl(out / "questions.jsonl", qs)
    write_jsonl(out / "answers.jsonl", ans)
    write_jsonl(out / "specs.jsonl", specs)
    return [Task.from_raw(q) for q in qs], {a["id"]: a["label"] for a in ans}


def cmd_make_dev(a: argparse.Namespace) -> int:
    if (a.out / "questions.jsonl").exists():
        raise SystemExit(f"{a.out} already has questions; pick a new folder")
    tasks, _ = make_dev(a.out, a.seed, a.per_task)
    print(f"{len(tasks)} dev puzzles in {a.out} (answers kept apart in answers.jsonl)")
    return 0


# ----------------------------------------------------------------------- import

def cmd_import(a: argparse.Namespace) -> int:
    runs = {}
    for s in a.set:
        name, _, path = s.partition("=")
        if not path:
            raise SystemExit("--set takes NAME=PATH")
        runs[name] = Path(path)
    cfg = sr.load_config(a.models)
    tasks, key, steps, roster, reports = import_sets(runs, cfg)
    save_bundle(a.out, tasks, key, steps, roster, {"source": "import", "sets": reports})
    for r in reports:
        print(f"{r['set']}: {r['tasks']} questions, {r['steps']} steps, {r['priced']:.1%} priced from the caches, "
              f"{r['load_outliers']} load-time outliers, {r.get('reparsed', 0)} blank answers read from their "
              f"saved reply; decision models {r['decision_models']}")
    return 0


# ------------------------------------------------------------------------ learn

def weights(a: argparse.Namespace) -> ln.Weights:
    return ln.Weights(a.w_tokens, a.w_seconds, a.w_usd)


def recipes_for(box) -> list:
    dms = [t for t in ("dm:jev:dm1", "dm:jev:read", "dm:d1-3b:dm1", "dm:d1-3b:read") if box.has(t)]
    return [r for r in menu(box.roster, dms) if r.fits(box)]


def learn_report(tasks: Sequence[Task], key: Mapping[str, str], steps, roster, *, train: Sequence[str] = (),
                 test: Sequence[str] = (), folds: int = 5, w: ln.Weights = ln.Weights()) -> Dict[str, Any]:
    box = ReplayBox(steps, roster)
    recipes = recipes_for(box)
    table = ln.evaluate(recipes, box, tasks, key)
    names, keep = ln.common(table, tasks)
    if train and test:
        tr = [t for t in keep if set_of(t.id) in train]
        te = [t for t in keep if set_of(t.id) in test]
        policy, routed = ln.held_out(table, tr, te, weights=w)
        how = f"learned on {'+'.join(train)} ({len(tr)}), scored on {'+'.join(test)} ({len(te)})"
        scored = te
    else:
        policy = ln.learn(table, keep, names=names, weights=w)
        routed = ln.k_fold(table, keep, k=folds, weights=w)
        how = f"{folds}-fold on {len(keep)} questions (each routed by a policy that never saw it)"
        scored = keep
    ids = {t.id for t in scored}
    outs = [o for n in names for o in table[n].values() if o.task in ids] + routed
    summ = sc.summary(outs, baseline="reader")
    ref = {"tokens": summ["reader"]["tokens"], "seconds": summ["reader"]["seconds"]}
    vals = {n: round(sum(ln.value(o, ref, w) for o in table[n].values() if o.task in ids) / len(ids), 4)
            for n in names} if ids else {}
    kind_of = {t.id: t.kind for t in scored}
    by_kind: Dict[str, Dict[str, float]] = {}
    for k in sorted({t.kind for t in scored if t.kind}):
        for label, rows in (("reader", table["reader"].values()), ("harness", routed)):
            got = [o.right for o in rows if kind_of.get(o.task) == k and o.right is not None]
            by_kind.setdefault(k, {})[label] = round(sum(got) / len(got), 3) if got else None
    test_ids = {t.id for t in scored}
    cost_by_kind: Dict[str, Dict[str, List[float]]] = {}
    for k in sorted({t.kind for t in scored if t.kind}):
        for label, rows in (("reader", table["reader"].values()), ("harness", routed)):
            got = [o for o in rows if o.task in test_ids and kind_of.get(o.task) == k]
            if got:
                cost_by_kind.setdefault(k, {})[label] = [round(sum(o.tokens for o in got) / len(got)),
                                                         round(sum(o.seconds for o in got) / len(got), 1)]
    return {"how": how, "n": len(scored), "recipes": len(names), "policy": policy.to_json(), "summary": summ,
            "values": vals, "pareto": sc.pareto(summ), "by_kind": by_kind, "cost_by_kind": cost_by_kind,
            "buckets": {b: sum(1 for t in scored if bucket(t) == b) for b in sorted({bucket(t) for t in scored})}}


def cmd_learn(a: argparse.Namespace) -> int:
    tasks, key, steps, roster, _ = load_bundle(a.data)
    train = [s for s in (a.train or "").split(",") if s]
    test = [s for s in (a.test or "").split(",") if s]
    rep = learn_report(tasks, key, steps, roster, train=train, test=test, folds=a.folds, w=weights(a))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "policy.json").write_text(json.dumps(rep["policy"], indent=1, sort_keys=True) + "\n")
    (a.out / "learn.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(learn_lines(rep, "saved work"))
    return 0


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def learn_lines(rep: Mapping[str, Any], label: str) -> str:
    s = rep["summary"]
    h, r = s.get("harness"), s["reader"]
    c = (h or {}).get("vs_reader") or {}
    lines = [f"{label}: {rep['how']}; {rep['recipes']} recipes compared"]
    if h and c.get("n"):
        lo, hi = c["accuracy_ci95"]
        lines.append(f"  harness {pct(h['accuracy'])} vs reader {pct(r['accuracy'])}: {c['accuracy_gain'] * 100:+.1f} "
                     f"points (range {lo * 100:+.1f} to {hi * 100:+.1f}); tokens x{c['tokens_ratio']}{ci(c, 'tokens')}, time "
                     f"x{c['seconds_ratio']}{ci(c, 'seconds')}, size-weighted tokens x{c['weighted_tokens_ratio']}; "
                     f"VERDICT {c['verdict']}")
    pol = rep["policy"]
    lines.append(f"  policy: default {pol['default']}; by kind of request {json.dumps(pol['buckets'], sort_keys=True)}")
    best = sorted(rep["values"].items(), key=lambda kv: -kv[1])[:5]
    lines.append("  top recipes by value: " + "; ".join(
        f"{n} {pct(s[n]['accuracy'])}, {s[n]['tokens']:.0f} tok, {s[n]['seconds']:.1f}s" for n, _ in best))
    lines.append(f"  on the accuracy/tokens/time frontier: {', '.join(rep['pareto'][:8])}")
    if rep.get("by_kind"):
        lines.append("  by dev kind (reader vs harness): " + "; ".join(
            f"{k} " + " vs ".join(pct(v) for v in d.values()) for k, d in rep["by_kind"].items()))
    if rep.get("cost_by_kind"):
        lines.append("  cost by dev kind (reader vs harness, tokens/seconds): " + "; ".join(
            f"{k} " + " vs ".join(f"{v[0]}/{v[1]}s" for v in d.values()) for k, d in rep["cost_by_kind"].items()))
    return "\n".join(lines)


def ci(c: Mapping[str, Any], what: str) -> str:
    r = c.get(f"{what}_ratio_ci95")
    return f" (range x{r[0]} to x{r[1]})" if r else ""


def tuned_report(tasks: Sequence[Task], key: Mapping[str, str], steps, roster, train: Sequence[str],
                 test: Sequence[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Learn again with the cost weight picked by :func:`learner.tune` on the training sets only, then score the
    test set once with it."""
    box = ReplayBox(steps, roster)
    table = ln.evaluate(recipes_for(box), box, tasks, key)
    _, keep = ln.common(table, tasks)
    w, grid = ln.tune(table, [t for t in keep if set_of(t.id) in train])
    rep = learn_report(tasks, key, steps, roster, train=train, test=test, w=w)
    rep["tuning"] = {"picked": w.tokens, "grid": grid}
    return rep, grid


def tuned_lines(rep: Mapping[str, Any], train: Sequence[str]) -> str:
    grid = rep["tuning"]["grid"]
    def row(r: Mapping[str, Any]) -> str:
        g = r.get("gain_ci95") or [0, 0]
        return (f"{r['weight']}: {pct(r['accuracy'])} ({(r['gain'] or 0) * 100:+.1f}, range {g[0] * 100:+.1f} to "
                f"{g[1] * 100:+.1f}), tokens x{r['tokens_ratio']}{ci(r, 'tokens')}, time x{r['seconds_ratio']}"
                f"{ci(r, 'seconds')}")
    why = next((r["why"] for r in grid if r.get("picked")), None)
    lines = [f"cost-tuned: the cost weight was picked by 5-fold on {'+'.join(train)} only (the test set unseen); "
             f"picked {rep['tuning']['picked']}" + (f" ({why})" if why else ""),
             "  tried on the training sets (points vs reader):"] + [f"    {row(r)}" for r in grid]
    full = learn_lines(rep, "cost-tuned").splitlines()
    lines += [x for x in full[1:3]]          # the harness-vs-reader line and the policy
    lines += [x for x in full if x.startswith(("  by dev kind", "  cost by dev kind"))]
    return "\n".join(lines)


# ------------------------------------------------------------------- live work

def config(path: Path) -> sr.Config:
    return sr.load_config(path)


def live_box(a: argparse.Namespace, out: Path, provider: Any, cfg: sr.Config, patience: Patience) -> LiveBox:
    runner = sr.Runner(provider, out / "cache.jsonl", concurrency=a.concurrency, check_names=False,
                       max_fail_streak=10 ** 6)
    return LiveBox(runner, cfg.reader, cfg.ants, parser=cfg.parser, code=a.code, patience=patience)


def merge_steps(path: Path, new: Mapping[str, Mapping[str, Step]], roster) -> Dict[str, Dict[str, Step]]:
    old: Dict[str, Dict[str, Step]] = {}
    if path.exists():
        old, _, _ = load_steps(path)
    for tid, per in new.items():
        old.setdefault(tid, {}).update(per)
    return old


def _root(base_url: str) -> str:
    return base_url.rstrip("/")[: -len("/v1")] if base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/")


def unloader(base_url: str, box: LiveBox):
    async def between(group: List[str]) -> None:
        tag = box.reader.tag if group[0] in ("reader", "notes", "code") else (
            box.roster[parse_ant(group[0])[0]].tag if parse_ant(group[0]) else "")
        if not tag:
            return
        import httpx

        try:
            await asyncio.to_thread(httpx.post, f"{_root(base_url)}/api/generate", json={"model": tag, "keep_alive": 0},
                                    timeout=120, trust_env=False)
        except httpx.HTTPError:
            pass
    return between


async def do_sweep(box: LiveBox, tasks: Sequence[Task], base_url: Optional[str]) -> Tuple[Dict[str, Dict[str, Step]], float]:
    t0 = time.monotonic()
    groups = batch_order(box)
    done = [0]

    def on_group(g: List[str]) -> None:
        done[0] += 1
        print(f"[harness] tool group {done[0]}/{len(groups)}: {', '.join(g[:3])}{' ...' if len(g) > 3 else ''} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)

    steps = await sweep_batch(box, tasks, between=unloader(base_url, box) if base_url else None, on_group=on_group)
    return steps, time.monotonic() - t0


def cmd_sweep(a: argparse.Namespace) -> int:
    tasks = load_tasks(a.dev / "questions.jsonl")
    key = load_key(a.dev / "answers.jsonl")
    cfg = config(a.models)
    provider = fake_provider(tasks, key, cfg) if a.dry_run else openai_provider(a)
    box = live_box(a, a.out, provider, cfg, Patience(drive=a.drive))
    raw, secs = asyncio.run(do_sweep(box, tasks, None if a.dry_run else a.base_url))
    steps, loads = trim_loads(raw)
    merged = merge_steps(a.out / "steps.json", steps, box.roster)
    old = load_tasks(a.out / "tasks.jsonl") if (a.out / "tasks.jsonl").exists() else []
    all_tasks = {t.id: t for t in [*old, *tasks]}
    old_key = load_key(a.out / "key.jsonl") if (a.out / "key.jsonl").exists() else {}
    save_bundle(a.out, list(all_tasks.values()), {**old_key, **key}, merged, box.roster, {"source": "sweep"})
    print(f"swept {len(tasks)} questions with {len(sweep_tools(box))} tools in {secs:.0f}s; "
          f"{box.runner.calls} new calls ({box.runner.failed} unusable); {loads} load-time outliers counted at the median")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    tasks = load_tasks(a.dev / "questions.jsonl")
    key = load_key(a.dev / "answers.jsonl") if (a.dev / "answers.jsonl").exists() else {}
    cfg = config(a.models)
    provider = fake_provider(tasks, key, cfg) if a.dry_run else openai_provider(a)
    box = live_box(a, a.out, provider, cfg, Patience(drive=a.drive))
    policy = ln.Policy.from_json(json.loads(a.policy.read_text()))
    planner = Planner(policy, by_name(recipes_for(box)), box, explore=a.explore, log=sc.Log(a.out / "outcomes.jsonl"))
    outs = asyncio.run(planner.answer_all(tasks, key))
    if planner.swept:
        save_steps(a.out / "steps.json", merge_steps(a.out / "steps.json", planner.swept, box.roster), box.roster,
                   {"source": "run sweeps"})
    t = sc.totals(outs)
    print(f"answered {t['n']}: accuracy {pct(t['accuracy'])}, {t['tokens']:.0f} tokens and {t['seconds']:.1f}s a "
          f"question on average; {len(planner.swept)} swept to learn from")
    return 0


def openai_provider(a: argparse.Namespace) -> Any:
    from antelligence.providers.openai_compat import OpenAICompatProvider

    # require_stop=False: a reply cut off at the length limit comes back with its text (finish reason "length")
    # instead of as an error with nothing in it, so the harness can see it, bill it and re-ask for the letter
    return OpenAICompatProvider(a.base_url, timeout_s=a.timeout, max_concurrency=a.concurrency,
                                require_model_match=False, require_stop=False)


def fake_provider(tasks: Sequence[Task], key: Mapping[str, str], cfg: sr.Config) -> Any:
    from antelligence.harness.fakes import FakeProvider

    skill = {cfg.reader.tag: 0.78, **{s.tag: 0.45 + 0.05 * i for i, s in enumerate(cfg.ants)}}
    size = {cfg.reader.tag: cfg.reader.gb, **{s.tag: s.gb for s in cfg.ants}}
    return FakeProvider(tasks, key, skill, size)


# --------------------------------------------------------------------- selftest

def selftest(out: Path, per_task: int = 12, seed: int = 11) -> Dict[str, Any]:
    """Make dev puzzles, sweep them with fake models (real sandbox for the code checker), save them in the night
    run's own format, import that back (the bills must all match), learn, and run the planner live. $0."""
    cfg = sr.load_config(MODELS)
    dev = out / "dev"
    tasks, key = make_dev(dev, seed, per_task)
    run = out / "set1" / "run"
    run.mkdir(parents=True, exist_ok=True)
    provider = fake_provider(tasks, key, cfg)
    runner = sr.Runner(provider, run / "cache.jsonl", concurrency=4, check_names=False, max_fail_streak=10 ** 6)
    box = LiveBox(runner, cfg.reader, cfg.ants, parser=cfg.parser, code=True)
    steps, _ = asyncio.run(do_sweep(box, tasks, None))
    # write the sweep the way the shared-read test saves a set, then read it back through the importer
    write_jsonl(run / "questions.jsonl", [task_row(t) for t in tasks])
    write_jsonl(run / "answers.jsonl", [{"id": i, "label": v} for i, v in key.items()])
    n = len(box.roster)
    write_jsonl(run / "rows.jsonl", [{
        "id": t.id, "group": t.kind, "reader": steps[t.id]["reader"].answer,
        "reader_notes": steps[t.id]["notes"].payload["notes"], "leak": steps[t.id]["notes"].payload["leak"],
        **{arm: [steps[t.id][f"ant{i}:{arm}"].answer for i in range(n)] for arm in ("alone", "notes", "both")}}
        for t in tasks])
    itasks, ikey, isteps, iroster, reports = import_sets({"set1": run}, cfg)
    rep = learn_report(itasks, ikey, isteps, iroster, folds=4)
    # the code checker isn't in a saved set; learn with it from the live sweep too
    live = learn_report(tasks, key, steps, box.roster, folds=4)
    policy = ln.Policy.from_json(live["policy"])
    fresh, fkey = make_dev(out / "dev2", seed + 1, 2)
    p2 = fake_provider(fresh, fkey, cfg)
    box2 = LiveBox(sr.Runner(p2, out / "cache2.jsonl", concurrency=4, check_names=False, max_fail_streak=10 ** 6),
                   cfg.reader, cfg.ants, parser=cfg.parser, code=True)
    planner = Planner(policy, by_name(recipes_for(box2)), box2, explore=0.25, log=sc.Log(out / "outcomes.jsonl"))
    outs = asyncio.run(planner.answer_all(fresh, fkey))
    return {"import": reports[0], "saved": rep, "live": live, "planner": sc.totals(outs),
            "planner_swept": len(planner.swept)}


def cmd_selftest(a: argparse.Namespace) -> int:
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty")
    r = selftest(a.out)
    imp = r["import"]
    print("HARNESS SELFTEST (fake models, $0)")
    print(f"import: {imp['tasks']} questions, {imp['steps']} steps, {imp['priced']:.0%} priced")
    print(learn_lines(r["saved"], "saved-format sweep"))
    print(learn_lines(r["live"], "live sweep with the code checker"))
    p = r["planner"]
    print(f"planner on new puzzles: {p['n']} answered, accuracy {pct(p['accuracy'])}, {p['tokens']:.0f} tokens a "
          f"question, {r['planner_swept']} swept to learn from")
    ok = imp["priced"] == 1.0 and p["n"] > 0
    print("SELFTEST " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


# -------------------------------------------------------------------------- mac

def _night_module():
    spec = importlib.util.spec_from_file_location("_night_run_for_harness", REPO / "scripts/night_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def night_sets(night: Path) -> Dict[str, Path]:
    return {n: night / n / "run" for n in ("set1", "set2", "set3") if (night / n / "run" / "rows.jsonl").exists()}


def cmd_mac(a: argparse.Namespace) -> int:
    """1. Import the night run's saved sets (free) and learn on two, score on the third.
    2. Start the MLX stand-in on its own ports, sweep new dev puzzles with every tool (free, local).
    3. Learn on that sweep (k folds) and run the planner live on a few more new puzzles.
    4. Print one block to paste back."""
    a.out.mkdir(parents=True, exist_ok=True)
    blocks: List[str] = ["HARNESS RESULT (paste this back to Claude in Duet)"]
    sets = night_sets(a.night) if a.night else {}
    if sets:
        cfg = sr.load_config(a.models)
        tasks, key, steps, roster, reports = import_sets(sets, cfg)
        save_bundle(a.out / "saved", tasks, key, steps, roster, {"source": "import", "sets": reports})
        blocks.append("import: " + "; ".join(f"{r['set']} {r['tasks']}q {r['priced']:.1%} priced "
                                             f"{r['load_outliers']} loads, {r.get('reparsed', 0)} blank answers read from their saved "
                                             f"reply, {r.get('cut_off_billed', 0)} cut-off replies billed at their "
                                             f"limit" for r in reports))
        names = sorted(sets)
        train, test = (names[:-1], names[-1:]) if len(names) > 1 else ((), ())
        rep = learn_report(tasks, key, steps, roster, train=train, test=test, folds=5)
        (a.out / "saved" / "learn.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        (a.out / "saved" / "policy.json").write_text(json.dumps(rep["policy"], indent=1, sort_keys=True) + "\n")
        blocks.append(learn_lines(rep, "saved night-run answers"))
        if train and test:
            trep, _ = tuned_report(tasks, key, steps, roster, train, test)
            (a.out / "saved" / "learn-tuned.json").write_text(json.dumps(trep, indent=1, sort_keys=True) + "\n")
            (a.out / "saved" / "policy-tuned.json").write_text(json.dumps(trep["policy"], indent=1, sort_keys=True)
                                                               + "\n")
            blocks.append(tuned_lines(trep, train))
    else:
        blocks.append("import: no saved sets found (skipped)")
    if a.smoke:
        blocks += smoke(a)
    blocks.append(DEV_NOTE)
    text = "\n".join(blocks)
    (a.out / "RESULT.txt").write_text(text + "\n")
    print("\n" + text)
    return 0


def smoke(a: argparse.Namespace) -> List[str]:
    cfg = config(a.models)
    seed = a.seed or random.SystemRandom().randrange(1, 10 ** 9)
    dev = a.out / "smoke" / "dev"
    if (dev / "questions.jsonl").exists():
        tasks, key = load_tasks(dev / "questions.jsonl"), load_key(dev / "answers.jsonl")
    else:
        tasks, key = make_dev(dev, seed, a.smoke)
    later = a.out / "smoke" / "dev-planner"
    if (later / "questions.jsonl").exists():
        ptasks, pkey = load_tasks(later / "questions.jsonl"), load_key(later / "answers.jsonl")
    else:
        ptasks, pkey = make_dev(later, seed + 1, 1)
    service = None
    patience = Patience(drive=a.drive)
    if a.dry_run:
        provider: Any = fake_provider(list(tasks) + list(ptasks), {**key, **pkey}, cfg)
        base = None
    else:
        nr = _night_module()
        for port in (a.port, a.mlx_port):
            if not nr.port_free(port):
                raise SystemExit(f"something already listens on port {port}; pick others with --port/--mlx-port")
        logs = a.out / "smoke" / "logs"
        service = nr.Service("MLX stand-in", [sys.executable, str(REPO / "scripts/mlx_standin.py"), "--models",
                                              str(a.models), "--mlx-server", a.mlx_server, "--port", str(a.port),
                                              "--mlx-port", str(a.mlx_port), "--log", str(logs / "mlx.log")],
                             f"http://127.0.0.1:{a.port}/v1/models", logs / "standin.log")
        service.up(seconds=300)
        base = f"http://127.0.0.1:{a.port}/v1"
        a.base_url = base
        provider = openai_provider(a)
    out = a.out / "smoke"
    lines: List[str] = []
    try:
        box = live_box(a, out, provider, cfg, patience)
        raw, secs = asyncio.run(do_sweep(box, tasks, base))
        steps, loads = trim_loads(raw)
        save_bundle(out, tasks, key, steps, box.roster, {"source": "smoke sweep", "seed": seed, "load_outliers": loads})
        code = [s["code"] for s in steps.values() if "code" in s]
        settled = [c for c, t in zip(code, tasks) if c.answer is not None]
        right = sum(1 for t in tasks if "code" in steps[t.id] and steps[t.id]["code"].answer == key[t.id])
        calls = sum(s.calls for per in steps.values() for s in per.values())
        asked, fixed = reask_counts(steps)
        checker = (f"code checker settled {len(settled)}/{len(code)}, right on {right}" if box.code_on
                   else "code checker benched (--code turns it on)")
        lines.append(f"smoke sweep: {len(tasks)} new dev puzzles (seed {seed}), {len(sweep_tools(box))} tools, "
                     f"{calls} calls in {secs / 60:.1f} min wall; {loads} load-time outliers counted at the median; "
                     f"{box.runner.failed} unusable replies; {fixed}/{asked} blank answers fixed by a short re-ask; "
                     f"{checker}")
        rep = learn_report(tasks, key, steps, box.roster, folds=min(4, max(2, len(tasks) // 4)))
        (out / "learn.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        lines.append(learn_lines(rep, "smoke (tiny, k-fold)"))
        policy = ln.Policy.from_json(rep["policy"])
        t0 = time.monotonic()
        planner = Planner(policy, by_name(recipes_for(box)), box, explore=0.0, log=sc.Log(out / "outcomes.jsonl"))
        outs = asyncio.run(planner.answer_all(ptasks, pkey))
        t = sc.totals(outs)
        lines.append(f"planner live on {t['n']} more new puzzles: accuracy {pct(t['accuracy'])}, {t['tokens']:.0f} "
                     f"tokens and {t['seconds']:.1f}s a question (model loads included, {time.monotonic() - t0:.0f}s wall); "
                     f"recipes used "
                     f"{sorted({o.recipe for o in outs})}")
    finally:
        if service:
            service.down()
    lines.append(f"waits for a stalled drive or server: {len(patience.events)} ({patience.waited_s:.0f}s)")
    return lines


# ------------------------------------------------------------------ fresh check

def _dm_addon_module():
    spec = importlib.util.spec_from_file_location("_dm_addon_for_harness", REPO / "scripts/dm_addon.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _syllogism_set(path: Path, seed: int, per_kind: int, label: str, prefix: str, maker) -> None:
    if (path / "questions.jsonl").exists():
        return
    qs, ans, specs = devsets.make_set(seed, per_kind, tasks=("syllogism",), makers={"syllogism": maker})
    bad = [q["id"] for q, sp in zip(qs, specs) if not devsets.check_syllogism(q, sp)]
    if bad:
        raise SystemExit(f"puzzles failed their own brute-force check: {bad}")
    for q, a_ in zip(qs, ans):
        q["id"] = a_["id"] = f"{prefix}{seed}-{q['id']}"
        q["task"] = label
    write_jsonl(path / "questions.jsonl", qs)
    write_jsonl(path / "answers.jsonl", ans)
    write_jsonl(path / "specs.jsonl", specs)


def fresh_sets(out: Path, seed: int, per_kind: int) -> Tuple[List[Task], Dict[str, str], List[Task], Dict[str, str]]:
    """New puzzles nobody has seen: every kind with the balanced syllogism maker, plus two extra syllogism sets so
    the makers can be compared on the same day with the same models: ``syllogism-old`` from the fresh-question
    maker (the one a no-logic lookup gets 61% on) and ``syllogism-reworded``, balanced and written in other words
    (other phrasings, opening line and question). Saved under ``out``; a rerun reuses them."""
    new = out / "fresh"
    if not (new / "questions.jsonl").exists():
        make_dev(new, seed, per_kind)
    _syllogism_set(out / "fresh-old-syllogisms", seed + 1, per_kind, "syllogism-old", "old", fq.make_syllogism)
    _syllogism_set(out / "fresh-reworded-syllogisms", seed + 2, per_kind, "syllogism-reworded", "rew",
                   devsets.make_reworded_syllogism)
    extra: List[Task] = []
    ekey: Dict[str, str] = {}
    for d in ("fresh-old-syllogisms", "fresh-reworded-syllogisms"):
        extra += load_tasks(out / d / "questions.jsonl")
        ekey.update(load_key(out / d / "answers.jsonl"))
    return load_tasks(new / "questions.jsonl"), load_key(new / "answers.jsonl"), extra, ekey


def jev_dms(a: argparse.Namespace, out: Path) -> Dict[str, Any]:
    """Jev for the fresh check, with its spend capped at ``--jev-cap`` dollars. The key is read in-process by the
    decision-model add-on's own reader and never printed. ``--dry-run`` uses the $0 fake."""
    da = _dm_addon_module()
    spec = dm.DMS["jev"]
    backend = dm.FakeDM() if a.dry_run else da.backend(spec, True, spec.url, spec.model)
    return {"jev": (spec, dm.DMRunner(backend, out / "dm_jev.jsonl", max_usd=a.jev_cap, price=spec.price))}


def fresh_lines(outs: Sequence[sc.Outcome], reader: Sequence[sc.Outcome], kinds: Mapping[str, str],
                label: str) -> List[str]:
    c = sc.compare(outs, reader)
    if not c.get("n"):
        return [f"{label}: nothing scored"]
    h, r = sc.totals(outs), sc.totals(reader)
    lo, hi = c["accuracy_ci95"]
    lines = [f"{label}: harness {pct(h['accuracy'])} vs reader {pct(r['accuracy'])} on {c['n']} new puzzles: "
             f"{c['accuracy_gain'] * 100:+.1f} points (range {lo * 100:+.1f} to {hi * 100:+.1f}); tokens "
             f"x{c['tokens_ratio']}{ci(c, 'tokens')}, time x{c['seconds_ratio']}{ci(c, 'seconds')}; VERDICT "
             f"{c['verdict']}"]
    by = []
    for k in sorted(set(kinds.values())):
        hs = [o for o in outs if kinds.get(o.task) == k]
        rs = [o for o in reader if kinds.get(o.task) == k]
        if hs and rs:
            th, tr = sc.totals(hs), sc.totals(rs)
            by.append(f"{k} {pct(tr['accuracy'])} vs {pct(th['accuracy'])} ({tr['tokens']:.0f}/{tr['seconds']:.1f}s "
                      f"vs {th['tokens']:.0f}/{th['seconds']:.1f}s)")
    lines.append("  by kind (reader vs harness, then tokens/seconds): " + "; ".join(by))
    lines.append("  recipes used: " + ", ".join(f"{n} x{k}" for n, k in sorted(Counter(o.recipe for o in outs).items())))
    return lines


def cmd_fresh(a: argparse.Namespace) -> int:
    """The frozen policy (learned before these puzzles existed) against the reader on brand-new puzzles."""
    a.out.mkdir(parents=True, exist_ok=True)
    seed = a.seed or random.SystemRandom().randrange(1, 10 ** 9)
    tasks, key, old, okey = fresh_sets(a.out, seed, a.per_kind)
    policy = ln.Policy.from_json(json.loads(a.policy.read_text()))
    cfg = config(a.models)
    patience = Patience(drive=a.drive)
    service = None
    if a.dry_run:
        provider: Any = fake_provider(list(tasks) + list(old), {**key, **okey}, cfg)
    else:
        nr = _night_module()
        for port in (a.port, a.mlx_port):
            if not nr.port_free(port):
                raise SystemExit(f"something already listens on port {port}; pick others with --port/--mlx-port")
        logs = a.out / "logs"
        service = nr.Service("MLX stand-in", [sys.executable, str(REPO / "scripts/mlx_standin.py"), "--models",
                                              str(a.models), "--mlx-server", a.mlx_server, "--port", str(a.port),
                                              "--mlx-port", str(a.mlx_port), "--log", str(logs / "mlx.log")],
                             f"http://127.0.0.1:{a.port}/v1/models", logs / "standin.log")
        service.up(seconds=300)
        a.base_url = f"http://127.0.0.1:{a.port}/v1"
        provider = openai_provider(a)
    blocks = ["FRESH CHECK (paste this back to Claude in Duet)",
              f"policy frozen from {a.policy.name}: default {policy.default}; by kind of request "
              f"{json.dumps(policy.buckets, sort_keys=True)}"]
    t0 = time.monotonic()
    try:
        runner = sr.Runner(provider, a.out / "cache.jsonl", concurrency=a.concurrency, check_names=False,
                           max_fail_streak=10 ** 6)
        dms = jev_dms(a, a.out)
        box = LiveBox(runner, cfg.reader, cfg.ants, parser=cfg.parser, code=False, dms=dms, patience=patience)
        recipes = by_name(recipes_for(box))
        missing = sorted({policy.default, *policy.buckets.values()} - set(recipes))
        if missing:
            raise SystemExit(f"the policy needs recipes this box can't run: {missing}")
        log = sc.Log(a.out / "outcomes.jsonl")
        reader_plan = Planner(ln.Policy("reader"), recipes, box, explore=0.0)
        planner = Planner(policy, recipes, box, explore=0.0, log=log)

        async def both(ts: Sequence[Task], k: Mapping[str, str]) -> Tuple[List[sc.Outcome], List[sc.Outcome]]:
            # the reader first, so when the policy also picks the reader its call comes from the cache: same answer,
            # same bill, both sides charged for it once each
            rs = await reader_plan.answer_all(ts, k)
            hs = await planner.answer_all(ts, k)
            return hs, [sc.Outcome(**{**o.to_json(), "recipe": "reader"}) for o in rs]

        hs, rs = asyncio.run(both(tasks, key))
        ohs, ors = asyncio.run(both(old, okey))
    finally:
        if service:
            service.down()
    kinds = {t.id: t.kind for t in list(tasks) + list(old)}
    blocks += fresh_lines(hs, rs, kinds, f"new puzzles, every kind (seed {seed}, {a.per_kind} per kind)")
    for label, what in (("syllogism-old", "old-maker syllogisms (the kind a no-logic lookup gets 61% on)"),
                        ("syllogism-reworded", "reworded syllogisms (balanced, and written in other words)")):
        keep = {t.id for t in old if t.kind == label}
        blocks += fresh_lines([o for o in ohs if o.task in keep], [o for o in ors if o.task in keep], kinds, what)
    spent = sum(r.spent for _, r in dms.values())
    blocks.append(f"Jev: {sum(r.calls for _, r in dms.values())} calls, ${spent:.4f} of a ${a.jev_cap:.2f} cap; "
                  f"unusable model replies {runner.failed}; {time.monotonic() - t0:.0f}s wall; waits for a stalled "
                  f"drive or server {len(patience.events)}")
    blocks.append(FRESH_NOTE)
    text = "\n".join(blocks)
    (a.out / "FRESH.txt").write_text(text + "\n")
    print("\n" + text)
    return 0


# --------------------------------------------------------------------- diagnose

class _SavedOnly:
    """A provider for replays: a call the run saved as failed fails the same way again; any other call that isn't
    in the saved log fails and is counted in ``missing``."""

    def __init__(self, rows: Sequence[Mapping[str, Any]] = ()) -> None:
        self.failed = {r["request_hash"]: r for r in rows if r.get("error") and r.get("request_hash")}
        self.missing = 0

    async def complete(self, request: ChatRequest) -> Any:
        from antelligence.providers.base import ProviderError, ProviderTruncated

        row = self.failed.get(request.request_hash)
        if row is None:
            self.missing += 1
            raise ProviderError("not in the saved calls")
        if row["error"] == "ProviderTruncated":
            raise ProviderTruncated(row.get("reason") or "generation did not finish normally (length)")
        raise ProviderError(row.get("reason") or row["error"])

    def describe(self) -> Dict[str, Any]:
        return {"provider": "saved calls only"}


def _tail(text: str, n: int = 110) -> str:
    t = " ".join(sr.strip_think(text or "").split()) or " ".join((text or "").split())
    return (t[-n:] if len(t) > n else t) or "(empty)"


class _Recording(sr.Runner):
    """A runner that notes every call it is asked for, so a replay knows which saved calls belong to this run."""

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.asked: List[str] = []

    async def ask(self, req: ChatRequest, usable: Any) -> Dict[str, Any]:
        self.asked.append(req.request_hash)
        return await super().ask(req, usable)


def reask_counts(steps: Mapping[str, Mapping[str, Step]]) -> Tuple[int, int]:
    """(re-asks, fixed): answer steps that needed a short re-ask for the letter, and how many it fixed."""
    notes = [s.note or "" for per in steps.values() for s in per.values()]
    return (sum("re-ask" in n for n in notes), sum("fixed by a short re-ask" in n for n in notes))


def diagnose(run: Path, models: Path) -> List[str]:
    """Explain a smoke run from its saved files, with no model calls. It replays the run's own sweep against its
    saved call log (so calls left over from an earlier run with other prompts don't count), then says which
    replies failed and why, which tools came back blank, and what each code-checker try ran into."""
    d = run / "smoke"
    tasks, key, steps, roster, meta = load_bundle(d)
    cfg = sr.load_config(models)
    every = [json.loads(line) for line in (d / "cache.jsonl").read_text().splitlines() if line.strip()]
    latest: Dict[str, Dict[str, Any]] = {}       # one row per call: a usable reply wins, else the newest try
    for r in every:
        h = r.get("request_hash") or str(id(r))
        if not r.get("error") or h not in latest or latest[h].get("error"):
            latest[h] = r

    tools = sorted({t for per in steps.values() for t in per})
    tmp = Path(tempfile.mkdtemp(prefix="harness-diag-")) / "cache.jsonl"
    shutil.copyfile(d / "cache.jsonl", tmp)
    saved = _SavedOnly(every)
    runner = _Recording(saved, tmp, check_names=False, max_fail_streak=10 ** 6)
    box = LiveBox(runner, cfg.reader, cfg.ants, parser=cfg.parser, code="code" in tools,
                  patience=Patience(waits=(), log=lambda _m: None))
    try:
        replay = asyncio.run(sweep_batch(box, tasks, [t for t in tools if box.has(t)]))
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    asked = list(dict.fromkeys(runner.asked))
    rows = [latest[h] for h in asked if h in latest]
    same = sum(1 for tid, per in steps.items() for t, st in per.items()
               if t in replay.get(tid, {}) and replay[tid][t].answer == st.answer)
    total = sum(len(per) for per in steps.values())
    lines = ["HARNESS DIAGNOSE (paste this back to Claude in Duet)",
             f"this run's calls: {len(rows)} (of {len(latest)} in the saved log; the rest are from earlier runs), "
             f"failed: {sum(1 for r in rows if r.get('error'))}; replay matched {same}/{total} saved answers"
             + (f"; {saved.missing} calls missing from the log" if saved.missing else "")]

    per_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        per_model.setdefault(r.get("model") or "?", []).append(r)
    lines.append("failed replies by model:")
    for model, rs in sorted(per_model.items()):
        bad = [r for r in rs if r.get("error")]
        if not bad:
            lines.append(f"  {model}: 0 of {len(rs)}")
            continue
        kinds = Counter(f"{r['error']} (stop {r.get('finish_reason') or '-'})" for r in bad)
        toks = [int(r.get("completion_tokens") or 0) for r in bad if r.get("completion_tokens")]
        lines.append(f"  {model}: {len(bad)} of {len(rs)}; " + ", ".join(f"{k} x{n}" for k, n in kinds.most_common())
                     + (f"; reply length up to {max(toks)} tokens" if toks else ""))
        for r in bad[:2]:
            lines.append(f"    e.g. ...{_tail(r.get('content') or r.get('reason') or '')}")

    blank = Counter(t for per in steps.values() for t, s in per.items() if s.answer is None and t not in ("code", "notes"))
    lines.append("blank answers by tool (of %d puzzles): " % len(tasks)
                 + (", ".join(f"{t} {n}" for t, n in sorted(blank.items())) or "none"))
    by_kind: Dict[str, List[int]] = {}
    for t in tasks:
        got = [s for name, s in steps.get(t.id, {}).items() if name not in ("code", "notes")]
        by_kind.setdefault(t.kind, [0, 0])
        by_kind[t.kind][0] += sum(1 for s in got if s.answer is None)
        by_kind[t.kind][1] += len(got)
    lines.append("blank answers by kind: " + "; ".join(f"{k} {b}/{n}" for k, (b, n) in sorted(by_kind.items())))
    asked_n, fixed_n = reask_counts(steps)
    lines.append(f"short re-asks for the letter: {asked_n}, fixed {fixed_n}, still blank {asked_n - fixed_n}")

    if "code" in tools:
        lines.append("code checker, try by try (replayed from the saved programs):")
        for t in tasks:
            if "code" not in steps.get(t.id, {}):
                continue
            step = replay.get(t.id, {}).get("code")
            mark = "right" if step and step.answer == key.get(t.id) else (
                "wrong" if step and step.answer else "unsettled")
            tries = " | ".join(p[:160] for p in box.code_trace.get(t.id, []))
            lines.append(f"  {t.kind} (answer {key.get(t.id)}): {mark}; {tries}")
    else:
        lines.append("code checker: off in this run")
    tok: Dict[str, List[int]] = {}
    for per in steps.values():
        for name, st in per.items():
            tok.setdefault(name.split(":")[0] if name.startswith("ant") else name, []).append(st.tokens)
    lines.append("average tokens a call by tool: " + ", ".join(
        f"{t} {sum(v) / len(v):.0f}" for t, v in sorted(tok.items())))
    return lines


def cmd_diagnose(a: argparse.Namespace) -> int:
    text = "\n".join(diagnose(a.run, a.models))
    print(text)
    (a.run / "DIAGNOSE.txt").write_text(text + "\n")
    return 0


# ------------------------------------------------------------------------- main

def main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def live_args(s: argparse.ArgumentParser) -> None:
        s.add_argument("--models", type=Path, default=MODELS)
        s.add_argument("--base-url", default=f"http://127.0.0.1:{PORT}/v1")
        s.add_argument("--concurrency", type=int, default=2)
        s.add_argument("--timeout", type=float, default=600.0)
        s.add_argument("--code", action="store_true",
                       help="turn the code checker on (benched by default: on the Mac smoke run its programs ran "
                            "past the length limit on most tries, at about 6,000 tokens a puzzle)")
        s.add_argument("--drive", type=Path, default=None, help="a folder on the model drive, checked before retries")
        s.add_argument("--dry-run", action="store_true", help="fake models, $0")

    def w_args(s: argparse.ArgumentParser) -> None:
        s.add_argument("--w-tokens", type=float, default=ln.Weights.tokens)
        s.add_argument("--w-seconds", type=float, default=ln.Weights.seconds)
        s.add_argument("--w-usd", type=float, default=ln.Weights.usd)

    s = sub.add_parser("selftest")
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("make-dev")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--per-task", type=int, default=3)
    s = sub.add_parser("import")
    s.add_argument("--set", action="append", required=True, help="NAME=PATH to a saved run folder")
    s.add_argument("--models", type=Path, default=MODELS)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("learn")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--train", default="")
    s.add_argument("--test", default="")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--out", type=Path, required=True)
    w_args(s)
    s = sub.add_parser("sweep")
    s.add_argument("--dev", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    live_args(s)
    s = sub.add_parser("run")
    s.add_argument("--dev", type=Path, required=True)
    s.add_argument("--policy", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--explore", type=float, default=0.1)
    live_args(s)
    s = sub.add_parser("diagnose", help="explain a smoke run from its saved files; no model calls")
    s.add_argument("--run", type=Path, required=True, help="the --out folder of a mac run")
    s.add_argument("--models", type=Path, default=MODELS)
    s = sub.add_parser("fresh", help="the frozen policy against the reader on brand-new puzzles (Jev capped)")
    s.add_argument("--policy", type=Path, required=True, help="a saved policy, e.g. replay/saved/policy-tuned.json")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--per-kind", type=int, default=10)
    s.add_argument("--jev-cap", type=float, required=True, help="most dollars Jev may spend")
    s.add_argument("--mlx-server", default="")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--port", type=int, default=PORT)
    s.add_argument("--mlx-port", type=int, default=MLX_PORT)
    live_args(s)
    s = sub.add_parser("mac")
    s.add_argument("--night", type=Path, default=None, help="the night run's folder (holds set1/run, set2/run, ...)")
    s.add_argument("--mlx-server", default="")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--smoke", type=int, default=3, help="new dev puzzles per kind for the live sweep (0: skip)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--port", type=int, default=PORT)
    s.add_argument("--mlx-port", type=int, default=MLX_PORT)
    live_args(s)
    a = p.parse_args(argv)
    if a.cmd == "mac" and a.smoke and not a.dry_run and not a.mlx_server:
        raise SystemExit("mac needs --mlx-server for the live smoke run (or --smoke 0, or --dry-run)")
    if a.cmd == "fresh" and not a.dry_run and not a.mlx_server:
        raise SystemExit("fresh needs --mlx-server (or --dry-run)")
    return {"selftest": cmd_selftest, "make-dev": cmd_make_dev, "import": cmd_import, "learn": cmd_learn,
            "sweep": cmd_sweep, "run": cmd_run, "mac": cmd_mac,
            "diagnose": cmd_diagnose, "fresh": cmd_fresh}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
