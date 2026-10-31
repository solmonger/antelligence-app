"""The decision-model add-on, end to end. See docs/plans/2026-10-30-dm-addon.md.

It reads swarms whose answers are already saved, so it makes no LLM call. Steps, in this order:

1. ``freeze`` -- writes ``prereg.json``: the SHA-256 of this code and of the fair test's own frozen plan, the decision
   models, the arms and the pass lines. Commit it before any real call.
2. ``export --source fair-test`` (after the fair test's live run) or ``export --source shared-read`` (after the
   shared-read run) -- turns a saved run into ``run/swarm_<source>.jsonl``: each ant's answer and reason.
3. ``plan --dm jev`` -- calls, tokens and dollars before any call.
4. ``smoke --dm jev --live`` -- 2 questions, 4 calls (well under a cent): prints the model the server reports, its
   answers and the cost. For a local model, start its server first (the plan says how) and use ``--dm d1-3b``.
5. ``run --dm jev --source fair-test --live`` -- asks the decision model ``dm1`` and ``read`` for every question.
   Jev stops at ``--max-usd`` (default $0.05). Every reply is cached, so a stopped run resumes without paying twice.
6. ``score --source fair-test`` -- only now reads the answer files. Prints every arm and the preregistered verdict.

Usage::

    uv run python scripts/dm_addon.py freeze
    uv run python scripts/dm_addon.py export --source fair-test
    uv run python scripts/dm_addon.py plan --dm jev --source fair-test
    uv run python scripts/dm_addon.py smoke --dm jev --live
    uv run python scripts/dm_addon.py run --dm jev --source fair-test --live
    uv run python scripts/dm_addon.py run --dm d1-3b --source fair-test --live      # free, local
    uv run python scripts/dm_addon.py score --source fair-test
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from antelligence.experiments import dm_addon as dm
from antelligence.experiments import fair_test as ft
from antelligence.providers.redact import scrub

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs/research/dm-addon-20261030"
FAIR = REPO / "docs/research/fair-test-20261029"
SHARED = REPO / "docs/research/shared-read-20261030"
CODE = (REPO / "antelligence/experiments/dm_addon.py", Path(__file__).resolve(),
        REPO / "antelligence/kernel/decision.py")
QFILES = {"exam": FAIR / "exam_questions.jsonl", "forecast": FAIR / "forecast_questions.jsonl"}
TYPESAFE_ENV_FILE_VAR = "ANTELLIGENCE_TYPESAFE_ENV_FILE"   # override for the env file holding TYPESAFE_API_KEY
HERMES_ENV = Path("~/.hermes/.env")


def rel(path: Path) -> str:
    """A path as the repo sees it, or as given when it lies outside the repo (tests write to a temp folder)."""
    path = Path(path)
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint() -> Dict[str, Any]:
    return {"code": {str(p.relative_to(REPO)): sha(p) for p in CODE},
            "fair_test_prereg": sha(FAIR / "prereg.json"),
            "dms": {n: {"model": s.model, "hosted": s.hosted} for n, s in dm.DMS.items()},
            "primary": {"dm": dm.PRIMARY_DM, "source": dm.PRIMARY_SOURCE, "arm": dm.PRIMARY_ARM,
                        "rivals": list(dm.RIVALS[dm.PRIMARY_SOURCE])},
            "pass": dm.PASS, "min_usable": dm.MIN_USABLE, "split_seed": dm.SPLIT_SEED, "bars": list(dm.BARS),
            "instructions": {f"{k}:{a}": v for (k, a), v in dm.DM_INSTRUCTIONS.items()},
            "reason_chars": dm.REASON_CHARS, "clip": list(dm.CLIP)}


def check_frozen(data: Path) -> None:
    path = data / "prereg.json"
    if not path.exists():
        raise SystemExit("no prereg.json: run `freeze` and commit it before any run")
    frozen, now = json.loads(path.read_text()), fingerprint()
    changed = sorted(k for k in set(frozen) | set(now) if frozen.get(k) != now.get(k))
    if changed:
        raise SystemExit(f"the plan changed since it was frozen ({', '.join(changed)}): that makes it a new test. "
                         "Re-freeze only if no real run has been scored, and say so.")


def questions(kinds: Sequence[str] = ft.KINDS) -> List[ft.Question]:
    return [q for k in kinds for q in ft.load_questions(QFILES[k], k)]


def options() -> Dict[str, Dict[str, str]]:
    return dm.load_options(QFILES["exam"])


def swarm_path(data: Path, source: str) -> Path:
    return data / "run" / f"swarm_{source}.jsonl"


def load_swarm(data: Path, source: str) -> Dict[str, dm.SwarmRow]:
    path = swarm_path(data, source)
    if not path.exists():
        raise SystemExit(f"no {rel(path)}: run `export --source {source}` first")
    return {r["id"]: dm.SwarmRow.from_json(r) for r in dm.read_jsonl(path)}


def read_typesafe_key() -> str:
    """TYPESAFE_API_KEY from the Hermes env file, read in-process and handed straight to the backend. Never printed,
    logged or put into the environment."""
    from dotenv import dotenv_values

    path = Path(os.environ.get(TYPESAFE_ENV_FILE_VAR) or HERMES_ENV).expanduser()
    key = (dotenv_values(path).get("TYPESAFE_API_KEY") or "").strip() if path.is_file() else ""
    if not key:
        raise SystemExit(f"TYPESAFE_API_KEY is not set in {path}")
    return key


def backend(spec: dm.DMSpec, live: bool, url: str, model: str) -> Any:
    if not live:
        return dm.FakeDM()
    from antelligence.experiments.v41_prod import USAGE_LEDGER
    from antelligence.providers.decision_backends import JevBackend, LedgeredDecision, SystemOneBackend
    from antelligence.providers.ledger import LedgerWriter
    from antelligence.providers.redact import register_secret

    if spec.hosted:
        key = read_typesafe_key()
        register_secret(key)
        return LedgeredDecision(JevBackend(api_key=key, model=model), LedgerWriter(USAGE_LEDGER, redact=[key]))
    return LedgeredDecision(SystemOneBackend(url, model=model, timeout_s=600.0), LedgerWriter(USAGE_LEDGER))


def spec_of(a: argparse.Namespace) -> dm.DMSpec:
    spec = dm.DMS[a.dm]
    return dm.DMSpec(spec.name, a.model or spec.model, a.url or spec.url, spec.price, spec.hosted,
                     a.concurrency or spec.concurrency, spec.about)


def cmd_freeze(a: argparse.Namespace) -> int:
    a.data.mkdir(parents=True, exist_ok=True)
    (a.data / "prereg.json").write_text(json.dumps(fingerprint(), indent=1, sort_keys=True) + "\n")
    print(f"froze -> {a.data / 'prereg.json'}")
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    if a.source == "fair-test":
        rows_path, cache_path = a.fair_rows or FAIR / "rows_live.jsonl", a.fair_cache or FAIR / "cache_live.jsonl"
        for p in (rows_path, cache_path):
            if not p.exists():
                raise SystemExit(f"no {p}: the fair test's live run has to finish first")
        rows, stats = dm.export_fair_test(questions(), dm.read_jsonl(rows_path), dm.read_jsonl(cache_path))
        print(f"{len(rows)} questions; ant reasons found {stats['reasons_found']}, missing {stats['reasons_missing']}")
        if stats["reasons_missing"] > 0.1 * max(1, stats["reasons_found"] + stats["reasons_missing"]):
            raise SystemExit("over 10% of the ants' reasons are missing from the cache: the cache and rows don't "
                             "match the frozen fair-test code, so `read` would not be a fair read")
    else:
        run = a.sr_run or SHARED / "run"
        if not (run / "rows.jsonl").exists():
            raise SystemExit(f"no {run / 'rows.jsonl'}: the shared-read run has to finish first")
        rows = dm.export_shared_read(dm.read_jsonl(run / "rows.jsonl"))
        print(f"{len(rows)} questions, {len(rows[0].ants) if rows else 0} ants each (answers only, no reasons)")
    dm.write_jsonl(swarm_path(a.data, a.source), [r.to_json() for r in rows])
    print(f"-> {rel(swarm_path(a.data, a.source))}")
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    check_frozen(a.data)
    spec = spec_of(a)
    est = dm.estimate(questions(), load_swarm(a.data, a.source), options(), spec)
    print(f"{spec.name} ({spec.model}) on {a.source}: {est['questions']} questions, {est['calls']} calls, about "
          f"{est['input_tokens']:,} input tokens, about ${est['usd']:.4f}" + (" (local, free)" if not spec.hosted else ""))
    return 0


def _cache(a: argparse.Namespace, spec: dm.DMSpec, live: bool) -> Path:
    return a.data / "run" / f"cache_{spec.name}{'' if live else '_dry'}.jsonl"


def _live(a: argparse.Namespace) -> bool:
    if a.live == a.dry_run:
        raise SystemExit("say --live (the real model) or --dry-run (a fake one, $0)")
    return bool(a.live)


def cmd_smoke(a: argparse.Namespace) -> int:
    live, spec = _live(a), spec_of(a)
    qs = questions()
    pick = [next(q for q in qs if q.kind == "exam"), next(q for q in qs if q.kind == "forecast")]
    path = swarm_path(a.data, "fair-test")
    swarm = {r["id"]: dm.SwarmRow.from_json(r) for r in dm.read_jsonl(path)} if path.exists() else {}
    runner = dm.DMRunner(backend(spec, live, spec.url, spec.model), _cache(a, spec, live), max_usd=a.max_usd,
                         price=spec.price)
    opts = options()
    ok = True
    for q in pick:
        ants = swarm[q.id].ants if q.id in swarm else tuple(dm.Ant(None) for _ in range(ft.K))
        for arm in ("dm1", "read"):
            start = time.monotonic()
            try:
                row = asyncio.run(runner.ask(dm.dm_request(spec.model, q, arm, opts.get(q.id), ants)))
            except Exception as exc:
                print(f"{q.id} {arm}: FAILED {type(exc).__name__}: {scrub(str(exc))[:300]}")
                ok = False
                continue
            ans = dm.to_answer(q.kind, row)
            ok &= ans is not None
            print(f"{q.id} {arm}: served {row.get('served_model')!r}, {time.monotonic() - start:.2f}s, "
                  f"{row.get('input_tokens')} tokens, ${row.get('cost_usd', 0):.6f} -> "
                  f"{json.dumps(ans) if ans else row.get('reason')}")
    print(f"spent ${runner.spent:.6f} in this cache. " + ("OK: start the full run." if ok else "NOT OK: fix the "
          "server or the model name before a full run."))
    return 0 if ok else 2


def cmd_run(a: argparse.Namespace) -> int:
    live, spec = _live(a), spec_of(a)
    check_frozen(a.data)
    swarm = load_swarm(a.data, a.source)
    qs = [q for q in questions() if q.id in swarm]
    if a.limit:
        qs = [q for k in ft.KINDS for q in [x for x in qs if x.kind == k][: a.limit]]
    est = dm.estimate(qs, swarm, options(), spec)
    runner = dm.DMRunner(backend(spec, live, spec.url, spec.model), _cache(a, spec, live), max_usd=a.max_usd,
                         price=spec.price)
    print(f"{spec.name}: {len(qs)} questions, {est['calls']} calls, about ${est['usd']:.4f}; cap ${a.max_usd:.2f}; "
          f"already ${runner.spent:.4f} in the cache")
    done: List[Dict[str, Any]] = []

    def save(row: Dict[str, Any]) -> None:
        done.append(row)
        if len(done) % 25 == 0:
            print(f"  {len(done)}/{len(qs)} questions, ${runner.spent:.4f} spent", flush=True)

    status = "complete"
    start = time.monotonic()
    try:
        asyncio.run(runner.run(qs, swarm, spec.model, options(), concurrency=spec.concurrency, on_done=save))
    except ft.SpendCap as exc:
        status = f"stopped at the cap: {exc}"
    except dm.BrokenRun as exc:
        status = f"STOPPED, the model's replies are not usable: {scrub(str(exc))[:300]}"
    except Exception as exc:  # keep what is done; a rerun resumes from the cache
        status = f"stopped: {type(exc).__name__}: {scrub(str(exc))[:300]}"
    out = a.data / "run" / f"dm_{spec.name}_{a.source}{'' if live else '_dry'}.jsonl"
    order = {q.id: i for i, q in enumerate(qs)}
    dm.write_jsonl(out, sorted(done, key=lambda r: order[r["id"]]))
    print(f"{status}\n{len(done)}/{len(qs)} questions in {time.monotonic() - start:.0f}s, {runner.calls} new calls "
          f"({runner.failed} failed), ${runner.spent:.4f} spent in total -> {rel(out)}")
    return 0 if status == "complete" else 2


def read_key() -> Dict[str, Any]:
    key: Dict[str, Any] = {}
    for r in dm.read_jsonl(FAIR / "exam_answers.jsonl"):
        key[r["id"]] = r["label"]
    for r in dm.read_jsonl(FAIR / "forecast_answers.jsonl"):
        key[r["id"]] = str(r["outcome"]).lower()
    return key


def read_crowd() -> Dict[str, float]:
    return {r["id"]: float(r["crowd_prob"]) for r in dm.read_jsonl(FAIR / "forecast_answers.jsonl")
            if r.get("crowd_prob") is not None}


def cost_of(cache: Path) -> Dict[str, float]:
    rows = dm.read_jsonl(cache) if cache.exists() else []
    ok = [r for r in rows if not r.get("error")]
    return {"calls": len(rows), "failed": len(rows) - len(ok), "usd": round(sum(r.get("cost_usd") or 0 for r in rows), 6),
            "input_tokens": sum(r.get("input_tokens") or 0 for r in rows),
            "median_s": round(st.median(r.get("elapsed_s") or 0 for r in ok), 3) if ok else 0.0}


def cmd_score(a: argparse.Namespace) -> int:
    check_frozen(a.data)
    swarm = load_swarm(a.data, a.source)
    suffix = "_dry" if a.dry_run else ""
    dms: Dict[str, Dict[str, Any]] = {}
    costs: Dict[str, Dict[str, float]] = {}
    for name in dm.DMS:
        path = a.data / "run" / f"dm_{name}_{a.source}{suffix}.jsonl"
        if path.exists():
            dms[name] = {r["id"]: r for r in dm.read_jsonl(path)}
            costs[name] = cost_of(a.data / "run" / f"cache_{name}{suffix}.jsonl")
    if not dms:
        raise SystemExit("no decision-model answers yet: run first")
    result = dm.score(a.source, list(swarm.values()), dms, read_key(), read_crowd(), costs)
    out = a.data / "run" / f"result_{a.source}{suffix}.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(report(result))
    print(f"-> {rel(out)}")
    return 0


def report(result: Mapping[str, Any]) -> str:
    """The result in a few plain lines: every arm's score and LLM calls, then what each decision model adds."""
    lines = [f"source: {result['source']}"]
    for kind, b in result["kinds"].items():
        metric = "share right (higher is better)" if kind == "exam" else "Brier (lower is better)"
        lines.append(f"\n{kind}: n={b['n']}, {b['ants']} ants, ants usable {b['ants_usable']:.0%}; {metric}")
        for arm, m in sorted(b["means"].items(), key=lambda kv: (-kv[1] if kind == "exam" else kv[1])):
            lines.append(f"  {arm:18s} {m:.4f}   LLM calls {b['llm_calls'][arm]:g}")
        if "crowd_brier" in b:
            lines.append(f"  {'market crowd':18s} {b['crowd_brier']:.4f}")
        for name, info in b["dms"].items():
            c = info["cheap"]
            lines.append(f"  {name}: usable {info['usable_dm1']:.0%}; vote vs ants {info['vote_vs_ants']['gain']:+.4f} "
                         f"{info['vote_vs_ants']['ci95']}; cheap asks the ants {c['test_asked_share']:.0%} of the time "
                         f"(bar {c['bar']}) for {c['score']} vs the ants' {c['ants_score_same_half']} on the same half")
            vs = [(k, v) for k, v in sorted(info.items()) if k.startswith(("vote_vs_", "read_vs_")) and k != "vote_vs_ants"]
            if vs:
                lines.append("    " + "; ".join(f"{k.replace('_vs_', ' vs ')} {v['gain']:+.4f} {v['ci95']}" for k, v in vs))
            if "cost" in info:
                lines.append(f"    cost: {info['cost']['calls']} calls, ${info['cost']['usd']:.4f}, "
                             f"median {info['cost']['median_s']}s a call")
        for g, row in b.get("by_group", {}).items():
            best = sorted(((k, v) for k, v in row.items() if k != "n"), key=lambda kv: -kv[1])[:4]
            lines.append(f"  {g} (n={row['n']}): " + ", ".join(f"{k} {v:.3f}" for k, v in best))
    p = result.get("primary")
    if p:
        lines.append(f"\nverdict (Jev vote vs single5 and swarm5): {p['verdict']}")
        for kind in ft.KINDS:
            if kind in p:
                v = p[kind]
                lines.append(f"  {kind}: " + "; ".join(f"vs {r} {v[r]['gain']:+.4f} {v[r]['ci95']}"
                                                        for r in dm.RIVALS[dm.PRIMARY_SOURCE])
                             + (" -> PASS" if v["pass"] else " -> no pass"))
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "export", "plan", "smoke", "run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=DATA)
    for name in ("export", "plan", "run", "score"):
        sub.choices[name].add_argument("--source", choices=dm.SOURCES, default="fair-test")
    for name in ("plan", "smoke", "run"):
        p = sub.choices[name]
        p.add_argument("--dm", choices=sorted(dm.DMS), default=dm.PRIMARY_DM)
        p.add_argument("--url", default="", help="a local model's server, e.g. http://127.0.0.1:8520")
        p.add_argument("--model", default="", help="the model id its server wants")
        p.add_argument("--concurrency", type=int, default=0)
    for name in ("smoke", "run"):
        p = sub.choices[name]
        p.add_argument("--live", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--max-usd", type=float, default=0.05)
    sub.choices["run"].add_argument("--limit", type=int, default=0, help="first N questions of each kind")
    sub.choices["score"].add_argument("--dry-run", action="store_true", help="score the dry-run files")
    ex = sub.choices["export"]
    ex.add_argument("--fair-rows", type=Path, default=None)
    ex.add_argument("--fair-cache", type=Path, default=None)
    ex.add_argument("--sr-run", type=Path, default=None)
    a = ap.parse_args(argv)
    return {"freeze": cmd_freeze, "export": cmd_export, "plan": cmd_plan, "smoke": cmd_smoke, "run": cmd_run,
            "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
