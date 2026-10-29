"""The fair test, end to end: freeze the plan, run the calls, then score. See docs/plans/2026-10-29-fair-test.md.

Five steps, in this order:

1. ``freeze`` -- writes ``prereg.json`` next to the question files: the SHA-256 of every question and answer file
   and of the harness code, the model, the arms and the pass lines. Commit it before any paid call.
2. ``plan`` -- prints what a run would cost (calls, tokens, dollars) and checks nothing changed since ``freeze``.
3. ``smoke`` -- one question through every arm (20 calls, well under a cent). It prints the model name the portal
   reports, how the replies finished, the first reply and any problem, and says whether to start the full run.
4. ``run`` -- asks the model. ``--dry-run`` uses a fake model ($0, for checking the plumbing); ``--live`` uses
   DeepSeek V4.1 Flash on the Nous portal with the key from the Hermes env file (never printed). Live calls are
   also written to the repo's usage ledger and stop at the existing Nous budget stop, on top of ``--max-usd``.
   Every usable reply is cached, so a stopped run resumes without paying twice; a failed call is asked again. The
   run stops by itself after 5 unusable replies in a row and prints why. Answer files are never opened here.
5. ``score`` -- only now reads the answer files, and gives the preregistered verdict, or "invalid run" when under
   90% of the answers were usable (then nothing was measured).

Usage::

    uv run python scripts/fair_test.py freeze
    uv run python scripts/fair_test.py plan
    uv run python scripts/fair_test.py smoke --live
    uv run python scripts/fair_test.py run --live --max-usd 3
    uv run python scripts/fair_test.py score
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from antelligence.experiments import fair_test as ft
from antelligence.providers.base import ChatRequest, ChatResponse
from antelligence.providers.redact import scrub

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs/research/fair-test-20261029"
FILES = {"exam": ("exam_questions.jsonl", "exam_answers.jsonl"),
         "forecast": ("forecast_questions.jsonl", "forecast_answers.jsonl")}
LATE_FROM = "2026-09-10"   # DeepSeek V4.1 Flash came out on this day: no training data can hold what resolved after it
CODE = (REPO / "antelligence/experiments/fair_test.py", Path(__file__).resolve())


class FakeModel:
    """A $0 stand-in that answers at random (seeded by the request), for checking the plumbing only."""

    async def complete(self, req: ChatRequest) -> ChatResponse:
        rng = random.Random(req.request_hash)
        text = req.messages[-1]["content"]
        if '"answer"' in text:
            body = {"reason": "fake", "answer": rng.choice("ABCDE")}
        else:
            body = {"reason": "fake", "probability": round(rng.random(), 2)}
        content = json.dumps(body)
        prompt_tokens = sum(len(m["content"]) for m in req.messages) // 4
        return ChatResponse(content=content, model=req.model, request_hash=req.request_hash,
                            prompt_tokens=prompt_tokens, completion_tokens=len(content) // 4, finish_reason="stop")


def load_all(data: Path, kinds: Sequence[str]) -> List[ft.Question]:
    out: List[ft.Question] = []
    for kind in kinds:
        path = data / FILES[kind][0]
        if path.exists():
            out += ft.load_questions(path, kind)
    return out


def fingerprint(data: Path) -> Dict[str, Any]:
    files = {}
    for kind, names in FILES.items():
        for name in names:
            if (data / name).exists():
                files[name] = ft.sha256_file(data / name)
    code = {str(p.relative_to(REPO)): ft.sha256_file(p) for p in CODE}
    return {"files": files, "code": code, "model": ft.MODEL, "temperature": ft.TEMPERATURE,
            "max_tokens": ft.MAX_TOKENS, "k": ft.K, "single_samples": ft.SINGLE_SAMPLES,
            "lenses": {k: list(v) for k, v in ft.LENSES.items()}, "pass": dict(ft.PASS),
            "primary": "swarm5 vs single5 (same model, same 5 calls per question)"}


def check_frozen(data: Path) -> Dict[str, Any]:
    path = data / "prereg.json"
    if not path.exists():
        raise SystemExit("no prereg.json: run `freeze` and commit it before any run")
    frozen = json.loads(path.read_text())
    now = fingerprint(data)
    changed = [k for k in ("files", "code", "pass", "lenses", "model") if frozen.get(k) != now.get(k)]
    if changed:
        raise SystemExit(f"changed since freeze: {', '.join(changed)}. A changed test is a new test: re-freeze, "
                         f"commit, and say so in the write-up")
    return frozen


def cmd_freeze(a: argparse.Namespace) -> int:
    fp = fingerprint(a.data)
    if not any(n.endswith("_questions.jsonl") for n in fp["files"]):
        raise SystemExit("no question files found")
    (a.data / "prereg.json").write_text(json.dumps(fp, indent=1, sort_keys=True) + "\n")
    print(f"froze {len(fp['files'])} files -> {a.data / 'prereg.json'}")
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    check_frozen(a.data)
    qs = load_all(a.data, a.kinds)
    for kind in a.kinds:
        sub = [q for q in qs if q.kind == kind]
        if sub:
            print(kind, ft.estimate(sub))
    total = ft.estimate(qs)
    print("total", total)
    try:
        from antelligence.experiments import v41_budget
        spent = v41_budget.spend_now()["nous"]
        print(f"Nous spend already in the usage ledger: ${spent:.2f} (the stop is ${v41_budget.STOPS['nous']:.2f})")
    except Exception as exc:  # no ledger here: say so, don't guess
        print(f"Nous spend so far: not readable here ({type(exc).__name__})")
    return 0


def live_provider(*, ledger: Optional[Path] = None, transport: Any = None) -> Any:
    from antelligence.experiments.v41_prod import USAGE_LEDGER, BudgetGuard, GuardedChat, read_nous_key
    from antelligence.providers.decision_backends import NOUS_URL
    from antelligence.providers.ledger import LedgerWriter, LedgeredChat
    from antelligence.providers.openai_compat import OpenAICompatProvider
    from antelligence.providers.redact import register_secret

    key = read_nous_key()
    register_secret(key)
    # The provider must hand back every billed reply, so the runner can count its cost and judge it: the served
    # model name (spelled differently by the portal) and the finish reason are checked in ft.Runner.ask instead of
    # being raised here, where a refused reply's cost was lost.
    ledger = Path(ledger or USAGE_LEDGER)
    chat = OpenAICompatProvider(NOUS_URL, api_key=key, allowed_models=[ft.MODEL], max_concurrency=4,
                                extra_body=ft.THINKING_OFF, require_model_match=False, require_stop=False,
                                transport=transport)
    ledgered = LedgeredChat(chat, LedgerWriter(ledger, redact=[key]), role="worker",
                            expected_model="", backend="nous", prefix=True, price=ft.PRICE)  # "" accepts any name
    return GuardedChat(ledgered, BudgetGuard("nous", usage=ledger))


def run_questions(runner: ft.Runner, qs: List[ft.Question], concurrency: int, every: int = 20):
    done: Dict[str, Dict[str, Any]] = {}

    def save(row: Dict[str, Any]) -> None:
        done[row["id"]] = row
        if len(done) % every == 0:
            print(f"  {len(done)}/{len(qs)} questions, ${runner.spent:.4f} spent", flush=True)

    status = "complete"
    try:
        asyncio.run(runner.run(qs, concurrency=concurrency, on_done=save))
    except ft.OutOfCredits as exc:
        status = f"STOPPED: {exc}"
    except ft.BrokenRun as exc:
        status = f"STOPPED, the model's replies are not usable: {exc}"
    except ft.SpendCap as exc:
        status = f"stopped at the cap: {exc}"
    except Exception as exc:  # an outage or a budget stop: keep what is done, a rerun resumes from the cache
        status = f"stopped: {type(exc).__name__}: {scrub(str(exc))[:300]}"
    return [done[q.id] for q in qs if q.id in done], status


def cmd_run(a: argparse.Namespace) -> int:
    if a.live == a.dry_run:
        raise SystemExit("say --live (paid, real model) or --dry-run (free, fake model)")
    check_frozen(a.data)
    qs = load_all(a.data, a.kinds)
    if a.limit:
        qs = [q for kind in a.kinds for q in [x for x in qs if x.kind == kind][: a.limit]]
    tag = "live" if a.live else "dry"
    out = a.out or a.data / f"rows_{tag}.jsonl"
    cache = a.cache or a.data / f"cache_{tag}.jsonl"
    provider = live_provider() if a.live else FakeModel()
    runner = ft.Runner(provider, cache, max_usd=a.max_usd)
    est = ft.estimate(qs)
    print(f"{len(qs)} questions, about {est['calls']} calls, about ${est['usd']}; cap ${a.max_usd:.2f}; "
          f"already spent ${runner.spent:.4f} on cached calls")
    if runner.ignored:
        print(f"  {runner.ignored} failed calls from an earlier run are in the cache; they will be asked again")
    rows, status = run_questions(runner, qs, a.concurrency)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"{status}\n{len(rows)}/{len(qs)} questions, {runner.calls} new calls ({runner.failed} failed), "
          f"${runner.spent:.4f} spent in total -> {out}")
    return 0 if status == "complete" else 2


def cmd_smoke(a: argparse.Namespace) -> int:
    """One question through every arm (20 calls, well under a cent), with what the model actually sent back."""
    if a.live == a.dry_run:
        raise SystemExit("say --live (paid, real model) or --dry-run (free, fake model)")
    check_frozen(a.data)
    qs = load_all(a.data, [a.kind])
    if not qs:
        raise SystemExit(f"no {a.kind} questions in {a.data}")
    tag = "live" if a.live else "dry"
    cache = a.data / f"cache_{tag}.jsonl"  # the full run reuses these replies; failed ones are asked again
    runner = ft.Runner(live_provider() if a.live else FakeModel(), cache, max_usd=a.max_usd)
    rows, status = run_questions(runner, qs[:1], concurrency=1, every=1)
    new = [json.loads(line) for line in cache.read_text().splitlines()[-runner.calls:]] if runner.calls else []
    served = sorted({r.get("served_model") or "none" for r in new})
    finish = sorted({r.get("finish_reason") or "none" for r in new})
    usable = sum(a is not None for r in rows for arm in ("single", "swarm", "seen") for a in r[arm])
    print(f"question {qs[0].id}: {usable}/20 usable answers; served as {', '.join(served)}; "
          f"finished: {', '.join(finish)}; ${runner.spent:.4f} spent so far in this cache")
    if new:
        print(f"first reply: {new[0].get('content', '')[:200]!r}")
    for reason in dict.fromkeys(r.get("reason") for r in new if r.get("reason")):
        print(f"problem: {reason}")
    ok = status == "complete" and usable >= 18
    print("SMOKE CHECK PASSED: start the full run." if ok else f"SMOKE CHECK FAILED: {status}. Do not start the full run.")
    return 0 if ok else 2


def read_key(data: Path) -> Dict[str, Any]:
    key: Dict[str, Any] = {}
    for kind, (_, answers) in FILES.items():
        path = data / answers
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                key[r["id"]] = r["label"] if kind == "exam" else str(r["outcome"]).lower()
    return key


def read_crowd(data: Path) -> Dict[str, float]:
    path = data / FILES["forecast"][1]
    crowd: Dict[str, float] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("crowd_prob") is not None:
                    crowd[r["id"]] = float(r["crowd_prob"])
    return crowd


def late_ids(data: Path) -> set:
    path = data / FILES["forecast"][1]
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"] for r in rows if str(r.get("resolution_date", "")) >= LATE_FROM}


def cmd_score(a: argparse.Namespace) -> int:
    check_frozen(a.data)
    rows_path = a.rows or a.data / "rows_live.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    key, crowd = read_key(a.data), read_crowd(a.data)
    result = ft.score(rows, key, crowd)
    late = late_ids(a.data)
    late_rows = [r for r in rows if r["id"] in late]
    if late_rows:  # secondary, no verdict: only forecasts that resolved after the model was released
        sub = ft.score(late_rows, key, crowd)
        result["secondary"]["forecast_resolved_after_release"] = {"from": LATE_FROM, **{k: v for k, v in sub["primary"]["forecast"].items() if k != "pass"},
                                                                  "crowd": sub["secondary"]["forecast"].get("crowd")}
    result["rows_file"] = rows_path.name
    result["rows_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    out = a.out or a.data / f"result_{rows_path.stem.removeprefix('rows_')}.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for kind, p in result["primary"].items():
        word = "PASS" if p["pass"] else "no pass"
        print(f"{kind}: swarm {p['swarm5']} vs single {p['single5']} ({p['metric']}, n={p['n']}), gain {p['gain']} "
              f"[{p['ci95'][0]}, {p['ci95'][1]}] -> {word}")
    print(f"verdict: {result['verdict']} -> {out}")
    return 0


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "plan", "smoke", "run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=DATA)
        p.add_argument("--kinds", default="exam,forecast")
    run = sub.choices["run"]
    run.add_argument("--live", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--max-usd", type=float, default=3.0)
    run.add_argument("--limit", type=int, default=0, help="first N questions of each kind (a smoke test)")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--cache", type=Path, default=None)
    run.add_argument("--out", type=Path, default=None)
    sm = sub.choices["smoke"]
    sm.add_argument("--live", action="store_true")
    sm.add_argument("--dry-run", action="store_true")
    sm.add_argument("--kind", choices=ft.KINDS, default="exam")
    sm.add_argument("--max-usd", type=float, default=0.05)
    sc = sub.choices["score"]
    sc.add_argument("--rows", type=Path, default=None)
    sc.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    a.kinds = [k for k in a.kinds.split(",") if k]
    bad = set(a.kinds) - set(ft.KINDS)
    if bad:
        raise SystemExit(f"unknown kinds: {sorted(bad)}")
    return {"freeze": cmd_freeze, "plan": cmd_plan, "smoke": cmd_smoke, "run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
