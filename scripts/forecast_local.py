"""The 101 real forecasts, free on the Mac with local models. See docs/plans/2026-11-02-forecast-local.md.

Steps, in this order:

1. ``freeze`` -- writes ``prereg.json``: the SHA-256 of the forecast question and answer files, the code and the model
   file, and the tests and pass line. Done once, before any model call, and committed.
2. ``check`` -- no model call: the plan is unchanged, the model folder answers within a few seconds (it sits on
   the external WD_BLACK drive, which froze once) and holds all five models, the port is free, there is disk and
   memory room, and ``mlx_lm.server`` exists.
3. ``copy-models`` (optional, only if Eshaan asks) -- copies the five models from one Hugging Face cache to another
   folder. Copy only; the source is read, never changed.
4. ``run`` -- ``--dry-run`` answers with a fake model (for the plumbing); ``--live`` starts ``mlx_lm.server`` on
   ``127.0.0.1`` one model at a time, reading models only from ``--hf-cache`` with downloads off. Every reply is
   cached, so a rerun resumes. Answer files are never opened here.
5. ``score`` -- only now reads the answers; prints the two verdicts and a block to paste back.

Usage::

    uv run python scripts/forecast_local.py freeze
    uv run python scripts/forecast_local.py check --mlx-server <path> --hf-cache ~/.cache/huggingface/hub
    uv run python scripts/forecast_local.py run --live --mlx-server <path> --hf-cache ~/.cache/huggingface/hub
    uv run python scripts/forecast_local.py score
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from antelligence.experiments import fair_test as ft
from antelligence.experiments import forecast_local as fl
from antelligence.providers.base import ChatRequest, ChatResponse

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs/research/fair-test-20261029"
OUT = REPO / "docs/research/forecast-local-20261102"
QUESTIONS, ANSWERS = "forecast_questions.jsonl", "forecast_answers.jsonl"
CODE = (REPO / "antelligence/experiments/forecast_local.py", REPO / "antelligence/experiments/fair_test.py",
        Path(__file__).resolve())


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(data: Path, models: Path) -> Dict[str, Any]:
    return {
        "files": {QUESTIONS: sha(data / QUESTIONS), ANSWERS: sha(data / ANSWERS)},
        "code": {str(p.relative_to(REPO)): sha(p) for p in CODE},
        "models": sha(models),
        "max_tokens": fl.MAX_TOKENS, "temperature": fl.TEMPERATURE, "k": fl.K, "single_samples": fl.SINGLE,
        "lenses": list(ft.LENSES["forecast"]), "late_from": fl.LATE_FROM,
        "pass": {"min_brier_gain": fl.PASS_GAIN, "ci95_low_above": 0.0, "min_usable": fl.MIN_USABLE},
        "tests": {"A": "swarm5 vs single5, the local reader, all forecasts",
                  "B": f"mix5 (reader single 1 + one plain answer from each ant) vs single5, forecasts resolved on "
                       f"or after {fl.LATE_FROM}"},
    }


def changed(data: Path, models: Path, prereg: Path) -> List[str]:
    if not prereg.exists():
        return ["prereg.json is missing: run `freeze` first"]
    frozen, now = json.loads(prereg.read_text()), fingerprint(data, models)
    out = []
    for k in sorted(set(frozen) | set(now)):
        if frozen.get(k) != now.get(k):
            out.append(k)
    return out


def need_frozen(a: argparse.Namespace) -> None:
    diff = changed(a.data, a.models, a.out / "prereg.json")
    if diff:
        raise SystemExit(f"the plan changed since it was frozen ({', '.join(diff)}): stop, this would be a new test")


def questions(a: argparse.Namespace) -> List[ft.Question]:
    qs = ft.load_questions(a.data / QUESTIONS, "forecast")
    return qs[: a.limit] if getattr(a, "limit", 0) else qs


class FakeModel:
    """A $0 stand-in that answers at random (seeded by the request), for the plumbing only."""

    model = None

    async def complete(self, req: ChatRequest) -> ChatResponse:
        rng = random.Random(req.request_hash)
        body = json.dumps({"reason": "fake", "probability": round(rng.random(), 2)})
        return ChatResponse(content=body, model=req.model, request_hash=req.request_hash,
                            prompt_tokens=sum(len(m["content"]) for m in req.messages) // 4,
                            completion_tokens=len(body) // 4, finish_reason="stop")


class NoServer:
    """The dry run's server: nothing to start."""

    model = None

    def start(self, tag: str) -> None:
        self.model = tag

    def stop(self) -> None:
        self.model = None


def read_key(data: Path) -> Dict[str, str]:
    rows = [json.loads(x) for x in (data / ANSWERS).read_text().splitlines() if x.strip()]
    return {r["id"]: str(r["outcome"]).lower() for r in rows}


def read_late(data: Path) -> List[str]:
    rows = [json.loads(x) for x in (data / ANSWERS).read_text().splitlines() if x.strip()]
    return [r["id"] for r in rows if str(r.get("resolution_date", "")) >= fl.LATE_FROM]


def read_crowd(data: Path) -> Dict[str, float]:
    rows = [json.loads(x) for x in (data / ANSWERS).read_text().splitlines() if x.strip()]
    return {r["id"]: float(r["crowd_prob"]) for r in rows if r.get("crowd_prob") is not None}


# ------------------------------------------------------------------- commands

def cmd_freeze(a: argparse.Namespace) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    fl.load_models(a.models)   # refuses a bad model file before it is frozen
    (a.out / "prereg.json").write_text(json.dumps(fingerprint(a.data, a.models), indent=1, sort_keys=True) + "\n")
    print(f"frozen -> {a.out / 'prereg.json'}")
    return 0


def cmd_copy(a: argparse.Namespace) -> int:
    reader, ants = fl.load_models(a.models)
    dst = Path(a.to).expanduser()
    sizes = fl.copy_models(Path(a.src).expanduser(), dst, (reader, *ants))
    print(f"copied {len(sizes)} models, {sum(sizes.values()) / 1e9:.1f} GB -> {dst}")
    return 0


def preflight(a: argparse.Namespace) -> List[str]:
    """Problems that would stop a live run, found without any model call."""
    problems = [f"plan changed since freeze: {d}" for d in changed(a.data, a.models, a.out / "prereg.json")]
    reader, ants = fl.load_models(a.models)
    hub = Path(a.hf_cache).expanduser()
    drive = fl.drive_problem(hub) if a.hf_cache else None
    if not a.hf_cache:
        problems.append("give --hf-cache: the Hugging Face hub folder that holds the models")
    elif drive:
        problems.append(drive)
    else:
        gone = fl.missing_models(hub, (reader, *ants))
        if gone:
            problems.append(f"not in {hub}: {', '.join(gone)}")
    if not fl.port_free(a.port):
        problems.append(f"port {a.port} is in use; pick a free one with --port")
    if not a.mlx_server or not Path(a.mlx_server).expanduser().is_file():
        problems.append(f"no mlx_lm.server at {a.mlx_server or '(give --mlx-server)'}")
    free_disk = shutil.disk_usage(a.out if a.out.exists() else REPO).free / 1e9
    if free_disk < 5:
        problems.append(f"only {free_disk:.1f} GB free on disk for the run's files")
    mem = fl.free_gb()
    if mem is not None and mem < reader.gb + a.headroom_gb:
        problems.append(f"only {mem:.1f} GB memory free; the reader needs about {reader.gb:.0f} GB plus "
                        f"{a.headroom_gb:.0f} GB headroom")
    return problems


def cmd_check(a: argparse.Namespace) -> int:
    problems = preflight(a)
    reader, ants = fl.load_models(a.models)
    p = fl.plan(len(questions(a)), len(ants))
    print(f"plan: {p['questions']} forecasts, {p['reader_calls']} reader calls, {p['ant_calls']} ant calls, $0")
    for line in problems:
        print(f"PROBLEM: {line}")
    print("CHECK PASSED: ready for the live run" if not problems else "CHECK FAILED: fix the problems above first")
    return 0 if not problems else 1


def run_dir(a: argparse.Namespace) -> Path:
    return a.out / ("run_dry" if a.dry_run else "run")


def cmd_run(a: argparse.Namespace) -> int:
    if a.live == a.dry_run:
        raise SystemExit("say --live or --dry-run")
    need_frozen(a)
    reader, ants = fl.load_models(a.models)
    d = run_dir(a)
    d.mkdir(parents=True, exist_ok=True)
    qs = questions(a)
    if a.dry_run:
        server: Any = NoServer()
        provider: Any = FakeModel()
        guard = None
    else:
        problems = preflight(a)
        if problems:
            for line in problems:
                print(f"PROBLEM: {line}")
            raise SystemExit("not starting: fix the problems above first (`check` lists them)")
        from antelligence.providers.openai_compat import OpenAICompatProvider

        hub = Path(a.hf_cache).expanduser().resolve()
        server = fl.MlxServer([str(Path(a.mlx_server).expanduser())], hub, port=a.port, log_dir=d / "logs")
        provider = fl.OwnServer(OpenAICompatProvider(server.base_url, allowed_models=[reader.tag, *(x.tag for x in ants)],
                                                     timeout_s=600, max_concurrency=a.concurrency,
                                                     require_model_match=False, require_stop=False), server)

        def guard(model: fl.Model) -> None:
            drive = fl.drive_problem(hub)                    # the drive is checked again before every model load
            if drive:
                raise fl.DriveNotAnswering(f"before loading {model.tag}: {drive}")
            have = fl.free_gb()
            if have is not None and have < model.gb + a.headroom_gb:
                raise fl.MemoryTight(f"only {have:.1f} GB free before loading {model.tag} (needs about "
                                     f"{model.gb:.0f} GB plus {a.headroom_gb:.0f} GB); free memory, then rerun")
    runner = ft.Runner(provider, d / "cache.jsonl", max_usd=0.0, price=(0.0, 0.0), max_call_usd=0.0,
                       max_fail_streak=a.max_fail_streak)
    p = fl.plan(len(qs), len(ants))
    print(f"run: {p['questions']} forecasts, {p['calls']} calls ({len(runner.cache)} already cached) -> {d}", flush=True)
    t0 = time.monotonic()
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "live": a.live, "port": a.port,
            "hf_cache": str(a.hf_cache) if a.live else None, "questions": len(qs)}
    lr = fl.LocalRun(runner, server, reader, ants, concurrency=a.concurrency, guard=guard,
                     log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
    try:
        rows = asyncio.run(lr.run(qs))
    except (ft.BrokenRun, RuntimeError) as exc:
        meta.update(stopped=str(exc), seconds=round(time.monotonic() - t0))
        (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"STOPPED: {exc}\nFinished calls are kept; rerun the same command to resume.")
        return 2
    (d / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    meta.update(seconds=round(time.monotonic() - t0), calls=runner.calls, failed=runner.failed, loaded=lr.loaded)
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"done: {runner.calls} new calls ({runner.failed} unusable) in {meta['seconds']}s -> {d / 'rows.jsonl'}")
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    diff = changed(a.data, a.models, a.out / "prereg.json")
    d = run_dir(a)
    rows = [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines() if x.strip()]
    result = fl.score(rows, read_key(a.data), read_late(a.data), read_crowd(a.data))
    result["plan_changed"] = diff
    result["rows_sha256"] = sha(d / "rows.jsonl")
    (d / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    text = fl.block(result)
    if diff:
        text += f"\nWARNING: the plan changed since freeze ({', '.join(diff)}), so this is a new test, not the frozen one"
    print(text)
    return 0


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "copy-models", "check", "run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=DATA)
        p.add_argument("--out", type=Path, default=OUT)
        p.add_argument("--models", type=Path, default=fl.MODELS_FILE)
    cp = sub.choices["copy-models"]
    cp.add_argument("--from", dest="src", required=True, help="the Hugging Face hub cache that has the models now")
    cp.add_argument("--to", required=True, help="the folder to copy into")
    for name in ("check", "run"):
        p = sub.choices[name]
        p.add_argument("--mlx-server", default="")
        p.add_argument("--hf-cache", default="")
        p.add_argument("--port", type=int, default=fl.PORT)
        p.add_argument("--headroom-gb", type=float, default=6.0)
        p.add_argument("--limit", type=int, default=0, help="first N forecasts only (a smoke test)")
    rn = sub.choices["run"]
    rn.add_argument("--live", action="store_true")
    rn.add_argument("--dry-run", action="store_true")
    rn.add_argument("--concurrency", type=int, default=2)
    rn.add_argument("--max-fail-streak", type=int, default=25)
    sub.choices["score"].add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return {"freeze": cmd_freeze, "copy-models": cmd_copy, "check": cmd_check, "run": cmd_run,
            "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
