"""The shared-read test on your own machine: one bigger reader, eight small mixed ants. See docs/plans/2026-10-30-shared-read.md.

Free: every call goes to a local model server (Ollama by default, at http://localhost:11434/v1). Nothing is sent
anywhere else, and the answer file is opened only by ``score``.

Steps, in this order::

    # once: get the five models (about 20 GB of downloads in all)
    ollama pull qwen3:14b && ollama pull qwen3:4b && ollama pull llama3.2:3b \\
      && ollama pull gemma3:4b && ollama pull phi4-mini:3.8b

    uv run python scripts/shared_read.py check          # is the server up, are the models there (no model call)
    uv run python scripts/shared_read.py bench          # 2 questions through everything: speed, and a look at the notes
    uv run python scripts/shared_read.py run --limit 60 # a quick pass: 15 questions from each of the 4 tasks
    uv run python scripts/shared_read.py run            # all 240 (resumes; nothing done is asked again)
    uv run python scripts/shared_read.py score          # only now reads the answers, prints the verdict

``--dry-run`` on ``bench`` and ``run`` uses a fake model, for checking the plumbing without a server. ``--base-url``
points at another OpenAI-compatible server (for MLX: ``mlx_lm.server``, then ``--base-url http://localhost:8080/v1
--loose-names``, since MLX reports a file path as the model name). ``--reader`` and ``--ants`` swap models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from antelligence.experiments import shared_read as sr
from antelligence.providers.base import ChatRequest, ChatResponse

REPO = Path(__file__).resolve().parents[1]
QUESTIONS = REPO / "docs/research/fair-test-20261029/exam_questions.jsonl"
ANSWERS = REPO / "docs/research/fair-test-20261029/exam_answers.jsonl"
OUT = REPO / "docs/research/shared-read-20261030"
OLLAMA = "http://localhost:11434/v1"
PREREG = OUT / "prereg.json"
FROZEN = {"questions": QUESTIONS, "answers": ANSWERS, "harness": REPO / "antelligence/experiments/shared_read.py",
          "script": Path(__file__).resolve(), "plan": REPO / "docs/plans/2026-10-30-shared-read.md"}


def fingerprint() -> Dict[str, Any]:
    import hashlib

    return {"files": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in FROZEN.items()},
            "pass": sr.PASS, "min_usable": sr.MIN_USABLE, "reader": sr.READER.tag,
            "ants": [s.tag for s in sr.ANTS], "lenses": list(sr.LENS_IDS), "arms": list(sr.ARMS),
            "temperature": sr.TEMPERATURE}


def frozen_changes() -> List[str]:
    """What differs from the frozen plan (empty when nothing does)."""
    if not PREREG.exists():
        return ["no prereg.json"]
    old, new = json.loads(PREREG.read_text()), fingerprint()
    out = [f"file {k}" for k in new["files"] if old.get("files", {}).get(k) != new["files"][k]]
    return out + [k for k in new if k != "files" and old.get(k) != new[k]]


class FakeModel:
    """A $0 stand-in for the plumbing: notes are filler, answers are random letters seeded by the request, and Qwen
    tags get a <think> block like the real thing."""

    async def complete(self, req: ChatRequest) -> ChatResponse:
        rng = random.Random(req.request_hash)
        last = req.messages[-1]["content"]
        if '"answer"' in last:
            body = json.dumps({"reason": "fake", "answer": rng.choice("ABCDE")})
        else:
            body = "The question asks which order works. Key rules: the stated conditions. Trap: misreading 'before'."
        if req.model.startswith("qwen3"):
            body = f"<think>\n\n</think>\n\n{body}"
        return ChatResponse(content=body, model=req.model, request_hash=req.request_hash,
                            prompt_tokens=sum(len(m["content"]) for m in req.messages) // 4,
                            completion_tokens=len(body) // 4, finish_reason="stop", elapsed_s=0.0)


def parse_specs(text: str) -> Tuple[sr.ModelSpec, ...]:
    """``tag=family`` items, comma separated; a Qwen3 tag gets the no-think switch."""
    out = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        tag, _, family = part.partition("=")
        out.append(sr.ModelSpec(tag, family or tag.split(":")[0], no_think=tag.lower().startswith("qwen3")))
    return tuple(out)


def setup(a: argparse.Namespace) -> Tuple[sr.ModelSpec, Tuple[sr.ModelSpec, ...]]:
    reader = parse_specs(a.reader)[0] if a.reader else sr.READER
    ants = parse_specs(a.ants) if a.ants else sr.ANTS
    return reader, ants


def provider(a: argparse.Namespace) -> Any:
    if a.dry_run:
        return FakeModel()
    from antelligence.providers.openai_compat import OpenAICompatProvider

    # the runner checks the served name itself (allowing Ollama's ":latest"), so the client doesn't
    return OpenAICompatProvider(a.base_url, timeout_s=a.timeout, max_concurrency=a.concurrency,
                                require_model_match=False)


def pick(items: Sequence[sr.Item], limit: int) -> List[sr.Item]:
    """The first ``limit`` items taken in turn from each task, so a short run covers all four tasks evenly."""
    if not limit or limit >= len(items):
        return list(items)
    by: Dict[str, List[sr.Item]] = defaultdict(list)
    for i in items:
        by[i.group].append(i)
    out: List[sr.Item] = []
    k = 0
    while len(out) < limit:
        for g in sorted(by):
            if k < len(by[g]) and len(out) < limit:
                out.append(by[g][k])
        k += 1
    return out


def out_dir(a: argparse.Namespace) -> Path:
    d = Path(a.out) if a.out else OUT / ("dry-run" if getattr(a, "dry_run", False) else "run")
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _run(a: argparse.Namespace, items: Sequence[sr.Item], runner: sr.Runner,
               reader: sr.ModelSpec, ants: Sequence[sr.ModelSpec], quiet: bool = False) -> List[Dict[str, Any]]:
    start = time.monotonic()
    done = [0]

    def tick(_: str) -> None:
        done[0] += 1
        if not quiet and (done[0] % 10 == 0 or done[0] == len(items)):
            print(f"  reader: {done[0]}/{len(items)} questions read ({time.monotonic() - start:.0f}s)", flush=True)

    if not quiet:
        print(f"Phase 1: the reader ({reader.tag}) writes notes and answers on its own", flush=True)
    reading = await sr.run_reader(runner, items, reader, on_done=tick)

    def model(tag: str) -> None:
        if not quiet:
            print(f"Phase 2: ant model {tag} answers alone, from notes, and both ({time.monotonic() - start:.0f}s)",
                  flush=True)

    answers = await sr.run_ants(runner, items, reading, ants, on_model=model)
    return sr.rows(items, reading, answers)


def cmd_freeze(a: argparse.Namespace) -> int:
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps({"date": "2026-10-30", **fingerprint()}, indent=1, sort_keys=True) + "\n")
    print(f"Frozen: {PREREG}. Commit it before the first real run.")
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    import httpx

    reader, ants = setup(a)
    need = [reader.tag] + [s.tag for s in ants]
    try:
        resp = httpx.get(f"{a.base_url.rstrip('/')}/models", timeout=10, trust_env=False)
    except httpx.HTTPError as exc:
        print(f"No server answered at {a.base_url} ({type(exc).__name__}). Start it first: `ollama serve` "
              "(or open the Ollama app).")
        return 2
    if resp.status_code != 200:
        print(f"{a.base_url}/models said HTTP {resp.status_code}")
        return 2
    have = {str(m.get("id")) for m in resp.json().get("data") or []}
    missing = [t for t in need if not any(sr.same_model(h, t) for h in have)]
    for t in need:
        print(f"  {'ok     ' if t not in missing else 'MISSING'} {t}")
    if missing:
        print("Pull the missing ones, then check again:\n  " + "\n  ".join(f"ollama pull {t}" for t in missing))
        return 1
    print(f"All {len(need)} models are there. Next: `uv run python scripts/shared_read.py bench`.")
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    reader, ants = setup(a)
    items = pick(sr.load_items(QUESTIONS), a.limit)
    p = sr.plan(len(items), ants)
    print(json.dumps(p, indent=1))
    print(f"reader {reader.tag}; ants {', '.join(s.tag for s in ants)} x {len(sr.LENS_IDS)} lenses")
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    reader, ants = setup(a)
    items = pick(sr.load_items(QUESTIONS), 4)[2:]   # one LSAT analytical and one LSAT logical question (both have passages)
    d = out_dir(a)
    runner = sr.Runner(provider(a), d / "cache.jsonl", concurrency=a.concurrency, check_names=not a.loose_names)
    t0 = time.monotonic()
    try:
        got = asyncio.run(_run(a, items, runner, reader, ants))
    except sr.BrokenRun as exc:
        print(f"STOPPED: {exc}")
        return 2
    wall = time.monotonic() - t0
    rows = [json.loads(l) for l in (d / "cache.jsonl").read_text().splitlines() if l.strip()]
    hashes = set()
    per: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r.get("elapsed_s") and r["request_hash"] not in hashes:
            hashes.add(r["request_hash"])
            per[r["model"]].append(r["elapsed_s"])
    first = got[0] if got else None
    if first:
        print(f"\nReader notes for {first['id']} (leak flag: {first['leak']}):\n{first['reader_notes']}\n")
        print(f"Reader's own answer: {first['reader']}; ants alone: {first['alone']}; ants with notes: {first['both']}")
    full = sr.plan(240, ants)
    if not per:
        print("No timings recorded (a dry run, or every call came from the cache; delete cache.jsonl in the output "
              "folder to time again).")
        return 0
    print("\nSeconds per call on this machine (median):")
    est = 0.0
    for spec in (reader, *ants):
        if per.get(spec.tag):
            sec = st.median(per[spec.tag])
            calls = full["reader_calls"] if spec is reader else full["ant_calls"] // len(ants)
            est += sec * calls / max(1, min(a.concurrency, 2))
            print(f"  {spec.tag:18s} {sec:6.1f}s")
    print(f"\nRough time for all 240 questions: {est / 3600:.1f} hours; for --limit 60: {est / 3600 / 4:.1f} hours. "
          f"(This bench took {wall:.0f}s, model loading included.)")
    print("Read the notes above: they should explain the question without picking an option. If they look right, "
          "start with `run --limit 60`.")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    reader, ants = setup(a)
    items = pick(sr.load_items(QUESTIONS), a.limit)
    d = out_dir(a)
    runner = sr.Runner(provider(a), d / "cache.jsonl", concurrency=a.concurrency, check_names=not a.loose_names)
    p = sr.plan(len(items), ants)
    print(f"{p['items']} questions, {p['calls']} calls ({len(runner.cache)} already done). Output: {d}", flush=True)
    t0 = time.monotonic()
    try:
        got = asyncio.run(_run(a, items, runner, reader, ants))
    except sr.BrokenRun as exc:
        print(f"STOPPED: {exc}")
        return 2
    meta = {"reader": reader.tag, "ants": [[s.tag, s.family, lens] for s, lens in sr.ant_roster(ants)],
            "families": [s.family for s, _ in sr.ant_roster(ants)], "items": len(got), "base_url": a.base_url,
            "dry_run": bool(a.dry_run), "new_calls": runner.calls, "failed_calls": runner.failed,
            "seconds": round(time.monotonic() - t0)}
    (d / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in got))
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"Done: {len(got)} questions, {runner.calls} new calls ({runner.failed} unusable) in "
          f"{meta['seconds'] / 60:.0f} min. Next: `uv run python scripts/shared_read.py score"
          f"{f' --out {a.out}' if a.out else ' --dry-run' if a.dry_run else ''}`.")
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    d = out_dir(a)
    rows_path, meta_path = d / "rows.jsonl", d / "meta.json"
    if not rows_path.exists():
        print(f"No rows at {rows_path}: run first.")
        return 2
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    meta = json.loads(meta_path.read_text())
    key = {r["id"]: r["label"] for r in map(json.loads, ANSWERS.read_text().splitlines()) if r}
    res = sr.score(rows, key, meta["families"])
    res["meta"] = meta
    changed = frozen_changes()
    res["frozen_plan_unchanged"] = not changed
    if changed:
        print("WARNING: the test changed since it was frozen (" + ", ".join(changed) + "). This score is a new test, "
              "not the preregistered one.")
    (d / "score.json").write_text(json.dumps(res, indent=1) + "\n")
    m = res["all"]
    print(f"{res['n']} questions; usable answers {res['usable_share']:.0%}; notes that leaked an option "
          f"{res['leak_share']:.0%}")
    print(f"Reader ({meta['reader']}) answering alone: {m['reader_alone']:.1%}")
    for arm in sr.ARMS:
        s = m["arms"][arm]
        print(f"  ants {arm:5s}: vote {s['vote']:.1%}, average ant {s['ant_mean']:.1%}, "
              f"mistakes alike {s['error_alikeness']:+.2f}, same wrong pick {s['same_wrong']:.0%}")
    b = m["both_vs_alone"]
    print(f"Notes + question vs alone (vote): {b['gain']:+.1%} (95% range {b['ci95'][0]:+.1%} to {b['ci95'][1]:+.1%})")
    r = m["alikeness_rise"]
    print(f"Rise in alike mistakes: {r['rise']:+.3f} (95% range {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f})")
    print("Lift per family (score with notes / score alone): "
          + ", ".join(f"{f} {v['both']}" for f, v in m["lift_by_family"].items()))
    print("Verdict: " + json.dumps(res["verdict"]))
    print(f"Full numbers: {d / 'score.json'}")
    return 0


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "check", "plan", "bench", "run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default=OLLAMA)
        p.add_argument("--reader", default="", help="tag=family, e.g. qwen3:8b=qwen")
        p.add_argument("--ants", default="", help="comma-separated tag=family list")
        p.add_argument("--out", default="", help="output folder (default docs/research/shared-read-20261030/run)")
        p.add_argument("--dry-run", action="store_true", help="fake model, no server")
        p.add_argument("--concurrency", type=int, default=2)
        p.add_argument("--timeout", type=float, default=600.0)
        p.add_argument("--loose-names", action="store_true", help="don't check the served model name (MLX)")
        p.add_argument("--limit", type=int, default=0, help="this many questions, taken evenly from the 4 tasks")
    a = ap.parse_args(argv)
    return {"freeze": cmd_freeze, "check": cmd_check, "plan": cmd_plan, "bench": cmd_bench, "run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
