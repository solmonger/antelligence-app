"""The shared-read test on your own machine: one bigger reader, eight small mixed ants. See docs/plans/2026-10-30-shared-read.md.

Free: every call goes to a local model server (Ollama by default, at http://localhost:11434/v1). Nothing is sent
anywhere else, and the answer file is opened only by ``score``. The models are listed in
``docs/research/shared-read-20261030/models.json``; change them there, not here. To have Claude Code on the Mac run
all of this with memory guards, paste ``docs/research/shared-read-20261030/CLAUDE_CODE_PROMPT.md`` into it.

Steps, in this order::

    # once: get the five models (about 30 GB of downloads in all)
    ollama pull qwen3.8:27b && ollama pull gemma4:e2b && ollama pull qwen3.5:4b \\
      && ollama pull granite4.2:3b && ollama pull lfm3:1.7b

    uv run python scripts/shared_read.py check          # server up, models there, which ones think (no model call)
    uv run python scripts/shared_read.py bench          # 2 questions through everything: speed, thinking, the notes
    uv run python scripts/shared_read.py run --limit 60 # a quick pass: 15 questions from each of the 4 tasks
    uv run python scripts/shared_read.py run            # all 240 (resumes; nothing done is asked again)
    uv run python scripts/shared_read.py score          # only now reads the answers, prints the verdict
    uv run python scripts/shared_read.py explore        # other swarm shapes from the same answers (exploratory)

On Ollama the script unloads each model when its phase ends and won't load the next one unless there is room for it
plus ``--headroom-gb`` (default 6) of free memory; if not, it stops and says so, and a rerun resumes. ``--dry-run``
on ``bench`` and ``run`` uses a fake model, for checking the plumbing without a server. ``--models`` reads another
model file.

MLX instead of Ollama (how run 1 ran): ``mlx_lm.server`` can't unload a model, so run 1 put a small Ollama-shaped
stand-in on port 11434 in front of it. The stand-in passes ``/v1`` calls through, lists the models at ``/v1/models``,
and restarts ``mlx_lm.server`` when it gets the unload call (``POST /api/generate`` with ``keep_alive: 0``), so the
memory guard sees the memory come back. Without a stand-in, point ``--base-url`` at ``mlx_lm.server`` itself
(``http://localhost:8080/v1``); the guard then can't free memory between models, so keep ``--headroom-gb`` and
restart the server yourself between models if memory gets tight. MLX turns thinking off through the chat template:
``"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}`` in the model file.

A reply that came back in the wrong form is kept and used again on a rerun instead of being asked again (each call
has a fixed seed, so it would mostly come back the same); ``--retry-unusable`` asks them again. ``--max-fail-streak``
sets how many unusable replies in a row stop the run (default 8; run 1 needed 100 for Gemma 4 E2B's LaTeX).
``--seed-cache`` reads another run's cache too, so ``run --parser 2 --offline --seed-cache .../run/cache.jsonl``
re-reads run 1's replies with parser 2 without asking any model (output in ``run-p2/``, never over run 1's; then
``score --parser 2`` and ``explore --parser 2``).
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
THINK_TOKENS_ANT = 350      # a JSON answer with a 60-word reason is ~100 tokens; far more means hidden reasoning
THINK_TOKENS_READER = 700   # 200-word notes are ~270 tokens
FROZEN = {"questions": QUESTIONS, "answers": ANSWERS, "harness": REPO / "antelligence/experiments/shared_read.py",
          "script": Path(__file__).resolve(), "plan": REPO / "docs/plans/2026-10-30-shared-read.md",
          "models": sr.MODELS_FILE}


def fingerprint() -> Dict[str, Any]:
    import hashlib

    return {"files": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in FROZEN.items()},
            "pass": sr.PASS, "min_usable": sr.MIN_USABLE, "reader": sr.READER.tag,
            "ants": [s.tag for s in sr.ANTS], "lenses": list(sr.LENS_IDS), "arms": list(sr.ARMS),
            "temperature": sr.TEMPERATURE, "parser": sr.PARSER}


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


def config(a: argparse.Namespace) -> sr.Config:
    cfg = sr.load_config(Path(a.models)) if getattr(a, "models", "") else sr.CONFIG
    parser = getattr(a, "parser", None)
    if parser is not None and parser != cfg.parser:
        if parser not in sr.PARSERS:
            raise SystemExit(f"--parser must be one of {sr.PARSERS}")
        cfg = sr.Config(cfg.reader, cfg.ants, parser)
    return cfg


def setup(a: argparse.Namespace) -> Tuple[sr.ModelSpec, Tuple[sr.ModelSpec, ...]]:
    cfg = config(a)
    return cfg.reader, cfg.ants


def make_runner(a: argparse.Namespace, d: Path) -> sr.Runner:
    return sr.Runner(provider(a), d / "cache.jsonl", concurrency=a.concurrency, check_names=not a.loose_names,
                     max_fail_streak=10 ** 9 if a.offline else a.max_fail_streak, retry_unusable=a.retry_unusable,
                     seed_paths=[Path(p) for p in a.seed_cache or ()])


class OfflineModel:
    """``--offline``: answers nothing. With ``--seed-cache`` a run then only re-reads replies it already has; any call
    it would have to ask is left unanswered (counted as unusable) instead of reaching a model."""

    async def complete(self, req: ChatRequest) -> ChatResponse:
        from antelligence.providers.base import ProviderError

        raise ProviderError("offline: not asked")


# ---------------------------------------------------------------- memory guard
#
# Only one model is ever asked at a time. On Ollama the script also unloads each model when its phase ends and checks
# there is room before the next one loads, so a 36 GB Mac never holds two big models or starts swapping.

class MemoryTight(sr.BrokenRun):
    """Not enough free memory to load the next model safely. Nothing is lost: free some memory and rerun."""


def free_gb() -> Optional[float]:
    """Memory the system can hand out now, in GB: macOS from ``vm_stat`` (free + inactive + speculative + purgeable
    pages), Linux from MemAvailable. None when it can't tell."""
    import platform
    import re
    import subprocess

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


def ollama_root(base_url: str) -> Optional[str]:
    """The Ollama server's own address (for its unload call), or None when the base URL isn't Ollama's port."""
    root = base_url.rstrip("/")
    root = root[: -len("/v1")] if root.endswith("/v1") else root
    return root if root.rsplit(":", 1)[-1] == "11434" else None


def unload(a: argparse.Namespace, spec: sr.ModelSpec) -> None:
    """Ask Ollama to drop the model from memory now (keep_alive 0) instead of holding it for minutes."""
    root = ollama_root(a.base_url)
    if a.dry_run or getattr(a, "offline", False) or root is None:
        return
    import httpx

    try:
        httpx.post(f"{root}/api/generate", json={"model": spec.tag, "keep_alive": 0}, timeout=60, trust_env=False)
    except httpx.HTTPError:
        pass  # a model that wasn't loaded, or a server that went away: the next call will say so


def guard(a: argparse.Namespace, spec: sr.ModelSpec) -> None:
    """Stop before loading ``spec`` unless there is room for it plus ``--headroom-gb``."""
    if a.dry_run or getattr(a, "offline", False) or not a.headroom_gb:
        return
    have = free_gb()
    need = spec.gb + a.headroom_gb
    if have is not None and have < need:
        raise MemoryTight(f"Only {have:.1f} GB free before loading {spec.tag} (needs about {spec.gb:.0f} GB plus "
                          f"{a.headroom_gb:.0f} GB headroom). Close other apps or unload other models "
                          "(`ollama ps`, `ollama stop <model>`), then rerun; finished calls are kept.")


def provider(a: argparse.Namespace) -> Any:
    if a.dry_run:
        return FakeModel()
    if getattr(a, "offline", False):
        return OfflineModel()
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
    """The run folder: ``run/`` (parser 1), ``run-p2/`` (parser 2), ``dry-run/`` for a fake model, or ``--out``."""
    parser = config(a).parser
    name = "dry-run" if getattr(a, "dry_run", False) else "run" if parser == 1 else f"run-p{parser}"
    d = Path(a.out) if a.out else OUT / name
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
    parser = config(a).parser
    guard(a, reader)
    reading = await sr.run_reader(runner, items, reader, on_done=tick, parser=parser)
    unload(a, reader)

    def model(spec: sr.ModelSpec) -> None:
        guard(a, spec)
        if not quiet:
            print(f"Phase 2: ant model {spec.tag} answers alone, from notes, and both "
                  f"({time.monotonic() - start:.0f}s)", flush=True)

    answers = await sr.run_ants(runner, items, reading, ants, on_model=model, after_model=lambda s: unload(a, s),
                                parser=parser)
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
    root = ollama_root(a.base_url)
    specs = {s.tag: s for s in (reader, *ants)}
    for t in need:
        extra = ""
        if root and t not in missing:
            try:  # Ollama's own model card: does this model think, and what will we send to stop it?
                caps = httpx.post(f"{root}/api/show", json={"model": t}, timeout=30, trust_env=False).json()
                if "thinking" in (caps.get("capabilities") or []):
                    how = specs[t].extra_body or ({"suffix": specs[t].suffix} if specs[t].suffix else None)
                    extra = f"  (a thinking model; we send {json.dumps(how) if how else 'nothing to turn it off'})"
            except (httpx.HTTPError, ValueError):
                pass
        print(f"  {'ok     ' if t not in missing else 'MISSING'} {t}{extra}")
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
    runner = make_runner(a, d)
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
    toks: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        if r.get("elapsed_s") and r["request_hash"] not in hashes:
            hashes.add(r["request_hash"])
            per[r["model"]].append(r["elapsed_s"])
            toks[r["model"]].append(int(r.get("completion_tokens") or 0))
    first = got[0] if got else None
    if first:
        print(f"\nReader notes for {first['id']} (leak flag: {first['leak']}):\n{first['reader_notes']}\n")
        print(f"Reader's own answer: {first['reader']}; ants alone: {first['alone']}; ants with notes: {first['both']}")
    full = sr.plan(240, ants)
    if not per:
        print("No timings recorded (a dry run, or every call came from the cache; delete cache.jsonl in the output "
              "folder to time again).")
        return 0
    print("\nPer call on this machine (median seconds, median tokens written):")
    est = 0.0
    thinking = []
    for spec in (reader, *ants):
        if per.get(spec.tag):
            sec, tok = st.median(per[spec.tag]), st.median(toks[spec.tag])
            calls = full["reader_calls"] if spec is reader else full["ant_calls"] // len(ants)
            est += sec * calls / max(1, min(a.concurrency, 2))
            long = tok > (THINK_TOKENS_READER if spec is reader else THINK_TOKENS_ANT)
            if long:
                thinking.append(spec.tag)
            print(f"  {spec.tag:18s} {sec:6.1f}s {tok:6.0f} tokens{'   <- LONG: probably still thinking' if long else ''}")
    print(f"\nRough time for all 240 questions: {est / 3600:.1f} hours; for --limit 60: {est / 3600 / 4:.1f} hours. "
          f"(This bench took {wall:.0f}s, model loading included.)")
    if thinking:
        print(f"WARNING: {', '.join(thinking)} wrote far more than a short answer needs, so its hidden reasoning is "
              "probably still on. That makes the run slow and can cut answers off. Change how models.json turns "
              "thinking off for it (see the notes there), rerun bench, and freeze again before the real run.")
        return 3
    print("Read the notes above: they should explain the question without picking an option. If they look right, "
          "start with `run --limit 60`.")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    reader, ants = setup(a)
    items = pick(sr.load_items(QUESTIONS), a.limit)
    d = out_dir(a)
    runner = make_runner(a, d)
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
            "reused_unusable": runner.reused, "parser": config(a).parser, "max_fail_streak": a.max_fail_streak,
            "retry_unusable": bool(a.retry_unusable), "seconds": round(time.monotonic() - t0)}
    (d / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in got))
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"Done: {len(got)} questions, {runner.calls} new calls ({runner.failed} unusable; {runner.reused} kept "
          f"unusable replies used again) in "
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
    if meta.get("parser", 1) != sr.PARSER:
        changed.append(f"replies read with parser {meta['parser']}")
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


def cmd_explore(a: argparse.Namespace) -> int:
    d = out_dir(a)
    rows_path = d / "rows.jsonl"
    if not rows_path.exists():
        print(f"No rows at {rows_path}: run first.")
        return 2
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    meta = json.loads((d / "meta.json").read_text())
    key = {r["id"]: r["label"] for r in map(json.loads, ANSWERS.read_text().splitlines()) if r}
    res = sr.explore(rows, key, meta["families"])
    res["meta"] = meta
    (d / "explore.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"EXPLORATORY (chosen after run 1, not preregistered): other swarm shapes from the same {res['n']} "
          "questions' answers, no new calls.")
    for name, v in res["shapes"].items():
        vs = v.get("vs_reader")
        cmp_ = f"   vs reader {vs['gain']:+.1%} ({vs['ci95'][0]:+.1%} to {vs['ci95'][1]:+.1%})" if vs else ""
        alike = f"   alike {v['error_alikeness']:+.2f}" if "error_alikeness" in v else ""
        print(f"  {name:32s} {v['accuracy']:6.1%}{cmp_}{alike}")
    print(f"Full numbers: {d / 'explore.json'}")
    return 0


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "check", "plan", "bench", "run", "score", "explore"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default=OLLAMA)
        p.add_argument("--models", default="", help="model config JSON (default: the frozen models.json)")
        p.add_argument("--headroom-gb", type=float, default=6.0,
                       help="free memory to keep on top of each model before loading it (0 turns the guard off)")
        p.add_argument("--out", default="", help="output folder (default docs/research/shared-read-20261030/run)")
        p.add_argument("--dry-run", action="store_true", help="fake model, no server")
        p.add_argument("--concurrency", type=int, default=2)
        p.add_argument("--timeout", type=float, default=600.0)
        p.add_argument("--loose-names", action="store_true", help="don't check the served model name (MLX)")
        p.add_argument("--limit", type=int, default=0, help="this many questions, taken evenly from the 4 tasks")
        p.add_argument("--max-fail-streak", type=int, default=8, help="unusable replies in a row that stop the run")
        p.add_argument("--retry-unusable", action="store_true",
                       help="ask again calls whose kept reply was unusable (default: use the kept reply)")
        p.add_argument("--seed-cache", action="append", default=[],
                       help="another run's cache.jsonl to read replies from (read only; repeatable)")
        p.add_argument("--parser", type=int, default=None, help="how replies are read (default: the model file's)")
        p.add_argument("--offline", action="store_true",
                       help="ask no model: only reuse replies from the caches (for re-reading a run)")
    a = ap.parse_args(argv)
    return {"freeze": cmd_freeze, "check": cmd_check, "plan": cmd_plan, "bench": cmd_bench, "run": cmd_run,
            "score": cmd_score, "explore": cmd_explore}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
