"""The shared-read test: does one bigger "reader" model help a swarm of small mixed ants, or just make them think alike?

Why: the Swarm Telepathy paper (arXiv 2610.15095, MIT CSAIL and the MIT-IBM lab) hands one model's reading of a
document to another model, and its best pair had an 8B reader lift a 3B receiver above its own normal score ("hire
the thinker, rent the talker"). The paper only ever tested one receiver at a time. A swarm has many, and our own
results say ants win by thinking alone (docs/plans/2026-10-29-swarm-at-scale.md). So before training any translator,
this asks the cheap question in plain text: if every ant gets the same reader's notes, does the swarm get smarter, or
do the ants just start making the same mistakes?

Preregistered in ``docs/plans/2026-10-30-shared-read.md``. Everything runs on local open models (Ollama or any
OpenAI-compatible server on the user's own machine), so it costs nothing.

Per exam question (the frozen 240 AGIEval questions of the fair test):

* the **reader** (a bigger model; the models are set in ``docs/research/shared-read-20261030/models.json``, by
  default Qwen3.8 27B, about 18 GB at 4-bit) writes short notes once: what is asked, the key facts and rules, the
  traps. It is told never to say which option is right; notes that name a winning option anyway are flagged as leaks
  and scored both with and without.
* the reader also answers the question itself once (``reader``): the "thinker alone" line the swarm has to beat.
* **8 ants** = 4 small models from 4 families (by default Gemma 4 E2B, Qwen3.5 4B, Granite 4.2 3B, LFM3 1.7B)
  x 2 ways of looking (lenses), each answering on its own, in 3 arms:

  - ``alone``: the ant reads the full question itself (our current default, think alone).
  - ``notes``: the ant gets the reader's notes in place of the passage, plus the question and options. This is the
    text version of the paper's set-up: the small model never reads the document, only the reader's reading of it.
    (AQuA maths items have no passage, so for them ``notes`` and ``both`` differ only in order.)
  - ``both``: the full question plus the reader's notes.

Ants never see each other. The swarm answer is a plain vote. Calls run one model at a time (all the reader's calls,
then each ant model's), so only one model needs to be in memory at a time.

Everything here is pure (prompts, parsing, combining, scoring). Calls go through any object with the provider
contract (``complete(ChatRequest) -> ChatResponse``), so tests use a fake model and no network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import statistics as st
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.experiments import fair_test as ft
from antelligence.providers.base import ChatRequest, ChatResponse, ProviderError, ProviderUnavailable
from antelligence.providers.redact import scrub

# --------------------------------------------------------------------- set-up


@dataclass(frozen=True)
class ModelSpec:
    tag: str                      # the name the local server knows it by, e.g. "qwen3:4b"
    family: str                   # model family, for the per-family lift
    gb: float = 0.0               # roughly how much memory it takes once loaded (for the memory guard)
    suffix: str = ""              # text appended to every user message, e.g. Qwen's "/no_think" switch
    extra_body: Optional[Mapping[str, Any]] = None   # extra request keys, e.g. {"reasoning_effort": "none"}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "ModelSpec":
        unknown = set(d) - {"tag", "family", "gb", "suffix", "extra_body", "note", "about"}
        if unknown:
            raise ValueError(f"unknown model keys {sorted(unknown)} in {d.get('tag')!r}")
        return cls(str(d["tag"]), str(d["family"]), float(d.get("gb") or 0.0), str(d.get("suffix") or ""),
                   dict(d["extra_body"]) if d.get("extra_body") else None)


# The models live in a config file, so they can change without touching code. The file is part of the frozen plan.
MODELS_FILE = Path(__file__).resolve().parents[2] / "docs/research/shared-read-20261030/models.json"


def load_models(path: Path = MODELS_FILE) -> Tuple[ModelSpec, Tuple[ModelSpec, ...]]:
    """(reader, ants) from a JSON file: {"reader": {...}, "ants": [{...}, ...]}. Ants must come from different
    families, because the test is about a mixed swarm."""
    cfg = json.loads(Path(path).read_text())
    reader = ModelSpec.from_json(cfg["reader"])
    ants = tuple(ModelSpec.from_json(a) for a in cfg["ants"])
    if not ants:
        raise ValueError("no ants in the model file")
    if len({a.family for a in ants}) != len(ants):
        raise ValueError("each ant model must come from a different family")
    if len({a.tag for a in ants} | {reader.tag}) != len(ants) + 1:
        raise ValueError("the reader and every ant must be different models")
    return reader, ants


READER, ANTS = load_models()
LENS_IDS = (0, 1)                 # the fair test's first two exam lenses: work it out / rule options out
ARMS = ("alone", "notes", "both")
TEMPERATURE = 0.7
ANT_MAX_TOKENS = 1200
READER_MAX_TOKENS = 1500
NOTES_WORDS = 200

READER_SYSTEM = ("You are the reader for a team. You read a question once and write notes so that others can answer "
                 "it well. You never say which option is right.")
NOTES_ASK = (f"Write notes for the team, at most {NOTES_WORDS} words: what the question asks, the key facts, rules or "
             "numbers, and any trap to watch for. Do not say which option is correct, do not rule options in or out "
             "by letter, and do not give a final answer. Reply with the notes only.")

# ------------------------------------------------------------------ questions


@dataclass(frozen=True)
class Item:
    id: str
    group: str                    # the AGIEval task
    passage: str
    stem: str
    options: Tuple[str, ...]

    @property
    def full(self) -> str:
        """Exactly what the fair test shows a model (fair_test._render)."""
        parts = [self.passage] if self.passage else []
        parts += [self.stem, "Options:\n" + "\n".join(self.options)]
        return "\n\n".join(parts)


def load_items(path: Path) -> List[Item]:
    """The fair test's exam questions. Refuses a file that carries answers."""
    ft.load_questions(path, "exam")  # same checks: no answer fields, no duplicate ids
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.append(Item(str(r["id"]), str(r.get("task") or ""), str(r.get("passage") or "").strip(),
                            str(r["question"]).strip(), tuple(str(o).strip() for o in r["options"])))
    return out


# -------------------------------------------------------------------- prompts

def _user(spec: ModelSpec, text: str) -> str:
    return f"{text} {spec.suffix}" if spec.suffix else text


def notes_messages(item: Item, spec: ModelSpec = READER) -> Tuple[Dict[str, str], ...]:
    return ({"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": _user(spec, f"{item.full}\n\n{NOTES_ASK}")})


def reader_answer_messages(item: Item, spec: ModelSpec = READER) -> Tuple[Dict[str, str], ...]:
    return ({"role": "system", "content": ft.SYSTEM},
            {"role": "user", "content": _user(spec, f"{item.full}\n\n{ft._format('exam')}")})


def ant_messages(item: Item, arm: str, lens: int, spec: ModelSpec, notes: Optional[str] = None
                 ) -> Tuple[Dict[str, str], ...]:
    """One ant's prompt. ``alone`` is the fair test's lens prompt exactly; the other arms add the reader's notes."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if arm != "alone" and not notes:
        raise ValueError(f"arm {arm!r} needs the reader's notes")
    tail = f"How to think about it: {ft.LENSES['exam'][lens]}\n\n{ft._format('exam')}"
    opts = "Options:\n" + "\n".join(item.options)
    if arm == "alone":
        body = item.full
    elif arm == "notes":
        lead = ("Notes from a reader who read the full passage (you do not see the passage):"
                if item.passage else "Notes from a reader who read this question:")
        body = f"{lead}\n{notes}\n\n{item.stem}\n\n{opts}"
    else:
        body = f"{item.full}\n\nNotes from a reader who read this question:\n{notes}"
    return ({"role": "system", "content": ft.SYSTEM}, {"role": "user", "content": _user(spec, f"{body}\n\n{tail}")})


# -------------------------------------------------------------------- parsing

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_OBJ = re.compile(r"\{[^{}]*\}")


def strip_think(text: str) -> str:
    """Drop hidden reasoning blocks (Qwen3 and others put them inline as <think>...</think>)."""
    text = _THINK.sub("", text or "")
    if "</think>" in text.lower():  # an opening tag the server already removed
        text = re.split(r"</think>", text, flags=re.I)[-1]
    return text.strip()


def parse_answer(text: str) -> Tuple[Optional[str], str]:
    """(letter A-E or None, reason). Reads the whole reply first, then each flat JSON object from the last one back,
    so stray braces in a reason or a leftover reasoning block don't hide the answer."""
    clean = strip_think(text)
    ans, reason = ft.parse_reply("exam", clean)
    if ans is not None:
        return ans, reason
    for m in reversed(_OBJ.findall(clean)):
        ans, reason = ft.parse_reply("exam", m)
        if ans is not None:
            return ans, reason
    return None, ""


_L = r"(?<![A-Za-z])\(?(?-i:[A-E])\)?(?![A-Za-z])"   # an option letter, capital, standing alone or in brackets
_LEAK = re.compile(
    r"(answer|correct|right|best)\s+(option\s+|choice\s+)?(is|would be|=|:)\s*" + _L
    + r"|\b(option|choice)\s*" + _L + r"\s+(is|would be)\s+(the\s+)?(correct|right|best|answer)"
    + r"|" + _L + r"\s+is\s+(the\s+)?(correct|right|best)\b"
    + r"|\b(eliminate|rule out|ruled out)\s+(option\s+)?" + _L,
    re.I)


def leaks(notes: str) -> bool:
    """Do the reader's notes name a winning (or losing) option by letter, against the instructions?"""
    return bool(_LEAK.search(notes or ""))


def clean_notes(text: str) -> str:
    return strip_think(text)[:4000]


# ------------------------------------------------------------------- running

def seed(qid: str, role: str, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"shared-read:{qid}:{role}:{index}".encode()).digest()[:4], "big")


def make_request(spec: ModelSpec, messages: Tuple[Dict[str, str], ...], qid: str, role: str, index: int,
                 max_tokens: int) -> ChatRequest:
    return ChatRequest(model=spec.tag, messages=messages, max_tokens=max_tokens, temperature=TEMPERATURE,
                       seed=seed(qid, role, index), extra_body=spec.extra_body)


class BrokenRun(RuntimeError):
    """Too many unusable replies in a row: the run measures the plumbing, not the swarm, so it stops and says why."""


class MissingModel(BrokenRun):
    """The local server doesn't have a model the test needs. Pull it, then rerun (the cache keeps what is done)."""


_MISSING = re.compile(r"model ['\"]?[^'\"]*['\"]? not found|not found, try pulling|no such model|model .* does not exist",
                      re.I)


class Runner:
    """Asks one call at a time per slot, caching each usable reply by request hash in ``cache_path`` (JSONL) so a
    stopped run resumes where it was. A failed call is logged but never reused: a rerun asks again. After
    ``max_fail_streak`` unusable replies in a row the run stops with :class:`BrokenRun`."""

    def __init__(self, provider: Any, cache_path: Path, *, max_fail_streak: int = 8, concurrency: int = 2,
                 backoff: Sequence[float] = (2, 5, 10), check_names: bool = True) -> None:
        self.provider, self.cache_path, self.check_names = provider, Path(cache_path), check_names
        self.max_fail_streak, self.backoff = max_fail_streak, tuple(backoff)
        self.sem = asyncio.Semaphore(concurrency)
        self.cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    if not row.get("error"):
                        self.cache[row["request_hash"]] = row
        self.calls = self.failed = self.streak = 0
        self.seconds = 0.0
        self.reasons: List[str] = []
        self.broken: Optional[str] = None
        self._lock = asyncio.Lock()

    async def ask(self, req: ChatRequest, usable: Callable[[str], bool]) -> Dict[str, Any]:
        h = req.request_hash
        if h in self.cache:
            return self.cache[h]
        if self.broken:
            raise BrokenRun(self.broken)
        async with self.sem:
            try:
                resp: ChatResponse = await self._complete(req)
                row = {"request_hash": h, "model": req.model, "content": resp.content, "served_model": resp.model,
                       "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                       "finish_reason": resp.finish_reason, "elapsed_s": round(resp.elapsed_s, 3), "error": None}
                if self.check_names and not same_model(resp.model, req.model):
                    row.update(error="ServedModelMismatch", reason=f"asked for {req.model!r}, served {resp.model!r}")
                elif not usable(resp.content):
                    row.update(error="Unusable", reason=f"reply not in the asked form: {resp.content[-160:]!r}")
            except ProviderUnavailable:
                raise  # an outage that outlived the retries: stop, cache nothing, a rerun asks again
            except ProviderError as exc:
                msg = scrub(str(exc))[:300]
                if _MISSING.search(msg) or getattr(exc, "http_status", None) == 404:
                    self.broken = (f"The local server has no model {req.model!r}. Pull it first "
                                   f"(for Ollama: `ollama pull {req.model}`), then rerun; finished calls are kept.")
                    raise MissingModel(self.broken) from None
                row = {"request_hash": h, "model": req.model, "content": "", "error": type(exc).__name__,
                       "reason": msg}
        async with self._lock:
            self.calls += 1
            self.seconds += row.get("elapsed_s") or 0.0
            with self.cache_path.open("a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            if row["error"] is None:
                self.cache[h] = row
                self.streak = 0
            else:
                self.failed += 1
                self.streak += 1
                if len(self.reasons) < 5:
                    self.reasons.append(f"{req.model}: {row.get('reason') or row['error']}")
                if self.streak >= self.max_fail_streak and not self.broken:
                    self.broken = (f"{self.streak} replies in a row were unusable. First reasons: "
                                   + " | ".join(dict.fromkeys(self.reasons)))
            if self.broken:
                raise BrokenRun(self.broken)
        return row

    async def _complete(self, req: ChatRequest) -> ChatResponse:
        for wait in self.backoff:
            try:
                return await self.provider.complete(req)
            except ProviderUnavailable:
                await asyncio.sleep(wait)
        return await self.provider.complete(req)


def same_model(served: str, asked: str) -> bool:
    """Ollama may report ``phi4-mini:latest`` for ``phi4-mini``; anything else must match exactly (case aside)."""
    def norm(name: str) -> str:
        name = (name or "").strip().lower()
        return name[: -len(":latest")] if name.endswith(":latest") else name
    return norm(served) == norm(asked)


def _usable_answer(text: str) -> bool:
    return parse_answer(text)[0] is not None


def _usable_notes(text: str) -> bool:
    return len(clean_notes(text).split()) >= 5


def _answer(row: Mapping[str, Any]) -> Optional[str]:
    return None if row.get("error") else parse_answer(row["content"])[0]


async def run_reader(runner: Runner, items: Sequence[Item], reader: ModelSpec = READER,
                     on_done: Optional[Callable[[str], None]] = None) -> Dict[str, Dict[str, Any]]:
    """Phase 1 (one model loaded): the reader's notes and its own answer for every item."""
    out: Dict[str, Dict[str, Any]] = {}

    async def one(item: Item) -> None:
        n = await runner.ask(make_request(reader, notes_messages(item, reader), item.id, "notes", 0,
                                          READER_MAX_TOKENS), _usable_notes)
        a = await runner.ask(make_request(reader, reader_answer_messages(item, reader), item.id, "reader", 0,
                                          ft.MAX_TOKENS), _usable_answer)
        notes = clean_notes(n["content"]) if not n.get("error") else ""
        out[item.id] = {"notes": notes, "leak": leaks(notes), "reader": _answer(a)}
        if on_done:
            on_done(item.id)

    await asyncio.gather(*(one(i) for i in items))
    return out


def ant_roster(ants: Sequence[ModelSpec] = ANTS, lenses: Sequence[int] = LENS_IDS) -> List[Tuple[ModelSpec, int]]:
    """Every ant as (model, lens), in a fixed order: the column order of every answer list."""
    return [(spec, lens) for spec in ants for lens in lenses]


async def run_ants(runner: Runner, items: Sequence[Item], reading: Mapping[str, Mapping[str, Any]],
                   ants: Sequence[ModelSpec] = ANTS, lenses: Sequence[int] = LENS_IDS,
                   on_model: Optional[Callable[[ModelSpec], None]] = None,
                   after_model: Optional[Callable[[ModelSpec], None]] = None
                   ) -> Dict[str, Dict[str, List[Optional[str]]]]:
    """Phase 2, one ant model at a time: every ant answers every item in every arm. An item whose notes failed gets
    no ``notes``/``both`` answers (None), which the usable-share check counts."""
    roster = ant_roster(ants, lenses)
    out: Dict[str, Dict[str, List[Optional[str]]]] = {i.id: {arm: [None] * len(roster) for arm in ARMS}
                                                       for i in items}
    for spec in ants:
        if on_model:
            on_model(spec)
        jobs = []
        for col, (s, lens) in enumerate(roster):
            if s is not spec:
                continue
            for item in items:
                notes = reading.get(item.id, {}).get("notes") or ""
                for arm in ARMS:
                    if arm != "alone" and not notes:
                        continue
                    req = make_request(spec, ant_messages(item, arm, lens, spec, notes or None), item.id,
                                       f"{arm}:{spec.tag}", lens, ANT_MAX_TOKENS)
                    jobs.append((item.id, arm, col, req))

        async def one(job):
            qid, arm, col, req = job
            out[qid][arm][col] = _answer(await runner.ask(req, _usable_answer))

        await asyncio.gather(*(one(j) for j in jobs))
        if after_model:
            after_model(spec)
    return out


def rows(items: Sequence[Item], reading: Mapping[str, Mapping[str, Any]],
         answers: Mapping[str, Mapping[str, List[Optional[str]]]]) -> List[Dict[str, Any]]:
    """One saved row per item: the reader's notes (``reader_notes``), their leak flag, the reader's own answer, and
    every ant's answer in each arm (``alone``, ``notes``, ``both``; one list entry per ant, in roster order)."""
    return [{"id": i.id, "group": i.group, "reader_notes": reading[i.id]["notes"], "leak": reading[i.id]["leak"],
             "reader": reading[i.id]["reader"], **{arm: answers[i.id][arm] for arm in ARMS}}
            for i in items if i.id in reading and i.id in answers]


# -------------------------------------------------------------------- scoring

# Preregistered (docs/plans/2026-10-30-shared-read.md). Changing these after a run voids the test.
PASS = {
    "helps_min_gain": 0.02,          # both-vote minus alone-vote accuracy, with the 95% interval above 0
    "alike_min_rise": 0.05,          # rise in how alike the ants' mistakes are (mean pairwise error correlation)
    "beats_reader_min_gain": 0.02,   # both-vote minus the reader answering alone
    "max_leak_share": 0.10,          # above this, the notes gave answers away and the main result is not a fair read
}
MIN_USABLE = 0.9


def _right(a: Optional[str], truth: str) -> float:
    return float(a is not None and a == truth)


def error_alikeness(answer_lists: Sequence[Sequence[Optional[str]]], truths: Sequence[str]) -> float:
    """Mean over ant pairs of the correlation between their "got it wrong" records across items (phi coefficient).
    0 means the ants fail on unrelated items; 1 means they always fail together. Pairs where either ant is always
    right or always wrong carry no information and are skipped."""
    n_ants = len(answer_lists[0]) if answer_lists else 0
    wrong = [[1.0 - _right(ans[k], t) for ans, t in zip(answer_lists, truths)] for k in range(n_ants)]
    vals = []
    for a, b in combinations(range(n_ants), 2):
        x, y = wrong[a], wrong[b]
        if len(set(x)) < 2 or len(set(y)) < 2:
            continue
        vals.append(st.correlation(x, y))
    return st.mean(vals) if vals else 0.0


def same_wrong(answer_lists: Sequence[Sequence[Optional[str]]], truths: Sequence[str]) -> float:
    """Of ant pairs that are both wrong on an item (with a usable answer each), the share that picked the same wrong
    letter. Five options, so ants guessing blindly would land near 0.25."""
    hit = tot = 0
    for ans, t in zip(answer_lists, truths):
        bad = [a for a in ans if a is not None and a != t]
        for a, b in combinations(bad, 2):
            tot += 1
            hit += a == b
    return hit / tot if tot else 0.0


def usable_share(rs: Sequence[Mapping[str, Any]]) -> float:
    vals = [a for r in rs for arm in ARMS for a in r[arm]] + [r["reader"] for r in rs]
    return sum(a is not None for a in vals) / len(vals) if vals else 0.0


def _arm_stats(rs: Sequence[Mapping[str, Any]], key: Mapping[str, str], arm: str,
               families: Sequence[str]) -> Dict[str, Any]:
    truths = [key[r["id"]] for r in rs]
    lists = [r[arm] for r in rs]
    per_ant = [st.mean(_right(lst[k], t) for lst, t in zip(lists, truths)) for k in range(len(lists[0]))]
    fam: Dict[str, List[float]] = {}
    for k, f in enumerate(families):
        fam.setdefault(f, []).append(per_ant[k])
    return {"vote": round(st.mean(_right(ft.vote(lst), t) for lst, t in zip(lists, truths)), 4),
            "ant_mean": round(st.mean(per_ant), 4),
            "by_family": {f: round(st.mean(v), 4) for f, v in fam.items()},
            "error_alikeness": round(error_alikeness(lists, truths), 4),
            "same_wrong": round(same_wrong(lists, truths), 4)}


def _boot(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Any]:
    g, lo, hi = ft.paired_boot([a - b for a, b in zip(xs, ys)])
    return {"gain": round(g, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def _alikeness_boot(rs: Sequence[Mapping[str, Any]], key: Mapping[str, str], reps: int = 2000,
                    seed_: int = 20261030) -> Dict[str, Any]:
    """Rise in error alikeness from ``alone`` to ``both``, with a 95% interval from resampling items."""
    import random
    rng = random.Random(seed_)
    truths = [key[r["id"]] for r in rs]

    def rise(idx: Sequence[int]) -> float:
        t = [truths[i] for i in idx]
        return (error_alikeness([rs[i]["both"] for i in idx], t) - error_alikeness([rs[i]["alone"] for i in idx], t))

    point = rise(range(len(rs)))
    draws = sorted(rise([rng.randrange(len(rs)) for _ in rs]) for _ in range(reps))
    return {"rise": round(point, 4), "ci95": [round(draws[int(0.025 * reps)], 4), round(draws[int(0.975 * reps) - 1], 4)]}


def score(rs: Sequence[Mapping[str, Any]], key: Mapping[str, str], families: Sequence[str],
          *, boot_reps: int = 2000) -> Dict[str, Any]:
    """Every arm's numbers and the preregistered verdicts. ``families`` names each ant column's model family."""
    rs = [r for r in rs if r["id"] in key]
    if not rs:
        raise ValueError("no scored rows")
    out: Dict[str, Any] = {"n": len(rs), "usable_share": round(usable_share(rs), 4),
                           "leak_share": round(sum(bool(r["leak"]) for r in rs) / len(rs), 4)}

    def block(sub: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        truths = [key[r["id"]] for r in sub]
        arms = {arm: _arm_stats(sub, key, arm, families) for arm in ARMS}
        reader = [_right(r["reader"], t) for r, t in zip(sub, truths)]
        vote = {arm: [_right(ft.vote(r[arm]), t) for r, t in zip(sub, truths)] for arm in ARMS}
        return {"n": len(sub), "reader_alone": round(st.mean(reader), 4), "arms": arms,
                "both_vs_alone": _boot(vote["both"], vote["alone"]),
                "notes_vs_alone": _boot(vote["notes"], vote["alone"]),
                "both_vs_reader": _boot(vote["both"], reader),
                "alikeness_rise": _alikeness_boot(sub, key, reps=boot_reps),
                "lift_by_family": {f: {arm: round(arms[arm]["by_family"][f] / arms["alone"]["by_family"][f], 3)
                                       if arms["alone"]["by_family"][f] else None for arm in ("notes", "both")}
                                   for f in arms["alone"]["by_family"]}}

    out["all"] = block(rs)
    clean = [r for r in rs if not r["leak"]]
    if len(clean) < len(rs) and len(clean) >= 20:
        out["no_leaks"] = block(clean)
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rs:
        groups.setdefault(r["group"], []).append(r)
    out["by_group"] = {g: {"n": len(v), "reader_alone": round(st.mean(_right(r["reader"], key[r["id"]]) for r in v), 4),
                           **{arm: round(st.mean(_right(ft.vote(r[arm]), key[r["id"]]) for r in v), 4)
                              for arm in ARMS}}
                       for g, v in sorted(groups.items())}

    main = out["all"]
    helps = main["both_vs_alone"]["gain"] >= PASS["helps_min_gain"] and main["both_vs_alone"]["ci95"][0] > 0
    hurts = main["both_vs_alone"]["ci95"][1] < 0
    alike = (main["alikeness_rise"]["rise"] >= PASS["alike_min_rise"] and main["alikeness_rise"]["ci95"][0] > 0)
    beats = (main["both_vs_reader"]["gain"] >= PASS["beats_reader_min_gain"] and main["both_vs_reader"]["ci95"][0] > 0)
    out["verdict"] = {
        "shared_read_helps": "yes" if helps else "no, it hurts" if hurts else "no clear change",
        "ants_think_alike": "yes" if alike else "no",
        "swarm_beats_reader_alone": "yes" if beats else "no",
    }
    if out["usable_share"] < MIN_USABLE:
        out["verdict"] = {"invalid": f"only {out['usable_share']:.0%} of answers were usable (at least "
                                     f"{MIN_USABLE:.0%} needed), so nothing was measured"}
    elif out["leak_share"] > PASS["max_leak_share"]:
        out["verdict"]["warning"] = (f"{out['leak_share']:.0%} of the reader's notes named an option by letter; "
                                     "read the no_leaks block as the fair result")
    return out


# --------------------------------------------------------------------- sizing

def plan(n_items: int, ants: Sequence[ModelSpec] = ANTS, lenses: Sequence[int] = LENS_IDS) -> Dict[str, int]:
    reader_calls = 2 * n_items
    ant_calls = n_items * len(ants) * len(lenses) * len(ARMS)
    return {"items": n_items, "ants": len(ants) * len(lenses), "reader_calls": reader_calls, "ant_calls": ant_calls,
            "calls": reader_calls + ant_calls}


__all__ = ["ANTS", "ARMS", "BrokenRun", "Item", "LENS_IDS", "MODELS_FILE", "MissingModel", "ModelSpec", "PASS", "READER", "Runner",
           "ant_messages", "ant_roster", "clean_notes", "error_alikeness", "leaks", "load_items", "load_models", "notes_messages",
           "parse_answer", "plan", "reader_answer_messages", "rows", "run_ants", "run_reader", "same_model", "same_wrong", "score",
           "strip_think", "usable_share"]
