"""The decision-model add-on: does a decision model make an LLM swarm better, per call spent?

Preregistered in ``docs/plans/2026-10-30-dm-addon.md``. A decision model (Jev, or an open one run locally: Liquid
d1, Perplexity's Decider) writes no text. It reads a state once and gives a probability to every answer it is
allowed. This test adds one to swarms whose answers are already saved, so it needs **no new LLM call**:

* the fair test's 5 Step 3.7 Flash ants on 240 exam questions and 101 real forecasts (``source="fair-test"``);
* the shared-read test's 8 small local ants, answering alone, on the 240 exam questions (``source="shared-read"``).

Arms, per decision model, per question:

* ``dm1``: the decision model alone, reading the question once (0 LLM calls).
* ``vote``: the ants plus the decision model as one more voter. Its whole probability spread counts as one vote
  (exam), or its probability joins the mean (forecast). Nobody reads anybody: a mixed swarm that thinks alone.
* ``read``: the decision model reads the question and every ant's answer and reason, then picks. It weighs the ants
  as evidence and says nothing back to them, so it combines the swarm without steering it.
* ``cheap``: the decision model answers alone when it is sure enough, and the ants are asked only when it is not.
  The bar is set on one half of the questions and scored on the other half.

The preregistered verdict uses the fair test, Jev and ``vote``: the mixed swarm must beat both LLM-only arms that
spend the same 5 LLM calls (the plain model asked 5 times, and the 5 think-alone ants). Everything else is secondary.

Everything here is pure: building requests, combining, scoring. Calls go through any object with the decision
backend contract (``decide(DecisionRequest) -> DecisionResponse``), so tests use a fake model and no network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.experiments import fair_test as ft
from antelligence.kernel.decision import DecisionError, DecisionRequest, DecisionResponse, Question

SOURCES = ("fair-test", "shared-read")
DM_ARMS = ("dm1", "vote", "read", "cheap")
PRIMARY_DM = "jev"
PRIMARY_SOURCE = "fair-test"
PRIMARY_ARM = "vote"
# The LLM-only arms the mixed swarm must beat, per source. Each spends the same LLM calls as ``vote``.
RIVALS = {"fair-test": ("single5", "swarm5"), "shared-read": ("reader", "alone{n}")}   # {n}: the number of ants
# Preregistered pass lines: the fair test's own, so the two tests read the same way. Changing these voids the test.
PASS = {"exam_min_gain": 0.01, "forecast_min_brier_gain": 0.005}
MIN_USABLE = 0.9          # below this share of usable answers, an arm measured the plumbing, not the model
SPLIT_SEED = 20261030     # which questions tune the ``cheap`` bar and which score it
CLIP = (0.01, 0.99)       # every forecast probability is clipped, as the fair test clips its ants
STATE_LIMIT = 24_000      # characters; the longest state built here is about 6,000 (Jev reads 32k tokens a question)
REASON_CHARS = 300        # how much of each ant's reason ``read`` shows, as the fair test's ``seen`` arm does
EXAM_GROUPS = {"logic": ("lsat-lr", "logiqa-en"), "puzzles": ("lsat-ar", "aqua-rat")}
BARS = tuple(round(0.05 * i, 2) for i in range(21)) + (1.01,)   # 1.01 means "always ask the ants"

DM_INSTRUCTIONS = {
    ("exam", "dm1"): "Which option correctly answers the question?",
    ("exam", "read"): ("Weigh the question and the answers these people gave, as evidence. Which option correctly "
                       "answers the question?"),
    ("forecast", "dm1"): "Will this question resolve YES?",
    ("forecast", "read"): ("Weigh the question and the forecasts these people gave, as evidence. Will this question "
                           "resolve YES?"),
}
NOUL_CRITERIA = {"true": "It resolves YES.", "false": "It resolves NO."}


class LongRequest(DecisionRequest):
    """A decision request with room for an exam passage and five ants' reasons."""

    MAX_STATE = STATE_LIMIT


@dataclass(frozen=True)
class DMSpec:
    """One decision model: a short name, the model id its server wants, where it runs, and its price (input, output
    in dollars per million tokens)."""

    name: str
    model: str
    url: str
    price: Tuple[float, float] = (0.0, 0.0)
    hosted: bool = False
    concurrency: int = 1
    about: str = ""


DMS: Dict[str, DMSpec] = {
    "jev": DMSpec("jev", "jev-1.13.0", "https://api.typesafe.ai", (0.042, 0.0), True, 4,
                  "TypeSafe Jev 1.13, hosted; the model v4 used, pinned by JevBackend"),
    "d1-3b": DMSpec("d1-3b", "LiquidAI/d1-3B", "http://127.0.0.1:8520", about=(
        "Liquid d1-3B, open weights (LFM Open License v1.0), through scripts/v41_d1_server.py on mps")),
    "d1-2b": DMSpec("d1-2b", "mlx-community/d1-2B-OptiQ-4bit", "http://127.0.0.1:8080", about=(
        "Liquid d1-2B at 4 bits on MLX, through `optiq serve` (mlx-optiq 0.7.9+)")),
    "pplx-flash": DMSpec("pplx-flash", "pplx-decider-flash-v1.2", "http://127.0.0.1:8081", about=(
        "Perplexity Decider Flash v1.2 (5B, Apache 2.0) Q8_0 GGUF, through a runtime that speaks its protocol")),
}


# ------------------------------------------------------------------ swarm rows

@dataclass(frozen=True)
class Ant:
    answer: Any                   # exam: a letter or None; forecast: a probability or None
    reason: str = ""


@dataclass(frozen=True)
class SwarmRow:
    """One question of a saved swarm: the ants' answers and reasons, and the source's own LLM-only arms."""

    id: str
    kind: str
    group: str
    ants: Tuple[Ant, ...]
    baselines: Mapping[str, Any]          # arm -> combined answer
    calls: Mapping[str, int]              # arm -> LLM calls per question

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "group": self.group,
                "ants": [{"answer": a.answer, "reason": a.reason} for a in self.ants],
                "baselines": dict(self.baselines), "calls": dict(self.calls)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "SwarmRow":
        return cls(str(d["id"]), str(d["kind"]), str(d.get("group") or ""),
                   tuple(Ant(a.get("answer"), str(a.get("reason") or "")) for a in d["ants"]),
                   dict(d["baselines"]), {k: int(v) for k, v in d["calls"].items()})


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def last_usable(cache_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    """Request hash -> the latest reply that did not fail (a failed call was asked again, so a later row wins)."""
    out: Dict[str, Mapping[str, Any]] = {}
    for row in cache_rows:
        if not row.get("error"):
            out[row["request_hash"]] = row
    return out


def export_fair_test(questions: Sequence[ft.Question], fair_rows: Sequence[Mapping[str, Any]],
                     cache_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[SwarmRow], Dict[str, int]]:
    """The fair test's saved run as swarm rows. Each ant's reason comes from the reply cache, found by rebuilding the
    exact request with the frozen fair-test code. An answer that differs from the fair test's own row is an error
    (the cache and the rows would be from different runs); a reason that can't be found is counted."""
    cache = last_usable(cache_rows)
    by_id = {r["id"]: r for r in fair_rows}
    out: List[SwarmRow] = []
    found = missing = 0
    for q in questions:
        row = by_id.get(q.id)
        if row is None:
            continue
        ants = []
        for i in range(ft.K):
            hit = cache.get(ft.request(ft.lens_messages(q, i), q.id, "swarm", i).request_hash)
            answer = row["swarm"][i]
            reason = ""
            if hit is not None:
                cached_answer, reason = ft._reply(q.kind, hit)
                if cached_answer != answer:
                    raise ValueError(f"{q.id} ant {i}: the cache says {cached_answer!r}, the rows say {answer!r}: "
                                     "they are not from the same run")
                found += 1
            elif answer is not None:
                missing += 1
            ants.append(Ant(answer, reason))
        out.append(SwarmRow(q.id, q.kind, q.group, tuple(ants), ft.arm_answers(row),
                            {"single1": 1, "single5": 5, "single10": 10, "swarm5": 5, "seen10": 10}))
    return out, {"reasons_found": found, "reasons_missing": missing}


def export_shared_read(sr_rows: Sequence[Mapping[str, Any]]) -> List[SwarmRow]:
    """The shared-read run's ants answering alone, as swarm rows. Its rows keep answers, not reasons, so ``read``
    shows each ant's answer only. Its LLM-only arms are the reader alone and the plain votes of each arm."""
    out = []
    for r in sr_rows:
        alone = list(r["alone"])
        n = len(alone)
        out.append(SwarmRow(str(r["id"]), "exam", str(r.get("group") or ""), tuple(Ant(a) for a in alone),
                            {"reader": r.get("reader"), f"alone{n}": ft.vote(alone),
                             f"notes{n}": ft.vote(r.get("notes") or [None]), f"both{n}": ft.vote(r.get("both") or [None])},
                            {"reader": 1, f"alone{n}": n, f"notes{n}": n + 1, f"both{n}": n + 1}))
    return out


# ------------------------------------------------------------------ requests

def call_seed(qid: str, arm: str) -> int:
    return int.from_bytes(hashlib.sha256(f"dm:{qid}:{arm}".encode()).digest()[:4], "big")


_LABEL = re.compile(r"^\s*\(?[A-E]\)\s*")


def option_texts(raw: Mapping[str, Any]) -> Dict[str, str]:
    """Letter -> option text without its "(A)" label, from a raw exam question."""
    return {ft.LETTERS[i]: _LABEL.sub("", str(o)).strip() for i, o in enumerate(raw["options"])}


def load_options(path: Path) -> Dict[str, Dict[str, str]]:
    return {str(r["id"]): option_texts(r) for r in read_jsonl(path)}


def _show(kind: str, answer: Any) -> str:
    if answer is None:
        return "nothing usable"
    return str(answer) if kind == "exam" else f"{float(answer):.2f} chance of YES"


def read_state(q: ft.Question, ants: Sequence[Ant]) -> str:
    lines = []
    for i, a in enumerate(ants, 1):
        verb = "answered" if q.kind == "exam" else "gave"
        reason = a.reason.strip()[:REASON_CHARS]
        lines.append(f"Person {i} {verb} {_show(q.kind, a.answer)}" + (f": {reason}" if reason else "."))
    return f"{q.text}\n\n{len(ants)} people answered this on their own:\n" + "\n".join(lines)


def dm_question(q: ft.Question, arm: str, options: Optional[Mapping[str, str]]) -> Question:
    text = DM_INSTRUCTIONS[(q.kind, arm)]
    if q.kind == "forecast":
        return Question("noul", text, criteria=NOUL_CRITERIA)
    if not options:
        raise ValueError(f"{q.id}: an exam question needs its options")
    return Question("choice", text, tuple(options), dict(options))


def dm_request(model: str, q: ft.Question, arm: str, options: Optional[Mapping[str, str]] = None,
               ants: Sequence[Ant] = ()) -> LongRequest:
    """``dm1`` reads the question; ``read`` also reads every ant. Choice options go in a seeded order, because a
    decision model can favour the first option."""
    if arm not in ("dm1", "read"):
        raise ValueError("only dm1 and read call the model")
    if arm == "read" and not ants:
        raise ValueError("read needs the ants")
    state = q.text if arm == "dm1" else read_state(q, ants)
    return LongRequest.build(model, state, {"answer": dm_question(q, arm, options)}, call_seed(q.id, arm))


# ------------------------------------------------------------------- calling

class DMRunner:
    """Asks a decision model, caching each usable reply by request hash in ``cache_path`` (JSONL), so a stopped run
    resumes without paying twice. A failed call is written with its reason but asked again on a rerun. Spend is
    capped: before each call, an upper estimate of its cost is held against ``max_usd``. After ``max_fail_streak``
    failures in a row the run stops, since it would be measuring the plumbing."""

    def __init__(self, backend: Any, cache_path: Path, *, max_usd: float, price: Tuple[float, float] = (0.0, 0.0),
                 max_fail_streak: int = 5) -> None:
        self.backend, self.cache_path, self.max_usd, self.price = backend, Path(cache_path), max_usd, price
        self.max_fail_streak = max_fail_streak
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.spent = 0.0
        if self.cache_path.exists():
            for row in read_jsonl(self.cache_path):
                self.spent += float(row.get("cost_usd") or 0.0)
                if not row.get("error"):
                    self.cache[row["request_hash"]] = row
        self.reserved = 0.0
        self.calls = self.failed = self.streak = 0
        self.broken: Optional[str] = None
        self._lock = asyncio.Lock()

    def hold(self, request: DecisionRequest) -> float:
        """An upper estimate of one call's cost: the payload at 3 characters a token, plus half again."""
        tokens = len(json.dumps(request.payload())) / 3
        return 1.5 * tokens * self.price[0] / 1_000_000

    async def ask(self, request: DecisionRequest) -> Dict[str, Any]:
        h = request.request_hash
        if h in self.cache:
            return self.cache[h]
        hold = self.hold(request)
        async with self._lock:
            if self.broken:
                raise BrokenRun(self.broken)
            if self.spent + self.reserved + hold > self.max_usd:
                raise ft.SpendCap(f"spent ${self.spent:.4f}; the cap is ${self.max_usd:.2f}")
            self.reserved += hold
        try:
            resp: DecisionResponse = await self.backend.decide(request)
        except DecisionError as exc:
            billed = getattr(exc, "billed", None) or {}
            row = {"request_hash": h, "answers": None, "served_model": None, "input_tokens": billed.get("input_tokens", 0),
                   "cost_usd": float(billed.get("cost_usd", 0.0)), "elapsed_s": 0.0, "error": type(exc).__name__,
                   "reason": str(exc)[:300], "http_status": getattr(exc, "http_status", None)}
        else:
            row = {"request_hash": h, "answers": {k: a.to_dict() for k, a in resp.answers.items()},
                   "served_model": resp.served_model, "input_tokens": resp.input_tokens, "cost_usd": resp.cost_usd,
                   "elapsed_s": round(resp.elapsed_s, 4), "error": None}
        async with self._lock:
            self.reserved -= hold
            self.spent += row["cost_usd"]
            self.calls += 1
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            if row["error"] is None:
                self.cache[h] = row
                self.streak = 0
            else:
                self.failed += 1
                self.streak += 1
                if self.streak >= self.max_fail_streak and not self.broken:
                    self.broken = f"{self.streak} calls in a row failed. Last: {row['reason']}"
            if self.broken:
                raise BrokenRun(self.broken)
        return row

    async def run(self, questions: Sequence[ft.Question], swarm: Mapping[str, SwarmRow], model: str,
                  options: Mapping[str, Mapping[str, str]], *, arms: Sequence[str] = ("dm1", "read"),
                  concurrency: int = 1, on_done: Optional[Callable[[Dict[str, Any]], None]] = None
                  ) -> List[Dict[str, Any]]:
        """``dm1`` and ``read`` for every question that has a swarm row. One output row per question."""
        sem = asyncio.Semaphore(concurrency)
        done: Dict[str, Dict[str, Any]] = {}

        async def one(q: ft.Question) -> None:
            async with sem:
                out: Dict[str, Any] = {"id": q.id, "kind": q.kind}
                for arm in arms:
                    req = dm_request(model, q, arm, options.get(q.id), swarm[q.id].ants if arm == "read" else ())
                    out[arm] = to_answer(q.kind, await self.ask(req))
                done[q.id] = out
                if on_done:
                    on_done(out)

        await asyncio.gather(*(one(q) for q in questions if q.id in swarm))
        return [done[q.id] for q in questions if q.id in done]


class BrokenRun(RuntimeError):
    """The decision model failed too many times in a row."""


def to_answer(kind: str, row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """A cached reply as a plain answer: exam ``{"choice", "probs", "conf"}``, forecast ``{"p"}``; None if failed."""
    if row.get("error") or not row.get("answers"):
        return None
    a = row["answers"]["answer"]
    if kind == "forecast":
        return {"p": float(a["value"])}
    return {"choice": a["value"], "probs": {k: float(v) for k, v in a["probabilities"].items()},
            "conf": a.get("confidence")}


# ------------------------------------------------------------------ combining

def clip(p: float) -> float:
    return min(max(float(p), CLIP[0]), CLIP[1])


def dm_value(kind: str, ans: Optional[Mapping[str, Any]]) -> Any:
    """What the decision model's own answer scores as: a letter, or a clipped probability (0.5 if it failed)."""
    if kind == "exam":
        return None if ans is None else ans["choice"]
    return 0.5 if ans is None else clip(ans["p"])


def confidence(kind: str, ans: Optional[Mapping[str, Any]]) -> float:
    """How sure the decision model is, from 0 to 1. Exam: the model's own confidence when it gives one, else its top
    probability rescaled to the number of options (0 when all are equal). Forecast: distance from 50%, doubled."""
    if ans is None:
        return 0.0
    if kind == "forecast":
        return round(abs(2 * float(ans["p"]) - 1), 9)
    if ans.get("conf") is not None:
        return float(ans["conf"])
    probs = list(ans["probs"].values())
    n = len(probs)
    # rounded so that a bar such as 0.2 compares as written (2 * 0.6 - 1 is 0.19999999999999996 in floats)
    return round(min(1.0, max(0.0, (n * max(probs) - 1) / (n - 1))), 9) if n > 1 else 1.0


def vote_with_dm(kind: str, ants: Sequence[Any], ans: Optional[Mapping[str, Any]]) -> Any:
    """The ants plus the decision model as one more voter. Exam: each ant's letter is one vote, and the model's
    probabilities are one vote spread over the options; the top total wins, a tie going to the earlier letter. A
    failed model call adds nothing. Forecast: the mean of the ants' and the model's clipped probabilities."""
    if kind == "forecast":
        return ft.mean_prob(list(ants) + ([None] if ans is None else [ans["p"]]))
    totals: Dict[str, float] = {}
    for a in ants:
        if a is not None:
            totals[a] = totals.get(a, 0.0) + 1.0
    for letter, p in (ans or {}).get("probs", {}).items():
        totals[letter] = totals.get(letter, 0.0) + p
    if not totals:
        return None
    best = max(totals.values())
    return min(k for k, v in totals.items() if math.isclose(v, best, abs_tol=1e-9))


def cheap_answer(kind: str, ants: Sequence[Any], ans: Optional[Mapping[str, Any]], bar: float) -> Tuple[Any, bool]:
    """(answer, asked the ants?). The model alone when its confidence reaches ``bar``, else the ants' plain vote."""
    if ans is not None and confidence(kind, ans) >= bar:
        return dm_value(kind, ans), False
    return ft.combine(kind, ants), True


def tune_half(qid: str) -> bool:
    """A fixed half of the questions tunes the ``cheap`` bar; the other half scores it."""
    return hashlib.sha256(f"{SPLIT_SEED}:{qid}".encode()).digest()[0] % 2 == 0


def quality(kind: str, scores: Sequence[float]) -> float:
    """Higher is better for both kinds: share right, or minus the Brier score."""
    m = st.mean(scores) if scores else 0.0
    return m if kind == "exam" else -m


def pick_bar(kind: str, rows: Sequence[Tuple[SwarmRow, Optional[Mapping[str, Any]]]], key: Mapping[str, Any]
             ) -> Dict[str, Any]:
    """The lowest bar (fewest ant calls) whose quality on the tuning rows is no worse than the ants' own vote by more
    than the pass line. Always asking the ants (bar 1.01) qualifies by definition."""
    margin = PASS["exam_min_gain"] if kind == "exam" else PASS["forecast_min_brier_gain"]
    ants_q = quality(kind, [ft.score_item(kind, ft.combine(kind, [a.answer for a in r.ants]), key[r.id])
                            for r, _ in rows])
    for bar in BARS:
        picked = [cheap_answer(kind, [a.answer for a in r.ants], ans, bar) for r, ans in rows]
        q = quality(kind, [ft.score_item(kind, a, key[r.id]) for (a, _), (r, _) in zip(picked, rows)])
        if q >= ants_q - margin - 1e-12:
            return {"bar": bar, "asked_share": round(sum(asked for _, asked in picked) / len(rows), 4) if rows else 0.0}
    return {"bar": BARS[-1], "asked_share": 1.0}


# -------------------------------------------------------------------- scoring

def _boot(kind: str, a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Gain of ``a`` over ``b`` (positive = better), with a paired bootstrap 95% interval."""
    sign = 1.0 if kind == "exam" else -1.0
    gain, lo, hi = ft.paired_boot([sign * (x - y) for x, y in zip(a, b)])
    return {"gain": round(gain, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def usable(kind: str, xs: Sequence[Any]) -> float:
    return sum(x is not None for x in xs) / len(xs) if xs else 0.0


def score(source: str, swarm: Sequence[SwarmRow], dms: Mapping[str, Mapping[str, Mapping[str, Any]]],
          key: Mapping[str, Any], crowd: Optional[Mapping[str, float]] = None,
          dm_cost: Optional[Mapping[str, Mapping[str, float]]] = None) -> Dict[str, Any]:
    """Every arm on every kind, the comparisons, and the preregistered verdict when this is the fair test and Jev.

    ``dms``: decision-model name -> question id -> {"dm1": answer, "read": answer} (answers as :func:`to_answer`)."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    out: Dict[str, Any] = {"source": source, "kinds": {}, "pass": PASS, "primary": None}
    for kind in ft.KINDS:
        items = [r for r in swarm if r.kind == kind and r.id in key]
        if not items:
            continue
        per: Dict[str, List[float]] = {}
        calls: Dict[str, float] = {}

        def add(arm: str, answers: Sequence[Any], llm_calls: float) -> None:
            per[arm] = [ft.score_item(kind, a, key[r.id]) for a, r in zip(answers, items)]
            calls[arm] = llm_calls

        for arm in items[0].baselines:
            add(arm, [r.baselines.get(arm) for r in items], items[0].calls[arm])
        ant_answers = [[a.answer for a in r.ants] for r in items]
        n_ants = len(items[0].ants)
        ant_share = usable(kind, [a for aa in ant_answers for a in aa])
        block: Dict[str, Any] = {"n": len(items), "ants": n_ants, "ants_usable": round(ant_share, 4), "dms": {}}
        for name, answers in dms.items():
            got = [answers.get(r.id, {}) for r in items]
            if not any(got):
                continue
            dm1 = [g.get("dm1") for g in got]
            rd = [g.get("read") for g in got]
            add(f"{name}:dm1", [dm_value(kind, a) for a in dm1], 0)
            add(f"{name}:vote", [vote_with_dm(kind, aa, a) for aa, a in zip(ant_answers, dm1)], n_ants)
            if any(x is not None for x in rd):
                add(f"{name}:read", [dm_value(kind, a) for a in rd], n_ants)
            tune = [(r, a) for r, a in zip(items, dm1) if tune_half(r.id)]
            test_idx = [i for i, r in enumerate(items) if not tune_half(r.id)]
            bar = pick_bar(kind, tune, key)
            picked = [cheap_answer(kind, ant_answers[i], dm1[i], bar["bar"]) for i in test_idx]
            asked = sum(a for _, a in picked) / len(picked) if picked else 0.0
            cheap_scores = [ft.score_item(kind, a, key[items[i].id]) for (a, _), i in zip(picked, test_idx)]
            ants_test = [ft.score_item(kind, ft.combine(kind, ant_answers[i]), key[items[i].id]) for i in test_idx]
            info: Dict[str, Any] = {
                "usable_dm1": round(usable(kind, dm1), 4), "usable_read": round(usable(kind, rd), 4),
                "cheap": {"bar": bar["bar"], "tune_asked_share": bar["asked_share"], "test_n": len(test_idx),
                          "test_asked_share": round(asked, 4), "llm_calls_per_question": round(n_ants * asked, 3),
                          "score": round(st.mean(cheap_scores), 4) if cheap_scores else None,
                          "ants_score_same_half": round(st.mean(ants_test), 4) if ants_test else None,
                          "vs_ants_same_half": _boot(kind, cheap_scores, ants_test) if cheap_scores else None}}
            info["valid"] = info["usable_dm1"] >= MIN_USABLE and ant_share >= MIN_USABLE
            if dm_cost and name in dm_cost:
                info["cost"] = dict(dm_cost[name])
            # Does the decision model add anything on top of the ants? And does the mix beat each LLM-only arm?
            info["vote_vs_ants"] = _boot(kind, per[f"{name}:vote"], per[_ants_arm(source, n_ants)])
            for rival in rivals(source, n_ants):
                if rival not in per:
                    continue
                info[f"vote_vs_{rival}"] = _boot(kind, per[f"{name}:vote"], per[rival])
                if f"{name}:read" in per:
                    info[f"read_vs_{rival}"] = _boot(kind, per[f"{name}:read"], per[rival])
            block["dms"][name] = info
        block["means"] = {arm: round(st.mean(v), 4) for arm, v in per.items()}
        block["llm_calls"] = calls
        if kind == "exam":
            block["by_group"] = {}
            for gname, members in EXAM_GROUPS.items():
                idx = [i for i, r in enumerate(items) if r.group in members]
                if idx:
                    block["by_group"][gname] = {"n": len(idx), **{arm: round(st.mean(v[i] for i in idx), 4)
                                                                  for arm, v in per.items()}}
        if kind == "forecast" and crowd:
            idx = [i for i, r in enumerate(items) if r.id in crowd]
            if idx:
                block["crowd_brier"] = round(st.mean(ft.score_item(kind, crowd[items[i].id], key[items[i].id])
                                                     for i in idx), 4)
        out["kinds"][kind] = block
    out["primary"] = primary(out) if source == PRIMARY_SOURCE else None
    return out


def rivals(source: str, n_ants: int) -> Tuple[str, ...]:
    """The LLM-only arms a mix is compared with."""
    return tuple(r.format(n=n_ants) for r in RIVALS[source])


def _ants_arm(source: str, n_ants: int) -> str:
    return "swarm5" if source == "fair-test" else f"alone{n_ants}"


def primary(result: Mapping[str, Any]) -> Dict[str, Any]:
    """The preregistered verdict: on each kind, Jev's ``vote`` must beat both ``single5`` and ``swarm5`` by the pass
    line, each with its 95% interval above zero. Supported if both kinds pass, mixed if one does."""
    verdicts: Dict[str, Any] = {}
    for kind, block in result["kinds"].items():
        info = block["dms"].get(PRIMARY_DM)
        if info is None:
            continue
        need = PASS["exam_min_gain"] if kind == "exam" else PASS["forecast_min_brier_gain"]
        checks = {rival: info[f"vote_vs_{rival}"] for rival in RIVALS[PRIMARY_SOURCE]}
        ok = all(c["gain"] >= need and c["ci95"][0] > 0 for c in checks.values())
        verdicts[kind] = {"valid": info["valid"], "pass": bool(ok and info["valid"]), **checks}
    if not verdicts:
        return {"verdict": f"no verdict: {PRIMARY_DM} has no answers"}
    if not all(v["valid"] for v in verdicts.values()):
        return {"verdict": f"invalid run: under {MIN_USABLE:.0%} of answers were usable", **verdicts}
    passed = [v["pass"] for v in verdicts.values()]
    word = "supported" if len(passed) == 2 and all(passed) else "mixed" if any(passed) else "not supported"
    return {"verdict": word, **verdicts}


# --------------------------------------------------------------- cost estimate

def estimate(questions: Sequence[ft.Question], swarm: Mapping[str, SwarmRow], options: Mapping[str, Mapping[str, str]],
             spec: DMSpec, arms: Sequence[str] = ("dm1", "read")) -> Dict[str, Any]:
    """Calls, tokens and dollars before any call: the payload at 4 characters a token."""
    calls = chars = 0
    for q in questions:
        if q.id not in swarm:
            continue
        for arm in arms:
            req = dm_request(spec.model, q, arm, options.get(q.id), swarm[q.id].ants if arm == "read" else ())
            chars += len(json.dumps(req.payload()))
            calls += 1
    tokens = chars // 4
    return {"questions": sum(q.id in swarm for q in questions), "calls": calls, "input_tokens": tokens,
            "usd": round(tokens * spec.price[0] / 1_000_000, 4)}


class FakeDM:
    """A $0 stand-in that answers at random (seeded by the request), for checking the plumbing only."""

    def describe(self) -> Dict[str, Any]:
        return {"backend": "fake"}

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        from antelligence.kernel.decision import Answer

        rng = random.Random(request.request_hash)
        answers = {}
        for qid, q in request.questions.items():
            if q.type == "noul":
                p = round(rng.random(), 3)
                answers[qid] = Answer("noul", p, {"yes": p, "no": 1 - p})
            else:
                w = [rng.random() + 0.01 for _ in q.options]
                probs = {o: x / sum(w) for o, x in zip(q.options, w)}
                answers[qid] = Answer("choice", max(probs, key=probs.get), probs, None)
        return DecisionResponse(answers, request.model, request.request_hash,
                                input_tokens=len(json.dumps(request.payload())) // 4)


__all__ = ["Ant", "BrokenRun", "DMRunner", "DMS", "DMSpec", "DM_ARMS", "FakeDM", "LongRequest", "PASS", "RIVALS",
           "SOURCES", "SwarmRow", "cheap_answer", "confidence", "dm_request", "estimate", "export_fair_test",
           "export_shared_read", "load_options", "pick_bar", "primary", "read_state", "score", "to_answer",
           "tune_half", "vote_with_dm"]
