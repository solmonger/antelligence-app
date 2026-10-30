"""The fair test: does a think-alone swarm beat the best single model at the same number of calls?

Preregistered in ``docs/plans/2026-10-29-fair-test.md`` (the pass lines are fixed there, before any call). Two task
sets we did not design, with answers kept out of every prompt:

* ``exam``: AGIEval English multiple choice (LSAT analytical and logical reasoning, LogiQA, AQuA-RAT maths).
* ``forecast``: real yes/no forecasting questions that resolved between July and October 2026, after the model was
  released (28 May 2026), so it cannot have seen the answer.

One model (StepFun Step 3.7 Flash on the Nous portal, free there, reasoning set to low; Amendment 2), so "the best
single model" is that model asked again and again. Arms, per question:

* ``single``: the same plain prompt sampled 10 times (temperature 0.7). The first 5 are the primary baseline
  ("best single model x 5", self-consistency); all 10 are the baseline for the 10-call secondary arm.
* ``swarm``: 5 ants that each answer **alone**, each told to look at the question a different way (a "lens"),
  same temperature. Combined by a plain vote (exam) or the mean probability (forecast). No checker, no judge, no
  ant sees another.
* ``seen``: the same 5 ants asked once more after reading the other four ants' first answers and reasons, then
  combined the same way (5 + 5 = 10 calls). This re-tests "seeing each other hurts" on fresh tasks.

Everything here is pure: prompts, parsing, combining, scoring and the cost estimate. Calls go through any object
with the provider contract (``complete(ChatRequest) -> ChatResponse``), so tests use a fake provider and no network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from antelligence.providers.base import ChatRequest, ChatResponse, ProviderError, ProviderUnavailable
from antelligence.providers.redact import scrub

# Amendment 2: the free Step 3.7 Flash replaces DeepSeek V4.1 Flash (which needed credit the account could not use).
MODEL = "stepfun/step-3.7-flash:free"   # the free route; plain "stepfun/step-3.7-flash" is the paid one
RELEASED = "2026-05-28"          # every forecast in the set resolved after this day
PRICE = (0.0, 0.0)               # free on the Nous portal; a cost the portal reports is still counted, and capped
REASONING = {"reasoning": {"effort": "low"}}   # a reasoning model; low is its lightest documented level
TEMPERATURE = 0.7
MAX_TOKENS = 4000                # room for its reasoning before the short JSON answer
K = 5                            # calls per question for the primary comparison
SINGLE_SAMPLES = 10
KINDS = ("exam", "forecast")
LETTERS = "ABCDE"

SYSTEM = ("You answer one question on your own. You cannot look anything up. "
          "Reply with one JSON object and nothing else.")

LENSES: Dict[str, Tuple[str, ...]] = {
    "exam": (
        "Work it out directly, step by step, then pick.",
        "First rule out every option you can show is wrong, then pick from what is left.",
        "First name the rule, principle or definition the question turns on, then apply it.",
        "Check each option against the facts given, one at a time, before you pick.",
        "Find the most tempting wrong option and say why it is wrong, then pick.",
    ),
    "forecast": (
        "Start from the base rate: how often do events like this happen?",
        "Focus on the specific situation: who decides, what has been said, what is scheduled.",
        "Assume things stay as they are unless something specific changes them.",
        "Imagine it resolved YES. What most likely made that happen? Then weigh how likely that is.",
        "Imagine it resolved NO. What most likely made that happen? Then weigh how likely that is.",
    ),
}

# ------------------------------------------------------------------ questions


@dataclass(frozen=True)
class Question:
    kind: str
    id: str
    group: str                    # exam: the AGIEval task; forecast: the field
    text: str                     # everything the model is shown about the question (never the answer)


def load_questions(path: Path, kind: str) -> List[Question]:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for banned in ("label", "answer", "outcome", "solution", "other", "resolution_date", "crowd_prob", "source_url"):
            if banned in r:
                raise ValueError(f"question {r.get('id')} carries {banned!r}: answers must stay out of the questions file")
        out.append(Question(kind, str(r["id"]), str(r.get("task") or r.get("field") or ""), _render(kind, r)))
    if len({q.id for q in out}) != len(out):
        raise ValueError("duplicate question ids")
    return out


def _render(kind: str, r: Mapping[str, Any]) -> str:
    if kind == "exam":
        parts = []
        if r.get("passage"):
            parts.append(str(r["passage"]).strip())
        parts.append(str(r["question"]).strip())
        parts.append("Options:\n" + "\n".join(str(o).strip() for o in r["options"]))
        return "\n\n".join(parts)
    parts = [f"Question: {str(r['question']).strip()}"]
    if r.get("background"):
        parts.append(f"Background: {str(r['background']).strip()}")
    if r.get("close_date"):
        parts.append(f"The question closes on {r['close_date']}.")
    return "\n".join(parts)


# -------------------------------------------------------------------- prompts

def _format(kind: str) -> str:
    if kind == "exam":
        return 'Reply as JSON: {"reason": "<at most 60 words>", "answer": "<one letter>"}'
    return ('Reply as JSON: {"reason": "<at most 60 words>", "probability": <a number from 0 to 1 that it resolves '
            'YES>}')


def plain_messages(q: Question) -> Tuple[Dict[str, str], ...]:
    return ({"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{q.text}\n\n{_format(q.kind)}"})


def lens_messages(q: Question, lens: int) -> Tuple[Dict[str, str], ...]:
    return ({"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{q.text}\n\nHow to think about it: {LENSES[q.kind][lens]}\n\n{_format(q.kind)}"})


def seen_messages(q: Question, lens: int, peers: Sequence[Tuple[int, Optional[str], str]]) -> Tuple[Dict[str, str], ...]:
    """``peers``: (lens, answer as text or None, reason) for the OTHER ants' first answers."""
    lines = []
    for i, (_, ans, reason) in enumerate(peers, 1):
        lines.append(f"Person {i} answered {ans if ans is not None else 'nothing usable'}: {reason[:300]}")
    shown = "\n".join(lines)
    return ({"role": "system", "content": SYSTEM},
            {"role": "user", "content": (f"{q.text}\n\nHow to think about it: {LENSES[q.kind][lens]}\n\n"
                                         f"Four other people answered this on their own:\n{shown}\n\n"
                                         f"Weigh what they said as evidence, then give your own final answer.\n\n"
                                         f"{_format(q.kind)}")})


# -------------------------------------------------------------------- parsing

_JSON = re.compile(r"\{.*\}", re.S)


def parse_reply(kind: str, text: str) -> Tuple[Optional[Any], str]:
    """(answer, reason). Exam answers are a letter A-E; forecasts a probability in [0, 1]. None when unusable."""
    m = _JSON.search(text or "")
    if not m:
        return None, ""
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, ""
    if not isinstance(data, dict):
        return None, ""
    reason = str(data.get("reason") or "")[:600]
    if kind == "exam":
        ans = str(data.get("answer") or "").strip().upper().strip("()")
        return (ans if len(ans) == 1 and ans in LETTERS else None), reason
    try:
        p = float(data.get("probability"))
    except (TypeError, ValueError):
        return None, reason
    if p != p or not 0 <= p <= 1:  # NaN or out of range
        if 1 < p <= 100:
            p = p / 100  # "70" meaning 70%
        else:
            return None, reason
    return p, reason


# ------------------------------------------------------------------ combining

def vote(answers: Sequence[Optional[str]]) -> Optional[str]:
    """Plurality over usable letters; a tie goes to the tied letter given first."""
    counts: Dict[str, int] = {}
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    return next(a for a in answers if a is not None and counts[a] == best)


def mean_prob(probs: Sequence[Optional[float]]) -> float:
    """Mean of usable probabilities, each clipped to [0.01, 0.99]; 0.5 when none is usable."""
    ok = [min(max(p, 0.01), 0.99) for p in probs if p is not None]
    return st.mean(ok) if ok else 0.5


def combine(kind: str, answers: Sequence[Any]) -> Any:
    return vote(answers) if kind == "exam" else mean_prob(answers)


# ------------------------------------------------------------------- running

def call_seed(qid: str, arm: str, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"fair:{qid}:{arm}:{index}".encode()).digest()[:4], "big")


def request(messages: Tuple[Dict[str, str], ...], qid: str, arm: str, index: int) -> ChatRequest:
    return ChatRequest(model=MODEL, messages=messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                       seed=call_seed(qid, arm, index), extra_body=REASONING)


class SpendCap(RuntimeError):
    """The run stopped because the next call could push spend over the cap."""


class BrokenRun(RuntimeError):
    """The run stopped because the last ``max_fail_streak`` calls in a row gave no usable answer. A run like that
    measures the plumbing, not the swarm, so it must stop and say why instead of scoring empty answers."""


class OutOfCredits(BrokenRun):
    """The provider refused because the account has no credit left. No call can succeed until it is topped up, so
    the run stops on the first such refusal instead of after a streak."""


_NO_CREDIT = re.compile(r"requires available credits|balance is too low|insufficient (credits|balance|funds)|"
                        r"out of credits|payment required", re.I)


def out_of_credits(exc: ProviderError) -> bool:
    return getattr(exc, "http_status", None) == 402 or bool(_NO_CREDIT.search(str(exc)))


def served_ok(served: str, model: str = MODEL) -> bool:
    """Does the model name the provider reports name the model we asked for? Providers spell it differently
    (``stepfun/step-3.7-flash``, ``stepfun-ai/Step-3.7-Flash``, a dated suffix), and a route tag such as ``:free``
    names how it is billed, not which model it is. So compare the bare, lower-case name without the tag and allow a
    suffix, but never a different model."""
    def bare(name: str) -> str:
        return name.strip().lower().split(":", 1)[0].rsplit("/", 1)[-1].replace("_", "-").replace(" ", "-")
    return bool(served) and bare(served).startswith(bare(model))


class Runner:
    """Runs every arm for every question, caching each reply by request hash in ``cache_path`` (JSONL), so a stopped
    run resumes without paying twice. Spend is the provider's reported cost when it gives one, else the token count
    times the price, and it is counted for every billed reply, usable or not.

    A call that failed (no reply, or a reply we refuse) is cached with its reason but never reused: a rerun asks
    again. After ``max_fail_streak`` unusable calls in a row the run stops with :class:`BrokenRun`."""

    def __init__(self, provider: Any, cache_path: Path, *, max_usd: float, price: Optional[Tuple[float, float]] = None,
                 max_call_usd: float = 0.002, backoff: Sequence[float] = (2, 4, 8, 16, 32, 64),
                 max_fail_streak: int = 5) -> None:
        self.backoff = tuple(backoff)
        self.provider, self.cache_path, self.max_usd = provider, Path(cache_path), max_usd
        self.price = PRICE if price is None else price
        self.max_call_usd = max_call_usd  # worst case for one call: 3k tokens in, 500 out
        self.max_fail_streak = max_fail_streak
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ignored = 0
        self.spent_before = 0.0
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.spent_before += row.get("cost_usd", 0.0)
                    if row.get("error"):
                        self.ignored += 1  # a failed call is asked again, never scored from the cache
                        continue
                    self.cache[row["request_hash"]] = row
        # settled dollars, failed-but-billed calls from earlier runs included: that money is gone either way
        self.spent = self.spent_before
        self.reserved = 0.0                                                       # held for calls in flight
        self.calls = self.failed = self.streak = 0
        self.reasons: List[str] = []
        self.broken: Optional[str] = None
        self._lock = asyncio.Lock()

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price[0] + completion_tokens * self.price[1]) / 1_000_000

    async def ask(self, req: ChatRequest, kind: str) -> Dict[str, Any]:
        h = req.request_hash
        if h in self.cache:
            return self.cache[h]
        async with self._lock:
            if self.broken:
                raise BrokenRun(self.broken)
            if self.spent + self.reserved + self.max_call_usd > self.max_usd:
                raise SpendCap(f"spent ${self.spent:.4f}; the cap is ${self.max_usd:.2f}")
            self.reserved += self.max_call_usd
        try:
            resp: ChatResponse = await self._complete(req)
        except ProviderUnavailable:
            async with self._lock:  # an outage that outlived the retries: release the reservation, cache nothing
                self.reserved -= self.max_call_usd
            raise
        except ProviderError as exc:  # refused before any reply: nothing billed
            if out_of_credits(exc):  # nothing cached: once topped up, every call is asked again
                async with self._lock:
                    self.reserved -= self.max_call_usd
                    self.broken = self.broken or (f"Out of credits: the provider refused {req.model} because the "
                                                  "account balance is too low. Run `fair_test.py account` to see "
                                                  "which account the key belongs to.")
                raise OutOfCredits(self.broken) from None
            row = {"request_hash": h, "content": "", "prompt_tokens": 0, "completion_tokens": 0,
                   "finish_reason": "error", "served_model": None, "cost_usd": 0.0, "error": type(exc).__name__,
                   "reason": scrub(str(exc))[:300], "http_status": getattr(exc, "http_status", None)}
        else:
            reported = (resp.extra or {}).get("cost")
            cost = float(reported) if isinstance(reported, (int, float)) else self.cost(resp.prompt_tokens,
                                                                                         resp.completion_tokens)
            row = {"request_hash": h, "content": resp.content, "prompt_tokens": resp.prompt_tokens,
                   "completion_tokens": resp.completion_tokens, "finish_reason": resp.finish_reason,
                   "served_model": resp.model, "cost_usd": cost, "error": None}
            if not served_ok(resp.model, req.model):
                row.update(error="ServedModelMismatch", reason=f"asked for {req.model!r}, served {resp.model!r}")
            elif resp.finish_reason != "stop":
                row.update(error="NotFinished", reason=f"reply stopped early ({resp.finish_reason}) after "
                                                       f"{resp.completion_tokens} tokens")
            elif parse_reply(kind, resp.content)[0] is None:
                row["reason"] = f"reply not in the asked format: {resp.content[:160]!r}"
        usable = _reply(kind, row)[0] is not None
        async with self._lock:
            self.reserved -= self.max_call_usd
            self.spent += row["cost_usd"]
            self.calls += 1
            self.failed += row["error"] is not None
            with self.cache_path.open("a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            if row["error"] is None:
                self.cache[h] = row
            if usable:
                self.streak = 0
            else:
                self.streak += 1
                if len(self.reasons) < 5:
                    self.reasons.append(row.get("reason") or row["error"] or "unusable")
                if self.streak >= self.max_fail_streak and not self.broken:
                    self.broken = (f"{self.streak} calls in a row gave no usable answer. First reasons: "
                                   + " | ".join(dict.fromkeys(self.reasons)))
            if self.broken:
                raise BrokenRun(self.broken)
        return row

    async def _complete(self, req: ChatRequest) -> ChatResponse:
        """One call, retrying an outage (HTTP 401/429/5xx or a dropped connection) with backoff. An outage that
        outlives the retries stops the run (nothing is cached for it, so a rerun asks again) rather than being
        scored as a wrong answer."""
        for wait in self.backoff:
            try:
                return await self.provider.complete(req)
            except ProviderUnavailable:
                await asyncio.sleep(wait)
        return await self.provider.complete(req)

    async def run_question(self, q: Question) -> Dict[str, Any]:
        single_reqs = [request(plain_messages(q), q.id, "single", i) for i in range(SINGLE_SAMPLES)]
        swarm_reqs = [request(lens_messages(q, i), q.id, "swarm", i) for i in range(K)]
        first = await asyncio.gather(*(self.ask(r, q.kind) for r in single_reqs + swarm_reqs))
        single = [_reply(q.kind, r) for r in first[:SINGLE_SAMPLES]]
        swarm = [_reply(q.kind, r) for r in first[SINGLE_SAMPLES:]]
        seen_reqs = []
        for i in range(K):
            peers = [(j, _show(q.kind, swarm[j][0]), swarm[j][1]) for j in range(K) if j != i]
            seen_reqs.append(request(seen_messages(q, i, peers), q.id, "seen", i))
        second = await asyncio.gather(*(self.ask(r, q.kind) for r in seen_reqs))
        seen = [_reply(q.kind, r) for r in second]
        return {"id": q.id, "kind": q.kind, "group": q.group,
                "single": [a for a, _ in single], "swarm": [a for a, _ in swarm], "seen": [a for a, _ in seen],
                "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in first + second)}

    async def run(self, questions: Sequence[Question], *, concurrency: int = 4,
                  on_done: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)
        results: Dict[str, Dict[str, Any]] = {}

        async def one(q: Question) -> None:
            async with sem:
                res = await self.run_question(q)
                results[q.id] = res
                if on_done:
                    on_done(res)

        await asyncio.gather(*(one(q) for q in questions))
        return [results[q.id] for q in questions if q.id in results]


def _reply(kind: str, row: Mapping[str, Any]) -> Tuple[Optional[Any], str]:
    """A cached call's (answer, reason); a call we refused (wrong model, cut off, no reply) has no answer."""
    return (None, "") if row.get("error") else parse_reply(kind, row["content"])


def _show(kind: str, answer: Any) -> Optional[str]:
    if answer is None:
        return None
    return str(answer) if kind == "exam" else f"{answer:.2f} chance of YES"


# -------------------------------------------------------------------- scoring

def arm_answers(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The combined answer of every arm for one question."""
    kind = row["kind"]
    single, swarm, seen = row["single"], row["swarm"], row["seen"]
    return {
        "single1": combine(kind, single[:1]),
        "single5": combine(kind, single[:K]),
        "single10": combine(kind, single[:SINGLE_SAMPLES]),
        "swarm5": combine(kind, swarm),
        "seen10": combine(kind, seen),
    }


def score_item(kind: str, answer: Any, truth: Any) -> float:
    """Exam: 1 if right else 0. Forecast: Brier score (lower is better)."""
    if kind == "exam":
        return float(answer is not None and answer == truth)
    y = 1.0 if truth == "yes" else 0.0
    return (answer - y) ** 2


def paired_boot(diffs: Sequence[float], reps: int = 10000, seed: int = 20261029) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return st.mean(diffs), means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


# The preregistered pass lines (docs/plans/2026-10-29-fair-test.md). Changing these after a run voids the test.
PASS = {"exam_min_gain": 0.01, "forecast_min_brier_gain": 0.005}
# Below this share of usable answers a run measured the plumbing, not the swarm: it gets no verdict at all.
MIN_USABLE = 0.9


def usable_share(rows: Sequence[Mapping[str, Any]]) -> float:
    answers = [a for r in rows for arm in ("single", "swarm", "seen") for a in r[arm]]
    return sum(a is not None for a in answers) / len(answers) if answers else 0.0


def score(rows: Sequence[Mapping[str, Any]], key: Mapping[str, Any],
          crowd: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    """Primary and secondary results with the preregistered verdict."""
    out: Dict[str, Any] = {"primary": {}, "secondary": {}}
    for kind in KINDS:
        items = [r for r in rows if r["kind"] == kind and r["id"] in key]
        if not items:
            continue
        per: Dict[str, List[float]] = {}
        for r in items:
            ans = arm_answers(r)
            for arm, a in ans.items():
                per.setdefault(arm, []).append(score_item(kind, a, key[r["id"]]))
        means = {arm: st.mean(v) for arm, v in per.items()}
        # Primary: swarm5 against single5. Gain is positive when the swarm is better.
        sign = 1.0 if kind == "exam" else -1.0
        diffs = [sign * (a - b) for a, b in zip(per["swarm5"], per["single5"])]
        gain, lo, hi = paired_boot(diffs)
        need = PASS["exam_min_gain"] if kind == "exam" else PASS["forecast_min_brier_gain"]
        out["primary"][kind] = {"n": len(items), "metric": "accuracy" if kind == "exam" else "brier",
                                "swarm5": round(means["swarm5"], 4), "single5": round(means["single5"], 4),
                                "gain": round(gain, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                "pass": gain >= need and lo > 0}
        sec = {"means": {k: round(v, 4) for k, v in means.items()}}
        d2 = [sign * (a - b) for a, b in zip(per["seen10"], per["single10"])]
        g2, l2, h2 = paired_boot(d2)
        sec["seen10_vs_single10"] = {"gain": round(g2, 4), "ci95": [round(l2, 4), round(h2, 4)]}
        d3 = [sign * (a - b) for a, b in zip(per["seen10"], per["swarm5"])]
        g3, l3, h3 = paired_boot(d3)
        sec["seen10_vs_swarm5"] = {"gain": round(g3, 4), "ci95": [round(l3, 4), round(h3, 4)]}
        groups: Dict[str, Dict[str, List[float]]] = {}
        for r, a5, s5 in zip(items, per["swarm5"], per["single5"]):
            g = groups.setdefault(r["group"], {"swarm5": [], "single5": []})
            g["swarm5"].append(a5)
            g["single5"].append(s5)
        sec["by_group"] = {g: {"n": len(v["swarm5"]), "swarm5": round(st.mean(v["swarm5"]), 4),
                               "single5": round(st.mean(v["single5"]), 4)} for g, v in sorted(groups.items())}
        lens_scores = []
        for i in range(K):
            lens_scores.append(round(st.mean(score_item(kind, (r["swarm"][i] if kind == "exam" else
                                                               mean_prob([r["swarm"][i]])), key[r["id"]])
                                             for r in items), 4))
        sec["per_lens"] = lens_scores
        if kind == "forecast" and crowd:
            with_crowd = [(r, crowd[r["id"]]) for r in items if r["id"] in crowd]
            if with_crowd:
                sec["crowd"] = {"n": len(with_crowd),
                                "crowd_brier": round(st.mean(score_item(kind, c, key[r["id"]]) for r, c in with_crowd), 4),
                                "swarm5_brier": round(st.mean(score_item(kind, arm_answers(r)["swarm5"], key[r["id"]])
                                                              for r, _ in with_crowd), 4)}
        out["secondary"][kind] = sec
    passed = [v["pass"] for v in out["primary"].values()]
    out["usable_share"] = round(usable_share(rows), 4)
    if out["usable_share"] < MIN_USABLE:
        out["verdict"] = (f"invalid run: only {out['usable_share']:.0%} of answers were usable "
                          f"(at least {MIN_USABLE:.0%} needed), so nothing was measured")
    else:
        out["verdict"] = ("supported" if passed and all(passed) else "mixed" if any(passed) else "not supported")
    return out


# --------------------------------------------------------------- cost estimate

def estimate(questions: Sequence[Question], *, reason_tokens: int = 160) -> Dict[str, Any]:
    """Tokens and dollars before any call: prompt tokens ~ characters / 4, replies ~ ``reason_tokens``."""
    calls = tin = 0
    for q in questions:
        base = len(SYSTEM) + len(q.text) + len(_format(q.kind)) + 40
        tin += SINGLE_SAMPLES * base + K * (base + 90) + K * (base + 90 + 4 * 330)
        calls += SINGLE_SAMPLES + 2 * K
    tin //= 4
    tout = calls * reason_tokens
    usd = (tin * PRICE[0] + tout * PRICE[1]) / 1_000_000
    return {"questions": len(questions), "calls": calls, "input_tokens": tin, "output_tokens": tout,
            "usd": round(usd, 2)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = ["BrokenRun", "OutOfCredits", "K", "KINDS", "LENSES", "MODEL", "PASS", "PRICE", "REASONING", "RELEASED", "Question", "Runner", "SpendCap", "arm_answers",
           "combine", "estimate", "lens_messages", "load_questions", "mean_prob", "parse_reply", "plain_messages",
           "out_of_credits", "score", "score_item", "seen_messages", "served_ok", "sha256_file", "usable_share", "vote"]
