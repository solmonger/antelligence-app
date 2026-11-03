"""Five parts borrowed from other harnesses and papers, each one a switch, so each can be scored alone and together.

Eshaan, 2026-11-02: "we can eval each of them as features and see what works best maybe?"

========  ==============================  ===========================================================  ==============
switch    borrowed from                   what it does here                                            needs
========  ==============================  ===========================================================  ==============
stop      Adaptive-Consistency            ask ants one at a time and stop when a small odds check      saved answers
          (Aggarwal et al., 2023)         says more answers are unlikely to change the vote (instead
                                          of "stop when k votes ahead")
gate      DeepConf (Fu et al., 2025),     read how sure a small ant was of its letter; if sure, stop   a live sweep
          AutoMix, FrugalGPT              there, else ask more ants or the reader; or weigh each       with answer
                                          ant's vote by how sure it was                                odds
router    RouteLLM (Ong et al., 2024),    pick a recipe per question from the most similar past        saved answers
          its similarity-weighted         questions (word overlap), not per coarse kind of question
          ranking router
cache     DeepSeek Harness's cost story   count only the prompt tokens a model had to read fresh:      saved answers
          (cache hits ~2% of the price)   the same question sent to the same model with another way    (estimate);
                                          of looking starts the same, and a prefix-caching server     live (measured)
                                          reads that start once. The learner then optimises that count.
prompt    GEPA, as used by Hermes Agent's a small loop that rewrites the ants' fixed instructions to  live
          self-evolution                  be shorter and keeps a rewrite only if accuracy holds        (promptopt.py)
========  ==============================  ===========================================================  ==============

The first four live here; ``prompt`` is :mod:`antelligence.harness.promptopt`. :func:`evaluate_features` scores
each switch alone and every mix of them the same way the harness is scored today: the cost weight is tuned and the
policy learned on the training sets only (k folds inside them), the mix is chosen on the training sets only, and the
test set is scored once.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from antelligence.experiments import fair_test as ft
from antelligence.experiments import shared_read as sr
from antelligence.harness import learner as ln
from antelligence.harness import scorecard as sc
from antelligence.harness.core import Step, Task, bucket, vote
from antelligence.harness.recipes import Ctx, Recipe, cheap_first, menu
from antelligence.harness.tools import ARMS, AntInfo, Box, Missing, ReplayBox, ant_tool, parse_ant

FEATURES = ("stop", "gate", "router", "cache")


# ===================================================================== 1. stop rule

def beta_conf(v1: int, v2: int) -> float:
    """Adaptive-Consistency's Beta stopping score for a leader with ``v1`` votes and a runner-up with ``v2``: the
    chance, under a flat prior, that the leader's true share is larger than the runner-up's.

    It is ``I_0.5(v2 + 1, v1 + 1)``, which for whole numbers is exactly ``P(Binomial(v1 + v2 + 1, 1/2) >= v2 + 1)``.
    2-0 gives 0.875, 3-0 0.9375, 4-0 0.969, 3-1 0.8125, 5-1 0.9375, 6-1 0.965."""
    if v1 <= v2:
        return 0.0 if v1 == v2 else 1.0 - beta_conf(v2, v1)
    n = v1 + v2 + 1
    return sum(math.comb(n, j) for j in range(v2 + 1, n + 1)) / 2 ** n


def top_two(answers: Sequence[Optional[str]]) -> Tuple[int, int]:
    counts = sorted((sum(1 for a in answers if a == x) for x in set(answers) if x is not None), reverse=True)
    return (counts[0] if counts else 0), (counts[1] if len(counts) > 1 else 0)


async def _escalate(ctx: Ctx, tool: str, answers: Sequence[Optional[str]]) -> Optional[str]:
    got = await ctx.call(tool)
    if got.answer is not None:
        return got.answer
    if got.passes:                                 # the code checker narrowed it down: vote inside that
        inside = [x for x in answers if x in got.passes]
        if inside:
            return vote(inside)
    return vote(list(answers))


@dataclass(frozen=True)
class SeqBeta(Recipe):
    """Ants one at a time, cheapest first; stop as soon as :func:`beta_conf` reaches ``conf``. With ``escalate``,
    hand the question up as soon as even every ant left agreeing with the leader could not get there."""

    arm: str = "alone"
    conf: float = 0.9
    escalate: Optional[str] = None
    n: int = 8
    order: Tuple[int, ...] = ()

    async def run(self, ctx: Ctx) -> Optional[str]:
        answers: List[Optional[str]] = []
        idx = list(self.order) or list(range(self.n))
        for k, i in enumerate(idx):
            answers.append((await ctx.call(ant_tool(i, self.arm))).answer)
            left = len(idx) - k - 1
            v1, v2 = top_two(answers)
            if v1 and beta_conf(v1, v2) >= self.conf:
                return vote(answers)
            if self.escalate and beta_conf(v1 + left, v2) < self.conf:
                break                              # can't get sure any more: hand it up now
            if not self.escalate and v1 - v2 > left:
                return vote(answers)               # the rest can't change the leader
        if self.escalate:
            return await _escalate(ctx, self.escalate, answers)
        return vote(answers)


STOP_CONFS = (0.8, 0.9, 0.95)


def stop_recipes(roster: Sequence[AntInfo], arms: Sequence[str] = ARMS) -> List[Recipe]:
    n, order = len(roster), tuple(cheap_first(roster))
    out: List[Recipe] = []
    for arm in arms:
        ants = tuple(ant_tool(i, arm) for i in range(n))
        for c in STOP_CONFS:
            tag = f"ac:{arm}:c{round(c * 100)}"
            out.append(SeqBeta(tag, ants, arm, c, None, n, order))
            for up in ("reader", "code"):
                out.append(SeqBeta(f"{tag}>{up}", ants + (up,), arm, c, up, n, order))
    return out


# ===================================================================== 2. confidence gate

@dataclass(frozen=True)
class Gate(Recipe):
    """Ask up to ``tries`` ants, cheapest first; take the first answer the ant was at least ``sure`` of (its chance
    for that letter, from the answer odds). If none was, ask ``escalate``; without it, keep the surest answer."""

    arm: str = "alone"
    sure: float = 0.9
    tries: int = 1
    escalate: Optional[str] = "reader"
    order: Tuple[int, ...] = ()

    async def run(self, ctx: Ctx) -> Optional[str]:
        seen: List[Step] = []
        for i in list(self.order)[: self.tries]:
            s = await ctx.call(ant_tool(i, self.arm))
            seen.append(s)
            if s.answer is not None and s.sure is not None and s.sure >= self.sure:
                return s.answer
        if self.escalate:
            got = await ctx.call(self.escalate)
            if got.answer is not None:
                return got.answer
        sure = [s for s in seen if s.answer is not None]
        return max(sure, key=lambda s: s.sure or 0.0).answer if sure else None


@dataclass(frozen=True)
class ConfVote(Recipe):
    """DeepConf's online vote: ants one at a time, cheapest first, each vote weighted by how sure the ant was of its
    letter (1 when unknown). Stop once the leader holds ``share`` of the weight with at least ``min_n`` answers. With
    ``escalate``, hand up as soon as even full-weight agreement from every ant left could not get there."""

    arm: str = "alone"
    share: float = 0.9
    escalate: Optional[str] = None
    min_n: int = 2
    order: Tuple[int, ...] = ()

    async def run(self, ctx: Ctx) -> Optional[str]:
        weight: Dict[str, float] = {}
        answers: List[Optional[str]] = []
        idx = list(self.order)
        for k, i in enumerate(idx):
            s = await ctx.call(ant_tool(i, self.arm))
            answers.append(s.answer)
            if s.answer is not None:
                weight[s.answer] = weight.get(s.answer, 0.0) + (s.sure if s.sure is not None else 1.0)
            total = sum(weight.values())
            left = len(idx) - k - 1
            lead = max(weight.values()) if weight else 0.0
            if total > 0 and len([a for a in answers if a]) >= self.min_n and lead / total >= self.share:
                break
            if self.escalate and (lead + left) / (total + left or 1.0) < self.share:
                return await _escalate(ctx, self.escalate, answers)
        else:
            if self.escalate:
                total = sum(weight.values())
                if not weight or max(weight.values()) / total < self.share:
                    return await _escalate(ctx, self.escalate, answers)
        if not weight:
            return None
        best = max(weight.values())
        return next(a for a in answers if a is not None and weight[a] == best)


GATE_SURE = (0.7, 0.85, 0.95)
GATE_TRIES = (1, 2, 4)
CVOTE_SHARE = (0.8, 0.9, 0.95)


def gate_recipes(roster: Sequence[AntInfo], arms: Sequence[str] = ("alone",)) -> List[Recipe]:
    n, order = len(roster), tuple(cheap_first(roster))
    out: List[Recipe] = []
    for arm in arms:
        for p in GATE_SURE:
            for t in GATE_TRIES:
                tools = tuple(ant_tool(i, arm) for i in order[:t]) + ("reader",)
                out.append(Gate(f"gate:{arm}:p{round(p * 100)}:t{t}>reader", tools, arm, p, t, "reader", order))
        ants = tuple(ant_tool(i, arm) for i in range(n))
        for c in CVOTE_SHARE:
            out.append(ConfVote(f"cvote:{arm}:c{round(c * 100)}", ants, arm, c, None, 2, order))
            out.append(ConfVote(f"cvote:{arm}:c{round(c * 100)}>reader", ants + ("reader",), arm, c, "reader", 2,
                                order))
    return out


def has_sure(steps: Mapping[str, Mapping[str, Step]], arm: str = "alone", min_share: float = 0.5) -> bool:
    """Do most of this arm's saved ant answers carry answer odds?"""
    got = [s for per in steps.values() for t, s in per.items()
           if (parse_ant(t) or (0, ""))[1] == arm and s.answer is not None]
    return bool(got) and sum(s.sure is not None for s in got) >= min_share * len(got)


# ===================================================================== 4. cache-aware token count

def _render(messages: Sequence[Mapping[str, str]]) -> str:
    return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)


def _lcp(a: str, b: str) -> int:
    """Length of the longest common start (binary search on slices, so it runs at C speed)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class Prompts:
    """Rebuilds the exact text each saved step was asked with (where it can), to see which calls start the same."""

    def __init__(self, roster: Sequence[AntInfo], reader: Optional[sr.ModelSpec] = None,
                 specs: Optional[Mapping[str, sr.ModelSpec]] = None,
                 ant_messages: Optional[Callable[..., Any]] = None) -> None:
        self.roster = tuple(roster)
        self.reader = reader
        self.specs = dict(specs or {})
        self.ant_messages = ant_messages or sr.ant_messages
        self._text: Dict[Tuple[str, str], Optional[str]] = {}
        self._shared: Dict[Tuple[str, str, str], int] = {}

    def shared(self, task: Task, a: str, b: str, done: Mapping[str, Step]) -> int:
        """Characters prompts ``a`` and ``b`` start with in common on ``task`` (remembered)."""
        k = (task.id, a, b) if a <= b else (task.id, b, a)
        if k not in self._shared:
            ta, tb = self.text(a, task, done), self.text(b, task, done)
            self._shared[k] = _lcp(ta, tb) if ta and tb else 0
        return self._shared[k]

    def model(self, tool: str) -> Optional[str]:
        a = parse_ant(tool)
        if a:
            return self.roster[a[0]].tag if a[0] < len(self.roster) else None
        if tool in ("reader", "notes"):
            return "__reader__"
        return None                                # code (its retries vary) and decision models: counted in full

    def text(self, tool: str, task: Task, done: Mapping[str, Step]) -> Optional[str]:
        k = (task.id, tool)
        if k not in self._text:
            self._text[k] = self._build(tool, task, done)
        return self._text[k]

    def _build(self, tool: str, task: Task, done: Mapping[str, Step]) -> Optional[str]:
        item = sr.Item(task.id, task.kind, task.passage, task.stem, task.options)
        a = parse_ant(tool)
        try:
            if a:
                info = self.roster[a[0]]
                spec = self.specs.get(info.tag) or sr.ModelSpec(info.tag, info.family, info.weight)
                notes = (done.get("notes").payload.get("notes") if done.get("notes") else "") or ""
                if a[1] != "alone" and not notes:
                    return None
                return _render(self.ant_messages(item, a[1], info.lens, spec, notes or None))
            spec = self.reader or sr.ModelSpec("reader", "reader")
            if tool == "reader":
                return _render(sr.reader_answer_messages(item, spec))
            if tool == "notes":
                return _render(sr.notes_messages(item, spec))
        except (IndexError, ValueError):
            return None
        return None


def fresh_in(steps: Sequence[Step], task: Task, prompts: Prompts, rate: Mapping[str, float],
             measured: bool = False) -> List[int]:
    """For each step (in the order a recipe asked them), the prompt tokens its model had to read fresh.

    Estimated: the longest start the step's prompt shares with an earlier prompt to the same model on this question,
    in characters, times that model's tokens per character. A step with a short re-ask also gets its first prompt back
    from the cache (the re-ask starts with it). Prompts of other questions are not counted as shared (only the short
    system line would be), so the estimate leans low.

    ``measured=True`` takes the server's own count (``payload["cached_in"]``) where a step has one. That is right only
    for steps replayed in the order they were really asked (a sweep's), not for a recipe that asks a subset in another
    order, so scoring recipes uses the estimate and the measured count is used to check it."""
    seen: Dict[str, List[str]] = {}
    done: Dict[str, Step] = {}
    out: List[int] = []
    for s in steps:
        cached: Optional[float] = None
        if measured and isinstance(s.payload.get("cached_in"), int):
            cached = float(s.payload["cached_in"])
        model = prompts.model(s.tool)
        text = prompts.text(s.tool, task, done) if model else None
        if cached is None and text is not None:
            r = rate.get(model, 0.0)
            prev = max((prompts.shared(task, s.tool, t, done) for t in seen.get(model, [])), default=0)
            cached = prev * r if prev >= 40 else 0.0
            if s.calls >= 2 and s.tool != "code":
                cached += len(text) * r            # the re-ask starts with the whole first prompt
        if text is not None and model:
            seen.setdefault(model, []).append(s.tool)
        done[s.tool] = s
        out.append(max(0, s.tokens_in - int(round(min(cached or 0.0, s.tokens_in)))))
    return out


def token_rates(tasks: Sequence[Task], steps: Mapping[str, Mapping[str, Step]], prompts: Prompts) -> Dict[str, float]:
    """Each model's prompt tokens per character, from its single-call steps (median), to turn shared text into
    tokens."""
    import statistics

    by: Dict[str, List[float]] = {}
    tid = {t.id: t for t in tasks}
    for i, per in steps.items():
        if i not in tid:
            continue
        for tool, s in per.items():
            model = prompts.model(tool)
            if not model or s.calls != 1 or s.tokens_in <= 0:
                continue
            text = prompts.text(tool, tid[i], per)
            if text:
                by.setdefault(model, []).append(s.tokens_in / len(text))
    return {m: statistics.median(v) for m, v in by.items() if v}


# ===================================================================== tables (raw and fresh counts)

@dataclass
class Tables:
    raw: ln.Table
    fresh: ln.Table
    names: List[str] = field(default_factory=list)


async def _tables(recipes: Sequence[Recipe], box: Box, tasks: Sequence[Task], key: Mapping[str, str],
                  prompts: Prompts, rate: Mapping[str, float]) -> Tables:
    raw: ln.Table = {r.name: {} for r in recipes}
    fresh: ln.Table = {r.name: {} for r in recipes}
    for r in recipes:
        for t in tasks:
            ctx = Ctx(box, t)
            try:
                answer = await r.run(ctx)
            except Missing:
                continue
            o = sc.bill(t, r.name, answer, ctx.steps, key)
            raw[r.name][t.id] = o
            f_in = fresh_in(ctx.steps, t, prompts, rate)
            fresh_tokens = sum(f_in) + sum(s.tokens_out for s in ctx.steps)
            fresh_w = sum((fi + s.tokens_out) * (s.weight or 1.0) for fi, s in zip(f_in, ctx.steps))
            fresh[r.name][t.id] = sc.Outcome(**{**o.to_json(), "tokens": fresh_tokens,
                                                 "weighted_tokens": round(fresh_w, 1)})
    return Tables(raw, fresh, [r.name for r in recipes])


def tables(recipes: Sequence[Recipe], box: Box, tasks: Sequence[Task], key: Mapping[str, str], prompts: Prompts,
           rate: Mapping[str, float]) -> Tables:
    return asyncio.run(_tables(recipes, box, tasks, key, prompts, rate))


# ===================================================================== 3. router (nearest past questions)

_WORD = re.compile(r"[a-z]+|\d+")


def words(task: Task) -> List[str]:
    """The task's words, plus its coarse kind as one more word, so the router also knows what the bucket learner
    knows."""
    return _WORD.findall(task.full.lower()) + [f"__{bucket(task)}"]


class Vectors:
    """TF-IDF over the training questions' words (fitted on training questions only). Words in more than
    ``max_df`` of the training questions are left out: words every question has ("which", the options' format) make
    every question look alike and so tell the router nothing."""

    def __init__(self, tasks: Sequence[Task], max_df: float = 0.5) -> None:
        docs = [set(words(t)) for t in tasks]
        n = len(docs)
        df: Dict[str, int] = {}
        for d in docs:
            for w in d:
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items() if c <= max_df * n}

    def __call__(self, task: Task) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for w in words(task):
            if w in self.idf:
                tf[w] = tf.get(w, 0.0) + 1.0
        v = {w: (1.0 + math.log(c)) * self.idf[w] for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}


def _cos(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


@dataclass
class KnnPolicy:
    """RouteLLM's similarity-weighted ranking, in miniature: for a new question, each recipe's value on the ``k`` most
    similar training questions, weighted by how similar they are, and pulled toward the recipe's overall value as if
    the overall value were ``shrink`` more (fully similar) questions. Picking costs no model call."""

    default: str
    names: List[str]
    vec: Vectors
    train: List[Tuple[Dict[str, float], Dict[str, float]]]   # (vector, recipe -> value) per training question
    overall: Dict[str, float]
    k: int = 20
    shrink: float = 3.0
    weights: ln.Weights = field(default_factory=ln.Weights)

    def pick(self, task: Task) -> str:
        v = self.vec(task)
        sims = sorted(((_cos(v, tv), vals) for tv, vals in self.train), key=lambda x: -x[0])[: self.k]
        sims = [(s, vals) for s, vals in sims if s > 0]
        wsum = sum(s for s, _ in sims)
        best, best_v = self.default, None
        for n in self.names:
            est = (sum(s * vals[n] for s, vals in sims) + self.shrink * self.overall[n]) / (wsum + self.shrink)
            if best_v is None or round(est, 9) > round(best_v, 9) or (round(est, 9) == round(best_v, 9)
                                                                          and n == self.default):
                best, best_v = n, est
        return best

    def to_json(self) -> Dict[str, Any]:
        return {"kind": "knn", "default": self.default, "k": self.k, "shrink": self.shrink,
                "weights": vars(self.weights), "recipes": len(self.names), "train": len(self.train)}


def _ref(table: ln.Table, ids: Sequence[str], baseline: str) -> Dict[str, float]:
    base = [table[baseline][i] for i in ids]
    return {"tokens": sum(o.tokens for o in base) / len(base), "seconds": sum(o.seconds for o in base) / len(base)}


def fit_knn(table: ln.Table, train: Sequence[Task], *, weights: ln.Weights = ln.Weights(), baseline: str = "reader",
            names: Optional[Sequence[str]] = None, k: int = 20, shrink: float = 3.0) -> KnnPolicy:
    names = list(names) if names is not None else ln.common(table, train)[0]
    keep = [t for t in train if all(t.id in table[n] for n in names)]
    if not keep:
        raise ValueError("no training task has every recipe scored")
    ids = [t.id for t in keep]
    ref = _ref(table, ids, baseline)
    val = {i: {n: ln.value(table[n][i], ref, weights) for n in names} for i in ids}
    overall = {n: sum(val[i][n] for i in ids) / len(ids) for n in names}
    default = max(names, key=lambda n: (round(overall[n], 9), n == baseline))
    vec = Vectors(keep)
    return KnnPolicy(default, names, vec, [(vec(t), val[t.id]) for t in keep], overall, k, shrink, weights)


def fit_bucket(table: ln.Table, train: Sequence[Task], *, weights: ln.Weights = ln.Weights(),
               baseline: str = "reader", names: Optional[Sequence[str]] = None) -> ln.Policy:
    return ln.learn(table, train, weights=weights, baseline=baseline, names=names)


Fit = Callable[..., Any]          # (table, train, weights=, names=) -> anything with .pick(task) -> recipe name


# ===================================================================== learning, generically

def picks_k_fold(fit: Fit, table: ln.Table, tasks: Sequence[Task], names: Sequence[str], w: ln.Weights, k: int = 5,
                 seed: int = 20261102) -> Dict[str, str]:
    """task id -> the recipe a policy learned without that task picks (folds cut within each kind, as
    :func:`learner.k_fold` does)."""
    rng = random.Random(seed)
    by_b: Dict[str, List[Task]] = {}
    for t in tasks:
        by_b.setdefault(bucket(t), []).append(t)
    fold: Dict[str, int] = {}
    for b in sorted(by_b):
        members = sorted(by_b[b], key=lambda t: t.id)
        rng.shuffle(members)
        for j, t in enumerate(members):
            fold[t.id] = j % k
    out: Dict[str, str] = {}
    for f in range(k):
        train = [t for t in tasks if fold[t.id] != f]
        test = [t for t in tasks if fold[t.id] == f]
        if not train or not test:
            continue
        pol = fit(table, train, weights=w, names=names)
        out.update({t.id: pol.pick(t) for t in test})
    return out


def routed(picks: Mapping[str, str], table: ln.Table, name: str = "harness") -> List[sc.Outcome]:
    return [sc.Outcome(**{**table[r][i].to_json(), "recipe": name}) for i, r in picks.items() if i in table.get(r, {})]


def tune_fit(fit: Fit, table: ln.Table, train: Sequence[Task], names: Sequence[str],
             grid: Sequence[float] = ln.COST_GRID, k: int = 5, baseline: str = "reader"
             ) -> Tuple[ln.Weights, List[Dict[str, Any]], Dict[str, Any]]:
    """:func:`learner.tune` for any learner: pick the cost weight by k folds inside the training sets, with the same
    three-step rule. Also returns the picked row, so mixes of switches can be compared on training questions."""
    base = [table[baseline][t.id] for t in train]
    rows: List[Dict[str, Any]] = []
    for c in grid:
        w = ln.Weights(tokens=c, seconds=c)
        outs = routed(picks_k_fold(fit, table, train, names, w, k), table)
        cmp = sc.compare(outs, base)
        acc = sum(o.right for o in outs if o.right is not None) / max(1, len(outs))
        rows.append({"weight": c, "accuracy": round(acc, 4), "gain": cmp.get("accuracy_gain"),
                     "gain_ci95": cmp.get("accuracy_ci95"), "tokens_ratio": cmp.get("tokens_ratio"),
                     "seconds_ratio": cmp.get("seconds_ratio"), "tokens_ratio_ci95": cmp.get("tokens_ratio_ci95"),
                     "seconds_ratio_ci95": cmp.get("seconds_ratio_ci95")})
    wins = [r for r in rows if ln._sure_cheaper(r) and (r["gain_ci95"] or [0, 0])[0] > 0]
    ties = [r for r in rows if ln._sure_cheaper(r) and (r["gain_ci95"] or [-1, -1])[1] > 0]

    def best(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return max(rs, key=lambda r: (r["accuracy"], -r["weight"]))
    if wins:
        pick, rank, why = best(wins), 2, "beats the reader on all three with confidence"
    elif ties:
        pick, rank, why = best(ties), 1, "cheaper on both with confidence, not surely less accurate"
    else:
        pick, rank, why = rows[0], 0, "no weight was surely cheaper on both, so accuracy first"
    return ln.Weights(tokens=pick["weight"], seconds=pick["weight"]), rows, {**pick, "rank": rank, "why": why}


# ===================================================================== scoring the switches

@dataclass(frozen=True)
class Config:
    """One mix of switches."""

    stop: bool = False
    gate: bool = False
    router: bool = False
    cache: bool = False

    @property
    def on(self) -> Tuple[str, ...]:
        return tuple(f for f in FEATURES if getattr(self, f))

    @property
    def label(self) -> str:
        return "harness" + "".join(f" +{f}" for f in self.on) if self.on else "harness (today)"


def configs(available: Sequence[str]) -> List[Config]:
    """Today's harness, then every mix of the available switches (one alone first)."""
    out = []
    for r in range(0, len(available) + 1):
        for combo in itertools.combinations(available, r):
            out.append(Config(**{f: True for f in combo}))
    return out


def recipe_pool(roster: Sequence[AntInfo], dms: Sequence[str], conf_arms: Sequence[str]) -> Dict[str, List[Recipe]]:
    return {"base": menu(roster, dms), "stop": stop_recipes(roster), "gate": gate_recipes(roster, conf_arms)}


def _names_for(cfg: Config, pool: Mapping[str, Sequence[Recipe]], fit_names: Sequence[str]) -> List[str]:
    want = [r.name for r in pool["base"]]
    if cfg.stop:
        want += [r.name for r in pool["stop"]]
    if cfg.gate:
        want += [r.name for r in pool["gate"]]
    have = set(fit_names)
    return [n for n in want if n in have]


def _side(outs: Sequence[sc.Outcome], base: Sequence[sc.Outcome]) -> Dict[str, Any]:
    t = sc.totals(outs)
    c = sc.compare(outs, base)
    return {"accuracy": t["accuracy"], "tokens": t["tokens"], "weighted_tokens": t["weighted_tokens"],
            "seconds": t["seconds"], "gain": c.get("accuracy_gain"), "gain_ci95": c.get("accuracy_ci95"),
            "tokens_ratio": c.get("tokens_ratio"), "tokens_ratio_ci95": c.get("tokens_ratio_ci95"),
            "seconds_ratio": c.get("seconds_ratio"), "seconds_ratio_ci95": c.get("seconds_ratio_ci95"),
            "weighted_tokens_ratio": c.get("weighted_tokens_ratio"), "verdict": c.get("verdict")}


def switch_use(choice: Mapping[str, str], tabs: Tables) -> Dict[str, int]:
    """On how many of the routed questions each switch actually did something: a stop-rule recipe was picked, a gate
    recipe was picked, or the picked route read fewer fresh prompt tokens than it was sent (the cache count changed
    its bill). A switch that is on but at zero here was never used on those questions, so its row says nothing about
    whether it helps."""
    def cached(i: str, r: str) -> bool:
        raw, fresh = tabs.raw.get(r, {}), tabs.fresh.get(r, {})
        return i in raw and i in fresh and fresh[i].tokens < raw[i].tokens
    picks = list(choice.items())
    return {"questions": len(picks),
            "reader": sum(r == "reader" for _, r in picks),
            "stop": sum(r.startswith("ac:") for _, r in picks),
            "gate": sum(r.startswith(("gate:", "cvote:")) for _, r in picks),
            "cache": sum(cached(i, r) for i, r in picks)}


def use_text(use: Mapping[str, int], on: Sequence[str]) -> str:
    """'stop 0, cache 0 of 160' for the switches that are on ('-' for today's harness). The router has no count: it
    changes every pick, so its effect is the row itself."""
    parts = [f"{f} {use.get(f, 0)}" for f in on if f in ("stop", "gate", "cache")]
    return f"{', '.join(parts)} of {use.get('questions', 0)}" if parts else "-"


def score_config(cfg: Config, tabs: Tables, pool: Mapping[str, Sequence[Recipe]], train: Sequence[Task],
                 test: Sequence[Task], fit_names: Sequence[str], k: int = 5) -> Dict[str, Any]:
    """Tune the cost weight and learn on ``train`` only, route ``test`` once. Reports the test score both ways: every
    prompt token counted, and only the ones the models had to read fresh."""
    names = _names_for(cfg, pool, fit_names)
    learn_on = tabs.fresh if cfg.cache else tabs.raw
    fit: Fit = fit_knn if cfg.router else fit_bucket
    w, grid, pick = tune_fit(fit, learn_on, train, names, k=k)
    pol = fit(learn_on, train, weights=w, names=names)
    choice = {t.id: pol.pick(t) for t in test}
    out_raw, out_fresh = routed(choice, tabs.raw), routed(choice, tabs.fresh)
    base_raw = [tabs.raw["reader"][t.id] for t in test]
    base_fresh = [tabs.fresh["reader"][t.id] for t in test]
    used: Dict[str, int] = {}
    for r in choice.values():
        used[r] = used.get(r, 0) + 1
    return {"config": cfg.label, "on": list(cfg.on), "recipes": len(names), "weight": w.tokens, "choice": choice,
            "train": {k2: pick.get(k2) for k2 in ("accuracy", "gain", "gain_ci95", "tokens_ratio", "seconds_ratio",
                                                  "tokens_ratio_ci95", "seconds_ratio_ci95", "rank", "why")},
            "test": _side(out_raw, base_raw), "test_fresh": _side(out_fresh, base_fresh),
            "used": dict(sorted(used.items(), key=lambda kv: -kv[1])[:6]), "switch_use": switch_use(choice, tabs),
            "policy": pol.to_json() if hasattr(pol, "to_json") else {}, "grid": grid}


def evaluate_features(tasks: Sequence[Task], key: Mapping[str, str], steps: Mapping[str, Mapping[str, Step]],
                      roster: Sequence[AntInfo], train_ids: Iterable[str], test_ids: Iterable[str], *,
                      reader: Optional[sr.ModelSpec] = None, specs: Optional[Mapping[str, sr.ModelSpec]] = None,
                      features: Optional[Sequence[str]] = None, k: int = 5,
                      log: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """Every mix of the switches on the same saved work; pick the best mix on training questions only; score the
    test questions once per mix (so each switch alone and the picked mix are all reported)."""
    box = ReplayBox(steps, roster)
    dms = [t for t in ("dm:jev:dm1", "dm:jev:read", "dm:d1-3b:dm1", "dm:d1-3b:read") if box.has(t)]
    gate_ok = has_sure(steps, "alone")
    available = [f for f in (features or FEATURES) if f != "gate" or gate_ok]
    pool = recipe_pool(roster, dms, ("alone",) if gate_ok else ())
    recipes = [r for group in pool.values() for r in group if r.fits(box)]
    prompts = Prompts(roster, reader, specs)
    rate = token_rates(tasks, steps, prompts)
    log(f"scoring {len(recipes)} recipes on {len(tasks)} questions (raw and fresh token counts)")
    tabs = tables(recipes, box, tasks, key, prompts, rate)
    fit_names, keep = ln.common(tabs.raw, tasks)
    keep_ids = {t.id for t in keep}
    train = [t for t in tasks if t.id in set(train_ids) and t.id in keep_ids]
    test = [t for t in tasks if t.id in set(test_ids) and t.id in keep_ids]
    if not train or not test:
        raise ValueError("need training and test questions with every recipe scored")
    rows = []
    for cfg in configs(available):
        log(f"  {cfg.label}")
        rows.append(score_config(cfg, tabs, pool, train, test, fit_names, k))
    mixes = [r for r in rows if len(r["on"]) >= 1]
    best = max(mixes, key=lambda r: (r["train"]["rank"], r["train"]["accuracy"], -len(r["on"]))) if mixes else None
    reader_test = sc.totals([tabs.raw["reader"][t.id] for t in test])
    return {"n_train": len(train), "n_test": len(test), "available": available,
            "gate_data": gate_ok, "token_rates": {m: round(v, 4) for m, v in rate.items()},
            "reader": reader_test, "rows": rows, "best_mix": best["config"] if best else None,
            "best_mix_why": (f"picked on the training sets only: {best['train']['why']}, "
                             f"{pct(best['train']['accuracy'])} on them by 5-fold") if best else None,
            "_tabs": tabs, "_recipes": recipes, "_prompts": prompts, "_rate": rate}


def public(rep: Mapping[str, Any]) -> Dict[str, Any]:
    """A report without its working parts (the ``_`` keys and each row's per-question choices), for saving."""
    return {k: ([{kk: vv for kk, vv in r.items() if kk != "choice"} for r in v] if k == "rows" else v)
            for k, v in rep.items() if not k.startswith("_")}


def swap_score(rep: Mapping[str, Any], label: str, new_steps: Mapping[str, Mapping[str, Step]],
               roster: Sequence[AntInfo], tasks: Sequence[Task], key: Mapping[str, str],
               prompts: Optional[Prompts] = None) -> Dict[str, Any]:
    """The same routing choices (learned on the training sets), replayed on test-set work done differently, e.g. by
    ants asked with shorter instructions: does the harness keep its accuracy and save tokens with them?"""
    row = next(r for r in rep["rows"] if r["config"] == label)
    choice = row["choice"]
    test = [t for t in tasks if t.id in choice and t.id in new_steps]
    box = ReplayBox(new_steps, roster)
    used = {r for r in rep["_recipes"] if r.name in set(choice.values()) | {"reader"}}
    tabs = tables(list(used), box, test, key, prompts or rep["_prompts"], rep["_rate"])
    picks = {t.id: choice[t.id] for t in test if t.id in tabs.raw.get(choice[t.id], {})}
    base = [tabs.raw["reader"][i] for i in picks if i in tabs.raw["reader"]]
    return {"config": label, "n": len(picks), "test": _side(routed(picks, tabs.raw), base),
            "test_fresh": _side(routed(picks, tabs.fresh), [tabs.fresh["reader"][i] for i in picks
                                                              if i in tabs.fresh["reader"]])}


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _rng(r: Optional[Sequence[float]], scale: float = 1.0, fmt: str = "{:+.1f}") -> str:
    return f"{fmt.format(r[0] * scale)} to {fmt.format(r[1] * scale)}" if r else "n/a"


def table_lines(rep: Mapping[str, Any], title: str) -> List[str]:
    """The plain table to paste back: one line per mix (today's harness, each switch alone, the mixes)."""
    r = rep["reader"]
    lines = [f"{title}: learned on {rep['n_train']} training questions, scored once on {rep['n_test']} test "
             f"questions; switches available: {', '.join(rep['available'])}"
             + ("" if rep["gate_data"] else " (gate needs the answer-odds sweep)"),
             f"  reader: {pct(r['accuracy'])}, {r['tokens']:.0f} tokens, {r['seconds']:.1f}s a question",
             "  mix | test accuracy (points vs reader, range) | tokens vs reader (range) | fresh tokens vs reader | "
             "time vs reader (range) | verdict | switch used on (test questions)"]
    for row in rep["rows"]:
        t, f = row["test"], row["test_fresh"]
        lines.append(f"  {row['config']} | {pct(t['accuracy'])} ({(t['gain'] or 0) * 100:+.1f}, "
                     f"{_rng(t['gain_ci95'], 100)}) | x{t['tokens_ratio']} ({_rng(t['tokens_ratio_ci95'], 1, '{}')}) | "
                     f"x{f['tokens_ratio']} | x{t['seconds_ratio']} ({_rng(t['seconds_ratio_ci95'], 1, '{}')}) | "
                     f"{t['verdict']} | {use_text(row.get('switch_use', {}), row['on'])}")
    if rep.get("best_mix"):
        lines.append(f"  best mix: {rep['best_mix']} ({rep['best_mix_why']})")
    return lines


# ===================================================================== overlay of a new sweep

def overlay(old: Mapping[str, Mapping[str, Step]], new: Mapping[str, Mapping[str, Step]]
            ) -> Dict[str, Dict[str, Step]]:
    """Saved steps with a newer sweep's steps put in their place (same question ids, same tool names)."""
    out = {tid: dict(per) for tid, per in old.items()}
    for tid, per in new.items():
        if tid in out:
            out[tid].update(per)
    return out


def custom_ants(system: str, fmt: str) -> Callable[..., Tuple[Dict[str, str], ...]]:
    """An ``ant_messages`` with its fixed text (system line and reply form) swapped, everything else as the shared-read
    test asks it: used by the prompt search."""

    def ant_messages(item: sr.Item, arm: str, lens: int, spec: sr.ModelSpec, notes: Optional[str] = None
                     ) -> Tuple[Dict[str, str], ...]:
        msgs = sr.ant_messages(item, arm, lens, spec, notes)
        user = msgs[1]["content"].replace(ft._format("exam"), fmt)
        return ({"role": "system", "content": system}, {"role": "user", "content": user})
    return ant_messages


class _HasPick(Protocol):
    def pick(self, task: Task) -> str: ...
