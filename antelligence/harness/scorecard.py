"""The scorecard: every request's answer and full bill, and a plain comparison with "always ask one model".

Each request becomes one :class:`Outcome`: which recipe ran, its answer, whether it was right (when the key is
known), and the totals of every step it took: tokens (in plus out), tokens weighted by model size, seconds, dollars
and model calls. Seconds are the sum of the steps' own times, because on one Mac the models run one after another;
time spent loading a model is not counted (the importer replaces a model's load-time outliers with its typical time).

The comparison against the baseline (``reader`` by default) is paired, question by question, with a bootstrap range
for the accuracy gap. The verdict "beats it on all three" needs all of:

* accuracy: the gap's 95% range above 0;
* tokens: fewer on average;
* time: less on average.

"Ties it, and is cheaper" means the accuracy range includes 0 but tokens and time are both lower.
"""

from __future__ import annotations

import json
import random
import statistics as st
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from antelligence.experiments import fair_test as ft
from antelligence.harness.core import Step, Task, bucket
from antelligence.harness.recipes import Ctx, Recipe
from antelligence.harness.tools import Box


@dataclass(frozen=True)
class Outcome:
    task: str
    kind: str                       # the dev tag, for reports only
    bucket: str
    recipe: str
    answer: Optional[str]
    right: Optional[float]          # 1.0, 0.0, or None when there is no key
    tokens: int
    weighted_tokens: float
    seconds: float
    cost_usd: float
    calls: int
    tools: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def bill(task: Task, recipe: str, answer: Optional[str], steps: Sequence[Step],
         key: Optional[Mapping[str, str]] = None) -> Outcome:
    truth = (key or {}).get(task.id)
    return Outcome(task.id, task.kind, bucket(task), recipe, answer,
                   None if truth is None else float(answer is not None and answer == truth),
                   sum(s.tokens for s in steps), round(sum(s.weighted_tokens for s in steps), 1),
                   round(sum(s.latency_s for s in steps), 4), round(sum(s.cost_usd for s in steps), 6),
                   sum(s.calls for s in steps), [s.tool for s in steps])


async def run_one(recipe: Recipe, box: Box, task: Task, key: Optional[Mapping[str, str]] = None) -> Outcome:
    ctx = Ctx(box, task)
    answer = await recipe.run(ctx)
    return bill(task, recipe.name, answer, ctx.steps, key)


class Log:
    """Append-only JSON-lines log of outcomes: the harness's memory of what worked."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def add(self, o: Outcome) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(o.to_json(), sort_keys=True) + "\n")

    def read(self) -> List[Outcome]:
        if not self.path.exists():
            return []
        return [Outcome(**json.loads(line)) for line in self.path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------- summary

def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return st.mean(xs) if xs else 0.0


def totals(outs: Sequence[Outcome]) -> Dict[str, Any]:
    scored = [o.right for o in outs if o.right is not None]
    return {"n": len(outs), "accuracy": round(_mean(scored), 4) if scored else None,
            "tokens": round(_mean(o.tokens for o in outs), 1),
            "weighted_tokens": round(_mean(o.weighted_tokens for o in outs), 1),
            "seconds": round(_mean(o.seconds for o in outs), 3),
            "cost_usd": round(sum(o.cost_usd for o in outs), 6),
            "calls": round(_mean(o.calls for o in outs), 2)}


def ratio_ranges(pairs: Sequence[Tuple[Outcome, Outcome]], reps: int = 2000, seed: int = 20261102
                 ) -> Dict[str, List[float]]:
    """95% bootstrap ranges for the tokens and time ratios (ours over the baseline's), resampling questions in
    pairs. A ratio whose whole range sits under 1 is cheaper with confidence, not just on this draw of questions."""
    rng = random.Random(seed)
    n = len(pairs)
    tok: List[float] = []
    sec: List[float] = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        tb, tp = sum(pairs[i][1].tokens for i in idx), sum(pairs[i][0].tokens for i in idx)
        sb, sp = sum(pairs[i][1].seconds for i in idx), sum(pairs[i][0].seconds for i in idx)
        tok.append(tp / tb if tb else 1.0)
        sec.append(sp / sb if sb else 1.0)
    tok.sort()
    sec.sort()
    lo, hi = int(0.025 * reps), int(0.975 * reps) - 1
    return {"tokens_ratio_ci95": [round(tok[lo], 3), round(tok[hi], 3)],
            "seconds_ratio_ci95": [round(sec[lo], 3), round(sec[hi], 3)]}


def compare(outs: Sequence[Outcome], base: Sequence[Outcome], ranges: bool = True) -> Dict[str, Any]:
    """Paired comparison on the tasks both have."""
    b = {o.task: o for o in base}
    pairs = [(o, b[o.task]) for o in outs if o.task in b and o.right is not None and b[o.task].right is not None]
    if not pairs:
        return {"n": 0}
    gain, lo, hi = ft.paired_boot([o.right - p.right for o, p in pairs], reps=4000)
    tb, tp = _mean(p.tokens for _, p in pairs), _mean(o.tokens for o, _ in pairs)
    sb, sp = _mean(p.seconds for _, p in pairs), _mean(o.seconds for o, _ in pairs)
    wb, wp = _mean(p.weighted_tokens for _, p in pairs), _mean(o.weighted_tokens for o, _ in pairs)
    return {"n": len(pairs), "accuracy_gain": round(gain, 4), "accuracy_ci95": [round(lo, 4), round(hi, 4)],
            "tokens_ratio": round(tp / tb, 3) if tb else None, "seconds_ratio": round(sp / sb, 3) if sb else None,
            "weighted_tokens_ratio": round(wp / wb, 3) if wb else None,
            "verdict": verdict(lo, hi, tp / tb if tb else 1.0, sp / sb if sb else 1.0),
            **(ratio_ranges(pairs) if ranges else {})}


def verdict(lo: float, hi: float, tokens_ratio: float, seconds_ratio: float) -> str:
    cheaper = tokens_ratio < 1.0 and seconds_ratio < 1.0
    if lo > 0 and cheaper:
        return "beats it on all three"
    if lo > 0:
        return "more accurate, but not cheaper on both tokens and time"
    if hi < 0:
        return "less accurate"
    if cheaper:
        return "ties it on accuracy, and is cheaper on both"
    return "ties it on accuracy, not cheaper on both"


def summary(outs: Sequence[Outcome], baseline: str = "reader") -> Dict[str, Any]:
    """Per recipe: totals, and the paired comparison with the baseline recipe."""
    by: Dict[str, List[Outcome]] = {}
    for o in outs:
        by.setdefault(o.recipe, []).append(o)
    base = by.get(baseline, [])
    return {name: {**totals(rs), "vs_" + baseline: compare(rs, base) if name != baseline and base else None}
            for name, rs in sorted(by.items())}


def pareto(table: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Recipes that no other recipe beats on accuracy, tokens and time all at once."""
    rows = {k: v for k, v in table.items() if v.get("accuracy") is not None}
    out = []
    for k, v in rows.items():
        dominated = any(w["accuracy"] >= v["accuracy"] and w["tokens"] <= v["tokens"] and w["seconds"] <= v["seconds"]
                        and (w["accuracy"], -w["tokens"], -w["seconds"]) != (v["accuracy"], -v["tokens"], -v["seconds"])
                        for j, w in rows.items() if j != k)
        if not dominated:
            out.append(k)
    return sorted(out, key=lambda k: -rows[k]["accuracy"])
