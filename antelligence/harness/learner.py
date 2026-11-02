"""The learner: which recipe wins for which kind of request, learned from saved steps.

How it learns (plain words):

1. Score every recipe on every saved task by replaying its steps (free). Each gets a value per task:
   ``right - w_tokens * tokens/T - w_seconds * seconds/S - w_usd * dollars``, where T and S are the baseline's
   (the reader's) average tokens and seconds. With the default weights, using as many tokens and as much time as
   the reader costs 2 + 2 accuracy points; accuracy still matters most.
2. The overall winner becomes the default.
3. For each bucket of tasks (a few plain text features, see :func:`antelligence.harness.core.bucket`), pick the
   recipe with the best value, but shrink each bucket's average toward the recipe's overall average as if the
   overall average were ``shrink`` extra tasks. A bucket with few tasks therefore can't pick a recipe that just got
   lucky there. Buckets with fewer than ``min_n`` tasks keep the default.

That lookup is the policy, and the planner is that lookup: picking a recipe costs no model call, so it can't eat
the savings (the RouteLLM lesson: a router is only worth it when it is much cheaper than what it routes).

How it is checked: the policy is always scored on tasks it did not learn from, either a whole held-out set or, with
one set, k folds (each task routed by a policy learned without it).
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.harness.core import Task, bucket
from antelligence.harness.recipes import Recipe
from antelligence.harness.scorecard import Outcome, run_one
from antelligence.harness.tools import Box, Missing

Table = Dict[str, Dict[str, Outcome]]           # recipe -> task id -> outcome


@dataclass(frozen=True)
class Weights:
    tokens: float = 0.02          # accuracy points given up for spending as many tokens as the baseline
    seconds: float = 0.02         # ... for taking as long as the baseline
    usd: float = 1.0              # ... per dollar (a cent costs 0.01)


@dataclass
class Policy:
    default: str
    buckets: Dict[str, str] = field(default_factory=dict)
    weights: Weights = field(default_factory=Weights)
    ref: Dict[str, float] = field(default_factory=dict)     # the baseline's mean tokens and seconds
    info: Dict[str, Any] = field(default_factory=dict)

    def pick(self, task: Task) -> str:
        return self.buckets.get(bucket(task), self.default)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Policy":
        return cls(str(d["default"]), dict(d.get("buckets") or {}), Weights(**(d.get("weights") or {})),
                   dict(d.get("ref") or {}), dict(d.get("info") or {}))


async def _evaluate(recipes: Sequence[Recipe], box: Box, tasks: Sequence[Task],
                    key: Optional[Mapping[str, str]]) -> Table:
    out: Table = {r.name: {} for r in recipes}
    for r in recipes:
        for t in tasks:
            try:
                out[r.name][t.id] = await run_one(r, box, t, key)
            except Missing:
                continue
    return out


def evaluate(recipes: Sequence[Recipe], box: Box, tasks: Sequence[Task], key: Optional[Mapping[str, str]] = None
             ) -> Table:
    """Every recipe on every task it can run on (replay boxes answer at once, so this is quick)."""
    return asyncio.run(_evaluate(recipes, box, tasks, key))


def common(table: Table, tasks: Sequence[Task], min_cover: float = 0.95) -> Tuple[List[str], List[Task]]:
    """The recipes that ran on at least ``min_cover`` of the tasks, and the tasks every one of them ran on, so all
    recipes are compared on exactly the same questions."""
    names = [n for n, per in table.items() if len(per) >= min_cover * len(tasks) and per]
    keep = [t for t in tasks if all(t.id in table[n] for n in names)]
    return names, keep


def value(o: Outcome, ref: Mapping[str, float], w: Weights) -> float:
    if o.right is None:
        raise ValueError("learning needs the answer key")
    return (o.right - w.tokens * (o.tokens / ref["tokens"] if ref.get("tokens") else 0.0)
            - w.seconds * (o.seconds / ref["seconds"] if ref.get("seconds") else 0.0) - w.usd * o.cost_usd)


def learn(table: Table, tasks: Sequence[Task], *, weights: Weights = Weights(), baseline: str = "reader",
          shrink: float = 10.0, min_n: int = 8, names: Optional[Sequence[str]] = None) -> Policy:
    names = list(names) if names is not None else common(table, tasks)[0]
    if baseline not in names:
        raise ValueError(f"the baseline {baseline!r} has no scored steps on these tasks")
    ids = [t.id for t in tasks if all(t.id in table[n] for n in names)]
    if not ids:
        raise ValueError("no task has every recipe scored")
    base = [table[baseline][i] for i in ids]
    ref = {"tokens": sum(o.tokens for o in base) / len(base), "seconds": sum(o.seconds for o in base) / len(base)}
    val = {n: {i: value(table[n][i], ref, weights) for i in ids} for n in names}
    overall = {n: sum(v.values()) / len(v) for n, v in val.items()}
    default = max(names, key=lambda n: (round(overall[n], 9), n == baseline))
    groups: Dict[str, List[str]] = {}
    by_id = {t.id: t for t in tasks}
    for i in ids:
        groups.setdefault(bucket(by_id[i]), []).append(i)
    picks: Dict[str, str] = {}
    info: Dict[str, Any] = {"n": len(ids), "recipes": len(names), "overall": {n: round(overall[n], 4) for n in names},
                            "buckets": {}}
    for b, members in sorted(groups.items()):
        nb = len(members)
        shrunk = {n: (sum(val[n][i] for i in members) + shrink * overall[n]) / (nb + shrink) for n in names}
        best = max(names, key=lambda n: (round(shrunk[n], 9), n == default))
        chosen = best if nb >= min_n else default
        if chosen != default:
            picks[b] = chosen
        info["buckets"][b] = {"n": nb, "pick": chosen, "value": round(shrunk[chosen], 4),
                              "default_value": round(shrunk[default], 4)}
    return Policy(default, picks, weights, ref, info)


def route(policy: Policy, table: Table, tasks: Sequence[Task], name: str = "harness") -> List[Outcome]:
    """What the policy would have done on these tasks, from the saved outcomes."""
    out = []
    for t in tasks:
        o = table.get(policy.pick(t), {}).get(t.id)
        if o is not None:
            out.append(Outcome(**{**o.to_json(), "recipe": name}))
    return out


def held_out(table: Table, train: Sequence[Task], test: Sequence[Task], **kw: Any) -> Tuple[Policy, List[Outcome]]:
    """Learn on ``train``, route ``test``."""
    names = common(table, list(train) + list(test))[0]
    pol = learn(table, train, names=names, **kw)
    return pol, route(pol, table, test)


def k_fold(table: Table, tasks: Sequence[Task], k: int = 5, seed: int = 20261102, **kw: Any) -> List[Outcome]:
    """Each task routed by a policy learned on the other folds. Folds are cut within each bucket, so every fold
    sees every kind of request."""
    names, keep = common(table, tasks)
    rng = random.Random(seed)
    by_b: Dict[str, List[Task]] = {}
    for t in keep:
        by_b.setdefault(bucket(t), []).append(t)
    fold: Dict[str, int] = {}
    for b in sorted(by_b):
        members = sorted(by_b[b], key=lambda t: t.id)
        rng.shuffle(members)
        for j, t in enumerate(members):
            fold[t.id] = j % k
    out: List[Outcome] = []
    for f in range(k):
        train = [t for t in keep if fold[t.id] != f]
        test = [t for t in keep if fold[t.id] == f]
        if not train or not test:
            continue
        out += route(learn(table, train, names=names, **kw), table, test)
    return out
