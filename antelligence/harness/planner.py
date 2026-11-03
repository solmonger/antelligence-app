"""The planner: for each request, look up the recipe the policy picked for its kind, run it, and log the bill.

Improving over time works like this. A share of live requests (``explore``, 10% by default) is *swept*: every tool
in the toolbox runs on it once and every step is saved. Sweeps are what the learner learns from, because on a swept
request every recipe can be scored on exactly the same work. The request still gets the policy's answer (read back
from the sweep), but its bill is the whole sweep, and the scorecard shows that honestly. Learning again from the
grown pile of sweeps gives the next policy.
"""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

from antelligence.harness.core import Step, Task
from antelligence.harness.learner import Policy
from antelligence.harness.recipes import Ctx, Recipe
from antelligence.harness.scorecard import Log, Outcome, bill, run_one
from antelligence.harness.tools import ARMS, Box, ReplayBox, ant_tool


def sweep_tools(box: Box) -> List[str]:
    """Every tool the toolbox has, in a cheap-to-dear order that also puts each tool's needs first."""
    n = len(box.roster)
    names = [ant_tool(i, "alone") for i in range(n)] + ["reader", "notes"]
    names += [ant_tool(i, arm) for arm in ARMS if arm != "alone" for i in range(n)]
    names += ["code"] + [f"dm:{d}:{a}" for d in ("jev", "d1-3b", "d1-2b", "pplx-flash") for a in ("dm1", "read")]
    return [t for t in names if box.has(t)]


async def sweep(box: Box, task: Task, tools: Optional[Sequence[str]] = None) -> Dict[str, Step]:
    ctx = Ctx(box, task)
    for t in tools or sweep_tools(box):
        await ctx.call(t)
    return dict(ctx.done)


def batch_order(box: Box, tools: Optional[Sequence[str]] = None) -> List[List[str]]:
    """The sweep's tools grouped by the model that serves them, so a one-model-at-a-time server loads each model
    once: the reader's tools (answer, notes, code) first, then each ant model's tools, then decision models (which
    read the ants' answers)."""
    tools = list(tools or sweep_tools(box))
    groups: List[List[str]] = [[t for t in ("reader", "notes", "code") if t in tools]]
    seen: Dict[str, List[str]] = {}
    for a in box.roster:
        seen.setdefault(a.tag, [])
        seen[a.tag] += [ant_tool(a.index, arm) for arm in ARMS if ant_tool(a.index, arm) in tools]
    groups += list(seen.values())
    groups.append([t for t in tools if t.startswith("dm:")])
    return [g for g in groups if g]


async def sweep_batch(box: Box, tasks: Sequence[Task], tools: Optional[Sequence[str]] = None,
                      between: Optional[Callable[[List[str]], Awaitable[None]]] = None,
                      on_group: Optional[Callable[[List[str]], None]] = None, order: str = "tool",
                      workers: int = 2) -> Dict[str, Dict[str, Step]]:
    """Every tool on every task, one model group at a time (see :func:`batch_order`). ``between`` runs after each
    group, e.g. to unload that model.

    ``order`` is how the calls inside a group are lined up. ``tool``: one tool on every task, then the next tool.
    ``question``: every tool of the group on one task, then the next task. The second keeps the calls that share a
    long start (the same question to the same model with another way of looking) next to each other, so a server
    with a prefix cache (``mlx_lm.server`` keeps the last 10 prompts by default) reads that start once, not twice."""
    if order not in ("tool", "question"):
        raise ValueError("order must be 'tool' or 'question'")
    ctxs = {t.id: Ctx(box, t) for t in tasks}
    for group in batch_order(box, tools):
        if on_group:
            on_group(group)
        if order == "tool":
            for tool in group:
                await asyncio.gather(*(ctxs[t.id].call(tool) for t in tasks))
        else:
            queue = list(tasks)

            async def worker() -> None:          # a few workers, each taking one whole task at a time, so a
                while queue:                     # task's calls stay together instead of queueing behind every
                    t = queue.pop(0)             # other task's first call
                    for tool in group:
                        await ctxs[t.id].call(tool)
            await asyncio.gather(*(worker() for _ in range(max(1, workers))))
        if between:
            await between(group)
    return {tid: dict(c.done) for tid, c in ctxs.items()}


class Planner:
    def __init__(self, policy: Policy, recipes: Mapping[str, Recipe], box: Box, *, explore: float = 0.1,
                 seed: int = 20261102, log: Optional[Log] = None) -> None:
        self.policy, self.recipes, self.box, self.explore = policy, dict(recipes), box, explore
        self.rng = random.Random(seed)
        self.log = log
        self.swept: Dict[str, Dict[str, Step]] = {}

    async def answer(self, task: Task, key: Optional[Mapping[str, str]] = None) -> Outcome:
        name = self.policy.pick(task)
        if name not in self.recipes or not self.recipes[name].fits(self.box):
            name = self.policy.default if self.policy.default in self.recipes else "reader"
        recipe = self.recipes[name]
        if self.explore and self.rng.random() < self.explore:
            steps = await sweep(self.box, task)
            self.swept[task.id] = steps
            got = await run_one(recipe, ReplayBox({task.id: steps}, self.box.roster), task, key)
            out = bill(task, f"harness:{name}+sweep", got.answer, list(steps.values()), key)
        else:
            got = await run_one(recipe, self.box, task, key)
            out = Outcome(**{**got.to_json(), "recipe": f"harness:{name}"})
        if self.log:
            self.log.add(out)
        return out

    async def answer_all(self, tasks: Sequence[Task], key: Optional[Mapping[str, str]] = None) -> List[Outcome]:
        return [await self.answer(t, key) for t in tasks]
