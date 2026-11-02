"""Recipes: small programs over the toolbox that turn one task into one answer.

A recipe never calls a model itself. It asks its :class:`Ctx` for steps (``await ctx.call("reader")``), and the
context asks the toolbox, counts each step once, and keeps the bill. The same recipe therefore runs live (a
:class:`~antelligence.harness.tools.LiveBox`) and on saved work (a :class:`~antelligence.harness.tools.ReplayBox`),
which is how the learner can score every recipe on logged questions for free, the way FrugalGPT scored its cascades
on cached replies.

The v0 recipes:

* ``reader``: ask the reader once. This is "always ask one model", the line to beat.
* ``code``, ``dm:jev:dm1`` and other single tools: ask that one tool.
* ``vote:<arm>``: every ant answers (alone, from notes, or both); plain vote.
* ``seq:<arm>:m<k>[><tool>]``: ants answer one at a time, cheapest model first, and stop as soon as the leading
  answer is ``k`` votes clear of the next (early-stopping self-consistency). If they run out still unsure, ask the
  escalation tool (``reader`` or ``code``) instead: a FrugalGPT-style cascade.
* ``code><tool>``: the code checker first; if it can't settle the question, ask the fallback tool.
* ``reader+check``: the reader answers, then the code checker checks it. A pick that the checker rules out is
  replaced by the checker's one passing option; otherwise the reader's answer stands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from antelligence.harness.core import Step, Task, margin, vote
from antelligence.harness.tools import ARMS, AntInfo, Box, Missing, ant_tool, deps


class Ctx:
    """One task's run: asks the toolbox for steps, at most once each, dependencies first, and keeps every step in
    the order it was asked."""

    def __init__(self, box: Box, task: Task) -> None:
        self.box, self.task = box, task
        self.done: Dict[str, Step] = {}
        self.order: List[str] = []

    async def call(self, tool: str) -> Step:
        if tool in self.done:
            return self.done[tool]
        if not self.box.has(tool):
            raise Missing(tool)
        for d in deps(tool, len(self.box.roster)):
            await self.call(d)
        step = await self.box.run(tool, self.task, self.done)
        self.done[tool] = step
        self.order.append(tool)
        return step

    @property
    def steps(self) -> List[Step]:
        return [self.done[t] for t in self.order]


def cheap_first(roster: Sequence[AntInfo]) -> List[int]:
    """Ant indexes, smallest model first (ties keep roster order)."""
    return [a.index for a in sorted(roster, key=lambda a: (a.weight, a.index))]


@dataclass(frozen=True)
class Recipe:
    """Base: ``name`` is unique and readable; ``tools`` lists what it may call (for checking a toolbox has them)."""

    name: str
    tools: Tuple[str, ...] = field(default=())

    def fits(self, box: Box) -> bool:
        return all(box.has(t) for t in self.tools)

    async def run(self, ctx: Ctx) -> Optional[str]:  # pragma: no cover - every recipe overrides it
        raise NotImplementedError


@dataclass(frozen=True)
class Single(Recipe):
    tool: str = "reader"

    async def run(self, ctx: Ctx) -> Optional[str]:
        return (await ctx.call(self.tool)).answer


@dataclass(frozen=True)
class Vote(Recipe):
    arm: str = "alone"
    n: int = 8

    async def run(self, ctx: Ctx) -> Optional[str]:
        return vote([(await ctx.call(ant_tool(i, self.arm))).answer for i in range(self.n)])


@dataclass(frozen=True)
class Seq(Recipe):
    arm: str = "alone"
    stop: int = 2
    escalate: Optional[str] = None
    n: int = 8
    order: Tuple[int, ...] = ()

    async def run(self, ctx: Ctx) -> Optional[str]:
        answers: List[Optional[str]] = []
        idx = list(self.order) or list(range(self.n))
        for k, i in enumerate(idx):
            answers.append((await ctx.call(ant_tool(i, self.arm))).answer)
            left = len(idx) - k - 1
            m = margin(answers)
            if m >= self.stop:                       # sure enough
                return vote(answers)
            if self.escalate and m + left < self.stop:
                break                                # can't get sure any more: hand it up now
            if not self.escalate and m > left:
                return vote(answers)                 # the rest can't change the leader
        if self.escalate:
            got = await ctx.call(self.escalate)
            if got.answer is not None:
                return got.answer
            if got.passes:                           # the code checker narrowed it down: vote inside that
                inside = [x for x in answers if x in got.passes]
                if inside:
                    return vote(inside)
        return vote(answers)


@dataclass(frozen=True)
class CodeThen(Recipe):
    fallback: str = "reader"

    async def run(self, ctx: Ctx) -> Optional[str]:
        got = (await ctx.call("code")).answer
        return got if got is not None else (await ctx.call(self.fallback)).answer


@dataclass(frozen=True)
class ReaderCheck(Recipe):
    async def run(self, ctx: Ctx) -> Optional[str]:
        pick = (await ctx.call("reader")).answer
        check = await ctx.call("code")
        if check.passes is None or pick in (check.passes or ()):
            return pick
        return check.answer if check.answer is not None else pick


# ------------------------------------------------------------------ the menu

STOPS = (2, 3, 4)


def menu(roster: Sequence[AntInfo], extra_single: Sequence[str] = ()) -> List[Recipe]:
    """Every v0 recipe for a roster. ``extra_single`` adds one-tool recipes such as ``dm:jev:dm1``."""
    n = len(roster)
    order = tuple(cheap_first(roster))
    ants = lambda arm: tuple(ant_tool(i, arm) for i in range(n))  # noqa: E731
    out: List[Recipe] = [Single("reader", ("reader",), "reader"), Single("code", ("code",), "code")]
    out += [Single(t, (t,), t) for t in extra_single]
    for arm in ARMS:
        out.append(Vote(f"vote:{arm}", ants(arm), arm, n))
        for k in STOPS:
            out.append(Seq(f"seq:{arm}:m{k}", ants(arm), arm, k, None, n, order))
            for up in ("reader", "code"):
                out.append(Seq(f"seq:{arm}:m{k}>{up}", ants(arm) + (up,), arm, k, up, n, order))
    out += [CodeThen("code>reader", ("code", "reader"), "reader"), ReaderCheck("reader+check", ("reader", "code"))]
    for t in extra_single:
        out.append(CodeThen(f"code>{t}", ("code", t), t))
    names = [r.name for r in out]
    assert len(set(names)) == len(names), "recipe names must be unique"
    return out


def by_name(recipes: Sequence[Recipe]) -> Dict[str, Recipe]:
    return {r.name: r for r in recipes}
