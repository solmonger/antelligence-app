"""Fading memory: marks that fade, with a half-life each reader can shorten or lengthen.

Every version before this one gave a mark kind one fixed half-life (``kernel.colony.marks.DEFAULT_HALF_LIFE``).
:class:`FadingMemory` keeps a half-life **per mark** and lets readers move it:

* **fixed** (``FadeRule.fixed``): readers' feedback is ignored; every mark keeps its starting half-life.
* **smart** (``FadeRule.smart``): a reader whose own view disagrees with a mark multiplies its half-life by
  ``contradict`` (< 1, so it fades faster); one whose view agrees multiplies it by ``confirm`` (>= 1). The half-life is
  clamped to ``[min_half_life, max_half_life]``.

There is no checker. Feedback comes only from ordinary readers, from whatever they hold as their own view; the memory
never decides who is right. A mark's strength at tick ``t`` integrates its half-life history exactly: a half-life
change at tick ``t0`` applies from ``t0`` on and never rewrites the strength already lost.

The store is world-agnostic (``place`` is any hashable: a voxel, a node id, a topic) and deterministic. The kernel
never imports a world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterator, List, Mapping, Optional, Tuple

Place = Hashable


@dataclass(frozen=True)
class FadeRule:
    """How readers' feedback moves a mark's half-life. ``smart=False`` ignores feedback (the fixed baseline)."""

    half_life: float = 10.0
    smart: bool = False
    confirm: float = 1.25
    contradict: float = 0.5
    min_half_life: float = 1.0
    max_half_life: float = 80.0

    def __post_init__(self) -> None:
        if not self.half_life > 0:
            raise ValueError("half_life must be positive")
        if not 0 < self.min_half_life <= self.half_life <= self.max_half_life:
            raise ValueError("need 0 < min_half_life <= half_life <= max_half_life")
        if not 0 < self.contradict <= 1 <= self.confirm:
            raise ValueError("need 0 < contradict <= 1 <= confirm")

    @classmethod
    def fixed(cls, half_life: float = 10.0) -> "FadeRule":
        return cls(half_life=half_life, smart=False, min_half_life=min(1.0, half_life),
                   max_half_life=max(80.0, half_life))

    @classmethod
    def smart_rule(cls, half_life: float = 10.0, confirm: float = 1.25, contradict: float = 0.5) -> "FadeRule":
        return cls(half_life=half_life, smart=True, confirm=confirm, contradict=contradict,
                   min_half_life=min(1.0, half_life), max_half_life=max(80.0, 8 * half_life))


@dataclass
class Mark:
    """One author's mark at one place. ``log2_lost`` is the number of halvings already accumulated at ``since``."""

    place: Place
    author: str
    value: int
    amount: float
    deposited: int
    half_life: float
    since: int
    log2_lost: float = 0.0
    confirms: int = 0
    contradicts: int = 0

    def strength(self, tick: int) -> float:
        if tick < self.deposited:
            return 0.0
        halvings = self.log2_lost + max(0, tick - self.since) / self.half_life
        return self.amount * math.pow(0.5, halvings)

    def _rebase(self, tick: int) -> None:
        """Fold the halvings since ``since`` into ``log2_lost`` so a new half-life applies from ``tick`` on only."""
        if tick > self.since:
            self.log2_lost += (tick - self.since) / self.half_life
            self.since = tick


@dataclass
class FadingMemory:
    """Marks keyed by (place, author). A new deposit by the same author at the same place replaces the old mark."""

    rule: FadeRule = field(default_factory=FadeRule)
    floor: float = 1e-3  # marks weaker than this are treated as gone and pruned
    _marks: Dict[Tuple[Place, str], Mark] = field(default_factory=dict)
    _by_place: Dict[Place, Dict[str, Mark]] = field(default_factory=dict)
    feedback_events: int = 0

    def deposit(self, place: Place, author: str, value: int, tick: int, amount: float = 1.0,
                half_life: Optional[float] = None) -> Mark:
        if amount <= 0:
            raise ValueError("amount must be positive")
        hl = self.rule.half_life if half_life is None else half_life
        if hl <= 0:
            raise ValueError("half_life must be positive")
        mark = Mark(place, author, int(value), float(amount), tick, float(hl), tick)
        self._marks[(place, author)] = mark
        self._by_place.setdefault(place, {})[author] = mark
        return mark

    def remove(self, place: Place, author: str) -> None:
        mark = self._marks.pop((place, author), None)
        if mark is not None:
            self._by_place[place].pop(author, None)

    def get(self, place: Place, author: str) -> Optional[Mark]:
        return self._marks.get((place, author))

    def at(self, place: Place, tick: int) -> List[Mark]:
        """Live marks at ``place`` (deposited before ``tick``: a mark is visible from the next tick), strongest first."""
        live = [m for m in self._by_place.get(place, {}).values()
                if m.deposited < tick and m.strength(tick) >= self.floor]
        return sorted(live, key=lambda m: (-m.strength(tick), m.author))

    def tally(self, place: Place, tick: int) -> Dict[int, float]:
        """Total live strength per value at ``place``."""
        out: Dict[int, float] = {}
        for m in self.at(place, tick):
            out[m.value] = out.get(m.value, 0.0) + m.strength(tick)
        return out

    def feedback(self, mark: Mark, agrees: bool, tick: int) -> None:
        """A reader's own view agrees or disagrees with ``mark``. Counted always; moves the half-life only if smart."""
        if agrees:
            mark.confirms += 1
        else:
            mark.contradicts += 1
        self.feedback_events += 1
        if not self.rule.smart:
            return
        mark._rebase(tick)
        factor = self.rule.confirm if agrees else self.rule.contradict
        mark.half_life = min(self.rule.max_half_life, max(self.rule.min_half_life, mark.half_life * factor))

    def prune(self, tick: int) -> int:
        dead = [k for k, m in self._marks.items() if m.strength(tick) < self.floor]
        for place, author in dead:
            self.remove(place, author)
        return len(dead)

    def __iter__(self) -> Iterator[Mark]:
        return iter(list(self._marks.values()))

    def __len__(self) -> int:
        return len(self._marks)

    def summary(self, tick: int) -> Mapping[str, float]:
        marks = [m for m in self._marks.values() if m.strength(tick) >= self.floor]
        return {"live_marks": len(marks), "feedback_events": self.feedback_events,
                "mean_half_life": (sum(m.half_life for m in marks) / len(marks)) if marks else 0.0}
