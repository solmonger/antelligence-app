"""Think alone, done included, in the SwarmBench foraging world. Scripted, seeded, no model calls.

The pilot world already has the think-alone rule built into its talk: an agent only broadcasts food or a nest it
can see right now. Two pieces test what that rule is worth and what an ant gains by using it:

* :class:`RelayForagingWorld` -- the pilot world with one switch, ``relay``. When it is on, an agent that sees
  nothing also broadcasts what it *remembers* ("food is north of me", pointing at its remembered spot), the way a
  colony relays a trail. Under ``v1`` ("food runs out") that memory goes stale, so a relayed message can send
  listeners to an empty patch. When it is off, talk is first-hand only (the pilot as published).
* :class:`LookForager` -- the think-alone ant. It trusts what it sees over what it was told: a remembered food spot
  that is in view and empty is forgotten (the pilot's S' for v1, here on by default, so "this patch is still
  good" is a fact seen, not a claim believed), and food seen anywhere else becomes the target. Only when it has
  no target of its own does it compare with others: it heads for the nearest peer that says food is near, and
  drops that lead as soon as its own view shows the spot with no food.

Waggle-dance marks (from the rumour world's ``graded`` policy): :class:`RelayForagingWorld` tags each message with how
sure its sender is (first-hand ``cap``, relayed 1; with ``shout=True`` relayed memories claim ``cap`` too, the liar
case), and :class:`GradedForager` follows the surest lead rather than just the nearest.

The pilot's S trusts its brief and ignores what it hears; the pilot's S' (``ExploreWhenEmpty``) trusts its eyes
but ignores what it hears. ``LookForager`` adds "then compare" on top of S'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from antelligence.kernel.types import Intent, LocalView
from antelligence.worlds.foraging_sb.nav import RADIUS, compass, plan_route, unstick
from antelligence.worlds.foraging_sb.scripts import ExploreWhenEmpty, _t
from antelligence.worlds.foraging_sb.world import HEIGHT, MSG_MAX, WIDTH, ForagingSBWorld

Cell = Tuple[int, int]
WORD_STEP = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
LEAD_REACH = 2      # a heard "food is <direction> of me" is taken to mean about this many cells from the speaker
LEAD_TICKS = 12     # a lead not confirmed by the ant's own eyes within this many ticks is dropped


class RelayForagingWorld(ForagingSBWorld):
    """The pilot foraging world; with ``relay=True`` agents also broadcast remembered (possibly stale) spots."""

    def __init__(self, seed: int, condition: str = "base", *, relay: bool = False, shout: bool = False, cap: int = 2,
                 **kw: Any) -> None:
        super().__init__(seed, condition, **kw)
        if cap < 1:
            raise ValueError("cap must be at least 1")
        self.relay, self.shout, self.cap = relay, shout, cap
        self.relayed = 0
        self._strength = 1

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "world": "foraging_sb_relay", "relay": self.relay, "shout": self.shout,
                "cap": self.cap}

    def _perceive(self, agent_id: str, tick: int) -> Dict[str, Any]:
        posted = len(self._outbox)
        data = super()._perceive(agent_id, tick)
        if len(self._outbox) > posted:  # this agent just posted: say how sure it is (listeners may ignore it)
            self._outbox[-1]["strength"] = self._strength
        return data

    def _message(self, agent_id: str, rows: List[str]) -> Optional[str]:
        seen = super()._message(agent_id, rows)
        self._strength = self.cap  # first-hand: seen right now
        if seen or not self.relay:
            return seen
        self._strength = self.cap if self.shout else 1  # a remembered spot, maybe stale (shouted at the cap if shout)
        mem = self.memory[agent_id]
        target = mem.nest_at if self.carrying[agent_id] else mem.food_at
        if target is None:
            return None
        px, py = self.positions[agent_id]
        word = compass(target[0] - px, target[1] - py)
        if word == "here":
            return None
        self.relayed += 1
        return f"{'nest' if self.carrying[agent_id] else 'food'} is {word} of me"[:MSG_MAX]


def lead_from(heard: List[Dict[str, Any]], pos: Cell, want: str = "food", graded: bool = False) -> Optional[Cell]:
    """The cell the nearest peer's "<want> is <direction> of me" points at, or None. With ``graded``, the surest
    peer's (highest ``strength``), nearest first among equally sure peers."""
    if graded:
        heard = sorted(heard, key=lambda m: -m.get("strength", 1))  # stable: nearest first within a strength
    for m in heard:  # already sorted nearest first
        text = m["text"]
        if not text.startswith(f"{want} is ") or not text.endswith(" of me"):
            continue
        word = text[len(want) + 4: -len(" of me")]
        dx = dy = 0
        for part in word.split("-"):
            step = WORD_STEP.get(part)
            if step is None:
                break
            dx, dy = dx + step[0], dy + step[1]
        else:
            x = min(max(pos[0] + m["dx"] + LEAD_REACH * dx, 0), WIDTH - 1)
            y = min(max(pos[1] + m["dy"] + LEAD_REACH * dy, 0), HEIGHT - 1)
            return (x, y)
    return None


class LookForager(ExploreWhenEmpty):
    """Trust your own eyes first; use what peers say only when you have nothing of your own. See module doc."""

    graded = False

    def __init__(self, sensor=None, *, use_heard: bool = True) -> None:
        super().__init__(sensor)
        self.use_heard = use_heard

    def describe(self) -> Dict[str, Any]:
        return {"policy": "foraging_sb.look", "version": 1, "use_heard": self.use_heard}

    def decide(self, view: LocalView) -> Intent:
        f = self.sensor.sense(view).fields
        agent, tick, pos = f["agent"], f["tick"], _t(f["pos"])
        st = self._state.get(agent)
        if st is None or tick <= st["tick"]:  # a new run reuses the policy
            st = self._state[agent] = {"tick": 0, "stale": None, "target": None, "lead": None, "lead_tick": 0,
                                       "leads": 0, "dead_leads": 0}
        st["tick"] = tick
        remembered = _t(f["food_at"])
        if f["food_gone"]:
            st["stale"], st["target"] = remembered, None
        seen = _t(f["food_seen"])
        if seen is not None and seen != st["stale"]:
            st["target"], st["lead"] = seen, None
        target = st["target"] or (remembered if remembered != st["stale"] else None)
        if target is None and not f["carrying"] and self.use_heard:
            lead = st["lead"]
            if lead is not None and (self._in_view(pos, lead) or tick - st["lead_tick"] > LEAD_TICKS):
                st["lead"], lead = None, None   # my own eyes reached it and showed no food, or it went cold
                st["dead_leads"] += 1
            if lead is None:
                lead = lead_from(f["heard"], pos, graded=self.graded)
                if lead is not None and lead != st["stale"]:
                    st["lead"], st["lead_tick"] = lead, tick
                    st["leads"] += 1
                else:
                    lead = None
            target = lead
        action, _ = plan_route(f["map"], pos, f["carrying"], target, _t(f["nest_at"]),
                               [tuple(c) for c in f["occupied"]], rot=int(agent) % 4)
        if target is not None and target == st["lead"] and action == "pick_up":
            action = "stay"  # next to a lead cell with no food in sight: nothing to pick up there
        return Intent(unstick(action, f["stuck"], f["view"], view.seed))

    @staticmethod
    def _in_view(pos: Cell, cell: Cell) -> bool:
        return max(abs(cell[0] - pos[0]), abs(cell[1] - pos[1])) <= RADIUS - 1

    def lead_stats(self) -> Dict[str, int]:
        return {"leads": sum(s.get("leads", 0) for s in self._state.values()),
                "dead_leads": sum(s.get("dead_leads", 0) for s in self._state.values())}


class GradedForager(LookForager):
    """:class:`LookForager` that follows the surest lead it hears (waggle-dance marks), not just the nearest."""

    graded = True

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "policy": "foraging_sb.graded"}


def default_ant() -> LookForager:
    """The default ant for new swarm runs in this world (the pilot's recorded arms keep their own scripts)."""
    return LookForager()


__all__ = ["GradedForager", "LookForager", "RelayForagingWorld", "default_ant", "lead_from"]
