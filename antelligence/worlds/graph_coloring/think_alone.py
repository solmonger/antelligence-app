"""Think alone, done included, in the AgentsNet colouring world. Scripted, seeded, no model calls.

Two pieces, both built on the pilot world and sensor unchanged:

* :class:`SwarmColouringWorld` -- the pilot world at any size, with any mix of its two stresses at once: a seeded
  set of liars (each announces a random colour to every neighbour every round, as in the pilot's v2) and lost
  messages (each message dropped with probability ``drop_p``, as in v1). Only edges between two honest agents are
  scored.
* :class:`LookBeforeDone` -- the think-alone ant for this world. The pilot's S waits for its higher-priority
  neighbours to report a settled colour, takes one colour, and never looks again: its "done" is a claim made once.
  This ant does the opposite on both counts:

  1. **Answer alone first.** In its first round it takes a colour of its own (seeded, any colour of the palette),
     without waiting for anyone.
  2. **Then compare, and call it done only when it sees it done.** Every round it looks at what each neighbour
     last said about itself. If a neighbour with priority (a lower id) holds its colour, it moves to the lowest
     colour none of its neighbours holds. Its colour counts as done only in a round where it sees no clash.
  3. **Say only what you hold, until it is heard.** It re-announces its own colour until every neighbour has
     acknowledged it, so a lost message is repeated (the pilot's S' for v1, here on by default).

  With ``gate=True`` it also stops hearing a neighbour whose announced colour has changed ``gate_flips`` times
  for no reason it can see (the earned-trust rule from the liar probe: an honest ant only moves when it clashes,
  so a neighbour that keeps changing for no visible reason stops being believed).

Waggle-dance marks (:class:`GradedColouringWorld` and :class:`GradedLook`, from the rumour world's ``graded``
policy): each announcement also says how sure its sender is, and on a clash the less sure ant gives way.

``done_claims`` reports, after a run, how many honest agents considered themselves done while really clashing
with an honest neighbour: for S every coloured agent counts itself done; for this ant only one whose last look
showed no clash.
"""

from __future__ import annotations

import random
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from antelligence.kernel.decision import Sensor
from antelligence.kernel.types import NOOP, Intent, LocalView, Observation
from antelligence.worlds.graph_coloring.sensing import ColoringSensor
from antelligence.worlds.graph_coloring.world import GraphColoringWorld, colour_action


class SwarmColouringWorld(GraphColoringWorld):
    """The pilot colouring world with many liars and, optionally, lost messages. ``base`` rules otherwise."""

    def __init__(self, seed: int, *, n_agents: int, rounds: int, family: str, liars: Iterable[str] = (),
                 drop_p: float = 0.0) -> None:
        super().__init__(seed, "base", n_agents=n_agents, rounds=rounds, family=family)
        if not 0 <= drop_p < 1:
            raise ValueError("drop_p must be in [0, 1)")
        self.liars: FrozenSet[str] = frozenset(liars)
        self.loss = drop_p

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "world": "graph_coloring_swarm", "liars": len(self.liars), "drop_p": self.loss}

    def step_environment(self, tick: int) -> None:
        outgoing = []
        for u in self.agents():
            liar = u in self.liars
            if u not in self._announcing and not liar:
                continue
            for v in self._adj[u]:
                announced = self._lie_rng.randrange(self.palette) if liar else self.colour[u]
                if liar:
                    self.counters["lied"] += 1
                if announced is None:
                    continue
                prior = self.known[u].get(v)
                outgoing.append((u, v, announced, prior["colour"] if prior else None))
        self._announcing = set()
        self.fresh = {a: set() for a in self._adj}
        for u, v, announced, ack in outgoing:
            self.counters["sent"] += 1
            if self.loss and self._drop_rng.random() < self.loss:
                self.counters["dropped"] += 1
                continue
            old = self.known[v].get(u)
            self.known[v][u] = {"colour": announced, "ack": ack, "round": tick,
                                "prev": old["colour"] if old else None}
            self.fresh[v].add(u)
        self._revision += 1

    def honest(self) -> List[str]:
        return [a for a in self.agents() if a not in self.liars]

    def scored_edges(self) -> List[Tuple[str, str]]:
        return [(str(u), str(v)) for u, v in self.graph.edges
                if str(u) not in self.liars and str(v) not in self.liars]

    def clashing(self) -> Set[str]:
        """Honest agents that are uncoloured or share a colour with an honest neighbour."""
        bad: Set[str] = set()
        for u, v in self.scored_edges():
            if not self._proper(u, v):
                bad.update(a for a in (u, v) if self.colour[a] is None or self.colour[u] == self.colour[v])
        bad.update(a for a in self.honest() if self.colour[a] is None)
        return bad


class LookBeforeDone:
    """Think alone, then compare; done only when seen done. See the module docstring."""

    name = "look_before_done"

    def __init__(self, sensor: Sensor, seed: int, *, gate: bool = False, gate_flips: int = 3,
                 ignore: FrozenSet[str] = frozenset()) -> None:
        self._sensor = sensor
        self.seed, self.gate, self.gate_flips = seed, gate, gate_flips
        self.ignore = frozenset(ignore)                 # oracle ceiling only: neighbours never heard
        self.flips: Dict[str, Dict[str, int]] = {}      # me -> neighbour -> unexplained changes seen
        self.seen_done: Dict[str, bool] = {}            # me -> did my last look show no clash?

    def describe(self) -> Dict[str, Any]:
        return {"policy": self.name, "gate": self.gate, "gate_flips": self.gate_flips, "ignore": len(self.ignore)}

    def decide(self, view: LocalView) -> Intent:
        choice = self.choose(self._sensor.sense(view).fields)
        return Intent(NOOP) if choice is None else Intent(colour_action(choice))

    def choose(self, f: Dict[str, Any]) -> Optional[int]:
        me, heard, mine = f["agent"], f["heard"], f["colour"]
        nbrs = [u for u in f["neighbours"] if u not in self.ignore]
        if self.gate:
            seen = self.flips.setdefault(me, {})
            for u in f["flipped"]:
                # The one reason for a change this ant can see is a neighbour moving off its own colour. Any
                # other change is unexplained from here (an honest ant may be fixing a clash this ant can't see,
                # which is why it takes several before a neighbour stops being heard).
                if heard[u]["prev"] != mine:
                    seen[u] = seen.get(u, 0) + 1
            ignored = {u for u, k in seen.items() if k >= self.gate_flips}
            nbrs = [u for u in nbrs if u not in ignored]
        held = {u: heard[u]["colour"] for u in nbrs if heard[u]["colour"] is not None}
        if mine is None:
            # Answer alone first: a colour of my own, before anyone has said anything.
            rng = random.Random(f"look:{self.seed}:{me}")
            self.seen_done[me] = False
            return rng.randrange(f["palette"])
        clash = [u for u, c in held.items() if c == mine and self._yields_to(u, me, heard)]
        if clash:
            self.seen_done[me] = False
            taken = set(held.values())
            return next((c for c in range(f["palette"]) if c not in taken), mine)
        self.seen_done[me] = not any(c == mine for c in held.values())
        if any(heard[u]["ack"] != mine for u in nbrs):
            return mine  # not everyone has heard my colour yet: say it again
        return None

    def _yields_to(self, u: str, me: str, heard: Dict[str, Dict[str, Any]]) -> bool:
        """On a clash, does this ant give way to neighbour ``u``? Here: to any neighbour with priority (a lower id)."""
        return int(u) < int(me)


class GradedColouringWorld(SwarmColouringWorld):
    """The swarm colouring world where each announcement also carries how sure its sender is (waggle-dance marks).

    A colour action may pass ``params={"strength": k}``; the strength is clamped to ``[1, cap]`` and sent with every
    later announcement of that agent until it sets a new one (default 1). Liars always claim ``cap``. Listeners see it
    as ``heard[u]["strength"]``. Nothing else changes, so an ant that ignores strength plays exactly as in
    :class:`SwarmColouringWorld` (tested).
    """

    def __init__(self, seed: int, *, cap: int = 2, **kw: Any) -> None:
        super().__init__(seed, **kw)
        if cap < 1:
            raise ValueError("cap must be at least 1")
        self.cap = cap
        self.strength: Dict[str, int] = {a: 1 for a in self.agents()}

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "world": "graph_coloring_swarm_graded", "cap": self.cap}

    def apply(self, agent_id: str, intent: Intent, tick: int):
        out = super().apply(agent_id, intent, tick)
        if out.accepted and intent.action != NOOP:
            self.strength[agent_id] = max(1, min(self.cap, int(intent.params.get("strength", 1))))
        return out

    def step_environment(self, tick: int) -> None:
        super().step_environment(tick)
        for v, senders in self.fresh.items():
            for u in senders:
                self.known[v][u]["strength"] = self.cap if u in self.liars else self.strength[u]

    def observe(self, agent_id: str, tick: int) -> Observation:
        obs = super().observe(agent_id, tick)
        data = dict(obs.data)
        for u, h in data["heard"].items():
            msg = self.known[agent_id].get(u)
            h["strength"] = msg.get("strength") if msg else None
        return Observation(data=data, sense=obs.sense)


class GradedLook(LookBeforeDone):
    """Waggle-dance marks for colouring: think alone, then compare, and say how sure you are.

    An ant's strength is 1 plus the number of rounds in a row its own look showed no clash, capped at ``cap``. It sends
    its strength with its colour, and says it again when the strength changes. On a clash it gives way to a neighbour
    that is surer than it is; between equally sure ants, the lower id keeps its colour, as in :class:`LookBeforeDone`.
    With ``cap=1`` every ant is equally sure and it plays exactly as :class:`LookBeforeDone` (tested).
    """

    name = "graded_look"

    def __init__(self, sensor: Sensor, seed: int, *, cap: int = 2, **kw: Any) -> None:
        super().__init__(sensor, seed, **kw)
        self.cap = cap
        self.streak: Dict[str, int] = {}
        self.said: Dict[str, int] = {}

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "policy": self.name, "cap": self.cap}

    def strength_of(self, me: str) -> int:
        return min(self.cap, 1 + self.streak.get(me, 0))

    def decide(self, view: LocalView) -> Intent:
        f = self._sensor.sense(view).fields
        me = f["agent"]
        choice = self.choose(f)
        self.streak[me] = self.streak.get(me, 0) + 1 if self.seen_done.get(me) else 0
        s = self.strength_of(me)
        if choice is None and f["colour"] is not None and self.cap > 1 and self.said.get(me) != s:
            choice = f["colour"]  # my evidence changed: say my colour again, with the new strength
        if choice is None:
            return Intent(NOOP)
        self.said[me] = s
        return Intent(colour_action(choice), params={"strength": s})

    def _yields_to(self, u: str, me: str, heard: Dict[str, Dict[str, Any]]) -> bool:
        theirs, mine = heard[u].get("strength") or 1, self.strength_of(me)
        return theirs > mine or (theirs == mine and int(u) < int(me))


def default_ant(seed: int) -> LookBeforeDone:
    """The default ant for new swarm runs in this world: think alone, then compare, done only when seen done.
    (The pilot's recorded arms keep their own scripts, so its published results still reproduce.)"""
    return LookBeforeDone(ColoringSensor(), seed)


def done_claims(world: SwarmColouringWorld, policy: Any) -> Dict[str, int]:
    """How many honest agents count themselves done, and how many of those really clash."""
    clashing = world.clashing()
    if isinstance(policy, LookBeforeDone):  # GradedLook included
        done = {a for a in world.honest() if policy.seen_done.get(a)}
    else:
        done = {a for a in world.honest() if world.colour[a] is not None}
    return {"claim_done": len(done), "claim_done_but_clashing": len(done & clashing),
            "clashing": len(clashing & set(world.honest()))}


__all__ = ["GradedColouringWorld", "GradedLook", "LookBeforeDone", "SwarmColouringWorld", "default_ant", "done_claims"]
