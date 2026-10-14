"""A tiny fake world for the domain-free colony kernel tests. Not a tumour world; knows nothing about one."""

from __future__ import annotations

from typing import Any, Dict, List

from antelligence.kernel.colony.episode import ArmSpec
from antelligence.kernel.colony.quorum import ZoneSpec
from antelligence.kernel.colony.wake import scan_band
from antelligence.kernel.types import Intent, Observation, Outcome

BASE_KEYS = ("squad", "tick")
SOLO = ("look", "uptake_reading", "washout_reading", "reading_log", "zone_need", "colony_note", "visited_self",
        "card_self", "notebook_self")
MARKS = SOLO + ("marks_near", "region_status", "cards_public")
TALK = MARKS + ("messages_in",)

ARMS = {
    "S": ArmSpec("S", False, False, False),
    "S-solo": ArmSpec("S-solo", True, False, False, allowed_keys=frozenset(SOLO)),
    "S-marks": ArmSpec("S-marks", True, True, False, allowed_keys=frozenset(MARKS)),
    "S-talk": ArmSpec("S-talk", True, True, True, allowed_keys=frozenset(TALK)),
    "S-talk-shuffled": ArmSpec("S-talk-shuffled", True, True, True, shuffled=True, allowed_keys=frozenset(TALK)),
}


class FakeWorld:
    """Ants on a line. ``params['move']`` teleports; ``params['release']`` drops 1 unit. Looks come from a table."""

    seed = 0
    case_id = "fake"
    voxel_mm = 1.0
    decision_every = 5
    queen_every = 30
    queen_hook = None
    revision = 0

    def __init__(self, n=3, ticks=40, looks=None, spacing=10):
        self.n, self.ticks = n, ticks
        self.looks = looks or {}
        self.pos = {i: (i * spacing, 0, 0) for i in range(n)}
        self.payload = {i: 10.0 for i in range(n)}
        self.applied: List[tuple] = []
        self.goal: Dict[int, Any] = {}

    def agents(self):
        return [f"squad-{i}" for i in range(self.n)]

    def done(self, tick):
        return tick >= self.ticks

    def observe(self, agent_id, tick):
        return Observation(data={"squad": int(agent_id.split("-")[1]), "tick": tick})

    def apply(self, agent_id, intent, tick):
        self.applied.append((agent_id, tick))
        if tick % self.decision_every:
            return Outcome(False, "not a decision tick")
        i = int(agent_id.split("-")[1])
        if intent.params.get("move") is not None:
            self.pos[i] = tuple(intent.params["move"])
        released = 0.0
        if intent.params.get("release"):
            released = 1.0
            self.payload[i] -= 1.0
        return Outcome(True, None, {"released": released})

    def step_environment(self, tick):
        pass

    def metrics(self):
        return {"applied": len(self.applied), "pos": {str(k): list(v) for k, v in self.pos.items()},
                "payload": sum(self.payload.values())}

    def describe(self):
        return {"world": "fake"}

    def ant_pos(self, i):
        return self.pos[i]

    def ant_goal(self, i):
        return self.goal.get(i)

    def ant_payload(self, i):
        return self.payload[i]

    def zones(self):
        c = (10, 0, 0)
        return [ZoneSpec(1, c, lambda p: sum((a - b) ** 2 for a, b in zip(p, c)) ** 0.5 <= 8.0)]

    def read_sensors(self, i, tick):
        out = []
        if self.pos[i] in self.looks:
            out.append({"kind": "look", "value": self.looks[self.pos[i]], "band": scan_band(self.looks[self.pos[i]])})
        out.append({"kind": "dose", "value": 0.0})
        return out

    def release_fraction(self, i, released):
        return released / 10.0

    def visited_self(self, i):
        return {"regions": [], "walked_count": 0}


class Scripted:
    """Policy from a function ``f(view) -> (params, emit, ask)``; records every observation it saw."""

    def __init__(self, fn, seen):
        self.fn, self.seen = fn, seen

    def decide(self, view):
        self.seen.append((view.agent_id, view.tick, dict(view.observation)))
        params, emit, ask = self.fn(view)
        params = dict(params)
        if ask is not None:
            params["ask"] = ask
        return Intent("decide", params, emit=tuple(emit))
