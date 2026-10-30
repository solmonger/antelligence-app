"""Tests for waggle-dance marks in the colouring (AgentsNet) and foraging (SwarmBench) worlds. Small and fast."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from antelligence.kernel.field import GridField
from antelligence.kernel.scheduler import RunConfig, Scheduler
from antelligence.worlds.foraging_sb.think_alone import GradedForager, LookForager, RelayForagingWorld, lead_from
from antelligence.worlds.foraging_sb.world import STEPS, V1_PICKUPS, V2_FLIP_TICK
from antelligence.worlds.graph_coloring.sensing import ColoringSensor
from antelligence.worlds.graph_coloring.think_alone import (GradedColouringWorld, GradedLook, LookBeforeDone,
                                                            SwarmColouringWorld)

REPO = Path(__file__).resolve().parents[2]
STRESS = {"liars": ["3", "11", "20"], "drop_p": 0.3}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _colour(world, policy, rounds):
    Scheduler(world, {}, GridField(), RunConfig("t", seed=world.seed, max_ticks=rounds), default_policy=policy).run()
    return world


def _forage(seed, condition, policy, **kw):
    world = RelayForagingWorld(seed, condition, n_pickups=V1_PICKUPS, flip_tick=V2_FLIP_TICK, max_steps=STEPS, **kw)
    Scheduler(world, {}, GridField(), RunConfig("t", seed=seed, max_ticks=STEPS), default_policy=policy,
              record_frames=False).run()
    return world


# ------------------------------------------------------------------ colouring

@pytest.mark.parametrize("family", ["ws", "dt"])
def test_think_alone_plays_the_same_in_the_graded_world(family):
    """The strength field changes nothing for an ant that ignores it."""
    for seed in (201, 202):
        a = _colour(SwarmColouringWorld(seed, n_agents=48, rounds=4, family=family, **STRESS),
                    LookBeforeDone(ColoringSensor(), seed), 4)
        b = _colour(GradedColouringWorld(seed, n_agents=48, rounds=4, family=family, **STRESS),
                    LookBeforeDone(ColoringSensor(), seed), 4)
        assert a.colour == b.colour


def test_graded_look_with_cap_one_is_think_alone():
    for seed in (201, 202, 203):
        a = _colour(GradedColouringWorld(seed, n_agents=48, rounds=4, family="ws", cap=1, **STRESS),
                    LookBeforeDone(ColoringSensor(), seed), 4)
        b = _colour(GradedColouringWorld(seed, n_agents=48, rounds=4, family="ws", cap=1, **STRESS),
                    GradedLook(ColoringSensor(), seed, cap=1), 4)
        assert a.colour == b.colour


def test_strength_travels_with_the_message_and_liars_claim_the_cap():
    w = _colour(GradedColouringWorld(204, n_agents=48, rounds=5, family="ws", liars=["3"], cap=2),
                GradedLook(ColoringSensor(), 204, cap=2), 5)
    sent = [m.get("strength") for inbox in w.known.values() for m in inbox.values()]
    assert set(sent) <= {1, 2} and 2 in sent
    assert all(inbox["3"]["strength"] == 2 for inbox in w.known.values() if "3" in inbox)


def test_graded_look_still_colours_properly():
    w = _colour(GradedColouringWorld(205, n_agents=48, rounds=8, family="dt"), GradedLook(ColoringSensor(), 205), 8)
    assert w.score() == 1.0


def test_less_sure_ant_gives_way():
    pol = GradedLook(ColoringSensor(), 0, cap=2)
    pol.streak["5"] = 3  # sure of its colour
    heard = {"9": {"strength": 1}, "2": {"strength": 2}, "1": {"strength": 1}}
    assert not pol._yields_to("9", "5", heard)   # less sure, higher id: keeps going
    assert not pol._yields_to("1", "5", heard)   # less sure even with priority: it does not win
    assert pol._yields_to("2", "5", heard)       # equally sure with priority: think alone's rule


def test_bad_cap_is_refused():
    with pytest.raises(ValueError):
        GradedColouringWorld(1, n_agents=16, rounds=2, family="ws", cap=0)
    with pytest.raises(ValueError):
        RelayForagingWorld(1, "base", cap=0)


# ------------------------------------------------------------------ foraging

def test_surest_lead_wins_over_the_nearest():
    heard = [{"from": "1", "text": "food is north of me", "dx": 1, "dy": 0, "strength": 1},
             {"from": "2", "text": "food is south of me", "dx": 3, "dy": 0, "strength": 2}]
    assert lead_from(heard, (5, 5)) == (6, 3)              # nearest
    assert lead_from(heard, (5, 5), graded=True) == (8, 7)  # surest


def test_messages_carry_strength_first_hand_above_relayed():
    w = _forage(201, "v1", LookForager(), relay=True)
    assert w.relayed > 0
    w2 = _forage(201, "v1", LookForager(), relay=True, shout=True)
    assert w2.relayed == w.relayed


@pytest.mark.parametrize("condition", ["base", "v1", "v2"])
def test_think_alone_forager_is_unchanged_by_the_strength_tags(condition):
    """Same deliveries as the recorded foraging check (``foraging.json``) on its own seeds."""
    rec = json.loads((REPO / "docs/research/swarm-scale-20261029/foraging.json").read_text())["rows"]
    for row in [r for r in rec if r["condition"] == condition and r["policy"] == "look"][:6]:
        w = _forage(row["seed"], condition, LookForager(), relay=(row["talk"] == "relay"))
        assert w.deliveries == row["deliveries"]


def test_graded_forager_equals_look_when_every_message_is_first_hand():
    for seed in (201, 202):
        assert _forage(seed, "v1", GradedForager()).deliveries == _forage(seed, "v1", LookForager()).deliveries


def test_ports_check_smoke(tmp_path):
    gpc = _load("graded_ports_check")
    out = tmp_path / "p.json"
    assert gpc.main(["--ns", "32", "--rounds", "3", "--colour-seeds", "200-200", "--forage-seeds", "200-201",
                     "--workers", "1", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert {c["condition"] for c in data["colouring"]["comparisons"]} == set(gpc.COLOUR_CONDITIONS)
    assert {(c["condition"], c["talk"]) for c in data["foraging"]["comparisons"]} == \
        {(c, t) for c in gpc.FORAGE_CONDITIONS for t in gpc.TALK}
