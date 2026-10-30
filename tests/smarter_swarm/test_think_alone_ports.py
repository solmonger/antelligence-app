"""Tests for think alone, done included, in the colouring (AgentsNet) and foraging (SwarmBench) worlds."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from antelligence.kernel.field import GridField
from antelligence.kernel.scheduler import RunConfig, Scheduler
from antelligence.worlds.foraging_sb.scripts import ExploreWhenEmpty, ShortestPath
from antelligence.worlds.foraging_sb.think_alone import LookForager, RelayForagingWorld, lead_from
from antelligence.worlds.foraging_sb.world import STEPS, V1_PICKUPS, V2_FLIP_TICK, ForagingSBWorld
from antelligence.worlds.graph_coloring.scripts import PatientGreedy
from antelligence.worlds.graph_coloring.sensing import ColoringSensor
from antelligence.worlds.graph_coloring.think_alone import LookBeforeDone, SwarmColouringWorld, done_claims
from antelligence.worlds.graph_coloring.world import GraphColoringWorld

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_colouring(world, policy, rounds):
    Scheduler(world, {}, GridField(), RunConfig("t", seed=world.seed, max_ticks=rounds), default_policy=policy).run()
    return world


# ------------------------------------------------------------------ colouring

def test_swarm_world_without_stress_matches_the_pilot_world():
    a = _run_colouring(GraphColoringWorld(101, "base", n_agents=32, rounds=6, family="ws"),
                       PatientGreedy(ColoringSensor()), 6)
    b = _run_colouring(SwarmColouringWorld(101, n_agents=32, rounds=6, family="ws"),
                       PatientGreedy(ColoringSensor()), 6)
    assert a.colour == b.colour and a.score() == b.score()


def test_look_colours_properly_and_only_claims_done_when_it_is():
    w = _run_colouring(SwarmColouringWorld(102, n_agents=48, rounds=8, family="dt"),
                       pol := LookBeforeDone(ColoringSensor(), 102), 8)
    assert w.score() == 1.0
    claims = done_claims(w, pol)
    assert claims["claim_done_but_clashing"] == 0 and claims["claim_done"] == 48


def test_look_beats_the_pilot_script_under_loss_and_liars():
    liars = ["3", "11", "20"]
    s = _run_colouring(SwarmColouringWorld(103, n_agents=48, rounds=6, family="ws", liars=liars, drop_p=0.3),
                       PatientGreedy(ColoringSensor()), 6)
    look = _run_colouring(SwarmColouringWorld(103, n_agents=48, rounds=6, family="ws", liars=liars, drop_p=0.3),
                          LookBeforeDone(ColoringSensor(), 103), 6)
    assert look.score() > s.score()
    assert not any(a in liars for a in look.honest())
    assert all(u not in liars and v not in liars for u, v in look.scored_edges())


def test_gate_stops_hearing_a_neighbour_that_keeps_changing_for_no_reason():
    pol = LookBeforeDone(ColoringSensor(), 0, gate=True, gate_flips=2)
    heard = {"1": {"colour": 2, "prev": 1, "ack": 0, "fresh": True}}
    f = {"agent": "5", "colour": 0, "palette": 4, "neighbours": ["1"], "flipped": ["1"], "heard": heard}
    pol.choose(f)
    pol.choose(f)
    assert pol.flips["5"]["1"] == 2
    heard["1"]["colour"] = 0  # the gated neighbour now "clashes" (and has heard my colour), but is no longer heard
    assert pol.choose(dict(f, flipped=[])) is None and pol.seen_done["5"]
    ungated = LookBeforeDone(ColoringSensor(), 0)
    assert ungated.choose(dict(f, flipped=[])) == 1  # a heard clash with a lower id makes it move


def test_world_refuses_bad_loss():
    with pytest.raises(ValueError):
        SwarmColouringWorld(1, n_agents=16, rounds=4, family="ws", drop_p=1.0)


def test_colouring_check_runs_and_pairs(tmp_path):
    cc = _load("colouring_think_alone_check")
    out = tmp_path / "c.json"
    assert cc.main(["--ns", "16", "--seeds", "100,101", "--rounds", "4", "--conditions", "base,both",
                    "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert {r["policy"] for r in data["rows"]} == set(cc.POLICIES)
    assert all(c["pairs"] >= 2 for c in data["summary"]["comparisons"])


# ------------------------------------------------------------------- foraging

def _run_forage(world, policy):
    Scheduler(world, {}, GridField(), RunConfig("f", seed=world.seed, max_ticks=STEPS), default_policy=policy,
              record_frames=False).run()
    return world


def test_relay_off_is_the_pilot_world():
    for seed in (100, 101):
        a = _run_forage(ForagingSBWorld(seed, "v1", n_pickups=V1_PICKUPS, flip_tick=V2_FLIP_TICK), ShortestPath())
        b = _run_forage(RelayForagingWorld(seed, "v1", relay=False, n_pickups=V1_PICKUPS, flip_tick=V2_FLIP_TICK),
                        ShortestPath())
        assert a.deliveries == b.deliveries and b.relayed == 0


def test_relay_on_broadcasts_remembered_spots():
    w = _run_forage(RelayForagingWorld(100, "v1", relay=True, n_pickups=V1_PICKUPS, flip_tick=V2_FLIP_TICK),
                    ShortestPath())
    assert w.relayed > 0


def test_lead_parsing():
    heard = [{"from": "3", "text": "nest is north of me", "dx": 1, "dy": 0},
             {"from": "4", "text": "food is north-west of me", "dx": 2, "dy": 2}]
    assert lead_from(heard, (5, 5)) == (5, 5)  # (5+2-2, 5+2-2)
    assert lead_from(heard, (5, 5), want="nest") == (6, 3)
    assert lead_from([{"from": "1", "text": "food is sideways of me", "dx": 0, "dy": 0}], (1, 1)) is None


def test_look_forager_is_the_pilot_s_prime_when_nothing_is_heard_and_beats_s_when_food_runs_out():
    for seed in (100, 102, 104):
        kw = dict(n_pickups=V1_PICKUPS, flip_tick=V2_FLIP_TICK)
        s = _run_forage(RelayForagingWorld(seed, "v1", **kw), ShortestPath()).deliveries
        sp = _run_forage(RelayForagingWorld(seed, "v1", **kw), ExploreWhenEmpty()).deliveries
        quiet = _run_forage(RelayForagingWorld(seed, "v1", **kw), LookForager(use_heard=False)).deliveries
        look = _run_forage(RelayForagingWorld(seed, "v1", **kw), LookForager()).deliveries
        assert quiet == sp and look > s


def test_foraging_check_runs(tmp_path):
    fc = _load("foraging_think_alone_check")
    out = tmp_path / "f.json"
    assert fc.main(["--seeds", "100,101", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert len(data["rows"]) == 2 * len(fc.CONDITIONS) * len(fc.TALK) * len(fc.POLICIES)


def test_think_alone_is_the_default_ant_everywhere():
    from antelligence.worlds.finish import FinishConfig, FinishWorld
    from antelligence.worlds.foraging_sb.think_alone import default_ant as forage_default
    from antelligence.worlds.graph_coloring.think_alone import default_ant as colour_default
    from antelligence.worlds.rumour import RumourConfig, RumourWorld

    assert RumourWorld(RumourConfig(n_ants=16, ticks=5), 0).run().policy == "alone"
    assert FinishWorld(FinishConfig(n_ants=8, sites_per_ant=2, max_ticks=50), 0).run().arm == "look"
    assert isinstance(colour_default(0), LookBeforeDone)
    assert isinstance(forage_default(), LookForager)  # think alone underneath ...
    from antelligence.worlds.foraging_sb.think_alone import SteadyEyesForager
    assert isinstance(forage_default(), SteadyEyesForager)  # ... with the eye check on top
