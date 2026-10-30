"""Tests for steady eyes: the think-alone forager that checks its own eyes (foraging v2). Small and fast."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from antelligence.kernel.field import GridField
from antelligence.kernel.scheduler import RunConfig, Scheduler
from antelligence.kernel.types import LocalView
from antelligence.worlds.foraging_sb.sensing import ForagingSensor
from antelligence.worlds.foraging_sb.think_alone import (LookForager, RelayForagingWorld, SelfCheckingSensor,
                                                         SteadyEyesForager)
from antelligence.worlds.foraging_sb.world import STEPS, V1_PICKUPS

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _forage(seed, condition, policy, flip_tick=30, relay=False, steps=STEPS):
    world = RelayForagingWorld(seed, condition, relay=relay, n_pickups=V1_PICKUPS, flip_tick=flip_tick, max_steps=steps)
    Scheduler(world, {}, GridField(), RunConfig("t", seed=seed, max_ticks=steps), default_policy=policy,
              record_frames=False).run()
    return world


def test_sensor_reports_exactly_what_the_plain_sensor_does_while_eyes_are_right():
    world = RelayForagingWorld(201, "base", relay=True, max_steps=STEPS)
    plain, checking = ForagingSensor(), SelfCheckingSensor()
    for tick in range(1, 25):
        for a in world.agents():
            obs = world.observe(a, tick)
            view = LocalView(a, tick, 0, 201, obs.data)
            assert checking.sense(view).fields == plain.sense(view).fields
            world.apply(a, LookForager().decide(view), tick)
        world.step_environment(tick)


def test_same_deliveries_as_think_alone_when_eyes_never_flip():
    for condition in ("base", "v1"):
        for seed in (201, 202, 203):
            for relay in (False, True):
                a = _forage(seed, condition, LookForager(), relay=relay).deliveries
                pol = SteadyEyesForager()
                assert _forage(seed, condition, pol, relay=relay).deliveries == a
                assert pol.sensor.stats()["switches"] == 0  # never doubts eyes that are right


def test_notices_the_flip_soon_after_it_happens_without_being_told():
    for seed, flip in ((201, 30), (202, 10), (203, 60)):
        pol = SteadyEyesForager()
        _forage(seed, "v2", pol, flip_tick=flip)
        ticks = pol.sensor.stats()["switch_ticks"]
        assert ticks and all(flip <= t <= flip + 10 for t in ticks)


def test_delivers_far_more_than_think_alone_when_eyes_flip():
    look = sum(_forage(s, "v2", LookForager()).deliveries for s in (201, 202, 203))
    steady = sum(_forage(s, "v2", SteadyEyesForager()).deliveries for s in (201, 202, 203))
    assert steady > 2 * look


def test_stops_listening_while_reading_mirrored():
    """Its partners' talk was read through flipped eyes too, so an ant that knows its eyes are flipped drops it."""
    world = RelayForagingWorld(204, "v2", relay=True, n_pickups=V1_PICKUPS, flip_tick=30, max_steps=STEPS)
    sensor, pol = SelfCheckingSensor(), LookForager()
    heard_while_mirrored = dropped = 0
    for tick in range(1, 61):
        for a in world.agents():
            view = LocalView(a, tick, 0, 204, world.observe(a, tick).data)
            sensed = sensor.sense(view)
            if sensor.mirrored(a) and view.observation["heard"]:
                heard_while_mirrored += 1
                dropped += sensed.fields["heard"] == []
            world.apply(a, pol.decide(view), tick)
        world.step_environment(tick)
    assert sensor.mirrored("0")
    assert heard_while_mirrored > 0 and dropped == heard_while_mirrored


def test_check_script_smoke(tmp_path):
    sec = _load("steady_eyes_check")
    out = tmp_path / "s.json"
    assert sec.main(["--seeds", "200-201", "--conditions", "base,v2", "--workers", "1", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert {c["arm"] for c in data["summary"]["cells"]} == set(sec.ARMS)
    assert all(c["seeds"] == 2 for c in data["summary"]["comparisons"])


def test_check_script_grid_smoke(tmp_path):
    sec = _load("steady_eyes_check")
    out = tmp_path / "g.json"
    assert sec.main(["--grid", "--seeds", "100-100", "--conditions", "base,v2", "--workers", "1",
                     "--out", str(out)]) == 0
    table = json.loads(out.read_text())["table"]
    assert len(table) == len(sec.GRID["switch"]) * len(sec.GRID["leak"])
    assert all(set(c) == {"switch", "leak", "base", "v2"} for c in table)
