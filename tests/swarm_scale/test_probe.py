"""Tests for scripts/swarm_scale_probe.py (many liars, many agents, one model-free gate). No model calls."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("swarm_scale_probe", REPO / "scripts" / "swarm_scale_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swarm_scale_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load()


def test_liars_are_seeded_and_sized():
    a = probe.pick_liars(64, 0.1, 100)
    assert a == probe.pick_liars(64, 0.1, 100)
    assert len(a) == round(64 * 0.1)
    assert probe.pick_liars(64, 0.0, 100) == []


def test_scored_edges_skip_every_liar():
    world = probe.ManyLiarsWorld(100, n_agents=16, rounds=4, family="ws", liars=["2", "5"])
    edges = world.scored_edges()
    assert edges and all("2" not in e and "5" not in e for e in edges)
    honest_edges = [e for e in world.graph.edges if 2 not in e and 5 not in e]
    assert len(edges) == len(honest_edges)


def test_liars_chatter_every_round_honest_agents_do_not():
    world = probe.ManyLiarsWorld(100, n_agents=8, rounds=4, family="ws", liars=["3"])
    world.step_environment(1)  # nobody has taken a colour yet: only the liar speaks
    senders = {u for v in world.known for u in world.known[v]}
    assert senders == {"3"}
    assert world.counters["lied"] == len(world.neighbours("3"))


@pytest.mark.parametrize("family", probe.FAMILIES)
def test_no_liars_all_policies_agree(family):
    rows = [probe.run_one(64, 0.0, family, 100, p, 8) for p in probe.POLICIES]
    assert len({r["score"] for r in rows}) == 1


def test_gate_recovers_the_oracle_on_a_small_graph_with_liars():
    for seed in (100, 101, 102):
        s = probe.run_one(8, 0.3, "ws", seed, "S", 8)
        gate = probe.run_one(8, 0.3, "ws", seed, "gate", 8)
        oracle = probe.run_one(8, 0.3, "ws", seed, "oracle", 8)
        assert oracle["score"] >= s["score"]
        assert gate["score"] == oracle["score"]


def test_gate_is_never_told_who_lies():
    pol = probe.IgnoreSet(probe.ColoringSensor(), gate=True)
    assert pol.fixed_ignore == frozenset()


def test_palette_guard_skips_graphs_the_sensor_cannot_name(monkeypatch):
    monkeypatch.setattr(probe, "COLOUR_NAMES", ("red", "blue"))
    assert probe.run_one(16, 0.0, "ws", 100, "S", 4) is None


def test_main_writes_summary(tmp_path):
    out = tmp_path / "p.json"
    assert probe.main(["--ns", "8", "--fracs", "0,0.3", "--seeds", "100", "--rounds", "4", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert [s["liar_frac"] for s in data["summary"]] == [0.0, 0.3]
    assert set(data["summary"][0]["mean_score"]) == set(probe.POLICIES)
