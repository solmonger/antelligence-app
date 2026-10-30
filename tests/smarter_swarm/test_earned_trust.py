"""Tests for earned trust in the finish world (the ``record`` and ``gossip`` arms) and its free check. Small and fast."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from antelligence.worlds.finish import FinishConfig, FinishWorld

REPO = Path(__file__).resolve().parents[2]
SPOT = FinishConfig(n_ants=32, sites_per_ant=4, max_ticks=400, look_cost=3, look_p=0.25)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _same(a, b):
    skip = {"arm", "caught", "caught_wrongly"}
    return {k: v for k, v in a.as_dict().items() if k not in skip} == {k: v for k, v in b.as_dict().items() if k not in skip}


@pytest.mark.parametrize("arm", ["record", "gossip"])
def test_with_full_looking_earned_trust_is_exactly_look(arm):
    cfg = replace(SPOT, look_p=1.0)
    for seed in (1, 2, 3):
        assert _same(FinishWorld(cfg, seed).run(arm, 0.2), FinishWorld(cfg, seed).run("look", 0.2))


@pytest.mark.parametrize("arm", ["record", "gossip"])
def test_with_no_shortcut_takers_nobody_is_caught_and_play_is_look(arm):
    for seed in (1, 2, 3):
        r = FinishWorld(SPOT, seed).run(arm, 0.0)
        assert r.caught == 0
        assert _same(r, FinishWorld(SPOT, seed).run("look", 0.0))


@pytest.mark.parametrize("arm", ["record", "gossip"])
def test_only_shortcut_takers_are_ever_caught(arm):
    held = 0
    for seed in range(1, 7):
        r = FinishWorld(replace(SPOT, n_ants=64), seed).run(arm, 0.2)
        assert r.caught_wrongly == 0
        held += r.caught
    assert held > 0


def test_gossip_spreads_names_further_than_a_private_record():
    cfg = replace(SPOT, n_ants=64)
    rec = sum(FinishWorld(cfg, s).run("record", 0.2).caught for s in range(1, 5))
    gos = sum(FinishWorld(cfg, s).run("gossip", 0.2).caught for s in range(1, 5))
    assert gos > rec


def test_look_arm_still_matches_its_recorded_results():
    """The new arms leave ``look`` untouched: the recorded spot-looking runs regenerate exactly."""
    rec = json.loads((REPO / "docs/research/swarm-scale-20261029/finish_n256_look3_spot25.json").read_text())
    cfg = FinishConfig(**rec["config"])
    rows = [r for r in rec["rows"] if r["arm"] == "look" and r["fade"] == "fixed" and r["shortcut_share"] == 0.2][:2]
    assert rows
    for row in rows:
        r = FinishWorld(replace(cfg, n_ants=row["n"]), row["seed"]).run("look", 0.2)
        assert (r.stop_tick, r.left_at_stop, r.look_ticks, r.lies) == \
            (row["stop_tick"], row["left_at_stop"], row["look_ticks"], row["lies"])


def test_earned_trust_check_smoke(tmp_path):
    etc = _load("earned_trust_check")
    out = tmp_path / "t.json"
    assert etc.main(["--ns", "32", "--shares", "0.2", "--patience", "10", "--seeds", "1-2", "--workers", "1",
                     "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert {c["arm"] for c in data["summary"]["cells"]} == set(etc.ARMS)
    assert all(c["seeds"] == 2 for c in data["summary"]["comparisons"])
