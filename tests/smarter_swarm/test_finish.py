"""Tests for the finish world (shortcut-takers who skip the last bit and call it done) and scripts/finish_check.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from antelligence.worlds.finish import ARMS, FinishConfig, FinishWorld

REPO = Path(__file__).resolve().parents[2]
SMALL = FinishConfig(n_ants=16, sites_per_ant=4, max_ticks=300)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_runs_are_deterministic():
    a = FinishWorld(SMALL, 3).run("trust", 0.2).as_dict()
    b = FinishWorld(SMALL, 3).run("trust", 0.2).as_dict()
    assert a == b


def test_world_is_shared_by_every_arm():
    w = FinishWorld(SMALL, 1)
    totals = {FinishWorld(SMALL, 1).run(arm).total_work for arm in ARMS}
    assert totals == {sum(w.work0)}
    assert len(w.work0) == SMALL.n_sites == 64


def test_shortcut_takers_are_nested_and_never_inspectors():
    w = FinishWorld(replace(SMALL, n_ants=40), 2)
    assert w.shortcut_ants(0.05) <= w.shortcut_ants(0.1) <= w.shortcut_ants(0.2)
    assert len(w.shortcut_ants(0.2)) == 8
    assert not (w.shortcut_ants(0.4) & w.inspector_ants())


def test_with_no_shortcuts_trust_and_look_do_the_same_work():
    for seed in range(3):
        w = FinishWorld(SMALL, seed)
        t, l = w.run("trust", 0.0), w.run("look", 0.0)
        assert t.lies == l.lies == 0
        assert t.work_ticks == l.work_ticks and t.left_at_stop == l.left_at_stop


def test_accounting_adds_up():
    for arm in ARMS:
        r = FinishWorld(SMALL, 4).run(arm, 0.2)
        assert r.work_ticks == r.total_work - r.left_at_stop
        assert 0 <= r.skipped_share <= 1 and r.open_pieces_at_stop <= SMALL.n_sites
        assert r.said_done_early == (r.stop_tick is not None and r.left_at_stop > 0)
        assert r.left_by_tick == sorted(r.left_by_tick, reverse=True)  # work left never grows


def test_trusted_claims_leave_work_undone_and_looking_finds_it():
    cfg = FinishConfig(n_ants=32, sites_per_ant=4, max_ticks=400)
    trust = [FinishWorld(cfg, s).run("trust", 0.2) for s in range(3)]
    look = [FinishWorld(cfg, s).run("look", 0.2) for s in range(3)]
    assert all(t.lies > 0 for t in trust)
    assert sum(t.left_at_stop for t in trust) > sum(l.left_at_stop for l in look)
    assert all(t.open_pieces_at_stop >= l.open_pieces_at_stop for t, l in zip(trust, look))


def test_look_arm_pays_for_rechecks():
    r = FinishWorld(SMALL, 0).run("look", 0.1)
    assert r.look_ticks > 0
    assert FinishWorld(SMALL, 0).run("trust", 0.1).look_ticks == 0
    free = FinishWorld(replace(SMALL, look_cost=0), 0).run("look", 0.1)
    assert free.look_ticks == 0


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        FinishWorld(SMALL, 0).run("boss")
    with pytest.raises(ValueError):
        FinishWorld(SMALL, 0).run("trust", 0.6)
    with pytest.raises(ValueError):
        FinishConfig(n_ants=4)
    with pytest.raises(ValueError):
        FinishConfig(tail=1.0)
    with pytest.raises(ValueError):
        FinishConfig(patience=0)


def test_finish_check_grid_is_paired_and_complete(tmp_path):
    fc = _load("finish_check")
    out = tmp_path / "f.json"
    assert fc.main(["--ns", "16", "--seeds", "0,1", "--shares", "0,0.2", "--max-ticks", "200", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert len(data["rows"]) == 2 * len(fc.ARMS) * 2
    names = {(c["a"], c["b"], c["metric"]) for c in data["summary"]["comparisons"]}
    assert ("trust/fixed", "look/fixed", "skipped_share") in names
    assert all(c["seeds"] == 2 for c in data["summary"]["comparisons"])


def test_spot_looking_rechecks_less():
    cfg = replace(SMALL, look_cost=3)
    full = FinishWorld(cfg, 2).run("look", 0.2)
    spot = FinishWorld(replace(cfg, look_p=0.25), 2).run("look", 0.2)
    assert spot.look_ticks < full.look_ticks
    assert FinishWorld(replace(cfg, look_p=1.0), 2).run("look", 0.2).as_dict() == full.as_dict()
    with pytest.raises(ValueError):
        FinishConfig(look_p=1.5)
