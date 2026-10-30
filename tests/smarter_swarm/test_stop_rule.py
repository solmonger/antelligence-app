"""Tests for the finish world's stop rules (``idle_rule``) and scripts/stop_rule_check.py. Small and fast."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from antelligence.worlds.finish import FinishConfig, FinishWorld
from antelligence.worlds.finish.world import IDLE_RULES

REPO = Path(__file__).resolve().parents[2]
SLOW = FinishConfig(n_ants=32, sites_per_ant=4, max_ticks=400, look_cost=3, look_p=0.25)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_unknown_stop_rule_is_refused():
    with pytest.raises(ValueError):
        FinishConfig(idle_rule="never")


@pytest.mark.parametrize("rule", IDLE_RULES)
def test_every_rule_is_deterministic_and_accounts_for_all_work(rule):
    cfg = replace(SLOW, idle_rule=rule)
    for arm in ("trust", "look"):
        a = FinishWorld(cfg, 4).run(arm, 0.2)
        assert a.as_dict() == FinishWorld(cfg, 4).run(arm, 0.2).as_dict()
        assert a.work_ticks + a.left_at_stop == a.total_work


def test_default_rule_is_ticks_and_recorded_runs_regenerate():
    assert FinishConfig().idle_rule == "ticks"
    rec = json.loads((REPO / "docs/research/swarm-scale-20261029/finish_n256.json").read_text())
    cfg = FinishConfig(**rec["config"])
    rows = [r for r in rec["rows"] if r["fade"] == "fixed" and r["shortcut_share"] == 0.2
            and r["arm"] in ("trust", "look")][:4]
    assert rows
    for row in rows:
        r = FinishWorld(replace(cfg, n_ants=row["n"]), row["seed"]).run(row["arm"], 0.2)
        assert (r.stop_tick, r.left_at_stop, r.look_ticks) == (row["stop_tick"], row["left_at_stop"], row["look_ticks"])


def test_looking_around_takes_more_ticks_per_idle_count_than_standing_still():
    """Under ``places`` an ant only counts a site once, so at the same patience it never stops sooner than ``walk``,
    which walks the same way but counts every tick."""
    for seed in range(1, 6):
        places = FinishWorld(replace(SLOW, idle_rule="places", patience=8), seed).run("look", 0.2)
        walk = FinishWorld(replace(SLOW, idle_rule="walk", patience=8), seed).run("look", 0.2)
        assert places.stop_tick >= walk.stop_tick


def test_stop_rule_check_smoke(tmp_path):
    src = _load("stop_rule_check")
    out = tmp_path / "s.json"
    assert src.main(["--ns", "32", "--dev-seeds", "0-1", "--seeds", "100-101", "--regimes", "slow", "--workers", "1",
                     "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert {c["rule"] for c in data["comparisons"]} == {"walk", "places"}
    assert all(c["skipped_pct"]["seeds"] == 2 for c in data["comparisons"])
    assert {c["rule"] for c in data["curve"]} == set(src.GRID)


def test_stop_rule_check_refuses_reused_seeds():
    src = _load("stop_rule_check")
    with pytest.raises(SystemExit):
        src.main(["--dev-seeds", "0-9", "--seeds", "5-20", "--workers", "1"])
