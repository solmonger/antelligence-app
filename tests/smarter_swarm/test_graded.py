"""Tests for waggle-dance marks (the ``graded`` policy in the rumour world) and its free check. Small and fast."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from antelligence.kernel import fading
from antelligence.worlds.rumour import SEEDS, RumourConfig, RumourWorld

REPO = Path(__file__).resolve().parents[2]
SMALL = RumourConfig(n_ants=48, ticks=30)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _deposits(monkeypatch, cfg: RumourConfig, policy: str, kind: str):
    """Run once and record every (author, amount) the world deposits."""
    seen = []
    real = fading.FadingMemory.deposit

    def spy(self, place, author, value, tick, amount=1.0, *args, **kwargs):
        seen.append((author, amount))
        return real(self, place, author, value, tick, amount, *args, **kwargs)

    monkeypatch.setattr(fading.FadingMemory, "deposit", spy)
    world = RumourWorld(cfg, 0)
    world.run(policy, kind)
    return world, seen


@pytest.mark.parametrize("smart", [False, True])
@pytest.mark.parametrize("kind", SEEDS)
def test_cap_one_is_exactly_think_alone(smart, kind):
    """With a cap of one reading, graded marks are plain think-alone marks, liars included."""
    for seed in (0, 1, 2):
        cfg = replace(SMALL, smart=smart, grade_cap=1.0)
        world = RumourWorld(cfg, seed)
        assert world.run("graded", kind).as_dict() | {"policy": "x"} == world.run("alone", kind).as_dict() | {"policy": "x"}


def test_honest_marks_are_whole_readings_up_to_the_cap(monkeypatch):
    cfg = replace(SMALL, q=0.5, grade_cap=3.0)
    _, seen = _deposits(monkeypatch, cfg, "graded", "none")
    amounts = {amount for _, amount in seen}
    assert amounts <= {1.0, 2.0, 3.0}
    assert 3.0 in amounts and 1.0 in amounts  # strong and weak evidence both show up with this much reading


def test_think_alone_marks_stay_at_one(monkeypatch):
    _, seen = _deposits(monkeypatch, replace(SMALL, q=0.5), "alone", "liars")
    assert {amount for _, amount in seen} == {1.0}


def test_liars_shout_at_the_cap(monkeypatch):
    cfg = replace(SMALL, grade_cap=2.5, liar_frac=0.1)
    world, seen = _deposits(monkeypatch, cfg, "graded", "liars")
    liars = {str(a) for a in world.liar_pool}
    assert {amount for author, amount in seen if author in liars} == {2.5}


def test_amount_rounds_away_float_residue():
    world = RumourWorld(SMALL, 0)
    llr = SMALL.llr
    assert world._amount("graded", 2 * llr - 1e-12, llr) == 2.0
    assert world._amount("graded", -(2 * llr + 1e-12), llr) == 2.0
    assert world._amount("graded", 1e-15, llr) == 1.0  # a mark is never weaker than one reading
    assert world._amount("graded", 9 * llr, llr) == 3.0
    assert world._amount("alone", 9 * llr, llr) == 1.0


def test_graded_is_deterministic():
    a = RumourWorld(SMALL, 4).run("graded", "burst").as_dict()
    assert a == RumourWorld(SMALL, 4).run("graded", "burst").as_dict()


def test_cap_below_one_reading_is_refused():
    for cap in (0.0, -1.0, 0.5):
        with pytest.raises(ValueError):
            RumourConfig(grade_cap=cap)


def test_graded_marks_help_on_these_seeds():
    """Regression guard for the headline: at the same listening weight, stronger marks for stronger evidence leave more
    ants right. Deterministic seeds, so this is a fixed fact about the code, not a statistical claim."""
    cfg = RumourConfig(n_ants=128, ticks=60)
    alone = [RumourWorld(cfg, s).run("alone", "none").right for s in range(4)]
    graded = [RumourWorld(cfg, s).run("graded", "none").right for s in range(4)]
    assert sum(graded) > sum(alone)


def test_check_script_smoke(tmp_path):
    gmc = _load("graded_marks_check")
    out = tmp_path / "g.json"
    assert gmc.main(["--tune-n", "16", "--tune-seeds", "0-1", "--test", "16:5-6", "--ticks", "8", "--workers", "1",
                     "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    s = data["summary"]
    assert set(data["tuning"]["picks"]) == {"fixed/alone", "fixed/graded", "smart/alone", "smart/graded"}
    assert s["best_alone"].endswith("/alone*") and s["best_graded"].endswith("/graded*")
    assert {r["seed"] for r in data["rows"]} == {5, 6}
    assert all(c["seeds"] == 2 for c in s["comparisons"])
    assert {r["seeded"] for r in data["rows"]} == set(gmc.KINDS)


def test_check_script_refuses_reused_seeds():
    gmc = _load("graded_marks_check")
    with pytest.raises(SystemExit):
        gmc.main(["--tune-seeds", "0-3", "--test", "16:3-5", "--workers", "1"])
