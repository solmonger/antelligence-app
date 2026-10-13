"""Oracle S' as a ceiling (L2b): beats the script on every seed, plans on projected dose, coordinates squads, is
deterministic and fast."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from antelligence.kernel.types import LocalView
from antelligence.worlds.tumor_mri import (Crop, ScriptS, TumorMRIWorld, oracle_for, run_episode, select_candidates)
from tests.v4_1.world.phantom import make_phantom, oracle_config, phantom_config

FIX = Path(__file__).parent / "fixtures"
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
BIG, SMALL = (18.5, 16.5, 17.5), (10.5, 23.5, 14.5)  # phantom tumour centres (voxels)


@pytest.fixture(scope="module")
def crop() -> Crop:
    return Crop.from_npz(FIX / "phantom.npz")


def _cov(crop, seed, oracle: bool, cfg=None) -> float:
    w = TumorMRIWorld(crop, cfg or oracle_config(), seed=seed)
    o = oracle_for(w) if oracle else None
    return run_episode(w, (lambda a: o) if oracle else (lambda a: ScriptS()))["coverage"]


def test_oracle_beats_script_every_seed(crop):
    for seed in SEEDS:
        s, sp = _cov(crop, seed, False), _cov(crop, seed, True)
        assert sp > s, (seed, s, sp)
        assert sp >= 0.6, (seed, sp)  # a sane oracle reaches 0.6 with this much drug


def test_oracle_deterministic(crop):
    runs = []
    for _ in range(2):
        w = TumorMRIWorld(crop, oracle_config(), seed=4)
        o = oracle_for(w)
        goals = []

        class Spy:
            def decide(self, view):
                it = o.decide(view)
                goals.append((view.tick, view.observation["squad"], it.params.get("target_vox"), it.params["release"], it.params["amount"]))
                return it

        m = run_episode(w, lambda a: Spy())
        runs.append((m, goals, w.field.dose.copy()))
    assert runs[0][0] == runs[1][0] and runs[0][1] == runs[1][1]
    assert np.array_equal(runs[0][2], runs[1][2])


def _start(crop, cfg=None, seed=0):
    w = TumorMRIWorld(crop, cfg or oracle_config(), seed=seed)
    w.set_candidates(select_candidates(w.regions))
    return w, oracle_for(w)


def test_oracle_uses_projected_dose(crop):
    # (1) the projection is the world's own future: with drug in flight and no more releases, projected == final dose
    w, o = _start(crop)
    for pos, m in (((18, 16, 17), 150.0), ((12, 22, 15), 40.0)):
        w.field.deposit(pos, m)
    w.step_environment(0)
    w.step_environment(1)
    proj = o._projected(2)
    assert (proj >= w.field.dose).all() and proj.sum() > w.field.dose.sum() * 1.5  # in-flight drug adds exposure
    for t in range(2, w.config.ticks):
        w.step_environment(t)
    truth = w.scorer_truth().truth
    final = w.field.dose
    assert np.abs(proj[truth] - final[truth]).max() < 0.02 * final[truth].max()  # box-limited, so close not bit-exact
    # (2) it plans on that projection: drug already in flight that will cover the big tumour is not sent there again
    w, o = _start(crop, oracle_config(total_drug=1500.0))
    for pos in ((18, 16, 17), (19, 17, 18), (17, 17, 17), (18, 16, 18)):
        w.field.deposit(pos, 220.0)
    sc = w.scorer_truth()
    projected_cov = float(((o._projected(0) >= sc.dose_threshold) & sc.truth).sum()) / sc.truth.sum()
    assert projected_cov > 0.5  # in flight, nothing has been delivered yet
    assert sc.coverage(w.field.dose) == 0.0
    o._plan(0, 0, w.squad_payloads()[0])
    goals = [a.goal for a in o._actions.values() if a.goal is not None]
    releases = [w.squad_positions()[i] for i, a in o._actions.items() if a.release_level]
    assert goals or releases
    targets = goals + releases
    big_hits = [p for p in targets if np.linalg.norm(np.subtract(p, BIG)) <= 4.0]
    assert len(big_hits) <= 1, (targets, "squads should leave the already-covered tumour core alone")
    assert any(np.linalg.norm(np.subtract(p, SMALL)) <= 6.0 for p in targets), targets  # ... and go for what is left


def test_oracle_squads_coordinate(crop):
    for seed in (1, 2, 3):
        w = TumorMRIWorld(crop, oracle_config(), seed=seed)
        o = oracle_for(w)
        w.set_candidates(select_candidates(w.regions))
        while not w.done(w._tick):
            tick = w._tick
            if tick % w.config.decision_every == 0:
                goals = []
                for a in w.agents():
                    obs = w.observe(a, tick).data
                    it = o.decide(LocalView(a, tick, w.revision, seed, obs))
                    if it.params.get("target_vox") is not None and tuple(it.params["target_vox"]) != tuple(obs["position_vox"]):
                        goals.append(tuple(it.params["target_vox"]))
                    w.apply(a, it, tick)
                # every point claimed this tick is distinct (a duplicate is allowed only if no alternative reaches 90% of its value)
                assert len(goals) == len(set(goals)), (seed, tick, goals)
            w.step_environment(tick)


def test_oracle_claimed_point_loses_to_a_90pct_alternative(crop):
    w, o = _start(crop)
    proj = o._projected(0)
    o._target_set(proj, 0, 400.0)
    pos = w.squad_positions()[0]
    best = o._best_walk(proj, pos, 0, 75.0, None)
    assert best is not None
    o._claimed = [best.point]
    second = o._best_walk(proj, pos, 0, 75.0, None)
    assert second is not None
    if second.point == best.point:  # still the pick: then no other point reached 90% of its value
        others = [o._option(proj, x, o._steps(pos, x, False), 0, 75.0) for x in o._candidate_points(proj) if x != best.point]
        assert all(v is None or v.eff < 0.9 * best.eff for v in others)
    else:
        assert second.eff >= 0.85 * best.eff  # beat the claimed point's discounted value


def test_oracle_speed_100_cube():
    n = 100
    g = np.indices((n, n, n)).astype(np.float32)
    c = (n - 1) / 2
    brain = ((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) <= 48 ** 2
    truth = ((g[0] - 55) ** 2 + (g[1] - 50) ** 2 + (g[2] - 45) ** 2) <= 10 ** 2
    p = ndimage.gaussian_filter(truth.astype(np.float64), 1.5) * 1.8
    big = Crop.from_arrays(dict(brain=brain, p_tumour=np.clip(p, 0, 1), unc=np.clip(4 * p * (1 - p), 0, 1), truth=truth))
    cfg = phantom_config(total_drug=2000.0)
    t0 = time.perf_counter()
    w = TumorMRIWorld(big, cfg, seed=0)
    o = oracle_for(w)
    m = run_episode(w, lambda a: o)
    assert time.perf_counter() - t0 < 5.0
    s = run_episode(TumorMRIWorld(big, cfg, seed=0), lambda a: ScriptS())
    assert m["coverage"] > s["coverage"] + 0.05  # a real ceiling on a big round tumour too


def test_oracle_reads_only_through_the_world_accessors(crop):
    """S' is the one policy allowed to see truth, and it gets it only via the scorer accessor."""
    w = TumorMRIWorld(crop, oracle_config(), seed=0)
    assert not hasattr(w, "truth") and not hasattr(w, "crop")
    o = oracle_for(w)
    assert o._scorer is w.scorer_truth()
