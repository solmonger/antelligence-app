"""W1: hidden facts F2 (poor uptake), F3 (washout), the oracle S' update, sensors (goal 5.2-5.5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from antelligence.kernel.types import Intent
from antelligence.worlds.tumor_mri import (OBS_KEYS, Crop, ScriptS, TumorMRIWorld, WorldConfig, oracle_for, run_episode)
from antelligence.worlds.tumor_mri.config import FactsConfig
from antelligence.worlds.tumor_mri.physics import DrugField, rng, seed64, uptake_source_a, uptake_source_b, washout_field
from antelligence.worlds.tumor_mri.store_adapter import load_case
from antelligence.worlds.tumor_mri.world import ACTION_DECIDE
from tests.v4_1.world.phantom import oracle_config
from tests.v4_1.world.test_store_adapter import N, _write_case, repo  # noqa: F401  (repo is a fixture)

PHANTOM = Path(__file__).resolve().parents[2] / "v4_1" / "world" / "fixtures" / "phantom.npz"
GOLDEN = json.loads((Path(__file__).parent / "golden_v41.json").read_text())
KEYS = ("coverage", "off_target_share", "released_total", "decayed", "lost_boundary", "mass_residual", "time_to_50_coverage")


@pytest.fixture(scope="module")
def crop() -> Crop:
    return Crop.from_npz(PHANTOM)


def _channels(crop: Crop):
    """Synthetic TC / ET for the phantom: TC = eroded WT, ET = a shell-free inner core."""
    from scipy import ndimage
    tc = ndimage.binary_erosion(crop.truth, iterations=1)
    et = ndimage.binary_erosion(crop.truth, iterations=2)
    return {"tc": tc, "et": et}


def _facts(**kw) -> FactsConfig:
    return FactsConfig(**kw)


def _run(crop, arm, seed, cfg, channels=None):
    w = TumorMRIWorld(crop, cfg, seed=seed, scorer_channels=channels)
    o = oracle_for(w) if arm == "Sprime" else None
    m = run_episode(w, (lambda a: o) if o else (lambda a: ScriptS()))
    return w, m


# --- F off == v4.1 -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("arm", ["S", "Sprime"])
def test_facts_off_reproduces_v41_golden(crop, arm, seed):
    for cfg in (oracle_config(), oracle_config(facts=FactsConfig())):  # default and explicit all-off
        w, m = _run(crop, arm, seed, cfg)
        g = GOLDEN[f"{arm}-{seed}"]
        for k in KEYS:
            assert m[k] == pytest.approx(g[k], rel=1e-9, abs=1e-9), (arm, seed, k)
        assert float(w.field.dose.sum()) == pytest.approx(g["dose_sum"], rel=1e-9)
        assert "facts" not in m and w.field.uptake is None and w.field.washout is None and w.field.dose_raw is None


def test_config_hash_unchanged_when_off_and_block_present_when_on():
    off = oracle_config()
    assert "facts" not in off.to_dict()
    on = oracle_config(facts=_facts(f3=True, f3_decay=0.04, f3_share=0.1))
    assert on.to_dict()["facts"]["f3_decay"] == 0.04
    assert WorldConfig.from_dict(on.to_dict()) == on
    with pytest.raises(ValueError):
        _facts(f3=True)  # on without decay/share
    with pytest.raises(ValueError):
        _facts(f2_source="C")


def test_observation_keys_unchanged_and_hidden_truth_not_observed(crop):
    assert OBS_KEYS == ("squad", "tick", "round", "rounds", "position_vox", "payload", "payload_fraction", "last_release",
                        "local_p_tumour", "local_unc", "local_look", "local_dose", "local_dose_rel", "target",
                        "target_region", "candidates", "goal_dist_mm", "target_source")
    cfg = oracle_config(facts=_facts(f2=True, u_core=0.3, f3=True, f3_decay=0.08, f3_share=0.2))
    w = TumorMRIWorld(crop, cfg, seed=1, scorer_channels=_channels(crop))
    obs = w.observe("squad-0", 0).data
    assert set(obs) == set(OBS_KEYS)
    assert not hasattr(crop, "tc") and not hasattr(crop, "et")
    assert w.scorer_truth().tc is not None and w.scorer_truth().et is not None
    assert crop.truth.shape == w.scorer_truth().truth.shape


# --- fields ------------------------------------------------------------------------------------------------------------


def test_seed64_matches_interfaces_rule():
    import hashlib
    assert seed64("f1_field", "BRATS_001") == int.from_bytes(hashlib.sha256(b"f1_field|BRATS_001").digest()[:8], "big")
    assert rng("a", 1).random() == rng("a", 1).random() != rng("a", 2).random()


def test_uptake_source_a_and_b(crop):
    ch = _channels(crop)
    u = uptake_source_a(ch["tc"], ch["et"], 0.5)
    core = ch["tc"] & ~ch["et"]
    assert core.any() and np.all(u[core] == 0.5) and np.all(u[~core] == 1.0)
    ub = uptake_source_b(crop.truth, "phantom", 0.5, 0.35, 2.0)
    assert np.array_equal(ub, uptake_source_b(crop.truth, "phantom", 0.5, 0.35, 2.0))  # deterministic
    assert not np.array_equal(ub, uptake_source_b(crop.truth, "other", 0.5, 0.35, 2.0))  # per case
    share = float((ub[crop.truth] < 1.0).mean())
    assert share == pytest.approx(0.35, abs=0.01) and np.all(ub[~crop.truth] == 1.0)


def test_washout_field_covers_share_of_tumour_plus_margin(crop):
    from scipy import ndimage
    d = washout_field(crop.truth, "phantom", 0.08, 0.2, 2.0)
    domain = ndimage.distance_transform_edt(~crop.truth) * 2.0 <= 5.0
    assert set(np.unique(d)) <= {0.0, 0.08}
    assert float((d[domain] > 0).mean()) == pytest.approx(0.2, abs=0.01)
    assert not (d[~domain] > 0).any()
    assert np.array_equal(d, washout_field(crop.truth, "phantom", 0.08, 0.2, 2.0))


# --- physics: F2 and F3 ------------------------------------------------------------------------------------------------


def _field(**kw):
    return DrugField((24, 24, 24), release_sigma_vox=1.0, diffuse_sigma_vox=1.0, decay=0.02, **kw)


def test_f2_dose_accrual_is_u_times_conc_and_mass_unchanged():
    u = np.ones((24, 24, 24))
    u[8:16, 8:16, 8:16] = 0.3
    base, f2 = _field(), _field(uptake=u)
    region = np.zeros((24, 24, 24), bool)
    region[6:18, 6:18, 6:18] = True
    for f in (base, f2):
        f.deposit((12, 12, 12), 50.0)
        for _ in range(20):
            f.step(region)
    assert np.array_equal(base.conc, f2.conc)  # mass accounting untouched
    assert f2.decayed == base.decayed and f2.decayed_in_region == base.decayed_in_region
    assert f2.mass_balance_residual() == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(f2.dose, u * base.dose) and np.allclose(f2.dose_raw, base.dose)
    assert f2.dose[12, 12, 12] == pytest.approx(0.3 * base.dose[12, 12, 12])


def test_f3_mass_balance_and_extra_decay():
    wo = np.zeros((24, 24, 24))
    wo[8:16, 8:16, 8:16] = 0.08
    region = np.zeros((24, 24, 24), bool)
    region[6:18, 6:18, 6:18] = True
    base, f3 = _field(), _field(washout=wo)
    for f in (base, f3):
        f.deposit((12, 12, 12), 50.0)
        f.deposit((4, 4, 20), 20.0)
        for _ in range(40):
            f.step(region)
    assert f3.mass_balance_residual() == pytest.approx(0.0, abs=1e-9)
    assert f3.decayed > base.decayed and f3.mass < base.mass
    assert 0.0 < f3.decayed_in_region <= f3.decayed
    assert f3.decayed_in_region > base.decayed_in_region
    # decayed mass in the region equals the per-voxel sum, recomputed independently
    g = _field(washout=wo)
    g.deposit((12, 12, 12), 50.0)
    manual = 0.0
    for _ in range(10):
        before = g.conc.copy()
        g.step(region)
        # after diffusion conc_d = conc_after / (1 - rate); recover the decayed amount from the post-step state
        rate = 0.02 + wo
        manual += float((g.conc / (1.0 - rate) * rate)[region].sum())
    assert g.decayed_in_region == pytest.approx(manual, rel=1e-9)
    with pytest.raises(ValueError):
        _field(washout=np.full((24, 24, 24), 0.99))


def test_world_f3_mass_balance_and_f2_dose_lower(crop):
    cfg = oracle_config(facts=_facts(f2=True, f2_source="B", u_core=0.3, f2_share=0.35,
                                     f3=True, f3_decay=0.08, f3_share=0.2))
    base_w, base = _run(crop, "S", 2, oracle_config())
    w, m = _run(crop, "S", 2, cfg)
    assert abs(m["mass_residual"]) < 1e-9
    assert 0.0 <= m["off_target_share"] <= 1.0
    assert m["coverage"] <= base["coverage"] + 1e-12
    assert float(w.field.dose.sum()) < float(base_w.field.dose.sum())
    assert w.field.washout.max() == 0.08 and set(np.unique(w.field.uptake)) <= {0.3, 1.0}
    assert "facts" in m  # counts are filled by whoever polls the sensors (K1), not by run_episode


def test_f2_source_a_needs_channels_else_use_b(crop):
    cfg = oracle_config(facts=_facts(f2=True, f2_source="A", u_core=0.5))
    with pytest.raises(ValueError, match="TC and ET"):
        TumorMRIWorld(crop, cfg, seed=1)
    w = TumorMRIWorld(crop, cfg, seed=1, scorer_channels=_channels(crop))
    ch = _channels(crop)
    assert np.array_equal(w.field.uptake < 1.0, ch["tc"] & ~ch["et"])


def test_determinism_all_facts(crop):
    cfg = oracle_config(facts=_facts(f2=True, f2_source="B", u_core=0.5, f2_share=0.2, f3=True, f3_decay=0.04, f3_share=0.1))
    a = _run(crop, "S", 3, cfg)
    b = _run(crop, "S", 3, cfg)
    assert a[1] == b[1] and np.array_equal(a[0].field.dose, b[0].field.dose)
    s1, s2 = _run(crop, "Sprime", 3, cfg)[1], _run(crop, "Sprime", 3, cfg)[1]
    assert s1 == s2


# --- sensors -----------------------------------------------------------------------------------------------------------


def _release_then_walk(w, i=0, target_vox=None):
    """Squad i releases at its entry voxel at tick 0 and walks away; returns (release voxel, tick-5 position)."""
    name = f"squad-{i}"
    v0 = tuple(w.squad_positions()[i])
    far = target_vox or (18, 16, 17)
    w.apply(name, Intent(ACTION_DECIDE, {"target": "S", "release": 1, "amount": 3, "target_vox": list(far)},
                         meta={"scripted": True}), 0)
    for t in range(5):
        w.step_environment(t)
    return v0, tuple(w.squad_positions()[i])


def test_uptake_reading_is_about_own_last_release_voxel(crop):
    cfg = oracle_config(facts=_facts(f2=True, f2_source="B", u_core=0.3, f2_share=1.0))
    w = TumorMRIWorld(crop, cfg, seed=5)
    w.field.uptake[:] = 0.3  # poor uptake everywhere, so the entry-site release reads slow
    assert w.sensor_readings("squad-0", 0)["uptake"] is None  # nothing released yet
    v0, pos5 = _release_then_walk(w)
    assert pos5 != v0  # the squad moved on
    r = w.sensor_readings("squad-0", 5)["uptake"]
    assert r["pos_vox"] == list(v0) and r["pos_vox"] != list(pos5)
    noise = float(rng("uptake_noise", crop.case_id, 5, 0, 5).normal(0.0, 0.1))
    assert r["reading"] == pytest.approx(0.3 + noise, abs=1e-9) and r["word"] == "slow"
    assert w.sensor_readings("squad-0", 5) is w.sensor_readings("squad-0", 5)  # one reading per squad and tick
    assert w.sensor_readings("squad-0", 10)["uptake"] is None  # one reading per release
    ev = w.pending_sensor_events("squad-0", 5)
    assert [e["kind"] for e in ev] == ["slow_uptake"] and ev[0]["pos_vox"] == list(v0) and ev[0]["ant"] == 0
    assert w.sensor_readings("squad-1", 5)["uptake"] is None  # other squads never read this release


def test_uptake_as_expected_when_u_is_one(crop):
    cfg = oracle_config(facts=_facts(f2=True, f2_source="B", u_core=0.5, f2_share=0.2))
    w = TumorMRIWorld(crop, cfg, seed=5)
    w.field.uptake[:] = 1.0
    _release_then_walk(w)
    r = w.sensor_readings("squad-0", 5)["uptake"]
    assert r["word"] == "as expected" and abs(r["reading"] - 1.0) < 0.5
    assert w.pending_sensor_events("squad-0", 5) == []


def test_uptake_pending_survives_call_order(crop):
    """Reading after the same tick's apply must still be about the older release."""
    cfg = oracle_config(facts=_facts(f2=True, f2_source="B", u_core=0.3, f2_share=1.0))
    w = TumorMRIWorld(crop, cfg, seed=5)
    w.field.uptake[:] = 0.3
    v0, _ = _release_then_walk(w)
    w.apply("squad-0", Intent(ACTION_DECIDE, {"target": "S", "release": 1, "amount": 1, "target_vox": [18, 16, 17]},
                              meta={"scripted": True}), 5)
    r = w.sensor_readings("squad-0", 5)["uptake"]
    assert r["pos_vox"] == list(v0)


def test_washout_reading_only_at_own_voxel_and_with_enough_drug(crop):
    cfg = oracle_config(facts=_facts(f3=True, f3_decay=0.08, f3_share=0.2))
    w = TumorMRIWorld(crop, cfg, seed=6)
    w.field.washout[:] = 0.0
    pos = tuple(w.squad_positions()[0])
    assert w.sensor_readings("squad-0", 0)["washout"] is None  # no drug here: no reading
    w.apply("squad-0", Intent(ACTION_DECIDE, {"target": "S", "release": 1, "amount": 5}, meta={"scripted": True}), 0)
    assert w.field.conc[pos] >= 0.01 * cfg.dose_threshold
    # patch somewhere else only: the ant reads delta at its own voxel (0), not the patch
    other = tuple((np.array(pos) + 6) % 30)
    w.field.washout[other] = 0.08
    r = w.sensor_readings("squad-0", 5)["washout"]
    assert r["pos_vox"] == list(pos) and r["word"] == "normal" and abs(r["reading"]) < 0.05
    # patch at the ant's own voxel: fast, with the seeded noise
    w.field.washout[pos] = 0.08
    r2 = w.sensor_readings("squad-0", 10)["washout"]
    noise = float(rng("washout_noise", crop.case_id, 6, 0, 10).normal(0.0, 0.01))
    assert r2["reading"] == pytest.approx(0.08 + noise, abs=1e-12) and r2["word"] == "fast"
    assert [e["kind"] for e in w.pending_sensor_events("squad-0", 10)] == ["washout"]
    assert w.pending_sensor_events("squad-0", 5) == []
    st = w.sensor_stats()
    assert st["washout_turns"] == 3 and st["washout_available"] == 2 and st["washout_fast"] == 1
    assert st["washout_available_share"] == pytest.approx(2 / 3)


def test_sensors_inert_when_facts_off(crop):
    w = TumorMRIWorld(crop, oracle_config(), seed=1)
    w.apply("squad-0", Intent(ACTION_DECIDE, {"target": "S", "release": 1, "amount": 2}, meta={"scripted": True}), 0)
    assert w.sensor_readings("squad-0", 5) == {"uptake": None, "washout": None}
    assert w.pending_sensor_events("squad-0", 5) == []
    assert w.sensor_stats()["washout_available_share"] is None


# --- oracle S' ---------------------------------------------------------------------------------------------------------


def _seeded_world(crop, **facts):
    w = TumorMRIWorld(crop, oracle_config(facts=_facts(**facts)) if facts else oracle_config(), seed=1)
    w.field.deposit((18, 16, 17), 60.0)
    for _ in range(4):
        w.field.step(w.scorer_truth().truth)
    return w


def test_sprime_projection_uses_u_exactly(crop):
    w = _seeded_world(crop, f2=True, f2_source="B", u_core=0.5, f2_share=0.2)
    o = oracle_for(w)
    w.field.uptake[:] = 1.0
    p1 = o._projected(4) - w.field.dose
    w.field.uptake[:] = 0.5
    p2 = o._projected(4) - w.field.dose
    assert p1.max() > 0 and np.allclose(p2, 0.5 * p1, rtol=1e-4, atol=1e-6)
    # a non-uniform u scales voxel by voxel
    w.field.uptake[:] = 1.0
    w.field.uptake[:18] = 0.2
    p3 = o._projected(4) - w.field.dose
    assert np.allclose(p3[:18], 0.2 * p1[:18], rtol=1e-4, atol=1e-6) and np.allclose(p3[18:], p1[18:], rtol=1e-4, atol=1e-6)


def test_sprime_puff_kernel_scales_with_u_and_release_voxel_decay(crop):
    w = _seeded_world(crop, f3=True, f3_decay=0.08, f3_share=0.2)
    o = oracle_for(w)
    v, n = (18, 16, 17), 60
    w.field.washout[:] = 0.0
    k0 = o._puff_kernel(v, n)
    w.field.washout[v] = 0.08
    k1 = o._puff_kernel(v, n)
    a, b = 0.98, 0.90
    expect = (b * (1 - b ** n) / (1 - b)) / (a * (1 - a ** n) / (1 - a))
    assert np.allclose(k1, k0 * expect, rtol=1e-5) and expect < 1.0
    # documented approximation: for a puff that sits in a uniform patch the scaled kernel is within 10% of the simulation
    w.field.washout[:] = 0.08
    wo = w.field.washout
    sim = DrugField(crop.shape, release_sigma_vox=1.0, diffuse_sigma_vox=1.0, decay=0.02, washout=wo)
    sim.deposit(v, 1.0)
    for _ in range(n):
        sim.step()
    win = o._window(crop.shape, v)[0]
    assert float(o._puff_kernel(v, n).sum()) == pytest.approx(float(sim.dose[win].sum()), rel=0.10)


def test_sprime_runs_with_facts_and_stays_valid(crop):
    cfg = oracle_config(facts=_facts(f2=True, f2_source="B", u_core=0.3, f2_share=0.35,
                                     f3=True, f3_decay=0.04, f3_share=0.2))
    for seed in (1, 2):
        w, m = _run(crop, "Sprime", seed, cfg)
        assert abs(m["mass_residual"]) < 1e-9 and 0.0 <= m["coverage"] <= 1.0


# --- store adapter: scorer-side channels --------------------------------------------------------------------------------


def test_store_adapter_keeps_tc_et_scorer_side(repo):
    case = repo["doc"]["dev_a"][0]
    a, _ = _write_case(repo["root"], case, outside=3)
    lc = load_case(case, root=repo["root"], repo_root=repo["repo"])
    assert np.array_equal(lc.scorer_channels["tc"], a["truth"][1].astype(bool))
    assert np.array_equal(lc.scorer_channels["et"], a["truth"][2].astype(bool))
    assert np.array_equal(lc.crop.truth, a["truth"][0])  # Crop.truth stays WT
    assert not any(hasattr(lc.crop, k) for k in ("tc", "et", "scorer_channels"))
    cfg = WorldConfig(total_drug=300.0, dose_threshold=4.0, entry_min_dist_mm=6.0,
                      facts=_facts(f2=True, u_core=0.5))
    w = TumorMRIWorld(lc.crop, cfg, seed=1, regions=lc.regions, scorer_channels=lc.scorer_channels)
    assert (w.field.uptake < 1.0).sum() == int((a["truth"][1] & ~a["truth"][2]).sum())
    assert set(w.observe("squad-0", 0).data) == set(OBS_KEYS)
