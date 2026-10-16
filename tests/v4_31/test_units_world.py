"""v4.31: the real-units world (physics_model: units). Synthetic phantom only; no case data.

Entry and motion are unchanged and unphysical: squads still walk through tissue carrying their whole dose. These tests say nothing
about delivery in a patient."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import ndimage

from antelligence.worlds.tumor_mri import ScriptS, TumorMRIWorld, WorldConfig, oracle_for, run_episode
from antelligence.worlds.tumor_mri import physical as P
from antelligence.worlds.tumor_mri.units_field import UnitsField
from tests.v4_1.world.phantom import make_phantom

REPO = Path(__file__).resolve().parents[2]
UNITS_CFG = {"low": REPO / "config/v4_31/world_units_k_low.yaml", "high": REPO / "config/v4_31/world_units_k_high.yaml"}


def ucfg(k: str = "low", thr: float = 1000.0, **kw) -> WorldConfig:
    doc = yaml.safe_load(UNITS_CFG[k].read_text())["world"]
    doc["physics"] = {**doc["physics"], "policy_threshold_umol_h_L": thr}
    doc.update(kw)
    return WorldConfig.from_dict(doc)


@pytest.fixture(scope="module")
def crop():
    return make_phantom(36)


# --- loader ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [{"drug_scale_k": 0.5338512801630682}, {"total_drug": 10.0}, {"drug_mode": "fixed"}, {"release_sigma_mm": 2.0},
                                 {"diffuse_sigma_vox": 1.0}, {"decay": 0.02}, {"dose_threshold": 5.0}])
def test_loader_refuses_legacy_physics_keys_in_a_units_config(bad):
    doc = yaml.safe_load(UNITS_CFG["low"].read_text())["world"]
    doc.update(bad)
    with pytest.raises(ValueError, match="refuses"):
        WorldConfig.from_dict(doc)


def test_units_config_refuses_hidden_facts_and_needs_a_policy_threshold():
    doc = yaml.safe_load(UNITS_CFG["low"].read_text())["world"]
    with pytest.raises(ValueError, match="F1/F2/F3 off"):
        WorldConfig.from_dict({**doc, "facts": {"f3": True, "f3_decay": 0.04, "f3_share": 0.1}})
    with pytest.raises(ValueError, match="policy_threshold"):
        WorldConfig.from_dict({**doc, "physics": {**doc["physics"], "policy_threshold_umol_h_L": 0.0}})
    with pytest.raises(ValueError, match="needs physics_model: units"):
        WorldConfig.from_dict({"physics": {"k_set": "low"}})


@pytest.mark.parametrize("k", ["low", "high"])
def test_derived_values_come_from_the_physical_block(k):
    cfg = ucfg(k)
    p = P.PhysicalParams.carmustine_wafer(k)
    assert cfg.units and cfg.physical_params() == p
    assert (cfg.decay, cfg.diffuse_sigma_vox, cfg.total_drug, cfg.dose_threshold) == (p.decay, p.diffuse_sigma_vox, 61.6, 1000.0)
    assert "drug_scale_k" not in cfg.to_dict() and "decay" not in cfg.to_dict()
    assert WorldConfig.from_dict(cfg.to_dict()) == cfg                    # round trip


def test_budget_is_the_cap_and_is_never_scaled_by_perceived_volume():
    a, b = make_phantom(36), make_phantom(36, seed=11, outside_truth=5)
    b = type(b).from_arrays({"brain": b.brain, "p_tumour": b.p_tumour * 0.5, "unc": b.unc, "truth": b.truth}, outside_crop_truth_voxels=5, case_id="phantom")
    cfg = ucfg("low")
    wa, wb = TumorMRIWorld(a, cfg, seed=1), TumorMRIWorld(b, cfg, seed=1)
    assert wa.total_drug == wb.total_drug == 61.6 == 8 * 7.7
    assert wa._squads[0].payload == pytest.approx(7.7)                     # split equally across the 8 squads
    assert a.p_tumour.sum() != pytest.approx(b.p_tumour.sum())            # the perceived volumes did differ


# --- the field ------------------------------------------------------------------------------------------------------------------


def test_single_release_dose_equals_the_closed_form_sum():
    """A release at the centre of a big grid: dose after n ticks is exactly M x the voxel-averaged finite-horizon kernel."""
    p = P.PhysicalParams.carmustine_wafer("high")
    shape = (41, 41, 41)
    f = UnitsField(shape, p, n_sub=5)
    f.deposit((20, 20, 20), 5.0)
    for _ in range(7):
        f.step()
    T = 7 * p.tick_seconds
    ref = 5.0 * float(P.voxel_average_kernel(np.array([[3, 1, 0]]), 2.0, p.d_star_mm2_s, p.k_clear_s, T, 5)[0]) * p.exposure_to_umol_h_L
    assert f.dose[23, 21, 20] == pytest.approx(ref, rel=1e-5)
    assert f.dose[20, 20, 20] > f.dose[22, 20, 20] > f.dose[24, 20, 20] > 0


def test_episode_truncation_late_release_gets_less_exposure():
    p = P.PhysicalParams.carmustine_wafer("low")
    shape = (41, 41, 41)
    early, late = UnitsField(shape, p, n_sub=5), UnitsField(shape, p, n_sub=5)
    early.deposit((20, 20, 20), 1.0)
    for t in range(p.ticks):
        if t == p.ticks - 3:
            late.deposit((20, 20, 20), 1.0)
        early.step()
        late.step()
    c = late.tables.radius_vox
    assert late.dose[20, 20, 20] == pytest.approx(float(late.tables.cum[3][c, c, c]), rel=1e-6)      # only 3 ticks of exposure
    assert late.dose[23, 20, 20] == pytest.approx(float(late.tables.cum[3][c + 3, c, c]), rel=1e-6)
    # truncation bites with distance: at the release voxel 94% of the exposure is in the first 3 ticks, 6 mm away only 14%
    assert late.dose[20, 20, 20] / early.dose[20, 20, 20] > 0.9
    assert late.dose[23, 20, 20] / early.dose[23, 20, 20] < 0.2


@pytest.mark.parametrize("k", ["low", "high"])
def test_mass_ledger_closes_on_an_episode(crop, k):
    cfg = ucfg(k, 1000.0)
    w = TumorMRIWorld(crop, cfg, seed=2)
    m = run_episode(w, lambda a: ScriptS())
    f = w.field
    assert m["released_total"] == pytest.approx(61.6)
    assert abs(m["mass_residual"]) < 1e-3                                  # mg of 61.6: table tail + quadrature, never defined as zero
    # the exact identity the ledger is built on: released = present + cleared (anywhere), and 'lost' is defined to close the crop's books
    assert f.present_total() + f.cleared_total() == pytest.approx(f.released, rel=1e-12)
    assert f.mass + f.decayed + f.lost_boundary == pytest.approx(f.released, rel=1e-12)
    assert 0.0 <= m["off_target_share"] <= 1.0
    # after 60 h at k >= 1.2e-4 /s nearly everything has been cleared
    assert f.present_total() < 0.01 * f.released
    assert 0.0 < f.release_voxel_share() < 0.8


def test_off_target_in_mg_is_released_minus_on_target(crop):
    w = TumorMRIWorld(crop, ucfg("low"), seed=1)
    run_episode(w, lambda a: ScriptS())
    on = w.field.on_target_mg(w._on_target)
    assert w.off_target_share() == pytest.approx(1.0 - on / w.field.released, rel=1e-9)


@pytest.mark.parametrize("arm", ["S", "Sp"])
def test_units_episode_is_deterministic(crop, arm):
    runs = []
    for _ in range(2):
        w = TumorMRIWorld(crop, ucfg("low", 1000.0), seed=4)
        pol = oracle_for(w) if arm == "Sp" else None
        m = run_episode(w, (lambda a: pol) if pol else (lambda a: ScriptS()))
        runs.append((json.dumps(m, sort_keys=True), hashlib.sha256(w.field.dose.tobytes()).hexdigest()))
    assert runs[0] == runs[1]


def test_units_oracle_plans_on_the_analytic_kernel_and_uses_the_budget_sparingly(crop):
    """S' planned for a threshold covers at least what the script covers at that threshold, on every seed of the phantom."""
    thr = 3000.0
    for seed in (1, 2, 3):
        ws = TumorMRIWorld(crop, ucfg("low", thr), seed=seed)
        ms = run_episode(ws, lambda a: ScriptS())
        wo = TumorMRIWorld(crop, ucfg("low", thr), seed=seed)
        pol = oracle_for(wo)
        mo = run_episode(wo, lambda a: pol)
        assert type(pol).__name__ == "UnitsOracle"
        assert mo["coverage"] >= ms["coverage"], (seed, ms["coverage"], mo["coverage"])
        assert mo["released_total"] <= 61.6 + 1e-9


def test_projection_is_the_dose_the_run_ends_with(crop):
    """projected_final() mid-episode, with no further releases, equals the final dose exactly: the oracle plans on the true future."""
    p = P.PhysicalParams.carmustine_wafer("high")
    f = UnitsField((41, 41, 41), p, n_sub=5)
    f.deposit((20, 20, 20), 2.0)
    for _ in range(4):
        f.step()
    f.deposit((10, 25, 22), 1.5)
    proj = f.projected_final()
    for _ in range(p.ticks - 4):
        f.step()
    assert np.allclose(proj, f.dose, rtol=1e-4, atol=1e-6)


# --- Step E: nothing legacy moved -----------------------------------------------------------------------------------------------


def _dg(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def test_legacy_configs_and_phantom_episodes_are_byte_identical_to_the_recorded_base():
    """Pins the v4.1, v4.2, v4.3 (and gate A) world configs and a phantom S and S' episode to hashes taken on the UNTOUCHED base a81bffe."""
    pins = json.loads((Path(__file__).parent / "fixtures" / "legacy_pin.json").read_text())["pins"]
    crop = make_phantom(36)
    ch = {"tc": ndimage.binary_erosion(crop.truth, iterations=1), "et": ndimage.binary_erosion(crop.truth, iterations=2)}
    for rel, want in pins.items():
        cfg = WorldConfig.from_yaml(REPO / rel)
        assert _dg(cfg.to_dict()) == want["cfg_dict_sha"], rel
        assert not cfg.units and "physics_model" not in cfg.to_dict()
        for arm in ("S", "Sp"):
            w = TumorMRIWorld(crop, cfg.with_(ticks=60), seed=3, scorer_channels=ch)
            pol = oracle_for(w) if arm == "Sp" else None
            m = run_episode(w, (lambda a: pol) if pol else (lambda a: ScriptS()))
            assert _dg(m) == want[arm]["metrics_sha"], (rel, arm)
            assert hashlib.sha256(np.ascontiguousarray(w.field.dose).tobytes()).hexdigest() == want[arm]["dose_sha"], (rel, arm)


def test_v431_legacy_world_is_the_v43_world_with_facts_off():
    v43 = yaml.safe_load((REPO / "config/v4_3/world.yaml").read_text())["world"]
    v431 = yaml.safe_load((REPO / "config/v4_31/world.yaml").read_text())["world"]
    assert v431.pop("physics_model") == "legacy"
    facts43, facts431 = v43.pop("facts"), v431.pop("facts")
    assert v43 == v431                                               # every world constant is the frozen v4.3 one
    assert not any(facts431[k] for k in ("f1_simlook", "f2", "f3"))
    assert facts43["f1_simlook"] and facts43["f2"] and facts43["f3"]  # v4.3 had them on; Phase A turns them off
    cfg = WorldConfig.from_yaml(REPO / "config/v4_31/world.yaml")
    assert not cfg.units and not cfg.facts.any_on


@pytest.mark.parametrize("k", ["low", "high"])
def test_oracle_planning_kernels_are_nonnegative_for_every_horizon(k):
    """The oracle indexes the table by ticks left; every horizon's planning window must be >= 0 (see the negative-roundoff regression)."""
    from antelligence.worlds.tumor_mri.units_oracle import UnitsOracle
    crop = make_phantom(36)
    w = TumorMRIWorld(crop, ucfg(k, 100.0), seed=1)
    o = oracle_for(w)
    assert isinstance(o, UnitsOracle)
    for n in range(1, w.config.ticks + 1):
        assert float(o._window_kernel(n).min()) >= 0.0
    assert o._window_kernel(0) is None


def test_oracle_survives_the_last_chance_decision_with_nothing_left_to_gain(crop):
    """Planned for a threshold the budget has already met everywhere, the oracle must still decide at every tick through the last one."""
    for k in ("low", "high"):
        w = TumorMRIWorld(crop, ucfg(k, 1.0), seed=2)
        pol = oracle_for(w)
        m = run_episode(w, lambda a: pol)
        assert m["ticks"] == 120 and 0.0 <= m["coverage"] <= 1.0
