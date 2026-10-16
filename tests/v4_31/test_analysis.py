"""v4.31: threshold curves, viability weights, the +/-1-voxel band, and the banded legacy stepper. Synthetic phantom only."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from antelligence.experiments import v431_phaseA as PA
from antelligence.worlds.tumor_mri import ScriptS, TumorMRIWorld, WorldConfig, run_episode
from antelligence.worlds.tumor_mri.physics import DrugField
from tests.v4_1.world.phantom import make_phantom, phantom_config


@pytest.fixture(scope="module")
def crop():
    return make_phantom(36)


def _channels(crop):
    return {"tc": ndimage.binary_erosion(crop.truth, iterations=1), "et": ndimage.binary_erosion(crop.truth, iterations=2)}


def test_band_masks_nest_and_differ_by_a_voxel(crop):
    b = PA.band_masks(crop.truth)
    assert not (b["eroded"] & ~b["nominal"]).any() and not (b["nominal"] & ~b["dilated"]).any()
    assert b["eroded"].sum() < b["nominal"].sum() < b["dilated"].sum()
    assert np.array_equal(b["nominal"], crop.truth)
    # a voxel is in the dilated mask iff some voxel of the 3x3x3 block around it is truth
    probe = np.argwhere(b["dilated"] & ~b["nominal"])[0]
    sl = tuple(slice(max(c - 1, 0), c + 2) for c in probe)
    assert crop.truth[sl].any()


def test_erosion_does_not_shave_voxels_at_the_crop_face():
    t = np.zeros((10, 10, 10), bool)
    t[:, 3:7, 3:7] = True                       # a tumour cut by two crop faces
    e = PA.band_masks(t)["eroded"]
    assert e[0, 4, 4] and e[9, 4, 4] and not e[5, 3, 3]


def test_curve_matches_the_scorer_at_a_grid_point(crop):
    cfg = phantom_config(total_drug=400.0, dose_threshold=6.0)
    w = TumorMRIWorld(crop, cfg, seed=1)
    m = run_episode(w, lambda a: ScriptS())
    c = PA.curve(w.field.dose, crop.truth, None, [6.0], crop.outside_crop_truth_voxels)[0]
    assert c == pytest.approx(m["coverage"], abs=1e-12)
    assert c == pytest.approx(w._scorer.coverage(w.field.dose), abs=1e-12)


def test_curve_is_nonincreasing_in_threshold_and_matches_brute_force(crop):
    rng = np.random.default_rng(0)
    dose = rng.lognormal(3.0, 2.0, crop.truth.shape)
    th = list(PA.THRESHOLDS_UMOL_H_L)
    cv = PA.curve(dose, crop.truth, None, th, 0)
    assert np.all(np.diff(cv) <= 1e-15) and cv[0] >= cv[-1]
    t = th[10]
    assert cv[10] == pytest.approx(float((dose[crop.truth] >= t).mean()))
    assert len(th) >= 8 * 4 + 1 and th[-1] / th[0] >= 1e8        # eight decades, well over the two required


def test_viability_weights_follow_the_goal(crop):
    ch = _channels(crop)
    w = PA.voxel_weights(crop.truth, crop.truth, ch["tc"], ch["et"], 0.35)
    assert set(np.unique(w[crop.truth])) <= {1.0, 0.3, 0.35}
    assert (w[ch["et"] & crop.truth] == 1.0).all()
    core = ch["tc"] & ~ch["et"]
    assert (w[core] == 0.3).all()
    assert (w[crop.truth & ~ch["tc"]] == 0.35).all() and (w[~crop.truth] == 0).all()
    w50 = PA.voxel_weights(crop.truth, crop.truth, ch["tc"], ch["et"], 0.5)
    assert (w50[crop.truth & ~ch["tc"]] == 0.5).all()


def test_weighted_curve_differs_from_plain_when_the_core_is_missed(crop):
    ch = _channels(crop)
    dose = np.where(ch["et"], 1.0e6, 0.0)           # only the enhancing core is covered
    th = [10.0]
    plain = PA.curve(dose, crop.truth, None, th, 0)[0]
    wgt = PA.curve(dose, crop.truth, PA.voxel_weights(crop.truth, crop.truth, ch["tc"], ch["et"], 0.35), th, 0)[0]
    assert plain == pytest.approx(ch["et"].sum() / crop.truth.sum())
    assert wgt > plain                                # the enhancing voxels weigh 1.0, the rest less
    full = PA.evaluate_dose(dose, crop.truth, ch, 0, th)
    assert set(full) == set(PA.BANDS) and all({"plain", "w35", "w50", "n_voxels"} <= set(v) for v in full.values())


def test_outside_crop_truth_counts_as_uncovered(crop):
    dose = np.full(crop.truth.shape, 1.0e9)
    assert PA.curve(dose, crop.truth, None, [1.0], 0)[0] == 1.0
    n = int(crop.truth.sum())
    assert PA.curve(dose, crop.truth, None, [1.0], n)[0] == pytest.approx(0.5)


def test_banded_legacy_stepper_changes_nothing_and_adds_the_band_offtarget(crop):
    cfg = phantom_config(total_drug=400.0, dose_threshold=6.0)
    plain = TumorMRIWorld(crop, cfg, seed=2)
    m_plain = run_episode(plain, lambda a: ScriptS())
    banded = TumorMRIWorld(crop, cfg, seed=2)
    bands = {k: v & crop.brain for k, v in PA.band_masks(crop.truth).items()}
    banded.field = PA.BandedDrugField(banded.shape, release_sigma_vox=cfg.release_sigma_mm / crop.voxel_mm,
                                      diffuse_sigma_vox=cfg.diffuse_sigma_vox, decay=cfg.decay, bands=bands)
    m_banded = run_episode(banded, lambda a: ScriptS())
    assert np.array_equal(plain.field.dose, banded.field.dose) and m_plain == m_banded       # identical arithmetic
    assert banded.field.off_target_share("nominal") == pytest.approx(banded.off_target_share(), rel=1e-12)
    ob = PA.off_target_by_band(banded)
    assert ob["dilated"]["off_target_share"] <= ob["nominal"]["off_target_share"] <= ob["eroded"]["off_target_share"]


def test_off_target_by_band_in_mg_for_units(crop):
    w = TumorMRIWorld(crop, WorldConfig.from_dict({"physics_model": "units", "physics": {"k_set": "low", "policy_threshold_umol_h_L": 1000.0}}), seed=1)
    run_episode(w, lambda a: ScriptS())
    ob = PA.off_target_by_band(w)
    assert ob["dilated"]["off_target_mg"] <= ob["nominal"]["off_target_mg"] <= ob["eroded"]["off_target_mg"] <= 61.6
    assert ob["nominal"]["off_target_share"] == pytest.approx(w.off_target_share(), rel=1e-9)
    assert ob["nominal"]["off_target_mg"] + ob["nominal"]["on_target_mg"] == pytest.approx(61.6)
