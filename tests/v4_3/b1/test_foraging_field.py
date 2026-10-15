"""B1 section 1: the foraging uptake field (formula, clipping, determinism, realised means, config)."""

import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from antelligence.worlds.tumor_mri import WorldConfig
from antelligence.worlds.tumor_mri.config import FactsConfig
from antelligence.worlds.tumor_mri.foraging import (UPTAKE_LEVELS, ForagingError, foraging_uptake, realised_means,
                                                    subregion_masks)
from antelligence.worlds.tumor_mri.physics import rng


def tiny():
    """A 12^3 volume: WT = cube [2:10], TC = [3:9], ET = [5:9] of the x axis inside TC (so core, rim and edema all exist)."""
    wt = np.zeros((12, 12, 12), bool)
    wt[2:10, 2:10, 2:10] = True
    tc = np.zeros_like(wt)
    tc[3:9, 3:9, 3:9] = True
    et = np.zeros_like(wt)
    et[5:9, 3:9, 3:9] = True
    return wt, tc, et


def sphere_volume(n=96, r_wt=36, r_tc=26, r_et_in=17):
    g = np.indices((n, n, n)).transpose(1, 2, 3, 0) - n / 2
    r = np.sqrt((g ** 2).sum(-1))
    wt, tc = r < r_wt, r < r_tc
    return wt, tc, tc & (r >= r_et_in)


def z_of(shape, case_id):
    """The documented recipe, written out independently of the module."""
    z = gaussian_filter(rng("f2_uptake", case_id).standard_normal(shape), 2.5, mode="reflect")
    return z / z.std()


# --- the formula -------------------------------------------------------------------------------------------------------

def test_base_values_clip_bounds_and_one_outside_wt(monkeypatch):
    wt, tc, et = tiny()
    monkeypatch.setitem(UPTAKE_LEVELS, "flat", {"rim": 1.0, "core": 0.4, "edema": 0.8, "sigma": 0.0})   # sigma 0: u == base
    u = foraging_uptake(wt, tc, et, "case-x", "flat")
    m = subregion_masks(wt, tc, et)
    assert m["rim"].sum() and m["core"].sum() and m["edema"].sum()
    assert np.all(u[m["rim"]] == 1.0) and np.all(u[m["core"]] == 0.4) and np.all(u[m["edema"]] == 0.8)
    assert np.all(u[~wt] == 1.0)                                    # outside the whole tumour
    assert u.dtype == np.float64 and u.shape == wt.shape
    monkeypatch.setitem(UPTAKE_LEVELS, "flat", {"rim": 0.01, "core": 0.01, "edema": 1.0, "sigma": 0.0})
    u2 = foraging_uptake(wt, tc, et, "case-x", "flat")
    assert np.all(u2[m["rim"] | m["core"]] == 0.05)                 # floor
    assert np.all(u2[m["edema"]] == 1.0)


def test_masks_follow_the_brief_and_are_disjoint():
    wt, tc, et = tiny()
    m = subregion_masks(wt, tc, et)
    assert np.array_equal(m["rim"], et & wt) and np.array_equal(m["core"], tc & ~et & wt) and np.array_equal(m["edema"], wt & ~tc)
    assert not (m["rim"] & m["core"]).any() and not (m["rim"] & m["edema"]).any() and not (m["core"] & m["edema"]).any()
    assert np.array_equal(m["rim"] | m["core"] | m["edema"], wt)
    # ET outside TC is not a thing in the expert labels; if it appears the voxel counts once, as rim
    et2 = et.copy()
    et2[2, 2, 2] = True
    m2 = subregion_masks(wt, tc, et2)
    assert m2["rim"][2, 2, 2] and not m2["edema"][2, 2, 2]


@pytest.mark.parametrize("level", ["low", "mid", "high"])
def test_formula_on_the_noise_recipe(level):
    wt, tc, et = tiny()
    u = foraging_uptake(wt, tc, et, "case-y", level)
    p = UPTAKE_LEVELS[level]
    base = np.zeros(wt.shape)
    m = subregion_masks(wt, tc, et)
    for k in ("rim", "core", "edema"):
        base[m[k]] = p[k]
    want = np.clip(base * np.exp(p["sigma"] * z_of(wt.shape, "case-y") - p["sigma"] ** 2 / 2), 0.05, 1.0)
    assert np.allclose(u[wt], want[wt], rtol=0, atol=1e-12) and np.all(u[~wt] == 1.0)
    assert u[wt].min() >= 0.05 and u.max() <= 1.0


def test_presets_are_the_named_assumptions():
    assert UPTAKE_LEVELS == {"low": {"rim": 1.0, "core": 0.6, "edema": 0.9, "sigma": 0.2},
                             "mid": {"rim": 1.0, "core": 0.4, "edema": 0.8, "sigma": 0.4},
                             "high": {"rim": 1.0, "core": 0.2, "edema": 0.7, "sigma": 0.6}}


def test_z_is_divided_by_the_crop_sd_without_mean_subtraction():
    z = z_of((20, 20, 20), "c")
    assert z.std() == pytest.approx(1.0)
    wt = np.ones((20, 20, 20), bool)
    tc = wt.copy()
    et = np.zeros_like(wt)               # all core: base 0.4 at mid
    u = foraging_uptake(wt, tc, et, "c", "mid")
    want = np.clip(0.4 * np.exp(0.4 * z - 0.08), 0.05, 1.0)
    assert np.allclose(u, want, atol=1e-12)
    raw = gaussian_filter(rng("f2_uptake", "c").standard_normal((20, 20, 20)), 2.5, mode="reflect")
    assert not np.allclose(raw / raw.std() - (raw / raw.std()).mean(), z)    # the mean was NOT removed


def test_needs_tc_and_et_channels():
    wt, tc, et = tiny()
    for bad in ((None, et), (tc, None), (None, None)):
        with pytest.raises(ForagingError, match="TC and ET"):
            foraging_uptake(wt, bad[0], bad[1], "c", "mid")
    with pytest.raises(ForagingError, match="level"):
        foraging_uptake(wt, tc, et, "c", "extreme")
    with pytest.raises(ForagingError, match="shapes"):
        foraging_uptake(wt, tc[:6], et, "c", "mid")


# --- determinism --------------------------------------------------------------------------------------------------------

def test_determinism_and_case_dependence():
    wt, tc, et = tiny()
    a = foraging_uptake(wt, tc, et, "case-1", "mid")
    assert np.array_equal(a, foraging_uptake(wt, tc, et, "case-1", "mid"))
    assert not np.array_equal(a, foraging_uptake(wt, tc, et, "case-2", "mid"))
    assert not np.array_equal(a, foraging_uptake(wt, tc, et, "case-1", "high"))
    # the noise stream is the documented one: the same case gives the same z at every level
    z = z_of(wt.shape, "case-1")
    hi = foraging_uptake(wt, tc, et, "case-1", "high")
    m = subregion_masks(wt, tc, et)
    rim = m["rim"]
    assert np.allclose(hi[rim], np.clip(np.exp(0.6 * z[rim] - 0.18), 0.05, 1.0))


# --- realised means -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("level,sigma", [("low", 0.2), ("mid", 0.4), ("high", 0.6)])
def test_realised_means_on_a_large_volume(level, sigma):
    wt, tc, et = sphere_volume()
    u = foraging_uptake(wt, tc, et, "big", level)
    rm = realised_means(u, wt, tc, et)
    base = UPTAKE_LEVELS[level]
    # rim: base 1, so clipping at 1 alone lowers the mean to 2 * (1 - Phi(sigma / 2)): 0.920, 0.841, 0.764 for z ~ N(0, 1).
    # The crop-wide SD includes the reflect-mode edges (more variance), so the realised rim is a little higher; tolerance 0.03.
    theory = 2.0 * 0.5 * math.erfc((sigma / 2) / math.sqrt(2.0))
    assert rm["rim"] == pytest.approx({"low": 0.92, "mid": 0.84, "high": 0.76}[level], abs=0.03)
    assert rm["rim"] == pytest.approx(theory, abs=0.03)
    # edema: clipping at 1 bites (base 0.7 to 0.9), so its realised mean is clearly below base. Core: base 0.2 to 0.6 is rarely
    # clipped, so the exact expectation is below base by less than 0.001; the realised mean differs from base by sampling noise only.
    for part in ("core", "edema"):
        want = clipped_mean(base[part], sigma)
        assert want < base[part]
        assert rm[part] == pytest.approx(want, abs=0.04)
    assert rm["edema"] < base["edema"] - 0.02
    assert rm["core"] < base["core"] + 0.02
    assert rm["rim"] < base["rim"]
    assert rm["n_rim"] == int((et & wt).sum()) and rm["n_wt"] == int(wt.sum())
    assert rm["n_rim"] + rm["n_core"] + rm["n_edema"] == rm["n_wt"]
    assert U_OK(u, wt)
    assert 0.0 <= rm["clipped_low"] <= 1.0 and 0.0 < rm["clipped_high"] < 1.0


def clipped_mean(b, sigma):
    """E[min(1, b * exp(sigma Z - sigma^2/2))] for Z ~ N(0, 1)."""
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    zs = (sigma * sigma / 2 - math.log(b)) / sigma if b < 1.0 else sigma / 2
    return b * phi(zs - sigma) + 1.0 - phi(zs)


def U_OK(u, wt):
    return u[wt].min() >= 0.05 and u[wt].max() <= 1.0 and np.all(u[~wt] == 1.0)


def test_realised_means_empty_subregion_is_none():
    wt, tc, et = tiny()
    rm = realised_means(np.ones(wt.shape), wt, np.zeros_like(wt), np.zeros_like(wt))   # no TC, no ET: all edema
    assert rm["rim"] is None and rm["core"] is None and rm["edema"] == 1.0 and rm["n_edema"] == int(wt.sum())


# --- config -------------------------------------------------------------------------------------------------------------

def test_facts_config_validates_and_omits_defaults():
    with pytest.raises(ValueError, match="f2_model"):
        FactsConfig(f2_model="bogus")
    with pytest.raises(ValueError, match="uptake_level"):
        FactsConfig(uptake_level="extreme")
    v42 = FactsConfig(f1_simlook=True, f2=True, f2_source="A", u_core=0.5, f3=True, f3_decay=0.04, f3_share=0.1)
    assert "f2_model" not in v42.to_dict() and "uptake_level" not in v42.to_dict() and not v42.foraging
    assert "f2_model" not in WorldConfig(facts=v42).to_dict()["facts"]
    fo = FactsConfig(f2=True, f2_model="foraging", uptake_level="mid")
    assert fo.foraging and fo.to_dict()["f2_model"] == "foraging" and fo.to_dict()["uptake_level"] == "mid"
    assert not FactsConfig(f2=False, f2_model="foraging").foraging          # F2 off: the model flag does nothing
    wc = WorldConfig(facts=fo)
    assert WorldConfig.from_dict(wc.to_dict()) == wc
    high_v42 = FactsConfig(f2=True, uptake_level="high")                     # a non-default level is stated even in v42 mode
    assert high_v42.to_dict()["uptake_level"] == "high" and "f2_model" not in high_v42.to_dict()
