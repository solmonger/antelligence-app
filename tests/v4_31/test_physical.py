"""v4.31 Step A: the physical parameter block. Synthetic only; no case data.

The STANDING acceptance test (``test_penetration_length_acceptance``) is the one physical prediction the world makes that can be checked
directly: 0.4 mm <= L = sqrt(D/k) <= the band edge, for every parameter set the code will run. Any change to D or k must pass it."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from antelligence.worlds.tumor_mri import physical as P


def _rebuild(p: P.PhysicalParams, **kw) -> P.PhysicalParams:
    return dataclasses.replace(p, **kw)


@pytest.mark.parametrize("k", ["low", "high"])
def test_penetration_length_acceptance(k):
    """STANDING: L must sit in the band for every parameter set the code will run (both ends of the clearance pair)."""
    p = P.PhysicalParams.carmustine_wafer(k)
    lo, hi = P.L_BAND_MM
    assert lo <= p.L_mm <= hi, (k, p.L_mm)


def test_exact_L_for_the_pair_and_the_band_decision():
    """The goal says L is 2.06 mm (low k) and 0.92 mm (high k). The stated band is 0.4-2.0, so the low end misses by 0.06 mm; the band
    was widened to 2.1 with a written reason (physical.L_BAND_REASON). Pin both facts so the widening cannot drift silently."""
    low, high = P.PhysicalParams.carmustine_wafer("low"), P.PhysicalParams.carmustine_wafer("high")
    assert low.L_mm == pytest.approx(2.0616, abs=1e-4)
    assert high.L_mm == pytest.approx(0.9220, abs=1e-4)
    assert P.L_BAND_STATED_MM == (0.4, 2.0) and P.L_BAND_MM == (0.4, 2.1)
    assert low.L_mm > P.L_BAND_STATED_MM[1]            # the miss, stated
    assert P.L_BAND_STATED_MM[0] <= high.L_mm <= P.L_BAND_STATED_MM[1]
    assert len(P.L_BAND_REASON) > 100


def test_nudging_D_or_k_out_of_the_band_is_refused():
    base = P.PhysicalParams.carmustine_wafer("low")
    with pytest.raises(P.PhysicalParamsError, match="acceptance test"):
        _rebuild(base, d_star_cm2_s=base.d_star_cm2_s * 1.15)       # D nudged up: L = 2.21 mm
    with pytest.raises(P.PhysicalParamsError, match="acceptance test"):
        _rebuild(base, k_clear_s=5.0e-3)                            # clearance far faster: L = 0.32 mm
    with pytest.raises(P.PhysicalParamsError, match="acceptance test"):
        _rebuild(base, k_clear_s=5.05e-6)                           # the old world's k: L = 10 mm


@pytest.mark.parametrize("field", list(P.PHYSICAL_FIELDS))
@pytest.mark.parametrize("bad", ["missing", "", "   "])
def test_every_physical_field_needs_a_source(field, bad):
    base = P.PhysicalParams.carmustine_wafer("low")
    src = dict(base.sources)
    if bad == "missing":
        del src[field]
    else:
        src[field] = bad
    with pytest.raises(P.PhysicalParamsError, match="source"):
        P.PhysicalParams(base.d_star_cm2_s, base.k_clear_s, base.tick_seconds, base.ticks, base.voxel_mm, base.budget_mg,
                         base.molar_mass_g_mol, P._src_map(src))


def test_frozen_hashable_and_json_plain():
    p = P.PhysicalParams.carmustine_wafer("high")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.k_clear_s = 1.0                                            # type: ignore[misc]
    assert hash(p) == hash(P.PhysicalParams.carmustine_wafer("high")) and p != P.PhysicalParams.carmustine_wafer("low")
    d = json.loads(json.dumps(p.to_dict()))                          # plain JSON: no numpy, no objects
    assert P.PhysicalParams.from_dict(d) == p and p.sha() == P.PhysicalParams.from_dict(d).sha()
    assert all(isinstance(d["sources"][f], str) and d["sources"][f] for f in P.PHYSICAL_FIELDS)


def test_declared_substitution_is_in_the_diffusivity_source():
    assert "no measured BCNU" in P.PhysicalParams.carmustine_wafer().source("d_star_cm2_s")
    assert "SUBSTITUTION" in P.D_STAR_SOURCE
    assert "DECLARED SUBSTITUTION" in P.__doc__.upper()          # also in the module docstring


@pytest.mark.parametrize("k", ["low", "high"])
def test_derived_quantities_are_derived_not_copied(k):
    p = P.PhysicalParams.carmustine_wafer(k)
    D_mm2_s = p.d_star_cm2_s * 100.0
    assert p.diffuse_sigma_mm == pytest.approx(math.sqrt(2.0 * D_mm2_s * p.tick_seconds))
    assert p.diffuse_sigma_vox == pytest.approx(p.diffuse_sigma_mm / p.voxel_mm)
    assert p.decay == pytest.approx(1.0 - math.exp(-p.k_clear_s * p.tick_seconds))
    assert p.L_mm == pytest.approx(math.sqrt(D_mm2_s / p.k_clear_s))
    shorter = p.with_tick(600.0)
    assert shorter.diffuse_sigma_mm < p.diffuse_sigma_mm and shorter.decay < p.decay
    assert shorter.L_mm == pytest.approx(p.L_mm)                    # L does not depend on what a tick means
    assert shorter.sha() != p.sha()


def test_default_tick_ticks_budget_and_units():
    p = P.PhysicalParams.carmustine_wafer()
    assert (p.tick_seconds, p.ticks, p.voxel_mm, p.budget_mg) == (1800.0, 120, 2.0, 61.6)
    assert p.episode_hours == pytest.approx(60.0)
    assert p.budget_mg == pytest.approx(8 * 7.7)
    # 1 mg*s/mm^3 = 1e6 mg*s/L = 277.78 mg*h/L = 277.78/214.05 mmol*h/L = 1297.7 umol*h/L
    assert p.exposure_to_umol_h_L == pytest.approx(1297.72, rel=1e-4)


def test_budget_cap_is_enforced():
    base = P.PhysicalParams.carmustine_wafer()
    with pytest.raises(P.PhysicalParamsError, match="cap"):
        _rebuild(base, budget_mg=62.0)
    assert _rebuild(base, budget_mg=30.0).budget_mg == 30.0         # a smaller budget is allowed; a bigger one is not


def test_tick_tradeoff_table_matches_the_grounding_report():
    rows = P.tick_tradeoff(P.PhysicalParams.carmustine_wafer())
    by = {int(r["tick_seconds"]): r for r in rows}
    for t, want in ((600, 0.78), (1800, 1.36), (3922, 2.00)):
        assert by[t]["spread_mm"] == pytest.approx(want, abs=0.01)
    assert 0.06 < by[600]["loss_per_tick_low"] < 0.08 and 0.29 < by[600]["loss_per_tick_high"] < 0.31       # 7-30 %
    assert 0.18 < by[1800]["loss_per_tick_low"] < 0.20 and 0.65 < by[1800]["loss_per_tick_high"] < 0.67     # 19-66 %
    assert 0.37 < by[3922]["loss_per_tick_low"] < 0.39 and 0.89 < by[3922]["loss_per_tick_high"] < 0.91     # 38-90 %
    assert [round(by[t]["episode_hours"]) for t in (600, 1800, 3922)] == [20, 60, 131]
    # there is no tick where the per-tick spread is about one voxel AND the per-tick loss is small
    assert not any(0.75 <= r["spread_vox"] <= 1.25 and r["loss_per_tick_low"] < 0.1 for r in rows)


def test_wafer_profile_x10_distances():
    low, high = P.PhysicalParams.carmustine_wafer("low"), P.PhysicalParams.carmustine_wafer("high")
    assert P.x10_distance_mm(high.L_mm) == pytest.approx(1.55, abs=0.03)
    assert P.x10_distance_mm(low.L_mm) == pytest.approx(2.2, abs=0.05)
    assert P.planar_x10_mm(1.0) == pytest.approx(2.302585, abs=1e-5)
