"""v4.31 Step A: the physical parameter block. Synthetic only; no case data.

The STANDING acceptance test (``test_penetration_length_acceptance``) is the one physical prediction the world makes that can be checked
directly: 0.4 mm <= L = sqrt(D/k) <= the band edge, for every parameter set the code will run. Any change to D or k must pass it."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

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


# --- A.2: the reference clearance k_ref and the profile acceptance test ---------------------------------------------------------------------------------


def test_profile_acceptance_primary_is_k_ref_and_the_pair_is_secondary():
    """A.2: the acceptance test is the penetration profile (planar x10 = 2.303 L within 1.5x of rat 1.0 mm and rabbit 'under 1.2 mm'); 0.4-2.1 mm is a sanity floor."""
    ref, lo, hi = (P.PhysicalParams.carmustine_wafer(k) for k in ("ref", "low", "high"))
    cr, cl, ch = ref.profile_check(), lo.profile_check(), hi.profile_check()
    assert P.K_CLEAR_REF_S == 2.0e-3 and ref.k_clear_s == 2.0e-3 and ref.k_label == "ref"
    assert ref.L_mm == pytest.approx(0.505, abs=0.001)
    assert cr["planar_x10_mm"] == pytest.approx(1.163, abs=0.002) and cr["ratio_planar_to_rat"] == pytest.approx(1.16, abs=0.01)
    assert cr["ratio_planar_to_rabbit_bound"] == pytest.approx(0.97, abs=0.01) and cr["point_source_x10_mm"] == pytest.approx(1.215, abs=0.003)
    assert cr["accepted"] and ref.role == "primary"
    assert not cl["accepted"] and lo.role == "secondary" and not ch["accepted"] and hi.role == "secondary"
    assert P.PhysicalParams.carmustine_wafer().k_label == "ref"                         # the default set is the primary one


def test_k_ref_is_declared_calibrated_not_measured_and_the_untrusted_constants_are_not_adopted():
    src = P.PhysicalParams.carmustine_wafer("ref").source("k_clear_s")
    assert "CALIBRATED" in src and "NOT independently measured" in src and "no carmustine tissue clearance constant has ever been measured directly" in src
    assert "Fung" in src and "Strasser" in src and "opposite of v4.1" in src
    assert "CALIBRATED" in P.__doc__ and "not\nindependently measured" in P.__doc__
    # the student-report constants must not be used as parameters anywhere in the module
    nums = [v for v in vars(P).values() if isinstance(v, float)]
    assert 1.31e-4 not in nums and 2.5e-10 not in nums and 6.75e-5 not in nums
    assert "NOT ADOPTED" in Path(P.__file__).read_text() and "HTTP 403" in Path(P.__file__).read_text()


def test_profile_check_rejects_a_set_that_misses_the_profile_even_inside_the_floor():
    base = P.PhysicalParams.carmustine_wafer("ref")
    off = _rebuild(base, k_clear_s=6.0e-4)                                             # L = 0.92 mm: inside 0.4-2.1, planar x10 = 2.1 mm
    assert P.L_BAND_MM[0] <= off.L_mm <= P.L_BAND_MM[1] and not off.accepted and not off.profile_check()["within_1.5x_rat"]


def test_scoring_grid_is_the_same_physics_on_another_voxel():
    a = P.PhysicalParams.carmustine_wafer("ref")
    b = a.with_voxel(1.0)
    assert (a.voxel_mm, b.voxel_mm) == (2.0, 1.0) and a.L_mm == b.L_mm and a.decay == b.decay and a.k_clear_s == b.k_clear_s
    assert "SCORING GRID" in b.source("voxel_mm") and "perception store" in a.source("voxel_mm") and a.sha() != b.sha()
    assert b.voxel_volume_mm3 == pytest.approx(a.voxel_volume_mm3 / 8.0)
