"""v4.31 Phase A: the drug field in real units (config ``physics_model: units``). Additive; the legacy :class:`DrugField` is untouched.

A release is an instantaneous point release of ``mass`` mg at a voxel centre. Its exposure (time integral of concentration) is the
closed form in :mod:`physical`, voxel-averaged and truncated to the time left in the episode, so ``dose`` at any tick is the exact
exposure accumulated so far, in umol*h/L. There is no stepper, no release sigma and no per-tick diffusion, so the stepper's error
(per-tick spread comparable to L) does not exist here.

Free-space kernel. The medium is homogeneous and unbounded: crop faces, the skull, ventricles and CSF are not sinks (Phase B). A cell
that falls outside the crop is simply not stored; the clearance that happens there is reported as ``lost_boundary`` (mg).

Mass ledger (milligrams). For a release of mass M that is T seconds old: still present = M exp(-kT); cleared so far = M (1 - exp(-kT)).
Cleared mass inside a region R is k * (volume of a voxel) * (exposure summed over R), because the loss rate is k times the concentration.
Present mass is the voxel-integrated Gaussian, an exact product of erf differences. Nothing is tuned or fitted.

Entry and motion are unchanged and unphysical: squads still walk through tissue with their whole dose.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from antelligence.worlds.tumor_mri.physical import ExposureTables, PhysicalParams, build_tables, mass_fraction_cube

Vox = Tuple[int, int, int]


class UnitsField:
    """Drop-in for :class:`DrugField` in the parts the world and the oracle use. ``dose`` is exposure in umol*h/L."""

    def __init__(self, shape: Sequence[int], params: PhysicalParams, tables: Optional[ExposureTables] = None, *, n_sub: int = 5) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.params = params
        self.tables = tables if tables is not None else build_tables(params, n_sub=n_sub)
        self.w = self.tables.radius_vox
        self.dose = np.zeros(self.shape, dtype=np.float64)
        self.uptake = None            # F2/F3 are off in Phase A (the loader refuses them with physics_model: units)
        self.washout = None
        self.dose_raw = None
        self._tick = 0
        self._rel: List[Tuple[Vox, float, int]] = []     # (position, mass_mg, release tick)
        self._region: Optional[np.ndarray] = None
        self._conc_cache: Optional[Tuple[int, np.ndarray]] = None
        # mg of cleared mass per unit of table exposure (umol*h/L per mg): k * V / (mg*s/mm^3 -> umol*h/L)
        self._clear_per_table = params.k_clear_s * params.voxel_volume_mm3 / params.exposure_to_umol_h_L

    # --- slicing ---------------------------------------------------------------------------------------------------

    def _slices(self, pos: Vox):
        w = self.w
        lo = [pos[a] - w for a in range(3)]
        hi = [pos[a] + w + 1 for a in range(3)]
        dst = tuple(slice(max(lo[a], 0), min(hi[a], self.shape[a])) for a in range(3))
        src = tuple(slice(dst[a].start - lo[a], dst[a].stop - lo[a]) for a in range(3))
        return dst, src

    # --- the DrugField protocol --------------------------------------------------------------------------------------

    def deposit(self, pos: Vox, mass: float) -> None:
        if mass <= 0.0:
            return
        self._rel.append(((int(pos[0]), int(pos[1]), int(pos[2])), float(mass), self._tick))
        self._conc_cache = None

    def step(self, region: Optional[np.ndarray] = None) -> None:
        """One tick of the clock: add to ``dose`` the exposure each release accrues during this tick."""
        if region is not None:
            self._region = region
        self._tick += 1
        tb = self.tables
        for pos, mass, t0 in self._rel:
            age = self._tick - t0
            if age < 1 or age > tb.n_tab:
                continue
            dst, src = self._slices(pos)
            inc = tb.cum[age][src] - tb.cum[age - 1][src]
            self.dose[dst] += mass * inc
        self._conc_cache = None

    # --- time-dependent counters (mg) --------------------------------------------------------------------------------

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def released(self) -> float:
        return float(sum(m for _, m, _ in self._rel))

    def _age_s(self, t0: int) -> float:
        return (self._tick - t0) * self.params.tick_seconds

    def cleared_total(self) -> float:
        """Cleared so far, anywhere (exact): sum of M (1 - exp(-kT))."""
        k = self.params.k_clear_s
        return float(sum(m * (-math.expm1(-k * self._age_s(t0))) for _, m, t0 in self._rel))

    def present_total(self) -> float:
        k = self.params.k_clear_s
        return float(sum(m * math.exp(-k * self._age_s(t0)) for _, m, t0 in self._rel))

    def _cum_at(self, t0: int) -> np.ndarray:
        return self.tables.cumulative(self._tick - t0)

    def cleared_in(self, mask: Optional[np.ndarray] = None) -> float:
        """Mass cleared so far inside ``mask`` (a boolean array on the crop grid) or, with no mask, anywhere in the crop (mg)."""
        tot = 0.0
        for pos, mass, t0 in self._rel:
            if self._tick - t0 < 1:
                continue
            dst, src = self._slices(pos)
            cube = self._cum_at(t0)[src]
            tot += mass * float((cube if mask is None else cube[mask[dst]]).sum(dtype=np.float64))
        return tot * self._clear_per_table

    def _present_cube(self, t0: int) -> Optional[np.ndarray]:
        k, p = self.params.k_clear_s, self.params
        T = self._age_s(t0)
        surv = math.exp(-k * T)
        if surv < 1e-14:
            return None
        return surv * mass_fraction_cube(self.w, p.voxel_mm, p.d_star_mm2_s, T)

    def present_in(self, mask: Optional[np.ndarray] = None) -> float:
        """Mass still present inside ``mask`` (or anywhere in the crop) (mg), exact separable erf."""
        tot = 0.0
        for pos, mass, t0 in self._rel:
            cube = self._present_cube(t0)
            if cube is None:
                continue
            dst, src = self._slices(pos)
            part = cube[src]
            tot += mass * float((part if mask is None else part[mask[dst]]).sum(dtype=np.float64))
        return tot

    def mass_in(self, region: np.ndarray) -> float:
        return self.present_in(region)

    @property
    def mass(self) -> float:
        return self.present_in(None)

    @property
    def decayed(self) -> float:
        """Cleared inside the crop (mg)."""
        return self.cleared_in(None)

    @property
    def decayed_in_region(self) -> float:
        return self.cleared_in(self._region) if self._region is not None else 0.0

    @property
    def lost_boundary(self) -> float:
        """Everything that is neither still in the crop nor cleared inside it: clearance outside the crop plus drug that diffused out
        and is still present there (mg). Defined so ``released == mass + decayed + lost_boundary`` exactly."""
        return self.released - self.mass - self.decayed

    def on_target_mg(self, mask: np.ndarray) -> float:
        """Mass cleared inside ``mask`` plus mass still present inside it (mg). Off-target mg is ``released - on_target_mg``."""
        return self.cleared_in(mask) + self.present_in(mask)

    def mass_balance_residual(self) -> float:
        """Honest closure test: released minus (present + cleared) summed over the full ``(2W+1)^3`` table cubes, ignoring the crop.
        What is left is the part of the plume beyond the table radius plus the sub-sampling error (mg); it is small by construction
        and never zero by definition."""
        tot = 0.0
        for pos, mass, t0 in self._rel:
            cube = self._cum_at(t0).astype(np.float64)
            pc = self._present_cube(t0)
            tot += mass * (float(cube.sum()) * self._clear_per_table + (0.0 if pc is None else float(pc.sum())))
        return self.released - tot

    @property
    def conc(self) -> np.ndarray:
        """Mass present per voxel (mg), exact. Computed on demand (cached per tick)."""
        if self._conc_cache is not None and self._conc_cache[0] == self._tick:
            return self._conc_cache[1]
        out = np.zeros(self.shape, dtype=np.float64)
        for pos, mass, t0 in self._rel:
            cube = self._present_cube(t0)
            if cube is None:
                continue
            dst, src = self._slices(pos)
            out[dst] += mass * cube[src]
        self._conc_cache = (self._tick, out)
        return out

    def release_voxel_share(self) -> float:
        """Grid honesty: the share of all exposure delivered so far that falls inside the release voxel itself (mass-weighted over releases).
        Above about 0.8 the model is counting single voxels, not fields."""
        c = self.w
        own = sum(m * float(self._cum_at(t0)[c, c, c]) for _, m, t0 in self._rel if self._tick - t0 >= 1) * self._clear_per_table
        tot = self.cleared_total()
        return own / tot if tot > 0 else 0.0

    # --- projection for the oracle -------------------------------------------------------------------------------------

    def projected_final(self, ticks_total: Optional[int] = None) -> np.ndarray:
        """``dose`` plus the exposure the releases already made will still add by the end of the episode (exact)."""
        total = self.params.ticks if ticks_total is None else int(ticks_total)
        proj = self.dose.copy()
        tb = self.tables
        for pos, mass, t0 in self._rel:
            now, end = self._tick - t0, total - t0
            if end <= now:
                continue
            dst, src = self._slices(pos)
            delta = tb.cumulative(end)[src] - tb.cumulative(now)[src]
            proj[dst] += mass * delta
        return proj

    # --- reporting ---------------------------------------------------------------------------------------------------

    def ledger(self, masks: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, object]:
        out: Dict[str, object] = {
            "released_mg": self.released, "present_total_mg": self.present_total(), "cleared_total_mg": self.cleared_total(),
            "present_in_crop_mg": self.mass, "cleared_in_crop_mg": self.decayed, "lost_boundary_mg": self.lost_boundary,
            "closure_residual_mg": self.mass_balance_residual()}
        if masks:
            out["on_target_mg"] = {k: self.on_target_mg(m) for k, m in masks.items()}
        return out
