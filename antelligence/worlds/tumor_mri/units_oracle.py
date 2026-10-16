"""v4.31 Phase A: oracle S' on the real-units exposure kernel (config ``physics_model: units``). Additive; the legacy planner is untouched.

It is the SAME planner as :class:`OracleSPrime` (target set, claimed points, amount levels, hill-climb, A* arrival). Only the physics
it plans with changes: the unit-puff response is the voxel-averaged analytic exposure kernel for the time left (umol*h/L per mg), and the
projected final dose is the live dose plus the exact future exposure of the releases already made (``UnitsField.projected_final``).
The kernel is free-space (no crop-face loss), as the field is.

Entry and motion are unchanged and unphysical: squads still walk through tissue with their whole dose.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from antelligence.worlds.tumor_mri.config import WorldConfig
from antelligence.worlds.tumor_mri.policies import OracleSPrime, _WIN_R
from antelligence.worlds.tumor_mri.physical import ExposureTables
from antelligence.worlds.tumor_mri.units_field import UnitsField, Vox


def _centre_window(cube: np.ndarray, win_r: int) -> np.ndarray:
    """A ``(2*win_r+1)^3`` float32 window around the cube centre (zero padded when the cube is smaller)."""
    w = (cube.shape[0] - 1) // 2
    out = np.zeros((2 * win_r + 1,) * 3, dtype=np.float32)
    h = min(w, win_r)
    out[win_r - h:win_r + h + 1, win_r - h:win_r + h + 1, win_r - h:win_r + h + 1] = cube[w - h:w + h + 1, w - h:w + h + 1, w - h:w + h + 1]
    return out


class UnitsOracle(OracleSPrime):
    def _init_kernels(self, config: WorldConfig, voxel_mm: float) -> None:
        if not isinstance(self._dose, UnitsField):
            raise TypeError("UnitsOracle needs a UnitsField")
        self._tables: ExposureTables = self._dose.tables
        self._win_cache: Dict[int, np.ndarray] = {}
        self._kern = {}      # unused: the legacy free-space kernels
        self._puff = None

    def _window_kernel(self, n: int) -> Optional[np.ndarray]:
        """Exposure (umol*h/L per mg) in the planning window for a release ``n`` ticks before the end (n >= 1)."""
        if n < 1:
            return None
        n = min(int(n), self.cfg.ticks)
        hit = self._win_cache.get(n)
        if hit is None:
            hit = _centre_window(self._tables.cumulative(n), _WIN_R)
            self._win_cache[n] = hit
        return hit

    def _kernel(self, n: int) -> Optional[np.ndarray]:
        return self._window_kernel(n)

    def _puff_kernel(self, v: Vox, n: int) -> Optional[np.ndarray]:
        return self._window_kernel(n)

    def _projected(self, tick: int) -> np.ndarray:
        return self._dose.projected_final(self.cfg.ticks)
