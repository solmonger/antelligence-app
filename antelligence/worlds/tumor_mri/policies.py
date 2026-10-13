"""No-model arms: script S (perception only) and oracle S' (sees truth), plus the S' -> candidate-label mapping.

Both implement the kernel ``Policy`` protocol and emit ``decide`` intents with the same typed fields a model arm
answers (target label, release 0/1, amount 1..5). S' additionally passes ``target_vox`` (its true goal), because a
label can only name a perceived region; the label is the router-training target (see :func:`sprime_candidate_label`).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from antelligence.kernel.types import Intent, LocalView
from antelligence.worlds.tumor_mri.data import Candidate
from antelligence.worlds.tumor_mri.questions import EXPLORE, STAY
from antelligence.worlds.tumor_mri.world import ACTION_DECIDE, Scorer

MAP_RADIUS_MM = 10.0


def sprime_candidate_label(target_vox: Sequence[float], candidates: Sequence[Mapping[str, Any]] | Sequence[Candidate],
                           voxel_mm: float = 2.0, radius_mm: float = MAP_RADIUS_MM) -> str:
    """Nearest candidate centroid within ``radius_mm`` of S' target, else ``X``. Ties go to the earlier label."""
    best_label, best_d = EXPLORE, radius_mm
    for c in candidates:
        label, centroid = (c.label, c.centroid_vox) if isinstance(c, Candidate) else (c["label"], c["centroid_vox"])
        d = voxel_mm * math.dist(target_vox, centroid)
        if d <= best_d and (best_label == EXPLORE or d < best_d):
            best_label, best_d = label, d
    return best_label


class ScriptS:
    """Perception-only script. Treat where the scan says tumour; otherwise walk to the nearest believed region.

    * release iff the local scan says tumour (p >= ``release_p``), is fairly certain (unc <= ``unc_max``), the local
      dose is below the threshold and the squad has payload; amount level ``amount``;
    * target ``S`` while the local spot is still under-dosed and tumour-like, else the nearest candidate with
      p >= ``target_p`` (skipping one the squad is already on and has saturated), else ``X``.
    """

    def __init__(self, *, release_p: float = 0.5, unc_max: float = 0.5, target_p: float = 0.5, amount: int = 3,
                 saturate_mm: float = 4.0) -> None:
        self.release_p, self.unc_max, self.target_p = release_p, unc_max, target_p
        self.amount, self.saturate_mm = amount, saturate_mm

    def decide(self, view: LocalView) -> Intent:
        o = view.observation
        p, unc, rel_dose = o["local_p_tumour"], o["local_unc"], o["local_dose_rel"]
        if o.get("local_look") is not None:
            p = 0.5 * (p + o["local_look"])
        tumour_here = p >= self.release_p and unc <= self.unc_max
        under = rel_dose < 1.0
        release = int(tumour_here and under and o["payload"] > 0.0)
        if tumour_here and under:
            target = STAY
        else:
            target = EXPLORE
            cands = [c for c in o["candidates"] if c["p_tumour"] >= self.target_p
                     and not (c["dist_mm"] < self.saturate_mm and not under)]
            if cands:
                target = min(cands, key=lambda c: (c["dist_mm"], c["label"]))["label"]
        return Intent(ACTION_DECIDE, {"target": target, "release": release, "amount": self.amount})

    def describe(self) -> Dict[str, Any]:
        return {"policy": "script_S", "release_p": self.release_p, "unc_max": self.unc_max,
                "target_p": self.target_p, "amount": self.amount, "saturate_mm": self.saturate_mm}


class OracleSPrime:
    """Oracle: knows the truth mask and the live dose. Steers each squad to a under-dosed tumour voxel in its
    own Voronoi cell, preferring dense clusters (so squads spread and work cores, not stragglers), and releases when it
    stands on under-dosed tumour with enough under-dosed tumour around it.

    Built with the world's scorer accessor and a ``positions`` callable (all squad positions). Not a fair arm; it
    sets the achievable ceiling for tuning and supplies router labels.
    """

    def __init__(self, scorer: Scorer, positions, dose_field, *, voxel_mm: float = 2.0, amount: int = 3,
                 map_radius_mm: float = MAP_RADIUS_MM, density_sigma_vox: float = 2.0,
                 release_density: float = 0.15, distance_penalty: float = 0.004) -> None:
        self._scorer, self._positions, self._dose = scorer, positions, dose_field
        self.voxel_mm, self.amount, self.map_radius_mm = voxel_mm, amount, map_radius_mm
        self.density_sigma_vox, self.release_density, self.distance_penalty = density_sigma_vox, release_density, distance_penalty
        self._density = np.zeros(0)
        self._cache_tick: Optional[int] = None
        self._goals: Dict[int, Optional[Tuple[int, int, int]]] = {}

    def _plan(self, tick: int) -> None:
        if self._cache_tick == tick:
            return
        sc = self._scorer
        need = sc.truth & (self._dose.dose < sc.dose_threshold)
        self._density = gaussian_filter(need.astype(np.float32), self.density_sigma_vox)
        pos = np.array(self._positions(), dtype=np.float64)
        self._goals = {i: None for i in range(len(pos))}
        pts = np.argwhere(need)
        if len(pts):
            d = np.sqrt(((pts[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))  # need voxels x squads
            owner = np.argmin(d, axis=1)
            dens = self._density[tuple(pts.T)]
            for i in range(len(pos)):
                mine = np.flatnonzero(owner == i)
                if len(mine) == 0:  # a squad with no cell takes from the whole set
                    mine = np.arange(len(pts))
                score = dens[mine] - self.distance_penalty * d[mine, i]
                j = mine[int(np.argmax(score))]
                self._goals[i] = (int(pts[j][0]), int(pts[j][1]), int(pts[j][2]))
        self._cache_tick = tick

    def decide(self, view: LocalView) -> Intent:
        o = view.observation
        self._plan(view.tick)
        pos = tuple(o["position_vox"])
        here_need = float(self._density[pos]) >= self.release_density and bool(self._scorer.truth[pos]) \
            and float(self._dose.dose[pos]) < self._scorer.dose_threshold
        goal = self._goals.get(o["squad"])
        if here_need:
            return Intent(ACTION_DECIDE, {"target": STAY, "release": int(o["payload"] > 0.0), "amount": self.amount})
        if goal is None:
            return Intent(ACTION_DECIDE, {"target": STAY, "release": 0, "amount": self.amount})
        label = sprime_candidate_label(goal, o["candidates"], self.voxel_mm, self.map_radius_mm)
        return Intent(ACTION_DECIDE, {"target": label, "release": 0, "amount": self.amount,
                                      "target_vox": list(goal)}, meta={"oracle_goal": list(goal)})

    def describe(self) -> Dict[str, Any]:
        return {"policy": "oracle_Sprime", "amount": self.amount, "map_radius_mm": self.map_radius_mm}


def oracle_for(world) -> OracleSPrime:
    """Wire S' to a world (scorer accessor + live dose + squad positions)."""
    return OracleSPrime(world.scorer_truth(), world.squad_positions, world.field, voxel_mm=world.voxel_mm)
