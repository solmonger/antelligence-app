"""No-model arms: script S (perception only) and oracle S' (sees truth), plus the S' -> candidate-label mapping.

Both implement the kernel ``Policy`` protocol and emit ``decide`` intents with the same typed fields a model arm
answers (target label, release 0/1, amount 1..5). S' additionally passes ``target_vox`` (its true goal), because a
label can only name a perceived region; the label is the router-training target (see :func:`sprime_candidate_label`).


Rules in brief
* S targets *candidates* (the labelled A..F pool) that are unvisited; "visited" is world state (a squad within 6 mm of
  the region centroid, ``visit_radius_mm``). S releases 50% of remaining payload whenever local p_tumour >= 0.5.
* S' efficiency heuristic. For a puff of mass m at voxel v it estimates marginal coverage as
  sum over uncovered truth voxels u in a 21^3 window of min(1, m * R(u - v) / (D_t - dose(u))), where R is the
  cumulative dose of a unit puff over the ticks left (computed once per 10-tick bucket with the same physics, no
  boundary effects). Efficiency = marginal coverage / m. Per point it picks the largest of the five amount levels whose
  efficiency is within 10% of the best level (big puffs saturate). Squads are assigned to uncovered truth masses
  largest first, discounted by 1 / (1 + squads already assigned). A squad releases where it stands when that is within
  80% of the best point's efficiency; otherwise it walks to the best point (S' passes ``target_vox`` for that).
* Explore and the walked-cluster rule live in the world (see ``world.py``); S' never explores.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from antelligence.kernel.types import Intent, LocalView
from antelligence.worlds.tumor_mri.config import WorldConfig
from antelligence.worlds.tumor_mri.data import Candidate
from antelligence.worlds.tumor_mri.physics import DrugField
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
    """Perception-only script (goal 4.2).

    * target: the unvisited perceived region with the highest ``p_tumour * volume / distance`` (distance floored at
      1 mm); if every region is visited the squad stays on its current target;
    * release iff the local scan says p_tumour >= ``release_p``, with ``amount`` level 3 (50% of remaining payload).
    """

    def __init__(self, *, release_p: float = 0.5, amount: int = 3) -> None:
        self.release_p, self.amount = release_p, amount

    def decide(self, view: LocalView) -> Intent:
        o = view.observation
        release = int(o["local_p_tumour"] >= self.release_p and o["payload"] > 0.0)
        best, best_score = STAY, -1.0
        for c in o["candidates"]:  # label order -> ties go to the earlier label
            if c["visited"]:
                continue
            score = c["p_tumour"] * c["volume_mm3"] / max(c["dist_mm"], 1.0)
            if score > best_score:
                best, best_score = c["label"], score
        return Intent(ACTION_DECIDE, {"target": best, "release": release, "amount": self.amount})

    def describe(self) -> Dict[str, Any]:
        return {"policy": "script_S", "release_p": self.release_p, "amount": self.amount}


_RESP_R, _WIN_R = 24, 10


class OracleSPrime:
    """Greedy oracle (goal 4.2): sees the truth mask and the live dose.

    1. Uncovered truth = WT voxels with dose < D_t. Squads are assigned to its connected masses, largest first
       (a mass with more squads already on it is discounted: size / (1 + assigned)), each taking the nearest free squad.
    2. Within its mass a squad evaluates up to ``n_points`` release points (densest uncovered voxels, plus its own
       position) and the five amount levels, scoring each by marginal coverage per unit of drug: the number of
       uncovered voxels a puff would lift to D_t, from a unit-response kernel for the time left. It releases where
       it stands if that is nearly the best option, otherwise it walks to the best point.
    Built with the world's scorer accessor; it is the achievable ceiling for tuning, not a fair arm.
    """

    def __init__(self, scorer: Scorer, positions, dose_field: DrugField, config: WorldConfig, *, voxel_mm: float = 2.0,
                 n_points: int = 16, here_tolerance: float = 0.8, map_radius_mm: float = MAP_RADIUS_MM) -> None:
        self._scorer, self._positions, self._dose, self.cfg = scorer, positions, dose_field, config
        self.voxel_mm, self.n_points, self.here_tolerance, self.map_radius_mm = voxel_mm, n_points, here_tolerance, map_radius_mm
        self._cache_tick: Optional[int] = None
        self._lab = np.zeros(0)
        self._assign: Dict[int, int] = {}
        self._resp: Dict[int, np.ndarray] = {}

    def _response(self, horizon: int) -> np.ndarray:
        """Cumulative dose around a unit puff after ``horizon`` ticks (window radius _WIN_R), cached per 10-tick bucket."""
        h = max(10, int(math.ceil(horizon / 10.0)) * 10)
        if h not in self._resp:
            n = 2 * _RESP_R + 1
            f = DrugField((n, n, n), release_sigma_vox=self.cfg.release_sigma_mm / self.voxel_mm,
                          diffuse_sigma_vox=self.cfg.diffuse_sigma_vox, decay=self.cfg.decay)
            f.deposit((_RESP_R,) * 3, 1.0)
            for _ in range(h):
                f.step()
            lo, hi = _RESP_R - _WIN_R, _RESP_R + _WIN_R + 1
            self._resp[h] = f.dose[lo:hi, lo:hi, lo:hi].copy()
        return self._resp[h]

    def _plan(self, tick: int) -> None:
        if self._cache_tick == tick:
            return
        sc = self._scorer
        need = sc.truth & (self._dose.dose < sc.dose_threshold)
        self._need = need
        self._lab, n = ndimage.label(need, structure=np.ones((3, 3, 3), dtype=bool))
        self._assign = {}
        if n:
            idx = np.arange(1, n + 1)
            sizes = ndimage.sum(need, self._lab, idx)
            cents = np.array(ndimage.center_of_mass(need, self._lab, idx))
            pos = np.array(self._positions(), dtype=np.float64)
            taken = np.zeros(n, dtype=int)
            free = list(range(len(pos)))
            while free:
                comp = int(np.argmax(sizes / (1.0 + taken)))
                sq = min(free, key=lambda i: (float(np.linalg.norm(pos[i] - cents[comp])), i))
                self._assign[sq] = comp + 1
                taken[comp] += 1
                free.remove(sq)
        self._cache_tick = tick

    def _gain(self, resp: np.ndarray, v: Tuple[int, int, int], mass: float) -> float:
        """Marginal coverage of a puff of ``mass`` at ``v``: uncovered truth voxels lifted toward D_t, each counted by the
        fraction of its remaining deficit the puff fills (capped at 1). Fractional so partial progress is valued; a
        single puff rarely crosses the threshold alone."""
        shape = self._need.shape
        lo = [max(v[a] - _WIN_R, 0) for a in range(3)]
        hi = [min(v[a] + _WIN_R + 1, shape[a]) for a in range(3)]
        rlo = [lo[a] - (v[a] - _WIN_R) for a in range(3)]
        rhi = [rlo[a] + hi[a] - lo[a] for a in range(3)]
        sl = tuple(slice(lo[a], hi[a]) for a in range(3))
        rs = tuple(slice(rlo[a], rhi[a]) for a in range(3))
        deficit = self._scorer.dose_threshold - self._dose.dose[sl]
        need = self._need[sl]
        return float(np.minimum(1.0, mass * resp[rs][need] / deficit[need]).sum())

    def decide(self, view: LocalView) -> Intent:
        o = view.observation
        self._plan(view.tick)
        i, pos = o["squad"], tuple(o["position_vox"])
        comp = self._assign.get(i)
        hold = Intent(ACTION_DECIDE, {"target": STAY, "release": 0, "amount": 3})
        if comp is None or o["payload"] <= 0.0:
            return hold
        pts = np.argwhere(self._lab == comp)
        if len(pts) == 0:
            return hold
        dens = ndimage.gaussian_filter(self._need.astype(np.float32), 2.0)[tuple(pts.T)]
        order = np.argsort(-dens, kind="stable")
        step = max(1, len(order) // self.n_points)
        cand_pts = [tuple(int(c) for c in pts[j]) for j in order[::step][: self.n_points]]
        remaining = self.cfg.ticks - view.tick
        fracs = self.cfg.amount_fractions

        def best_at(v):
            horizon = remaining - max(abs(v[a] - pos[a]) for a in range(3))
            resp = self._response(max(horizon, 1))
            effs = [self._gain(resp, v, o["payload"] * fracs[l]) / (o["payload"] * fracs[l]) for l in range(5)]
            top = max(effs)
            if top <= 0.0:
                return (0.0, 3)
            # largest amount whose efficiency is within 10% of the best (saturation makes big puffs less efficient)
            lvl = max(l for l in range(5) if effs[l] >= 0.9 * top)
            return (effs[lvl], lvl + 1)

        here = best_at(pos)
        scored = [(best_at(v), v) for v in cand_pts]
        (top_eff, top_lvl), top_v = max(scored, key=lambda t: (t[0][0], -math.dist(t[1], pos)))
        if here[0] > 0.0 and here[0] >= self.here_tolerance * max(top_eff, here[0]):
            return Intent(ACTION_DECIDE, {"target": STAY, "release": 1, "amount": here[1]})
        if top_eff <= 0.0:
            return hold
        label = sprime_candidate_label(top_v, o["candidates"], self.voxel_mm, self.map_radius_mm)
        return Intent(ACTION_DECIDE, {"target": label, "release": 0, "amount": top_lvl, "target_vox": list(top_v)},
                      meta={"oracle_goal": list(top_v)})

    def describe(self) -> Dict[str, Any]:
        return {"policy": "oracle_Sprime", "n_points": self.n_points, "here_tolerance": self.here_tolerance,
                "map_radius_mm": self.map_radius_mm}


def oracle_for(world) -> OracleSPrime:
    """Wire S' to a world (scorer accessor + live dose + squad positions)."""
    return OracleSPrime(world.scorer_truth(), world.squad_positions, world.field, world.config, voxel_mm=world.voxel_mm)
