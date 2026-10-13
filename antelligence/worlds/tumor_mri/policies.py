"""No-model arms: script S (perception only) and oracle S' (sees truth), plus the S' -> candidate-label mapping.

Both implement the kernel ``Policy`` protocol and emit ``decide`` intents with the same typed fields a model arm
answers (target label, release 0/1, amount 1..5). S' additionally passes ``target_vox`` (its true goal), because a
label can only name a perceived region; the label is the router-training target (see :func:`sprime_candidate_label`).


Rules in brief
* S targets *candidates* (the labelled A..F pool) that are unvisited; "visited" is world state (a squad within 6 mm of
  the region centroid, ``visit_radius_mm``). S releases 50% of remaining payload whenever local p_tumour >= 0.5.
* S' is a greedy oracle (goal 4.2) that maximises the number of truth voxels whose *final* dose reaches D_t.
  Each decision tick it builds one plan for all squads:
    1. projected final dose P = dose so far + the future exposure of the drug already in flight (a forward run of
       the world's own diffusion/decay on the part of the crop the drug can reach);
    2. the *target set*: the cheapest truth voxels the drug left can afford to bring to D_t. A thin lattice of puffs fills
       a region at far less drug than a few stacked ones (a stacked puff overshoots D_t at its centre and leaves a
       shell short), so a puff is valued by the dose it delivers into the target set without overshoot, plus the count
       of voxels it lifts past D_t;
    3. squads are committed one after another. A squad scores release points (cluster centres of the uncovered target
       mass at several scales, hill-climbed, plus its own position and previous goal) at the time it would arrive
       (A* path length, rounded up to the next decision tick), for all five amount levels, with a unit-puff response
       kernel for the horizon left;
    4. a squad releases where it stands iff that is at least as good (per unit of drug) as the best point after
       walking there; the amount is the biggest level within half the best gain per unit of drug (all gain at the last chance);
    5. every committed puff, now or planned, is added to P before the next squad plans, so squads spread out.
  Explore and the walked-cluster rule live in the world (see ``world.py``); S' never explores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy import ndimage

from antelligence.kernel.types import Intent, LocalView
from antelligence.worlds.tumor_mri.config import WorldConfig
from antelligence.worlds.tumor_mri.data import Candidate
from antelligence.worlds.tumor_mri.physics import DrugField, PuffResponse, Vox
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


_KERN_R, _WIN_R = 24, 16  # simulation grid radius / gain-window radius (voxels) of the unit-puff response kernels
_COUNT_W = 1.0  # weight of the integer count of voxels lifted past D_t next to the dose delivered to the target set
_SPREAD_SIGMA = 3.0  # voxels: width of the plume a release smears over (leakage estimate for the target set)
_LEVEL_TOL = 0.5  # take the biggest amount whose gain per unit of drug is within this share of the best level's
_WASTE = 1.3  # overshoot / unevenness allowance when sizing the target set to the drug left
_PROJ_REL_TOL = 1e-4  # projection ignores concentration below this share of the peak
_PROJ_REACH = 2.8  # ... and sources reach 2.8 diffusion widths (sqrt(ticks left) voxels each) beyond the drug's bounding box
_EXACT_BUDGET = 3e7  # voxel-ticks above which the projection runs on a 2x coarser grid
_KCACHE: Dict[tuple, Dict[int, np.ndarray]] = {}


def _response_kernels(cfg: WorldConfig, voxel_mm: float) -> Dict[int, np.ndarray]:
    """Cumulative dose in a (2*_WIN_R+1)^3 window around a unit puff deposited ``n`` ticks before the end, for every ``n``
    a decision tick can have (``ticks - k * decision_every``). Same physics as the world, away from the crop boundary."""
    key = (cfg.release_sigma_mm, voxel_mm, cfg.diffuse_sigma_vox, cfg.decay, cfg.ticks, cfg.decision_every)
    if key not in _KCACHE:
        needed = {cfg.ticks - cfg.decision_every * j for j in range(cfg.rounds + 1)}
        needed = {n for n in needed if n >= 1}
        size = 2 * _KERN_R + 1
        f = DrugField((size, size, size), release_sigma_vox=cfg.release_sigma_mm / voxel_mm,
                      diffuse_sigma_vox=cfg.diffuse_sigma_vox, decay=cfg.decay)
        f.deposit((_KERN_R,) * 3, 1.0)
        lo, hi = _KERN_R - _WIN_R, _KERN_R + _WIN_R + 1
        out: Dict[int, np.ndarray] = {}
        for k in range(1, (max(needed) if needed else 0) + 1):
            f.step()
            if k in needed:
                out[k] = f.dose[lo:hi, lo:hi, lo:hi].astype(np.float32)
        _KCACHE[key] = out
    return _KCACHE[key]


@dataclass(frozen=True)
class _Option:
    """A release point for one squad: where, when it would arrive and release, the amount level, and its value."""
    point: Vox
    steps: int
    release_tick: int
    level: int  # 1..5
    eff: float  # voxels lifted past D_t per unit of drug (plus a tiny progress term)
    gain: float
    mass: float


@dataclass
class _Action:
    release_level: int = 0  # >0: release this level where the squad stands
    goal: Optional[Vox] = None  # walk here next (None: stay put)
    goal_level: int = 3


class OracleSPrime:
    """Greedy oracle ceiling (goal 4.2): sees the truth mask and the live dose, plans all squads together each decision tick.

    Objective: the number of truth voxels whose *final* dose reaches D_t (coverage is binary at D_t).
    * Projected dose: dose so far plus the exposure the drug in flight will still deliver (forward run of the world's own
      diffusion and decay, box-limited, on a 2x coarser grid when the box is large).
    * Target set: truth voxels, cheapest first (deficit over what a unit of drug delivers there, discounted by the share
      of the plume that leaves the tumour), until the drug left is spent. Pure per-puff counts are myopic: no single
      puff can lift a voxel when D_t needs several overlapping ones, and stacking puffs wastes drug above D_t. So a
      puff's value is the dose it adds toward D_t inside the target set (no credit above D_t) plus the count of voxels
      it lifts past D_t.
    * Squads commit in index order. Each scores candidate release points with a unit-puff response kernel for the
      horizon left when it would arrive (A* path length, rounded up to the next decision tick), for all five amount levels.
    * Release here iff that is at least as good per unit of drug as the best point after walking there. Amount: the
      biggest level within ``_LEVEL_TOL`` of the best gain per unit of drug (earlier drug is exposed longer, so spend it).
    * Every committed puff (now or planned) joins the projected dose before the next squad plans, and a point another squad
      already claimed this tick is worth 0.89x, so squads spread unless nothing within 90% of its value exists.
    Not a fair arm: it is the achievable ceiling for tuning the world and the router's training target.
    """

    CLAIM_PENALTY = 0.89
    GOAL_BONUS = 1.05  # hysteresis for the previous goal
    MAX_ASTAR = 12  # A* path lengths computed per squad per plan (they are checked best-first)

    def __init__(self, scorer: Scorer, positions, dose_field: DrugField, config: WorldConfig, *, voxel_mm: float = 2.0,
                 payloads=None, pathfinder=None, n_points: int = 5, map_radius_mm: float = MAP_RADIUS_MM) -> None:
        self._scorer, self._positions, self._dose, self.cfg = scorer, positions, dose_field, config
        self._payloads, self._paths = payloads, pathfinder
        self.voxel_mm, self.n_points, self.map_radius_mm = voxel_mm, n_points, map_radius_mm
        self._kern = _response_kernels(config, voxel_mm)  # free-space; sizes the target set
        self._puff = PuffResponse(release_sigma_vox=config.release_sigma_mm / voxel_mm,
                                  diffuse_sigma_vox=config.diffuse_sigma_vox, decay=config.decay, ticks=config.ticks,
                                  win_r=_WIN_R, kern_r=_KERN_R)
        self._shape = tuple(scorer.truth.shape)
        self._brain = pathfinder.brain if pathfinder is not None else np.ones(scorer.truth.shape, dtype=bool)
        idx = np.argwhere(scorer.truth)
        self._tbox = (idx.min(axis=0), idx.max(axis=0) + 1) if len(idx) else None
        self._rho = ndimage.gaussian_filter(scorer.truth.astype(np.float32), _SPREAD_SIGMA, mode="constant", truncate=3.0)
        self._target = np.zeros(scorer.truth.shape, dtype=bool)
        self._plan_tick: Optional[int] = None
        self._actions: Dict[int, _Action] = {}
        self._last_goal: Dict[int, Vox] = {}
        self._claimed: list = []

    # --- projected dose ---------------------------------------------------------------------------------------

    def _projected(self, tick: int) -> np.ndarray:
        """Dose so far + exposure the drug now in flight will still add by the end of the run (exact, box-limited)."""
        f, cfg = self._dose, self.cfg
        proj = f.dose.copy()
        n = cfg.ticks - tick
        peak = float(f.conc.max())
        if n < 1 or peak <= 0.0:
            return proj
        idx = np.argwhere(f.conc > peak * _PROJ_REL_TOL)
        pad = int(math.ceil(_PROJ_REACH * math.sqrt(n) * cfg.diffuse_sigma_vox)) + 2
        lo = np.maximum(idx.min(axis=0) - pad, 0)
        hi = np.minimum(idx.max(axis=0) + 1 + pad, proj.shape)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        box = f.conc[sl].astype(np.float32)
        if box.size * n > _EXACT_BUDGET:
            acc = self._expose(box, n, coarse=True)
        else:
            acc = self._expose(box, n, coarse=False)
        proj[sl] += acc
        return proj

    def _expose(self, conc: np.ndarray, n: int, *, coarse: bool) -> np.ndarray:
        """Cumulative dose over the next ``n`` ticks from concentration ``conc`` under the world's diffusion and decay.
        ``coarse`` runs it on a 2x down-sampled grid (about 8x cheaper) for big boxes; the dose is a smooth field."""
        cfg = self.cfg
        sigma, keep = cfg.diffuse_sigma_vox, np.float32(1.0 - cfg.decay)
        shape = conc.shape
        if coarse:
            even = [s + (s % 2) for s in shape]
            conc = np.pad(conc, [(0, e - s) for e, s in zip(even, shape)])
            conc = conc.reshape(even[0] // 2, 2, even[1] // 2, 2, even[2] // 2, 2).sum(axis=(1, 3, 5))
            sigma *= 0.5
        box, acc = conc, np.zeros(conc.shape, dtype=np.float32)
        for _ in range(n):
            box = ndimage.gaussian_filter(box, sigma, mode="constant", cval=0.0, truncate=3.0)
            np.maximum(box, 0.0, out=box)
            box *= keep
            acc += box
        if coarse:
            acc = np.repeat(np.repeat(np.repeat(acc, 2, axis=0), 2, axis=1), 2, axis=2)[tuple(slice(0, s) for s in shape)] / 8.0
        return acc

    # --- scoring ----------------------------------------------------------------------------------------------

    def _kernel(self, n: int) -> Optional[np.ndarray]:
        """Free-space unit-puff response (nearest available horizon); only for sizing the target set."""
        if n < 1 or not self._kern:
            return None
        return self._kern[min(self._kern, key=lambda k: (abs(k - n), k))]

    def _puff_kernel(self, v: Vox, n: int) -> Optional[np.ndarray]:
        """Unit-puff response at ``v`` for ``n`` ticks left, exact at the crop faces: the part that crosses a face (at
        release or by diffusion) is lost, so a puff near an edge delivers less than the free-space kernel says."""
        if n < 1:
            return None
        return self._puff.window(self._shape, v, n)

    @staticmethod
    def _window(shape, v):
        lo = [max(v[a] - _WIN_R, 0) for a in range(3)]
        hi = [min(v[a] + _WIN_R + 1, shape[a]) for a in range(3)]
        sl = tuple(slice(lo[a], hi[a]) for a in range(3))
        ks = tuple(slice(lo[a] - (v[a] - _WIN_R), hi[a] - (v[a] - _WIN_R)) for a in range(3))
        return sl, ks

    def _gains(self, proj: np.ndarray, v: Vox, kern: np.ndarray, masses: np.ndarray):
        """(count of truth voxels lifted past D_t, dose delivered into the target set as a share of D_t) per mass."""
        sl, ks = self._window(proj.shape, v)
        d_t = self._scorer.dose_threshold
        pw = proj[sl]
        m = self._scorer.truth[sl] & (pw < d_t)
        if not m.any():
            return np.zeros(len(masses)), np.zeros(len(masses))
        k = kern[ks][m].astype(np.float64)
        d = d_t - pw[m]
        counts = (d[None, :] <= masses[:, None] * k[None, :]).sum(axis=1).astype(np.float64)
        x0 = np.clip(pw[m], 0.0, None) / d_t
        gain = np.minimum(1.0 - x0[None, :], masses[:, None] * k[None, :] / d_t)  # dose toward D_t, no credit above it
        frac = (gain * self._target[sl][m][None, :]).sum(axis=1)  # ... and only inside the target set
        return counts, frac

    def _commit(self, proj: np.ndarray, v: Vox, kern: np.ndarray, mass: float) -> None:
        sl, ks = self._window(proj.shape, v)
        proj[sl] += mass * kern[ks]

    def _option(self, proj, v: Vox, steps: int, tick: int, payload: float) -> Optional[_Option]:
        cfg = self.cfg
        every = cfg.decision_every
        release_tick = tick + int(math.ceil(steps / every)) * every
        n = cfg.ticks - release_tick
        kern = self._puff_kernel(v, n)
        if kern is None:
            return None
        masses = payload * np.asarray(cfg.amount_fractions, dtype=np.float64)
        counts, frac = self._gains(proj, v, kern, masses)
        score = _COUNT_W * counts + frac
        if n <= every:  # last chance to release: take all the gain
            lvl = int(np.flatnonzero(score >= score.max() * (1.0 - 1e-9)).max())
        else:
            eff = score / masses
            lvl = int(np.flatnonzero(eff >= eff.max() * _LEVEL_TOL).max())
        return _Option(v, steps, release_tick, lvl + 1, float(score[lvl] / masses[lvl]), float(counts[lvl]), float(masses[lvl]))

    def _target_set(self, proj: np.ndarray, tick: int, budget: float) -> None:
        """The truth voxels the drug left can afford to bring to D_t, cheapest first. A voxel's cost is its dose deficit over
        what a unit of drug delivers there (the response kernel's total, times the share of the plume that stays in the
        tumour), times a waste allowance. Spreading dose thinly over this set, not stacking it, is what covers it."""
        self._target[:] = False
        if self._tbox is None or budget <= 0.0:
            return
        lo, hi = self._tbox
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        d_t = self._scorer.dose_threshold
        need = self._scorer.truth[box] & (proj[box] < d_t)
        kern = self._kernel(max(self.cfg.ticks - tick - 25, 1))
        if kern is None or not need.any():
            return
        cost = _WASTE * (d_t - proj[box][need]) / (float(kern.sum()) * np.maximum(self._rho[box][need], 0.05))
        order = np.argsort(cost, kind="stable")
        keep = order[np.cumsum(cost[order]) <= budget]
        flat = np.zeros(int(need.sum()), dtype=bool)
        flat[keep] = True
        sub = np.zeros(need.shape, dtype=bool)
        sub[need] = flat
        self._target[box] = sub

    # --- candidate points -------------------------------------------------------------------------------------

    def _snap(self, v: Sequence[float]) -> Vox:
        w = tuple(int(min(max(round(c), 0), s - 1)) for c, s in zip(v, self._scorer.truth.shape))
        if self._brain[w]:
            return w  # type: ignore[return-value]
        return self._paths.nearest_brain_voxel(w) if self._paths is not None else w  # type: ignore[return-value]

    def _candidate_points(self, proj: np.ndarray):
        if self._tbox is None:
            return []
        lo, hi = self._tbox
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        need = self._target[box] & (proj[box] < self._scorer.dose_threshold)
        if not need.any():
            return []
        nf = need.astype(np.float32)
        lab, nc = ndimage.label(need, structure=np.ones((3, 3, 3), dtype=bool))
        idx = np.arange(1, nc + 1)
        sizes = ndimage.sum(need, lab, idx)
        big = idx[np.argsort(-sizes, kind="stable")[: self.n_points + 3]]  # largest uncovered masses
        pts: list = []
        for sigma in (1.5, 3.0, 6.0):
            dens = ndimage.gaussian_filter(nf, sigma, mode="constant")
            peak = (dens == ndimage.maximum_filter(dens, size=int(2 * sigma) + 1)) & (dens >= 0.1 * float(dens.max()))
            cand = np.argwhere(peak)
            order = np.argsort(-dens[tuple(cand.T)], kind="stable")[: self.n_points]
            pts += [tuple(cand[j] + lo) for j in order]
            pts += [tuple(np.asarray(q) + lo) for q in ndimage.maximum_position(dens, lab, big)]  # one per mass
        cents = ndimage.center_of_mass(need, lab, big)
        pts += [tuple(np.asarray(c) + lo) for c in cents]
        seen, out = set(), []
        for p in pts:
            s = self._snap(p)
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # --- planning ---------------------------------------------------------------------------------------------

    def _steps(self, a: Vox, b: Vox, exact: bool) -> int:
        if exact and self._paths is not None:
            return len(self._paths.path(a, b))
        return max(abs(a[i] - b[i]) for i in range(3))

    def _value(self, o: _Option) -> float:
        return o.eff * (self.CLAIM_PENALTY if o.point in self._claimed else 1.0)

    def _best_walk(self, proj, pos: Vox, tick: int, payload: float, prev: Optional[Vox]) -> Optional[_Option]:
        """Best place to walk to and release. Ranked by A* path length: the straight-line count is only an optimistic
        bound (a step is never shorter than the Chebyshev distance), used to order the search and stop it early, so the
        pick is the best by walked path even when a nearby point sits behind a wall of non-brain voxels."""
        pts = [p for p in self._candidate_points(proj) if p != pos]
        if prev is not None and prev != pos and prev not in pts:
            pts.append(prev)
        pool: Dict[Vox, _Option] = {}
        for p in pts:
            o = self._option(proj, p, self._steps(pos, p, False), tick, payload)
            if o is not None and o.eff > 0.0:
                pool[p] = o
        if not pool:
            return None

        def val(o: _Option) -> float:
            return self._value(o) * (self.GOAL_BONUS if o.point == prev else 1.0)

        best = min(pool.values(), key=lambda o: (-val(o), o.steps, o.point))
        for _ in range(2):  # hill-climb: release points may sit off the mass
            improved = False
            for a in range(3):
                for d in (-2, 2):
                    q = list(best.point)
                    q[a] += d
                    if not 0 <= q[a] < self._brain.shape[a] or not self._brain[tuple(q)]:
                        continue
                    qv: Vox = (q[0], q[1], q[2])
                    o = self._option(proj, qv, self._steps(pos, qv, False), tick, payload)
                    if o is not None and o.eff > 0.0:
                        pool.setdefault(qv, o)
                    if o is not None and val(o) > val(best) * (1.0 + 1e-9):
                        best, improved = o, True
            if not improved:
                break
        order = sorted(pool.values(), key=lambda o: (-val(o), o.steps, o.point))
        winner: Optional[_Option] = None
        for n_exact, o in enumerate(order):
            if winner is not None and (val(o) <= val(winner) or n_exact >= self.MAX_ASTAR):
                break  # the optimistic value of everything left cannot beat the exact winner
            r = self._option(proj, o.point, self._steps(pos, o.point, True), tick, payload)
            if r is not None and r.eff > 0.0 and (winner is None or (val(r), -r.steps) > (val(winner), -winner.steps)):
                winner = r
        return winner

    def _plan_squad(self, i: int, pos: Vox, payload: float, tick: int, proj: np.ndarray) -> _Action:
        act = _Action()
        if payload <= 1e-12:
            return act
        prev = self._last_goal.get(i)
        here = self._option(proj, pos, 0, tick, payload)
        walk = self._best_walk(proj, pos, tick, payload, prev)
        if here is not None and here.eff > 0.0 and (walk is None or here.eff >= self._value(walk)):
            act.release_level = here.level
            self._commit(proj, pos, self._puff_kernel(pos, self.cfg.ticks - tick), here.mass)  # type: ignore[arg-type]
            payload -= here.mass
            walk = self._best_walk(proj, pos, tick, payload, prev) if payload > 1e-12 else None
        if walk is not None:
            act.goal, act.goal_level = walk.point, walk.level
            self._commit(proj, walk.point, self._puff_kernel(walk.point, self.cfg.ticks - walk.release_tick), walk.mass)  # type: ignore[arg-type]
            self._claimed.append(walk.point)
        elif act.release_level:
            self._claimed.append(pos)
        return act

    def _plan(self, tick: int, caller: int, caller_payload: float) -> None:
        proj = self._projected(tick)
        pos = [tuple(int(c) for c in p) for p in self._positions()]
        if self._payloads is not None:
            pay = list(self._payloads())
        else:
            pay = [self.cfg.total_drug / self.cfg.n_squads] * len(pos)
            pay[caller] = caller_payload
        self._target_set(proj, tick, float(sum(pay)))
        self._claimed, self._actions = [], {}
        for i in range(len(pos)):
            self._actions[i] = self._plan_squad(i, pos[i], pay[i], tick, proj)  # type: ignore[arg-type]
            g = self._actions[i].goal
            if g is not None:
                self._last_goal[i] = g
            else:
                self._last_goal.pop(i, None)
        self._plan_tick = tick

    def decide(self, view: LocalView) -> Intent:
        o = view.observation
        i, pos = o["squad"], tuple(int(c) for c in o["position_vox"])
        if self._plan_tick != view.tick:
            self._plan(view.tick, i, o["payload"])
        act = self._actions.get(i, _Action())
        release = int(act.release_level > 0 and o["payload"] > 0.0)
        amount = act.release_level if release else act.goal_level
        if act.goal is not None:
            label = sprime_candidate_label(act.goal, o["candidates"], self.voxel_mm, self.map_radius_mm)
            return Intent(ACTION_DECIDE, {"target": label, "release": release, "amount": amount,
                                          "target_vox": list(act.goal)},
                          meta={"oracle_goal": list(act.goal), "scripted": True})
        label = sprime_candidate_label(pos, o["candidates"], self.voxel_mm, self.map_radius_mm)
        return Intent(ACTION_DECIDE, {"target": STAY if label == EXPLORE else label, "release": release,
                                      "amount": max(amount, 1), "target_vox": list(pos)},
                      meta={"scripted": True})  # stay put: no stale path

    def describe(self) -> Dict[str, Any]:
        return {"policy": "oracle_Sprime", "n_points": self.n_points, "map_radius_mm": self.map_radius_mm}


def oracle_for(world) -> OracleSPrime:
    """Wire S' to a world (scorer accessor + live dose + squad positions)."""
    return OracleSPrime(world.scorer_truth(), world.squad_positions, world.field, world.config, voxel_mm=world.voxel_mm,
                        payloads=world.squad_payloads, pathfinder=world.pathfinder)
