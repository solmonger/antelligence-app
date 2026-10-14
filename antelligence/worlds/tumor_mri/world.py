"""3D drug-delivery world on an MRI crop (v4.1, goal section 4.2).

Eight squads enter on the crop's outer shell, walk one voxel per tick along an A* path inside the brain mask toward
a target chosen every 5 ticks, and release drug as 3D Gaussian puffs that diffuse and decay. Score = how many
ground-truth whole-tumour voxels reach the dose threshold.

Rules worth knowing (all goal 4.2 unless marked):

* Perceived region = L1's definition (6-connected components of p >= 0.3, >= 100 mm^3, split above 20 cm^3), computed
  from the perception arrays or passed in as ``regions`` (INTERFACES knowledge payloads). Entry exclusion, Explore and
  the candidate pool all use it. Candidates A..F are at most six regions chosen by the queen (``set_candidates``);
  the fallback is :func:`select_candidates` (rank by p x volume).
* Visited-radius rule: a region is *visited* once any squad stands within ``visit_radius_mm`` (6 mm) of its centroid;
  visits are tracked for every perceived region, not just candidates.
* Stay keeps the current target. Explore = nearest unvisited perceived region (all regions, not only candidates); if
  none is left, the nearest highest-uncertainty cluster (unc >= the ``explore_unc_quantile`` brain quantile, 26-connected)
  that is not yet *walked*: a cluster counts as walked once more than half of its voxels have been stood on by a squad.
* Off-target share = (drug anywhere except tumour + drug that left the crop) / released. Tumour = truth WT inside the
  brain. Drug in non-brain voxels inside the crop counts as off-target: conservative, since squads cannot enter
  those voxels but diffusion reaches them.

Information barrier: the observation builder reads only perception arrays (``_Percept``). Ground truth lives in
:class:`Scorer`, reachable only through :meth:`TumorMRIWorld.scorer_truth` (scorer / oracle S' use). Nothing in an
observation is a function of truth, except what the harness itself injects through ``look_fn``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from antelligence.kernel.types import NOOP, Intent, LocalView, Observation, Outcome
from antelligence.worlds.tumor_mri.config import WorldConfig
from antelligence.worlds.tumor_mri.data import (Candidate, Crop, Region, entry_sites_with_info, perceived_regions,
                                                select_candidates)
from antelligence.worlds.tumor_mri.physics import (DrugField, Pathfinder, Vox, rng, uptake_source_a, uptake_source_b,
                                                   washout_field)
from antelligence.worlds.tumor_mri.questions import CANDIDATE_LABELS, EXPLORE, STAY

ACTION_DECIDE = "decide"
QUEEN_HOOK_AGENT = "queen"
OBS_KEYS = ("squad", "tick", "round", "rounds", "position_vox", "payload", "payload_fraction", "last_release",
            "local_p_tumour", "local_unc", "local_look", "local_dose", "local_dose_rel", "target", "target_region", "candidates", "goal_dist_mm", "target_source")

LookFn = Callable[[int, Vox, int], Optional[float]]
QueenHook = Callable[[int, "TumorMRIWorld"], None]


@dataclass(frozen=True)
class _Percept:
    brain: np.ndarray
    p_tumour: np.ndarray
    unc: np.ndarray
    flair: Optional[np.ndarray]
    t1c: Optional[np.ndarray]


class Scorer:
    """Ground truth and scoring. Never passed to the observation builder."""

    def __init__(self, truth: np.ndarray, outside_crop_truth_voxels: int, dose_threshold: float, *,
                 tc: Optional[np.ndarray] = None, et: Optional[np.ndarray] = None) -> None:
        self.truth = truth
        self.tc = None if tc is None else np.asarray(tc, dtype=bool)  # scorer-side truth channels (F2 Source A); never observed
        self.et = None if et is None else np.asarray(et, dtype=bool)
        self.outside_crop_truth_voxels = int(outside_crop_truth_voxels)
        self.dose_threshold = float(dose_threshold)
        self._idx = np.flatnonzero(truth.ravel())
        self.denominator = int(len(self._idx)) + self.outside_crop_truth_voxels

    def coverage(self, dose: np.ndarray) -> float:
        if self.denominator == 0:
            return 0.0
        hit = int((dose.ravel()[self._idx] >= self.dose_threshold).sum())
        return hit / self.denominator

    @property
    def truth_share_outside_crop(self) -> float:
        return self.outside_crop_truth_voxels / self.denominator if self.denominator else 0.0


@dataclass
class _Squad:
    index: int
    pos: Vox
    payload: float
    target: str = STAY
    target_region: Optional[int] = None   # perception region id the current target refers to (None: none, or exploring)
    target_source: Optional[str] = None   # 'queen' while the target is the queen's assignment, 'worker' once it chose
    goal: Optional[Vox] = None
    path: Optional[List[Vox]] = None
    last_release: Optional[Dict[str, Any]] = None
    uptake_pending: List[Dict[str, Any]] = field(default_factory=list)   # F2: releases not yet read by the uptake sensor


class TumorMRIWorld:
    """Implements :class:`antelligence.kernel.types.World`. Deterministic in (crop, config, seed)."""

    def __init__(self, crop: Crop, config: Optional[WorldConfig] = None, seed: int = 0, *,
                 look_fn: Optional[LookFn] = None, queen_hook: Optional[QueenHook] = None,
                 regions: Optional[Sequence[Region]] = None,
                 scorer_channels: Optional[Mapping[str, np.ndarray]] = None) -> None:
        self.config = config or WorldConfig()
        self.seed = int(seed)
        self.voxel_mm = crop.voxel_mm
        self.shape = crop.shape
        self.case_id = crop.case_id
        self._percept = _Percept(crop.brain, crop.p_tumour, crop.unc, crop.flair, crop.t1c)
        ch = scorer_channels or {}
        self._scorer = Scorer(crop.truth, crop.outside_crop_truth_voxels, self.config.dose_threshold,
                              tc=ch.get("tc"), et=ch.get("et"))
        self.look_fn = look_fn
        self.regions: List[Region] = list(regions) if regions is not None else perceived_regions(
            crop.p_tumour, crop.unc, crop.brain, crop.voxel_mm)
        self.queen_hook = queen_hook
        cfg = self.config
        uptake, washout = self._hidden_fields(crop)
        self.field = DrugField(self.shape, release_sigma_vox=cfg.release_sigma_mm / crop.voxel_mm,
                               diffuse_sigma_vox=cfg.diffuse_sigma_vox, decay=cfg.decay, uptake=uptake, washout=washout)
        self._paths = Pathfinder(crop.brain, weight=cfg.path_weight, max_expand=cfg.path_max_expand)
        self.entry_sites, self.entry_shell_widened_mm = entry_sites_with_info(
            crop.brain, crop.p_tumour, self.seed, n=cfg.n_squads, voxel_mm=crop.voxel_mm,
            shell_mm=cfg.entry_shell_mm, min_dist_mm=cfg.entry_min_dist_mm)
        self._on_target = crop.truth & crop.brain
        self.total_drug = cfg.resolve_total_drug(crop.p_tumour)  # perception only (never the truth mask)
        share = self.total_drug / cfg.n_squads
        self._squads = [_Squad(i, site, share) for i, site in enumerate(self.entry_sites)]
        self._visited = np.zeros(self.shape, dtype=bool)
        for s in self._squads:
            self._visited[s.pos] = True
        self._visited_regions: set = set()
        self._clusters = self._uncertainty_clusters(crop, cfg.explore_unc_quantile)
        self._candidates: List[Candidate] = []
        self._squad_candidates: Dict[int, List[Candidate]] = {}
        self._revision = 0
        self._tick = 0
        self._coverage_at_queen: Dict[int, float] = {}
        self._time_to_half: Optional[int] = None
        self._coverage_trace: List[float] = []
        self._sensor_cache: Dict[tuple, Dict[str, Optional[Dict[str, Any]]]] = {}
        self._sensor_counts = {"washout_turns": 0, "washout_available": 0, "washout_fast": 0,
                               "uptake_readings": 0, "uptake_slow": 0}

    def _hidden_fields(self, crop: Crop):
        """F2 / F3 fields, scorer-side. (None, None) when both facts are off: the v4.1 arithmetic path."""
        f = self.config.facts
        uptake = washout = None
        if f.f2:
            if f.f2_source == "A":
                sc = self._scorer
                if sc.tc is None or sc.et is None:
                    raise ValueError("f2_source A needs the store's TC and ET truth channels (scorer_channels); "
                                     "use f2_source B when they are missing")
                uptake = uptake_source_a(sc.tc, sc.et, f.u_core)
            else:
                uptake = uptake_source_b(crop.truth, crop.case_id, f.u_core, f.f2_share, crop.voxel_mm)
        if f.f3:
            washout = washout_field(crop.truth, crop.case_id, f.f3_decay, f.f3_share, crop.voxel_mm)
        return uptake, washout

    # --- World protocol -----------------------------------------------------

    @property
    def revision(self) -> int:
        return self._revision

    def agents(self) -> List[str]:
        return [f"squad-{i}" for i in range(self.config.n_squads)]

    def done(self, tick: int) -> bool:
        return tick >= self.config.ticks

    def describe(self) -> Dict[str, Any]:
        p = self._percept
        h = hashlib.sha256()
        for arr in (p.brain, p.p_tumour, p.unc):
            h.update(np.ascontiguousarray(arr).tobytes())
        return {"world": "tumor_mri", "case_id": self.case_id, "shape": list(self.shape), "voxel_mm": self.voxel_mm,
                "seed": self.seed, "config": self.config.to_dict(), "total_drug": self.total_drug, "perception_sha256": h.hexdigest(),
                "entry_sites": [list(s) for s in self.entry_sites], "entry_shell_widened_mm": self.entry_shell_widened_mm,
                "n_regions": len(self.regions),
                "outside_crop_truth_voxels": self._scorer.outside_crop_truth_voxels}

    # --- perception inputs (queen / harness) -----------------------------------

    def set_candidates(self, candidates: Sequence[Candidate], squad: Optional[int] = None) -> None:
        """Candidate regions offered to the squads (all, or one squad if ``squad`` is given). Perception-derived."""
        cands = list(candidates)
        if len(cands) > len(CANDIDATE_LABELS) or [c.label for c in cands] != list(CANDIDATE_LABELS[:len(cands)]):
            raise ValueError("candidates must be labelled A, B, C ... in order, at most six")
        if squad is None:
            self._candidates = cands
            self._squad_candidates.clear()
        else:
            self._squad_candidates[int(squad)] = cands
        self._revision += 1

    def assign_route(self, squad: int, region_id: int) -> bool:
        """The queen's route gives ``squad`` a first region. A squad with no current target adopts it; a squad that
        already has one keeps it (the queen advises, never gates). Perception regions only, never the truth."""
        sq = self._squads[int(squad)]
        if sq.target != STAY or sq.goal is not None:
            return False
        for c in self.candidates_for(int(squad)):
            if c.region_id == region_id:
                sq.target, sq.target_source, sq.target_region = c.label, "queen", c.region_id
                sq.goal = self._paths.nearest_brain_voxel(c.centroid_vox)
                sq.path = self._paths.path(sq.pos, sq.goal) if sq.goal != sq.pos else []
                self._revision += 1
                return True
        return False

    def candidates_for(self, squad: int) -> List[Candidate]:
        return list(self._squad_candidates.get(squad, self._candidates))

    def scorer_truth(self) -> Scorer:
        """Scorer-only accessor. Policies other than oracle S' must never be handed this."""
        return self._scorer

    def squad_payloads(self) -> List[float]:
        """Remaining payload per squad (scripted/oracle arms only; model arms see their own payload in the observation)."""
        return [sq.payload for sq in self._squads]

    @property
    def pathfinder(self) -> Pathfinder:
        """The world's A* (brain mask included); for the oracle's path-aware planning."""
        return self._paths

    def squad_positions(self) -> List[Vox]:
        return [s.pos for s in self._squads]

    def local_patches(self, agent_id: str, size: int = 64) -> Dict[str, np.ndarray]:
        """Axial FLAIR / T1c patches (size x size, zero padded) centred on the squad, for d1 local look."""
        sq = self._squads[self._idx(agent_id)]
        out: Dict[str, np.ndarray] = {}
        for name in ("flair", "t1c"):
            vol = getattr(self._percept, name)
            if vol is None:
                continue
            patch = np.zeros((size, size), dtype=np.float32)
            x0, y0 = sq.pos[0] - size // 2, sq.pos[1] - size // 2
            xs, xe = max(x0, 0), min(x0 + size, self.shape[0])
            ys, ye = max(y0, 0), min(y0 + size, self.shape[1])
            patch[xs - x0:xe - x0, ys - y0:ye - y0] = vol[xs:xe, ys:ye, sq.pos[2]]
            out[name] = patch
        return out

    def observe(self, agent_id: str, tick: int) -> Observation:
        i = self._idx(agent_id)
        sq = self._squads[i]
        cfg = self.config
        pr = self._percept
        dose = float(self.field.dose[sq.pos])
        share = self.total_drug / cfg.n_squads
        cands = []
        for c in self.candidates_for(i):
            d = c.to_dict()
            d["dist_mm"] = round(self.voxel_mm * math.dist(sq.pos, c.centroid_vox), 3)
            d["visited"] = c.region_id in self._visited_regions
            cands.append(d)
        look = self.look_fn(i, sq.pos, tick) if self.look_fn else None
        data = {
            "squad": i, "tick": int(tick), "round": int(tick) // cfg.decision_every, "rounds": cfg.rounds,
            "position_vox": list(sq.pos), "payload": sq.payload, "payload_fraction": sq.payload / share,
            "last_release": dict(sq.last_release) if sq.last_release else None,
            "local_p_tumour": float(pr.p_tumour[sq.pos]), "local_unc": float(pr.unc[sq.pos]),
            "local_look": None if look is None else float(look),
            "local_dose": dose, "local_dose_rel": dose / cfg.dose_threshold,
            "target": sq.target, "target_region": sq.target_region, "candidates": cands,
            "target_source": sq.target_source,
            "goal_dist_mm": None if sq.goal is None else round(self.voxel_mm * math.dist(sq.pos, sq.goal), 3),
        }
        return Observation(data=data)

    def apply(self, agent_id: str, intent: Intent, tick: int) -> Outcome:
        if intent.action == NOOP:
            return Outcome(True, "noop")
        if intent.action != ACTION_DECIDE:
            return Outcome(False, f"unknown action {intent.action!r}")
        cfg = self.config
        if tick % cfg.decision_every != 0:
            return Outcome(False, "not a decision tick")
        i = self._idx(agent_id)
        sq = self._squads[i]
        p = intent.params
        # target_vox (a raw goal voxel) is for scripted arms only (oracle S'), which tag meta["scripted"] = True. Any other
        # intent that carries it is refused outright, so a label is the only way a model arm can steer a squad.
        if p.get("target_vox") is not None and intent.meta.get("scripted") is not True:
            return Outcome(False, "target_vox is allowed only for scripted policies")
        target, release, amount = p.get("target"), int(p.get("release", 0)), int(p.get("amount", 1))
        valid = {c.label for c in self.candidates_for(i)} | {STAY, EXPLORE}
        if target not in valid:
            return Outcome(False, f"target {target!r} not offered")
        if release not in (0, 1) or not 1 <= amount <= 5:
            return Outcome(False, "release must be 0/1 and amount 1..5")
        deposited = 0.0
        if release and sq.payload > 0.0:
            deposited = sq.payload * cfg.amount_fractions[amount - 1]
            self.field.deposit(sq.pos, deposited)
            sq.payload -= deposited
            sq.last_release = {"tick": int(tick), "amount_level": amount,
                               "fraction": cfg.amount_fractions[amount - 1], "mass": deposited}
            if self.field.uptake is not None:
                sq.uptake_pending.append({"tick": int(tick), "pos": sq.pos, "dose0": float(self.field.dose[sq.pos]),
                                          "raw0": float(self.field.dose_raw[sq.pos])})
        override = p.get("target_vox")
        if override is not None or target != STAY:  # Stay keeps the current target (and its path)
            sq.target = target
            sq.target_source = "worker"
            sq.target_region = next((c.region_id for c in self.candidates_for(i) if c.label == target), None)
            sq.goal = self._resolve_goal(i, sq, target, override)
            sq.path = self._paths.path(sq.pos, sq.goal) if sq.goal is not None and sq.goal != sq.pos else []
        self._revision += 1
        return Outcome(True, None, {"released": deposited, "goal": list(sq.goal) if sq.goal else None})

    def step_environment(self, tick: int) -> None:
        for sq in self._squads:
            if sq.path:
                sq.pos = sq.path.pop(0)
                self._visited[sq.pos] = True
        self._mark_visits()
        self.field.step(self._on_target)
        self._tick = tick + 1
        cov = self._scorer.coverage(self.field.dose)
        self._coverage_trace.append(cov)
        if self._time_to_half is None and cov >= 0.5:
            self._time_to_half = tick + 1
        if self._tick % self.config.queen_every == 0 or self._tick == self.config.ticks:
            self._coverage_at_queen[self._tick] = cov
        self._revision += 1

    def metrics(self) -> Dict[str, Any]:
        f, sc = self.field, self._scorer
        cov = sc.coverage(f.dose)
        at_queen = dict(self._coverage_at_queen)
        at_queen.setdefault(0, 0.0)
        return {
            "coverage": cov, "coverage_at_queen": {int(k): v for k, v in sorted(at_queen.items())},
            "time_to_50_coverage": self._time_to_half,
            "off_target_share": self.off_target_share(),
            "truth_share_outside_crop": sc.truth_share_outside_crop,
            "released_total": f.released, "decayed": f.decayed, "lost_boundary": f.lost_boundary,
            "lost_boundary_share": f.lost_boundary / f.released if f.released > 0 else 0.0,
            "mass_residual": f.mass_balance_residual(), "ticks": self._tick,
            "entry_shell_widened_mm": self.entry_shell_widened_mm,
            **({"facts": self.sensor_stats()} if self.config.facts.any_on else {}),
        }

    # --- hidden-fact sensors (F2, F3) ---------------------------------------------------------------------------

    def sensor_readings(self, agent_id: str, tick: int) -> Dict[str, Optional[Dict[str, Any]]]:
        """What the ant's sensors read at its decision turn ``tick`` (computed once per squad and tick, then cached).

        * ``uptake`` (F2): about the ant's own last release voxel v, even if it has moved on: (dose gain at v since the
          release) / (the gain u = 1 would have given) + N(0, 0.1) noise, stream ``uptake_noise``. Words: "as expected"
          (>= 0.7) or "slow". One reading per release, on the first turn after it.
        * ``washout`` (F3): delta(v) + N(0, 0.01) noise (stream ``washout_noise``) at the ant's own voxel, only when the local
          concentration is at least 0.01 x D_t. Words: "normal", or "fast" (>= 0.5 x f3_decay).
        Each is ``{"pos_vox", "reading", "word"}`` or None (fact off, nothing to read, or no signal). Never in observe()."""
        i = self._idx(agent_id)
        key = (i, int(tick))
        hit = self._sensor_cache.get(key)
        if hit is not None:
            return hit
        sq, f, fl = self._squads[i], self.config.facts, self.field
        out: Dict[str, Optional[Dict[str, Any]]] = {"uptake": None, "washout": None}
        if fl.uptake is not None:
            ready = [p for p in sq.uptake_pending if p["tick"] < tick]
            sq.uptake_pending = [p for p in sq.uptake_pending if p["tick"] >= tick]
            if ready:
                p = ready[-1]
                v = p["pos"]
                raw_gain = float(fl.dose_raw[v]) - p["raw0"]
                if raw_gain > 1e-12:
                    ratio = (float(fl.dose[v]) - p["dose0"]) / raw_gain
                    val = ratio + float(rng("uptake_noise", self.case_id, self.seed, i, int(tick)).normal(0.0, 0.1))
                    word = "as expected" if val >= 0.7 else "slow"
                    out["uptake"] = {"pos_vox": list(v), "reading": val, "word": word}
                    self._sensor_counts["uptake_readings"] += 1
                    self._sensor_counts["uptake_slow"] += word == "slow"
        if fl.washout is not None:
            self._sensor_counts["washout_turns"] += 1
            v = sq.pos
            if float(fl.conc[v]) >= 0.01 * self.config.dose_threshold:
                val = float(fl.washout[v]) + float(rng("washout_noise", self.case_id, self.seed, i, int(tick)).normal(0.0, 0.01))
                word = "fast" if val >= 0.5 * f.f3_decay else "normal"
                out["washout"] = {"pos_vox": list(v), "reading": val, "word": word}
                self._sensor_counts["washout_available"] += 1
                self._sensor_counts["washout_fast"] += word == "fast"
        self._sensor_cache[key] = out
        return out

    def pending_sensor_events(self, agent_id: str, tick: int) -> List[Dict[str, Any]]:
        """Automatic-mark triggers for this squad and tick: a ``slow_uptake`` event when the uptake reading is slow, a
        ``washout`` event when the washout reading is fast. K1 turns them into marks; the world places none itself.
        Each event is ``{"kind", "ant", "tick", "pos_vox", "reading"}``."""
        r = self.sensor_readings(agent_id, tick)
        ant = self._idx(agent_id)
        ev: List[Dict[str, Any]] = []
        if r["uptake"] is not None and r["uptake"]["word"] == "slow":
            ev.append({"kind": "slow_uptake", "ant": ant, "tick": int(tick), **{k: r["uptake"][k] for k in ("pos_vox", "reading")}})
        if r["washout"] is not None and r["washout"]["word"] == "fast":
            ev.append({"kind": "washout", "ant": ant, "tick": int(tick), **{k: r["washout"][k] for k in ("pos_vox", "reading")}})
        return ev

    def sensor_stats(self) -> Dict[str, Any]:
        """Sensor counts for the run; ``washout_available_share`` = how often a washout reading was available."""
        c = dict(self._sensor_counts)
        c["washout_available_share"] = c["washout_available"] / c["washout_turns"] if c["washout_turns"] else None
        return c

    def off_target_share(self) -> float:
        """(drug in non-tumour voxels + drug that left the crop) / all released drug, "tumour" = truth WT.

        Every unit of released mass has one fate: left the crop, decayed (counted where it decayed) or is still
        present (counted where it is). On target = decayed inside WT + still inside WT.
        """
        f = self.field
        if f.released <= 0.0:
            return 0.0
        on_target = f.decayed_in_region + f.mass_in(self._on_target)
        return min(1.0, max(0.0, 1.0 - on_target / f.released))

    # --- queen cadence ------------------------------------------------------------

    def queen_ticks(self) -> List[int]:
        return list(range(0, self.config.ticks, self.config.queen_every))

    # --- internals --------------------------------------------------------------------

    def _idx(self, agent_id: str) -> int:
        try:
            i = int(agent_id.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            raise KeyError(agent_id) from None
        if not agent_id.startswith("squad-") or not 0 <= i < len(self._squads):
            raise KeyError(agent_id)
        return i

    def _resolve_goal(self, i: int, sq: _Squad, target: str, override: Any) -> Optional[Vox]:
        if override is not None:
            return self._paths.nearest_brain_voxel(override)
        if target == STAY:
            return None
        if target == EXPLORE:
            return self._explore_goal(i, sq)
        for c in self.candidates_for(i):
            if c.label == target:
                return self._paths.nearest_brain_voxel(c.centroid_vox)
        return None

    def _mark_visits(self) -> None:
        r2 = (self.config.visit_radius_mm / self.voxel_mm) ** 2
        for r in self.regions:
            if r.region_id not in self._visited_regions and any(
                    sum((a - b) ** 2 for a, b in zip(sq.pos, r.centroid_vox)) <= r2 for sq in self._squads):
                self._visited_regions.add(r.region_id)

    @staticmethod
    def _uncertainty_clusters(crop: Crop, quantile: float) -> List[np.ndarray]:
        from scipy import ndimage
        if not crop.brain.any():
            return []
        thr = float(np.quantile(crop.unc[crop.brain], quantile))
        lab, n = ndimage.label((crop.unc >= thr) & crop.brain, structure=np.ones((3, 3, 3), dtype=bool))
        return [np.argwhere(lab == k) for k in range(1, n + 1)]

    def _explore_goal(self, i: int, sq: _Squad) -> Optional[Vox]:
        """Explore = nearest perceived region no squad has visited; if none, nearest highest-uncertainty voxel
        cluster that has not been mostly walked over."""
        best, best_d = None, math.inf
        for c in self.regions:  # all perceived regions, not only the candidates
            if c.region_id in self._visited_regions:
                continue
            d = math.dist(sq.pos, c.centroid_vox)
            if d < best_d:
                best, best_d = c, d
        if best is not None:
            return self._paths.nearest_brain_voxel(best.centroid_vox)
        goal, best_d = None, math.inf
        for pts in self._clusters:
            if len(pts) and self._visited[tuple(pts.T)].mean() > 0.5:
                continue
            centre = pts.mean(axis=0)
            d = math.dist(sq.pos, centre)
            if d < best_d:
                goal, best_d = centre, d
        return None if goal is None else self._paths.nearest_brain_voxel(goal)


# --- episode driver ----------------------------------------------------------------------


def run_episode(world: TumorMRIWorld, policy_for: Callable[[str], Any],
                default_candidates: bool = True) -> Dict[str, Any]:
    """Run one episode. ``policy_for(agent_id)`` returns an object with ``decide(LocalView) -> Intent``.

    The queen hook fires at tick 0 and every ``queen_every`` ticks. With no hook and ``default_candidates`` the
    candidate pool is the top six perceived regions by p x volume.
    """
    cfg = world.config
    agents = world.agents()
    policies = {a: policy_for(a) for a in agents}
    if default_candidates and world.queen_hook is None:
        world.set_candidates(select_candidates(world.regions))
    tick = 0
    while not world.done(tick):
        if world.queen_hook is not None and tick % cfg.queen_every == 0:
            world.queen_hook(tick, world)
        if tick % cfg.decision_every == 0:
            for a in agents:
                obs = world.observe(a, tick)
                view = LocalView(agent_id=a, tick=tick, revision=world.revision, seed=world.seed,
                                 observation=obs.data)
                world.apply(a, policies[a].decide(view), tick)
        world.step_environment(tick)
        tick += 1
    return world.metrics()
