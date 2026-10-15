"""v4.3 B1: the foraging uptake field (replaces v4.2's F2), its sensor words and automatic marks. Scorer-side, never observed.

How well each part of the tumour takes up drug. The scan cannot show it; an ant can sense it only through its own readings
(the uptake sensor at its own release voxel, and its own dose reading).

    u(v) = clip(base(v) * exp(sigma * z(v) - sigma^2 / 2), 0.05, 1.0)   inside the whole tumour (WT),   u = 1 outside WT.

* ``base`` comes from the case's own scorer-side sub-region channels: rim = ``et & wt``; core = ``tc & ~et & wt``;
  edema = ``wt & ~tc``. Where the channels disagree with each other (ET voxels outside TC, which the expert labels do not
  produce) the voxel takes the first of rim, core, edema it matches, so the three masks are disjoint.
* ``z`` is ``rng("f2_uptake", case_id).standard_normal(shape)`` smoothed by a Gaussian (sigma 2.5 voxels, reflect) and divided
  by its standard deviation over the whole crop (no mean subtraction).
* The three presets in :data:`UPTAKE_LEVELS` are named assumptions. They are never tuned.

The sensor reading (``world.py``) keeps the v4.2 form: the ratio of the dose gain to the gain u = 1 would have given, plus
N(0, 0.1) noise. This module only gives the reading its word (:func:`uptake_word`) and the automatic marks (:func:`automatic_marks_v43`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from antelligence.worlds.tumor_mri.config import UPTAKE_LEVEL_NAMES
from antelligence.worlds.tumor_mri.physics import rng

U_MIN, U_MAX = 0.05, 1.0
Z_SMOOTH_SIGMA_VOX = 2.5
GOOD_AT, POOR_BELOW = 0.7, 0.4        # reading >= 0.7 good, < 0.4 poor, else middling

# level -> (base rim, base core, base edema, sigma). Named assumptions, NEVER tuned (goal B1).
UPTAKE_LEVELS: Dict[str, Dict[str, float]] = {
    "low": {"rim": 1.0, "core": 0.6, "edema": 0.9, "sigma": 0.2},
    "mid": {"rim": 1.0, "core": 0.4, "edema": 0.8, "sigma": 0.4},
    "high": {"rim": 1.0, "core": 0.2, "edema": 0.7, "sigma": 0.6},
}
assert tuple(UPTAKE_LEVELS) == UPTAKE_LEVEL_NAMES


class ForagingError(ValueError):
    pass


def subregion_masks(wt: Any, tc: Any, et: Any) -> Dict[str, np.ndarray]:
    """Disjoint boolean masks ``rim`` (ET), ``core`` (TC without ET) and ``edema`` (WT without TC), all inside WT."""
    wt, tc, et = (np.asarray(a, dtype=bool) for a in (wt, tc, et))
    if not (wt.shape == tc.shape == et.shape):
        raise ForagingError(f"wt / tc / et shapes differ: {wt.shape} {tc.shape} {et.shape}")
    rim = et & wt
    core = tc & ~et & wt
    edema = wt & ~tc & ~rim
    return {"rim": rim, "core": core, "edema": edema}


def foraging_uptake(wt: Any, tc: Any, et: Any, case_id: str, level: str) -> np.ndarray:
    """The uptake field u(v) (float64, shape of ``wt``). ``tc`` and ``et`` are required: there is no Source B for it."""
    if tc is None or et is None:
        raise ForagingError("the foraging uptake field needs the TC and ET channels (scorer_channels); there is no "
                            "Source B for it. Use f2_model: v42 with f2_source: B when the channels are missing")
    if level not in UPTAKE_LEVELS:
        raise ForagingError(f"uptake level must be one of {tuple(UPTAKE_LEVELS)}, got {level!r}")
    wt_b = np.asarray(wt, dtype=bool)
    preset = UPTAKE_LEVELS[level]
    masks = subregion_masks(wt_b, tc, et)
    base = np.zeros(wt_b.shape, dtype=np.float64)
    for name, m in masks.items():
        base[m] = preset[name]
    z = gaussian_filter(rng("f2_uptake", case_id).standard_normal(wt_b.shape), Z_SMOOTH_SIGMA_VOX, mode="reflect")
    sd = float(z.std())
    if sd > 0.0:
        z = z / sd                      # no mean subtraction
    sigma = preset["sigma"]
    u = np.clip(base * np.exp(sigma * z - 0.5 * sigma * sigma), U_MIN, U_MAX)
    out = np.ones(wt_b.shape, dtype=np.float64)
    out[wt_b] = u[wt_b]
    return out


def realised_means(u: np.ndarray, wt: Any, tc: Any, et: Any) -> Dict[str, Any]:
    """Mean u per sub-region (``rim``, ``core``, ``edema``) and over the whole tumour (``wt``), plus voxel counts
    (``n_*``) and the share of tumour voxels clipped at the floor / ceiling (``clipped_low``, ``clipped_high``).
    A sub-region with no voxels has mean None."""
    u = np.asarray(u, dtype=np.float64)
    wt_b = np.asarray(wt, dtype=bool)
    out: Dict[str, Any] = {}
    parts = dict(subregion_masks(wt_b, tc, et))
    parts["wt"] = wt_b
    for name, m in parts.items():
        n = int(m.sum())
        out[name] = float(u[m].mean()) if n else None
        out[f"n_{name}"] = n
    n_wt = out["n_wt"]
    out["clipped_low"] = float((u[wt_b] <= U_MIN).mean()) if n_wt else None
    out["clipped_high"] = float((u[wt_b] >= U_MAX).mean()) if n_wt else None
    return out


# --- sensor words ----------------------------------------------------------------------------------------------------------

def uptake_word(reading: float) -> str:
    """good (>= 0.7), middling (0.4 to 0.7), poor (< 0.4). The v4.2 words ("as expected" / "slow") stay in ``world.py``."""
    if reading >= GOOD_AT:
        return "good"
    return "middling" if reading >= POOR_BELOW else "poor"


# --- automatic marks (colony) ------------------------------------------------------------------------------------------------

def automatic_marks_v43(ctx: Mapping[str, Any]) -> List[Tuple[str, Sequence[int], float, int]]:
    """``colony_policies.automatic_marks`` plus the uptake words: ``slow_uptake`` for the words ``slow`` or ``poor``, and a new
    ``good_uptake`` for ``good``; ``middling`` and ``as expected`` place nothing. Both uptake marks sit at the RELEASE voxel the
    reading is about, amount 1.0, reading in ppm. The base rule is untouched (it is the v4.2 rule, and B2's ant logic is elsewhere)."""
    from antelligence.worlds.tumor_mri.colony_policies import automatic_marks     # lazy: colony_policies imports world.py
    readings: List[Mapping[str, Any]] = []
    extra: List[Tuple[str, Sequence[int], float, int]] = []
    for r in ctx["readings"]:
        if r["kind"] == "uptake" and r.get("word") == "poor":
            readings.append({**r, "word": "slow"})            # the base rule turns a slow word into slow_uptake
        elif r["kind"] == "uptake" and r.get("word") == "good":
            ppm = int(min(1_000_000, max(0, round(float(r["value"]) * 1_000_000))))
            extra.append(("good_uptake", r.get("pos_vox") or ctx["pos_vox"], 1.0, ppm))
        else:
            readings.append(r)
    return list(automatic_marks({**ctx, "readings": readings})) + extra


def select_auto_mark_rule(world_cfg: Any):
    """The harness's ``auto_mark_rule``: the v4.3 rule when the foraging model is on, else the v4.2 ``automatic_marks``."""
    from antelligence.worlds.tumor_mri.colony_policies import automatic_marks
    f = world_cfg.facts
    return automatic_marks_v43 if (f.f2 and f.f2_model == "foraging") else automatic_marks


def queen_uptake_scaling(world_cfg: Any, sharing: bool) -> bool:
    """The queen's B3 flag: on only in the foraging model, and only in sharing arms (S-solo's queen sees no board)."""
    f = world_cfg.facts
    return bool(sharing and f.f2 and f.f2_model == "foraging")
