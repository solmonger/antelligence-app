"""Tune total_drug and D_t with scripts S and S' only, on dev cases only (goal section 4.2 targets)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np

from antelligence.worlds.tumor_mri.config import WorldConfig
from antelligence.worlds.tumor_mri.data import Crop
from antelligence.worlds.tumor_mri.policies import ScriptS, oracle_for
from antelligence.worlds.tumor_mri.world import TumorMRIWorld, run_episode

STRICT = {"sprime": (0.70, 0.80), "s": (0.30, 0.50), "gap": 0.15, "gap_frac": 0.80}
WIDENED = {"sprime": (0.65, 0.85), "s": (0.25, 0.55), "gap": 0.15, "gap_frac": 0.70}


class SplitError(RuntimeError):
    """A case outside the dev split was requested."""


def load_split(path: Union[str, Path]) -> Dict[str, set]:
    """Committed format (config/v4_1/split.json): keys ``dev_a``, ``dev_b``, ``eval``, ``reserve`` (lists of case ids).
    Dev = dev_a + dev_b. Returns those four sets plus ``dev``."""
    data = json.loads(Path(path).read_text())
    out: Dict[str, set] = {k: {str(c) for c in data.get(k, [])} for k in ("dev_a", "dev_b", "eval", "reserve")}
    out["dev"] = out["dev_a"] | out["dev_b"]
    return out


def assert_dev_only(case_ids: Iterable[str], split: Mapping[str, set]) -> None:
    dev = split.get("dev", set())
    bad = sorted(c for c in case_ids if c not in dev)
    if bad:
        held = sorted(c for c in bad if c in split.get("eval", set()) | split.get("reserve", set()))
        raise SplitError(f"refusing to tune on cases not in dev (dev_a + dev_b): {bad}"
                         + (f"; held-out (eval/reserve): {held}" if held else ""))


def episode_coverages(crop: Crop, cfg: WorldConfig, seed: int = 0) -> Tuple[float, float]:
    """(coverage under S, coverage under S') for one case."""
    w = TumorMRIWorld(crop, cfg, seed=seed)
    cov_s = run_episode(w, lambda a: ScriptS())["coverage"]
    w2 = TumorMRIWorld(crop, cfg, seed=seed)
    oracle = oracle_for(w2)
    cov_sp = run_episode(w2, lambda a: oracle)["coverage"]
    return cov_s, cov_sp


def summarise(cov_s: Sequence[float], cov_sp: Sequence[float]) -> Dict[str, float]:
    s, sp = np.asarray(cov_s), np.asarray(cov_sp)
    return {"median_s": float(np.median(s)), "median_sprime": float(np.median(sp)),
            "gap_frac": float(np.mean(sp - s >= STRICT["gap"])), "n_cases": int(len(s))}


def in_band(summary: Mapping[str, float], band: Mapping[str, Any]) -> bool:
    return (band["sprime"][0] <= summary["median_sprime"] <= band["sprime"][1]
            and band["s"][0] <= summary["median_s"] <= band["s"][1]
            and summary["gap_frac"] >= band["gap_frac"])


def penalty(summary: Mapping[str, float]) -> float:
    """Distance from the band centres plus a shortfall term for the gap criterion."""
    return (abs(summary["median_sprime"] - 0.75) + abs(summary["median_s"] - 0.40)
            + max(0.0, STRICT["gap_frac"] - summary["gap_frac"]))


def grid_search(crops: Mapping[str, Crop], base: WorldConfig, drug_grid: Sequence[float], dt_grid: Sequence[float],
                *, seed: int = 0, evaluate: Callable[[Crop, WorldConfig, int], Tuple[float, float]] = episode_coverages
                ) -> Dict[str, Any]:
    """Search (total_drug, D_t). Both scale one-for-one, so only D_t / total_drug changes behaviour; equal ratios are
    evaluated once and shared."""
    cache: Dict[float, Dict[str, float]] = {}
    points: List[Dict[str, Any]] = []
    for dt in dt_grid:
        for drug in drug_grid:
            ratio = round(dt / drug, 9)
            if ratio not in cache:
                cfg = base.with_(total_drug=float(drug), dose_threshold=float(dt))
                res = [evaluate(crops[c], cfg, seed) for c in sorted(crops)]
                cache[ratio] = summarise([r[0] for r in res], [r[1] for r in res])
            points.append({"total_drug": float(drug), "dose_threshold": float(dt), **cache[ratio]})
    for band_name, band in (("strict", STRICT), ("widened", WIDENED)):
        ok = [p for p in points if in_band(p, band)]
        if ok:
            best = min(ok, key=lambda p: (penalty(p), p["total_drug"], p["dose_threshold"]))
            return {"status": "ok" if band_name == "strict" else "fallback_widened_band", "band": band_name,
                    "best": best, "n_points": len(points), "cases": sorted(crops)}
    best = min(points, key=lambda p: (penalty(p), p["total_drug"], p["dose_threshold"]))
    return {"status": "no_feasible_point", "band": None, "best": best, "n_points": len(points), "cases": sorted(crops)}
