"""Tune total_drug and D_t with scripts S and S' only, on dev cases only (goal section 4.2 targets)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from antelligence.worlds.tumor_mri.config import WorldConfig
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


def episode_coverages(case: Any, cfg: WorldConfig, seed: int = 0, *, timed: bool = False) -> Tuple[float, ...]:
    """(coverage under S, coverage under S') for one case: a :class:`Crop`, or a store-adapter ``LoadedCase`` (crop plus
    the stored perception regions). With ``timed`` also the wall seconds of each episode."""
    crop = getattr(case, "crop", case)
    regions = getattr(case, "regions", None)
    t0 = time.perf_counter()
    w = TumorMRIWorld(crop, cfg, seed=seed, regions=regions)
    cov_s = run_episode(w, lambda a: ScriptS())["coverage"]
    t1 = time.perf_counter()
    w2 = TumorMRIWorld(crop, cfg, seed=seed, regions=regions)
    oracle = oracle_for(w2)
    cov_sp = run_episode(w2, lambda a: oracle)["coverage"]
    t2 = time.perf_counter()
    return (cov_s, cov_sp, t1 - t0, t2 - t1) if timed else (cov_s, cov_sp)


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


def serial_map(fn: Callable, arg_tuples: Sequence[Tuple]) -> List[Any]:
    return [fn(*a) for a in arg_tuples]


def grid_search(crops: Mapping[str, Any], base: WorldConfig, drug_grid: Sequence[float], dt_grid: Sequence[float],
                *, seed: int = 0, seeds: Optional[Sequence[int]] = None,
                evaluate: Callable[[Any, WorldConfig, int], Tuple[float, float]] = episode_coverages,
                map_fn: Callable[[Callable, Sequence[Tuple]], List[Any]] = serial_map,
                cache: Optional[MutableMapping[float, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Search (total_drug, D_t). Both scale one-for-one (dose is linear in drug), so only D_t / total_drug changes
    behaviour; equal ratios are evaluated once and shared.

    A case's coverage is the mean over ``seeds`` (default ``(seed,)``). ``map_fn(evaluate, [(crop, cfg, seed), ...])``
    runs the episodes (a process pool in the tuning script; serial by default). ``cache`` (ratio -> result) lets staged
    searches reuse episodes. ``per_case`` in the result is for the best point."""
    seeds = tuple(int(x) for x in (seeds if seeds is not None else (seed,)))
    cache = cache if cache is not None else {}
    ids = sorted(crops)
    todo: Dict[float, WorldConfig] = {}
    for dt in dt_grid:
        for drug in drug_grid:
            ratio = round(dt / drug, 9)
            if ratio not in cache and ratio not in todo:
                todo[ratio] = base.with_(total_drug=float(drug), dose_threshold=float(dt))
    jobs = [(ratio, c, sd) for ratio in todo for c in ids for sd in seeds]
    out = map_fn(evaluate, [(crops[c], todo[ratio], sd) for ratio, c, sd in jobs]) if jobs else []
    by: Dict[Tuple[float, str], List[Tuple[float, float]]] = {}
    for (ratio, c, _), r in zip(jobs, out):
        by.setdefault((ratio, c), []).append((float(r[0]), float(r[1])))
    for ratio in todo:
        rows = {c: by[(ratio, c)] for c in ids}
        per_case = {c: {"s": float(np.mean([x[0] for x in v])), "sprime": float(np.mean([x[1] for x in v])),
                        "s_by_seed": [x[0] for x in v], "sprime_by_seed": [x[1] for x in v]} for c, v in rows.items()}
        cache[ratio] = {"summary": summarise([per_case[c]["s"] for c in ids], [per_case[c]["sprime"] for c in ids]),
                        "per_case": per_case}
    points: List[Dict[str, Any]] = []
    for dt in dt_grid:
        for drug in drug_grid:
            ratio = round(dt / drug, 9)
            points.append({"total_drug": float(drug), "dose_threshold": float(dt), **cache[ratio]["summary"]})
    for band_name, band in (("strict", STRICT), ("widened", WIDENED)):
        ok = [p for p in points if in_band(p, band)]
        if ok:
            best = min(ok, key=lambda p: (penalty(p), p["total_drug"], p["dose_threshold"]))
            status = "ok" if band_name == "strict" else "fallback_widened_band"
            break
    else:
        # Neither band is met (the gap criterion is the usual casualty). Prefer a point whose medians sit in the widened
        # bands and take the one with the most cases showing the gap; else the point nearest the band centres.
        med = [p for p in points if WIDENED["sprime"][0] <= p["median_sprime"] <= WIDENED["sprime"][1]
               and WIDENED["s"][0] <= p["median_s"] <= WIDENED["s"][1]]
        if med:
            best = min(med, key=lambda p: (-p["gap_frac"], penalty(p), p["total_drug"], p["dose_threshold"]))
            band_name, status = "widened_medians_only", "fallback_gap_unmet"
        else:
            best = min(points, key=lambda p: (penalty(p), p["total_drug"], p["dose_threshold"]))
            band_name, status = None, "no_feasible_point"
    best_ratio = round(best["dose_threshold"] / best["total_drug"], 9)
    return {"status": status, "band": band_name, "best": best, "n_points": len(points), "cases": ids, "seeds": list(seeds),
            "points": points, "per_case": cache[best_ratio]["per_case"]}


# --- process-pool worker (cases are loaded once per process through the L1 store adapter) -----------------------

_WORKER: Dict[str, Any] = {}


def init_worker(cases_root: Optional[str] = None, repo_root: Optional[str] = None) -> None:
    _WORKER.clear()
    _WORKER.update(cases_root=cases_root, repo_root=repo_root, loaded={})


def _loaded(case_id: str):
    from antelligence.worlds.tumor_mri.store_adapter import load_case
    cache = _WORKER.setdefault("loaded", {})
    if case_id not in cache:
        cache[case_id] = load_case(case_id, root=_WORKER.get("cases_root"), repo_root=_WORKER.get("repo_root"))
    return cache[case_id]


def episode_coverages_for_case(case_id: str, cfg: WorldConfig, seed: int) -> Tuple[float, float]:
    """(S coverage, S' coverage) for a dev case id, with the stored perception regions."""
    return episode_coverages(_loaded(case_id), cfg, seed)


def timed_episode_coverages_for_case(case_id: str, cfg: WorldConfig, seed: int) -> Tuple[float, float, float, float]:
    """(S coverage, S' coverage, seconds for S, seconds for S')."""
    return episode_coverages(_loaded(case_id), cfg, seed, timed=True)  # type: ignore[return-value]
