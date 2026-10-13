"""v4.1 analysis (brief S5, goal 4.7 and 7): cell records -> results.json.

Reuses ``v4_metrics.mean``/``percentile``. Scores are fractions in [0,1]; one "point" is 0.01.
Pre-registered choices that the brief leaves open are named constants below and are printed in the
preregistration (``scripts/v41_prereg.py``) and in results.json under ``thresholds``.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps

from antelligence.experiments.v4_metrics import mean

POINT = 0.01
ALPHA = 0.05
BOOTSTRAP_SEED = 20261013
N_RESAMPLES = 9999
P2_MARGIN_POINTS = -3.0  # noninferiority: lower bound of (SW-B - FW) must exceed -3 points
S9_MIN_GAP = 0.15
POWER = 0.80
WILCOXON_ARE = 3 / math.pi  # asymptotic relative efficiency of Wilcoxon vs t under normality
P1_TREATMENT, P1_CONTROL = "SW-B", "SW-noBoard"
P2_COMPARATOR = "FW"
P4_TREATMENT, P4_CONTROL = "SW-D", "SW-B"
SINGLE, SINGLE_PRIME = "S", "S-prime"
# Explained-null checklist (goal 8). The deferral band is an assumption fixed here and in the prereg.
NULL_UNREAD_MAX = 0.50
NULL_BRIEF_AGE_MAX = 30.0
NULL_DEFER_BAND = (0.05, 0.60)
NULL_DEFER_RUNS_MAX = 0.25
NULL_ROUTER_SHARE_MAX = 0.80
NULL_ECE_MAX = 0.10

SECONDARY: Dict[str, Optional[str]] = {
    "S1": "wall_s", "S2": "ticks_to_50", "S3": "seconds_to_50", "S4": "queen_calls", "S5": "board_size",
    "S6": "anchors", "S7": "visibility_latency_ms", "S8": "coverage", "S9": None, "S10": "off_target",
    "S11": "defer_rate", "S12": "citations",
}


def thresholds() -> Dict[str, Any]:
    return {"alpha": ALPHA, "bootstrap_resamples": N_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
            "p2_margin_points": P2_MARGIN_POINTS, "s9_min_gap": S9_MIN_GAP, "power": POWER,
            "wilcoxon_are": WILCOXON_ARE, "unread_max": NULL_UNREAD_MAX, "brief_age_max_ticks": NULL_BRIEF_AGE_MAX,
            "defer_band_low": NULL_DEFER_BAND[0], "defer_band_high": NULL_DEFER_BAND[1],
            "defer_runs_max": NULL_DEFER_RUNS_MAX, "router_share_max": NULL_ROUTER_SHARE_MAX,
            "perception_ece_max": NULL_ECE_MAX}


# --- aggregation -------------------------------------------------------------


def usable(cells: Sequence[Mapping[str, Any]], split: Optional[str] = None) -> List[Mapping[str, Any]]:
    return [c for c in cells if not c.get("error") and c.get("score") is not None
            and (split is None or c.get("split") == split)]


def case_means(cells: Sequence[Mapping[str, Any]], arm: str, field: str = "score") -> Dict[str, float]:
    """Mean over seeds per case for one arm."""
    by: Dict[str, List[float]] = defaultdict(list)
    for c in cells:
        if c["arm"] == arm and c.get(field) is not None:
            by[c["case"]].append(float(c[field]))
    return {k: sum(v) / len(v) for k, v in by.items()}


def paired_case_diffs(cells: Sequence[Mapping[str, Any]], treatment: str, control: str,
                      field: str = "score") -> Tuple[List[str], np.ndarray]:
    t, c = case_means(cells, treatment, field), case_means(cells, control, field)
    shared = sorted(set(t) & set(c))
    return shared, np.array([t[k] - c[k] for k in shared], dtype=float)


def bca_ci(diffs: np.ndarray, level: float = 0.95, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """BCa bootstrap CI of the mean (scipy.stats.bootstrap, fixed seed). Falls back, and says so, when BCa is undefined."""
    n = len(diffs)
    if n == 0:
        return {"low": None, "high": None, "method": "none"}
    m = float(np.mean(diffs))
    if n < 2 or float(np.ptp(diffs)) == 0.0:
        return {"low": m, "high": m, "method": "degenerate"}
    for method in ("BCa", "percentile"):
        try:
            res = sps.bootstrap((diffs,), np.mean, confidence_level=level, method=method, n_resamples=N_RESAMPLES,
                                random_state=np.random.default_rng(seed))
            lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
            if not (math.isnan(lo) or math.isnan(hi)):
                return {"low": lo, "high": hi, "method": method}
        except (ValueError, FloatingPointError):
            continue
    return {"low": m, "high": m, "method": "degenerate"}


def wilcoxon_p(diffs: np.ndarray) -> Optional[float]:
    if len(diffs) == 0:
        return None
    if not np.any(diffs != 0):
        return 1.0
    return float(sps.wilcoxon(diffs, alternative="two-sided").pvalue)


# --- hypotheses --------------------------------------------------------------


def p1(cells: Sequence[Mapping[str, Any]], treatment: str = P1_TREATMENT, control: str = P1_CONTROL) -> Dict[str, Any]:
    """Two-sided Wilcoxon on case-level paired diffs + 95% BCa CI. Passes only if p < 0.05 and mean > 0."""
    shared, d = paired_case_diffs(cells, treatment, control)
    out: Dict[str, Any] = {"treatment": treatment, "control": control, "n_cases": len(shared)}
    if not len(d):
        return {**out, "mean_diff": None, "ci_low": None, "ci_high": None, "ci_method": "none", "p_value": None,
                "passed": False}
    ci = bca_ci(d, 0.95)
    p = wilcoxon_p(d)
    m = float(np.mean(d))
    return {**out, "mean_diff": m, "ci_low": ci["low"], "ci_high": ci["high"], "ci_method": ci["method"],
            "p_value": p, "passed": bool(p is not None and p < ALPHA and m > 0)}


def p2(cells: Sequence[Mapping[str, Any]], p1_result: Mapping[str, Any], treatment: str = P1_TREATMENT,
       comparator: str = P2_COMPARATOR) -> Dict[str, Any]:
    """Noninferiority of SW-B vs FW: one-sided 95% lower bound of the mean diff > -3 points. Gated on P1."""
    shared, d = paired_case_diffs(cells, treatment, comparator)
    margin = P2_MARGIN_POINTS * POINT
    gated = bool(p1_result.get("passed"))
    out: Dict[str, Any] = {"treatment": treatment, "comparator": comparator, "n_cases": len(shared),
                           "margin": margin, "p1_passed": gated, "status": "confirmatory" if gated else "exploratory"}
    if not len(d):
        return {**out, "mean_diff": None, "lower_bound": None, "noninferior": False, "passed": False}
    ci = bca_ci(d, 0.90)  # two-sided 90% lower end == one-sided 95% lower bound
    ni = bool(ci["low"] is not None and ci["low"] > margin)
    return {**out, "mean_diff": float(np.mean(d)), "lower_bound": ci["low"], "ci_method": ci["method"],
            "noninferior": ni, "passed": bool(gated and ni)}


def p4(cells: Sequence[Mapping[str, Any]], treatment: str = P4_TREATMENT, control: str = P4_CONTROL) -> Dict[str, Any]:
    """On-chain mode (SW-D) vs SW-B: score diff + 95% CI and time-to-50% overhead. No significance test."""
    shared, d = paired_case_diffs(cells, treatment, control)
    ci = bca_ci(d, 0.95)
    out: Dict[str, Any] = {"treatment": treatment, "control": control, "n_cases": len(shared),
                           "mean_diff": float(np.mean(d)) if len(d) else None, "ci_low": ci["low"], "ci_high": ci["high"],
                           "ci_method": ci["method"], "significance_tested": False}
    for field, key in (("seconds_to_50", "seconds_to_50_overhead"), ("ticks_to_50", "ticks_to_50_overhead")):
        _, o = paired_case_diffs(cells, treatment, control, field)
        oc = bca_ci(o, 0.95)
        out[key] = {"n_cases": len(o), "mean": float(np.mean(o)) if len(o) else None,
                    "ci_low": oc["low"], "ci_high": oc["high"], "ci_method": oc["method"]}
    return out


# --- secondary, S9, power, explained null ------------------------------------


def s9_normalised(cells: Sequence[Mapping[str, Any]], arms: Sequence[str], min_gap: float = S9_MIN_GAP) -> Dict[str, Any]:
    """Normalised score = ratio of means: mean(arm - S) / mean(S' - S), only on cases with S' - S >= min_gap."""
    s, sp = case_means(cells, SINGLE), case_means(cells, SINGLE_PRIME)
    kept = sorted(k for k in set(s) & set(sp) if sp[k] - s[k] >= min_gap)
    out: Dict[str, Any] = {"n_cases_gap": len(kept), "min_gap": min_gap, "arms": {}}
    if not kept:
        return out
    denom = sum(sp[k] - s[k] for k in kept) / len(kept)
    for arm in arms:
        a = case_means(cells, arm)
        ks = [k for k in kept if k in a]
        out["arms"][arm] = ({"n_cases": len(ks), "ratio": (sum(a[k] - s[k] for k in ks) / len(ks)) / (
            sum(sp[k] - s[k] for k in ks) / len(ks))} if ks else {"n_cases": 0, "ratio": None})
    out["mean_gap"] = denom
    return out


def secondary(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    arms = sorted({c["arm"] for c in cells})
    out: Dict[str, Any] = {}
    for sid, field in SECONDARY.items():
        if field is None:
            out[sid] = s9_normalised(cells, [a for a in arms if a not in (SINGLE, SINGLE_PRIME)])
            continue
        out[sid] = {"field": field, "arms": {}}
        for arm in arms:
            vals = [float(c[field]) for c in cells if c["arm"] == arm and c.get(field) is not None]
            out[sid]["arms"][arm] = {"n": len(vals), "mean": mean(vals)}
    return out


def minimum_detectable_diff(sd: float, n: int, *, alpha: float = ALPHA, power: float = POWER,
                            are: float = WILCOXON_ARE) -> Optional[float]:
    """Smallest true paired mean diff the Wilcoxon test detects with the given power.

    Normal approximation: (z_{1-alpha/2} + z_{power}) * sd / sqrt(n * ARE). The ARE factor (3/pi, ~0.955)
    inflates the t-test requirement to cover the Wilcoxon test's efficiency loss under normality.
    """
    if n <= 0 or sd is None:
        return None
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return float(z * sd / math.sqrt(n * are))


def power_analysis(cells: Sequence[Mapping[str, Any]], n_eval: int = 80) -> Dict[str, Any]:
    dev = usable(cells, "dev")
    shared, d = paired_case_diffs(dev, P1_TREATMENT, P1_CONTROL)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else None
    return {"dev_cases": len(shared), "dev_paired_sd": sd, "eval_cases": n_eval,
            "mdd_p1": minimum_detectable_diff(sd, n_eval) if sd is not None else None,
            "method": "Wilcoxon ARE-adjusted normal approximation"}


def explained_null(cells: Sequence[Mapping[str, Any]], arm: str = P1_TREATMENT) -> Dict[str, Any]:
    """Flags that would explain a null P1 result. Each entry: value, threshold, flagged (strict >)."""
    runs = [c for c in cells if c["arm"] == arm and not c.get("error")]

    def col(f: str) -> List[float]:
        return [float(c[f]) for c in runs if c.get(f) is not None]

    unread, age, share, ece = (mean(col("unread_frac")), (statistics.median(col("brief_age_median"))
                               if col("brief_age_median") else None), mean(col("router_top_share")),
                               mean(col("perception_ece")))
    d = col("defer_rate")
    lo, hi = NULL_DEFER_BAND
    out_frac = (sum(1 for v in d if v < lo or v > hi) / len(d)) if d else None

    def item(value: Optional[float], threshold: float) -> Dict[str, Any]:
        return {"value": value, "threshold": threshold, "flagged": bool(value is not None and value > threshold)}

    items = {"unread": item(unread, NULL_UNREAD_MAX), "brief_age_median": item(age, NULL_BRIEF_AGE_MAX),
             "defer_out_of_band": item(out_frac, NULL_DEFER_RUNS_MAX), "router_one_model": item(share, NULL_ROUTER_SHARE_MAX),
             "perception_ece": item(ece, NULL_ECE_MAX)}
    return {"arm": arm, "n_runs": len(runs), "items": items, "any_flagged": any(i["flagged"] for i in items.values())}


# --- driver ------------------------------------------------------------------


def analyze(cells: Sequence[Mapping[str, Any]], split: Optional[str] = "eval") -> Dict[str, Any]:
    """Full results.json payload. Hypothesis tests use ``split`` cells; power uses dev cells."""
    scoped = usable(cells, split)
    r1 = p1(scoped)
    counts: Dict[str, int] = defaultdict(int)
    for c in scoped:
        counts[c["arm"]] += 1
    return {"study": "v4.1", "split": split, "n_cells": len(scoped), "cells_by_arm": dict(sorted(counts.items())),
            "thresholds": thresholds(), "p1": r1, "p2": p2(scoped, r1), "p4": p4(scoped),
            "secondary": secondary(scoped), "power": power_analysis(usable(cells)),
            "explained_null": explained_null(scoped)}


def write_results(results: Mapping[str, Any], path: Path) -> None:
    Path(path).write_text(json.dumps(results, indent=1, sort_keys=True, allow_nan=False) + "\n")
