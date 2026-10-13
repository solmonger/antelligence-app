"""v4.1 analysis (brief S5, goal 4.7 and 7): cell records -> results.json.

Reuses ``v4_metrics.mean``/``percentile``. The outcome is ``coverage`` (goal 7), a fraction in [0,1]; one "point" is 0.01.
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

from antelligence.experiments import v41_cells, v41_override
from antelligence.experiments.v4_metrics import mean

POINT = 0.01
OUTCOME = "coverage"
ALPHA = 0.05
BOOTSTRAP_SEED = 20261013
N_RESAMPLES = 9999
P2_MARGIN_POINTS = -3.0  # noninferiority: lower bound of (SW-B - FW) must exceed -3 points
S9_MIN_GAP = 0.15  # S' - S size for S9 and for the P1 gap secondary; the share of cases above it is reported, never a gate
OVERRIDE_FOLLOWED_MAX = 0.10  # worker override rate below this in BOTH SW-B and FW: P2 reads "workers followed the queen"
P2_FOLLOWED_LABEL = "workers followed the queen"
P2_FRONTIER_LABEL = "small matches frontier"
POWER = 0.80
WILCOXON_ARE = 3 / math.pi  # asymptotic relative efficiency of Wilcoxon vs t under normality
P1_TREATMENT, P1_CONTROL = "SW-B", "SW-noBoard"
P2_COMPARATOR = "FW"
P4_TREATMENT, P4_CONTROL = "SW-D", "SW-B"
SINGLE, SINGLE_PRIME = "S", "S-prime"
# Explained-null checklist (goal 7). Pooled/mean choices are documented in docs/v4.1/CELLS.md.
NULL_UNREAD_MAX = 0.50
NULL_BRIEF_AGE_MAX = 30.0
NULL_DEFER_BAND = (0.05, 0.40)  # goal 4.5 alarm band for the per-run defer rate
LIVE_CHECK_DEFER_BAND = (0.10, 0.30)  # goal 5.10 live check, a separate check
NULL_DEFER_RUNS_MAX = 0.25
NULL_ROUTER_SHARE_MAX = 0.80
NULL_ECE_MAX = 0.10

# Secondary outcomes S1 to S12 (goal 7). Comparisons are coverage differences; the rest are per-arm field means.
SECONDARY_COMPARISONS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "S1": (("SW-noQueen", "SW-B"),), "S2": (("SW-staticQueen", "SW-B"),), "S3": (("SW-Bonsai", "SW-B"),),
    "S4": (("SW-Byz-local", "SW-Byz-chain"), ("SW-Byz-local", "SW-B"), ("SW-Byz-chain", "SW-B")),
    "S10": (("SW-B-missingFLAIR", "SW-B"),),
}
SECONDARY_FIELDS: Dict[str, Tuple[str, ...]] = {
    "S5": ("perception_dice", "perception_ece", "locallook_ece", "truth_outside_crop"),
    "S6": ("defer_rate", "alarms", "audit_conformal_coverage"),
    "S7": ("citation_validity", "uncited_decisions", "unread_frac"),
    "S8": ("off_target", "seconds_to_50", "ticks_to_50"),
    "S11": ("cost_usd", "tokens_in", "tokens_out", "pod_minutes", "gas"),
}
SECONDARY_IDS = tuple(f"S{i}" for i in range(1, 13))


def thresholds() -> Dict[str, Any]:
    return {"alpha": ALPHA, "bootstrap_resamples": N_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
            "p2_margin_points": P2_MARGIN_POINTS, "s9_min_gap": S9_MIN_GAP, "override_followed_max": OVERRIDE_FOLLOWED_MAX, "power": POWER,
            "wilcoxon_are": WILCOXON_ARE, "unread_max": NULL_UNREAD_MAX, "brief_age_max_ticks": NULL_BRIEF_AGE_MAX,
            "defer_band_low": NULL_DEFER_BAND[0], "defer_band_high": NULL_DEFER_BAND[1],
            "live_check_band_low": LIVE_CHECK_DEFER_BAND[0], "live_check_band_high": LIVE_CHECK_DEFER_BAND[1],
            "defer_runs_max": NULL_DEFER_RUNS_MAX, "router_share_max": NULL_ROUTER_SHARE_MAX,
            "perception_ece_max": NULL_ECE_MAX}


# --- aggregation -------------------------------------------------------------


def usable(cells: Sequence[Mapping[str, Any]], split: Optional[str] = None) -> List[Mapping[str, Any]]:
    return [c for c in cells if not c.get("error") and c.get(OUTCOME) is not None
            and (split is None or c.get("split") == split)]


def case_means(cells: Sequence[Mapping[str, Any]], arm: str, field: str = OUTCOME) -> Dict[str, float]:
    """Mean over seeds per case for one arm (the case is the unit, goal 7)."""
    by: Dict[str, List[float]] = defaultdict(list)
    for c in cells:
        if c["arm"] == arm and c.get(field) is not None:
            by[c["case"]].append(float(c[field]))
    return {k: sum(v) / len(v) for k, v in by.items()}


def paired_case_diffs(cells: Sequence[Mapping[str, Any]], treatment: str, control: str,
                      field: str = OUTCOME) -> Tuple[List[str], np.ndarray]:
    """Per-case diffs. Arms that share cases use the same seeds, so a single-seed arm is compared with the
    control's runs on those seeds only (P2 uses the FW seed, P4 the SW-D seed)."""
    both = {c["seed"] for c in cells if c["arm"] == treatment} & {c["seed"] for c in cells if c["arm"] == control}
    cells = [c for c in cells if c["arm"] not in (treatment, control) or c["seed"] in both]
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


# --- worker override rate (pre-registered diagnostic) --------------------------------------------------------


def worker_override(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per arm: overrides / comparable decisions (pooled over runs). A decision is comparable when the squad had a queen
    assignment and the decision was not an error fallback. Arms with no assignments have rate None."""
    out: Dict[str, Any] = {"threshold": OVERRIDE_FOLLOWED_MAX, "arms": {}}
    for arm in sorted({c["arm"] for c in cells}):
        k, n, rate = v41_override.pooled_override(cells, arm)
        out["arms"][arm] = {"overrides": k, "decisions": n, "rate": rate}
    return out


def p2_reading(cells: Sequence[Mapping[str, Any]], treatment: str = P1_TREATMENT,
               comparator: str = P2_COMPARATOR) -> Dict[str, Any]:
    """Pre-registered reporting rule: if the worker override rate is below 10% in both SW-B and FW, P2 is reported as
    "workers followed the queen", not as "small matches frontier". Strictly below; a missing rate gives no label."""
    arms = worker_override(cells)["arms"]
    rates = {a: (arms.get(a) or {}).get("rate") for a in (treatment, comparator)}
    if any(r is None for r in rates.values()):
        return {"override_rates": rates, "followed_the_queen": None, "reading": None,
                "reading_note": "override rate unavailable for SW-B or FW; the rule cannot be applied"}
    followed = all(r < OVERRIDE_FOLLOWED_MAX for r in rates.values())
    return {"override_rates": rates, "followed_the_queen": followed,
            "reading": P2_FOLLOWED_LABEL if followed else P2_FRONTIER_LABEL, "reading_note": None}


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
    out.update(p2_reading(cells, treatment, comparator))
    if not len(d):
        return {**out, "mean_diff": None, "lower_bound": None, "noninferior": False, "passed": False}
    ci = bca_ci(d, 0.90)  # two-sided 90% lower end == one-sided 95% lower bound
    ni = bool(ci["low"] is not None and ci["low"] > margin)
    return {**out, "mean_diff": float(np.mean(d)), "lower_bound": ci["low"], "ci_method": ci["method"],
            "noninferior": ni, "passed": bool(gated and ni)}


def p4(cells: Sequence[Mapping[str, Any]], treatment: str = P4_TREATMENT, control: str = P4_CONTROL) -> Dict[str, Any]:
    """On-chain mode (SW-D) vs SW-B: coverage diff + 95% CI and time-to-50% overhead. No significance test."""
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
    """Normalised score (coverage - S)/(S' - S) = ratio of means: mean(arm - S) / mean(S' - S), only on cases with S' - S >= min_gap."""
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


def gap_cases(cells: Sequence[Mapping[str, Any]], min_gap: float = S9_MIN_GAP) -> Dict[str, Any]:
    """Cases with S' - S >= min_gap, from the S and S' runs of ``cells`` (case = mean over seeds). Computed after the lock."""
    s, sp = case_means(cells, SINGLE), case_means(cells, SINGLE_PRIME)
    both = sorted(set(s) & set(sp))
    kept = sorted(k for k in both if sp[k] - s[k] >= min_gap)
    return {"min_gap": min_gap, "n_cases": len(both), "n_cases_gap": len(kept),
            "share": (len(kept) / len(both)) if both else None, "cases": kept}


def p1_gap_secondary(cells: Sequence[Mapping[str, Any]], min_gap: float = S9_MIN_GAP) -> Dict[str, Any]:
    """Pre-registered secondary: P1 (same test) restricted to the evaluation cases with S' - S >= 0.15. Reported, never
    used to declare a win, never unlocks P2. The share of such cases is reported with it (it is not a gate)."""
    g = gap_cases(cells, min_gap)
    keep = set(g["cases"])
    out = p1([c for c in cells if c["case"] in keep])
    return {**out, "secondary": True, "min_gap": min_gap, "n_cases_total": g["n_cases"], "n_cases_gap": g["n_cases_gap"],
            "gap_share": g["share"]}


def pair_summary(cells: Sequence[Mapping[str, Any]], treatment: str, control: str) -> Dict[str, Any]:
    """Coverage difference (treatment - control) over shared cases, with a 95% BCa interval. Descriptive only."""
    shared, d = paired_case_diffs(cells, treatment, control)
    ci = bca_ci(d, 0.95)
    return {"treatment": treatment, "control": control, "n_cases": len(shared),
            "mean_diff": float(np.mean(d)) if len(d) else None, "ci_low": ci["low"], "ci_high": ci["high"],
            "ci_method": ci["method"]}


def secondary(cells: Sequence[Mapping[str, Any]], latency: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    """S1 to S12 as defined in goal 7. Reported, never used to declare a win."""
    arms = sorted({c["arm"] for c in cells})
    out: Dict[str, Any] = {}
    for sid in SECONDARY_IDS:
        if sid in SECONDARY_COMPARISONS:
            out[sid] = {"comparisons": {f"{t}_vs_{c}": pair_summary(cells, t, c) for t, c in SECONDARY_COMPARISONS[sid]}}
        elif sid in SECONDARY_FIELDS:
            block: Dict[str, Any] = {}
            for field in SECONDARY_FIELDS[sid]:
                block[field] = {}
                for arm in arms:
                    vals = [float(c[field]) for c in cells if c["arm"] == arm and c.get(field) is not None]
                    block[field][arm] = {"n": len(vals), "mean": mean(vals)}
            out[sid] = {"fields": block}
        elif sid == "S9":
            out[sid] = s9_normalised(cells, [a for a in arms if a not in (SINGLE, SINGLE_PRIME)])
        else:  # S12: the latency benchmark is a separate run (goal 4.7); its numbers come from latency/results.json
            out[sid] = {"available": latency is not None, "source": "latency/results.json",
                        "metrics": dict(latency or {})}
    return out


DECISION_ERROR_SHARE_MAX = 0.05  # goal 8 failure policy: above this in any arm, the sensitivity analysis is run
DECISION_ERROR_CAVEAT = (
    "P2 (SW-B vs FW noninferiority) compares arms whose decision error rates may differ: an error decision is a neutral "
    "fallback (target S, release no), and FW alone gets a corrective note on retries 2 and 3. Read P2 together with this "
    "sensitivity result.")


def decision_error_sensitivity(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Secondary: decision_error_share per arm, and (only if some arm's mean share exceeds 5%) P1 and P2 recomputed on the
    runs with no error decision. Error decisions are not separable inside a run's coverage, so the unit excluded is the
    run. Reported, never used to declare a win; the sensitivity P2 is always exploratory."""
    per_arm: Dict[str, Any] = {}
    for arm in sorted({c["arm"] for c in cells}):
        rows = [c for c in cells if c["arm"] == arm]
        shares = [float(c.get("decision_error_share") or 0.0) for c in rows]
        per_arm[arm] = {"n_runs": len(rows), "mean_share": mean(shares),
                        "decision_errors": sum(int(c.get("decision_errors") or 0) for c in rows),
                        "look_errors": sum(int(c.get("look_errors") or 0) for c in rows),
                        "runs_with_errors": sum(1 for c in rows if int(c.get("decision_errors") or 0) > 0),
                        "runs_alarmed": sum(1 for c in rows if int(c.get("decision_error_alarm") or 0))}
    triggered = any((a["mean_share"] or 0.0) > DECISION_ERROR_SHARE_MAX for a in per_arm.values())
    out: Dict[str, Any] = {"threshold": DECISION_ERROR_SHARE_MAX, "per_arm": per_arm, "triggered": triggered,
                           "caveat": DECISION_ERROR_CAVEAT, "sensitivity": None}
    if triggered:
        clean = [c for c in cells if int(c.get("decision_errors") or 0) == 0]
        r1 = p1(clean)
        r2 = {**p2(clean, r1), "status": "exploratory"}
        out["sensitivity"] = {"excluded_runs": len(cells) - len(clean), "kept_runs": len(clean), "p1": r1, "p2": r2}
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


def explained_null(cells: Sequence[Mapping[str, Any]], arm: str = P1_TREATMENT,
                   dev_cells: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """Flags that would explain a null P1 result (goal 7). Each item: value, threshold, flagged (strict >), basis.

    Bases (also in docs/v4.1/CELLS.md): unread = mean of per-run unread_frac; brief age = median of per-run
    medians; defer = share of runs whose defer_rate is outside the 5-40% band; router = mean of per-run
    top-model share; perception ECE = mean over development runs of the post-refit perception_ece.
    """
    runs = [c for c in cells if c["arm"] == arm and not c.get("error")]
    dev = [c for c in (dev_cells if dev_cells is not None else []) if c["arm"] == arm and not c.get("error")]

    def col(rows: Sequence[Mapping[str, Any]], f: str) -> List[float]:
        return [float(c[f]) for c in rows if c.get(f) is not None]

    ages, d = col(runs, "brief_age_median"), col(runs, "defer_rate")
    lo, hi = NULL_DEFER_BAND
    out_frac = (sum(1 for v in d if v < lo or v > hi) / len(d)) if d else None

    def item(value: Optional[float], threshold: float, basis: str) -> Dict[str, Any]:
        return {"value": value, "threshold": threshold, "basis": basis,
                "flagged": bool(value is not None and round(value, 12) > threshold)}

    items = {
        "unread": item(mean(col(runs, "unread_frac")), NULL_UNREAD_MAX, "mean of per-run unread_frac"),
        "brief_age_median": item(statistics.median(ages) if ages else None, NULL_BRIEF_AGE_MAX,
                                 "median of per-run median brief entry age, ticks"),
        "defer_out_of_band": item(out_frac, NULL_DEFER_RUNS_MAX, "share of runs with defer_rate outside 5-40%"),
        "router_one_model": item(mean(col(runs, "router_top_share")), NULL_ROUTER_SHARE_MAX,
                                 "mean of per-run top-model share of decisions"),
        "perception_ece": item(mean(col(dev, "perception_ece")), NULL_ECE_MAX,
                               "mean development-run perception ECE after the refit"),
    }
    return {"arm": arm, "n_runs": len(runs), "items": items, "any_flagged": any(i["flagged"] for i in items.values())}


def live_check(dev_cells: Sequence[Mapping[str, Any]], arm: str = P1_TREATMENT) -> Dict[str, Any]:
    """Separate from the explained-null band: the live-check defer band is 10-30% (goal 5.10), on mean defer rate."""
    d = [float(c["defer_rate"]) for c in dev_cells if c["arm"] == arm and not c.get("error") and c.get("defer_rate") is not None]
    rate = mean(d)
    lo, hi = LIVE_CHECK_DEFER_BAND
    return {"arm": arm, "n_runs": len(d), "mean_defer_rate": rate, "in_band": bool(rate is not None and lo <= rate <= hi)}


# --- driver ------------------------------------------------------------------


class IsolationError(ValueError):
    """Cells from a fake backend, a non-live backend or a bootstrap router reached an evaluation analysis."""


def analyze(cells: Sequence[Mapping[str, Any]], split: Optional[str] = "eval",
            latency: Optional[Mapping[str, float]] = None, *, allow_fake: bool = False) -> Dict[str, Any]:
    """Full results.json payload. Hypothesis tests use ``split`` cells; power, perception ECE and the live check use dev cells.

    Evaluation (``split == "eval"``) is strict: every eval cell must be backends_kind "live", router_kind "frozen" and carry
    no fake identity, else :class:`IsolationError`. The dev cells that feed power and the live check may be Phase 0
    calibration cells (router_kind "none") but never fake. Any other ``split`` is the explicit development / calibration
    path; fake cells are read only with ``allow_fake=True`` there, never for evaluation."""
    scoped, dev = usable(cells, split), usable(cells, "dev")
    if split == "eval":
        bad = (v41_cells.isolation_problems(scoped) +
               v41_cells.isolation_problems([c for c in dev if c.get("split") != "eval"], allow_router_none=True))
    else:
        bad = v41_cells.isolation_problems(scoped, allow_router_none=True, allow_fake=allow_fake)
    if bad:
        raise IsolationError("cells refused for analysis:\n" + "\n".join(bad[:20]))
    r1 = p1(scoped)
    counts: Dict[str, int] = defaultdict(int)
    for c in scoped:
        counts[c["arm"]] += 1
    return {"study": "v4.1", "split": split, "n_cells": len(scoped), "cells_by_arm": dict(sorted(counts.items())),
            "thresholds": thresholds(), "p1": r1, "p2": p2(scoped, r1), "p4": p4(scoped),
            "secondary": secondary(scoped, latency), "worker_override": worker_override(scoped),
            "p1_gap_secondary": p1_gap_secondary(scoped), "gap_share": gap_cases(scoped),
            "decision_errors": decision_error_sensitivity(scoped), "power": power_analysis(dev),
            "explained_null": explained_null(scoped, dev_cells=dev), "live_check": live_check(dev)}


def write_results(results: Mapping[str, Any], path: Path) -> None:
    Path(path).write_text(json.dumps(results, indent=1, sort_keys=True, allow_nan=False) + "\n")
