"""v4.3 statistics (goal section 4): the replication verdict, the headroom report and the patchiness curve.

Everything reuses the v4.2 functions (``v42_stats``: ``paired_diffs``, ``group_by_case``, ``cluster_bootstrap``, ``holm``,
``primary_verdict``). Pairing is by (case, seed); the bootstrap resamples CASES. Inputs are ``results.json`` dicts
(``{"meta": ..., "runs": [{"arm", "case_id", "seed", "split", "coverage", ...}]}``). Nothing here runs an episode.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from antelligence.experiments.v42_report import case_splits, coverage_values
from antelligence.experiments.v42_stats import (MIN_MEDIAN_DIFF, N_RESAMPLES, TOL, cluster_bootstrap, group_by_case, holm,
                                                median_of, paired_diffs, primary_verdict)
from antelligence.kernel.colony import rng

SOLO = "S-solo"
PRIME = "S'"
CONTROL_OF = {"S-marks": "S-marks-displaced", "S-talk": "S-talk-shuffled"}     # each primary arm and its own control
REPLICATION_PAIRS = 80                                                        # 40 cases x seeds 7 and 8
HEADROOM_GAP = 0.10                                                           # goal 4: S' - S-solo >= 0.10
CURVE_ARMS = ("S-marks", "S-talk")
CURVE_MEANS = ("S", SOLO, "S-marks", "S-talk", PRIME)
LEVELS = ("low", "mid", "high")


def _filter(runs: Sequence[Mapping[str, Any]], seeds: Optional[Iterable[int]] = None,
            split: Optional[str] = None) -> List[Mapping[str, Any]]:
    sd = None if seeds is None else {int(s) for s in seeds}
    return [r for r in runs if (sd is None or int(r["seed"]) in sd) and (split is None or r.get("split") == split)]


def cluster_bootstrap_mean(by_case: Mapping[str, Sequence[float]], name: str, *, n: int = N_RESAMPLES, ci: float = 0.95) -> Dict[str, Any]:
    """Case-clustered percentile bootstrap of the MEAN paired difference (same resampling as ``cluster_bootstrap``, stream
    ``("bootstrap", name)``). Returns the estimate and the central ``ci`` interval."""
    cases = sorted(by_case)
    if not cases:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_resamples": n}
    arrs = [np.asarray(by_case[c], float) for c in cases]
    idx = rng("bootstrap", name).integers(0, len(cases), size=(n, len(cases)))
    if len({len(a) for a in arrs}) == 1:
        boot = np.stack(arrs)[idx].reshape(n, -1).mean(axis=1)
    else:
        boot = np.array([np.concatenate([arrs[j] for j in row]).mean() for row in idx])
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {"mean": float(np.concatenate(arrs).mean()), "ci_low": float(np.percentile(boot, lo)),
            "ci_high": float(np.percentile(boot, hi)), "n_resamples": n}


# --- S' ceiling check ---------------------------------------------------------------------------------------------------

CEILING_ARMS = ("S", "S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled")
CEILING_SHARE = 0.95


def ceiling_check(runs: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> Dict[str, Any]:
    """S' >= every verdict arm on >= 95% of case-seeds (a sanity check, as in v4.2). Logged, never fatal."""
    by: Dict[tuple, Dict[str, float]] = {}
    for r in runs:
        by.setdefault((r["case_id"], r["seed"]), {})[r["arm"]] = float(r["coverage"])
    others = [a for a in arms if a in CEILING_ARMS]
    keys = [k for k, v in by.items() if PRIME in v]
    viol = []
    for k in sorted(keys):
        worse = [a for a in others if a in by[k] and by[k][a] > by[k][PRIME] + 1e-12]
        if worse:
            viol.append({"case_id": k[0], "seed": k[1], "s_prime": by[k][PRIME], "beaten_by": {a: by[k][a] for a in worse}})
    share = None if not keys else 1.0 - len(viol) / len(keys)
    return {"n_case_seeds": len(keys), "share_s_prime_ge_all": share, "required": CEILING_SHARE,
            "ok": share is not None and share >= CEILING_SHARE, "violations": viol}


# --- replication --------------------------------------------------------------------------------------------------------

def replication_verdict(results: Mapping[str, Any], arm: str, control: str, *, n: int = N_RESAMPLES,
                        expected_pairs: Optional[int] = None, seeds: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """The full v4.2 bar for ONE comparison (``arm`` minus S-solo) on the pooled replication pairs (40 cases x seeds 7 and 8):

    1. median paired difference vs S-solo >= 0.03;
    2. Holm-adjusted p < 0.05, where the replicated comparison is one of the two primaries and the other primary was not
       replicated, so it gets p = 1 in the Holm step (adjusted p = min(1, 2p), conservative);
    3. the same sign on dev-A and dev-B (both medians strictly positive);
    4. it beats its own control: median paired difference (arm minus control) > 0 and the interval's lower end > 0.

    ``expected_pairs`` (the script passes 80): fewer or more pooled pairs than that is reported and the verdict FAILS (a partial
    replication is not a replication). ``seeds`` restricts the runs to those seeds first."""
    runs = _filter(results["runs"], seeds)
    vals = coverage_values(runs)
    missing = [a for a in (arm, SOLO, control) if a not in vals]
    if missing:
        raise ValueError(f"replication needs arms {arm!r}, {SOLO!r} and {control!r}; missing in the results: {missing}")
    splits = case_splits(runs)
    d = paired_diffs(vals, arm, SOLO)
    boot = cluster_bootstrap(group_by_case(d), f"replication:coverage:{arm}-minus-{SOLO}", n=n)
    by_split = {s: median_of({k: v for k, v in d.items() if splits.get(k[0]) == s}) for s in ("dev-A", "dev-B")}
    cd = paired_diffs(vals, arm, control)
    cboot = cluster_bootstrap(group_by_case(cd), f"replication:coverage:{arm}-minus-{control}", n=n)
    holm_p = None
    if boot["p"] is not None:
        holm_p = holm({"replicated": float(boot["p"]), "not_replicated": 1.0})["replicated"]
    v = primary_verdict(median_diff=boot["median"], holm_p=holm_p, median_dev_a=by_split["dev-A"], median_dev_b=by_split["dev-B"],
                        control_median=cboot["median"], control_ci_low=cboot["ci_low"])
    complete = expected_pairs is None or boot["n_pairs"] == int(expected_pairs)
    verdict = {**v, "pairs_complete": bool(complete), "pass": bool(v["pass"] and complete)}
    return {"arm": arm, "control": control, "comparison": f"{arm} - {SOLO}", "control_comparison": f"{arm} - {control}",
            "seeds": sorted({int(r["seed"]) for r in runs}), "n_cases": boot["n_cases"], "n_pairs": boot["n_pairs"],
            "expected_pairs": expected_pairs, "boot": boot, "raw_p": boot["p"], "holm_p": holm_p,
            "holm_note": "the other primary was not replicated: p = 1 in the Holm step, adjusted p = min(1, 2p)",
            "by_split": by_split, "control_boot": cboot, "verdict": verdict, "pass": verdict["pass"],
            "bar": {"min_median_diff": MIN_MEDIAN_DIFF, "alpha": 0.05, "n_resamples": n}}


# --- headroom ------------------------------------------------------------------------------------------------------------

def headroom_report(results: Mapping[str, Any], *, seeds: Optional[Iterable[int]] = None, split: Optional[str] = None) -> Dict[str, Any]:
    """S' minus S-solo on the (case, seed) pairs both arms have: the share of pairs with a gap >= 0.10, the mean and median gap,
    and the same share by case (a case counts when its mean gap over the seeds is >= 0.10). Report only: no pass flag."""
    runs = _filter(results["runs"], seeds, split)
    vals = coverage_values(runs)
    if PRIME not in vals or SOLO not in vals:
        raise ValueError(f"headroom needs {SOLO!r} and {PRIME!r} in the results")
    d = paired_diffs(vals, PRIME, SOLO)
    gaps = np.array(list(d.values()), float)
    by_case = group_by_case(d)
    case_gap = {c: float(np.mean(v)) for c, v in by_case.items()}
    n = int(len(gaps))
    return {"n_pairs": n, "n_cases": len(by_case), "seeds": sorted({int(r["seed"]) for r in runs}), "split": split,
            "threshold": HEADROOM_GAP,
            "share_pairs_ge_threshold": float((gaps >= HEADROOM_GAP - TOL).mean()) if n else None,
            "share_cases_ge_threshold": float(np.mean([g >= HEADROOM_GAP - TOL for g in case_gap.values()])) if case_gap else None,
            "mean_gap": float(gaps.mean()) if n else None, "median_gap": float(np.median(gaps)) if n else None,
            "mean_s_solo": float(np.mean([vals[SOLO][k] for k in d])) if n else None,
            "mean_s_prime": float(np.mean([vals[PRIME][k] for k in d])) if n else None,
            "note": "report only; nothing is tuned from this number"}


# --- patchiness curve -----------------------------------------------------------------------------------------------------

def curve_level(results: Mapping[str, Any], *, n: int = N_RESAMPLES, level: str = "") -> Dict[str, Any]:
    """One level: arm means, and for S-marks and S-talk against S-solo the mean and the median paired difference, each with a
    case-clustered 95% bootstrap interval. No pass/fail."""
    runs = results["runs"]
    vals = coverage_values(runs)
    out: Dict[str, Any] = {"level": level, "arm_means": {a: (float(np.mean(list(vals[a].values()))) if a in vals else None) for a in CURVE_MEANS},
                           "n_runs": {a: len(vals.get(a, {})) for a in CURVE_MEANS}, "vs_s_solo": {}}
    for arm in CURVE_ARMS:
        if arm not in vals or SOLO not in vals:
            out["vs_s_solo"][arm] = None
            continue
        d = paired_diffs(vals, arm, SOLO)
        bc = group_by_case(d)
        name = f"curve:{level}:{arm}-minus-{SOLO}"
        med = cluster_bootstrap(bc, name, n=n)
        mean = cluster_bootstrap_mean(bc, name, n=n)
        out["vs_s_solo"][arm] = {"n_pairs": med["n_pairs"], "n_cases": med["n_cases"],
                                 "median": med["median"], "median_ci": [med["ci_low"], med["ci_high"]],
                                 "mean": mean["mean"], "mean_ci": [mean["ci_low"], mean["ci_high"]], "n_resamples": n}
    return out


def curve(by_level: Mapping[str, Mapping[str, Any]], *, n: int = N_RESAMPLES) -> Dict[str, Any]:
    """``by_level``: ``{"low": results, "mid": results, "high": results}``."""
    return {"levels": {lvl: curve_level(by_level[lvl], n=n, level=lvl) for lvl in LEVELS if lvl in by_level},
            "note": "patchiness curve: the sharing effect at each uptake level, with intervals; no pass/fail"}


def curve_markdown(doc: Mapping[str, Any]) -> str:
    def f(x: Any, nd: int = 4) -> str:
        return "n/a" if x is None else f"{x:.{nd}f}"

    def ci(p: Any) -> str:
        return "n/a" if p is None or p[0] is None else f"[{p[0]:.4f}, {p[1]:.4f}]"

    lv = doc["levels"]
    rows = ["## Patchiness curve (sharing effect by uptake level)", "", "No pass/fail. Paired by (case, seed); case-clustered 95% bootstrap.", "",
            "| level | " + " | ".join(f"mean {a}" for a in CURVE_MEANS) + " |", "|---|" + "---|" * len(CURVE_MEANS)]
    for name, d in lv.items():
        rows.append(f"| {name} | " + " | ".join(f(d["arm_means"][a], 3) for a in CURVE_MEANS) + " |")
    rows += ["", "| level | arm minus S-solo | pairs | mean diff | mean 95% CI | median diff | median 95% CI |", "|---|---|---|---|---|---|---|"]
    for name, d in lv.items():
        for arm in CURVE_ARMS:
            v = d["vs_s_solo"].get(arm)
            rows.append(f"| {name} | {arm} | n/a | n/a | n/a | n/a | n/a |" if v is None else
                        f"| {name} | {arm} | {v['n_pairs']} | {f(v['mean'])} | {ci(v['mean_ci'])} | {f(v['median'])} | {ci(v['median_ci'])} |")
    return "\n".join(rows) + "\n"
