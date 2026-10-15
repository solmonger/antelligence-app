"""v4.2 M1 statistics (goal sections 8, 9.3): paired differences, case-clustered bootstrap, Holm, the verdict.

Pairing is by (case, seed). The bootstrap resamples CASES (with replacement) and pools their paired differences, so the
two seeds of one case stay together. The statistic is the median paired difference. 10,000 resamples, stream
``("bootstrap", comparison_name)`` (INTERFACES section 1).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from antelligence.kernel.colony import rng

N_RESAMPLES = 10_000
MIN_MEDIAN_DIFF = 0.03
ALPHA = 0.05
TOL = 1e-12
PRIMARY = {"a": ("S-marks", "S-solo", "S-marks-displaced"), "b": ("S-talk", "S-solo", "S-talk-shuffled")}

Key = Tuple[str, int]


def paired_diffs(values: Mapping[str, Mapping[Key, float]], a: str, b: str) -> Dict[Key, float]:
    """``a - b`` over the (case, seed) keys both arms have."""
    va, vb = values[a], values[b]
    return {k: float(va[k]) - float(vb[k]) for k in sorted(set(va) & set(vb))}


def group_by_case(diffs: Mapping[Key, float]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for (case, _seed), d in sorted(diffs.items()):
        out.setdefault(case, []).append(float(d))
    return out


def cluster_bootstrap(by_case: Mapping[str, Sequence[float]], name: str, *, n: int = N_RESAMPLES,
                      ci: float = 0.95) -> Dict[str, Any]:
    """Case-clustered percentile bootstrap of the median paired difference. Returns the estimate, the central ``ci``
    interval and a two-sided p-value for "median = 0": ``min(1, 2 * min(P(b <= 0), P(b >= 0)))`` with the +1 correction
    ``(count + 1) / (n + 1)`` so p is never exactly 0."""
    cases = sorted(by_case)
    if not cases:
        return {"n_cases": 0, "n_pairs": 0, "median": None, "ci_low": None, "ci_high": None, "p": None, "n_resamples": n}
    arrs = [np.asarray(by_case[c], float) for c in cases]
    g = rng("bootstrap", name)
    idx = g.integers(0, len(cases), size=(n, len(cases)))
    if len({len(a) for a in arrs}) == 1:
        boot = np.median(np.stack(arrs)[idx].reshape(n, -1), axis=1)
    else:
        boot = np.array([np.median(np.concatenate([arrs[j] for j in row])) for row in idx])
    pooled = np.concatenate(arrs)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    le, ge = int((boot <= 0).sum()), int((boot >= 0).sum())
    p = min(1.0, 2.0 * min((le + 1) / (n + 1), (ge + 1) / (n + 1)))
    return {"n_cases": len(cases), "n_pairs": int(len(pooled)), "median": float(np.median(pooled)),
            "mean": float(pooled.mean()), "ci_low": float(np.percentile(boot, lo)), "ci_high": float(np.percentile(boot, hi)),
            "p": float(p), "n_resamples": n}


def holm(pvals: Mapping[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def median_of(diffs: Mapping[Key, float]) -> Optional[float]:
    return float(np.median(list(diffs.values()))) if diffs else None


def primary_verdict(*, median_diff: Optional[float], holm_p: Optional[float], median_dev_a: Optional[float],
                    median_dev_b: Optional[float], control_median: Optional[float],
                    control_ci_low: Optional[float]) -> Dict[str, Any]:
    """Goal 9.3 for ONE primary comparison. Passes only if all four hold:
    1. median paired difference >= 0.03;
    2. Holm-adjusted p < 0.05;
    3. dev-A and dev-B medians have the same sign (both strictly positive; a zero or missing median fails);
    4. it beats its own control: median paired difference (arm minus control) > 0 with the interval's lower end > 0."""
    def ok(x: Optional[float]) -> bool:
        return x is not None
    c1 = ok(median_diff) and median_diff >= MIN_MEDIAN_DIFF - TOL
    c2 = ok(holm_p) and holm_p < ALPHA
    same = ok(median_dev_a) and ok(median_dev_b) and ((median_dev_a > 0 and median_dev_b > 0) or (median_dev_a < 0 and median_dev_b < 0))
    c3 = bool(same and median_dev_a > 0)
    c4 = ok(control_median) and ok(control_ci_low) and control_median > 0 and control_ci_low > 0
    return {"median_ge_0.03": bool(c1), "holm_p_lt_0.05": bool(c2), "same_sign_dev_a_b": bool(same),
            "dev_sign_positive": c3, "beats_control": bool(c4), "pass": bool(c1 and c2 and c3 and c4)}


def analyse(values: Mapping[str, Mapping[Key, float]], splits: Mapping[str, str], *, n: int = N_RESAMPLES,
            name: str = "coverage") -> Dict[str, Any]:
    """Full goal-9.3 analysis on one metric (coverage). ``values[arm][(case, seed)]``; ``splits[case]`` is ``"dev-A"`` or
    ``"dev-B"``. Primary a/b, their controls, Holm over the two primaries, the secondary S-talk - S-marks, and
    "sharing helps scripts" (at least one primary passes)."""
    prim: Dict[str, Any] = {}
    for tag, (arm, solo, ctrl) in PRIMARY.items():
        d = paired_diffs(values, arm, solo)
        boot = cluster_bootstrap(group_by_case(d), f"{name}:{arm}-minus-{solo}", n=n)
        by_split = {s: median_of({k: v for k, v in d.items() if splits.get(k[0]) == s}) for s in ("dev-A", "dev-B")}
        cd = paired_diffs(values, arm, ctrl)
        cboot = cluster_bootstrap(group_by_case(cd), f"{name}:{arm}-minus-{ctrl}", n=n)
        prim[tag] = {"comparison": f"{arm} - {solo}", "control": f"{arm} - {ctrl}", "boot": boot, "by_split": by_split,
                     "control_boot": cboot}
    adj = holm({t: p["boot"]["p"] for t, p in prim.items() if p["boot"]["p"] is not None})
    for t, p in prim.items():
        p["holm_p"] = adj.get(t)
        p["verdict"] = primary_verdict(
            median_diff=p["boot"]["median"], holm_p=p["holm_p"], median_dev_a=p["by_split"]["dev-A"],
            median_dev_b=p["by_split"]["dev-B"], control_median=p["control_boot"]["median"],
            control_ci_low=p["control_boot"]["ci_low"])
    sd = paired_diffs(values, "S-talk", "S-marks")
    return {"primary": prim, "secondary": {"comparison": "S-talk - S-marks",
                                            "boot": cluster_bootstrap(group_by_case(sd), f"{name}:S-talk-minus-S-marks", n=n)},
            "sharing_helps_scripts": any(p["verdict"]["pass"] for p in prim.values())}
