"""v4.3 measurement (goal section 5): everything in v4.2 section 8 (re-exported from ``v42_metrics``), plus

* probes per run and probes saved by others' uptake marks (``script_counters``: the B2 builder names them ``probes`` and
  ``probes_saved_by_marks``; a missing key counts as 0);
* drug released on poor-uptake tissue (u < 0.4), from the ``dosed`` marks of the bundle against the case's uptake field;
* the coverage gap to S' split by cause: false-alarm releases (drug placed more than 4 mm from the tumour), poor-patch releases
  (drug placed at u < 0.4), unreleased drug, and the remainder;
* the effect rate of uptake marks, separately.

Inputs are plain dicts. Scorer-side lookups (distance to the tumour, u) come in as callables from an offline analysis; this
module imports no world and no store, and writes nothing.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from antelligence.experiments.v42_metrics import (Timings, claim_accuracy_by_ant, effect_study, effect_summary,  # noqa: F401
                                                  honesty, mark_quality, pearson, run_metrics, sample_items, sample_runs)

POOR_U = 0.4                    # drug released where true u is below this is "on poor-uptake tissue"
FAR_MM = 4.0                    # a release farther than this from the tumour is a false alarm
UPTAKE_KINDS = ("good_uptake", "slow_uptake")
N_SQUADS = 8
ASSUMPTION = ("This split is an estimate and an association, not a causal result. It assumes the colony arms' tumour-side yield "
              "(coverage per unit of drug released on the tumour side) equals S''s. S has no saved decision log, so it enters "
              "through its totals only (unreleased drug).")
CAUSES = ("false_alarm", "poor_patch", "unreleased")


# --- probes ---------------------------------------------------------------------------------------------------------------

def probe_counts(script_counters: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    """``probes`` and ``probes_saved_by_marks`` from a run's ``script_counters`` (a missing key, or no counters, is 0)."""
    c = script_counters or {}
    return {"probes": int(c.get("probes", 0) or 0), "probes_saved_by_marks": int(c.get("probes_saved_by_marks", 0) or 0)}


def run_metrics_v43(metrics: Mapping[str, Any], colony: Optional[Mapping[str, Any]] = None, *,
                    script_counters: Optional[Mapping[str, Any]] = None, total_drug: Optional[float] = None,
                    accuracy: Optional[Mapping[str, Mapping[str, int]]] = None, voxel_mm: float = 1.0) -> Dict[str, Any]:
    """The v4.2 per-run record plus ``probes``, ``probes_saved_by_marks``, ``released_total``, ``total_drug``, ``unreleased``."""
    out = run_metrics(metrics, colony, accuracy=accuracy, voxel_mm=voxel_mm)
    out.update(probe_counts(script_counters))
    released = float(metrics.get("released_total", 0.0))
    out["released_total"] = released
    out["total_drug"] = None if total_drug is None else float(total_drug)
    out["unreleased"] = None if total_drug is None else max(0.0, float(total_drug) - released)
    return out


# --- where the drug went -----------------------------------------------------------------------------------------------------

def release_pools(colony: Mapping[str, Any], total_drug: float, *, dist_mm_at: Callable[[Sequence[int]], float],
                  u_at: Callable[[Sequence[int]], float], n_squads: int = N_SQUADS) -> Dict[str, Any]:
    """Masses released by class, from the ``dosed`` marks (each mark's ``amount_milli / 1000`` is the fraction of one squad's
    share ``total_drug / n_squads``, at ``pos_vox``).

    * ``false_alarm``: farther than 4 mm from the tumour (``dist_mm_at`` is 0 inside it, +inf outside the crop);
    * ``poor_patch``: true u < 0.4 at the voxel (u is 1 outside the tumour, so this is tumour tissue by construction);
    * ``other``: everything else (good or middling tissue, or within 4 mm of the tumour).
    The classes are disjoint (a voxel with u < 0.4 is inside the tumour, hence not a false alarm; a voxel where u is unknown
    counts by distance only). ``released_from_marks`` is their sum; the caller checks it against ``released_total``."""
    share = float(total_drug) / float(n_squads)
    far = poor = other = 0.0
    n = n_far = n_poor = 0
    for m in colony.get("marks", ()):
        if m["mark_kind"] != "dosed":
            continue
        mass = m["amount_milli"] / 1000.0 * share
        pos = tuple(int(c) for c in m["pos_vox"])
        n += 1
        if float(dist_mm_at(pos)) > FAR_MM:
            far += mass
            n_far += 1
        elif float(u_at(pos)) < POOR_U:
            poor += mass
            n_poor += 1
        else:
            other += mass
    return {"false_alarm": far, "poor_patch": poor, "other": other, "released_from_marks": far + poor + other,
            "n_releases": n, "n_false_alarm": n_far, "n_poor_patch": n_poor}


def gap_by_cause(*, cov_arm: float, cov_prime: float, prime_released: float, total_drug: float, false_alarm: Optional[float],
                 poor_patch: Optional[float], unreleased: float) -> Dict[str, Any]:
    """Split ``gap = S' - arm`` (coverage). Each estimated cause is ``(pool / total_drug) * c`` where
    ``c = cov_prime / (prime_released / total_drug)`` is S''s coverage per unit of the budget it released. The remainder is the
    gap minus the three estimates, so the four parts always sum to the gap. A pool that is None (S has no log) is left out
    and its estimate is None; the remainder then holds it."""
    gap = float(cov_prime) - float(cov_arm)
    share_prime = float(prime_released) / float(total_drug) if total_drug else 0.0
    if share_prime <= 0.0:
        return {"gap": gap, "per_released_drug": None, "false_alarm": None, "poor_patch": None, "unreleased": None, "remainder": gap}
    c = float(cov_prime) / share_prime

    def est(pool: Optional[float]) -> Optional[float]:
        return None if pool is None else float(pool) / float(total_drug) * c

    parts = {"false_alarm": est(false_alarm), "poor_patch": est(poor_patch), "unreleased": est(unreleased)}
    rem = gap - sum(v for v in parts.values() if v is not None)
    return {"gap": gap, "per_released_drug": c, **parts, "remainder": rem}


def summarise_gaps(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Mean over runs of each part (None parts skipped; the count is reported) and the assumption, stated once."""
    out: Dict[str, Any] = {"n_runs": len(rows), "assumption": ASSUMPTION}
    for k in ("gap", "false_alarm", "poor_patch", "unreleased", "remainder"):
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
        out[f"n_{k}"] = len(v)
    return out


# --- message effect, uptake marks apart -----------------------------------------------------------------------------------------

def tag_mark_kinds(rows: Iterable[Mapping[str, Any]], marks: Mapping[tuple, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Add ``mark_kind`` to each effect row about a mark (``marks[(case, seed)]`` = that baseline run's mark records)."""
    out = []
    for r in rows:
        row = dict(r)
        item = row.get("item") or {}
        if item.get("kind") == "mark":
            recs = marks.get((row["case_id"], int(row["seed"])), ())
            if 0 <= int(item["id"]) < len(recs):
                row["mark_kind"] = recs[int(item["id"])]["mark_kind"]
        out.append(row)
    return out


def effect_by_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """``all``, ``uptake_marks`` (good_uptake and slow_uptake), ``other_marks`` and ``messages`` through ``effect_summary``."""
    def pick(f: Callable[[Mapping[str, Any]], bool]) -> Dict[str, Any]:
        return effect_summary([r for r in rows if f(r)])

    is_mark = lambda r: (r.get("item") or {}).get("kind") == "mark"  # noqa: E731
    return {"all": effect_summary(rows), "uptake_marks": pick(lambda r: is_mark(r) and r.get("mark_kind") in UPTAKE_KINDS),
            "other_marks": pick(lambda r: is_mark(r) and r.get("mark_kind") not in UPTAKE_KINDS),
            "messages": pick(lambda r: (r.get("item") or {}).get("kind") == "message")}
