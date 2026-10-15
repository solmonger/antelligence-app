#!/usr/bin/env python
"""v4.3 diagnostics: why did sharing help or not help?  Offline, read-only analysis of saved verdict runs.

    python scripts/v43_diagnose.py --run-dir runs/verdict-mid --cases-root /path/to/cases --out diag-mid \\
        [--extra-run-dir runs/other] [--limit 0] [--level mid]

A v4.3 copy of the v4.2 diagnostics. The generic log readers (decisions, positions, mark replay, divergence, shared inputs,
release anatomy, look quality, beliefs) are the v4.2 functions, imported as library code; this script does not run a v4.2 script.
What is new, for every colony arm:

* probes per run and probes saved by others' uptake marks (``script_counters``; a missing key is 0);
* drug released on poor-uptake tissue (true u < 0.4), from the ``dosed`` marks of the bundle (each ``amount_milli / 1000`` is the
  fraction of one squad's share ``total_drug / 8``, at ``pos_vox``), against the uptake field rebuilt from the case store with
  ``foraging.foraging_uptake`` (offline, same level as the run);
* the coverage gap to S' split by cause: false-alarm releases (drug placed more than 4 mm from the tumour), poor-patch releases
  (u < 0.4), unreleased drug, and the remainder. Each cause is the pool's share of the total drug times S''s coverage per released
  drug (S' coverage divided by S''s released share of the budget). S has no saved log, so it enters through its totals only.
  The split is an association and assumes the colony arms' tumour-side yield equals S''s (stated in the output);
* the accuracy of the uptake marks against true u;
* the realised mean u per sub-region and level (``docs/research/v4.3-20261015/uptake/realised_u.json``, B1's file, copied in).

Reads ``results.json`` and each arm / case / seed ``record.json`` and ``bundle.json``. Writes ONLY ``<out>/diagnostics_numbers.json``
(``--out`` is required and may not be inside a v4.1 / v4.2 folder). It never runs an episode and never changes a saved result.
Scorer-side truth (tumour mask, uptake field) is read from the dev-case store INSIDE this offline analysis only; only aggregate
numbers are written, no voxel coordinate tied to truth leaves the script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments.v43_drive import OutRefused, assert_out_allowed  # noqa: E402
from antelligence.experiments.v43_metrics import (ASSUMPTION, POOR_U, UPTAKE_KINDS, gap_by_cause, probe_counts, release_pools,  # noqa: E402
                                                  summarise_gaps)
from antelligence.worlds.tumor_mri.data import select_candidates  # noqa: E402
from antelligence.worlds.tumor_mri.foraging import foraging_uptake  # noqa: E402

REALISED_U = "docs/research/v4.3-20261015/uptake/realised_u.json"


def _load_v42_readers():
    """The v4.2 log readers, as library code (the module defines functions only; nothing runs on import)."""
    # NOTE: the imported v4.2 script must stay free of import-time side effects (no file written, no network, no main()). It is
    # loaded by path, so it is not covered by the freeze hash; tests/v4_3/c/test_freeze_fix.py imports this script in a
    # subprocess from an empty directory and fails if a file appears there or under docs/research/v4.2-20261014.
    spec = importlib.util.spec_from_file_location("v43_diagnose_readers", REPO / "scripts" / "v42_diagnose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_v42_readers()
SOLO, PRIME = R.SOLO, "S'"
SHARING = ("S-marks", "S-talk", "S-marks-perfect")
CONTROLS = ("S-marks-displaced", "S-talk-shuffled", "S-marks-quiet")
COLONY_ARMS = (SOLO, "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S-marks-perfect", "S-marks-quiet")
ALL_ARMS = ("S", *COLONY_ARMS, PRIME)
POSITIONS_MOVED = ("S-marks-displaced",)        # their marks sit at random places on purpose: position-based pools are meaningless


# --- cases (scorer side, offline only) ---------------------------------------------------------------------------------------

class CaseInfo43:
    """The v4.2 ``CaseInfo`` interface over the v4.3 world: the uptake field is the foraging one, at ``level``."""

    def __init__(self, case_id: str, cases_root: Optional[Path], facts: Mapping[str, Any], level: str,
                 loader: Optional[Callable[[str, Optional[Path]], Any]] = None) -> None:
        from scipy import ndimage
        from antelligence.experiments.v42_harness import load_dev_case
        from antelligence.worlds.tumor_mri.physics import washout_field
        c = loader(case_id, cases_root) if loader is not None else load_dev_case(case_id, root=cases_root)
        f = dict(facts or {})
        ch = c.scorer_channels or {}
        self.u = None
        if f.get("f2") and ch.get("tc") is not None and ch.get("et") is not None:
            self.u = foraging_uptake(c.crop.truth, ch["tc"], ch["et"], c.crop.case_id, level)
        self.delta = None
        if f.get("f3"):
            self.delta = washout_field(c.crop.truth, c.crop.case_id, float(f["f3_decay"]), float(f["f3_share"]), c.crop.voxel_mm)
        self.f3_decay = float(f.get("f3_decay") or 0.0)
        self.crop, self.regions = c.crop, list(c.regions or [])
        self.truth = c.crop.truth.astype(bool)
        self.dist_mm = ndimage.distance_transform_edt(~self.truth) * R.VOX_MM          # 0 inside the tumour
        self.cands = {cd.label: cd for cd in select_candidates(self.regions)}
        self.zones = []
        import math
        from antelligence.kernel.colony.quorum import ZoneSpec
        for r in self.regions:
            rad = (3.0 * r.volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0) + 6.0
            ctr = tuple(r.centroid_vox)
            self.zones.append(ZoneSpec(r.region_id, ctr, (lambda ctr, rad: lambda p: R.VOX_MM * math.dist(p, ctr) <= rad * (1 + 1e-9))(ctr, rad)))
        self.shape = self.truth.shape

    def inside(self, pos: Sequence[int]) -> bool:
        return all(0 <= p < s for p, s in zip(pos, self.shape))

    def truth_at(self, pos: Sequence[int]) -> Optional[bool]:
        return bool(self.truth[tuple(pos)]) if self.inside(pos) else None

    def dist_at(self, pos: Sequence[int]) -> float:
        return float(self.dist_mm[tuple(pos)]) if self.inside(pos) else float("inf")

    def u_at(self, pos: Sequence[int]) -> float:
        """True uptake at the voxel (1.0 outside the tumour, outside the crop, or when F2 is off)."""
        return float(self.u[tuple(pos)]) if self.u is not None and self.inside(pos) else 1.0

    def slow_tissue(self, pos: Sequence[int]) -> bool:
        """Poor-uptake tissue (u < 0.4) in the v4.3 world."""
        return self.u is not None and self.inside(pos) and float(self.u[tuple(pos)]) < POOR_U

    def fast_washout(self, pos: Sequence[int]) -> bool:
        return self.delta is not None and self.inside(pos) and float(self.delta[tuple(pos)]) >= 0.5 * self.f3_decay > 0

    def fact_shares(self) -> Dict[str, float]:
        t = self.truth
        return {"tumour_voxels": int(t.sum()),
                "share_tumour_poor_uptake": float((self.u[t] < POOR_U).mean()) if self.u is not None and t.any() else 0.0,
                "share_tumour_fast_washout": float((self.delta[t] >= 0.5 * self.f3_decay).mean()) if self.delta is not None and t.any() else 0.0}

    def scan_p(self, pos: Sequence[int]) -> float:
        return float(self.crop.p_tumour[tuple(pos)])


# --- the v4.3 additions -----------------------------------------------------------------------------------------------------------

def uptake_mark_accuracy(col: Mapping[str, Any], case: CaseInfo43) -> Dict[str, int]:
    """How right are the uptake marks (offline check against true u)."""
    c: Counter = Counter()
    for m in col["marks"]:
        k = m["mark_kind"]
        if k not in UPTAKE_KINDS:
            continue
        u = case.u_at(m["pos_vox"])
        c["n_" + k] += 1
        c[f"{k}_true_u_ge_0.7"] += u >= 0.7
        c[f"{k}_true_u_middling"] += POOR_U <= u < 0.7
        c[f"{k}_true_u_lt_0.4"] += u < POOR_U
    return dict(c)


def run_pools(run: Mapping[str, Any], case: CaseInfo43) -> Dict[str, Any]:
    rec, col = run["rec"], run["col"]
    pools = release_pools(col, rec["total_drug"], dist_mm_at=case.dist_at, u_at=case.u_at)
    pools["released_total"] = float(rec["metrics"]["released_total"])
    pools["accounting_error"] = pools["released_from_marks"] - pools["released_total"]    # amounts are rounded to 1/1000 of a share
    return pools


def load_rec(run_dir: Path, arm: str, case: str, seed: int) -> Dict[str, Any]:
    return json.loads((run_dir / arm / case / f"seed{seed}" / "record.json").read_text())


def gap_rows(run_dir: Path, arm: str, keys: Sequence[Tuple[str, int]], cases: Mapping[str, CaseInfo43]) -> List[Dict[str, Any]]:
    """Per (case, seed) gap split for one arm. Needs the S' record. S and position-moved arms get the unreleased pool only."""
    rows = []
    for case, seed in keys:
        arm_rec, prime = load_rec(run_dir, arm, case, seed), load_rec(run_dir, PRIME, case, seed)
        total = float(arm_rec["total_drug"])
        far = poor = None
        if arm not in ("S", PRIME) and arm not in POSITIONS_MOVED:
            p = run_pools(R.load_run(run_dir, arm, case, seed), cases[case])
            far, poor = p["false_alarm"], p["poor_patch"]
        rows.append(gap_by_cause(cov_arm=float(arm_rec["metrics"]["coverage"]), cov_prime=float(prime["metrics"]["coverage"]),
                                 prime_released=float(prime["metrics"]["released_total"]), total_drug=total, false_alarm=far,
                                 poor_patch=poor, unreleased=max(0.0, total - float(arm_rec["metrics"]["released_total"]))))
    return rows


def mean_pools(rows: Sequence[Mapping[str, Any]], total_drugs: Sequence[float]) -> Dict[str, Any]:
    n = len(rows)
    mean = lambda k: float(np.mean([r[k] for r in rows])) if n else None  # noqa: E731
    rel = mean("released_from_marks")
    poor = mean("poor_patch")
    return {"n_runs": n, "total_drug": float(np.mean(total_drugs)) if n else None, "released_from_marks": rel, "poor_patch": poor,
            "false_alarm": mean("false_alarm"), "other": mean("other"), "n_releases": mean("n_releases"),
            "n_poor_patch_releases": mean("n_poor_patch"), "poor_share_of_released": (poor / rel if rel else None),
            "max_abs_accounting_error": float(max((abs(r["accounting_error"]) for r in rows), default=0.0))}


def realised_u_block() -> Optional[Dict[str, Any]]:
    p = REPO / REALISED_U
    if not p.is_file():
        return None
    doc = json.loads(p.read_text())
    return {"source": REALISED_U, "cases": doc.get("cases"),
            "levels": {lvl: d.get("overall_voxel_weighted") for lvl, d in doc["levels"].items()}}


# --- driver -----------------------------------------------------------------------------------------------------------------

def present(run_dir: Path, arms: Sequence[str]) -> List[str]:
    return [a for a in arms if (run_dir / a).is_dir()]


def analyse(run_dir: Path, cases_root: Optional[Path], *, limit: int = 0, level: Optional[str] = None,
            extra_run_dirs: Sequence[Path] = (), loader: Optional[Callable] = None) -> Dict[str, Any]:
    keys = R.run_keys(run_dir, SOLO)
    if limit:
        keys = keys[:limit]
    meta = json.loads((run_dir / "results.json").read_text())["meta"]
    facts = meta["frozen_facts"]
    level = level or meta.get("level") or facts.get("uptake_level") or "mid"
    cases = {c: CaseInfo43(c, cases_root, facts, level, loader) for c in sorted({c for c, _ in keys})}
    arms = present(run_dir, ALL_ARMS)
    out: Dict[str, Any] = {"study": "v4.3", "run_dir": str(run_dir), "level": level, "pairs": len(keys), "arms": arms, "frozen_facts": facts,
                           "sources": {"results": str(run_dir / "results.json"), "records": str(run_dir / "<arm>/<case>/seed<N>/record.json"),
                                       "bundles": str(run_dir / "<arm>/<case>/seed<N>/bundle.json"), "realised_u": REALISED_U},
                           "assumption": ASSUMPTION}
    sh = [cases[c].fact_shares() for c, _ in keys]
    out["world_fact_shares_over_pairs"] = {k: round(st.mean(x[k] for x in sh), 4) for k in sh[0]} if sh else {}
    out["arm_means"] = {a: R.arm_means(run_dir, a, keys) for a in arms}
    for ex in extra_run_dirs:
        out.setdefault("robustness_arm_means", {})[ex.name] = {a: R.arm_means(ex, a, keys) for a in present(ex, ("S", SOLO, "S-marks", "S-marks-perfect", PRIME))}
    out["divergence_from_solo"] = {a: R.divergence(run_dir, a, keys) for a in present(run_dir, ("S-marks", "S-talk", "S-marks-perfect", *CONTROLS))}

    colony = present(run_dir, COLONY_ARMS)
    shared: Dict[str, Any] = {}
    for arm in present(run_dir, SHARING):
        tot: Dict[str, float] = {}
        firsts: Counter = Counter()
        closure: List[int] = []
        for case, seed in keys:
            r = R.shared_inputs(R.load_run(run_dir, arm, case, seed), cases[case], R.load_run(run_dir, SOLO, case, seed))
            R.add(tot, r)
            if r["_first_closure_tick"] is not None:
                closure.append(r["_first_closure_tick"])
            if r["_first_cause"]:
                firsts[r["_first_cause"][2]] += 1
        per_run = {k: v for k, v in R.mean_of(tot, len(keys)).items() if "75pct" not in k}      # v4.2's 75% amount does not exist in v4.3
        shared[arm] = {"per_run_mean": per_run, "first_divergence_cause_counts": dict(firsts.most_common()),
                       "runs_with_any_region_closed": len(closure),
                       "first_region_closed_tick_median": st.median(closure) if closure else None}
    out["shared_inputs"] = shared

    counters: Dict[str, Any] = {}
    probes: Dict[str, Any] = {}
    for arm in colony:
        t: Dict[str, float] = {}
        pr = {"probes": [], "probes_saved_by_marks": []}
        for case, seed in keys:
            r = R.load_run(run_dir, arm, case, seed)
            R.add(t, r["rec"].get("script_counters") or {})
            for k, v in r["col"]["counters"].items():
                if isinstance(v, (int, float)):
                    t["colony_" + k] = t.get("colony_" + k, 0) + v
            for k, v in probe_counts(r["rec"].get("script_counters")).items():
                pr[k].append(v)
        counters[arm] = R.mean_of(t, len(keys))
        probes[arm] = {k: (float(np.mean(v)) if v else None) for k, v in pr.items()} | {"runs": len(keys)}
    out["saved_counters_per_run"] = counters
    out["probes_per_run"] = probes

    anat: Dict[str, Any] = {}
    lq: Dict[str, Any] = {}
    pools_out: Dict[str, Any] = {}
    umark: Dict[str, Any] = {}
    for arm in colony:
        ta: Dict[str, float] = {}
        tl: Dict[str, float] = {}
        mc: Dict[str, float] = {}
        um: Dict[str, float] = {}
        prows, totals = [], []
        for case, seed in keys:
            run = R.load_run(run_dir, arm, case, seed)
            a = R.release_anatomy(run, cases[case])
            for k, v in a["mass_by_class"].items():
                mc[k] = mc.get(k, 0) + v
            R.add(ta, {k: v for k, v in a.items() if k in ("releases", "released_mass", "mass_unreleased", "ants_never_released",
                                                           "releases_look_false_positive", "releases_fp_far", "releases_on_slow_tissue")})
            R.add(tl, R.look_quality(run, cases[case]))
            R.add(um, uptake_mark_accuracy(run["col"], cases[case]))
            if arm not in POSITIONS_MOVED:
                prows.append(run_pools(run, cases[case]))
                totals.append(float(run["rec"]["total_drug"]))
        n = len(keys)
        anat[arm] = {"per_run": R.mean_of(ta, n), "mass_by_class_per_run": {k: round(v / n, 1) for k, v in sorted(mc.items())},
                     "note": "releases_on_slow_tissue counts releases at true u < 0.4 (v4.3)"}
        lq[arm] = {k: round(v, 1) for k, v in sorted(tl.items())}
        umark[arm] = {k: round(v / n, 3) for k, v in sorted(um.items())}
        if prows:
            pools_out[arm] = mean_pools(prows, totals)
        else:
            pools_out[arm] = {"n_runs": 0, "note": "mark positions are moved on purpose in this arm; position-based pools are not meaningful"}
    out["release_anatomy_per_run"] = anat
    out["look_quality_totals"] = lq
    out["poor_tissue_release"] = pools_out
    out["uptake_mark_accuracy_per_run"] = umark

    gaps: Dict[str, Any] = {}
    if PRIME in arms:
        for arm in present(run_dir, ("S", *COLONY_ARMS)):
            gaps[arm] = summarise_gaps(gap_rows(run_dir, arm, keys, cases))
    out["gap_to_s_prime_by_cause"] = gaps

    b: Dict[str, Any] = {}
    for arm in present(run_dir, (SOLO, "S-marks")):
        tb2: Dict[str, float] = {}
        tm: Dict[str, float] = {}
        for case, seed in keys:
            run = R.load_run(run_dir, arm, case, seed)
            R.add(tb2, R.beliefs(run, cases[case]))
            R.add(tm, R.mark_overlap(run, cases[case], {}))
        b[arm] = {"beliefs_per_run": R.mean_of(tb2, len(keys)), "mark_overlap_per_run": R.mean_of(tm, len(keys))}
    out["beliefs_and_overlap"] = b
    ma: Dict[str, Any] = {}
    for arm in present(run_dir, (SOLO, "S-marks", "S-marks-perfect")):
        t2: Dict[str, float] = {}
        for case, seed in keys:
            R.add(t2, R.mark_accuracy(R.load_run(run_dir, arm, case, seed), cases[case]))
        ma[arm] = {k: round(v / len(keys), 2) for k, v in sorted(t2.items())}
    out["judgement_mark_accuracy_per_run"] = ma
    out["realised_u"] = realised_u_block()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cases-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output folder; only diagnostics_numbers.json is written there")
    ap.add_argument("--extra-run-dir", action="append", default=[], type=Path)
    ap.add_argument("--level", default=None, choices=("low", "mid", "high"), help="default: the level recorded in results.json")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N (case, seed) pairs (for a quick check)")
    a = ap.parse_args(argv)
    try:
        assert_out_allowed(a.out)
    except OutRefused as exc:
        raise SystemExit(f"refused: {exc}")
    out = analyse(a.run_dir, a.cases_root, limit=a.limit, level=a.level, extra_run_dirs=a.extra_run_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / "diagnostics_numbers.json"
    target.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
