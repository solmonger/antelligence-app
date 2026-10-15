#!/usr/bin/env python
"""A1: why did sharing not help in verdict attempt 1?  Offline, read-only analysis of saved runs.

    python scripts/v42_diagnose.py --run-dir /Volumes/WD_BLACK/antelligence-v4.2/runs/verdict1 \
        --cases-root /Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases \
        [--extra-run-dir runs/verdict-R1 --extra-run-dir runs/verdict-R2] [--limit 0]

Reads ``results.json``, and each arm / case / seed ``record.json`` and ``bundle.json``. Writes only
``docs/research/v4.2-20261014/diagnostics_numbers.json`` (and, with ``--tables``, a markdown table dump next to it).
It never runs an episode and never changes a saved result.

Scorer-side truth (the tumour mask) is read from the dev-case store INSIDE this offline analysis only, to classify where
drug went. Only aggregate numbers are written. No voxel coordinate tied to truth leaves this script.

Method notes (what is exact and what is reconstructed):
* Decisions, readings, marks, need bands, counters and final region status are read from the bundle (exact).
* What an ant saw in ``marks_near`` and ``region_status`` at a decision turn is RECONSTRUCTED by replaying the saved mark
  records into the kernel's ``MarkField`` / ``Quorum`` (in memory, deterministic). The replay is checked against the saved
  final region status and mark counts; the check count is written to the output.
* "Used" means the rule's condition was true on the saved turn. A true counterfactual needs a replay with the input removed;
  ``effect.json`` of the verdict run already holds that for single marks (cited, not recomputed).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.kernel.colony.marks import MarkField, MarkParams  # noqa: E402
from antelligence.kernel.colony.quorum import Quorum, ZoneSpec  # noqa: E402
from antelligence.worlds.tumor_mri.colony_policies import NEED_NORM  # noqa: E402
from antelligence.worlds.tumor_mri.data import select_candidates  # noqa: E402

OUT_DIR = REPO / "docs" / "research" / "v4.2-20261014"
VOX_MM = 2.0
READ_MM, CONFIRM_MM = 6.0, 4.0
SOLO = "S-solo"
SHARING = ("S-marks", "S-talk", "S-marks-perfect")
CONTROLS = ("S-marks-displaced", "S-talk-shuffled", "S-marks-quiet")
ALL_ARMS = ("S", SOLO, "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S-marks-perfect", "S-marks-quiet", "S'")
AMOUNT = (0.10, 0.25, 0.50, 0.75, 1.00)
RELEASE_LOOK = 0.5
STEP = 5


# --- loading ------------------------------------------------------------------------------------------------------------

def run_keys(run_dir: Path, arm: str) -> List[Tuple[str, int]]:
    out = []
    for rec in sorted((run_dir / arm).glob("*/seed*/record.json")):
        out.append((rec.parent.parent.name, int(rec.parent.name[4:])))
    return out


def load_run(run_dir: Path, arm: str, case: str, seed: int, bundle: bool = True) -> Dict[str, Any]:
    d = run_dir / arm / case / f"seed{seed}"
    rec = json.loads((d / "record.json").read_text())
    if not bundle:
        return {"rec": rec}
    col = json.loads((d / "bundle.json").read_text())["extra"]["colony"]
    return {"rec": rec, "col": col}


def decisions(col: Mapping[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    out = {}
    for a in col["action_log"]:
        if a.get("kind") == "decision":
            p = a["params"]
            w = p["world"]
            out[(int(a["tick"]), int(a["ant"]))] = {
                "target": w["target"], "release": int(w["release"]), "amount": int(w["amount"]),
                "tvox": tuple(w["target_vox"]) if w.get("target_vox") is not None else None,
                "job": p["meta"]["job"], "reason": p["meta"]["reason"], "stim": p["meta"]["stimuli"],
                "emit": [(e["kind"], tuple(e["pos_vox"])) for e in p["emit"]], "ask": p.get("ask")}
    return out


def sig(d: Mapping[str, Any]) -> Tuple[Any, ...]:
    """What the world sees of a decision: target label, release flag, amount level, goal override."""
    return (d["target"], d["release"], d["amount"], d["tvox"])


def positions(col: Mapping[str, Any]) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """Ant position at each decision tick (the ``dose`` reading is always logged at the ant's own voxel)."""
    pos = {}
    for ant, log in col["reading_logs"].items():
        for e in log:
            if e["kind"] == "dose":
                pos[(int(e["tick"]), int(ant))] = tuple(e["pos_vox"])
    return pos


def looks(col: Mapping[str, Any]) -> Dict[Tuple[int, int], float]:
    return {(int(e["tick"]), int(ant)): float(e["value"]) for ant, log in col["reading_logs"].items()
            for e in log if e["kind"] == "look"}


def mm(a: Sequence[float], b: Sequence[float]) -> float:
    return VOX_MM * math.dist(a, b)


# --- cases (scorer side, offline only) --------------------------------------------------------------------------------

class CaseInfo:
    def __init__(self, case_id: str, cases_root: Path, facts: Optional[Mapping[str, Any]] = None) -> None:
        from scipy import ndimage
        from antelligence.experiments.v42_harness import load_dev_case
        from antelligence.worlds.tumor_mri.physics import uptake_source_a, washout_field
        c = load_dev_case(case_id, root=cases_root)
        f = dict(facts or {})
        self.u = self.delta = None
        if f.get("f2") and c.scorer_channels:
            self.u = uptake_source_a(c.scorer_channels["tc"], c.scorer_channels["et"], float(f["u_core"]))
        if f.get("f3"):
            self.delta = washout_field(c.crop.truth, case_id, float(f["f3_decay"]), float(f["f3_share"]), c.crop.voxel_mm)
        self.f3_decay = float(f.get("f3_decay") or 0.0)
        self.crop, self.regions = c.crop, list(c.regions or [])
        self.truth = c.crop.truth.astype(bool)
        self.dist_mm = ndimage.distance_transform_edt(~self.truth) * VOX_MM       # 0 inside the tumour
        self.cands = {cd.label: cd for cd in select_candidates(self.regions)}     # static pool A..F (queen hook unset)
        self.zones = []
        for r in self.regions:
            rad = (3.0 * r.volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0) + 6.0
            ctr = tuple(r.centroid_vox)
            self.zones.append(ZoneSpec(r.region_id, ctr, (lambda ctr, rad: lambda p: VOX_MM * math.dist(p, ctr) <= rad * (1 + 1e-9))(ctr, rad)))
        self.shape = self.truth.shape

    def inside(self, pos: Sequence[int]) -> bool:
        return all(0 <= p < s for p, s in zip(pos, self.shape))

    def truth_at(self, pos: Sequence[int]) -> Optional[bool]:
        return bool(self.truth[tuple(pos)]) if self.inside(pos) else None

    def dist_at(self, pos: Sequence[int]) -> float:
        return float(self.dist_mm[tuple(pos)]) if self.inside(pos) else float("inf")

    def slow_tissue(self, pos: Sequence[int]) -> bool:
        return self.u is not None and self.inside(pos) and float(self.u[tuple(pos)]) < 1.0

    def fast_washout(self, pos: Sequence[int]) -> bool:
        return self.delta is not None and self.inside(pos) and float(self.delta[tuple(pos)]) >= 0.5 * self.f3_decay > 0

    def fact_shares(self) -> Dict[str, float]:
        t = self.truth
        return {"tumour_voxels": int(t.sum()),
                "share_tumour_slow": float((self.u[t] < 1.0).mean()) if self.u is not None and t.any() else 0.0,
                "share_tumour_fast_washout": float((self.delta[t] >= 0.5 * self.f3_decay).mean()) if self.delta is not None and t.any() else 0.0}

    def scan_p(self, pos: Sequence[int]) -> float:
        return float(self.crop.p_tumour[tuple(pos)])


# --- reconstruction of shared inputs ---------------------------------------------------------------------------------

def replay(col: Mapping[str, Any], case: CaseInfo) -> Dict[str, Any]:
    """Replay the saved marks into MarkField / Quorum. Returns per-(tick, ant) marks_near rows, the region status at each
    decision tick and the quorum events, plus a check against the saved final status and mark counts."""
    prm = col["params"]
    mf = MarkField(col["arm"], MarkParams(voxel_mm=VOX_MM, read_radius_mm=prm["read_radius_mm"],
                                          confirm_radius_mm=prm["confirm_radius_mm"], half_life=prm["half_life"]))
    quorum = Quorum(case.zones, mf, k=prm["quorum_k"])
    pos = positions(col)
    by_tick: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for m in col["marks"]:
        by_tick[int(m["tick"])].append(m)
    ticks = sorted({t for t, _ in pos})
    last = max(ticks + list(by_tick) + [0])
    rows: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    status: Dict[int, Dict[int, str]] = {}
    seen_records = -1
    for t in range(last + 1):
        mf.expire(t)
        if len(mf.records) != seen_records:
            seen_records = len(mf.records)
            quorum.evaluate(t)
        if t % STEP == 0:
            status[t] = dict(quorum.status)
            for a in range(8):
                if (t, a) in pos:
                    rows[(t, a)] = mf.marks_near(pos[(t, a)], t)
        for m in by_tick.get(t, ()):
            mf.deposit(m["ant"], m["mark_kind"], m["pos_vox"], t, amount=m["amount_milli"] / 1000.0, reading_ppm=m["reading_ppm"])
    final = status[max(status)] if status else {}
    saved = {int(k): v for k, v in col["region_status"].items()}
    # final status in the bundle is at the end of the run; the last replayed decision tick is 115, so re-evaluate once more
    mf.expire(last + 1)
    quorum.evaluate(last + 1)
    ok = all(quorum.status.get(r) == s for r, s in saved.items()) if saved else True
    return {"rows": rows, "status": status, "events": [(e.tick_first, e.region_id, e.status) for e in quorum.events],
            "status_ok": ok, "marks_ok": len(mf.records) == len(col["marks"]), "pos": pos}


def foreign(rows: Sequence[Mapping[str, Any]], ant: int) -> List[Mapping[str, Any]]:
    return [r for r in rows if any(a != ant for a in r["authors"])]


# --- section 1: divergence from S-solo ---------------------------------------------------------------------------------

def divergence(run_dir: Path, arm: str, keys: Sequence[Tuple[str, int]], base: str = SOLO) -> Dict[str, Any]:
    first, n_turns, n_diff, field_diff, ident = [], 0, 0, Counter(), 0
    per_bin = Counter()
    per_bin_n = Counter()
    d_cov, same_final = [], 0
    cause_input = []
    for case, seed in keys:
        a = load_run(run_dir, base, case, seed)
        b = load_run(run_dir, arm, case, seed)
        da, db = decisions(a["col"]), decisions(b["col"])
        common = sorted(set(da) & set(db))
        ft = None
        for k in common:
            n_turns += 1
            per_bin_n[k[0] // 30 * 30] += 1
            if sig(da[k]) != sig(db[k]):
                n_diff += 1
                per_bin[k[0] // 30 * 30] += 1
                for i, f in enumerate(("target", "release", "amount", "target_vox")):
                    field_diff[f] += sig(da[k])[i] != sig(db[k])[i]
                if ft is None:
                    ft = k[0]
        if ft is None:
            ident += 1
        else:
            first.append(ft)
        d_cov.append(b["rec"]["metrics"]["coverage"] - a["rec"]["metrics"]["coverage"])
        same_final += abs(d_cov[-1]) < 1e-12
    return {"pairs": len(keys), "pairs_identical_all_turns": ident, "first_divergence_tick_median": st.median(first) if first else None,
            "first_divergence_tick_mean": round(st.mean(first), 2) if first else None,
            "first_divergence_tick_min": min(first) if first else None,
            "share_pairs_diverge_by_tick30": round(sum(t <= 30 for t in first) / len(keys), 3),
            "share_turns_differ": round(n_diff / n_turns, 4), "turns": n_turns, "turns_differ": n_diff,
            "field_diff_share": {k: round(v / n_turns, 4) for k, v in field_diff.items()},
            "share_turns_differ_by_tick_bin": {str(k): round(per_bin[k] / per_bin_n[k], 4) for k in sorted(per_bin_n)},
            "coverage_delta_mean": round(st.mean(d_cov), 5), "coverage_delta_abs_mean": round(st.mean(abs(x) for x in d_cov), 5),
            "coverage_delta_sd": round(st.pstdev(d_cov), 5), "pairs_same_coverage": same_final}


# --- section 2: which shared inputs fired -----------------------------------------------------------------------------

def predicted_top(case: CaseInfo, label_pos: Tuple[int, int, int], need: Mapping[int, str], closed: Sequence[int],
                  exclude: Sequence[int] = ()) -> Optional[str]:
    """Top-scoring zone target the script would pick (need x scan p / distance), from perception-side numbers only."""
    best, best_s = None, -1.0
    for label, c in sorted(case.cands.items()):
        if c.region_id in closed or c.region_id in exclude:
            continue
        band = need.get(c.region_id)
        value = (1.0 if band is None else NEED_NORM[band]) * c.p_tumour
        if value <= 0:
            continue
        s = value / max(mm(label_pos, c.centroid_vox), 1.0)
        if s > best_s:
            best, best_s = label, s
    return best


def need_at(col: Mapping[str, Any]) -> Dict[int, Dict[int, str]]:
    by: Dict[int, Dict[int, str]] = defaultdict(dict)
    for r in col["need_per_zone"]:
        by[int(r["tick"])][int(r["region_id"])] = r["need"]
    return by


def latest_need(by: Mapping[int, Mapping[int, str]], t: int) -> Mapping[int, str]:
    ks = [k for k in by if k <= t]
    return by[max(ks)] if ks else {}


def own_reading_near(col: Mapping[str, Any], ant: int, tick: int, kind: str, word: str, pos: Sequence[int], radius_mm: float) -> bool:
    for e in col["reading_logs"][str(ant)]:
        if e["tick"] <= tick and e["kind"] == kind and e.get("word") == word and mm(e["pos_vox"], pos) <= radius_mm * (1 + 1e-9):
            return True
    return False


def shared_inputs(run: Mapping[str, Any], case: CaseInfo, solo: Mapping[str, Any]) -> Dict[str, Any]:
    """Per run counts of the turns on which each shared input was present / fired, plus a cause for the first divergence."""
    col = run["col"]
    rp = replay(col, case)
    dec = decisions(col)
    pos = rp["pos"]
    need = need_at(col)
    need_solo = need_at(solo["col"])
    dsolo = decisions(solo["col"])
    c = Counter()
    first_cause: Optional[Tuple[int, int, str]] = None
    first_tick = None
    for k in sorted(dec):
        t, ant = k
        d = dec[k]
        rows = rp["rows"].get(k, [])
        fr = foreign(rows, ant)
        closed = [r for r, s in rp["status"][t].items() if s != "open"]
        c["turns"] += 1
        c["turns_foreign_mark_visible"] += bool(fr)
        c["turns_any_mark_visible"] += bool(rows)
        for r in fr:
            c["foreign_rows_" + r["kind"]] += 1
        c["turns_region_closed"] += bool(closed)
        cause: List[str] = []
        # 1. strong not_tumour / done status removed a candidate zone from the dose targets
        if d["job"] == "dose" and d["target"] != "S":
            top_open = predicted_top(case, pos[k], latest_need(need, t), closed)
            top_all = predicted_top(case, pos[k], latest_need(need, t), [])
            c["dose_zone_turns"] += 1
            c["dose_zone_pred_match"] += top_open == d["target"]
            if top_all != top_open and top_open == d["target"]:
                c["dose_target_changed_by_status"] += 1
                cause.append("status_skip")
            top_solo_need = predicted_top(case, pos[k], latest_need(need_solo, t), [])
            if top_solo_need != top_all and top_all == d["target"]:
                c["dose_target_changed_by_queen_need"] += 1
                cause.append("queen_need_ranking")
        # 2. tumour_here / need_more of another ant became a spot target (job dose, label Stay, goal override)
        if d["job"] == "dose" and d["target"] == "S" and d["tvox"] is not None:
            c["spot_targets_from_marks"] += 1
            tt = case.truth_at(d["tvox"])
            c["spot_targets_on_tumour"] += bool(tt)
            c["spot_targets_within_4mm_of_tumour"] += case.dist_at(d["tvox"]) <= 4.0
            cause.append("spot_target")
        # 3. strong dosed / dose_ok at the spot ('dose already high here')
        if d["reason"] == "shared_dose_high":
            c["shared_dose_high"] += 1
            cause.append("shared_dose_high")
        # 4. washout: the only move-on; counts as shared when no own fast reading within 4 mm
        if d["reason"] == "washout":
            c["washout_blocked_release"] += 1
            if not own_reading_near(col, ant, t, "washout", "fast", pos[k], CONFIRM_MM):
                c["washout_blocked_by_foreign_mark"] += 1
                cause.append("foreign_washout")
        # 5. 75 percent release without an own slow reading near (so a foreign slow_uptake mark did it)
        if d["release"] and d["amount"] == 4:
            c["released_75pct"] += 1
            if not own_reading_near(col, ant, t, "uptake", "slow", pos[k], READ_MM):
                c["released_75pct_from_foreign_mark"] += 1
                cause.append("foreign_slow_uptake")
        # 6. explore aimed at a region others already marked (goal override on the explore job)
        if d["job"] == "explore" and d["tvox"] is not None:
            c["explore_aimed_by_mark"] += 1
            cause.append("explore_aim")
        # 7. queen need band differs from S-solo's (the queen reads the board in sharing arms only)
        if latest_need(need, t) != latest_need(need_solo, t):
            c["turns_need_differs_from_solo"] += 1
        if k in dsolo and sig(dsolo[k]) != sig(d) and first_tick is None:
            first_tick = t
            first_cause = (t, ant, "+".join(cause) if cause else ("need_band" if latest_need(need, t) != latest_need(need_solo, t) else "state"))
        c["decisions_with_cause"] += bool(cause)
        if d["release"]:
            tr = case.truth_at(pos[k])
            if tr is False and case.dist_at(pos[k]) > 4.0:
                c["releases_far_from_tumour"] += 1
                c["releases_far_with_foreign_not_tumour_visible"] += any(r["kind"] == "not_tumour" for r in fr)
                c["releases_far_with_any_foreign_mark_visible"] += bool(fr)
                c["releases_far_with_foreign_tumour_here_visible"] += any(r["kind"] == "tumour_here" for r in fr)
    c["quorum_events"] = len(rp["events"])
    c["quorum_done"] = sum(1 for e in rp["events"] if e[2] == "done")
    c["quorum_not_tumour"] = sum(1 for e in rp["events"] if e[2] == "not_tumour")
    c["replay_status_ok"] = int(rp["status_ok"])
    c["replay_marks_ok"] = int(rp["marks_ok"])
    out = dict(c)
    out["_first_cause"] = first_cause
    out["_first_closure_tick"] = min((e[0] for e in rp["events"]), default=None)
    return out


# --- section 3: release anatomy (colony arms) ------------------------------------------------------------------------

def release_anatomy(run: Mapping[str, Any], case: CaseInfo) -> Dict[str, Any]:
    """Every release with its mass, where it landed relative to the true tumour, and the look that triggered it."""
    col, rec = run["col"], run["rec"]
    dec = decisions(col)
    pos = positions(col)
    lk = looks(col)
    share = rec["total_drug"] / 8.0
    payload = {a: share for a in range(8)}
    out = Counter()
    mass_by_class: Dict[str, float] = defaultdict(float)
    mass_by_tick: Dict[int, float] = defaultdict(float)
    jobs, reasons = Counter(), Counter()
    first_release = {}
    for k in sorted(dec):
        t, a = k
        d = dec[k]
        jobs[d["job"]] += 1
        reasons[d["reason"]] += 1
        if case.truth_at(pos[k]):
            out["turns_inside_tumour"] += 1
            out["inside_released"] += bool(d["release"])
            out["inside_reason_" + d["reason"]] += 1
            out["inside_job_" + d["job"]] += 1
        if d["release"] and payload[a] > 0:
            m = payload[a] * AMOUNT[d["amount"] - 1]
            payload[a] -= m
            p = pos[k]
            tr, dist = case.truth_at(p), case.dist_at(p)
            lv = lk.get(k)
            cls = "inside_tumour" if tr else ("within_4mm" if dist <= 4.0 else "far_from_tumour")
            mass_by_class[cls] += m
            mass_by_tick[t // 30 * 30] += m
            out["releases"] += 1
            out["releases_" + cls] += 1
            if lv is not None and lv >= RELEASE_LOOK and not tr:
                out["releases_look_false_positive"] += 1
                mass_by_class["look_false_positive"] += m
                mass_by_class["fp_tick_%s" % ("0" if t == 0 else "1_29" if t < 30 else "30_plus")] += m
                mass_by_class["fp_job_" + d["job"]] += m
                if dist > 4.0:
                    mass_by_class["look_false_positive_far"] += m
                    out["releases_fp_far"] += 1
            if case.slow_tissue(p):
                mass_by_class["on_slow_tissue_f2"] += m
                out["releases_on_slow_tissue"] += 1
            if case.fast_washout(p):
                mass_by_class["on_fast_washout_f3"] += m
                out["releases_on_fast_washout"] += 1
            if t == 0:
                mass_by_class["released_at_tick0"] += m
                out["releases_tick0"] += 1
            sp = case.scan_p(p)
            if sp < 0.5:
                out["releases_scan_below_half"] += 1
                mass_by_class["scan_below_half"] += m
            if sp >= 0.5 and not tr:
                mass_by_class["scan_false_positive"] += m
            first_release.setdefault(a, t)
    released = sum(mass_by_class[c] for c in ("inside_tumour", "within_4mm", "far_from_tumour"))
    res = {"releases": out["releases"], "released_mass": released, "mass_unreleased": sum(payload.values()),
           "mass_by_class": dict(mass_by_class), "mass_by_tick_bin": dict(mass_by_tick), "jobs": dict(jobs),
           "reasons": dict(reasons), "first_release_tick_mean": st.mean(first_release.values()) if first_release else None,
           "ants_never_released": 8 - len(first_release), "releases_look_false_positive": out["releases_look_false_positive"],
           "releases_scan_below_half": out["releases_scan_below_half"], "releases_fp_far": out["releases_fp_far"],
           "releases_tick0": out["releases_tick0"], "releases_on_slow_tissue": out["releases_on_slow_tissue"],
           "releases_on_fast_washout": out["releases_on_fast_washout"], "inside": {k: v for k, v in out.items() if k.startswith(("inside", "turns_inside"))},
           "check_released_vs_metrics": released - rec["metrics"]["released_total"]}
    # mass-weighted tumour exposure proxy: share of tumour voxels within 2 voxels of the release voxel
    return res


def look_quality(run: Mapping[str, Any], case: CaseInfo) -> Dict[str, Any]:
    col = run["col"]
    pos = positions(col)
    lk = looks(col)
    c = Counter()
    for k, v in lk.items():
        p = pos[k]
        tr = case.truth_at(p)
        if tr is None:
            continue
        sp = case.scan_p(p)
        c["looks"] += 1
        c["look_pos"] += v >= 0.5
        c["look_pos_true"] += (v >= 0.5) and tr
        c["scan_pos"] += sp >= 0.5
        c["scan_pos_true"] += (sp >= 0.5) and tr
        c["truth"] += tr
        c["look_scan_disagree"] += (v >= 0.5) != (sp >= 0.5)
        dist = case.dist_at(p)
        cls = "inside" if tr else ("out_0_4" if dist <= 4.0 else "out_4_10" if dist <= 10.0 else "out_gt10")
        c["n_" + cls] += 1
        c["look_ge_half_" + cls] += v >= 0.5
        c["scan_ge_half_" + cls] += sp >= 0.5
        c["look_lt_half_" + cls] += v < 0.5
        c["look_grey_" + cls] += 0.35 <= v <= 0.65
    return dict(c)


# --- section 4: sensors, marks and beliefs ---------------------------------------------------------------------------

def beliefs(run: Mapping[str, Any], case: CaseInfo) -> Dict[str, Any]:
    """Per zone: how many distinct ants looked inside it, and whether two ants hold different beliefs (mean look on opposite
    sides of 0.5) or the same wrong belief."""
    col = run["col"]
    pos = positions(col)
    per_zone: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for k, v in looks(col).items():
        z = next((zs.region_id for zs in case.zones if zs.contains(pos[k])), None)
        if z is not None:
            per_zone[z][k[1]].append(v)
    c = Counter()
    for z, by_ant in per_zone.items():
        c["zones_looked_at"] += 1
        n = len(by_ant)
        c["zones_by_ants_%d" % min(n, 3)] += 1
        if n >= 2:
            means = [st.mean(v) for v in by_ant.values()]
            c["zones_two_plus_ants"] += 1
            if min(means) < 0.5 <= max(means):
                c["zones_ants_disagree"] += 1
            if max(means) - min(means) >= 0.2:
                c["zones_ants_differ_by_0.2"] += 1
    c["zones_total"] = len(case.zones)
    return dict(c)


def mark_overlap(run: Mapping[str, Any], case: CaseInfo, rp: Mapping[str, Any]) -> Dict[str, Any]:
    """How often could another ant read a mark at all: for each mark, was any other ant within the read radius during the
    mark's life (decision turns only), and the distance of the nearest foreign mark at each decision."""
    col = run["col"]
    pos = positions(col)
    c = Counter()
    nearest = []
    for k, p in pos.items():
        t, ant = k
        best = None
        for m in col["marks"]:
            if m["ant"] == ant or m["tick"] >= t:
                continue
            d = mm(p, m["pos_vox"])
            if best is None or d < best:
                best = d
        if best is not None:
            nearest.append(best)
    c["turns"] = len(pos)
    c["turns_nearest_foreign_mark_le6mm_any_age"] = sum(d <= 6.0 for d in nearest)
    c["turns_nearest_foreign_mark_le20mm_any_age"] = sum(d <= 20.0 for d in nearest)
    out = dict(c)
    out["nearest_foreign_mark_median_mm"] = st.median(nearest) if nearest else None
    return out


def mark_accuracy(run: Mapping[str, Any], case: CaseInfo) -> Dict[str, Any]:
    """How right are the judgement marks the ants wrote (truth-side check, offline)."""
    c = Counter()
    for m in run["col"]["marks"]:
        k = m["mark_kind"]
        if k not in ("tumour_here", "not_tumour", "need_more"):
            continue
        tr = case.truth_at(m["pos_vox"])
        if tr is None:
            continue
        c["n_" + k] += 1
        c["on_tumour_" + k] += bool(tr)
        low = case.scan_p(m["pos_vox"]) < 0.5
        c["scan_below_half_" + k] += low
        c["scan_below_half_on_tumour_" + k] += low and bool(tr)
        c["scan_at_least_half_off_tumour_" + k] += (not low) and not tr
    return dict(c)


# --- driver -----------------------------------------------------------------------------------------------------------

def add(total: Dict[str, float], part: Mapping[str, Any]) -> None:
    for k, v in part.items():
        if isinstance(v, (int, float)) and not k.startswith("_"):
            total[k] = total.get(k, 0) + v


def mean_of(totals: Mapping[str, float], n: int) -> Dict[str, float]:
    return {k: round(v / n, 3) for k, v in sorted(totals.items())}


def arm_means(run_dir: Path, arm: str, keys: Sequence[Tuple[str, int]]) -> Dict[str, Any]:
    cov, rel, off, dec, td, deca, lost = [], [], [], [], [], [], []
    facts: Dict[str, List[float]] = defaultdict(list)
    marks: Dict[str, List[float]] = defaultdict(list)
    for case, seed in keys:
        r = load_run(run_dir, arm, case, seed, bundle=False)["rec"]
        m = r["metrics"]
        cov.append(m["coverage"])
        rel.append(m["released_total"])
        td.append(r["total_drug"])
        off.append(m["off_target_share"])
        deca.append(m["decayed"])
        lost.append(m["lost_boundary"])
        for k, v in (m.get("facts") or {}).items():
            if v is not None:
                facts[k].append(v)
    n = len(keys)
    mean = lambda xs: round(st.mean(xs), 4)  # noqa: E731
    on = [r * (1 - o) for r, o in zip(rel, off)]
    return {"runs": n, "coverage": mean(cov), "released": mean(rel), "total_drug": mean(td),
            "released_share_of_total": round(sum(rel) / sum(td), 4), "off_target_share": mean(off),
            "on_target_mass": mean(on), "decayed": mean(deca), "lost_boundary": mean(lost),
            "facts_mean": {k: round(st.mean(v), 3) for k, v in facts.items()}}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cases-root", required=True, type=Path)
    ap.add_argument("--extra-run-dir", action="append", default=[], type=Path)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N (case, seed) pairs (for a quick check)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "diagnostics_numbers.json")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_DIR not in args.out.resolve().parents and args.out.resolve().parent != OUT_DIR.resolve():
        raise SystemExit("refusing to write outside docs/research/v4.2-20261014/")
    keys = run_keys(args.run_dir, SOLO)
    if args.limit:
        keys = keys[: args.limit]
    meta = json.loads((args.run_dir / "results.json").read_text())["meta"]
    facts = meta["frozen_facts"]
    cases = {c: CaseInfo(c, args.cases_root, facts) for c in sorted({c for c, _ in keys})}
    out: Dict[str, Any] = {"run_dir": str(args.run_dir), "pairs": len(keys), "frozen_facts": facts}
    sh = [cases[c].fact_shares() for c, _ in keys]
    out["world_fact_shares_over_pairs"] = {k: round(st.mean(x[k] for x in sh), 4) for k in sh[0]}

    out["arm_means"] = {a: arm_means(args.run_dir, a, keys) for a in ALL_ARMS}
    for ex in args.extra_run_dir:
        out.setdefault("robustness_arm_means", {})[ex.name] = {a: arm_means(ex, a, keys) for a in ("S", SOLO, "S-marks", "S-marks-perfect", "S'")}

    out["divergence_from_solo"] = {a: divergence(args.run_dir, a, keys) for a in ("S-marks", "S-talk", "S-marks-perfect", *CONTROLS, "S-marks-displaced")}
    out["divergence_other_pairs"] = {"S-marks-perfect_vs_S-marks": divergence(args.run_dir, "S-marks-perfect", keys, "S-marks"),
                                     "S-talk_vs_S-marks": divergence(args.run_dir, "S-talk", keys, "S-marks"),
                                     "S-talk-shuffled_vs_S-marks": divergence(args.run_dir, "S-talk-shuffled", keys, "S-marks")}

    shared: Dict[str, Any] = {}
    firsts: Dict[str, Counter] = {}
    for arm in SHARING:
        tot: Dict[str, float] = {}
        firsts[arm] = Counter()
        n_first = 0
        closure: List[int] = []
        for case, seed in keys:
            run = load_run(args.run_dir, arm, case, seed)
            solo = load_run(args.run_dir, SOLO, case, seed)
            r = shared_inputs(run, cases[case], solo)
            add(tot, r)
            if r["_first_closure_tick"] is not None:
                closure.append(r["_first_closure_tick"])
            if r["_first_cause"]:
                firsts[arm][r["_first_cause"][2]] += 1
                n_first += 1
        shared[arm] = {"per_run_mean": mean_of(tot, len(keys)), "first_divergence_cause_counts": dict(firsts[arm].most_common()),
                       "runs_with_divergence": n_first,
                       "runs_with_any_region_closed": len(closure),
                       "first_region_closed_tick_median": st.median(closure) if closure else None,
                       "first_region_closed_tick_mean": round(st.mean(closure), 1) if closure else None}
    out["shared_inputs"] = shared

    # saved counters (script_counters, colony counters)
    cnt: Dict[str, Dict[str, float]] = {}
    for arm in (SOLO, *SHARING):
        t: Dict[str, float] = {}
        for case, seed in keys:
            r = load_run(args.run_dir, arm, case, seed)
            add(t, r["rec"].get("script_counters") or {})
            for k, v in r["col"]["counters"].items():
                if isinstance(v, (int, float)):
                    t["colony_" + k] = t.get("colony_" + k, 0) + v
            for k, v in r["col"]["counters"].get("marks", {}).items():
                t["marks_" + k] = t.get("marks_" + k, 0) + v
        cnt[arm] = mean_of(t, len(keys))
    out["saved_counters_per_run"] = cnt

    # release anatomy and look quality for the colony arms; S has no saved logs
    anat: Dict[str, Any] = {}
    lq: Dict[str, Any] = {}
    for arm in (SOLO, "S-marks", "S-talk", "S-marks-perfect"):
        ta: Dict[str, float] = {}
        tl: Dict[str, float] = {}
        mc: Dict[str, float] = {}
        jobs: Counter = Counter()
        reasons: Counter = Counter()
        tb: Dict[str, float] = defaultdict(float)
        insd: Dict[str, float] = {}
        chk: List[float] = []
        fr = []
        for case, seed in keys:
            run = load_run(args.run_dir, arm, case, seed)
            a = release_anatomy(run, cases[case])
            fr.append(a["first_release_tick_mean"] if a["first_release_tick_mean"] is not None else 120)
            chk.append(abs(a["check_released_vs_metrics"]))
            for k, v in a["mass_by_class"].items():
                mc[k] = mc.get(k, 0) + v
            for k, v in a["mass_by_tick_bin"].items():
                tb[str(k)] += v
            jobs.update(a["jobs"])
            reasons.update(a["reasons"])
            for k, v in a["inside"].items():
                insd[k] = insd.get(k, 0) + v
            add(ta, {k: v for k, v in a.items() if k in ("releases_on_slow_tissue", "releases_on_fast_washout", "releases", "released_mass", "mass_unreleased", "ants_never_released",
                                                         "releases_look_false_positive", "releases_scan_below_half", "releases_fp_far",
                                                         "releases_tick0")})
            add(tl, look_quality(run, cases[case]))
        n = len(keys)
        anat[arm] = {"per_run": mean_of(ta, n), "mass_by_class_per_run": {k: round(v / n, 1) for k, v in sorted(mc.items())},
                     "mass_by_tick_bin_per_run": {k: round(v / n, 1) for k, v in sorted(tb.items(), key=lambda kv: int(kv[0]))},
                     "decision_turns_by_job": {k: round(v / n, 2) for k, v in jobs.items()},
                     "decision_turns_by_reason": {k: round(v / n, 2) for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])},
                     "max_abs_release_accounting_error_vs_released_total": max(chk),
                     "turns_inside_tumour_per_run": {k: round(v / n, 2) for k, v in sorted(insd.items())},
                     "first_release_tick_mean_over_ants_and_runs": round(st.mean(fr), 2)}
        lq[arm] = {k: round(v, 1) for k, v in sorted(tl.items())}
    out["release_anatomy_per_run"] = anat
    out["look_quality_totals"] = lq

    # section 4
    b: Dict[str, Any] = {}
    for arm in (SOLO, "S-marks"):
        tb2: Dict[str, float] = {}
        tm: Dict[str, float] = {}
        for case, seed in keys:
            run = load_run(args.run_dir, arm, case, seed)
            add(tb2, beliefs(run, cases[case]))
            add(tm, mark_overlap(run, cases[case], {}))
        b[arm] = {"beliefs_per_run": mean_of(tb2, len(keys)), "mark_overlap_per_run": mean_of(tm, len(keys))}
    out["beliefs_and_overlap"] = b

    ma: Dict[str, Any] = {}
    for arm in (SOLO, "S-marks", "S-marks-perfect"):
        t2: Dict[str, float] = {}
        for case, seed in keys:
            add(t2, mark_accuracy(load_run(args.run_dir, arm, case, seed), cases[case]))
        ma[arm] = {k: round(v / len(keys), 2) for k, v in sorted(t2.items())}
    out["judgement_mark_accuracy_per_run"] = ma

    args.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
