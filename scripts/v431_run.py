#!/usr/bin/env python
"""v4.31 Phase A runner: scripted S and oracle S' on the development cases, legacy constants or real units. $0, no keys, no model.

    python scripts/v431_run.py --physics units --k low --arms S,Sp --plan-thrs 1,10,100,1000,10000,100000,1000000 \
        --seeds 9,10 --out /Volumes/WD_BLACK/antelligence-v4.31/runs/units_low [--shard 0/3] [--save-dose]
    python scripts/v431_run.py --physics legacy --arms S,Sp --seeds 9,10 --out .../runs/legacy

* Development cases only (``dev_a`` + ``dev_b`` of config/v4_1/split.json, 40 cases). The store adapter's gate refuses anything else.
* Output only under /Volumes/WD_BLACK/antelligence-v4.31/{runs,data}/ or docs/research/v4.31-20261016/. Nothing under v4.1-v4.3 is written.
* S is threshold-blind (it releases on the local scan), so it runs once. S' plans against a threshold: with units it runs once per
  ``--plan-thrs`` value, and S' planned for T and scored at T is the attainable ceiling at T. Legacy S' plans against its D_t.
* Entry and motion are UNCHANGED and UNPHYSICAL: squads still walk through tissue carrying their whole dose.
* This script never imports v41_pod or v41_prod, reads no key and calls no model or network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v431_phaseA as PA  # noqa: E402
from antelligence.worlds.tumor_mri import ScriptS, TumorMRIWorld, WorldConfig, oracle_for, run_episode  # noqa: E402
from antelligence.worlds.tumor_mri.store_adapter import load_case  # noqa: E402

CASES_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")      # read-only
ALLOWED_OUT = (Path("/Volumes/WD_BLACK/antelligence-v4.31/runs"), Path("/Volumes/WD_BLACK/antelligence-v4.31/data"),
               REPO / "docs" / "research" / "v4.31-20261016")      # NOT the worktrees folder: it holds config/v4_3 and the v4.1-v4.3 evidence


def assert_out_allowed(out: Path) -> None:
    o = Path(out).resolve()
    if not any(o == a.resolve() or a.resolve() in o.parents for a in ALLOWED_OUT):
        raise SystemExit(f"refused: output {o} is not under the v4.31 runs/evidence folders")


def dev_cases() -> List[Dict[str, str]]:
    sp = json.loads((REPO / "config" / "v4_1" / "split.json").read_text())
    return [{"case": c, "split": s} for s in ("dev_a", "dev_b") for c in sp[s]]


def build_config(physics: str, k: str, plan_thr: Optional[float]) -> WorldConfig:
    if physics == "legacy":
        return WorldConfig.from_yaml(REPO / "config" / "v4_31" / "world.yaml")
    doc = yaml.safe_load((REPO / "config" / "v4_31" / f"world_units_k_{k}.yaml").read_text())["world"]
    if plan_thr is not None:
        doc["physics"] = {**doc["physics"], "policy_threshold_umol_h_L": float(plan_thr)}
    return WorldConfig.from_dict(doc)


def run_one(cfg: WorldConfig, arm: str, lc: Any, seed: int, thresholds: List[float]) -> Dict[str, Any]:
    world = TumorMRIWorld(lc.crop, cfg, seed=seed, regions=lc.regions, scorer_channels=lc.scorer_channels)
    brain = lc.crop.brain
    if not cfg.units:   # the same stepper arithmetic, plus off-target accounting under the +/-1-voxel band
        from antelligence.worlds.tumor_mri.physics import DrugField  # noqa: F401
        bands = {k: m & brain for k, m in PA.band_masks(lc.crop.truth).items()}
        world.field = PA.BandedDrugField(world.shape, release_sigma_vox=cfg.release_sigma_mm / lc.crop.voxel_mm,
                                         diffuse_sigma_vox=cfg.diffuse_sigma_vox, decay=cfg.decay, bands=bands)
    pol = oracle_for(world) if arm == "Sp" else None
    t0 = time.time()
    m = run_episode(world, (lambda a: pol) if pol is not None else (lambda a: ScriptS()))
    dt = time.time() - t0
    dose = world.field.dose
    truth = lc.crop.truth
    ch = lc.scorer_channels
    curves = PA.evaluate_dose(dose, truth, ch, lc.crop.outside_crop_truth_voxels, thresholds)
    tv = dose[truth]
    rec: Dict[str, Any] = {
        "n_truth_voxels": int(truth.sum()), "outside_crop_truth_voxels": int(lc.crop.outside_crop_truth_voxels),
        "n_regions": int(len(world.regions)), "total_drug": float(world.total_drug),
        "metrics": {k: m[k] for k in ("coverage", "off_target_share", "released_total", "decayed", "lost_boundary", "lost_boundary_share",
                                       "mass_residual", "time_to_50_coverage")},
        "curves": curves, "off_target": PA.off_target_by_band(world),
        "dose_truth": {"median": float(np.median(tv)) if tv.size else 0.0, "p10": float(np.percentile(tv, 10)) if tv.size else 0.0,
                       "p90": float(np.percentile(tv, 90)) if tv.size else 0.0, "max_all": float(dose.max())},
    }
    if cfg.units:
        f = world.field
        rec["ledger"] = f.ledger()
        rec["release_voxel_share"] = f.release_voxel_share()
        rec["n_releases"] = len(f._rel)
    rec["_seconds"] = dt
    rec["_dose"] = dose
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--physics", choices=("legacy", "units"), required=True)
    ap.add_argument("--k", choices=("low", "high"), default="low")
    ap.add_argument("--arms", default="S,Sp")
    ap.add_argument("--plan-thrs", default="", help="units S': comma list of planning thresholds (umol*h/L)")
    ap.add_argument("--seeds", default="9,10")
    ap.add_argument("--cases", default="dev", help="'dev' (all 40) or a comma list of development case ids")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-dose", action="store_true")
    a = ap.parse_args(argv)
    assert_out_allowed(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(",") if x]
    seeds = [int(s) for s in a.seeds.split(",") if s]
    si, sn = (int(x) for x in a.shard.split("/"))
    cases = dev_cases()
    if a.cases != "dev":
        want = set(a.cases.split(","))
        cases = [c for c in cases if c["case"] in want]
        if len(cases) != len(want):
            raise SystemExit(f"refused: {sorted(want - {c['case'] for c in cases})} is not a development case")
    cases = cases[si::sn]
    thresholds = list(PA.THRESHOLDS_UMOL_H_L if a.physics == "units" else PA.THRESHOLDS_LEGACY)
    plan = [float(x) for x in a.plan_thrs.split(",") if x] if a.physics == "units" else [None]
    out_file = a.out / f"runs_shard{si}of{sn}.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["arm"], r["plan_thr"], r["case"], r["seed"]))
    (a.out / "dose").mkdir(exist_ok=True) if a.save_dose else None
    n_new = 0
    with out_file.open("a") as fh, (a.out / f"timing_shard{si}of{sn}.jsonl").open("a") as tf:
        for c in cases:
            lc = load_case(c["case"], root=CASES_ROOT, repo_root=REPO)
            for seed in seeds:
                for arm in arms:
                    for pt in (plan if (arm == "Sp" and a.physics == "units") else [None]):
                        key = (arm, pt, c["case"], seed)
                        if key in done:
                            continue
                        cfg = build_config(a.physics, a.k, pt)
                        rec = run_one(cfg, arm, lc, seed, thresholds)
                        dose = rec.pop("_dose")
                        secs = rec.pop("_seconds")
                        rec.update({"label": a.out.name, "physics": a.physics, "k": a.k if a.physics == "units" else None, "arm": arm,
                                    "plan_thr": pt, "case": c["case"], "split": c["split"], "seed": seed,
                                    "config_sha": PA.digest(cfg.to_dict()), "thresholds": thresholds})
                        fh.write(json.dumps(rec, sort_keys=True) + "\n")
                        fh.flush()
                        tf.write(json.dumps({"case": c["case"], "seed": seed, "arm": arm, "plan_thr": pt, "seconds": secs}) + "\n")
                        if a.save_dose:
                            np.savez_compressed(a.out / "dose" / f"{arm}_p{pt}_{c['case']}_s{seed}.npz", dose=dose.astype(np.float32))
                        n_new += 1
                        print(f"{a.out.name} {c['case']} s{seed} {arm} plan={pt}: cov@policy={rec['metrics']['coverage']:.3f} ({secs:.1f}s)", flush=True)
    print(f"done: {n_new} new runs -> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
