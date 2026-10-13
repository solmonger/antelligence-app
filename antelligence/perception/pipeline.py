"""Per-case orchestration: infer -> calibrate -> crop -> regions -> store. Resumable by file existence."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import MODEL_PROV, PIPELINE_VERSION, TARGET_SPACING_MM
from . import calibrate as cal
from . import crop as crp
from . import msd, regions, store

CHANNELS = ("WT", "TC", "ET")  # downstream channel order
CAVEAT = (
    "MSD Task01 cases are a subset of BraTS 2016/2017 data, and the MONAI brats_mri_segmentation "
    "bundle was trained on BraTS 2018, which overlaps these cases. Scores are therefore optimistic "
    "and are not an independent generalisation estimate."
)


def _stack(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32)


def infer_case(root: Path, case_id: str, model, device, raw_dir: Path, missing_flair: bool = False,
               roi_size=None, overlap=None, flips=None) -> Path:
    """Run TTA inference on one case and save the 2 mm raw bundle to raw_dir/<case>.npz."""
    from . import infer

    out = raw_dir / f"{case_id}.npz"
    if out.exists():
        return out
    t0 = time.time()
    case = msd.load_case(root, case_id)
    ds = msd.read_dataset_json(root)
    reorder = msd.assert_msd_channel_order(ds)
    ldef = msd.label_definition(ds)
    brain = msd.brain_mask(case.image)
    x = infer.prepare_input(case.image, reorder, case.channels if missing_flair else None)
    kw = {k: v for k, v in (("roi_size", roi_size), ("overlap", overlap), ("flips", flips)) if v is not None}
    mean_b, std_b = infer.predict_tta(model, x, device=device, **kw)
    mean = infer.to_msd_truth_order(mean_b)  # WT, TC, ET
    std = infer.to_msd_truth_order(std_b)
    truth = msd.truth_channels(case.label, ldef)
    full_dice = [cal.dice(mean[i] >= 0.5, truth[i]) for i in range(3)]

    sp = case.spacing_mm
    rs = lambda a, o=1: crp.resample(a, sp, TARGET_SPACING_MM, o)  # noqa: E731
    names_b = list(msd.BUNDLE_CHANNELS)
    raw = {
        "mean": np.stack([rs(m) for m in mean]).astype(np.float32),
        "std": np.stack([rs(s) for s in std]).astype(np.float32),
        "flair": rs(x[names_b.index("FLAIR")]).astype(np.float32),
        "t1c": rs(x[names_b.index("T1c")]).astype(np.float32),
        "brain": rs(brain.astype(np.float32)) >= 0.5,
        "truth": np.stack([rs(t.astype(np.float32), 0) >= 0.5 for t in truth]),
        "full_res_dice": np.asarray(full_dice, dtype=np.float64),
        "spacing_mm": np.asarray([TARGET_SPACING_MM] * 3),
        "seconds": np.asarray(time.time() - t0),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **raw)
    tmp.replace(out)
    return out


def load_raw(raw_dir: Path, case_id: str) -> dict:
    with np.load(raw_dir / f"{case_id}.npz") as z:
        return {k: z[k] for k in z.files}


def fit_from_dev(raw_dir: Path, dev_ids: list[str], out_path: Path) -> cal.Calibrator:
    """Fit the WT calibrator on brain voxels of DEV cases only and save it."""
    ps, ys = [], []
    for cid in dev_ids:
        r = load_raw(raw_dir, cid)
        b = r["brain"]
        ps.append(r["mean"][0][b])
        ys.append(r["truth"][0][b])
    calibrator, report = cal.fit_calibrator(np.concatenate(ps), np.concatenate(ys))
    out_path.write_text(json.dumps({"calibrator": calibrator.to_dict(), "dev_cases": sorted(dev_ids),
                                    "report": report}, indent=1))
    return calibrator


def finalize_case(case_id: str, raw_dir: Path, out_root: Path, calibrator: cal.Calibrator, split: str) -> dict:
    """Calibrate, crop, extract regions, write the per-case store + metrics.json."""
    r = load_raw(raw_dir, case_id)
    spacing = float(r["spacing_mm"][0])
    brain = r["brain"]
    p_raw = r["mean"][0].astype(np.float64)
    p_cal = calibrator.apply(p_raw).astype(np.float32)
    truth = r["truth"]

    box = crp.compute_crop(p_cal, brain, spacing)
    found = box is not None
    if box is None:  # nothing above threshold: keep the brain bounding box so downstream still has data
        idx = np.nonzero(brain)
        box = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx) if idx[0].size else tuple(
            slice(0, s) for s in brain.shape)
        outside = crp.truth_share_outside(truth[0], None)
    else:
        outside = crp.truth_share_outside(truth[0], box)

    sub = lambda a: np.ascontiguousarray(a[box])  # noqa: E731
    p_c, u_c = sub(p_cal), sub(r["std"][0])
    ptc, pet = sub(r["mean"][1]), sub(r["mean"][2])
    payloads, _ = regions.extract_regions(p_c, spacing, unc=u_c, p_tc=ptc, p_et=pet)
    origin = [int(s.start) for s in box]
    case_dir = out_root / case_id
    store.save_case(
        case_id, out_root,
        brain_mask=sub(brain), p_tumour=p_c, unc=u_c, flair=sub(r["flair"]), t1c=sub(r["t1c"]),
        truth=np.stack([sub(t) for t in truth]), p_tc=ptc, p_et=pet,
        origin_vox=np.asarray(origin), spacing_mm=np.asarray([spacing] * 3),
    )

    bv = brain
    y_wt = truth[0][bv]
    metrics = {
        "case": case_id, "split": split, "pipeline": PIPELINE_VERSION, "prov": MODEL_PROV,
        "calibrator": calibrator.method,
        "dice_2mm": {"WT": cal.dice(p_cal >= 0.5, truth[0]),
                     "TC": cal.dice(r["mean"][1] >= 0.5, truth[1]),
                     "ET": cal.dice(r["mean"][2] >= 0.5, truth[2])},
        "dice_full_res_raw": dict(zip(CHANNELS, map(float, r["full_res_dice"]))),
        "ece_wt_raw": cal.ece(p_raw[bv], y_wt), "ece_wt_cal": cal.ece(p_cal[bv], y_wt),
        "brier_wt_raw": cal.brier(p_raw[bv], y_wt), "brier_wt_cal": cal.brier(p_cal[bv], y_wt),
        "truth_wt_share_outside_crop": outside, "crop_found": found,
        "crop_origin_vox": origin, "crop_shape_vox": [int(s.stop - s.start) for s in box],
        "n_regions": len(payloads), "regions": payloads,
        "seconds_infer": float(r["seconds"]),
    }
    (case_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def summarise(out_root: Path, only_split: str | None = None) -> dict:
    """Aggregates only (no per-voxel data, no per-case rows)."""
    ms = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/metrics.json"))]

    def agg(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0}

        def mm(vals):
            v = np.asarray(vals, dtype=np.float64)
            return {"mean": float(v.mean()), "median": float(np.median(v))}

        return {
            "n": len(rows),
            "dice_2mm": {c: mm([m["dice_2mm"][c] for m in rows]) for c in CHANNELS},
            "dice_full_res_raw": {c: mm([m["dice_full_res_raw"][c] for m in rows]) for c in CHANNELS},
            "ece_wt_raw": mm([m["ece_wt_raw"] for m in rows]),
            "ece_wt_cal": mm([m["ece_wt_cal"] for m in rows]),
            "brier_wt_raw": mm([m["brier_wt_raw"] for m in rows]),
            "brier_wt_cal": mm([m["brier_wt_cal"] for m in rows]),
            "truth_wt_share_outside_crop": mm([m["truth_wt_share_outside_crop"] for m in rows]),
            "seconds_per_case": mm([m["seconds_infer"] for m in rows]),
            "crop_found_fraction": float(np.mean([m["crop_found"] for m in rows])),
        }

    splits = sorted({m["split"] for m in ms})
    return {
        "pipeline": PIPELINE_VERSION, "prov": MODEL_PROV,
        "caveat_brats2018_overlap": CAVEAT,
        "note": "Dice/ECE/Brier are computed on the 2 mm grid (brain voxels for ECE/Brier); "
                "dice_full_res_raw is the uncalibrated 1 mm Dice at p>=0.5.",
        "all": agg(ms),
        **{f"split_{s}": agg([m for m in ms if m["split"] == s]) for s in splits},
    }
