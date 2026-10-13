#!/usr/bin/env python
"""v4.1 perception CLI: MSD Task01 cases -> calibrated tumour maps, crops and regions.

Resumable: existing raw/<case>.npz and cases/<case>/metrics.json are skipped.

  .venv/bin/python scripts/v41_perception.py --cases splits.json --device mps --out data/v4_1/cases
  .venv/bin/python scripts/v41_perception.py --summary --out data/v4_1/cases

A split file is JSON {"dev": [...], "eval": [...]} (calibrator is fit on dev only) or a text file with
one case id per line (then pass --dev). --cases also accepts a comma-separated id list.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.perception import calibrate as cal  # noqa: E402
from antelligence.perception import pipeline  # noqa: E402

SUMMARY_PATH = REPO / "docs" / "research" / "v4.1-20261013" / "perception" / "summary.json"


def _ids(spec: str) -> list[str]:
    p = Path(spec)
    if p.is_file():
        return [s.strip() for s in p.read_text().splitlines() if s.strip() and not s.startswith("#")]
    return [s.strip() for s in spec.split(",") if s.strip()]


def parse_cases(cases: str, dev: str | None) -> tuple[list[str], list[str], list[str]]:
    """Return (all_ids, dev_ids, eval_ids)."""
    p = Path(cases)
    if p.is_file() and p.suffix == ".json":
        d = json.loads(p.read_text())
        dev_ids, eval_ids = list(d.get("dev", [])), list(d.get("eval", []))
    else:
        dev_ids, eval_ids = [], _ids(cases)
    if dev:
        dev_ids = _ids(dev)
    eval_ids = [c for c in eval_ids if c not in set(dev_ids)]
    return dev_ids + eval_ids, dev_ids, eval_ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", help="split file (json/text) or comma-separated case ids")
    ap.add_argument("--dev", help="dev case ids/file for calibration (overrides split file)")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--out", default=str(REPO / "data" / "v4_1" / "cases"))
    ap.add_argument("--msd-root", default=str(REPO / "data" / "v4_1" / "Task01_BrainTumour"))
    ap.add_argument("--models-dir", default=None, help="default <out>/../models")
    ap.add_argument("--missing-flair", action="store_true", help="zero FLAIR before inference (separate output dirs)")
    ap.add_argument("--summary", action="store_true", help="write aggregate summary.json and exit")
    ap.add_argument("--summary-path", default=str(SUMMARY_PATH))
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.missing_flair:
        out = out.with_name(out.name + "_missing_flair")
    base = out.parent
    raw_dir = base / ("raw_missing_flair" if args.missing_flair else "raw")

    if args.summary:
        s = pipeline.summarise(out)
        dest = Path(args.summary_path)
        if args.missing_flair:
            dest = dest.with_name("summary_missing_flair.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(s, indent=1) + "\n")
        print(f"wrote {dest} (n={s['all']['n']})")
        return 0

    if not args.cases:
        ap.error("--cases is required unless --summary")
    all_ids, dev_ids, eval_ids = parse_cases(args.cases, args.dev)
    split_of = {**{c: "dev" for c in dev_ids}, **{c: "eval" for c in eval_ids}}
    models_dir = Path(args.models_dir) if args.models_dir else base / "models"
    calib_path = base / ("calibration_missing_flair.json" if args.missing_flair else "calibration.json")

    # 1. inference (skips cases already in raw/)
    todo = [c for c in all_ids if not (raw_dir / f"{c}.npz").exists()]
    if todo:
        from antelligence.perception import infer, model as mdl

        device = infer.select_device(args.device)
        net = mdl.load_bundle_model(models_dir, device)
        print(f"device={device} inferring {len(todo)}/{len(all_ids)} cases", flush=True)
        for c in todo:
            t0 = time.time()
            pipeline.infer_case(Path(args.msd_root), c, net, device, raw_dir, args.missing_flair)
            print(f"[infer] {c} {time.time() - t0:.1f}s", flush=True)

    # 2. calibrator: fit on dev only
    if calib_path.exists() and not dev_ids:
        calibrator = cal.Calibrator.from_dict(json.loads(calib_path.read_text())["calibrator"])
    else:
        if not dev_ids:
            ap.error("need dev cases (split file 'dev' key or --dev) to fit the calibrator")
        calibrator = pipeline.fit_from_dev(raw_dir, dev_ids, calib_path)
        print(f"calibrator={calibrator.method} T={calibrator.temperature:.3f} -> {calib_path}", flush=True)

    # 3. crop / regions / store
    for c in all_ids:
        if (out / c / "metrics.json").exists():
            continue
        t0 = time.time()
        m = pipeline.finalize_case(c, raw_dir, out, calibrator, split_of.get(c, "unsplit"))
        print(f"[store] {c} split={m['split']} WT dice={m['dice_2mm']['WT']:.3f} regions={m['n_regions']} "
              f"outside={m['truth_wt_share_outside_crop']:.3f} {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
