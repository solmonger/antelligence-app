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


GROUPS = ("dev_a", "dev_b", "eval", "reserve")
DEFAULT_GROUPS = ("dev_a", "dev_b", "eval")


def parse_cases(cases: str, dev_a: str | None = None, dev_b: str | None = None,
                include: tuple[str, ...] = DEFAULT_GROUPS) -> dict[str, list[str]]:
    """Return {group: ids} for the requested groups. ``reserve`` only if explicitly included.

    ``cases`` is config/v4_1/split.json (keys dev_a, dev_b, eval, reserve) or an id list/file,
    in which case ids go to ``eval`` and dev-A/dev-B come from --dev-a/--dev-b.
    """
    p = Path(cases)
    if p.is_file() and p.suffix == ".json":
        d = json.loads(p.read_text())
        groups = {g: list(d.get(g, [])) for g in GROUPS}
    else:
        groups = {"dev_a": [], "dev_b": [], "eval": _ids(cases), "reserve": []}
    if dev_a:
        groups["dev_a"] = _ids(dev_a)
    if dev_b:
        groups["dev_b"] = _ids(dev_b)
    seen: set[str] = set()
    for g in ("dev_a", "dev_b", "eval", "reserve"):
        dup = seen & set(groups[g])
        if dup:
            raise SystemExit(f"case(s) in more than one group: {sorted(dup)[:3]}")
        seen |= set(groups[g])
    return {g: groups[g] for g in GROUPS if g in include or g in ("dev_a", "dev_b")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", help="split file (json/text) or comma-separated case ids")
    ap.add_argument("--dev-a", help="dev-A ids/file (calibrator candidates are fit here)")
    ap.add_argument("--dev-b", help="dev-B ids/file (held-out calibrator selection)")
    ap.add_argument("--include", default=",".join(DEFAULT_GROUPS),
                    help="groups to process: dev_a,dev_b,eval,reserve (reserve only when asked)")
    ap.add_argument("--eval-metrics", action="store_true",
                    help="compute truth metrics for eval/reserve cases (ONLY after the preregistration lock)")
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
        s = pipeline.summarise(out, base / ('calibration_missing_flair.json' if args.missing_flair else 'calibration.json'))
        dest = Path(args.summary_path)
        if args.missing_flair:
            dest = dest.with_name("summary_missing_flair.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(s, indent=1) + "\n")
        print(f"wrote {dest} ")
        return 0

    if not args.cases:
        ap.error("--cases is required unless --summary")
    include = tuple(g.strip() for g in args.include.split(",") if g.strip())
    bad = set(include) - set(GROUPS)
    if bad:
        ap.error(f"unknown groups {sorted(bad)}")
    groups = parse_cases(args.cases, args.dev_a, args.dev_b, include)
    dev_a, dev_b = groups["dev_a"], groups["dev_b"]
    split_of = {c: g for g, ids in groups.items() for c in ids}
    all_ids = [c for ids in groups.values() for c in ids]
    metrics_ok = lambda c: split_of[c] in ("dev_a", "dev_b") or args.eval_metrics  # noqa: E731
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
            pipeline.infer_case(Path(args.msd_root), c, net, device, raw_dir, args.missing_flair,
                                truth_metrics=metrics_ok(c))
            print(f"[infer] {c} {time.time() - t0:.1f}s", flush=True)

    # 2. calibrator: candidates on dev-A, selection on held-out dev-B, refit on A+B
    if dev_a and dev_b:
        calibrator = pipeline.fit_from_dev(raw_dir, dev_a, dev_b, calib_path)
        print(f"calibrator={calibrator.method} T={calibrator.temperature:.3f} -> {calib_path}", flush=True)
    elif calib_path.exists():
        calibrator = cal.Calibrator.from_dict(json.loads(calib_path.read_text())["calibrator"])
    else:
        ap.error("need dev_a and dev_b (split file or --dev-a/--dev-b) to fit the calibrator")

    # 3. crop / regions / store
    for c in all_ids:
        if (out / c / "metrics.json").exists():
            continue
        t0 = time.time()
        m = pipeline.finalize_case(c, raw_dir, out, calibrator, split_of[c], truth_metrics=metrics_ok(c))
        extra = f" WT dice={m['dice_2mm']['WT']:.3f}" if "dice_2mm" in m else " (metrics withheld)"
        print(f"[store] {c} split={m['split']}{extra} regions={m['n_regions']} {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
