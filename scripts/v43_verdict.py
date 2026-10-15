#!/usr/bin/env python
"""v4.3 Phase 0 verdict driver (goal section 4). All arms, the v4.3 config, development cases only, one uptake level.

    python scripts/v43_verdict.py --out runs/verdict-mid --level mid --dev-a 30 --dev-b 10 --seeds 5,6 [--workers 4]
    python scripts/v43_verdict.py --out runs/curve-low  --level low --dev-a 30 --dev-b 10 --seeds 5,6 \\
        --arms S,S-solo,S-marks,S-talk,"S'"

Arms (default): S, S-solo, S-marks, S-talk, S-marks-displaced, S-talk-shuffled, S', S-marks-perfect (diagnostic), S-marks-quiet
(quiet pair). Config: ``config/v4_3/world.yaml``; its fact flags must be on.

FREEZE: a sharing arm or a control (anything but S, S-solo, S') on a development case runs only if ``--freeze-file`` (default
``docs/research/v4.3-20261015/freeze.json``) exists and its ``freeze_hash`` equals the current code's. Otherwise the driver exits
with status 2 before it loads anything. Gate-A style runs (S and S-solo only) do not need the file. ``--ticks`` shortens the
episodes for dry runs only; it is recorded in ``results.json`` meta (``dry_overrides``).

Resumable (an episode whose ``record.json`` is complete is skipped), process pool of at most 6 workers, then: assemble
``results.json`` -> analyse -> ``REPORT.md`` -> S' ceiling check (``verdict_checks.json``) -> ``stats.json``. Everything is
written under ``--out`` (``--report`` can point the report elsewhere, never into a v4.1 / v4.2 folder).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v43_arms  # noqa: E402
from antelligence.experiments import v43_drive as drive  # noqa: E402
from antelligence.experiments.v41_gate import load_checked_split  # noqa: E402
from antelligence.experiments.v42_harness import EvalRefused  # noqa: E402
from antelligence.experiments.v42_report import compute_stats  # noqa: E402
from antelligence.experiments.v43_freeze import (DEFAULT_FREEZE_FILE, FreezeRefused, freeze_hash, freeze_needed,  # noqa: E402
                                                 head_commit, require_frozen)
from antelligence.experiments.v43_harness import code_sha_v43  # noqa: E402
from antelligence.experiments.v43_report import render  # noqa: E402
from antelligence.experiments.v43_results import write_results  # noqa: E402
from antelligence.experiments.v43_stats import CEILING_SHARE, ceiling_check  # noqa: E402
from antelligence.worlds.tumor_mri.combined import DEFAULT_COEFFICIENTS_PATH  # noqa: E402

VERDICT_ARMS = ("S", "S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S'")
EXTRA_ARMS = ("S-marks-perfect", "S-marks-quiet")
DEFAULT_ARMS = VERDICT_ARMS + EXTRA_ARMS


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--level", default="mid", choices=drive.LEVELS, help="uptake level (mid is primary; low and high are the curve)")
    ap.add_argument("--cases", default="", help="comma list of development case ids")
    ap.add_argument("--dev-a", type=int, default=0, help="take the first N dev-A cases")
    ap.add_argument("--dev-b", type=int, default=0, help="take the first N dev-B cases")
    ap.add_argument("--seeds", default="5,6")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--workers", type=int, default=4, help=f"at most {drive.MAX_WORKERS}")
    ap.add_argument("--config", type=Path, default=drive.DEFAULT_CONFIG)
    ap.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS_PATH)
    ap.add_argument("--cases-root", type=Path, default=drive.DEFAULT_CASES_ROOT if drive.DEFAULT_CASES_ROOT.is_dir() else None)
    ap.add_argument("--freeze-file", type=Path, default=DEFAULT_FREEZE_FILE)
    ap.add_argument("--ticks", type=int, default=None, help="override episode length (dry runs only; recorded in meta)")
    ap.add_argument("--report", type=Path, default=None, help="REPORT.md path (default: <out>/REPORT.md)")
    ap.add_argument("--cite", default="results.json", help="committed path the report cites for results.json")
    ap.add_argument("--resamples", type=int, default=None, help="bootstrap resamples (default: the v4.2 10,000)")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    split = load_checked_split(REPO)
    cases = [c.strip() for c in a.cases.split(",") if c.strip()] + split["dev_a"][:a.dev_a] + split["dev_b"][:a.dev_b]
    cases = list(dict.fromkeys(cases))
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    v43_arms.register_all()
    try:
        arms = [v43_arms.resolve_arm(x.strip()).name for x in a.arms.split(",") if x.strip()]
        drive.guard_dev_cases(cases)
        drive.assert_out_allowed(a.out)
        if a.report is not None:
            drive.assert_out_allowed(a.report)
    except (EvalRefused, KeyError, drive.OutRefused) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if not cases or not seeds:
        print("no cases or seeds given", file=sys.stderr)
        return 2
    if a.workers > drive.MAX_WORKERS:
        print(f"refused: at most {drive.MAX_WORKERS} workers", file=sys.stderr)
        return 2
    needs_freeze = freeze_needed(arms, on_dev_case=True)         # every case here passed the dev-case guard
    freeze_doc: Dict[str, Any] = {}
    if needs_freeze:
        try:
            freeze_doc = require_frozen(a.freeze_file)
        except FreezeRefused as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
    wc, cc, rules = drive.load_configs(a.config)
    if not wc.facts.any_on:
        print("refused: the config has every hidden fact off", file=sys.stderr)
        return 2
    tasks = [{"kind": "episode", "arm": arm, "case_id": c, "seed": s, "config": str(a.config), "out": str(a.out),
              "cases_root": str(a.cases_root) if a.cases_root else None, "ticks": a.ticks, "board": True,
              "uptake_level": a.level, "coefficients": str(a.coefficients), "freeze_file": str(a.freeze_file)}
             for c in cases for s in seeds for arm in arms]
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    times: List[Dict[str, Any]] = []
    print(f"{len(tasks)} episodes ({len(arms)} arms x {len(cases)} cases x {len(seeds)} seeds), level {a.level}, {a.workers} workers",
          flush=True)

    def on_result(task: Mapping[str, Any], r: Dict[str, Any]) -> None:
        if r.get("skipped"):
            print(f"  skip {task['arm']:18s} {task['case_id']:10s} seed {task['seed']} (done)", flush=True)
            return
        times.append({"arm": r["arm"], "case_id": r["case_id"], "seed": r["seed"], **r["timing_s"]})
        print(f"  ran  {r['arm']:18s} {r['case_id']:10s} seed {r['seed']}  coverage {r['coverage']:.3f}  {r['timing_s']['wall']:.2f}s", flush=True)

    drive.run_pool(tasks, a.workers, on_result)

    meta = {"study": "v4.3", "level": a.level, "config": str(a.config), "config_sha256": sha256_file(a.config),
            "frozen_facts": wc.facts.__dict__, "ant_rules": rules.to_dict(), "ant_rules_sha": rules.sha(),
            "code_sha": code_sha_v43(), "coefficients": str(a.coefficients), "coefficients_sha256": sha256_file(a.coefficients),
            "freeze": {"required": needs_freeze, "file": str(a.freeze_file), "freeze_hash_recorded": freeze_doc.get("freeze_hash"),
                       "commit_recorded": freeze_doc.get("commit"), "freeze_hash_now": freeze_hash(), "commit_now": head_commit()},
            "dry_overrides": {"ticks": a.ticks}, "workers": a.workers, "seeds": seeds, "arms": arms, "cases": cases}
    results = write_results(a.out, truth_for=lambda c: drive.case_truth(c, a.cases_root), meta=meta)
    results["meta"]["ceiling_check"] = chk = ceiling_check(results["runs"], arms)
    (a.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True, default=str))
    (a.out / "verdict_checks.json").write_text(json.dumps({"ceiling": chk, "episode_times": times}, indent=1, sort_keys=True))
    stats = compute_stats(results, **({"n": a.resamples} if a.resamples else {}))
    text = render(results, stats, cite=a.cite)
    report = a.report or (a.out / "REPORT.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text)
    if stats is not None:
        (a.out / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True, default=str))
    if not chk["ok"]:
        print(f"LOG: S' >= every verdict arm on only {chk['share_s_prime_ge_all']} of {chk['n_case_seeds']} case-seeds "
              f"(need {CEILING_SHARE}); see verdict_checks.json", file=sys.stderr)
    if times:
        w = sorted(t["wall"] for t in times)
        print(f"episodes run {len(w)}; wall per episode median {w[len(w) // 2]:.2f}s max {w[-1]:.2f}s; total {time.perf_counter() - t0:.1f}s")
    print(f"results {a.out / 'results.json'}\nreport  {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
