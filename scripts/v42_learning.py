#!/usr/bin/env python
"""Learning plumbing check (goal section 10, L1). Development cases only, scripts only, no model, no network.

    python scripts/v42_learning.py --out runs/learning --report docs/research/v4.2-20261014/learning

This is a PLUMBING CHECK, NOT A FINDING. Scripts nudging their own thresholds is not models learning lessons.

Three parts:
1. Carry. One colony of S-marks ants runs the 30 dev-A cases in split order, seed 1, frozen config ``config/v4_2/world.yaml``.
   After each run the 8 name cards are saved (``Card.save``); before the next run they are loaded (``Card.load``) and only the
   job THRESHOLDS are copied into the new run (card and script). Nothing else is carried: not marks, messages, readings,
   claims, notebooks or cross-run counts.
   INTERFACES 8 says cards reset at the start of every run. That rule is untouched everywhere else. The carry is an explicit
   option of THIS script only (``carry_thresholds`` below): it wraps ``ColonyKit`` for the duration of a carried run and
   overwrites the freshly reset thresholds. The harness, kernel and every other script are unchanged.
2. Measure. The 10 dev-B cases, seeds 3 and 4, S-marks started from the carried cards versus S-marks with fresh cards
   (the ordinary harness path), paired by (case, seed). Every carried measurement run starts from the SAME trained cards (it does
   not keep learning), so the pairs are independent. Reported: the median paired coverage difference and the case-clustered
   bootstrap from ``v42_stats`` (descriptive; not a verdict; not part of Holm).
3. Notebook round trip: write, cap, archive, freeze, reload, with the output saved.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v42_drive as drive  # noqa: E402
from antelligence.experiments import v42_harness as hmod  # noqa: E402
from antelligence.experiments.v41_gate import load_checked_split  # noqa: E402
from antelligence.experiments.v42_harness import V42Harness, load_dev_case  # noqa: E402
from antelligence.experiments.v42_stats import cluster_bootstrap, group_by_case, paired_diffs  # noqa: E402
import antelligence.experiments.v42_arms  # noqa: E402,F401
from antelligence.kernel.colony.cards import Card  # noqa: E402
from antelligence.kernel.colony.notebook import Notebook, NotebookError  # noqa: E402
from antelligence.kernel.canonical import canonical_json  # noqa: E402

ARM = "S-marks"
TRAIN_SEED = 1
MEASURE_SEEDS = (3, 4)
COMPARISON = "learning_carried_vs_fresh"
N_ANTS = 8


# --- carry (the explicit option; resets stay the rule everywhere else) -------------------------------------------------------

def load_carried(colony_dir: Path, n: int = N_ANTS) -> Dict[str, Dict[str, float]]:
    """Thresholds from the saved cards, via ``Card.load``. Nothing else is read from them."""
    return {f"ant-{i:02d}": dict(Card.load(colony_dir, f"ant-{i:02d}").thresholds) for i in range(n)}


@contextlib.contextmanager
def carry_thresholds(carried: Mapping[str, Mapping[str, float]]) -> Iterator[None]:
    """For the duration of the block, every ColonyKit the harness builds starts from ``carried`` thresholds (card and script)
    instead of 0.5. Restores the harness on exit."""
    original = hmod.ColonyKit

    class CarriedKit(original):                                    # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            for i, card in enumerate(self.cards):
                th = dict(carried[card.ant_id])
                card.thresholds = dict(th)
                self.scripts[i].state.thresholds.theta = dict(th)

    hmod.ColonyKit = CarriedKit
    try:
        yield
    finally:
        hmod.ColonyKit = original


def save_cards(record: Mapping[str, Any], colony_dir: Path) -> None:
    for card in record["cards_final"].values():
        Card.from_dict(card).save(colony_dir)


def _harness_for(config: Path, case: Any, ticks: Optional[int]) -> V42Harness:
    wc, cc = drive.load_configs(config)
    if ticks:
        wc = wc.with_(ticks=int(ticks))
    wc = wc.with_(facts=drive.facts_for(wc.facts, case.scorer_channels is not None))
    return V42Harness(wc, cc, with_board=True)


# --- parts 1 and 2 -----------------------------------------------------------------------------------------------------------

def train(cases: Sequence[str], colony_dir: Path, *, config: Path, cases_root: Optional[Path], ticks: Optional[int],
          log=print) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k, cid in enumerate(cases):
        case = load_dev_case(cid, root=cases_root)
        h = _harness_for(config, case, ticks)
        t0 = time.perf_counter()
        if k == 0:
            rec = h.run_episode(ARM, case, TRAIN_SEED)                        # first run: fresh cards (no saved cards yet)
            start = {f"ant-{i:02d}": {j: 0.5 for j in ("dose", "explore", "verify")} for i in range(N_ANTS)}
        else:
            start = load_carried(colony_dir)
            with carry_thresholds(start):
                rec = h.run_episode(ARM, case, TRAIN_SEED)
        save_cards(rec, colony_dir)
        end = {a: dict(c["thresholds"]) for a, c in rec["cards_final"].items()}
        rows.append({"run": k + 1, "case_id": cid, "seed": TRAIN_SEED, "coverage": rec["metrics"]["coverage"],
                     "start_mean_theta": _mean_theta(start), "end_mean_theta": _mean_theta(end),
                     "end_thresholds": end, "wall_s": round(time.perf_counter() - t0, 2)})
        log(f"train {k + 1:2d}/{len(cases)} {cid:10s} coverage {rec['metrics']['coverage']:.3f} "
            f"theta {rows[-1]['start_mean_theta']} -> {rows[-1]['end_mean_theta']}")
    return rows


def _mean_theta(th: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    return {j: round(sum(t[j] for t in th.values()) / len(th), 4) for j in ("dose", "explore", "verify")}


def measure(cases: Sequence[str], colony_dir: Path, *, config: Path, cases_root: Optional[Path], ticks: Optional[int],
            n_resamples: int, log=print) -> Dict[str, Any]:
    carried = load_carried(colony_dir)
    runs: List[Dict[str, Any]] = []
    cov: Dict[str, Dict[Any, float]] = {"carried": {}, "fresh": {}}
    for cid in cases:
        case = load_dev_case(cid, root=cases_root)
        h = _harness_for(config, case, ticks)
        for seed in MEASURE_SEEDS:
            fresh = h.run_episode(ARM, case, seed)
            with carry_thresholds(carried):
                car = h.run_episode(ARM, case, seed)
            for label, rec in (("fresh", fresh), ("carried", car)):
                cov[label][(cid, seed)] = float(rec["metrics"]["coverage"])
                runs.append({"colony": label, "case_id": cid, "seed": seed, "coverage": float(rec["metrics"]["coverage"])})
            log(f"measure {cid:10s} seed {seed}  fresh {fresh['metrics']['coverage']:.3f}  carried {car['metrics']['coverage']:.3f}")
    diffs = paired_diffs(cov, "carried", "fresh")
    boot = cluster_bootstrap(group_by_case(diffs), COMPARISON, n=n_resamples)
    return {"runs": runs, "paired_differences": [{"case_id": c, "seed": s, "carried_minus_fresh": d} for (c, s), d in sorted(diffs.items())],
            "median_paired_difference": boot["median"], "bootstrap": boot,
            "note": "descriptive only; not a verdict; not part of Holm"}


# --- part 3 ------------------------------------------------------------------------------------------------------------------

def notebook_round_trip(workdir: Path) -> Dict[str, Any]:
    """Write, cap, archive, freeze, reload. Returns a step-by-step log (every step records what happened, errors included)."""
    steps: List[Dict[str, Any]] = []
    nb = Notebook("ant-00", cap_chars=1500)

    def step(name: str, **kw: Any) -> None:
        steps.append({"step": name, "active_chars": nb.active_chars(), "n_lessons": len(nb.lessons), "frozen": nb.frozen, **kw})

    a = nb.add("Dose spots where two ants agree before releasing a large amount.", ["demo-1"])
    b = nb.add("Check grey spots near a disputed mark before moving on.", ["demo-1"])
    nb.patch(b, "Check grey spots within 20 mm of a disputed mark before moving on.")
    step("write", ids=[a, b], snapshot=nb.snapshot())
    n_ok = 0
    refused = None
    try:
        while True:
            nb.add("x" * 200, ["demo-fill"])
            n_ok += 1
    except NotebookError as exc:
        refused = str(exc)
    step("cap", filler_lessons_accepted=n_ok, refused_with=refused, cap_chars=nb.cap_chars)
    nb.remove(a)
    nb.pin(b)
    nb.mark_used(b, effect=0.02)
    gone = nb.curate(run_index=20, stale_runs=10)
    step("archive", archived_by_remove=a, archived_by_curate=gone, pinned_kept=b,
         statuses={l["id"]: l["status"] for l in nb.lessons}, lessons_kept_in_file=len(nb.lessons))
    nb.freeze()
    refusals = {}
    for name, fn in (("add", lambda: nb.add("late lesson")), ("patch", lambda: nb.patch(b, "changed")), ("remove", lambda: nb.remove(b))):
        try:
            fn()
            refusals[name] = "NOT REFUSED"
        except NotebookError as exc:
            refusals[name] = str(exc)
    step("freeze", writes_after_freeze=refusals, snapshot=nb.snapshot())
    path = nb.save(workdir / "notebook_ant-00.json")
    back = Notebook.load(path)
    same = back.to_dict() == nb.to_dict()
    step("reload", path=path.name, identical_to_saved=same, frozen_after_reload=back.frozen, snapshot=back.snapshot(),
         archived_still_in_file=[l["id"] for l in back.lessons if l["status"] == "archived"][:3] + ["..."],
         export_skills=back.export_skills())
    return {"steps": steps, "ok": bool(same and back.frozen and all(v != "NOT REFUSED" for v in refusals.values()))}


# --- driver ------------------------------------------------------------------------------------------------------------------

README = """# Learning plumbing check (v4.2, goal section 10)

**This is a plumbing check, not a finding.** The "learning" here is a script nudging its own job thresholds (a number goes
down a little when a job works, up a little when it does not) and a file being saved and loaded. No model is involved and
nothing was learned in the sense of a model keeping a lesson. Do not cite the number below as evidence that ants learn.

What was run (development cases only, scripts only, no model, no network, seed and config as stated):

1. One colony of 8 S-marks ants ran the 30 dev-A cases in split order, seed 1, frozen `config/v4_2/world.yaml`. After each
   run the 8 name cards were saved and the next run loaded them back; only the job thresholds were carried. Nothing else was.
   The v4.2 rule that cards reset every run is unchanged everywhere else; carrying is an explicit option of
   `scripts/v42_learning.py` only.
2. The 10 dev-B cases, seeds 3 and 4: S-marks started from those carried thresholds versus S-marks with fresh cards, paired by
   (case, seed). Every carried run starts from the same trained cards. Descriptive only; not a verdict, not part of Holm.
3. A notebook round trip (write, cap, archive, freeze, reload) is in `results.json` under `notebook_round_trip`.

{result}

Files: `results.json` (everything), `cards/` (final saved name cards), `notebook_ant-00.json` (the demo notebook).
"""


def run(out: Path, report: Path, *, config: Path, cases_root: Optional[Path], ticks: Optional[int] = None,
        n_dev_a: Optional[int] = None, n_dev_b: Optional[int] = None, n_resamples: int = 10_000, log=print) -> Dict[str, Any]:
    split = load_checked_split(REPO)
    dev_a, dev_b = split["dev_a"][:n_dev_a], split["dev_b"][:n_dev_b]
    drive.guard_dev_cases(dev_a + dev_b)
    colony_dir = out / "colony"
    t0 = time.perf_counter()
    training = train(dev_a, colony_dir, config=config, cases_root=cases_root, ticks=ticks, log=log)
    meas = measure(dev_b, colony_dir, config=config, cases_root=cases_root, ticks=ticks, n_resamples=n_resamples, log=log)
    nbook = notebook_round_trip(out)
    wc, _ = drive.load_configs(config)
    results = {
        "kind": "plumbing_check_not_a_finding", "arm": ARM, "config": str(config.relative_to(REPO) if config.is_relative_to(REPO) else config),
        "train_seed": TRAIN_SEED, "measure_seeds": list(MEASURE_SEEDS), "ticks_override": ticks,
        "dev_a_cases_in_split_order": dev_a, "dev_b_cases": dev_b,
        "carried_state": "job thresholds only (via Card.save/Card.load); cards otherwise reset every run (INTERFACES 8)",
        "training": training, "final_cards": {f"ant-{i:02d}": Card.load(colony_dir, f"ant-{i:02d}").to_dict() for i in range(N_ANTS)},
        "measure": meas, "notebook_round_trip": nbook, "wall_s": round(time.perf_counter() - t0, 1),
        "facts": wc.facts.__dict__}
    for d in (out, report):
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True, default=str))
        if d is report:
            for i in range(N_ANTS):
                Card.load(colony_dir, f"ant-{i:02d}").save(report / "colony_cards")
    b = meas["bootstrap"]
    result = (f"Result (descriptive): median paired coverage difference, carried minus fresh, {b['n_pairs']} pairs over {b['n_cases']} "
              f"cases: {b['median']:+.4f} (95% case-clustered bootstrap interval {b['ci_low']:+.4f} to {b['ci_high']:+.4f}, "
              f"{b['n_resamples']} resamples). Notebook round trip ok: {nbook['ok']}.") if b["median"] is not None else "No pairs."
    (report / "README.md").write_text(README.format(result=result))
    (out / "README.md").write_text(README.format(result=result))
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path, help="docs directory for results.json and README.md")
    ap.add_argument("--config", type=Path, default=drive.DEFAULT_CONFIG)
    ap.add_argument("--cases-root", type=Path, default=drive.DEFAULT_CASES_ROOT if drive.DEFAULT_CASES_ROOT.is_dir() else None)
    ap.add_argument("--ticks", type=int, default=None, help="override the episode length (dry runs only)")
    ap.add_argument("--dev-a", type=int, default=None, help="first N dev-A cases (dry runs only)")
    ap.add_argument("--dev-b", type=int, default=None, help="first N dev-B cases (dry runs only)")
    ap.add_argument("--resamples", type=int, default=10_000)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    run(a.out, a.report, config=a.config, cases_root=a.cases_root, ticks=a.ticks, n_dev_a=a.dev_a, n_dev_b=a.dev_b,
        n_resamples=a.resamples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
