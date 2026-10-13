"""Phase 0 fit (goal 5, steps 7, 9, 10 and 12): calibration logs -> frozen conformal thresholds and kNN router, plus the
power and throughput numbers. Development calibration logs only; an evaluation-case log is refused.

Input is one or more run output directories (each run has ``decisions.jsonl``, ``run_log.json`` and ``cell.json``) and,
optionally, ``cells.jsonl``. A calibration run logs, per decision, the driving model's answer (Qwen3.5-9B, or Bonsai 8B
for the Bonsai variant) at the row's top level and d1 and Qwen3.5-4B in ``counterfactual``, plus the S' labels and the
eleven router ``features``.

* Conformal: label = the driving model's decision on the same state, from both arms; ``freeze_thresholds`` does the
  per-question split-conformal fit with one joint alpha search for the ANY-of-3 defer rate (~20%).
* Router: label = the cheapest model whose three answers best match S' (ties to the cheaper), kNN k=15 on standardised
  features, trained on both arms. Reported: pick shares, leave-one-case-out agreement with the labels, and the share of
  decisions where each model matches S'.
* Power: paired SD of (SW-B - SW-noBoard) case coverage -> minimum detectable P1 difference via
  ``v41_analysis.power_analysis`` (n=80, alpha 0.05, power 0.8).
* Throughput: decisions per second and seconds per episode from the cells and the run logs.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from antelligence.experiments import v41_analysis, v41_fallbacks
from antelligence.experiments.v41_conformal import (DEFER_TARGET, QTYPES, ConformalDeferral, ConformalThresholds,
                                                    DevRecord, freeze_thresholds)
from antelligence.experiments.v41_router import (BONSAI_ROUTER_MODELS, FEATURE_NAMES, K, ROUTER_MODELS, KNNRouter,
                                                 TrainRow, best_model, match_count)
from antelligence.experiments.v41_split import load_split, which_split

D1, Q4, Q9, BONSAI = "d1", "qwen3.5-4b", "qwen3.5-9b", "bonsai-8b"
CALIBRATION_ARMS = ("SW-B", "SW-noBoard")
TARGET_BACKEND = {False: Q9, True: BONSAI}


class FitError(ValueError):
    """Refused input: an evaluation or non-calibration log, a missing field, or an unusable set."""


@dataclass(frozen=True)
class Run:
    path: Path
    arm: str
    case: str
    seed: int
    wall_s: float
    cell: Mapping[str, Any]
    rows: Tuple[Mapping[str, Any], ...]   # decision rows only (look rows dropped)


@dataclass(frozen=True)
class Sample:
    """One usable calibration decision."""

    arm: str
    case: str
    probs: Mapping[str, Mapping[str, float]]      # d1's label distributions (conformal scores)
    labels: Mapping[str, str]                     # the driving model's decision (conformal labels)
    features: Tuple[float, ...]
    decisions: Mapping[str, Mapping[str, Any]]    # {model: {target, release, amount}} incl. the driving model
    sprime: Mapping[str, str]


# --- loading ---------------------------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover(paths: Iterable[Path]) -> List[Path]:
    """Run directories (those holding ``decisions.jsonl``) at or under each path, sorted, without repeats."""
    found: Dict[str, Path] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FitError(f"{p}: no such directory")
        files = [p / "decisions.jsonl"] if (p / "decisions.jsonl").is_file() else sorted(p.rglob("decisions.jsonl"))
        for f in files:
            found[str(f.parent.resolve())] = f.parent
    return [found[k] for k in sorted(found)]


def check_dev(split_doc: Optional[Mapping[str, Any]], case: str, where: str) -> None:
    if split_doc is None:
        return
    try:
        kind = which_split(dict(split_doc), case)
    except KeyError:
        raise FitError(f"{where}: case {case!r} is not in the split file; refusing") from None
    if kind != "dev":
        raise FitError(f"{where}: case {case!r} is an {kind} case; the fit uses development calibration logs only")


def load_run(path: Path, split_doc: Optional[Mapping[str, Any]]) -> Run:
    run_log_p, cell_p = path / "run_log.json", path / "cell.json"
    if not run_log_p.is_file() or not cell_p.is_file():
        raise FitError(f"{path}: needs run_log.json and cell.json next to decisions.jsonl")
    run_log, cell = _read_json(run_log_p), _read_json(cell_p)
    where = str(path)
    if run_log.get("split") != "dev" or cell.get("split") != "dev":
        raise FitError(f"{where}: split is {run_log.get('split')!r}/{cell.get('split')!r}; only dev logs may be fitted")
    if not run_log.get("calibration"):
        raise FitError(f"{where}: not a --calibration run; the fit needs the counterfactual logs")
    if run_log.get("arm", "").split(":")[0] not in CALIBRATION_ARMS:
        raise FitError(f"{where}: arm {run_log.get('arm')!r} is not SW-B or SW-noBoard")
    check_dev(split_doc, run_log["case"], where)
    rows = []
    for line in (path / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind", "decision") != "decision":
            continue
        if row.get("case") != run_log["case"]:
            raise FitError(f"{where}: a decision row names case {row.get('case')!r}, the run is {run_log['case']!r}")
        rows.append(row)
    return Run(path, run_log["arm"].split(":")[0], run_log["case"], int(run_log["seed"]), float(run_log["wall_s"]),
               cell, tuple(rows))


def load_cells(path: Path, split_doc: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cells = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for c in cells:
        if c.get("split") != "dev":
            raise FitError(f"{path}: cell {c.get('arm')}/{c.get('case')} has split {c.get('split')!r}; dev cells only")
        if c.get("error") is None:
            check_dev(split_doc, c["case"], str(path))
    return cells


def _decision(d: Mapping[str, Any]) -> Dict[str, Any]:
    return {"target": d["target"], "release": bool(d["release"]), "amount": int(d["amount"])}


def _labels(d: Mapping[str, Any]) -> Dict[str, str]:
    return {"target": str(d["target"]), "release": "yes" if d["release"] else "no", "amount": str(int(d["amount"]))}


def _label_probs(cf_probs: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    """Counterfactual probs as logged (amount index "0".."4") -> conformal label distributions (levels "1".."5")."""
    amount = {str(int(i) + 1): float(p) for i, p in sorted(cf_probs["amount"].items(), key=lambda kv: int(kv[0]))}
    return {"target": {k: float(v) for k, v in cf_probs["target"].items()},
            "release": {"yes": float(cf_probs["release"]["yes"]), "no": float(cf_probs["release"]["no"])},
            "amount": amount}


def samples_from(runs: Sequence[Run], *, bonsai: bool = False) -> Tuple[List[Sample], Dict[str, int]]:
    """Usable decisions of all runs. Error decisions, rows without counterfactual answers, S' labels or features are
    counted in the returned ``skipped`` tally, never guessed."""
    driver = TARGET_BACKEND[bonsai]
    skipped: Counter = Counter()
    out: List[Sample] = []
    for run in runs:
        for row in run.rows:
            if row.get("error"):
                skipped["error_decision"] += 1
                continue
            if row.get("backend") != driver:
                raise FitError(f"{run.path}: decisions were made by {row.get('backend')!r}, expected {driver!r}"
                               + ("" if bonsai else " (use --bonsai for a Bonsai calibration)"))
            cf = row.get("counterfactual")
            if not cf or D1 not in cf or Q4 not in cf:
                skipped["no_counterfactual"] += 1
                continue
            sp = row.get("sprime")
            feats = row.get("features")
            if not sp:
                skipped["no_sprime"] += 1
            if feats is None:
                skipped["no_features"] += 1
            decisions = {D1: _decision(cf[D1]), Q4: _decision(cf[Q4]), driver: _decision(row)}
            out.append(Sample(run.arm, run.case, _label_probs(cf[D1]["probs"]), _labels(row),
                              tuple(float(x) for x in feats) if feats is not None else (), decisions, sp or {}))
    return out, dict(skipped)


# --- conformal -------------------------------------------------------------------------------------------------------


def fit_conformal(samples: Sequence[Sample], *, bonsai: bool = False, target: float = DEFER_TARGET) -> ConformalThresholds:
    if not samples:
        raise FitError("no usable decisions for the conformal fit")
    records = [DevRecord(s.probs, s.labels) for s in samples]
    return freeze_thresholds(records, arm="SW-Bonsai" if bonsai else "SW-B+SW-noBoard",
                             target_model=TARGET_BACKEND[bonsai], target=target)


def conformal_report(th: ConformalThresholds, samples: Sequence[Sample]) -> Dict[str, Any]:
    dec = ConformalDeferral(th)
    flags = [(s.arm, dec.decide(s.probs)) for s in samples]
    by_arm: Dict[str, List[bool]] = defaultdict(list)
    for arm, r in flags:
        by_arm[arm].append(r.deferred)
    agree = {q: float(np.mean([max(s.probs[q], key=lambda l: (s.probs[q][l], -list(s.probs[q]).index(l))) == s.labels[q]
                               for s in samples])) for q in QTYPES}
    return {"n_decisions": len(samples), "alpha": {q: th.per_question[q]["alpha"] for q in QTYPES},
            "qhat": {q: (None if th.qhat(q) == float("inf") else th.qhat(q)) for q in QTYPES},
            "per_question_defer_rate": {q: th.per_question[q]["dev_defer_rate"] for q in QTYPES},
            "joint_defer_rate": th.joint_dev_defer_rate, "joint_on_target": th.joint_on_target,
            "defer_target": th.defer_target,
            "defer_rate_by_arm": {a: float(np.mean(v)) for a, v in sorted(by_arm.items())},
            "empty_set_share": float(np.mean([r.empty for _, r in flags])),
            "d1_argmax_matches_label": agree, "sha256": th.sha256}


# --- router ----------------------------------------------------------------------------------------------------------


def _models(bonsai: bool) -> Tuple[str, ...]:
    return BONSAI_ROUTER_MODELS if bonsai else ROUTER_MODELS


def router_rows(samples: Sequence[Sample], *, bonsai: bool = False) -> Tuple[List[Sample], List[TrainRow]]:
    """Samples with features and S' labels, and their router labels (cheapest best-matching model, ties cheaper)."""
    models = _models(bonsai)
    keep = [s for s in samples if s.features and s.sprime]
    return keep, [TrainRow(list(s.features), best_model(s.decisions, s.sprime, models)) for s in keep]


def predict_many(router: KNNRouter, features: np.ndarray) -> List[str]:
    """``KNNRouter.predict`` for many rows at once: same standardisation, (distance, index) neighbour order, plurality
    vote with ties to the cheaper (first) model."""
    z = (np.asarray(features, dtype=np.float64) - router.mean) / router.std
    d2 = ((z[:, None, :] - router.x[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(d2, axis=1, kind="stable")[:, : router.k]   # stable: ties keep the lower training index
    votes = np.zeros((len(z), len(router.models)), dtype=int)
    for j in range(len(router.models)):
        votes[:, j] = (router.y[order] == j).sum(axis=1)
    return [router.models[int(i)] for i in np.argmax(votes, axis=1)]


def cross_validate(samples: Sequence[Sample], rows: Sequence[TrainRow], *, bonsai: bool = False, k: int = K
                   ) -> Dict[str, Any]:
    """Leave-one-case-out: each case's decisions are routed by a router trained on all other cases."""
    models = _models(bonsai)
    cases = sorted({s.case for s in samples})
    if len(cases) < 2:
        return {"folds": len(cases), "agreement": None, "pick_shares": None, "note": "needs at least 2 cases"}
    feats = np.asarray([r.features for r in rows], dtype=np.float64)
    preds: List[Optional[str]] = [None] * len(rows)
    for case in cases:
        test = [i for i, s in enumerate(samples) if s.case == case]
        train = [i for i, s in enumerate(samples) if s.case != case]
        router = KNNRouter.fit([rows[i] for i in train], models, k)
        for i, p in zip(test, predict_many(router, feats[test])):
            preds[i] = p
    labels = [r.label for r in rows]
    agree = float(np.mean([p == l for p, l in zip(preds, labels)]))
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    return {"folds": len(cases), "agreement": agree, "majority_baseline": majority,
            "pick_shares": _shares(preds, models), "per_arm_agreement": {
                a: float(np.mean([p == l for p, l, s in zip(preds, labels, samples) if s.arm == a]))
                for a in sorted({s.arm for s in samples})}}


def _shares(items: Sequence[Optional[str]], models: Sequence[str]) -> Dict[str, float]:
    c = Counter(items)
    return {m: c.get(m, 0) / len(items) for m in models}


def fit_router(samples: Sequence[Sample], *, bonsai: bool = False, k: int = K) -> Tuple[KNNRouter, Dict[str, Any]]:
    keep, rows = router_rows(samples, bonsai=bonsai)
    if not rows:
        raise FitError("no decision has both router features and S' labels; the logs were written before the harness "
                       "logged `features` (re-run the calibration with a harness that logs them)")
    models = _models(bonsai)
    router = KNNRouter.fit(rows, models, k)
    driver = TARGET_BACKEND[bonsai]
    full = {m: float(np.mean([match_count(s.decisions[m], s.sprime) == 3 for s in keep])) for m in models}
    per_q = {m: {q: float(np.mean([_labels(s.decisions[m])[q] == s.sprime[q] for s in keep])) for q in QTYPES}
             for m in models}
    in_sample = _shares(predict_many(router, np.asarray([r.features for r in rows])), models)
    report = {"n_decisions": len(keep), "k": router.k, "models": list(models), "driver": driver,
              "label_shares": _shares([r.label for r in rows], models),
              "in_sample_pick_shares": in_sample,
              "cross_validated": cross_validate(keep, rows, bonsai=bonsai, k=k),
              "sprime_match_share_all_three": full, "sprime_match_share_per_question": per_q,
              "sha256": router.sha256}
    return router, report


# --- power and throughput --------------------------------------------------------------------------------------------


def power_report(cells: Sequence[Mapping[str, Any]], n_eval: int = 80) -> Dict[str, Any]:
    out = dict(v41_analysis.power_analysis(cells, n_eval))
    out.update(alpha=v41_analysis.ALPHA, power=v41_analysis.POWER)
    mdd = out.get("mdd_p1")
    out["mdd_p1_points"] = None if mdd is None else mdd / v41_analysis.POINT
    dev = v41_analysis.usable(cells, "dev")
    for arm in v41_analysis.P1_TREATMENT, v41_analysis.P1_CONTROL:
        m = list(v41_analysis.case_means(dev, arm).values())
        out[f"mean_coverage_{arm}"] = float(np.mean(m)) if m else None
    return out


def throughput_report(runs: Sequence[Run], concurrency: Optional[int] = None) -> Dict[str, Any]:
    n_dec = sum(len(r.rows) for r in runs)
    wall = sum(r.wall_s for r in runs)
    walls = [r.wall_s for r in runs]
    server: Dict[str, List[float]] = defaultdict(list)
    for r in runs:
        for row in r.rows:
            for c in row.get("calls") or []:
                server[c.get("worker", row.get("backend", "?"))].append(float(c["server_ms"]))
            for name, cf in (row.get("counterfactual") or {}).items():
                if cf.get("call"):
                    server[name].append(float(cf["call"]["server_ms"]))
    out: Dict[str, Any] = {
        "runs": len(runs), "decisions": n_dec, "total_run_wall_s": wall,
        "seconds_per_episode_mean": statistics.fmean(walls) if walls else None,
        "seconds_per_episode_median": statistics.median(walls) if walls else None,
        "decisions_per_episode_mean": n_dec / len(runs) if runs else None,
        "decisions_per_second_per_run": n_dec / wall if wall > 0 else None,
        "server_ms_median_by_model": {m: statistics.median(v) for m, v in sorted(server.items())}}
    if concurrency:
        out["concurrency_assumed"] = concurrency
        out["decisions_per_second_at_concurrency"] = n_dec / wall * concurrency if wall > 0 else None
        out["episodes_per_hour_at_concurrency"] = 3600.0 * concurrency / statistics.fmean(walls) if walls else None
    return out


# --- orchestration ---------------------------------------------------------------------------------------------------


def run_fit(inputs: Sequence[Path], *, tag: str, cells_path: Optional[Path] = None, bonsai: bool = False,
            out_conformal: Optional[Path] = None, out_router: Optional[Path] = None, report_path: Optional[Path] = None,
            split_file: Optional[Path] = None, concurrency: Optional[int] = None, n_eval: int = 80,
            force: bool = False, k: int = K, allow_provider_fallbacks: bool = False) -> Dict[str, Any]:
    """Load, fit, freeze (when output paths are given) and report. Returns the aggregates-only report.

    Runs whose queen fell back because the PROVIDER failed (mistake-2039) are invalid for the fit: they are excluded and
    counted (``inputs.provider_fallback_runs``, ``skipped.provider_fallback_run`` decisions) unless
    ``allow_provider_fallbacks``."""
    split_doc = load_split(split_file) if split_file else None
    dirs = discover(inputs)
    if not dirs:
        raise FitError("no run directories (decisions.jsonl) found under the inputs")
    runs = [load_run(d, split_doc) for d in dirs]
    tainted = [r for r in runs if v41_fallbacks.run_dir_provider_fallbacks(r.path, r.cell) > 0]
    dropped_decisions = 0
    if tainted and not allow_provider_fallbacks:
        gone = {id(r) for r in tainted}
        dropped_decisions = sum(len(r.rows) for r in tainted)
        runs = [r for r in runs if id(r) not in gone]
        if not runs:
            raise FitError(f"all {len(tainted)} run(s) have provider-caused queen fallbacks; pass --allow-provider-fallbacks to fit them anyway")
    cells = load_cells(cells_path, split_doc) if cells_path else [dict(r.cell) for r in runs]
    if tainted and not allow_provider_fallbacks and cells_path:
        bad = {(r.arm, r.case, r.seed) for r in tainted}
        cells = [c for c in cells if (c.get("arm"), c.get("case"), c.get("seed")) not in bad
                 and not int(c.get("queen_fallbacks_provider") or 0)]
    samples, skipped = samples_from(runs, bonsai=bonsai)
    if dropped_decisions:
        skipped["provider_fallback_run"] = dropped_decisions
    for out in (out_conformal, out_router):
        if out is not None and Path(out).exists() and not force:
            raise FitError(f"{out} already exists; frozen files are not overwritten without --force")
    th = fit_conformal(samples, bonsai=bonsai)
    report: Dict[str, Any] = {
        "study": "v4.1", "kind": "phase0-fit", "tag": tag, "variant": "bonsai" if bonsai else "main",
        "inputs": {"run_dirs": len(runs), "arms": dict(Counter(r.arm for r in runs)),
                   "cases": len({r.case for r in runs}), "decisions": sum(len(r.rows) for r in runs),
                   "usable_decisions": len(samples), "skipped": skipped,
                   "cells_file_lines": len(cells) if cells_path else None,
                   "provider_fallback_runs": len(tainted), "provider_fallback_runs_excluded": bool(tainted) and not allow_provider_fallbacks,
                   "dev_only": True},
        "conformal": conformal_report(th, samples)}
    if out_conformal is not None:
        report["conformal"]["frozen_to"] = str(out_conformal)
        th.save(out_conformal)
    status = "ok"
    try:
        router, rr = fit_router(samples, bonsai=bonsai, k=k)
        report["router"] = rr
        if out_router is not None:
            rr["frozen_to"] = str(out_router)
            router.save(out_router)
    except FitError as exc:
        report["router"] = {"error": str(exc)}
        status = "router_not_fitted"
    report["power"] = power_report(cells, n_eval)
    report["throughput"] = throughput_report(runs, concurrency)
    report["status"] = status
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return report
