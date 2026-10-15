"""v4.2 M1 report generator: ``results.json`` -> ``REPORT.md`` (tables only; FINDINGS.md is hand-written by the lead).

``results.json``::

    {"meta": {...free-form, echoed...},
     "runs": [{"arm", "case_id", "seed", "split": "dev-A"|"dev-B", "coverage", "off_target_share", "time_to_50_coverage",
               "coverage_at_queen": {tick: v}, "drug_on_non_tumour", "colony": {...run_metrics()...}, "timings": {...}}],
     "effect": {"S-marks": effect_study(...), "S-talk": effect_study(...)}}   # optional
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from antelligence.experiments.v42_stats import N_RESAMPLES, analyse

ARM_ORDER = ["S", "S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S-marks-perfect",
             "S-marks-quiet", "S'"]
PRIMARY_FIELDS = [("coverage", "coverage"), ("off_target_share", "off-target share"),
                  ("time_to_50_coverage", "time to 50% coverage"), ("drug_on_non_tumour", "drug on non-tumour")]


def _f(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _arms(runs: Sequence[Mapping[str, Any]]) -> List[str]:
    present = {r["arm"] for r in runs}
    return [a for a in ARM_ORDER if a in present] + sorted(present - set(ARM_ORDER))


def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def coverage_values(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[tuple, float]]:
    out: Dict[str, Dict[tuple, float]] = {}
    for r in runs:
        out.setdefault(r["arm"], {})[(r["case_id"], int(r["seed"]))] = float(r["coverage"])
    return out


SPLITS = ("dev-A", "dev-B")


def case_splits(runs: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """case -> "dev-A" | "dev-B". A run without a valid ``split``, or a case with two different ones, is an error: the
    same-sign check across dev-A and dev-B must never run on guessed labels."""
    out: Dict[str, str] = {}
    for r in runs:
        sp = r.get("split")
        if sp not in SPLITS:
            raise ValueError(f"run {r.get('arm')}/{r.get('case_id')}/seed {r.get('seed')}: split {sp!r} is missing or not one of {SPLITS}")
        if out.setdefault(r["case_id"], sp) != sp:
            raise ValueError(f"case {r['case_id']!r} has two splits: {out[r['case_id']]!r} and {sp!r}")
    return out


def never_reached_50(runs: Sequence[Mapping[str, Any]], arm: str) -> int:
    """Runs of ``arm`` whose coverage never reached 50% (``time_to_50_coverage`` is None). The mean of the time column
    skips them, so they are counted next to it."""
    return sum(1 for r in runs if r["arm"] == arm and r.get("time_to_50_coverage") is None)


def compute_stats(results: Mapping[str, Any], n: int = N_RESAMPLES) -> Optional[Dict[str, Any]]:
    runs = results["runs"]
    vals = coverage_values(runs)
    if not all(a in vals for a in ("S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled")):
        return None
    splits = case_splits(runs)
    return analyse(vals, splits, n=n)


def render(results: Mapping[str, Any], stats: Optional[Mapping[str, Any]] = None) -> str:
    runs = results["runs"]
    case_splits(runs)                                      # fails loudly on a missing or inconsistent split
    L: List[str] = ["# v4.2 Phase 0 report (generated)", "",
                    "Generated from `results.json` by `scripts/v42_report.py`. Tables only. No interpretation here.", ""]
    meta = results.get("meta") or {}
    if meta:
        L += ["## Run info", "", _table(["key", "value"], [(k, json.dumps(v, sort_keys=True)) for k, v in sorted(meta.items())]), ""]
    arms = _arms(runs)
    L += ["## Primary metrics (mean over runs)", "",
          "The time-to-50% mean covers only runs that reached 50%; the last column counts the runs that never did.", "",
          _table(["arm", "runs"] + [h for _, h in PRIMARY_FIELDS] + ["runs never at 50%"],
                 [[a, sum(r["arm"] == a for r in runs)] + [_f(_mean([r.get(k) for r in runs if r["arm"] == a])) for k, _ in PRIMARY_FIELDS]
                  + [never_reached_50(runs, a)] for a in arms]), ""]
    qticks = sorted({int(t) for r in runs for t in (r.get("coverage_at_queen") or {})})
    if qticks:
        L += ["## Coverage at queen steps (mean)", "",
              _table(["arm"] + [f"t={t}" for t in qticks],
                     [[a] + [_f(_mean([(r.get("coverage_at_queen") or {}).get(str(t)) for r in runs if r["arm"] == a])) for t in qticks]
                      for a in arms]), ""]
    col = [a for a in arms if any(r["arm"] == a and r.get("colony") for r in runs)]
    if col:
        def cm(a: str, key: str) -> Optional[float]:
            return _mean([r["colony"].get(key) for r in runs if r["arm"] == a and r.get("colony")])
        L += ["## Colony metrics (mean per run)", "",
              _table(["arm", "marks", "share strong", "share disputed", "asks sent", "answered", "lapsed", "refused",
                      "quorum events", "turns skipped", "honesty r"],
                     [[a, _f(cm(a, "marks_total"), 1), _f(cm(a, "share_strong")), _f(cm(a, "share_disputed")),
                       _f(cm(a, "asks_sent"), 1), _f(cm(a, "asks_answered"), 1), _f(cm(a, "asks_lapsed"), 1),
                       _f(cm(a, "asks_refused"), 1), _f(cm(a, "quorum_events"), 1), _f(cm(a, "turns_skipped"), 1),
                       _f(_mean([(r["colony"].get("honesty") or {}).get("correlation") for r in runs
                                 if r["arm"] == a and r.get("colony")]))] for a in col]), ""]
        kinds = sorted({k for r in runs if r.get("colony") for k in r["colony"].get("marks_by_kind", {})})
        L += ["### Marks by kind (mean per run)", "",
              _table(["arm"] + kinds, [[a] + [_f(_mean([r["colony"].get("marks_by_kind", {}).get(k, 0) for r in runs
                                                         if r["arm"] == a and r.get("colony")]), 1) for k in kinds] for a in col]), ""]
    tim = [a for a in arms if any(r["arm"] == a and r.get("timings") for r in runs)]
    if tim:
        def tm(a: str, k: str) -> Optional[float]:
            return _mean([(r["timings"].get(k) or {}).get("mean") for r in runs if r["arm"] == a and r.get("timings")])
        L += ["## Timings (mean of per-run means, ms)", "",
              _table(["arm", "look_ms", "queen_ms"], [[a, _f(tm(a, "look_ms"), 2), _f(tm(a, "queen_ms"), 2)] for a in tim]), ""]
    qp = coverage_values(runs)
    if "S-marks" in qp and "S-marks-quiet" in qp:
        sk = {a: _mean([r["colony"]["turns_skipped"] for r in runs if r["arm"] == a and r.get("colony")]) for a in ("S-marks", "S-marks-quiet")}
        L += ["## Quiet-rule pair", "",
              _table(["arm", "mean coverage", "mean turns skipped"],
                     [[a, _f(_mean(list(qp[a].values()))), _f(sk[a], 1)] for a in ("S-marks", "S-marks-quiet")]), ""]
    if "effect" in results and results["effect"]:
        L += ["## Message effect (removal replay)", "",
              _table(["arm", "runs", "items", "effect rate", "mean delta", "median delta", "spam rate", "unapplied"],
                     [[a, e.get("runs"), e.get("n_items"), _f(e.get("effect_rate")), _f(e.get("mean_delta"), 4),
                       _f(e.get("median_delta"), 4), _f(e.get("spam_rate")), e.get("unapplied")]
                      for a, e in sorted(results["effect"].items())]), ""]
    st = stats if stats is not None else compute_stats(results)
    if st is not None:
        L += ["## Verdict statistics (coverage, paired by case and seed)", "",
              f"Case-clustered bootstrap, {next(iter(st['primary'].values()))['boot']['n_resamples']} resamples, seeded. "
              "Holm over the two primary comparisons.", ""]
        rows = []
        for tag, p in st["primary"].items():
            b, cb, v = p["boot"], p["control_boot"], p["verdict"]
            rows.append([f"({tag}) {p['comparison']}", b["n_pairs"], _f(b["median"], 4), f"[{_f(b['ci_low'], 4)}, {_f(b['ci_high'], 4)}]",
                         _f(b["p"], 4), _f(p["holm_p"], 4), _f(p["by_split"]["dev-A"], 4), _f(p["by_split"]["dev-B"], 4),
                         f"{_f(cb['median'], 4)} [{_f(cb['ci_low'], 4)}, {_f(cb['ci_high'], 4)}]", "PASS" if v["pass"] else "fail"])
        L += [_table(["comparison", "pairs", "median diff", "95% CI", "p", "Holm p", "dev-A median", "dev-B median",
                      "vs own control: median [CI]", "verdict"], rows), "",
              _table(["comparison", "median >= 0.03", "Holm p < 0.05", "same sign dev-A/B", "beats control"],
                     [[f"({t}) {p['comparison']}", *(str(p["verdict"][k]) for k in
                                                     ("median_ge_0.03", "holm_p_lt_0.05", "same_sign_dev_a_b", "beats_control"))]
                      for t, p in st["primary"].items()]), "",
              f"Sharing helps scripts (at least one primary passes): **{st['sharing_helps_scripts']}**", ""]
        s = st["secondary"]["boot"]
        L += ["### Secondary (no threshold)", "",
              _table(["comparison", "pairs", "median diff", "95% CI", "p"],
                     [[st["secondary"]["comparison"], s["n_pairs"], _f(s["median"], 4), f"[{_f(s['ci_low'], 4)}, {_f(s['ci_high'], 4)}]", _f(s["p"], 4)]]), ""]
        if "S-marks-perfect" in qp:
            from antelligence.experiments.v42_stats import cluster_bootstrap, group_by_case, paired_diffs
            d = paired_diffs(qp, "S-marks-perfect", "S-solo")
            b = cluster_bootstrap(group_by_case(d), "coverage:S-marks-perfect-minus-S-solo", n=st["primary"]["a"]["boot"]["n_resamples"])
            L += ["### Diagnostic: S-marks-perfect - S-solo (not a verdict arm)", "",
                  _table(["pairs", "median diff", "95% CI", "p"], [[b["n_pairs"], _f(b["median"], 4), f"[{_f(b['ci_low'], 4)}, {_f(b['ci_high'], 4)}]", _f(b["p"], 4)]]), ""]
    else:
        L += ["## Verdict statistics", "", "Not computed: the five arms S-solo, S-marks, S-talk, S-marks-displaced and S-talk-shuffled are not all present.", ""]
    return "\n".join(L).rstrip() + "\n"


def write_report(results_path: Path, out_path: Path) -> str:
    results = json.loads(Path(results_path).read_text())
    text = render(results)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text
