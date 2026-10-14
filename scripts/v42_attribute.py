#!/usr/bin/env python
"""C0 step 1: attribute v4.1 episode wall time from the decision logs (read-only; no network, no keys, no model calls).

  HF_HUB_OFFLINE=1 PYTHONPATH=. python scripts/v42_attribute.py --runs /Volumes/WD_BLACK/antelligence-v4.1/runs \\
      --out docs/research/v4.2-20261014/throughput/attribution.json

Reads ``<runs>/<set>/runs/<run_id>/{run_log.json,decisions.jsonl}`` for the sets named in ``SETS`` (decision rows only;
never ``pod-up*.log``, keys, ledgers or caches). Each decision row carries per-call ``server_ms``, ``network_ms`` and
``queue_ms`` (v4.1 timing: network_ms is the client round trip minus server and queue time, so for vLLM, which reports no
server time, it holds the whole GPU wait).

Method. Calls inside one round run concurrently, so wall time is the critical path, not the sum:
  round wait = max over squads of the squad's decision wait (looks excluded, see below)
  squad wait = max over the calls of a calibration decision, or d1 first then the max of the others (router mode)
  look wait  = NOT logged by v4.1 (look rows are written only on error), so it cannot be attributed; it stays in the
               unlogged remainder. (A first draft used the longest d1 decision call as a stand-in; that overshot the
               wall time in calib1, 596 s modelled against 479 s measured, so it was dropped.)
A call served from the reply cache waited ~0 in that run, so the *actual-wall view* counts only non-cached calls. The
*recorded view* counts every call at its recorded timing (the cache stores the original timing) and is the latency source
for the sleeping fake backend. Remainder = wall - decision wait: the d1 looks (not logged), queen calls (not logged), engine and event-loop time. It is an
UPPER bound on the engine share; v42_profile.py measures the engine directly. Components of the modelled wait come from the critical call of each round.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SETS = ("calib1", "calib1f", "step9", "bonsai-calib")
D1 = "d1"
QUANTILES = [i / 100 for i in range(0, 101)]
# What else ran during each set, from log/session-log.md (v4.1 vault) lines cited in ``notes``.
CONTEXT = {
    "calib1": "step 7, concurrency 8, one pod, static queen, mock anchors; nothing else heavy on the Mac "
              "(session-log 2026-10-13 'Phase 0 step 7 running')",
    "calib1f": "step 7 refit replay: replies come from the reply cache, so wall is mostly engine time "
               "(session-log 2026-10-13 'step 7 fit (calib1f)')",
    "step9": "step 9, --log-counterfactuals all, concurrency 16, plus Bonsai calibration (concurrency 4) in parallel on "
             "the same pod, the Mac Bonsai server (6.7 GB RSS), and the lead Claude process at ~20 GB "
             "(session-log 2026-10-14 ~09:50 and 10:10); the 135 s runs are cache replays after the restart",
    "bonsai-calib": "Bonsai calibration, concurrency 4, Bonsai 8B on the Mac (:8521) driving, while step 9 ran on the pod "
                    "(session-log 2026-10-14 09:21, 10:09)",
}


def load_set(root: Path, name: str) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    out = []
    for d in sorted((root / name / "runs").glob("*")):
        log, dec = d / "run_log.json", d / "decisions.jsonl"
        if not (log.is_file() and dec.is_file()):
            continue
        rows = [json.loads(line) for line in dec.read_text().splitlines() if line.strip()]
        out.append((json.loads(log.read_text()), [r for r in rows if r.get("kind") == "decision"]))
    return out


def call_ms(c: Dict[str, Any]) -> Dict[str, float]:
    return {"server": float(c.get("server_ms", 0)), "network": float(c.get("network_ms", 0)),
            "queue": float(c.get("queue_ms", 0))}


def total(c: Dict[str, Any]) -> float:
    return sum(call_ms(c).values())


def all_calls(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every call this decision made: the main/used calls plus the counterfactual and audit calls, once per model."""
    by_model: Dict[str, Dict[str, Any]] = {}
    for c in row.get("calls") or []:
        by_model.setdefault(c["worker"], c)
    for v in (row.get("counterfactual") or {}).values():
        if isinstance(v, dict) and v.get("call"):
            by_model.setdefault(v["call"]["worker"], v["call"])
    for c in ((row.get("audit") or {}).get("calls") or []):
        by_model.setdefault(c["worker"], c)
    return list(by_model.values())


def squad_wait(calls: List[Dict[str, Any]], calibration: bool, use_cached: bool) -> Tuple[float, Optional[Dict[str, Any]]]:
    """(wait, critical call) of one squad's decision. ``use_cached`` False = actual-wall view (cached calls wait 0)."""
    live = [c for c in calls if use_cached or not c.get("cached")]
    if not live:
        return 0.0, None
    if calibration:
        crit = max(live, key=total)
        return total(crit), crit
    first = [c for c in live if c["worker"] == D1]
    rest = [c for c in live if c["worker"] != D1]
    t0 = total(first[0]) if first else 0.0
    crit = max(rest, key=total) if rest else None
    return t0 + (total(crit) if crit else 0.0), (crit or (first[0] if first else None))


def episode(run_log: Dict[str, Any], rows: List[Dict[str, Any]], use_cached: bool) -> Dict[str, Any]:
    calibration = bool(run_log.get("calibration"))
    rounds: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rounds[int(r["round"])].append(r)
    comp = {"server": 0.0, "network": 0.0, "queue": 0.0}
    per_model_crit: Dict[str, float] = defaultdict(float)
    modelled = 0.0
    overlap = 0.0     # what-if: every call of a decision started at once (cf-all runs only)
    for rnd in sorted(rounds):
        best, crit = 0.0, None
        flat_best = 0.0
        for r in rounds[rnd]:
            calls = all_calls(r)
            w, c = squad_wait(calls, calibration, use_cached)
            if w > best:
                best, crit = w, c
            live = [x for x in calls if use_cached or not x.get("cached")]
            if live:
                flat_best = max(flat_best, max(total(x) for x in live))
        modelled += best / 1000.0
        overlap += flat_best / 1000.0
        if crit is not None:
            for k, v in call_ms(crit).items():
                comp[k] += v / 1000.0
            per_model_crit[crit["worker"]] += best / 1000.0
    wall = float(run_log["wall_s"])
    return {"wall_s": wall, "modelled_wait_s": modelled, "remainder_s": wall - modelled, "whatif_overlap_wait_s": overlap,
            "components_s": comp, "critical_by_model_s": dict(per_model_crit), "rounds": len(rounds),
            "decisions": len(rows), "calls": sum(len(all_calls(r)) for r in rows),
            "cached_calls": sum(1 for r in rows for c in all_calls(r) if c.get("cached"))}


def shares(parts: Dict[str, float], whole: float) -> Dict[str, Optional[float]]:
    return {k: (v / whole if whole > 0 else None) for k, v in parts.items()}


def summarise(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not eps:
        return {"episodes": 0}
    wall = sum(e["wall_s"] for e in eps)
    comp = {k: sum(e["components_s"][k] for e in eps) for k in ("server", "network", "queue")}
    modelled = sum(e["modelled_wait_s"] for e in eps)
    remainder = sum(e["remainder_s"] for e in eps)
    walls = sorted(e["wall_s"] for e in eps)
    wait_parts = dict(comp)
    return {"episodes": len(eps), "wall_s_mean": wall / len(eps), "wall_s_median": statistics.median(walls),
            "wall_s_min": walls[0], "wall_s_max": walls[-1],
            "time_s_per_episode": {**{k: v / len(eps) for k, v in wait_parts.items()}, "unlogged_remainder": remainder / len(eps),
                                   "modelled_wait": modelled / len(eps)},
            "share_of_wall": {**shares(wait_parts, wall), "unlogged_remainder": remainder / wall if wall else None},
            "whatif_all_calls_started_together_wait_s": sum(e["whatif_overlap_wait_s"] for e in eps) / len(eps),
            "calls_per_episode": sum(e["calls"] for e in eps) / len(eps),
            "cached_call_share": (sum(e["cached_calls"] for e in eps) / max(1, sum(e["calls"] for e in eps)))}


def per_model_summary(sets_rows: Iterable[Dict[str, Any]], use_cached: bool) -> Dict[str, Any]:
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"server": [], "network": [], "queue": [], "total": []})
    for r in sets_rows:
        for c in all_calls(r):
            if not use_cached and c.get("cached"):
                continue
            m = call_ms(c)
            for k, v in m.items():
                acc[c["worker"]][k].append(v / 1000.0)
            acc[c["worker"]]["total"].append(total(c) / 1000.0)
    out = {}
    for w, d in sorted(acc.items()):
        t = sum(d["total"]) or 1.0
        out[w] = {"calls": len(d["total"]), "sum_s": t,
                  "share_server": sum(d["server"]) / t, "share_network": sum(d["network"]) / t,
                  "share_queue": sum(d["queue"]) / t,
                  "mean_s": statistics.mean(d["total"]), "p50_s": statistics.median(d["total"]),
                  "p95_s": sorted(d["total"])[int(0.95 * (len(d["total"]) - 1))]}
    return out


def quantile_table(values: List[float]) -> List[float]:
    v = sorted(values)
    n = len(v)
    return [round(v[min(n - 1, int(q * (n - 1) + 0.5))], 4) for q in QUANTILES]


def latency_distribution(root: Path, source: str = "calib1") -> Dict[str, Any]:
    """Per-model total-latency quantiles (seconds, 101 points) from the live runs of ``source``: the fake backend's source."""
    vals: Dict[str, List[float]] = defaultdict(list)
    for run_log, rows in load_set(root, source):
        for r in rows:
            for c in all_calls(r):
                if not c.get("cached"):
                    vals[c["worker"]].append(total(c) / 1000.0)
    return {"source": f"{source} live (non-cached) calls", "unit": "s", "quantiles": QUANTILES,
            "models": {m: {"n": len(v), "q": quantile_table(v)} for m, v in sorted(vals.items()) if v}}


def build(root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"schema": "v42-c0-attribution/1", "method": __doc__.split("Method.")[1].strip(),
                              "sets": {}}
    for name in SETS:
        data = load_set(root, name)
        if not data:
            result["sets"][name] = {"episodes": 0, "note": "no decision logs"}
            continue
        arms = sorted({rl["arm"] for rl, _ in data})
        actual = [episode(rl, rows, use_cached=False) for rl, rows in data]
        recorded = [episode(rl, rows, use_cached=True) for rl, rows in data]
        flat = [r for _, rows in data for r in rows]
        result["sets"][name] = {
            "arms": arms, "context": CONTEXT[name],
            "actual_wall_view": summarise(actual),
            "recorded_latency_view": {**summarise(recorded), "note": "recorded timings of every call, cached ones "
                                      "included; the remainder is meaningful only where cached_call_share is 0"},
            "per_model_recorded": per_model_summary(flat, use_cached=True),
            "per_model_live_only": per_model_summary(flat, use_cached=False),
            "critical_by_model_s_per_episode": {
                k: sum(e["critical_by_model_s"].get(k, 0.0) for e in recorded) / len(recorded)
                for k in sorted({k for e in recorded for k in e["critical_by_model_s"]})},
            "ticks": sorted({int(rl["ticks"]) for rl, _ in data}), "rounds": sorted({int(rl["rounds"]) for rl, _ in data}),
        }
    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="the v4.1 runs/ directory (read-only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latency-out", help="also write the per-model latency quantile table the fake backend samples")
    args = ap.parse_args(argv)
    root = Path(args.runs)
    doc = build(root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    if args.latency_out:
        Path(args.latency_out).write_text(json.dumps(latency_distribution(root), indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
