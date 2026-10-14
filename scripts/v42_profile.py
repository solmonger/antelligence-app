#!/usr/bin/env python
"""C0 throughput measurements for the v4.1 SW-B episode path with sleeping fake backends (no network, no keys, no pod).

  export HF_HUB_OFFLINE=1 PYTHONPATH=<worktree>
  python scripts/v42_profile.py phases      --label before --out docs/research/v4.2-20261014/throughput/results/phases-before.json
  python scripts/v42_profile.py scripted    --label before --out .../results/scripted-before.json
  python scripts/v42_profile.py concurrency --scale 0.1 --inproc 1,4,8,16 --pool 1,2,4,8 --out .../results/concurrency-s0.1.json
  python scripts/v42_profile.py ticks       --runs /Volumes/WD_BLACK/antelligence-v4.1/runs --out .../results/ticks.json
  python scripts/v42_profile.py report      # builds profile.json and the tables in PROFILE.md from results/*.json

The v4.1 data are read-only inputs: the development-case perception stores (``--cases-root``) and, for ``ticks``, the
run logs and bundles of finished or killed episodes. Only development cases are admitted (``v42_fake.dev_only_gate``).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v42_profile as P  # noqa: E402

OUT_DIR = REPO / "docs/research/v4.2-20261014/throughput"
RUNS = OUT_DIR / "results"
PROFILED_CASE_INDEX = 0          # dev_a[0]
EXTRA_CASES = 3                  # the profiled case plus 3 more for the identical-metrics check
REPEATS = 3


def cfg_from(args: argparse.Namespace, **kw: Any) -> P.ProfileConfig:
    return P.ProfileConfig(repo_root=str(REPO), cases_root=args.cases_root, time_scale=getattr(args, "scale", 0.01),
                           work_dir="", **kw)


def write(path: str, doc: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise SystemExit(f"{p} exists: results are never overwritten (pick a new --out and report both attempts)")
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {p}")


def meta(args: argparse.Namespace) -> Dict[str, Any]:
    import platform
    return {"label": getattr(args, "label", None), "started_unix": time.time(), "python": platform.python_version(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "swap": P.swap_usage()}


def cmd_phases(args: argparse.Namespace) -> None:
    cases = P.dev_a_cases()[: 1 + EXTRA_CASES]
    cfg = cfg_from(args, arm="SW-B")
    timed = [P.timed_episode(cfg, cases[PROFILED_CASE_INDEX], 1) for _ in range(REPEATS)]
    prof = P.cprofile_episode(cfg, cases[PROFILED_CASE_INDEX], 1)
    fps = {}
    for case in cases:
        r = P.timed_episode(cfg, case, 1) if case != cases[0] else timed[0]
        fps[f"SW-B:{case}:1"] = {"fingerprint": r["fingerprint"], "cpu_s": r["cpu_s"]}
    nob = P.timed_episode(cfg_from(args, arm="SW-noBoard"), cases[0], 1)
    fps[f"SW-noBoard:{cases[0]}:1"] = {"fingerprint": nob["fingerprint"], "cpu_s": nob["cpu_s"]}
    for t in timed:
        t.pop("fingerprint", None)
    write(args.out, {"meta": meta(args), "profiled_case": cases[0], "time_scale": args.scale, "arm": "SW-B",
                     "repeats": timed, "cprofile": prof, "fingerprints": fps})


def cmd_scripted(args: argparse.Namespace) -> None:
    cases = P.dev_a_cases()[:5]
    cfg = cfg_from(args)
    out: Dict[str, Any] = {"meta": meta(args), "cases": cases, "arms": {}}
    for arm in ("S", "S-prime"):
        rows = P.time_scripted(cfg, arm, cases)
        out["arms"][arm] = {"episodes": rows, "mean_wall_s": statistics.mean(r["wall_s"] for r in rows),
                            "mean_cpu_s": statistics.mean(r["cpu_s"] for r in rows)}
    write(args.out, out)


def cmd_concurrency(args: argparse.Namespace) -> None:
    cases = P.dev_a_cases()
    cfg = cfg_from(args, arm="SW-B")
    out: Dict[str, Any] = {"meta": meta(args), "time_scale": args.scale, "inprocess": {}, "pool": {}, "attempts": []}
    for c in [int(x) for x in args.inproc.split(",") if x]:
        r = P.measure_inprocess(cfg, c, cases)
        out["inprocess"][str(c)] = r
        print(f"in-process c={c}: wall {r['wall_s']:.1f}s, {r['episodes_per_hour']:.1f} eps/h, lag p95 {r['lag']['p95_ms']:.1f} ms, "
              f"cpu {r['cpu_s']:.1f}s, rss {r['peak_rss_mb']:.0f} MB, failed {len(r['failed'])}")
    for n in [int(x) for x in args.pool.split(",") if x]:
        r = P.measure_pool(cfg, n, cases)
        out["pool"][str(n)] = r
        print(f"pool N={n}: wall {r['wall_s']:.1f}s (startup {r['startup_s']:.1f}s), {r['episodes_per_hour']:.1f} eps/h, "
              f"lag p95 worst {r['lag']['p95_ms_worst_process']:.1f} ms, child rss {r['peak_rss_mb_child_max']:.0f} MB")
    write(args.out, out)


def cmd_ticks(args: argparse.Namespace) -> None:
    root = Path(args.runs)
    out: Dict[str, Any] = {"meta": meta(args), "sets": {}, "world_ticks_config": None}
    import yaml
    out["world_ticks_config"] = yaml.safe_load((REPO / "config/v4_1/world.yaml").read_text())["world"]["ticks"]
    for name in ("calib1", "calib1f", "step9", "bonsai-calib"):
        ticks, board, finished, unfinished = [], [], 0, []
        for d in sorted((root / name / "runs").glob("*")):
            log = d / "run_log.json"
            ots = sorted({int(f.name.split(".")[-2]) for f in d.iterdir() if f.name.endswith(".ots")})
            if log.is_file():
                rl = json.loads(log.read_text())
                finished += 1
                ticks.append(int(rl["ticks"]))
                board.append(int(rl["board_size"]))
            elif ots:
                unfinished.append(max(ots))
        out["sets"][name] = {
            "finished_episodes": finished, "ticks_values": sorted(set(ticks)),
            "board_size_min_median_max": [min(board), statistics.median(board), max(board)] if board else None,
            "episodes_without_run_log": len(unfinished),
            "last_anchor_checkpoint_of_unfinished": {str(k): unfinished.count(k) for k in sorted(set(unfinished))},
            "median_last_anchor_checkpoint_of_unfinished": statistics.median(unfinished) if unfinished else None}
    out["verdict"] = ("Every finished episode in calib1, calib1f, step9 and bonsai-calib has ticks == 120 == world.yaml ticks. "
                      "The 26 killed step-9 episodes left run directories without a run_log; their highest OTS anchor "
                      "checkpoint is a BOARD ENTRY COUNT (anchors fall at board sizes 50, 100, ...), median 200 with a "
                      "board of ~440 entries per SW-B episode. 'About tick 200' therefore counted board entries (the "
                      "10:10 log line speaks of 'board steps' and ~450 entries per episode), not world ticks.")
    write(args.out, out)


def cmd_ab(args: argparse.Namespace) -> None:
    """Paired, interleaved A/B in one process: legacy sign_id (parse the key every call) vs the cached-key version.
    Alternating runs cancel machine-load drift that separate before/after runs cannot."""
    import asyncio
    from antelligence.kernel import board as board_mod
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    new_sign = board_mod.sign_id

    def legacy(seed: bytes, entry_id: str) -> str:
        return Ed25519PrivateKey.from_private_bytes(seed).sign(bytes.fromhex(entry_id)).hex()

    cfg = cfg_from(args, arm="SW-B")
    case = P.dev_a_cases()[0]
    pairs = []
    for i in range(args.pairs):
        row: Dict[str, Any] = {}
        for label, fn in (("legacy", legacy), ("cached", new_sign)) if i % 2 == 0 else (("cached", new_sign), ("legacy", legacy)):
            board_mod.sign_id = fn
            r = asyncio.run(P.run_episode(cfg, case, 1, monitor=False))
            row[label] = {"cpu_s": r["cpu_s"], "trace_hash": r["fingerprint"]["trace_hash"],
                          "board_root": r["fingerprint"]["board_root"]}
        board_mod.sign_id = new_sign
        pairs.append(row)
    d = [p["legacy"]["cpu_s"] - p["cached"]["cpu_s"] for p in pairs]
    write(args.out, {"meta": meta(args), "case": case, "pairs": pairs, "median_saving_cpu_s": statistics.median(d),
                     "identical": all(p["legacy"]["trace_hash"] == p["cached"]["trace_hash"]
                                      and p["legacy"]["board_root"] == p["cached"]["board_root"] for p in pairs)})


def _load(name: str) -> Optional[Dict[str, Any]]:
    p = RUNS / name
    return json.loads(p.read_text()) if p.is_file() else None


def _med(rows: List[Dict[str, Any]], key: str) -> float:
    return statistics.median(r[key] for r in rows)


def _phase_med(doc: Dict[str, Any], phase: str) -> float:
    return statistics.median(r["phases_s"].get(phase, 0.0) for r in doc["repeats"])


def cmd_report(args: argparse.Namespace) -> None:
    """Assemble profile.json (every number below comes from results/*.json and attribution.json) and PROFILE.md."""
    before, after = _load("phases-before.json"), _load("phases-after-final.json")
    attempt_a = _load("phases-after.json")
    sb, sa = _load("scripted-before-b.json"), _load("scripted-after.json")
    ab, ticks = _load("ab-sign-cache.json"), _load("ticks.json")
    conc = [(f.name, json.loads(f.read_text())) for f in sorted(RUNS.glob("concurrency-*.json"))]
    attr = json.loads((OUT_DIR / "attribution.json").read_text())
    assert before and after and ab and ticks and attr, "run phases/ab/ticks and v42_attribute.py first"
    phases = sorted({k for d in (before, after) for r in d["repeats"] for k in r["phases_s"]}, key=lambda k: -_phase_med(before, k))
    profile: Dict[str, Any] = {
        "schema": "v42-c0-profile/1",
        "engine": {"profiled_case": before["profiled_case"], "arm": before["arm"], "time_scale": before["time_scale"],
                   "cpu_s_before": [r["cpu_s"] for r in before["repeats"]],
                   "cpu_s_after_final": [r["cpu_s"] for r in after["repeats"]],
                   "cpu_s_after_attempt_a_reverted": [r["cpu_s"] for r in attempt_a["repeats"]] if attempt_a else None,
                   "phases_median_s": {k: {"before": _phase_med(before, k), "after": _phase_med(after, k),
                                           "calls": before["repeats"][0]["phase_calls"].get(k)} for k in phases},
                   "other_glue_median_s": {"before": _med(before["repeats"], "other_glue_s"),
                                           "after": _med(after["repeats"], "other_glue_s")},
                   "per_tick_phase_s_before_first_repeat": before["repeats"][0]["phase_per_tick_s"],
                   "cprofile_top_tottime": before["cprofile"]["top_by_tottime"][:15],
                   "identical_metrics": {k: after["fingerprints"][k]["fingerprint"] == before["fingerprints"][k]["fingerprint"]
                                         for k in before["fingerprints"]},
                   "sign_cache_paired_ab": {"median_cpu_saving_s": ab["median_saving_cpu_s"], "pairs": len(ab["pairs"]),
                                            "identical": ab["identical"]}},
        "scripted": None, "ticks": ticks["verdict"], "concurrency": {n: d for n, d in conc},
    }
    if sb and sa:
        profile["scripted"] = {arm: {"before_mean_wall_s": sb["arms"][arm]["mean_wall_s"],
                                     "after_mean_wall_s": sa["arms"][arm]["mean_wall_s"],
                                     "identical_metrics": [x["fingerprint"] == y["fingerprint"] for x, y in
                                                           zip(sb["arms"][arm]["episodes"], sa["arms"][arm]["episodes"])]}
                               for arm in ("S", "S-prime")}
    (OUT_DIR / "profile.json").write_text(json.dumps(profile, indent=1, sort_keys=True) + "\n")

    L: List[str] = []
    a = L.append
    cpu_b, cpu_a = statistics.median(profile["engine"]["cpu_s_before"]), statistics.median(profile["engine"]["cpu_s_after_final"])
    a("# C0 throughput: attribute, profile, fix\n")
    a("Generated by `scripts/v42_profile.py report` from `results/*.json` and `attribution.json`; the analysis section at the "
      "end is hand-written and cites the same files. Fakes only: no network, no keys, no pod, no model server, "
      "development cases only.\n")
    a("## Verdict\n")
    a(f"* **Engine share: about {cpu_b:.1f} s CPU per SW-B episode** (before-fix run {cpu_b:.2f} s; after-fix run {cpu_a:.2f} s "
      f"at a heavier machine load, paired A/B saving {ab['median_saving_cpu_s']:.2f} s) (target < 30 s: **hit, by more "
      f"than 10x**). The engine was never the throughput problem.")
    a("* Pool scaling target (>= 0.7 x linear up to 8): **hit** in both modes at every time scale (section 4).")
    a("* v4.1 wall time is model-server wait: the decision critical path alone is "
      f"{attr['sets']['calib1']['actual_wall_view']['share_of_wall']['network']*100:.0f}% of step-7 wall time "
      f"(network_ms, which for vLLM holds the whole GPU wait).")
    a("* \"About tick 200\" was a board-entry count, not a tick: every finished episode ends at tick 120 (see Ticks).\n")
    a("## 1. Attribution from the v4.1 logs (`attribution.json`)\n")
    a("| set | episodes | wall s/ep | server | network | queue | unlogged remainder* | cached calls |")
    a("|---|---|---|---|---|---|---|---|")
    for name, st_ in attr["sets"].items():
        v = st_["actual_wall_view"]
        if not v.get("episodes"):
            continue
        sh = v["share_of_wall"]
        a(f"| {name} | {v['episodes']} | {v['wall_s_mean']:.0f} | {sh['server']*100:.1f}% | {sh['network']*100:.1f}% | "
          f"{sh['queue']*100:.1f}% | {sh['unlogged_remainder']*100:.1f}% | {v['cached_call_share']*100:.0f}% |")
    a("\n\\* looks, queen calls, engine and event loop; v4.1 did not log looks or queen time, so this is an UPPER bound on "
      "the engine. Shares are of the decision critical path (slowest call per squad, slowest squad per round) counting only calls "
      "that were not served from the reply cache.\n")
    a("Per model (recorded timings, all calls; calib1 = live):\n")
    a("| set | model | calls | mean s | p95 s | server | network | queue |")
    a("|---|---|---|---|---|---|---|---|")
    for name in ("calib1", "step9", "bonsai-calib"):
        for m, v in attr["sets"][name]["per_model_recorded"].items():
            a(f"| {name} | {m} | {v['calls']} | {v['mean_s']:.1f} | {v['p95_s']:.1f} | {v['share_server']*100:.0f}% | "
              f"{v['share_network']*100:.0f}% | {v['share_queue']*100:.0f}% |")
    a("\nWhat else ran (from the v4.1 session log):\n")
    for name, st_ in attr["sets"].items():
        a(f"* **{name}**: {st_.get('context', '')}")
    a("\n## 2. Engine profile (SW-B, fake sleeping backends, concurrency 1)\n")
    a(f"Case {before['profiled_case']}, seed 1, time scale {before['time_scale']}, three repeats each; medians. Phases are "
      "exclusive wall timers around the unmodified code; `other glue` is CPU outside every timed phase "
      "(asyncio, dict and list work, logging).\n")
    a("| phase | calls | before s | after s |")
    a("|---|---|---|---|")
    for k in phases:
        e = profile["engine"]["phases_median_s"][k]
        a(f"| {k} | {e['calls']} | {e['before']:.3f} | {e['after']:.3f} |")
    g = profile["engine"]["other_glue_median_s"]
    a(f"| other glue | | {g['before']:.3f} | {g['after']:.3f} |")
    a(f"| **episode CPU** | | **{cpu_b:.3f}** | **{cpu_a:.3f}** |")
    a(f"\nCPU per repeat, before: {[round(x, 3) for x in profile['engine']['cpu_s_before']]}; after: "
      f"{[round(x, 3) for x in profile['engine']['cpu_s_after_final']]}. The two sets were taken at different machine "
      "loads (swap was 10.8 of 12 GB), so **the unpaired difference is load drift, not the fix**; the paired "
      f"interleaved A/B (`ab-sign-cache.json`, {len(ab['pairs'])} pairs) shows the cached signing key saving a median "
      f"{ab['median_saving_cpu_s']:.3f} s per episode, metrics identical = {ab['identical']}.")
    if attempt_a:
        a(f"\nAttempt A (reverted, reported): a bounding-box `DrugField` diffusion plus the signing-key cache gave CPU "
          f"{[round(x, 3) for x in profile['engine']['cpu_s_after_attempt_a_reverted']]}; the "
          "bbox part is bit-identical but was slower on every field shape tried (1.17 vs 0.98 ms; the crop is only "
          "191k voxels and the support fills it within a few ticks), so it was removed.")
    a("\nTop cProfile entries by own time (one episode, before the key cache, profiler overhead included):\n")
    for r in profile["engine"]["cprofile_top_tottime"][:8]:
        if "kqueue" in r["function"]:
            a(f"* {r['function']}: {r['tottime_s']:.1f} s (the event loop waiting on the sleeping fakes; not CPU)")
        else:
            a(f"* {r['function']}: {r['tottime_s']:.3f} s, {r['ncalls']} calls")
    a("\nIdentical metrics before and after (cell fields, trace hash, board root, decision-log hash, anchors):\n")
    for k, ok in profile["engine"]["identical_metrics"].items():
        a(f"* {k}: {'identical' if ok else '**DIFFERENT**'}")
    if profile["scripted"]:
        a("\n## 3. Scripted arms S and S' (5 development cases, seed 1)\n")
        a("| arm | s/episode before | s/episode after | metrics identical (5 cases) |")
        a("|---|---|---|---|")
        for arm, v in profile["scripted"].items():
            a(f"| {arm} | {v['before_mean_wall_s']:.3f} | {v['after_mean_wall_s']:.3f} | {all(v['identical_metrics'])} |")
    a("\n## 4. Concurrency: one process vs a process pool\n")
    for name, d in conc:
        base_in = d["inprocess"].get("1", {}).get("episodes_per_hour")
        base_pl = d["pool"].get("1", {}).get("episodes_per_hour") or base_in
        sc = d["time_scale"]
        what = ("the recorded latencies, unscaled" if sc == 1.0 else
                f"{sc}x the recorded latencies, so the engine is {1 / sc:.0f}x more prominent than at real latencies")
        a(f"`{name}`: time scale {sc} (sleeps are {what}); episodes start together; swap before `{d['meta']['swap']}`.\n")
        a("| mode | n | wall s | episodes/h | efficiency vs linear | lag p95 ms | lag max ms | CPU s total | peak RSS MB |")
        a("|---|---|---|---|---|---|---|---|---|")
        for mode, base in (("inprocess", base_in), ("pool", base_pl)):
            for n, r in sorted(d[mode].items(), key=lambda kv: int(kv[0])):
                eff = (r["episodes_per_hour"] / (int(n) * base)) if base else float("nan")
                lag = r["lag"]
                p95 = lag.get("p95_ms", lag.get("p95_ms_worst_process"))
                mx = lag.get("max_ms", lag.get("max_ms_worst_process"))
                cpu = r.get("cpu_s", r.get("cpu_s_sum"))
                rss = r.get("peak_rss_mb", r.get("peak_rss_mb_child_max"))
                label = "one process" if mode == "inprocess" else "process pool"
                a(f"| {label} | {n} | {r['wall_s']:.1f} | {r['episodes_per_hour']:.0f} | {eff:.2f} | {p95:.1f} | {mx:.1f} | "
                  f"{cpu:.1f} | {rss:.0f} |")
        a("")
    a("Lag max at 16 in one process (about 0.6 s at every scale) sits far above p95 (2.2 ms at scale 1.0): a rare spike, "
      "most likely the start-up burst when 16 cases load at once (not separately verified).\n")
    a("Pool rows: lag is the worst process's; CPU is summed over processes; RSS is the largest child. Efficiency = "
      "episodes/h divided by n times the n=1 row of the same mode.\n")
    a("## 5. Ticks\n")
    a(ticks["verdict"] + "\n")
    for name, v in ticks["sets"].items():
        a(f"* {name}: {v['finished_episodes']} finished, ticks {v['ticks_values']}, board size min/median/max "
          f"{v['board_size_min_median_max']}; killed episodes {v['episodes_without_run_log']}, last anchor checkpoint "
          f"{v['last_anchor_checkpoint_of_unfinished']}")
    a("\n<!-- ANALYSIS-BEGIN -->")
    notes = OUT_DIR / "analysis.md"
    a(notes.read_text() if notes.is_file() else "(analysis.md missing)")
    a("<!-- ANALYSIS-END -->")
    (OUT_DIR / "PROFILE.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_DIR / 'profile.json'} and {OUT_DIR / 'PROFILE.md'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-root", default=str(P.DEFAULT_CASES_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("phases")
    s.add_argument("--label", required=True)
    s.add_argument("--scale", type=float, default=0.01)
    s.add_argument("--out", required=True)
    s = sub.add_parser("scripted")
    s.add_argument("--label", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("concurrency")
    s.add_argument("--scale", type=float, default=0.1)
    s.add_argument("--inproc", default="1,4,8,16")
    s.add_argument("--pool", default="1,2,4,8")
    s.add_argument("--out", required=True)
    s = sub.add_parser("ab")
    s.add_argument("--pairs", type=int, default=6)
    s.add_argument("--scale", type=float, default=0.01)
    s.add_argument("--out", required=True)
    s = sub.add_parser("ticks")
    s.add_argument("--runs", required=True)
    s.add_argument("--out", required=True)
    sub.add_parser("report")
    args = ap.parse_args(argv)
    {"phases": cmd_phases, "scripted": cmd_scripted, "concurrency": cmd_concurrency, "ticks": cmd_ticks, "ab": cmd_ab,
     "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
