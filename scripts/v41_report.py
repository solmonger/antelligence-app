#!/usr/bin/env python3
"""results.json -> REPORT.md. Every number is a link to its results.json key: [0.031](results.json#p1.mean_diff)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from antelligence.experiments.v41_analysis import P2_FOLLOWED_LABEL, P2_FRONTIER_LABEL

LINK_TARGET = "results.json"


def resolve(results: Dict[str, Any], key: str) -> Any:
    cur: Any = results
    for part in key.split("."):
        cur = cur[part]
    return cur


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


class Report:
    def __init__(self, results: Dict[str, Any]) -> None:
        self.r = results
        self.lines: list[str] = []

    def n(self, key: str, digits: int = 4) -> str:
        return f"[{fmt(resolve(self.r, key), digits)}]({LINK_TARGET}#{key})"

    def add(self, text: str = "") -> None:
        self.lines.append(text)


def render(results: Dict[str, Any]) -> str:
    R = Report(results)
    n, add = R.n, R.add
    add("# Antelligence v4.1 report")
    add("")
    add("Generated from results.json; each number links to its key. Do not edit by hand.")
    add("")
    add(f"Split analysed: {results['split']}. Cells used: {n('n_cells')}.")
    add("")
    add("## P1 board effect (SW-B vs SW-noBoard, coverage)")
    add(f"- Cases: {n('p1.n_cases')}; mean paired difference {n('p1.mean_diff')} "
        f"(BCa confidence interval {n('p1.ci_low')} to {n('p1.ci_high')}); Wilcoxon p = {n('p1.p_value')}.")
    add(f"- Passed: {n('p1.passed')} (needs p below {n('thresholds.alpha', 2)} and a positive mean).")
    add("")
    add("## P2 noninferiority to the frontier-worker arm (SW-B vs FW)")
    add(f"- Status: {results['p2']['status']} (gated on P1: {n('p2.p1_passed')}).")
    add(f"- Cases: {n('p2.n_cases')}; mean difference {n('p2.mean_diff')}; one-sided lower bound "
        f"{n('p2.lower_bound')} against margin {n('p2.margin')}.")
    add(f"- Noninferior: {n('p2.noninferior')}; passed: {n('p2.passed')}.")
    add("")
    add(f"- Worker override rate: SW-B {n('p2.override_rates.SW-B')}, FW {n('p2.override_rates.FW')} "
        f"(rule: below {n('thresholds.override_followed_max', 2)} in both reads as \"{P2_FOLLOWED_LABEL}\", not \"{P2_FRONTIER_LABEL}\").")
    add(f"- Reading of P2: {results['p2'].get('reading') or 'not available (override rate missing)'}.")
    add("")
    add("## P1 gap secondary (P1 on evaluation cases with a large S' - S gap; reported, not used to declare a win)")
    add(f"- Gap threshold: {n('thresholds.s9_min_gap', 2)}.")
    add(f"- Cases in the gap subset: {n('p1_gap_secondary.n_cases_gap')} of {n('p1_gap_secondary.n_cases_total')} "
        f"(share {n('p1_gap_secondary.gap_share')}, reported, not a gate).")
    add(f"- Mean paired difference {n('p1_gap_secondary.mean_diff')} (BCa interval {n('p1_gap_secondary.ci_low')} to "
        f"{n('p1_gap_secondary.ci_high')}); Wilcoxon p = {n('p1_gap_secondary.p_value')}; passed: {n('p1_gap_secondary.passed')}.")
    add("")
    add("## Worker override rate (pre-registered diagnostic, per arm)")
    for arm in results["worker_override"]["arms"]:
        k = f"worker_override.arms.{arm}"
        add(f"- {arm}: {n(k + '.overrides')} overrides in {n(k + '.decisions')} comparable decisions, rate {n(k + '.rate')}.")
    add("")
    add("## P4 on-chain mode (SW-D vs SW-B), no significance test")
    add(f"- Cases: {n('p4.n_cases')}; mean score difference {n('p4.mean_diff')} "
        f"(confidence interval {n('p4.ci_low')} to {n('p4.ci_high')}).")
    add(f"- Time-to-half-score overhead, seconds: {n('p4.seconds_to_50_overhead.mean')} "
        f"(CI {n('p4.seconds_to_50_overhead.ci_low')} to {n('p4.seconds_to_50_overhead.ci_high')}); "
        f"ticks: {n('p4.ticks_to_50_overhead.mean')}.")
    add("")
    add("## Power")
    add(f"- Dev cases {n('power.dev_cases')}, dev paired SD {n('power.dev_paired_sd')}, eval cases "
        f"{n('power.eval_cases')}: minimum detectable P1 difference {n('power.mdd_p1')} ({results['power']['method']}).")
    add("")
    add("## Explained-null checklist (SW-B)")
    for name, item in results["explained_null"]["items"].items():
        add(f"- {name}: {n(f'explained_null.items.{name}.value')} against {n(f'explained_null.items.{name}.threshold', 2)}; "
            f"flagged: {n(f'explained_null.items.{name}.flagged')}.")
    add("")
    add("## Secondary measures (reported, not used to declare a win)")
    for sid in (f"S{i}" for i in range(1, 13)):
        block = results["secondary"][sid]
        add(f"### {sid}")
        if "comparisons" in block:
            for name in block["comparisons"]:
                k = f"secondary.{sid}.comparisons.{name}"
                add(f"- {name}: coverage difference {n(k + '.mean_diff')} (BCa interval {n(k + '.ci_low')} to "
                    f"{n(k + '.ci_high')}) over {n(k + '.n_cases')} cases.")
        elif "fields" in block:
            for field, arms in block["fields"].items():
                for arm in arms:
                    k = f"secondary.{sid}.fields.{field}.{arm}"
                    add(f"- {field}, {arm}: mean {n(k + '.mean')} over {n(k + '.n')} runs.")
        elif sid == "S9":
            add(f"- Cases with gap at least {n('secondary.S9.min_gap', 2)}: {n('secondary.S9.n_cases_gap')}.")
            for arm in block["arms"]:
                add(f"- {arm}: ratio of means {n(f'secondary.S9.arms.{arm}.ratio')} over "
                    f"{n(f'secondary.S9.arms.{arm}.n_cases')} cases.")
        else:
            add(f"- Latency benchmark available: {n('secondary.S12.available')} (source: {block['source']}).")
            for name in block["metrics"]:
                add(f"- {name}: {n(f'secondary.S12.metrics.{name}')}.")
    add("")
    add("## Live check (separate from the explained-null band)")
    add(f"- Mean development defer rate {n('live_check.mean_defer_rate')}; inside the live-check band: {n('live_check.in_band')}.")
    add("")
    return "\n".join(R.lines) + "\n"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: REPORT.md next to results.json")
    a = ap.parse_args(argv)
    out = a.out or a.results.with_name("REPORT.md")
    out.write_text(render(json.loads(a.results.read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
