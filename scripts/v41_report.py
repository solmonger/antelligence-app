#!/usr/bin/env python3
"""results.json -> REPORT.md. Every number is a link to its results.json key: [0.031](results.json#p1.mean_diff)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

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
    add("## P1 board effect (SW-B vs SW-noBoard)")
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
    add("## Secondary measures")
    for sid, block in results["secondary"].items():
        if "arms" in block and sid != "S9":
            add(f"### {sid} ({block['field']})")
            for arm, v in block["arms"].items():
                add(f"- {arm}: mean {n(f'secondary.{sid}.arms.{arm}.mean')} over {n(f'secondary.{sid}.arms.{arm}.n')} runs.")
    s9 = results["secondary"]["S9"]
    add("### S9 (normalised score, ratio of means)")
    add(f"- Cases with gap at least {n('secondary.S9.min_gap', 2)}: {n('secondary.S9.n_cases_gap')}.")
    for arm in s9["arms"]:
        add(f"- {arm}: {n(f'secondary.S9.arms.{arm}.ratio')} over {n(f'secondary.S9.arms.{arm}.n_cases')} cases.")
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
