#!/usr/bin/env python3
"""Render PREREGISTRATION.md from docs/v4.1/PREREGISTRATION.template.md. Refuses if any input is missing."""

from __future__ import annotations

import argparse
import hashlib
import json
import string
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from antelligence.experiments import v41_analysis as A
from antelligence.experiments import v41_budget as B

TEMPLATE = Path(__file__).resolve().parent.parent / "docs" / "v4.1" / "PREREGISTRATION.template.md"
FILE_INPUTS = ("world_yaml", "harness_yaml", "conformal_json", "split_json")


class PreregError(RuntimeError):
    """One or more inputs are missing; nothing is rendered."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha(repo: Path) -> str:
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def _static_sections() -> Dict[str, str]:
    t = A.thresholds()
    hyp = "\n".join([
        f"- P1: {A.P1_TREATMENT} vs {A.P1_CONTROL}, case-level mean over seeds, two-sided Wilcoxon signed-rank and 95% BCa "
        f"bootstrap CI of the mean paired difference. Passes only if p < {t['alpha']} and the mean is > 0.",
        f"- P2: {A.P1_TREATMENT} vs {A.P2_COMPARATOR} noninferiority: one-sided 95% bootstrap lower bound of the mean "
        f"difference > {t['p2_margin_points']:g} points (1 point = 0.01). Gated on P1; otherwise exploratory.",
        f"- P4: {A.P4_TREATMENT} vs {A.P4_CONTROL}: score difference with 95% CI and time-to-50% overhead. No significance test.",
        f"- S9: normalised score as a ratio of means, only on cases with S' - S >= {t['s9_min_gap']}.",
        f"- Bootstrap: {t['bootstrap_resamples']} resamples, fixed seed {t['bootstrap_seed']}.",
        "- Power: minimum detectable P1 difference from the dev paired SD (Wilcoxon ARE-adjusted normal approximation, "
        f"ARE = 3/pi, power {t['power']}).",
    ])
    arms = "\n".join(f"- {a}: {c} x {s}" for a, (c, s) in B.PLAN.items())
    budgets = (f"- Nous: ${B.BUDGETS['nous']:.2f} (stop ${B.STOPS['nous']:.2f})\n"
               f"- RunPod: ${B.BUDGETS['runpod']:.2f} (stop ${B.STOPS['runpod']:.2f})\n- Testnet: {B.TESTNET_ETH} ETH")
    cuts = "\n".join([
        "Nous, in order: " + "; ".join(f"{n} -> {v}" for n, v in B.NOUS_CUT_ORDER),
        "RunPod, in order: " + "; ".join(f"{n} -> {v}" for n, v in B.RUNPOD_CUT_ORDER),
        "Never cut SW-B or SW-noBoard below 80 x 2, or FW below 30.",
    ])
    en = "\n".join([
        f"- Unread briefed entries > {t['unread_max']:.0%} of the board.",
        f"- Median brief entry age > {t['brief_age_max_ticks']:g} ticks.",
        f"- Deferral rate outside [{t['defer_band_low']}, {t['defer_band_high']}] in > {t['defer_runs_max']:.0%} of runs.",
        f"- Router sends > {t['router_share_max']:.0%} of calls to one model.",
        f"- Perception ECE > {t['perception_ece_max']}.",
    ])
    return {"hypotheses": hyp, "arms": arms, "budgets": budgets, "cut_orders": cuts, "explained_null": en}


def render_prereg(*, code_sha: Optional[str], world_yaml: Optional[Path], harness_yaml: Optional[Path],
                  conformal_json: Optional[Path], split_json: Optional[Path], router_model: Optional[str],
                  model_revisions: Optional[Sequence[str]], contracts: Optional[Mapping[str, Mapping[str, str]]],
                  template: Path = TEMPLATE) -> str:
    missing: List[str] = []
    if not code_sha:
        missing.append("code_sha")
    values: Dict[str, Any] = {}
    for key, path in zip(FILE_INPUTS, (world_yaml, harness_yaml, conformal_json, split_json)):
        if path is None or not Path(path).is_file() or Path(path).stat().st_size == 0:
            missing.append(f"{key} ({path})")
        else:
            values[key.split("_")[0] + "_sha256"] = sha256_file(Path(path))
    if not router_model:
        missing.append("router_model")
    if not model_revisions:
        missing.append("model_revisions")
    if not contracts or any(not v.get("address") or not v.get("code_hash") for v in contracts.values()):
        missing.append("contracts (each needs address and code_hash)")
    if missing:
        raise PreregError("cannot render preregistration, missing: " + ", ".join(missing))
    values.update(code_sha=code_sha, router_model=router_model,
                  model_revisions="\n".join(f"- {m}" for m in model_revisions or ()),
                  contracts="\n".join(f"- {n}: `{v['address']}`, `{v['code_hash']}`" for n, v in sorted((contracts or {}).items())),
                  **_static_sections())
    return string.Template(Path(template).read_text()).substitute(values)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--world", type=Path, default=Path("config/v4_1/world.yaml"))
    ap.add_argument("--harness", type=Path, default=Path("config/v4_1/harness.yaml"))
    ap.add_argument("--conformal", type=Path, default=Path("config/v4_1/conformal.json"))
    ap.add_argument("--split", type=Path, default=Path("config/v4_1/split.json"))
    ap.add_argument("--router-model")
    ap.add_argument("--model-revisions", type=Path, help="JSON list of 'model@revision' strings")
    ap.add_argument("--contracts", type=Path, help="JSON {name: {address, code_hash}}")
    ap.add_argument("--out", type=Path, default=Path("PREREGISTRATION.md"))
    a = ap.parse_args(argv)
    try:
        text = render_prereg(
            code_sha=git_sha(a.repo), world_yaml=a.world, harness_yaml=a.harness, conformal_json=a.conformal,
            split_json=a.split, router_model=a.router_model,
            model_revisions=json.loads(a.model_revisions.read_text()) if a.model_revisions and a.model_revisions.is_file() else None,
            contracts=json.loads(a.contracts.read_text()) if a.contracts and a.contracts.is_file() else None)
    except PreregError as exc:
        raise SystemExit(str(exc))
    a.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
