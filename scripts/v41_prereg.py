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
    """Numbers that live in code are substituted from it; the prose is verbatim goal text in the template."""
    t = A.thresholds()
    return {"bootstrap_resamples": str(t["bootstrap_resamples"]), "bootstrap_seed": str(t["bootstrap_seed"]),
            "power": str(t["power"])}


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
