#!/usr/bin/env python3
"""Render PREREGISTRATION.md from docs/v4.1/PREREGISTRATION.template.md. Refuses if any input is missing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from antelligence.anchoring.pins import PinError, reject_raw_html
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
                  conformal_json: Optional[Path], split_json: Optional[Path], router_model_file: Optional[Path],
                  model_revisions_json: Optional[Path], contracts: Optional[Mapping[str, Mapping[str, str]]],
                  study_key_sha256: Optional[str] = None, template: Path = TEMPLATE) -> str:
    missing: List[str] = []
    if not code_sha:
        missing.append("code_sha")
    values: Dict[str, Any] = {}
    for key, path in zip(FILE_INPUTS, (world_yaml, harness_yaml, conformal_json, split_json)):
        if path is None or not Path(path).is_file() or Path(path).stat().st_size == 0:
            missing.append(f"{key} ({path})")
        else:
            values[key.split("_")[0] + "_sha256"] = sha256_file(Path(path))
    revisions: List[str] = []
    for key, slot, path in (("router_model_file", "router_sha256", router_model_file),
                            ("model_revisions_json", "revisions_sha256", model_revisions_json)):
        if path is None or not Path(path).is_file() or Path(path).stat().st_size == 0:
            missing.append(f"{key} ({path})")
        else:
            values[slot] = sha256_file(Path(path))
    if model_revisions_json is not None and Path(model_revisions_json).is_file():
        entries = json.loads(Path(model_revisions_json).read_text()).get("models", [])
        if not entries or any(not e.get("name") or not e.get("revision") or "TODO" in e["revision"] for e in entries):
            missing.append("model_revisions_json (empty list or TODO revision)")
        revisions = [f"{e.get('name')}@{e.get('revision')}" for e in entries]
    if not contracts or any(not v.get("address") or not v.get("code_hash") for v in contracts.values()):
        missing.append("contracts (each needs address and code_hash)")
    if not study_key_sha256 or not re.fullmatch(r"[0-9a-f]{64}", study_key_sha256):
        missing.append("study_key_sha256 (sha256 of the study Rekor key's SPKI DER, 64 lowercase hex)")
    if missing:
        raise PreregError("cannot render preregistration, missing: " + ", ".join(missing))
    values.update(code_sha=code_sha, study_key_sha256=study_key_sha256, router_model=Path(router_model_file).name,
                  model_revisions="\n".join(f"- {m}" for m in revisions),
                  contracts="\n".join(f"- {n}: `{v['address']}`, `{v['code_hash']}`" for n, v in sorted((contracts or {}).items())),
                  **_static_sections())
    text = string.Template(Path(template).read_text()).substitute(values)
    try:
        reject_raw_html(text)         # the verifier refuses such a document, so never render one
    except PinError as exc:
        raise PreregError(f"cannot render preregistration: {exc}") from exc
    return text


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--world", type=Path, default=Path("config/v4_1/world.yaml"))
    ap.add_argument("--harness", type=Path, default=Path("config/v4_1/harness.yaml"))
    ap.add_argument("--conformal", type=Path, default=Path("config/v4_1/conformal.json"))
    ap.add_argument("--split", type=Path, default=Path("config/v4_1/split.json"))
    ap.add_argument("--router-model-file", type=Path, help="the frozen router model file")
    ap.add_argument("--model-revisions", type=Path, default=Path("config/v4_1/model_revisions.json"))
    ap.add_argument("--contracts", type=Path, help="JSON {name: {address, code_hash}}")
    ap.add_argument("--study-key-sha256", help="fingerprint of the study Rekor key (printed by the study-key generator)")
    ap.add_argument("--out", type=Path, default=Path("PREREGISTRATION.md"))
    a = ap.parse_args(argv)
    try:
        text = render_prereg(
            code_sha=git_sha(a.repo), world_yaml=a.world, harness_yaml=a.harness, conformal_json=a.conformal,
            split_json=a.split, router_model_file=a.router_model_file, model_revisions_json=a.model_revisions,
            contracts=json.loads(a.contracts.read_text()) if a.contracts and a.contracts.is_file() else None,
            study_key_sha256=a.study_key_sha256)
    except PreregError as exc:
        raise SystemExit(str(exc))
    a.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
