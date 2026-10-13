"""v4.1 eval gate (brief S5): eval cases need a committed, unmodified prereg, a confirmed chain anchor
receipt and the operator's G1 approval file. Dev cases are always allowed; reserve and unknown cases never are.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Union

from antelligence.experiments.v41_split import load_split, which_split

CHAIN_ID = 300


class GateError(RuntimeError):
    """An eval run was refused."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _check_prereg(prereg_path: Path) -> None:
    if not prereg_path.is_file():
        raise GateError(f"preregistration not found: {prereg_path}")
    repo = prereg_path.resolve().parent
    name = prereg_path.name
    if _git(repo, "ls-files", "--error-unmatch", "--", name).returncode != 0:
        raise GateError("preregistration is not committed")
    if _git(repo, "status", "--porcelain", "--", name).stdout.strip():
        raise GateError("preregistration has uncommitted changes")


def _check_receipt(receipt_path: Path, prereg_path: Path) -> None:
    if not receipt_path.is_file():
        raise GateError(f"anchor receipt not found: {receipt_path}")
    try:
        rec: Mapping[str, Any] = json.loads(receipt_path.read_text())
    except json.JSONDecodeError as exc:
        raise GateError(f"anchor receipt is not JSON: {exc}") from exc
    if not isinstance(rec, dict) or not rec.get("tx_hash"):
        raise GateError("anchor receipt has no tx_hash")
    if rec.get("status") != 1:
        raise GateError("anchor receipt status is not 1")
    if rec.get("readback_ok") is not True:
        raise GateError("anchor receipt readback_ok is not true")
    if rec.get("chain_id") != CHAIN_ID:
        raise GateError(f"anchor receipt chain_id is not {CHAIN_ID}")
    claimed = rec.get("prereg_sha256")  # optional binding of the receipt to the exact document
    if claimed is not None and claimed != hashlib.sha256(prereg_path.read_bytes()).hexdigest():
        raise GateError("anchor receipt prereg_sha256 does not match the preregistration file")


def assert_eval_allowed(case: str, split: Union[Mapping[str, Any], str, Path], prereg_path: Union[str, Path],
                        anchor_receipt: Union[str, Path], g1_file: Union[str, Path]) -> str:
    """Return the case's split name ('dev' or 'eval') or raise GateError."""
    doc = split if isinstance(split, Mapping) else load_split(Path(split))
    try:
        kind = which_split(doc, case)
    except KeyError as exc:
        raise GateError(str(exc)) from exc
    if kind == "dev":
        return kind
    if kind != "eval":
        raise GateError(f"case {case!r} is in the reserve set and may not be run")
    _check_prereg(Path(prereg_path))
    _check_receipt(Path(anchor_receipt), Path(prereg_path))
    g1 = Path(g1_file)
    if not g1.is_file() or not g1.read_text().strip():
        raise GateError("G1 approval file is missing or empty")
    return kind
