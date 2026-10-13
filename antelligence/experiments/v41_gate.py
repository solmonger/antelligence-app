"""v4.1 eval gate (brief S5): eval cases need a committed, unmodified prereg inside the study repo, a confirmed
chain anchor receipt bound to that prereg, and the operator's G1 approval file. Dev cases are always allowed;
reserve and unknown cases never are. The split is always loaded from the repo; callers cannot supply one.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Union

from antelligence.experiments.v41_split import load_split, which_split

CHAIN_ID = 300
SPLIT_REL = Path("config/v4_1/split.json")
_TX = re.compile(r"0x[0-9a-f]{64}")
_SPLIT_SHA = re.compile(r"split file sha256[^`\n]*`([0-9a-f]{64})`")
_LISTS = ("dev_a", "dev_b", "eval", "reserve")


class GateError(RuntimeError):
    """An eval run was refused."""


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checked_split(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / SPLIT_REL
    if not path.is_file():
        raise GateError(f"split file not found: {path}")
    doc = load_split(path)
    seen: Dict[str, str] = {}
    for key in _LISTS:
        for cid in doc[key]:
            if cid in seen:
                raise GateError(f"case id {cid!r} appears in both {seen[cid]} and {key}")
            seen[cid] = key
    return doc


def _check_prereg(repo_root: Path, prereg_path: Path) -> None:
    if not prereg_path.is_file():
        raise GateError(f"preregistration not found: {prereg_path}")
    try:
        rel = prereg_path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise GateError("preregistration is outside the study repo") from exc
    if _git(repo_root, "cat-file", "-e", f"HEAD:{rel.as_posix()}").returncode != 0:
        raise GateError("preregistration is not committed in HEAD")
    head = _git(repo_root, "show", f"HEAD:{rel.as_posix()}", text=False)
    if head.returncode != 0 or head.stdout != prereg_path.read_bytes() or \
            _git(repo_root, "diff", "--quiet", "HEAD", "--", rel.as_posix()).returncode != 0:
        raise GateError("preregistration has uncommitted changes")


def _check_receipt(receipt_path: Path, prereg_path: Path) -> None:
    if not receipt_path.is_file():
        raise GateError(f"anchor receipt not found: {receipt_path}")
    try:
        rec = json.loads(receipt_path.read_text())
    except json.JSONDecodeError as exc:
        raise GateError(f"anchor receipt is not JSON: {exc}") from exc
    if not isinstance(rec, dict):
        raise GateError("anchor receipt is not an object")
    if not isinstance(rec.get("tx_hash"), str) or not _TX.fullmatch(rec["tx_hash"]):
        raise GateError("anchor receipt tx_hash is not 0x plus 64 lowercase hex")
    status = rec.get("status")
    if not (type(status) is int and status == 1):
        raise GateError("anchor receipt status is not 1")
    if rec.get("readback_ok") is not True:
        raise GateError("anchor receipt readback_ok is not true")
    chain = rec.get("chain_id")
    if not (type(chain) is int and chain == CHAIN_ID):
        raise GateError(f"anchor receipt chain_id is not {CHAIN_ID}")
    if rec.get("prereg_sha256") != sha256_file(prereg_path):
        raise GateError("anchor receipt prereg_sha256 is missing or does not match the preregistration file")


def _check_split_hash(repo_root: Path, prereg_path: Path) -> None:
    m = _SPLIT_SHA.search(prereg_path.read_text())
    if not m:
        raise GateError("preregistration does not record the split file sha256")
    if m.group(1) != sha256_file(repo_root / SPLIT_REL):
        raise GateError("split file sha256 differs from the value recorded in the preregistration")


def assert_eval_allowed(case: str, repo_root: Union[str, Path], prereg_path: Union[str, Path],
                        anchor_receipt: Union[str, Path], g1_file: Union[str, Path]) -> str:
    """Return the case's split name ('dev' or 'eval') or raise GateError."""
    root, prereg = Path(repo_root), Path(prereg_path)
    doc = load_checked_split(root)
    try:
        kind = which_split(doc, case)
    except KeyError as exc:
        raise GateError(str(exc)) from exc
    if kind == "dev":
        return kind
    if kind != "eval":
        raise GateError(f"case {case!r} is in the reserve set and may not be run")
    _check_prereg(root, prereg)
    _check_split_hash(root, prereg)
    _check_receipt(Path(anchor_receipt), prereg)
    g1 = Path(g1_file)
    if not g1.is_file() or not g1.read_text().strip():
        raise GateError("G1 approval file is missing or empty")
    return kind
