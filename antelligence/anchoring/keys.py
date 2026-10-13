"""v4.1 key custody (goal 2.5): where private keys live, who can read them, and how they are destroyed.

* Per-run ed25519 board keys and the per-run P-256 anchor key are made by ``KeyVault`` (v41_harness) under ``ledger/``
  (``<label>.keys.json``, ``<label>.anchor.pem``), mode 0600.
* One **study key** (P-256) at ``ledger/v4.1-study-key`` signs only the preregistration and run-plan Rekor entries.
* :func:`delete_run_keys` removes every per-run private file after the tamper suite and appends one row per deletion to
  ``ledger/v4.1-key-deletions.jsonl``. Rows carry the file name and the public fingerprint, never key material.

Nothing here prints or returns a private key. Private files can only be created or deleted inside a directory whose
path has a ``ledger`` component.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cryptography.hazmat.primitives import serialization

from antelligence.anchoring.rekor import AnchorKey, fingerprint_of

STUDY_KEY_NAME = "v4.1-study-key"
STUDY_PUB_NAME = "v4.1-study-key.pub.pem"
DELETION_LOG = "v4.1-key-deletions.jsonl"
RUN_KEY_SUFFIXES = (".keys.json", ".anchor.pem")


class KeyError_(RuntimeError):
    """A key operation was refused."""


def _ledger(ledger_dir: Union[str, Path]) -> Path:
    directory = Path(ledger_dir).resolve()
    if "ledger" not in directory.parts:
        raise KeyError_("private keys live only under a ledger/ directory")
    return directory


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.chmod(path, 0o600)


def create_study_key(ledger_dir: Union[str, Path]) -> str:
    """Generate the study key once. Returns its fingerprint (sha256 of the SPKI DER). Refuses to overwrite."""
    directory = _ledger(ledger_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / STUDY_KEY_NAME
    if target.exists():
        raise KeyError_(f"{target.name} already exists; the study key is made once")
    key = AnchorKey()
    _write_private(target, key._key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                   serialization.NoEncryption()))
    (directory / STUDY_PUB_NAME).write_bytes(key.pem)
    return key.fingerprint


def load_study_key(ledger_dir: Union[str, Path]) -> AnchorKey:
    path = _ledger(ledger_dir) / STUDY_KEY_NAME
    if not path.exists():
        raise KeyError_(f"{STUDY_KEY_NAME} not found; run `scripts/v41_keys.py init-study` first")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise KeyError_(f"{STUDY_KEY_NAME} is readable by group/others; refusing to use it (chmod 600)")
    return AnchorKey(serialization.load_pem_private_key(path.read_bytes(), password=None))


def study_fingerprint(ledger_dir: Union[str, Path]) -> str:
    """From the public half only (no private read)."""
    pub = _ledger(ledger_dir) / STUDY_PUB_NAME
    if not pub.exists():
        raise KeyError_(f"{STUDY_PUB_NAME} not found")
    return fingerprint_of(serialization.load_pem_public_key(pub.read_bytes()))


def run_key_files(ledger_dir: Union[str, Path]) -> List[Path]:
    directory = _ledger(ledger_dir)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.name.endswith(RUN_KEY_SUFFIXES))


def audit(ledger_dir: Union[str, Path]) -> Dict[str, Any]:
    """Names and modes only. ``bad_mode`` lists any private file that group or others can read."""
    files = run_key_files(ledger_dir)
    study = _ledger(ledger_dir) / STUDY_KEY_NAME
    allfiles = files + ([study] if study.exists() else [])
    return {"run_key_files": len(files), "study_key_present": study.exists(),
            "bad_mode": [p.name for p in allfiles if stat.S_IMODE(p.stat().st_mode) & 0o077]}


def _fingerprint_of_pem(path: Path) -> Optional[str]:
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        return fingerprint_of(key.public_key())
    except Exception:
        return None


def delete_run_keys(ledger_dir: Union[str, Path], *, confirm: bool, now=time.time) -> List[Dict[str, Any]]:
    """Delete every per-run private key file (never the study key) and log each deletion."""
    if confirm is not True:
        raise KeyError_("deleting run keys is irreversible; pass confirm=True (--confirm)")
    directory = _ledger(ledger_dir)
    rows: List[Dict[str, Any]] = []
    for path in run_key_files(directory):
        row: Dict[str, Any] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now())), "event": "delete_run_key",
                               "file": path.name, "anchor_fingerprint": _fingerprint_of_pem(path) if path.name.endswith(".pem") else None}
        path.unlink()
        rows.append(row)
    with (directory / DELETION_LOG).open("a", encoding="utf-8") as log:
        for row in rows:
            log.write(json.dumps(row, sort_keys=True) + "\n")
        if not rows:
            log.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now())),
                                  "event": "delete_run_keys", "files": 0}) + "\n")
    return rows
