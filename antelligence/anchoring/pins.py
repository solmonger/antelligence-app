"""Trust anchors for the outside verifier: pinned from public, operator-independent inputs only.

A verifier that is told "the registry is at X, the deployer is Y, the study key is Z" by the same party it is
auditing has verified nothing. The pins come from two places the operator cannot quietly change:

* the **deployments file at a git commit** (``blockchain/deployments/<network>.json``, key ``v4_1``): the
  ``AnchorRegistry`` address, its ``code_hash`` (keccak256 of the code the chain serves there) and the deployer;
* the **preregistration**, whose digest is ``anchorOnce``'d on chain: it carries the study key fingerprint
  (sha256 of the SPKI DER of the key that signs the study's Rekor entries).

Anything a caller supplies on top (a registry, a deployer, a study PEM) is only compared against these pins.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from web3 import Web3

from antelligence.anchoring.rekor import RekorError, fingerprint_of, load_public_pem

DEFAULT_DEPLOYMENTS = "blockchain/deployments/zksync-era-sepolia.json"
REGISTRY = "AnchorRegistry"
# The one line of the preregistration that fixes the study key. Exactly one must exist.
STUDY_KEY_RE = re.compile(r"^Study key fingerprint \(sha256 of SPKI DER\): `([0-9a-f]{64})`\s*$", re.MULTILINE)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class PinError(ValueError):
    """The public inputs do not yield a complete, consistent set of pins."""


@dataclass(frozen=True)
class TrustPins:
    registry: str            # checksum address
    deployer: str            # checksum address
    code_hash: str           # "0x" + keccak256(code at registry)
    study_key_sha256: str    # lowercase hex

    def check_study_pem(self, pem: bytes) -> Optional[str]:
        """``None`` if ``pem`` is the pinned study key, else the reason it is not."""
        try:
            fp = fingerprint_of(load_public_pem(pem))
        except (RekorError, ValueError) as exc:
            return f"study public key unreadable: {exc}"
        return None if fp == self.study_key_sha256 else "study public key is not the key fixed in the preregistration"


def study_key_in(prereg: Union[bytes, str]) -> str:
    """The study key fingerprint written into a preregistration document (exactly one such line)."""
    text = prereg.decode("utf-8", "replace") if isinstance(prereg, (bytes, bytearray)) else prereg
    found = STUDY_KEY_RE.findall(text)
    if len(found) != 1:
        raise PinError(f"the preregistration must state the study key fingerprint exactly once (found {len(found)})")
    return found[0]


def load_pins(deployments: Mapping[str, Any], prereg: Union[bytes, str]) -> TrustPins:
    """Build pins from a parsed deployments file and the preregistration document."""
    record = deployments.get("v4_1") or {}
    if record.get("complete") is not True:
        raise PinError("deployments file has no complete v4_1 record")
    contracts = [c for c in record.get("contracts", []) if c.get("name") == REGISTRY]
    if len(contracts) != 1:
        raise PinError(f"deployments v4_1 record must list {REGISTRY} exactly once")
    entry = contracts[0]
    code_hash = str(entry.get("code_hash", "")).lower()
    if not re.fullmatch(r"0x[0-9a-f]{64}", code_hash):
        raise PinError(f"{REGISTRY} has no code_hash in the deployments record (redeploy with the v4.1 deploy script)")
    try:
        registry = Web3.to_checksum_address(entry.get("address", ""))
        deployer = Web3.to_checksum_address(record.get("deployer", ""))
    except (ValueError, TypeError) as exc:
        raise PinError(f"bad address in the deployments record: {exc}") from exc
    if str(entry.get("deployer", record["deployer"])).lower() != deployer.lower():
        raise PinError(f"{REGISTRY} entry names a different deployer than the v4_1 record")
    return TrustPins(registry=registry, deployer=deployer, code_hash=code_hash, study_key_sha256=study_key_in(prereg))


def deployments_at_commit(repo: Union[str, Path], commit: str, path: str = DEFAULT_DEPLOYMENTS) -> Dict[str, Any]:
    """Read the deployments file as it was at a full 40-hex git commit (read-only ``git show``)."""
    if not _COMMIT_RE.match(commit or ""):
        raise PinError("a full 40-hex commit id is required (no branch names: they move)")
    out = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True, text=True)
    if out.returncode != 0:
        raise PinError(f"cannot read {path} at {commit}: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout)


def pins_from_repo(repo: Union[str, Path], commit: str, prereg_path: Union[str, Path], path: str = DEFAULT_DEPLOYMENTS) -> TrustPins:
    return load_pins(deployments_at_commit(repo, commit, path), Path(prereg_path).read_bytes())


def prereg_digest(prereg: bytes) -> bytes:
    return hashlib.sha256(prereg).digest()
