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


_HIDDEN_RE = re.compile(r"<!--.*?(?:-->|\Z)|^(?P<f>`{3,}|~{3,}).*?(?:^(?P=f)[`~]*[ \t]*$|\Z)", re.MULTILINE | re.DOTALL)


def _rendered(text: str) -> str:
    """The text a reader sees: HTML comments and fenced code blocks (also unterminated ones) removed."""
    return _HIDDEN_RE.sub("", text)


# A code span cannot cross a blank line (a paragraph break), so an unmatched backtick cannot shield later text.
_CODE_SPAN_RE = re.compile(r"(`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)\1(?!`)", re.DOTALL)
_RAW_HTML_RE = re.compile(r"<[A-Za-z!/]")


def reject_raw_html(text: str) -> None:
    """The preregistration is plain Markdown: raw HTML (``<div hidden>``, ``<details>``, comments, closing tags) can
    hide or restyle text from a reader while the verifier's regex still sees it. Inline code spans are exempt."""
    stripped = _CODE_SPAN_RE.sub("", text)
    m = _RAW_HTML_RE.search(stripped)
    if m:
        line = stripped.count("\n", 0, m.start()) + 1
        raise PinError("the preregistration must contain no raw HTML (found "
                       f"{stripped[m.start():m.start() + 12]!r} near line {line} outside inline code)")


def study_key_in(prereg: Union[bytes, str]) -> str:
    """The study key fingerprint written into a preregistration document.

    The document must contain no raw HTML (``PinError``). Exactly one matching line in the raw text AND exactly one in the rendered text (outside HTML comments and
    code fences, where a reader would not see it): a line hidden in a comment or code block can neither be the
    key nor sit beside the real one.
    """
    text = prereg.decode("utf-8", "replace") if isinstance(prereg, (bytes, bytearray)) else prereg
    reject_raw_html(text)
    raw, shown = STUDY_KEY_RE.findall(text), STUDY_KEY_RE.findall(_rendered(text))
    if len(raw) != 1 or len(shown) != 1 or raw != shown:
        raise PinError("the preregistration must state the study key fingerprint exactly once, in visible text "
                       f"(found {len(raw)} in the raw text, {len(shown)} outside comments and code blocks)")
    return shown[0]


def load_pins(deployments: Mapping[str, Any], prereg: Union[bytes, str]) -> TrustPins:
    """Build pins from a parsed deployments file and the preregistration document."""
    record = deployments.get("v4_1") if isinstance(deployments, Mapping) else None
    if not isinstance(record, Mapping) or record.get("complete") is not True:
        raise PinError("deployments file has no complete v4_1 record")
    listed = record.get("contracts")
    if not isinstance(listed, list) or not all(isinstance(c, Mapping) for c in listed):
        raise PinError("deployments v4_1 record has no contracts list")
    contracts = [c for c in listed if c.get("name") == REGISTRY]
    if len(contracts) != 1:
        raise PinError(f"deployments v4_1 record must list {REGISTRY} exactly once")
    entry = contracts[0]
    code_hash = entry.get("code_hash", "")          # exact: lowercase 0x + 64 hex, as the deploy script writes it
    if not isinstance(code_hash, str) or not re.fullmatch(r"0x[0-9a-f]{64}", code_hash):
        raise PinError(f"{REGISTRY} has no code_hash in the deployments record (redeploy with the v4.1 deploy script)")
    try:
        registry = Web3.to_checksum_address(entry.get("address", ""))
        deployer = Web3.to_checksum_address(record.get("deployer", ""))
    except (ValueError, TypeError) as exc:
        raise PinError(f"bad address in the deployments record: {exc}") from exc
    try:                                   # no default: a missing per-contract deployer is a gap, not a match
        entry_deployer = Web3.to_checksum_address(entry.get("deployer", ""))
    except (ValueError, TypeError) as exc:
        raise PinError(f"{REGISTRY} entry has no valid deployer: {exc}") from exc
    if entry_deployer != deployer:
        raise PinError(f"{REGISTRY} entry names a different deployer than the v4_1 record")
    return TrustPins(registry=registry, deployer=deployer, code_hash=code_hash, study_key_sha256=study_key_in(prereg))


def _git(repo: Union[str, Path], *args: str) -> "subprocess.CompletedProcess[str]":
    # --no-replace-objects: refs/replace/* in a (caller-supplied) repo must not be able to swap the pinned blob.
    return subprocess.run(["git", "--no-replace-objects", "-C", str(repo), *args], capture_output=True, text=True)


def _require_commit(repo: Union[str, Path], commit: str) -> None:
    if not _COMMIT_RE.match(commit or ""):
        raise PinError("a full 40-hex commit id is required (no branch names: they move)")
    out = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if out.returncode != 0 or out.stdout.strip() != commit:
        raise PinError(f"{commit} is not a commit object in the repository")


def _show(repo: Union[str, Path], commit: str, path: str) -> bytes:
    out = subprocess.run(["git", "--no-replace-objects", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True)
    if out.returncode != 0:
        raise PinError(f"cannot read {path} at {commit}: {out.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return out.stdout


def _no_duplicates(pairs: list) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not allowed")


def parse_strict_json(raw: Union[bytes, str]) -> Any:
    """``json.loads`` that refuses duplicate object keys (last-one-wins hides a forged entry) and NaN/Infinity."""
    return json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)


def deployments_at_commit(repo: Union[str, Path], commit: str, path: str = DEFAULT_DEPLOYMENTS) -> Dict[str, Any]:
    """Read the deployments file as it was at a full 40-hex git commit (read-only ``git show``)."""
    _require_commit(repo, commit)
    try:
        parsed = parse_strict_json(_show(repo, commit, path))
    except ValueError as exc:
        raise PinError(f"{path} at {commit} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise PinError(f"{path} at {commit} is not a JSON object")
    return parsed


def pins_from_repo(repo: Union[str, Path], commit: str, prereg_path: Union[str, Path], path: str = DEFAULT_DEPLOYMENTS,
                   prereg_commit: Optional[str] = None) -> TrustPins:
    """Pins from the deployments file at ``commit`` and the preregistration.

    With ``prereg_commit`` the preregistration is read from git too (``prereg_path`` is then a path inside the
    repo) and ``commit`` must be an ancestor of, or equal to, ``prereg_commit``: the deployment the study key was
    locked against cannot be a side-branch commit made afterwards. Without it the working-tree file is only as
    good as its digest on chain (the verifier checks that); ``repo`` is caller-supplied either way, so pass a
    clone of the public repository.
    """
    deployments = deployments_at_commit(repo, commit, path)
    if prereg_commit is None:
        return load_pins(deployments, Path(prereg_path).read_bytes())
    _require_commit(repo, prereg_commit)
    if _git(repo, "merge-base", "--is-ancestor", commit, prereg_commit).returncode != 0:
        raise PinError("the deployments commit is not an ancestor of the preregistration commit")
    return load_pins(deployments, _show(repo, prereg_commit, str(prereg_path)))


def prereg_digest(prereg: bytes) -> bytes:
    return hashlib.sha256(prereg).digest()
