"""Sigstore Rekor client: log an anchor digest as a ``hashedrekord`` entry and find a run's entries by public key.

Key type (checked against the public hashedrekord schema and entry.go on 2026-10-12): Rekor verifies
Ed25519 hashedrekord signatures only as **Ed25519ph** (RFC 8032 pre-hashed), which the Python
``cryptography`` library does not provide. The run's anchor key is therefore an **ECDSA P-256** key and
the signature is ECDSA-SHA256 over the anchor-digest preimage, i.e. a pre-hashed signature over the
digest that goes in ``data.hash.value``. INTERFACES keeps ``anchor_pubkey`` as 32 bytes: it is
``sha256(SubjectPublicKeyInfo DER)`` of the P-256 key, and the public PEM travels next to the export
(Rekor can only search by the full key, not by a fingerprint).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

from antelligence.kernel.board import BoardError, verify_inclusion

DEFAULT_REKOR_URL = "https://rekor.sigstore.dev"
API_VERSION = "0.0.1"
# sha256 of the SubjectPublicKeyInfo DER of the public Rekor log key (ECDSA P-256), fetched once on
# 2026-10-13 from GET https://rekor.sigstore.dev/api/v1/log/publicKey. Also the production logID.
REKOR_LOG_KEY_SHA256 = "c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d"


class RekorError(RuntimeError):
    """A Rekor request failed or returned something unusable."""


class RekorConflict(RekorError):
    """Rekor already has this exact entry (HTTP 409); ``uuid`` is the existing one."""

    def __init__(self, uuid: str) -> None:
        super().__init__(f"entry already exists: {uuid}")
        self.uuid = uuid


class RekorTransport(Protocol):
    def create(self, proposed: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/v1/log/entries -> ``{uuid: entry}``."""

    def search_by_pubkey(self, pem: bytes) -> List[str]:
        """POST /api/v1/index/retrieve (read-only lookup) -> entry UUIDs logged under this public key."""

    def get_entry(self, uuid: str) -> Dict[str, Any]:
        """GET /api/v1/log/entries/{uuid} -> ``{uuid: entry}``."""

    def log_public_key(self) -> bytes:
        """GET /api/v1/log/publicKey -> PEM of the log's signing key."""


class HttpRekorTransport:
    def __init__(self, base_url: str = DEFAULT_REKOR_URL, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _check(self, resp) -> Dict[str, Any]:
        if resp.status_code >= 400:
            raise RekorError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def create(self, proposed: Dict[str, Any]) -> Dict[str, Any]:
        import httpx

        resp = httpx.post(f"{self.base_url}/api/v1/log/entries", json=proposed, timeout=self.timeout_s)
        if resp.status_code == 409:
            loc = resp.headers.get("Location", "")
            raise RekorConflict(loc.rsplit("/", 1)[-1])
        return self._check(resp)

    def search_by_pubkey(self, pem: bytes) -> List[str]:
        import httpx

        body = {"publicKey": {"format": "x509", "content": base64.b64encode(pem).decode()}}
        resp = httpx.post(f"{self.base_url}/api/v1/index/retrieve", json=body, timeout=self.timeout_s)
        return list(self._check(resp))

    def get_entry(self, uuid: str) -> Dict[str, Any]:
        import httpx

        return self._check(httpx.get(f"{self.base_url}/api/v1/log/entries/{uuid}", timeout=self.timeout_s))

    def log_public_key(self) -> bytes:
        import httpx

        resp = httpx.get(f"{self.base_url}/api/v1/log/publicKey", timeout=self.timeout_s)
        if resp.status_code >= 400:
            raise RekorError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.content


# ---------------------------------------------------------------------------
# Anchor key (ECDSA P-256)
# ---------------------------------------------------------------------------

def spki_der(public_key: ec.EllipticCurvePublicKey) -> bytes:
    return public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)


def spki_pem(public_key: ec.EllipticCurvePublicKey) -> bytes:
    return public_key.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def fingerprint_of(public_key: ec.EllipticCurvePublicKey) -> str:
    """The 32-byte ``anchor_pubkey`` of the run header: sha256 of the SPKI DER, lowercase hex."""
    return hashlib.sha256(spki_der(public_key)).hexdigest()


def load_public_pem(pem: bytes) -> ec.EllipticCurvePublicKey:
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, ec.EllipticCurvePublicKey) or not isinstance(key.curve, ec.SECP256R1):
        raise RekorError("anchor key must be an ECDSA P-256 public key")
    return key


class AnchorKey:
    """Per-run Rekor signing key. The private half never leaves the signer or a ``ledger/`` directory."""

    def __init__(self, private_key: Optional[ec.EllipticCurvePrivateKey] = None) -> None:
        self._key = private_key or ec.generate_private_key(ec.SECP256R1())

    @property
    def public_key(self) -> ec.EllipticCurvePublicKey:
        return self._key.public_key()

    @property
    def pem(self) -> bytes:
        return spki_pem(self.public_key)

    @property
    def fingerprint(self) -> str:
        return fingerprint_of(self.public_key)

    def sign_digest(self, digest: bytes) -> bytes:
        """DER ECDSA signature over ``digest`` as a pre-hashed SHA-256 value (equals signing its preimage)."""
        if len(digest) != 32:
            raise RekorError("digest must be 32 bytes")
        # RFC 6979 deterministic: a retried log call builds the identical entry, so Rekor answers 409, not a duplicate
        return self._key.sign(digest, ec.ECDSA(utils.Prehashed(hashes.SHA256()), deterministic_signing=True))

    def save(self, ledger_dir: Union[str, Path], label: str = "run") -> Path:
        directory = Path(ledger_dir).resolve()
        if "ledger" not in directory.parts:
            raise BoardError("private keys may only be written under a ledger/ directory")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{label}.anchor.pem"
        pem = self._key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as handle:
            handle.write(pem)
        return target


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

def build_hashedrekord(digest: bytes, key: AnchorKey) -> Dict[str, Any]:
    """The proposed ``hashedrekord`` v0.0.1 entry for one anchor digest."""
    return {
        "apiVersion": API_VERSION,
        "kind": "hashedrekord",
        "spec": {
            "signature": {
                "content": base64.b64encode(key.sign_digest(digest)).decode(),
                "publicKey": {"content": base64.b64encode(key.pem).decode()},
            },
            "data": {"hash": {"algorithm": "sha256", "value": digest.hex()}},
        },
    }


@dataclass
class RekorRecord:
    uuid: str
    log_index: int
    integrated_time: int
    body: bytes                      # the canonical entry body: the Merkle leaf preimage
    digest: str                      # data.hash.value, hex
    pubkey_pem: bytes
    signature: bytes
    inclusion_proof: Optional[Dict[str, Any]] = None
    raw_bytes: int = 0               # size of the response, for the verifier's byte accounting
    log_id: str = ""
    body_b64: str = ""
    set_b64: str = ""                # signedEntryTimestamp
    time_trusted: bool = False       # integrated_time is covered by a verified SET


def parse_entry(uuid: str, entry: Dict[str, Any], raw_bytes: int = 0) -> RekorRecord:
    try:
        body = base64.b64decode(entry["body"])
        spec = json.loads(body)["spec"]
        if json.loads(body)["kind"] != "hashedrekord":
            raise RekorError("not a hashedrekord entry")
        verification = entry.get("verification") or {}
        return RekorRecord(
            uuid=uuid,
            log_index=int(entry["logIndex"]),
            integrated_time=int(entry["integratedTime"]),
            body=body,
            digest=spec["data"]["hash"]["value"],
            pubkey_pem=base64.b64decode(spec["signature"]["publicKey"]["content"]),
            signature=base64.b64decode(spec["signature"]["content"]),
            inclusion_proof=verification.get("inclusionProof"),
            raw_bytes=raw_bytes,
            log_id=str(entry.get("logID", "")),
            body_b64=str(entry["body"]),
            set_b64=str(verification.get("signedEntryTimestamp", "")),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise RekorError(f"malformed rekor entry {uuid}: {exc}") from exc


def pinned_log_key(pem: bytes, pinned_sha256: str) -> ec.EllipticCurvePublicKey:
    """The log's public key, accepted only if its SPKI DER hashes to the pinned value."""
    key = load_public_pem(pem)
    if fingerprint_of(key) != pinned_sha256.lower():
        raise RekorError("Rekor log public key does not match the pinned sha256")
    return key


def _checkpoint_problems(proof: Dict[str, Any], log_key: ec.EllipticCurvePublicKey) -> List[str]:
    """Verify the signed checkpoint (a C2SP signed note) and that it states the proof's size and root."""
    cp = proof.get("checkpoint")
    if not cp:
        return ["no signed checkpoint"]
    text, sep, sigs = cp.partition("\n\n")
    if not sep:
        return ["malformed checkpoint"]
    note = text + "\n"
    lines = note.split("\n")
    try:
        size_ok = int(lines[1]) == int(proof["treeSize"])
        root_ok = base64.b64decode(lines[2]).hex() == proof["rootHash"].lower()
    except (IndexError, ValueError, KeyError):
        return ["malformed checkpoint"]
    out = [] if size_ok and root_ok else ["checkpoint size/root differ from the inclusion proof"]
    verified = False
    for line in sigs.splitlines():
        if not line.startswith("\u2014 "):
            continue
        try:
            raw = base64.b64decode(line.split(" ", 2)[2])
            log_key.verify(raw[4:], note.encode(), ec.ECDSA(hashes.SHA256()))
            verified = True
        except (InvalidSignature, ValueError, IndexError):
            continue
    if not verified:
        out.append("checkpoint signature does not verify under the pinned Rekor log key")
    return out


def set_payload(record: "RekorRecord") -> bytes:
    """Canonical JSON the log signs for the signed entry timestamp (SET)."""
    return json.dumps(
        {"body": record.body_b64, "integratedTime": record.integrated_time, "logID": record.log_id, "logIndex": record.log_index},
        sort_keys=True, separators=(",", ":"),
    ).encode()


def check_record(
    record: RekorRecord, searched_pem: bytes, log_key: Optional[ec.EllipticCurvePublicKey]
) -> List[str]:
    """Problems with one record; empty means: right key, valid signature, inclusion proof reaching a
    checkpoint signed by the pinned log key. Sets ``record.time_trusted`` iff the SET verifies.

    Fail closed: no log key, no proof, or no checkpoint is a problem, never a pass.
    """
    problems: List[str] = []
    record.time_trusted = False
    try:
        pub = load_public_pem(record.pubkey_pem)
        if spki_der(pub) != spki_der(load_public_pem(searched_pem)):
            problems.append("entry public key is not the searched anchor key")
        pub.verify(record.signature, bytes.fromhex(record.digest), ec.ECDSA(utils.Prehashed(hashes.SHA256())))
    except (InvalidSignature, RekorError, ValueError) as exc:
        problems.append(f"signature: {exc or 'invalid'}")
    if log_key is None:
        problems.append("no pinned Rekor log key: inclusion cannot be authenticated")
        return problems
    proof = record.inclusion_proof
    if not proof:
        problems.append("no inclusion proof")
    else:
        try:
            ok = verify_inclusion(
                record.body, int(proof["logIndex"]), int(proof["treeSize"]),
                [bytes.fromhex(h) for h in proof["hashes"]], bytes.fromhex(proof["rootHash"]),
            )
        except (KeyError, ValueError, TypeError):
            ok = False
        if not ok:
            problems.append("inclusion proof does not reach the stated root")
        problems += _checkpoint_problems(proof, log_key)
    try:
        log_key.verify(base64.b64decode(record.set_b64), set_payload(record), ec.ECDSA(hashes.SHA256()))
        record.time_trusted = True
    except (InvalidSignature, ValueError):
        record.time_trusted = False       # integratedTime is reported but unverified
    return problems


class RekorClient:
    def __init__(self, transport: Optional[RekorTransport] = None, pinned_log_key_sha256: str = REKOR_LOG_KEY_SHA256) -> None:
        self.transport: RekorTransport = transport or HttpRekorTransport()
        self.pinned_log_key_sha256 = pinned_log_key_sha256
        self.calls = 0
        self._log_key: Optional[ec.EllipticCurvePublicKey] = None

    def log_key(self) -> ec.EllipticCurvePublicKey:
        """The log's public key, fetched once and accepted only if it matches the pinned sha256."""
        if self._log_key is None:
            self.calls += 1
            self._log_key = pinned_log_key(self.transport.log_public_key(), self.pinned_log_key_sha256)
        return self._log_key

    def log_digest(self, digest: bytes, key: AnchorKey) -> RekorRecord:
        """Create the entry (idempotent: an existing identical entry is fetched instead)."""
        self.calls += 1
        proposed = build_hashedrekord(digest, key)
        try:
            resp = self.transport.create(proposed)
        except RekorConflict as exc:
            return self.get(exc.uuid)
        (uuid, entry), = resp.items()
        return parse_entry(uuid, entry)

    def get(self, uuid: str) -> RekorRecord:
        self.calls += 1
        resp = self.transport.get_entry(uuid)
        (key, entry), = resp.items()
        return parse_entry(key, entry, raw_bytes=len(json.dumps(resp)))

    def entries_for_key(self, pem: bytes) -> List[RekorRecord]:
        """Every entry logged under this public key, ordered by log index."""
        self.calls += 1
        uuids = self.transport.search_by_pubkey(pem)
        records = [self.get(u) for u in uuids]
        return sorted(records, key=lambda r: r.log_index)
