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
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise RekorError(f"malformed rekor entry {uuid}: {exc}") from exc


def check_record(record: RekorRecord) -> List[str]:
    """Problems with one record; empty list means the signature and (if present) the inclusion proof hold."""
    problems: List[str] = []
    try:
        pub = load_public_pem(record.pubkey_pem)
        pub.verify(record.signature, bytes.fromhex(record.digest), ec.ECDSA(utils.Prehashed(hashes.SHA256())))
    except (InvalidSignature, RekorError, ValueError) as exc:
        problems.append(f"signature: {exc or 'invalid'}")
    proof = record.inclusion_proof
    if proof:
        try:
            ok = verify_inclusion(
                record.body, int(proof["logIndex"]), int(proof["treeSize"]),
                [bytes.fromhex(h) for h in proof["hashes"]], bytes.fromhex(proof["rootHash"]),
            )
        except (KeyError, ValueError, TypeError):
            ok = False
        if not ok:
            problems.append("inclusion proof does not reach the stated root")
    return problems


class RekorClient:
    def __init__(self, transport: Optional[RekorTransport] = None) -> None:
        self.transport: RekorTransport = transport or HttpRekorTransport()
        self.calls = 0

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
