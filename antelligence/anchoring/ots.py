"""OpenTimestamps client: stamp a 32-byte anchor digest, upgrade pending proofs, verify a ``.ots`` file.

All network access goes through an injectable :class:`OtsTransport`, so the tests run offline.
The proof format is the standard detached ``.ots`` file (``opentimestamps`` library), so any
third-party OTS tool can verify what this module writes.

The stamped file digest is the anchor digest itself (INTERFACES section 3). As in ``ots stamp``, a
random nonce is appended and hashed before the value goes to the calendars, so a calendar never sees
the digest it is timestamping; the nonce is part of the proof.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Union

from bitcoin.core import CBlockHeader
from opentimestamps.core.notary import (
    BitcoinBlockHeaderAttestation,
    PendingAttestation,
    VerificationError,
)
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.serialize import (
    BytesDeserializationContext,
    BytesSerializationContext,
    DeserializationError,
)
from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

DEFAULT_CALENDARS = (
    "https://alice.btc.calendar.opentimestamps.org",
    "https://bob.btc.calendar.opentimestamps.org",
    "https://finney.calendar.eternitywall.com",
)
_ACCEPT = {"Accept": "application/vnd.opentimestamps.v1", "User-Agent": "antelligence-v4.1"}


class OtsError(RuntimeError):
    """Stamping or upgrading failed."""


class CommitmentNotReady(KeyError):
    """The calendar has no (final) timestamp for this commitment yet."""


class OtsTransport(Protocol):
    def submit(self, calendar_url: str, digest: bytes) -> Timestamp:
        """POST ``digest`` to the calendar; returns the calendar's timestamp for it."""

    def get_timestamp(self, calendar_url: str, commitment: bytes) -> Timestamp:
        """GET the calendar's timestamp for ``commitment``; raise :class:`CommitmentNotReady` if pending."""


class HttpOtsTransport:
    """Real calendars over HTTPS (the only network users of this module)."""

    def __init__(self, timeout_s: float = 20.0) -> None:
        self.timeout_s = timeout_s

    def submit(self, calendar_url: str, digest: bytes) -> Timestamp:
        import httpx

        resp = httpx.post(calendar_url.rstrip("/") + "/digest", content=digest, headers=_ACCEPT, timeout=self.timeout_s)
        resp.raise_for_status()
        return Timestamp.deserialize(BytesDeserializationContext(resp.content[:10000]), digest)

    def get_timestamp(self, calendar_url: str, commitment: bytes) -> Timestamp:
        import httpx

        resp = httpx.get(calendar_url.rstrip("/") + "/timestamp/" + commitment.hex(), headers=_ACCEPT, timeout=self.timeout_s)
        if resp.status_code == 404:
            raise CommitmentNotReady(commitment.hex())
        resp.raise_for_status()
        return Timestamp.deserialize(BytesDeserializationContext(resp.content[:10000]), commitment)


HeaderProvider = Callable[[int], CBlockHeader]


class BlockstreamHeaderProvider:
    """Read-only GETs to a public Esplora instance for Bitcoin block headers."""

    def __init__(self, base_url: str = "https://blockstream.info/api", timeout_s: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def __call__(self, height: int) -> CBlockHeader:
        import httpx

        block_hash = httpx.get(f"{self.base_url}/block-height/{height}", timeout=self.timeout_s).text.strip()
        raw = httpx.get(f"{self.base_url}/block/{block_hash}/header", timeout=self.timeout_s).text.strip()
        return CBlockHeader.deserialize(bytes.fromhex(raw))


@dataclass
class StampResult:
    path: Path
    calendars_ok: List[str] = field(default_factory=list)
    calendars_failed: Dict[str, str] = field(default_factory=dict)


@dataclass
class UpgradeResult:
    upgraded: int = 0
    pending: int = 0
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.pending == 0


@dataclass
class OtsVerification:
    """``ok``: the proof commits to the expected digest and is structurally sound.

    ``status``: ``bitcoin`` (a Bitcoin attestation was checked against a block header), ``bitcoin_unchecked``
    (attestation present but no header source given), ``pending`` (calendar promise only) or ``invalid``.
    """

    ok: bool
    status: str
    digest_matches: bool
    attestations: List[dict] = field(default_factory=list)
    time: Optional[int] = None
    error: Optional[str] = None


def _serialise(detached: DetachedTimestampFile) -> bytes:
    ctx = BytesSerializationContext()
    detached.serialize(ctx)
    return ctx.getbytes()


def load_ots(source: Union[str, Path, bytes]) -> DetachedTimestampFile:
    raw = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    return DetachedTimestampFile.deserialize(BytesDeserializationContext(bytes(raw)))


class OtsClient:
    def __init__(
        self,
        transport: OtsTransport,
        calendars: Sequence[str] = DEFAULT_CALENDARS,
        min_ok: int = 1,
        nonce_fn: Callable[[], bytes] = lambda: os.urandom(16),
    ) -> None:
        self.transport = transport
        self.calendars = tuple(calendars)
        self.min_ok = min_ok
        self.nonce_fn = nonce_fn

    # -- stamp -----------------------------------------------------------------------------------
    def stamp(self, digest: bytes, path: Union[str, Path]) -> StampResult:
        """Stamp the 32-byte ``digest`` against the calendars and save the (pending) proof to ``path``."""
        if len(digest) != 32:
            raise OtsError("digest must be 32 bytes")
        detached = DetachedTimestampFile(OpSHA256(), Timestamp(digest))
        salted = detached.timestamp.ops.add(OpAppend(self.nonce_fn())).ops.add(OpSHA256())
        result = StampResult(path=Path(path))
        for url in self.calendars:
            try:
                salted.merge(self.transport.submit(url, salted.msg))
                result.calendars_ok.append(url)
            except Exception as exc:  # network and parse errors both just mean "this calendar failed"
                result.calendars_failed[url] = f"{type(exc).__name__}: {exc}"
        if len(result.calendars_ok) < self.min_ok:
            raise OtsError(f"only {len(result.calendars_ok)} of {self.min_ok} required calendars accepted: {result.calendars_failed}")
        result.path.parent.mkdir(parents=True, exist_ok=True)
        result.path.write_bytes(_serialise(detached))
        return result

    # -- upgrade ---------------------------------------------------------------------------------
    def upgrade(self, path: Union[str, Path]) -> UpgradeResult:
        """Replace calendar promises by Bitcoin attestations where the calendars have them; rewrite the file."""
        detached = load_ots(path)
        out = UpgradeResult()
        for msg, att in list(detached.timestamp.all_attestations()):
            if not isinstance(att, PendingAttestation):
                continue
            sub = self._find(detached.timestamp, msg)
            try:
                upgraded = self.transport.get_timestamp(att.uri, msg)
            except CommitmentNotReady:
                out.pending += 1
                continue
            except Exception as exc:
                out.pending += 1
                out.errors[att.uri] = f"{type(exc).__name__}: {exc}"
                continue
            if any(isinstance(a, BitcoinBlockHeaderAttestation) for _, a in upgraded.all_attestations()):
                sub.merge(upgraded)
                out.upgraded += 1
            else:
                out.pending += 1
        if out.upgraded:
            Path(path).write_bytes(_serialise(detached))
        return out

    @staticmethod
    def _find(root: Timestamp, msg: bytes) -> Timestamp:
        stack = [root]
        while stack:
            ts = stack.pop()
            if ts.msg == msg and any(isinstance(a, PendingAttestation) for a in ts.attestations):
                return ts
            stack.extend(ts.ops.values())
        raise OtsError("pending attestation vanished")

    # -- verify ----------------------------------------------------------------------------------
    def verify(
        self,
        source: Union[str, Path, bytes],
        expected_digest: bytes,
        header_provider: Optional[HeaderProvider] = None,
    ) -> OtsVerification:
        return verify_ots(source, expected_digest, header_provider)


def verify_ots(
    source: Union[str, Path, bytes],
    expected_digest: bytes,
    header_provider: Optional[HeaderProvider] = None,
) -> OtsVerification:
    """Check a ``.ots`` proof commits to ``expected_digest`` and, given headers, that its Bitcoin attestations hold."""
    try:
        detached = load_ots(source)
    except (DeserializationError, ValueError, OSError) as exc:
        return OtsVerification(False, "invalid", False, error=f"unreadable proof: {exc}")
    matches = detached.file_digest == expected_digest and detached.file_hash_op.__class__ is OpSHA256
    if not matches:
        return OtsVerification(False, "invalid", False, error="proof is for a different digest")
    records: List[dict] = []
    best: Optional[int] = None
    status = "pending"
    for msg, att in detached.timestamp.all_attestations():
        if isinstance(att, BitcoinBlockHeaderAttestation):
            rec = {"kind": "bitcoin", "height": att.height}
            if header_provider is None:
                rec["checked"] = False
                status = "bitcoin_unchecked" if status != "bitcoin" else status
            else:
                try:
                    t = att.verify_against_blockheader(msg, header_provider(att.height))
                except VerificationError as exc:
                    return OtsVerification(False, "invalid", True, records, error=f"bitcoin attestation fails: {exc}")
                except Exception as exc:
                    return OtsVerification(False, "invalid", True, records, error=f"header lookup failed: {exc}")
                rec.update(checked=True, time=int(t))
                best = int(t) if best is None else min(best, int(t))
                status = "bitcoin"
            records.append(rec)
        elif isinstance(att, PendingAttestation):
            records.append({"kind": "pending", "uri": att.uri})
        else:
            records.append({"kind": "unknown"})
    if not records:
        return OtsVerification(False, "invalid", True, error="proof has no attestation")
    return OtsVerification(True, status, True, records, time=best)
