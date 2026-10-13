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


MAINNET_POW_LIMIT = 0xFFFF << (8 * (0x1D - 3))       # target of nBits 0x1d00ffff: difficulty 1
# Bitcoin mainnet difficulty has been far above this since 2020. A header cheaper than this is not a real block,
# so a lying header source would have to burn ~2^32 * MIN_DIFFICULTY hashes per forged header.
MIN_DIFFICULTY = 1e12


class HeaderError(RuntimeError):
    """A block header failed a local sanity check (hash, proof of work, difficulty floor)."""


def nbits_target(nbits: int) -> int:
    """Decode Bitcoin's compact difficulty target; raises :class:`HeaderError` for negative or zero targets."""
    exponent, mantissa = nbits >> 24, nbits & 0x007FFFFF
    if nbits & 0x00800000 or mantissa == 0:
        raise HeaderError(f"invalid compact target 0x{nbits:08x}")
    return mantissa << (8 * (exponent - 3)) if exponent > 3 else mantissa >> (8 * (3 - exponent))


def check_header_pow(header: CBlockHeader, *, pow_limit: int = MAINNET_POW_LIMIT, min_difficulty: float = MIN_DIFFICULTY) -> None:
    """The header's double-SHA256 must be below its own target, the target must be a real Bitcoin one
    (at or below ``pow_limit`` and at least ``min_difficulty`` times harder than difficulty 1)."""
    target = nbits_target(int(header.nBits))
    if target > pow_limit:
        raise HeaderError("header target is easier than the proof-of-work limit")
    if MAINNET_POW_LIMIT / target < min_difficulty:
        raise HeaderError(f"header difficulty {MAINNET_POW_LIMIT / target:.3g} is below the floor {min_difficulty:.3g}")
    if int.from_bytes(header.GetHash(), "little") > target:
        raise HeaderError("header hash does not meet its own target (no proof of work)")


class BlockstreamHeaderProvider:
    """Read-only GETs to a public Esplora instance for Bitcoin block headers.

    The server is NOT trusted for the header's content: the header must hash to the block hash the server named,
    and must carry proof of work at a real Bitcoin difficulty (:func:`check_header_pow`), so a lying server needs
    to out-mine the network rather than just return a header whose merkle root it chose. What this cannot
    check without a node is that the header sits on the best chain (documented in DESIGN-anchors.md).
    """

    def __init__(self, base_url: str = "https://blockstream.info/api", timeout_s: float = 20.0, *,
                 fetch: Optional[Callable[[str], str]] = None, pow_limit: int = MAINNET_POW_LIMIT,
                 min_difficulty: float = MIN_DIFFICULTY) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.pow_limit, self.min_difficulty = pow_limit, min_difficulty
        self._fetch = fetch or self._http_get

    def _http_get(self, path: str) -> str:
        import httpx

        resp = httpx.get(f"{self.base_url}/{path}", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.text.strip()

    def __call__(self, height: int) -> CBlockHeader:
        block_hash = self._fetch(f"block-height/{height}")
        header = CBlockHeader.deserialize(bytes.fromhex(self._fetch(f"block/{block_hash}/header")))
        if header.GetHash()[::-1].hex() != block_hash.lower():
            raise HeaderError("header does not hash to the block hash the server named")
        check_header_pow(header, pow_limit=self.pow_limit, min_difficulty=self.min_difficulty)
        return header


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
    """``ok`` is True ONLY for status ``bitcoin``: a Bitcoin attestation whose merkle path ends in a
    block header that the header provider returned and that matches.

    Everything else fails closed: ``unverified`` (calendar promise only, no header source, unknown
    attestation types) or ``invalid`` (wrong digest, unreadable, header mismatch).
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
    except Exception as exc:        # a garbage proof raises anything (struct, recursion, type errors): it is invalid, never "unverifiable"
        return OtsVerification(False, "invalid", False, error=f"unreadable proof: {type(exc).__name__}: {exc}")
    matches = detached.file_digest == expected_digest and detached.file_hash_op.__class__ is OpSHA256
    if not matches:
        return OtsVerification(False, "invalid", False, error="proof is for a different digest")
    records: List[dict] = []
    best: Optional[int] = None
    invalid: List[str] = []
    lookup_failed: List[str] = []
    for msg, att in detached.timestamp.all_attestations():
        if isinstance(att, BitcoinBlockHeaderAttestation):
            rec = {"kind": "bitcoin", "height": att.height, "checked": False}
            if header_provider is not None:
                try:
                    t = att.verify_against_blockheader(msg, header_provider(att.height))
                    rec.update(checked=True, time=int(t))
                    best = int(t) if best is None else min(best, int(t))
                except VerificationError as exc:
                    invalid.append(f"bitcoin attestation at height {att.height} fails: {exc}")
                except Exception as exc:
                    lookup_failed.append(f"header lookup failed at height {att.height}: {exc}")
            records.append(rec)
        elif isinstance(att, PendingAttestation):
            records.append({"kind": "pending", "uri": att.uri})
        else:
            records.append({"kind": "unknown"})
    # The verdict is the WORST over all attestations: one proven-bad attestation makes the proof invalid no matter
    # what order they are visited in or whether another header lookup happened to fail first.
    if invalid:
        return OtsVerification(False, "invalid", True, records, error=invalid[0])
    if best is not None:
        return OtsVerification(True, "bitcoin", True, records, time=best)
    if lookup_failed:
        return OtsVerification(False, "unverified", True, records, error=lookup_failed[0])
    reason = ("no Bitcoin attestation (calendar promise only; upgrade the proof)"
              if not any(r["kind"] == "bitcoin" for r in records)
              else "no block header source: Bitcoin attestation not checked")
    return OtsVerification(False, "unverified", True, records, error=reason)
