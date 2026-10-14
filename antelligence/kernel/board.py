"""v4.1 board: a typed, hash-chained, optionally signed, Merkle-committed log.

Design decision (new module, not an extension of BoardField/EvidenceMemory):
``BoardField`` is a TTL-expiring signal store keyed by topic, and ``EvidenceMemory``
is a mutable record store with admission and invalidation. The v4.1 board is the
opposite on both counts: append-only, never expiring, never mutated, and every
entry must be byte-for-byte reproducible on the Python, JS and Solidity sides. Bending
either existing type to that contract would break their current guarantees. What is
reused is ``kernel.canonical`` (extended with the strict ``jcs_bytes`` encoder) and the
same sqlite3 storage style as ``EvidenceMemory``. The kernel never imports a world.

Contract: ``docs/v4.1/INTERFACES.md`` (sections 1-4, 6, 7). Vectors:
``tests/v4_1/vectors/interfaces.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from antelligence.kernel.canonical import MAX_SAFE_INT, CanonicalError, jcs_bytes

ENTRY_TYPES = ("task", "knowledge", "evidence", "route", "decision", "edit", "alarm")
TYPE_CODES = {name: code for code, name in enumerate(ENTRY_TYPES)}
ZERO_HASH = "0" * 64
ANCHOR_EVERY = 50
DEFAULT_AUTHORS = ("perception", "queen", "harness", "router") + tuple(f"squad-{i}" for i in range(8))

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX128 = re.compile(r"^[0-9a-f]{128}$")
_AUTHOR = re.compile(r"^(queen|perception|harness|router|squad-[0-7])$")
HEADER_KEYS = (
    "study", "arm", "case", "seed", "split", "world_cfg_sha256", "harness_cfg_sha256",
    "pubkeys", "anchor_pubkey", "code_sha",
)


class BoardError(ValueError):
    """The board refused an entry or a header."""


class ValidationError(BoardError):
    """An entry or payload does not match the contract."""


class BoardIntegrityError(BoardError):
    """A stored or imported board fails id, signature, chain or header checks."""


# ---------------------------------------------------------------------------
# Payload validators (INTERFACES.md section 4): exact keys, integers only.
# ---------------------------------------------------------------------------

def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and -MAX_SAFE_INT <= v <= MAX_SAFE_INT


def _int(lo: int = -MAX_SAFE_INT, hi: int = MAX_SAFE_INT) -> Callable[[Any], bool]:
    return lambda v: _is_int(v) and lo <= v <= hi


def _str(max_len: int) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, str) and len(v) <= max_len


def _enum(*options: str) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, str) and v in options


def _triple(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 3 and all(_is_int(x) for x in v)


def _int_list(v: Any) -> bool:
    return isinstance(v, list) and all(_is_int(x) for x in v)


def _id_list(max_len: int) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, list) and len(v) <= max_len and all(isinstance(x, str) and _HEX64.match(x) for x in v)


def _prediction(v: Any) -> bool:
    return (
        isinstance(v, dict) and set(v) == {"metric", "delta_ppm"}
        and isinstance(v["metric"], str) and _is_int(v["delta_ppm"])
    )


_PPM = _int(0, 1_000_000)
_FLAG = _int(0, 1)
_NONNEG = _int(0)

_PAYLOADS: Dict[str, Dict[str, Callable[[Any], bool]]] = {
    "task": {"objective": _str(280), "squads": _int_list, "until_tick": _int()},
    "knowledge": {
        "region_id": _int(), "centroid_dmm": _triple, "cls": _enum("WT", "TC", "ET"),
        "p_tumour_ppm": _PPM, "volume_mm3": _NONNEG, "unc_ppm": _PPM,
        "source": _enum("perception", "local_look"), "prov": _str(1 << 16),
    },
    "evidence": {
        "squad": _int(), "pos_vox": _triple, "look_p_ppm": _PPM, "p_tumour_ppm": _PPM,
        "dose_ug": _NONNEG, "released_ug": _NONNEG,
    },
    "route": {"squad": _int(), "region_ids": _int_list, "priority": _int(1, 5), "brief_ids": _id_list(12)},
    "decision": {
        "squad": _int(), "round": _int(), "target": _enum("A", "B", "C", "D", "E", "F", "S", "X"),
        "release": _FLAG, "amount_level": _int(1, 5), "p_target_ppm": _PPM, "p_release_ppm": _PPM,
        "p_amount_ppm": _PPM, "backend": _str(1 << 16), "deferred": _FLAG, "router_pick": _str(1 << 16),
        "brief_ids": _id_list(12),
    },
    "edit": {
        "round": _int(), "patch_sha256": lambda v: isinstance(v, str) and bool(_HEX64.match(v)),
        "keys": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
        "prediction": _prediction, "kept": lambda v: v is None or (_is_int(v) and v in (0, 1)),
    },
    "alarm": {
        "kind": _enum("defer_rate", "timeout", "identity", "other"), "value_ppm": _int(), "note": _str(140),
    },
}


# v4.2 colony evidence subtypes (INTERFACES v4.2 sections 2.4, 3.3, 4). An ``evidence`` payload WITHOUT ``subtype`` keeps the
# six-key v4.1 spec above, byte for byte; one WITH ``subtype`` is checked against that subtype's exact key set.
MARK_KINDS = ("dosed", "dose_ok", "slow_uptake", "washout", "tumour_here", "not_tumour", "need_more")

_EVIDENCE_SUBTYPES: Dict[str, Dict[str, Callable[[Any], bool]]] = {
    "mark": {
        "subtype": _enum("mark"), "mark_kind": _enum(*MARK_KINDS), "ant": _int(0), "pos_vox": _triple,
        "amount_milli": _NONNEG, "reading_ppm": _PPM,
    },
    "message": {
        "subtype": _enum("message"), "msg_type": _enum("ask", "answer"), "msg_id": _int(0), "from_ant": _int(0),
        "to_ant": _int(0), "about_pos_vox": _triple, "about_region": _int(-1), "question": _int(1, 3),
        "sender_reading_ppm": _PPM, "reply_to": _int(-1), "answer": _int(-1, 2),
        "answerer_reading_ppm": _int(-1, 1_000_000), "note": _str(120),
    },
    "status": {
        "subtype": _enum("status"), "region_id": _int(), "status": _enum("done", "not_tumour"), "k": _int(0),
        "tick_first": _int(0),
    },
    "need": {
        "subtype": _enum("need"), "region_id": _int(), "need": _enum("none", "low", "medium", "high"),
        "need_milli": _NONNEG, "delivered_milli": _NONNEG,
    },
}


def validate_payload(entry_type: str, payload: Any) -> None:
    """Raise :class:`ValidationError` unless ``payload`` has exactly the keys for ``entry_type``."""
    if entry_type not in _PAYLOADS:
        raise ValidationError(f"unknown entry type {entry_type!r}")
    if not isinstance(payload, dict):
        raise ValidationError("payload must be an object")
    spec = _PAYLOADS[entry_type]
    if entry_type == "evidence" and "subtype" in payload:
        subtype = payload["subtype"]
        if not isinstance(subtype, str) or subtype not in _EVIDENCE_SUBTYPES:
            raise ValidationError(f"unknown evidence subtype {subtype!r}")
        spec = _EVIDENCE_SUBTYPES[subtype]
    extra = set(payload) - set(spec)
    missing = set(spec) - set(payload)
    if extra or missing:
        raise ValidationError(f"{entry_type} payload keys: extra={sorted(extra)} missing={sorted(missing)}")
    for key, check in spec.items():
        if not check(payload[key]):
            raise ValidationError(f"{entry_type}.{key}: invalid value {payload[key]!r}")


def _normalise_author(author: Union[str, Mapping[str, Any]]) -> Dict[str, str]:
    if isinstance(author, str):
        author = {"agent": author, "model": "none", "rev": "none"}
    if not isinstance(author, Mapping) or set(author) != {"agent", "model", "rev"}:
        raise ValidationError("author must have exactly agent, model, rev")
    out = {k: author[k] for k in ("agent", "model", "rev")}
    if not all(isinstance(v, str) and v for v in out.values()):
        raise ValidationError("author fields must be non-empty strings")
    if not _AUTHOR.match(out["agent"]):
        raise ValidationError(f"unknown author agent {out['agent']!r}")
    return out


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

_ENTRY_KEYS = {"run_id", "seq", "type", "author", "cites", "payload", "tick", "prev_hash", "id", "sig"}


def _detach(value: Any) -> Any:
    try:
        return json.loads(jcs_bytes(value))
    except CanonicalError as exc:
        raise ValidationError(str(exc)) from exc


@dataclass(frozen=True, eq=False)
class Entry:
    run_id: str
    seq: int
    type: str
    author: Mapping[str, str]
    cites: Tuple[str, ...]
    payload: Mapping[str, Any]
    tick: int
    prev_hash: str
    id: str
    sig: Optional[str]

    def body(self) -> Dict[str, Any]:
        """The entry without ``id`` and ``sig``: the thing that is hashed."""
        return {
            "run_id": self.run_id, "seq": self.seq, "type": self.type, "author": dict(self.author),
            "cites": list(self.cites), "payload": _detach(self.payload), "tick": self.tick,
            "prev_hash": self.prev_hash,
        }

    def to_dict(self) -> Dict[str, Any]:
        out = self.body()
        out["id"] = self.id
        out["sig"] = self.sig
        return out

    def line(self) -> str:
        return jcs_bytes(self.to_dict()).decode("utf-8")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entry) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Entry":
        if not isinstance(data, Mapping) or set(data) != _ENTRY_KEYS:
            raise ValidationError(f"entry keys must be exactly {sorted(_ENTRY_KEYS)}")
        data = _detach(dict(data))
        return cls(
            run_id=data["run_id"], seq=data["seq"], type=data["type"], author=data["author"],
            cites=tuple(data["cites"]), payload=data["payload"], tick=data["tick"],
            prev_hash=data["prev_hash"], id=data["id"], sig=data["sig"],
        )


def compute_id(body: Mapping[str, Any]) -> str:
    """sha256 hex of the JCS encoding of the entry without ``id`` and ``sig``."""
    return hashlib.sha256(jcs_bytes({k: v for k, v in body.items() if k not in ("id", "sig")})).hexdigest()


def _validate_shape(e: Entry) -> None:
    if not (isinstance(e.run_id, str) and _HEX64.match(e.run_id)):
        raise ValidationError("run_id must be 64 hex")
    if not (_is_int(e.seq) and e.seq >= 0):
        raise ValidationError("seq must be a non-negative integer")
    if e.type not in TYPE_CODES:
        raise ValidationError(f"unknown entry type {e.type!r}")
    _normalise_author(e.author)
    if not (_is_int(e.tick) and e.tick >= 0):
        raise ValidationError("tick must be a non-negative integer")
    cites = list(e.cites)
    if cites != sorted(set(cites)) or not all(isinstance(c, str) and _HEX64.match(c) for c in cites):
        raise ValidationError("cites must be sorted, unique, 64-hex entry ids")
    for field in ("prev_hash", "id"):
        if not (isinstance(getattr(e, field), str) and _HEX64.match(getattr(e, field))):
            raise ValidationError(f"{field} must be 64 hex")
    if e.sig is not None and not (isinstance(e.sig, str) and _HEX128.match(e.sig)):
        raise ValidationError("sig must be 128 hex or null")
    validate_payload(e.type, e.payload if isinstance(e.payload, dict) else dict(e.payload))


# ---------------------------------------------------------------------------
# Keys and signatures (ed25519 over the 32 raw bytes of the id)
# ---------------------------------------------------------------------------

def pubkey_hex(seed: bytes) -> str:
    return Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def sign_id(seed: bytes, entry_id: str) -> str:
    return Ed25519PrivateKey.from_private_bytes(seed).sign(bytes.fromhex(entry_id)).hex()


def verify_sig(pubkey: str, entry_id: str, sig: str) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey)).verify(bytes.fromhex(sig), bytes.fromhex(entry_id))
        return True
    except (InvalidSignature, ValueError):
        return False


@dataclass(frozen=True)
class RunKeys:
    pubkeys: Dict[str, str]        # author -> 32-byte hex, goes in the run header
    seeds: Dict[str, bytes]        # author -> 32-byte private seed, never leaves the ledger


def generate_run_keys(
    authors: Iterable[str] = DEFAULT_AUTHORS,
    ledger_dir: Optional[Union[str, Path]] = None,
    label: str = "run",
) -> RunKeys:
    """Fresh ed25519 key per author. Private seeds are written only under a ``ledger/`` directory."""
    seeds: Dict[str, bytes] = {}
    pubkeys: Dict[str, str] = {}
    for author in authors:
        if not _AUTHOR.match(author):
            raise BoardError(f"unknown author {author!r}")
        seed = os.urandom(32)
        seeds[author] = seed
        pubkeys[author] = pubkey_hex(seed)
    if ledger_dir is not None:
        directory = Path(ledger_dir).resolve()
        if "ledger" not in directory.parts:
            raise BoardError("private keys may only be written under a ledger/ directory")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{label}.keys.json"
        payload = json.dumps({a: s.hex() for a, s in seeds.items()}, sort_keys=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
    return RunKeys(pubkeys=pubkeys, seeds=seeds)


# ---------------------------------------------------------------------------
# RFC 6962 / RFC 9162 Merkle tree over entry ids (32 raw bytes each)
# ---------------------------------------------------------------------------

def _sha(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _leaf_hash(d: bytes) -> bytes:
    return _sha(b"\x00" + d)


def _node_hash(left: bytes, right: bytes) -> bytes:
    return _sha(b"\x01" + left + right)


def _split(n: int) -> int:
    """Largest power of two strictly smaller than n (n >= 2)."""
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    """Merkle Tree Hash of the raw leaf data."""
    n = len(leaves)
    if n == 0:
        return _sha(b"")
    if n == 1:
        return _leaf_hash(leaves[0])
    k = _split(n)
    return _node_hash(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


def inclusion_path(index: int, leaves: Sequence[bytes]) -> List[bytes]:
    n = len(leaves)
    if not 0 <= index < n:
        raise BoardError("leaf index out of range")
    if n == 1:
        return []
    k = _split(n)
    if index < k:
        return inclusion_path(index, leaves[:k]) + [merkle_root(leaves[k:])]
    return inclusion_path(index - k, leaves[k:]) + [merkle_root(leaves[:k])]


def _subproof(m: int, leaves: Sequence[bytes], complete: bool) -> List[bytes]:
    n = len(leaves)
    if m == n:
        return [] if complete else [merkle_root(leaves)]
    k = _split(n)
    if m <= k:
        return _subproof(m, leaves[:k], complete) + [merkle_root(leaves[k:])]
    return _subproof(m - k, leaves[k:], False) + [merkle_root(leaves[:k])]


def consistency_path(old_size: int, leaves: Sequence[bytes]) -> List[bytes]:
    """RFC 6962 PROOF(old_size, D[n]). Empty when old_size is 0 or equals n."""
    n = len(leaves)
    if not 0 <= old_size <= n:
        raise BoardError("old_size out of range")
    if old_size == 0 or old_size == n:
        return []
    return _subproof(old_size, leaves, True)


def verify_inclusion(leaf: bytes, index: int, size: int, path: Sequence[bytes], root: bytes) -> bool:
    """RFC 9162 section 2.1.3.2."""
    if index >= size or index < 0:
        return False
    fn, sn = index, size - 1
    r = _leaf_hash(leaf)
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = _node_hash(p, r)
            while not fn & 1 and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            r = _node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def verify_consistency(old_size: int, new_size: int, old_root: bytes, new_root: bytes, proof: Sequence[bytes]) -> bool:
    """RFC 9162 section 2.1.4.2."""
    if old_size < 0 or old_size > new_size:
        return False
    if old_size == new_size:
        return not proof and old_root == new_root
    if old_size == 0:
        return not proof
    if not proof:
        return False
    proof = list(proof)
    if old_size & (old_size - 1) == 0:
        proof = [old_root] + proof
    fn, sn = old_size - 1, new_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    fr = sr = proof[0]
    for c in proof[1:]:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            fr = _node_hash(c, fr)
            sr = _node_hash(c, sr)
            while not fn & 1 and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            sr = _node_hash(sr, c)
        fn >>= 1
        sn >>= 1
    return fr == old_root and sr == new_root and sn == 0


def anchor_digest(run_id: str, size: int, root: bytes) -> bytes:
    """sha256(runId(32) || uint64_be(size) || root(32)): the 72-byte preimage in INTERFACES section 3."""
    if size < 0 or size >= 1 << 64:
        raise BoardError("size out of range")
    return _sha(bytes.fromhex(run_id) + size.to_bytes(8, "big") + root)


def anchor_sizes(final_size: int) -> List[int]:
    """Anchor points: every multiple of 50, plus the final size (once)."""
    sizes = list(range(ANCHOR_EVERY, final_size + 1, ANCHOR_EVERY))
    if final_size > 0 and (not sizes or sizes[-1] != final_size):
        sizes.append(final_size)
    return sizes


# ---------------------------------------------------------------------------
# Run header
# ---------------------------------------------------------------------------

def validate_header(header: Mapping[str, Any]) -> None:
    if not isinstance(header, Mapping) or set(header) != set(HEADER_KEYS):
        raise ValidationError(f"run header keys must be exactly {sorted(HEADER_KEYS)}")
    if header["split"] not in ("dev", "eval"):
        raise ValidationError("header.split must be dev or eval")
    pubkeys = header["pubkeys"]
    if not isinstance(pubkeys, Mapping) or not all(
        _AUTHOR.match(a) and isinstance(k, str) and _HEX64.match(k) for a, k in pubkeys.items()
    ):
        raise ValidationError("header.pubkeys must map authors to 32-byte hex keys")
    if not (isinstance(header["anchor_pubkey"], str) and _HEX64.match(header["anchor_pubkey"])):
        raise ValidationError("header.anchor_pubkey must be 32-byte hex")
    try:
        jcs_bytes(header)
    except CanonicalError as exc:
        raise ValidationError(str(exc)) from exc


def run_id_of(header: Mapping[str, Any]) -> str:
    return hashlib.sha256(jcs_bytes(header)).hexdigest()


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entries (seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, body TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS entries_no_update BEFORE UPDATE ON entries
    BEGIN SELECT RAISE(ABORT, 'board entries are append-only'); END;
CREATE TRIGGER IF NOT EXISTS entries_no_delete BEFORE DELETE ON entries
    BEGIN SELECT RAISE(ABORT, 'board entries are append-only'); END;
"""


class Board:
    """One run's log. Mode ``A`` is unsigned (prev_hash chain only); mode ``B`` signs every entry."""

    def __init__(
        self,
        run_header: Mapping[str, Any],
        mode: str = "A",
        keys: Optional[Mapping[str, bytes]] = None,
        db_path: Union[str, Path] = ":memory:",
    ) -> None:
        if mode not in ("A", "B"):
            raise BoardError("mode must be 'A' or 'B'")
        validate_header(run_header)
        self.header: Dict[str, Any] = _detach(dict(run_header))
        self.run_id = run_id_of(self.header)
        self.mode = mode
        self._seeds: Dict[str, bytes] = dict(keys or {})
        if mode == "B":
            for author, seed in self._seeds.items():
                if self.header["pubkeys"].get(author) != pubkey_hex(seed):
                    raise BoardError(f"key for {author!r} does not match the run header")
        self._entries: List[Entry] = []
        self._by_id: Dict[str, int] = {}
        self._db = sqlite3.connect(str(db_path))
        self._db.executescript(_DDL)
        stored = self._db.execute("SELECT v FROM meta WHERE k='run_id'").fetchone()
        if stored is None:
            with self._db:
                self._db.execute("INSERT INTO meta VALUES ('run_id', ?)", (self.run_id,))
                self._db.execute("INSERT INTO meta VALUES ('mode', ?)", (mode,))
        elif stored[0] != self.run_id:
            raise BoardError("database belongs to a different run")
        else:
            for _, body in self._db.execute("SELECT seq, body FROM entries ORDER BY seq"):
                self._accept(Entry.from_dict(json.loads(body)))

    # -- reads ---------------------------------------------------------------
    def size(self) -> int:
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> List[Entry]:
        return list(self._entries)

    def get(self, key: Union[int, str]) -> Entry:
        """By seq (int) or by entry id (64 hex)."""
        if isinstance(key, str):
            if key not in self._by_id:
                raise KeyError(key)
            return self._entries[self._by_id[key]]
        return self._entries[key]

    def __contains__(self, entry_id: object) -> bool:
        return entry_id in self._by_id

    def by_type(self, entry_type: str) -> List[Entry]:
        if entry_type not in TYPE_CODES:
            raise BoardError(f"unknown entry type {entry_type!r}")
        return [e for e in self._entries if e.type == entry_type]

    # -- writes --------------------------------------------------------------
    def append(
        self,
        entry_type: str,
        author: Union[str, Mapping[str, Any]],
        payload: Mapping[str, Any],
        tick: int,
        cites: Iterable[str] = (),
    ) -> Entry:
        author = _normalise_author(author)
        payload = _detach(dict(payload))
        validate_payload(entry_type, payload)
        cites_list = sorted(set(cites))
        for cite in cites_list:
            if cite not in self._by_id:
                raise ValidationError(f"cite {cite!r} is not an earlier entry on this board")
        seq = len(self._entries)
        body = {
            "run_id": self.run_id, "seq": seq, "type": entry_type, "author": author, "cites": cites_list,
            "payload": payload, "tick": tick,
            "prev_hash": self._entries[-1].id if self._entries else ZERO_HASH,
        }
        entry_id = compute_id(body)
        sig: Optional[str] = None
        if self.mode == "B":
            seed = self._seeds.get(author["agent"])
            if seed is None:
                raise BoardError(f"no signing key for author {author['agent']!r}")
            sig = sign_id(seed, entry_id)
        entry = Entry(
            run_id=self.run_id, seq=seq, type=entry_type, author=author, cites=tuple(cites_list),
            payload=payload, tick=tick, prev_hash=body["prev_hash"], id=entry_id, sig=sig,
        )
        _validate_shape(entry)
        with self._db:
            self._db.execute(
                "INSERT INTO entries VALUES (?, ?, ?)", (seq, entry.id, jcs_bytes(entry.to_dict()).decode("utf-8"))
            )
        self._register(entry)
        return entry

    def _register(self, entry: Entry) -> None:
        self._by_id[entry.id] = len(self._entries)
        self._entries.append(entry)

    def _accept(self, entry: Entry) -> None:
        """Verify ``entry`` as the next link and register it (used by import and reopen)."""
        _validate_shape(entry)
        seq = len(self._entries)
        if entry.run_id != self.run_id:
            raise BoardIntegrityError(f"entry {seq}: run_id mismatch")
        if entry.seq != seq:
            raise BoardIntegrityError(f"entry {seq}: seq is {entry.seq}")
        expected_prev = self._entries[-1].id if self._entries else ZERO_HASH
        if entry.prev_hash != expected_prev:
            raise BoardIntegrityError(f"entry {seq}: prev_hash chain broken")
        if compute_id(entry.body()) != entry.id:
            raise BoardIntegrityError(f"entry {seq}: id does not match content")
        for cite in entry.cites:
            if cite not in self._by_id:
                raise BoardIntegrityError(f"entry {seq}: cites unknown entry {cite}")
        if self.mode == "B":
            pub = self.header["pubkeys"].get(entry.author["agent"])
            if entry.sig is None or pub is None or not verify_sig(pub, entry.id, entry.sig):
                raise BoardIntegrityError(f"entry {seq}: bad or missing signature")
        elif entry.sig is not None:
            raise BoardIntegrityError(f"entry {seq}: signed entry on a mode A board")
        self._register(entry)

    # -- Merkle --------------------------------------------------------------
    def _leaves(self, size: Optional[int]) -> List[bytes]:
        size = len(self._entries) if size is None else size
        if not 0 <= size <= len(self._entries):
            raise BoardError(f"size {size} outside 0..{len(self._entries)}")
        return [bytes.fromhex(e.id) for e in self._entries[:size]]

    def root(self, size: Optional[int] = None) -> bytes:
        return merkle_root(self._leaves(size))

    def anchor_digest(self, size: Optional[int] = None) -> bytes:
        size = len(self._entries) if size is None else size
        return anchor_digest(self.run_id, size, self.root(size))

    def inclusion_proof(self, index: int, size: Optional[int] = None) -> List[bytes]:
        return inclusion_path(index, self._leaves(size))

    def consistency_proof(self, old_size: int, new_size: Optional[int] = None) -> List[bytes]:
        return consistency_path(old_size, self._leaves(new_size))

    def anchors(self) -> List[Dict[str, Any]]:
        """Anchor records for the current board, in the ``<run_id>.anchors.json`` shape."""
        return [
            {"size": s, "root": self.root(s).hex(), "digest": self.anchor_digest(s).hex()}
            for s in anchor_sizes(len(self._entries))
        ]

    # -- export / import -----------------------------------------------------
    def export_jsonl(self, path: Union[str, Path]) -> None:
        lines = [jcs_bytes(self.header).decode("utf-8")] + [e.line() for e in self._entries]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load_jsonl(
        cls, path: Union[str, Path], verify: bool = True, mode: Optional[str] = None
    ) -> "Board":
        """Rebuild a board from a JSONL export. ``verify`` checks ids, signatures and the chain."""
        raw = Path(path).read_text(encoding="utf-8").splitlines()
        if not raw:
            raise BoardIntegrityError("empty export")
        header = json.loads(raw[0])
        rows = [json.loads(line) for line in raw[1:] if line.strip()]
        if mode is None:
            signed = [r.get("sig") is not None for r in rows]
            if signed and any(signed) and not all(signed):
                raise BoardIntegrityError("export mixes signed and unsigned entries")
            mode = "B" if signed and all(signed) else "A"
        board = cls(header, mode=mode)
        for row in rows:
            entry = Entry.from_dict(row)
            if verify:
                board._accept(entry)
            else:
                board._register(entry)
            with board._db:
                board._db.execute(
                    "INSERT INTO entries VALUES (?, ?, ?)", (entry.seq, entry.id, jcs_bytes(entry.to_dict()).decode())
                )
        return board

    def close(self) -> None:
        self._db.close()


def load_knowledge_from(run_path: Union[str, Path]) -> List[Entry]:
    """Verified ``knowledge`` entries of an earlier run, for cross-run persistence (not used in v4.1)."""
    board = Board.load_jsonl(run_path, verify=True)
    try:
        return board.by_type("knowledge")
    finally:
        board.close()
