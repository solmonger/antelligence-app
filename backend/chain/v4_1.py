"""Antelligence v4.1 chain client: AnchorRegistry, BoardLog, ReputationRegistry.

Mode-C anchoring must never block a run, so every write is first *queued* in a SQLite file (default
``ledger/v4_1_chain_queue.sqlite``, gitignored) and a separate :meth:`ChainClientV41.pump` call moves
the queue forward:

* one nonce manager per signer, with pipelined nonces (at most ``max_in_flight`` = 8 unconfirmed);
* a transaction that never shows up (dropped from the mempool) is **replaced at the same nonce** with a
  higher gas price, never given a fresh nonce. A SENT job is never requeued or resent under another
  nonce: if the nonce was consumed and none of our hashes is known, the job becomes ``failed_review``;
* receipts are polled; a job is **confirmed** only when the receipt has status 1 *and* the effect is
  readable through ``eth_call`` (``rootAt`` / ``onceDigestOf`` / ``nextSeq`` ...);
* the queue survives a restart: sent jobs keep their nonce and tx hashes.

Safety, reusing :mod:`chain.config` and :mod:`chain.submit`: before signing, the RPC must serve the
selected testnet (chain id 300 by default, :func:`chain.config.assert_rpc_matches_network`), every
contract must have code at its address, and the signer must be the deployer recorded in the
deployments file. The key is read only through :func:`chain.config.get_private_key` (the W1 mechanism,
environment-driven); tests pass a throw-away local key and a fake provider.

INTERFACES (docs/v4.1/INTERFACES.md) helpers: :func:`anchor_digest`, :func:`encode_entry_compact`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from eth_abi import encode as abi_encode
from eth_account import Account
from web3 import Web3

from chain.config import (
    REPO_ROOT,
    ChainConfigError,
    assert_rpc_matches_network,
    get_private_key,
    load_deployment,
)

# --------------------------------------------------------------------------------------------------
# Reputation rule (mirrored exactly by ReputationRegistry.update in Solidity)
# --------------------------------------------------------------------------------------------------

REP_START_PPM = 1_000_000
REP_STEP_UP_PPM = 100_000
REP_EXCLUDE_BELOW_PPM = 300_000


def reputation_update(r_ppm: int, contradicted: bool) -> int:
    """Integer-ppm update. Contradicted: ``r // 2``. Otherwise ``min(1_000_000, r + 100_000)``."""
    if r_ppm < 0 or r_ppm > REP_START_PPM:
        raise ValueError(f"reputation must be in [0, {REP_START_PPM}] ppm, got {r_ppm}")
    if contradicted:
        return r_ppm // 2
    return min(REP_START_PPM, r_ppm + REP_STEP_UP_PPM)


def is_excluded(r_ppm: int) -> bool:
    """Excluded from briefs when the score is strictly below 300,000 ppm (0.3)."""
    return r_ppm < REP_EXCLUDE_BELOW_PPM


# --------------------------------------------------------------------------------------------------
# INTERFACES helpers
# --------------------------------------------------------------------------------------------------

ZERO32 = b"\x00" * 32
TYPE_CODES = {"task": 0, "knowledge": 1, "evidence": 2, "route": 3, "decision": 4, "edit": 5, "alarm": 6}


def _b32(value: Any, name: str) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        text = str(value)
        raw = bytes.fromhex(text[2:] if text.startswith("0x") else text)
    if len(raw) != 32:
        raise ValueError(f"{name} must be 32 bytes, got {len(raw)}")
    return raw


def anchor_digest(run_id: Any, size: int, root: Any) -> bytes:
    """INTERFACES section 3: sha256(runId(32) || uint64_be(size) || root(32))."""
    import hashlib

    return hashlib.sha256(_b32(run_id, "run_id") + int(size).to_bytes(8, "big") + _b32(root, "root")).digest()


def encode_entry_compact(seq: int, type_name: str, entry_id: Any, prev_hash: Any, payload_ints: Sequence[int]) -> bytes:
    """INTERFACES section 4 compact entry for ``BoardLog.postBatch``.

    ``abi.encode(uint64 seq, uint8 typeCode, bytes32 id, bytes32 prevHash, bytes payloadCompact)`` where
    ``payloadCompact = abi.encode(uint256 ... )`` of the payload's integer fields in table order. (The
    spec says "abi.encode of the integer fields" without naming a width; uint256 is used, see report.)
    """
    payload = abi_encode(["uint256"] * len(payload_ints), [int(v) for v in payload_ints])
    return abi_encode(
        ["uint64", "uint8", "bytes32", "bytes32", "bytes"],
        [int(seq), TYPE_CODES[type_name], _b32(entry_id, "id"), _b32(prev_hash, "prev_hash"), payload],
    )


# --------------------------------------------------------------------------------------------------
# ABIs (only what the client calls; kept inline because artifacts are not committed)
# --------------------------------------------------------------------------------------------------


def _fn(name: str, inputs: List[tuple], outputs: List[str] = (), mutability: str = "nonpayable") -> dict:
    return {
        "type": "function", "name": name, "stateMutability": mutability,
        "inputs": [{"name": n, "type": t} for n, t in inputs],
        "outputs": [{"name": "", "type": t} for t in outputs],
    }


ABIS: Dict[str, list] = {
    "AnchorRegistry": [
        _fn("anchor", [("runId", "bytes32"), ("size", "uint64"), ("root", "bytes32"), ("prevRoot", "bytes32")]),
        _fn("anchorOnce", [("id", "bytes32"), ("digest", "bytes32")]),
        _fn("rootAt", [("runId", "bytes32"), ("size", "uint64")], ["bytes32"], "view"),
        _fn("onceDigestOf", [("id", "bytes32")], ["bytes32"], "view"),
        _fn("digestOf", [("runId", "bytes32"), ("size", "uint64")], ["bytes32"], "view"),
        _fn("anchorCount", [("runId", "bytes32")], ["uint64"], "view"),
    ],
    "BoardLog": [
        _fn("postBatch", [("runId", "bytes32"), ("firstSeq", "uint64"), ("prevHash", "bytes32"), ("entries", "bytes[]")]),
        _fn("nextSeq", [("runId", "bytes32")], ["uint64"], "view"),
        _fn("lastId", [("runId", "bytes32")], ["bytes32"], "view"),
    ],
    "ReputationRegistry": [
        _fn("report", [("runId", "bytes32"), ("author", "bytes32"), ("claimId", "bytes32"), ("contradicted", "bool")], ["uint32"]),
        _fn("reported", [("runId", "bytes32"), ("author", "bytes32"), ("claimId", "bytes32")], ["bool"], "view"),
        _fn("reputationOf", [("author", "bytes32")], ["uint32"], "view"),
    ],
}

_OFFLINE_W3 = Web3()


def _calldata(contract: str, fn: str, args: list) -> str:
    c = _OFFLINE_W3.eth.contract(abi=ABIS[contract])
    return c.encode_abi(fn, args=args)


# --------------------------------------------------------------------------------------------------
# Signer
# --------------------------------------------------------------------------------------------------


class LocalSigner:
    """Signs legacy transactions in-process. The key never appears in repr or logs."""

    def __init__(self, private_key: str):
        self._account = Account.from_key(private_key)
        self.address: str = self._account.address

    def sign(self, tx: dict) -> bytes:
        return bytes(self._account.sign_transaction(tx).raw_transaction)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"LocalSigner({self.address})"


def signer_from_config() -> LocalSigner:
    """The production signer: the key comes from :func:`chain.config.get_private_key` (W1 mechanism)."""
    key = get_private_key()
    if not key:
        raise ChainConfigError("no signing key configured (PRIVATE_KEY / ANTELLIGENCE_DEPLOYER_PRIVATE_KEY)")
    return LocalSigner(key)


# --------------------------------------------------------------------------------------------------
# Persistent queue
# --------------------------------------------------------------------------------------------------

PENDING, SENT, CONFIRMED, FAILED = "pending", "sent", "confirmed", "failed"
# The nonce was consumed on chain and none of our hashes is known: a human must look. Never auto-resent.
FAILED_REVIEW = "failed_review"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedupe_key TEXT UNIQUE,
  kind TEXT NOT NULL,
  to_addr TEXT NOT NULL,
  calldata TEXT NOT NULL,
  check_data TEXT,
  check_expect TEXT,
  check_cmp TEXT NOT NULL DEFAULT 'any',
  status TEXT NOT NULL DEFAULT 'pending',
  nonce INTEGER,
  gas_limit INTEGER,
  gas_price INTEGER,
  tx_hashes TEXT NOT NULL DEFAULT '[]',
  sent_at REAL,
  attempts INTEGER NOT NULL DEFAULT 0,
  block_number INTEGER,
  error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, id);
"""


def default_queue_path() -> Path:
    return REPO_ROOT / "ledger" / "v4_1_chain_queue.sqlite"


class TxQueue:
    """Small SQLite-backed job queue. Enqueueing is one INSERT, so it never waits on the chain."""

    def __init__(self, path: Optional[Path] = None, clock: Callable[[], float] = time.time):
        self.path = Path(path) if path else default_queue_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def enqueue(self, *, kind: str, to_addr: str, calldata: str, dedupe_key: Optional[str] = None,
                check_data: Optional[str] = None, check_expect: Optional[str] = None, check_cmp: str = "any") -> int:
        """Insert a job; a repeated ``dedupe_key`` returns the existing job id (idempotent)."""
        now = self._clock()
        with self._lock:
            if dedupe_key:
                row = self._db.execute("SELECT id FROM jobs WHERE dedupe_key=?", (dedupe_key,)).fetchone()
                if row:
                    return int(row["id"])
            cur = self._db.execute(
                "INSERT INTO jobs(dedupe_key,kind,to_addr,calldata,check_data,check_expect,check_cmp,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (dedupe_key, kind, to_addr, calldata, check_data, check_expect, check_cmp, now, now),
            )
            return int(cur.lastrowid)

    def get(self, job_id: int) -> Optional[dict]:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def by_status(self, status: str) -> List[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs WHERE status=? ORDER BY id", (status,)).fetchall()
        return [self._row(r) for r in rows]

    def all(self) -> List[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs ORDER BY id").fetchall()
        return [self._row(r) for r in rows]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
        out = {PENDING: 0, SENT: 0, CONFIRMED: 0, FAILED: 0, FAILED_REVIEW: 0}
        out.update({r["status"]: int(r["n"]) for r in rows})
        return out

    def max_nonce(self) -> Optional[int]:
        with self._lock:
            row = self._db.execute("SELECT MAX(nonce) m FROM jobs WHERE nonce IS NOT NULL").fetchone()
        return None if row["m"] is None else int(row["m"])

    def update(self, job_id: int, **fields: Any) -> None:
        if "tx_hashes" in fields and not isinstance(fields["tx_hashes"], str):
            fields["tx_hashes"] = json.dumps(list(fields["tx_hashes"]))
        fields["updated_at"] = self._clock()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["tx_hashes"] = json.loads(d["tx_hashes"] or "[]")
        return d


# --------------------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------------------


def _is_tx_not_found(exc: Exception) -> bool:
    return type(exc).__name__ in ("TransactionNotFound", "TransactionIndexingInProgress") or "not found" in str(exc).lower()


@dataclass
class PumpStats:
    sent: int = 0
    replaced: int = 0
    confirmed: int = 0
    failed: int = 0
    deferred: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class ChainClientV41:
    """Queue-first client for the three v4.1 contracts.

    ``contracts`` maps ``AnchorRegistry`` / ``BoardLog`` / ``ReputationRegistry`` to addresses; if omitted
    they come from the ``v4_1`` key of the selected network's deployments file.
    """

    MAX_ATTEMPTS = 8

    def __init__(self, w3, signer, queue: TxQueue, contracts: Optional[Dict[str, str]] = None, *,
                 expected_deployer: Optional[str] = None, max_in_flight: int = 8, replace_after_s: float = 60.0,
                 gas_bump_pct: int = 15, gas_limit_pad_pct: int = 20, clock: Callable[[], float] = time.time):
        if not 1 <= max_in_flight <= 8:
            raise ValueError("max_in_flight must be in 1..8")
        record = (load_deployment() or {}).get("v4_1") or {}
        self.w3 = w3
        self.signer = signer
        self.queue = queue
        self.contracts = dict(contracts) if contracts is not None else {
            c["name"]: c["address"] for c in record.get("contracts", [])
        }
        self.expected_deployer = expected_deployer if expected_deployer is not None else record.get("deployer", "")
        self.max_in_flight = max_in_flight
        self.replace_after_s = replace_after_s
        self.gas_bump_pct = gas_bump_pct
        self.gas_limit_pad_pct = gas_limit_pad_pct
        self._clock = clock
        self._next_nonce: Optional[int] = None

    # ---- safety -------------------------------------------------------------------------------

    def preflight(self) -> int:
        """Refuse to sign unless: RPC serves the selected testnet, signer is the deployer, contracts have code."""
        chain_id = assert_rpc_matches_network(self.w3)
        if not self.expected_deployer:
            raise ChainConfigError("no deployer address recorded for v4_1; refusing to sign")
        if self.signer.address.lower() != str(self.expected_deployer).lower():
            raise ChainConfigError(
                f"signer {self.signer.address} is not the recorded deployer {self.expected_deployer}; refusing to sign"
            )
        for name in ABIS:
            addr = self.contracts.get(name)
            if not addr:
                raise ChainConfigError(f"{name} has no address (v4_1 not deployed on this network); refusing to sign")
            if not self.w3.eth.get_code(Web3.to_checksum_address(addr)):
                raise ChainConfigError(f"no contract code for {name} at {addr}; refusing to sign")
        return chain_id

    # ---- enqueue (never touches the network) --------------------------------------------------

    def _addr(self, name: str) -> str:
        addr = self.contracts.get(name)
        if not addr:
            raise ChainConfigError(f"{name} has no address configured")
        return Web3.to_checksum_address(addr)

    def enqueue_anchor(self, run_id: Any, size: int, root: Any, prev_root: Any = ZERO32) -> int:
        rid, rt, pr = _b32(run_id, "run_id"), _b32(root, "root"), _b32(prev_root, "prev_root")
        return self.queue.enqueue(
            kind="anchor", to_addr=self._addr("AnchorRegistry"),
            calldata=_calldata("AnchorRegistry", "anchor", [rid, int(size), rt, pr]),
            dedupe_key=f"anchor:{rid.hex()}:{int(size)}",
            check_data=_calldata("AnchorRegistry", "rootAt", [rid, int(size)]), check_expect=rt.hex(), check_cmp="eq",
        )

    def enqueue_anchor_once(self, single_shot_id: Any, digest: Any) -> int:
        i, d = _b32(single_shot_id, "id"), _b32(digest, "digest")
        return self.queue.enqueue(
            kind="anchor_once", to_addr=self._addr("AnchorRegistry"),
            calldata=_calldata("AnchorRegistry", "anchorOnce", [i, d]), dedupe_key=f"once:{i.hex()}",
            check_data=_calldata("AnchorRegistry", "onceDigestOf", [i]), check_expect=d.hex(), check_cmp="eq",
        )

    def enqueue_board_batch(self, run_id: Any, first_seq: int, prev_hash: Any, entries: Sequence[bytes]) -> int:
        rid, ph = _b32(run_id, "run_id"), _b32(prev_hash, "prev_hash")
        return self.queue.enqueue(
            kind="board_batch", to_addr=self._addr("BoardLog"),
            calldata=_calldata("BoardLog", "postBatch", [rid, int(first_seq), ph, [bytes(e) for e in entries]]),
            dedupe_key=f"batch:{rid.hex()}:{int(first_seq)}",
            check_data=_calldata("BoardLog", "nextSeq", [rid]),
            check_expect=hex(int(first_seq) + len(entries)), check_cmp="ge",
        )

    def enqueue_report(self, run_id: Any, author: Any, claim_id: Any, contradicted: bool) -> int:
        """Queue ReputationRegistry.report. ``claim_id`` is mandatory: it is the dedupe key
        (runId:author:claimId) here and the on-chain idempotency key there."""
        rid, au, cl = _b32(run_id, "run_id"), _b32(author, "author"), _b32(claim_id, "claim_id")
        return self.queue.enqueue(
            kind="report", to_addr=self._addr("ReputationRegistry"),
            calldata=_calldata("ReputationRegistry", "report", [rid, au, cl, bool(contradicted)]),
            dedupe_key=f"report:{rid.hex()}:{au.hex()}:{cl.hex()}",
            check_data=_calldata("ReputationRegistry", "reported", [rid, au, cl]), check_expect="0x1", check_cmp="ge",
        )

    # ---- pump ---------------------------------------------------------------------------------

    def pump(self) -> PumpStats:
        """One pass: poll receipts, replace dropped txs at the same nonce, send pending jobs."""
        stats = PumpStats()
        self.preflight()
        if self._next_nonce is None:
            self._sync_nonce()
        self._poll_sent(stats)
        self._replace_dropped(stats)
        self._send_pending(stats)
        return stats

    def drain(self, timeout_s: float = 600.0, poll_s: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> dict:
        """Pump until nothing is pending or sent, or the timeout passes. For batch/finalisation only."""
        deadline = self._clock() + timeout_s
        while True:
            self.pump()
            counts = self.queue.counts()
            if counts[PENDING] == 0 and counts[SENT] == 0:
                return counts
            if self._clock() >= deadline:
                return counts
            sleep(poll_s)

    # ---- internals ----------------------------------------------------------------------------

    def _sync_nonce(self) -> None:
        chain_pending = int(self.w3.eth.get_transaction_count(self.signer.address, "pending"))
        persisted = self.queue.max_nonce()
        self._next_nonce = max(chain_pending, (persisted + 1) if persisted is not None else 0)

    def _receipt(self, tx_hash: str):
        try:
            return self.w3.eth.get_transaction_receipt(tx_hash)
        except Exception as exc:  # web3 raises TransactionNotFound for unknown hashes
            if _is_tx_not_found(exc):
                return None
            raise

    def _readable(self, job: dict) -> bool:
        if not job.get("check_data"):
            return True
        try:
            out = bytes(self.w3.eth.call({"to": job["to_addr"], "data": job["check_data"]}))
        except Exception:
            return False
        cmp_ = job["check_cmp"]
        if cmp_ == "any":
            return True
        expect = job["check_expect"] or ""
        if cmp_ == "eq":
            return out.hex().removeprefix("0x") == expect.removeprefix("0x")
        if cmp_ == "ge":
            return int.from_bytes(out, "big") >= int(expect, 16)
        raise ValueError(f"unknown check_cmp {cmp_!r}")

    def _poll_sent(self, stats: PumpStats) -> None:
        for job in self.queue.by_status(SENT):
            for h in job["tx_hashes"]:
                rc = self._receipt(h)
                if rc is None:
                    continue
                if int(rc["status"]) != 1:
                    self.queue.update(job["id"], status=FAILED, error=f"reverted in tx {h}", block_number=int(rc["blockNumber"]))
                    stats.failed += 1
                elif self._readable(job):
                    self.queue.update(job["id"], status=CONFIRMED, error=None, block_number=int(rc["blockNumber"]))
                    stats.confirmed += 1
                else:
                    self.queue.update(job["id"], error="receipt ok, effect not yet readable via eth_call")
                break

    def _build(self, job: dict, nonce: int, gas_limit: int, gas_price: int) -> dict:
        return {
            "to": job["to_addr"], "data": job["calldata"], "value": 0, "nonce": nonce,
            "gas": gas_limit, "gasPrice": gas_price, "chainId": int(self.w3.eth.chain_id),
        }

    def _broadcast(self, job_id: int, raw: bytes) -> Optional[str]:
        """Returns None on success/known, 'nonce_low' when the nonce is already used, or an error string."""
        try:
            self.w3.eth.send_raw_transaction(raw)
            return None
        except Exception as exc:
            msg = str(exc).lower()
            if "already known" in msg or "known transaction" in msg:
                return None
            if "nonce too low" in msg:
                return "nonce_low"
            return str(exc)

    def _lookup(self, tx_hash: str) -> str:
        """'mined' (receipt), 'known' (node has the tx), 'absent' (node says not found), 'unresolved' (error)."""
        try:
            if self._receipt(tx_hash) is not None:
                return "mined"
        except Exception:
            return "unresolved"
        try:
            self.w3.eth.get_transaction(tx_hash)
            return "known"
        except Exception as exc:
            return "absent" if _is_tx_not_found(exc) else "unresolved"

    def _replace_dropped(self, stats: PumpStats) -> None:
        """Same-nonce gas-bump replacement only. A SENT job is never requeued under a fresh nonce."""
        now = self._clock()
        for job in self.queue.by_status(SENT):
            if job["sent_at"] is None or now - job["sent_at"] < self.replace_after_s:
                continue
            states = [self._lookup(h) for h in job["tx_hashes"]]
            if "mined" in states or "unresolved" in states:
                continue  # mined (next poll confirms, even if the receipt index lags) or unknown: keep SENT
            latest = int(self.w3.eth.get_transaction_count(self.signer.address, "latest"))
            if latest > job["nonce"]:
                if "known" in states:
                    continue  # node still knows one of our txs: wait for its receipt
                self.queue.update(job["id"], status=FAILED_REVIEW,
                                  error=f"nonce {job['nonce']} consumed on chain, none of our hashes known; manual review")
                stats.failed += 1
                continue
            if job["attempts"] >= self.MAX_ATTEMPTS:
                self.queue.update(job["id"], status=FAILED, error=f"dropped after {job['attempts']} attempts")
                stats.failed += 1
                continue
            old = int(job["gas_price"])
            bumped = max(int(self.w3.eth.gas_price), old * (100 + self.gas_bump_pct) // 100) + 1
            raw = self.signer.sign(self._build(job, job["nonce"], int(job["gas_limit"]), bumped))
            tx_hash = Web3.keccak(raw).hex()
            tx_hash = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash
            self.queue.update(job["id"], gas_price=bumped, tx_hashes=job["tx_hashes"] + [tx_hash],
                              sent_at=now, attempts=job["attempts"] + 1)
            err = self._broadcast(job["id"], raw)
            if err and err != "nonce_low":
                self.queue.update(job["id"], error=err)
            stats.replaced += 1

    def _send_pending(self, stats: PumpStats) -> None:
        for job in self.queue.by_status(PENDING):
            in_flight = len(self.queue.by_status(SENT))
            if in_flight >= self.max_in_flight:
                stats.deferred += 1
                break
            if self._next_nonce is None:
                self._sync_nonce()
            gas_price = int(self.w3.eth.gas_price)
            try:
                est = int(self.w3.eth.estimate_gas({
                    "from": self.signer.address, "to": job["to_addr"], "data": job["calldata"], "value": 0,
                }))
            except Exception as exc:
                if in_flight:
                    # Later jobs can depend on earlier in-flight ones (anchor prevRoot chain), and
                    # estimate_gas cannot see the mempool. Wait for the pipeline to settle, then retry.
                    self.queue.update(job["id"], error=f"estimate deferred behind in-flight txs: {exc}")
                    stats.deferred += 1
                    break
                self.queue.update(job["id"], status=FAILED, error=f"would revert: {exc}")
                stats.failed += 1
                continue
            gas_limit = est * (100 + self.gas_limit_pad_pct) // 100
            nonce = self._next_nonce
            raw = self.signer.sign(self._build(job, nonce, gas_limit, gas_price))
            tx_hash = Web3.keccak(raw).hex()
            tx_hash = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash
            # Persist before broadcasting: a crash between the two is healed by same-nonce replacement.
            self.queue.update(job["id"], status=SENT, nonce=nonce, gas_limit=gas_limit, gas_price=gas_price,
                              tx_hashes=[tx_hash], sent_at=self._clock(), attempts=1, error=None)
            self._next_nonce = nonce + 1
            err = self._broadcast(job["id"], raw)
            if err == "nonce_low":
                # Keep the job SENT with its nonce and hash. It is never moved to a fresh nonce: the poll
                # confirms it if it landed, _replace_dropped sends it to FAILED_REVIEW if the nonce is foreign.
                self.queue.update(job["id"], error="nonce too low; resyncing, awaiting resolution")
                self._next_nonce = None
                stats.sent += 1
                break
            if err:
                self.queue.update(job["id"], error=err)
            stats.sent += 1
