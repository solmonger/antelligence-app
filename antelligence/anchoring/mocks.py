"""Offline stand-ins for OTS calendars, Rekor and an ``AnchorRegistry`` node.

They enforce the same rules as the real systems where those rules matter to the tamper suite:
the registry is deployer-only, write-once and prevRoot-linked (``blockchain/contracts/AnchorRegistry.sol``);
Rekor is append-only, verifies signatures on create and hands out RFC 6962 inclusion proofs;
the OTS calendar promises first and attests to a Bitcoin block after ``mine_block``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from bitcoin.core import CBlockHeader
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.timestamp import Timestamp
from web3 import Web3

from antelligence.anchoring.ots import CommitmentNotReady
from antelligence.anchoring.readers import (
    EVENT_ANCHORED,
    EVENT_ANCHORED_ONCE,
    SELECTORS,
    ZERO32,
    decode_view,
)
from antelligence.anchoring.rekor import RekorConflict, load_public_pem
from antelligence.kernel.board import inclusion_path, merkle_root

# --------------------------------------------------------------------------------------------------
# OpenTimestamps
# --------------------------------------------------------------------------------------------------


class MockOtsTransport:
    """One fake calendar set. ``fail`` lists calendar URLs that refuse requests."""

    def __init__(self, fail: Tuple[str, ...] = (), clock: Callable[[], float] = time.time) -> None:
        self.fail = set(fail)
        self.calls = 0
        self.submitted: List[Tuple[str, bytes]] = []
        self._commitments: Dict[bytes, int] = {}     # commitment -> bitcoin height once mined
        self._salt = 0
        self.clock = clock
        self.blocks: Dict[int, Tuple[bytes, int]] = {}   # height -> (merkle root, time)
        self._next_height = 800_000

    def submit(self, calendar_url: str, digest: bytes) -> Timestamp:
        self.calls += 1
        if calendar_url in self.fail:
            raise ConnectionError(f"{calendar_url} unreachable")
        self.submitted.append((calendar_url, digest))
        self._salt += 1
        ts = Timestamp(digest)
        committed = ts.ops.add(OpAppend(self._salt.to_bytes(4, "big"))).ops.add(OpSHA256())
        self._commitments[committed.msg] = -1
        committed.attestations.add(PendingAttestation(calendar_url))
        return ts

    def get_timestamp(self, calendar_url: str, commitment: bytes) -> Timestamp:
        self.calls += 1
        height = self._commitments.get(commitment, -1)
        if height < 0:
            raise CommitmentNotReady(commitment.hex())
        ts = Timestamp(commitment)
        ts.attestations.add(BitcoinBlockHeaderAttestation(height))
        return ts

    def mine_block(self) -> int:
        """Confirm every pending commitment in one new Bitcoin block (merkle root = the commitment)."""
        pending = [c for c, h in self._commitments.items() if h < 0]
        height = self._next_height
        self._next_height += 1
        if pending:
            # one block holds many commitments; the mock keeps one root per height, so use one block per commitment
            for c in pending:
                self._commitments[c] = height
                self.blocks[height] = (c, int(self.clock()))
                height = self._next_height
                self._next_height += 1
        return height

    def header(self, height: int) -> CBlockHeader:
        root, when = self.blocks[height]
        return CBlockHeader(hashMerkleRoot=root, nTime=when)


# --------------------------------------------------------------------------------------------------
# Rekor
# --------------------------------------------------------------------------------------------------


class MockRekor:
    """Append-only transparency log with ``hashedrekord`` signature checking and RFC 6962 proofs."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self.bodies: List[bytes] = []
        self.meta: List[Dict[str, Any]] = []
        self.requests = 0
        self.bytes_out = 0

    @staticmethod
    def _uuid(body: bytes) -> str:
        return hashlib.sha256(b"\x00" + body).hexdigest()

    def create(self, proposed: Dict[str, Any]) -> Dict[str, Any]:
        self.requests += 1
        if proposed.get("kind") != "hashedrekord" or proposed.get("apiVersion") != "0.0.1":
            raise ValueError("unsupported entry type")
        spec = proposed["spec"]
        pem = base64.b64decode(spec["signature"]["publicKey"]["content"])
        digest = bytes.fromhex(spec["data"]["hash"]["value"])
        try:
            load_public_pem(pem).verify(
                base64.b64decode(spec["signature"]["content"]), digest, ec.ECDSA(utils.Prehashed(hashes.SHA256()))
            )
        except (InvalidSignature, Exception) as exc:
            raise ValueError(f"signature does not verify: {exc}") from exc
        body = json.dumps(proposed, sort_keys=True, separators=(",", ":")).encode()
        uuid = self._uuid(body)
        for i, existing in enumerate(self.bodies):
            if existing == body:
                raise RekorConflict(self._uuid(existing))
        self.bodies.append(body)
        self.meta.append({"time": int(self.clock()), "pem": pem})
        return {uuid: self._entry(len(self.bodies) - 1)}

    def _entry(self, index: int) -> Dict[str, Any]:
        leaves = self.bodies[: len(self.bodies)]
        # Rekor leaf = the canonical body; merkle code hashes with the 0x00 prefix itself
        path = inclusion_path(index, leaves)
        root = merkle_root(leaves)
        return {
            "body": base64.b64encode(self.bodies[index]).decode(),
            "integratedTime": self.meta[index]["time"],
            "logID": "mock",
            "logIndex": index,
            "verification": {
                "inclusionProof": {
                    "logIndex": index, "treeSize": len(leaves), "rootHash": root.hex(),
                    "hashes": [h.hex() for h in path],
                },
            },
        }

    def search_by_pubkey(self, pem: bytes) -> List[str]:
        self.requests += 1
        return [self._uuid(b) for b, m in zip(self.bodies, self.meta) if m["pem"] == pem]

    def get_entry(self, uuid: str) -> Dict[str, Any]:
        self.requests += 1
        for i, b in enumerate(self.bodies):
            if self._uuid(b) == uuid:
                out = {uuid: self._entry(i)}
                self.bytes_out += len(json.dumps(out))
                return out
        raise KeyError(uuid)


# --------------------------------------------------------------------------------------------------
# AnchorRegistry node
# --------------------------------------------------------------------------------------------------


class Revert(Exception):
    """A contract call reverted (write-time rejection)."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class MockRegistry:
    """Python model of ``AnchorRegistry.sol``: deployer-only, write-once, size increasing, prevRoot linked."""

    def __init__(self, deployer: str, clock: Callable[[], float] = time.time) -> None:
        self.deployer_address = Web3.to_checksum_address(deployer)
        self.clock = clock
        self.runs: Dict[bytes, Dict[str, Any]] = {}
        self.root_at: Dict[Tuple[bytes, int], bytes] = {}
        self.times: Dict[Tuple[bytes, int], int] = {}
        self.once: Dict[bytes, bytes] = {}
        self.once_times: Dict[bytes, int] = {}
        self.block = 0

    def anchor(self, sender: str, run_id: bytes, size: int, root: bytes, prev_root: bytes) -> None:
        if Web3.to_checksum_address(sender) != self.deployer_address:
            raise Revert("NotDeployer")
        run = self.runs.setdefault(run_id, {"owner": self.deployer_address, "last_size": 0, "last_root": ZERO32, "count": 0})
        if root == ZERO32:
            raise Revert("EmptyRoot")
        if size <= run["last_size"]:
            raise Revert("SizeNotIncreasing")
        if prev_root != run["last_root"]:
            raise Revert("PrevRootMismatch")
        if self.root_at.get((run_id, size), ZERO32) != ZERO32:
            raise Revert("AlreadyAnchored")
        self.block += 1
        self.root_at[(run_id, size)] = root
        self.times[(run_id, size)] = int(self.clock())
        run.update(last_size=size, last_root=root, count=run["count"] + 1)

    def anchor_once(self, sender: str, single_shot_id: bytes, digest: bytes) -> None:
        if Web3.to_checksum_address(sender) != self.deployer_address:
            raise Revert("NotDeployer")
        if digest == ZERO32:
            raise Revert("EmptyDigest")
        if self.once.get(single_shot_id, ZERO32) != ZERO32:
            raise Revert("AlreadyAnchored")
        self.block += 1
        self.once[single_shot_id] = digest
        self.once_times[single_shot_id] = int(self.clock())


class MockW3:
    """The slice of ``web3`` the :class:`Web3ChainReader` uses, backed by a :class:`MockRegistry`."""

    ADDRESS = "0x" + "a1" * 20

    def __init__(self, registry: MockRegistry) -> None:
        self.registry = registry
        self.eth = self
        self.requests = 0

    def call(self, tx: Dict[str, Any]) -> bytes:
        self.requests += 1
        data = bytes.fromhex(tx["data"][2:])
        sel, args = data[:4], data[4:]
        r = self.registry
        if sel == SELECTORS["deployer"]:
            return abi_encode(["address"], [r.deployer_address])
        if sel == SELECTORS["runOwner"]:
            (run,) = abi_decode(["bytes32"], args)
            return abi_encode(["address"], [r.runs.get(run, {}).get("owner", "0x" + "00" * 20)])
        if sel == SELECTORS["lastSize"]:
            (run,) = abi_decode(["bytes32"], args)
            return abi_encode(["uint64"], [r.runs.get(run, {}).get("last_size", 0)])
        if sel == SELECTORS["anchorCount"]:
            (run,) = abi_decode(["bytes32"], args)
            return abi_encode(["uint64"], [r.runs.get(run, {}).get("count", 0)])
        if sel == SELECTORS["rootAt"]:
            run, size = abi_decode(["bytes32", "uint64"], args)
            return abi_encode(["bytes32"], [r.root_at.get((run, size), ZERO32)])
        if sel == SELECTORS["onceDigestOf"]:
            (i,) = abi_decode(["bytes32"], args)
            return abi_encode(["bytes32"], [r.once.get(i, ZERO32)])
        raise ValueError("unknown view")

    def get_logs(self, flt: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.requests += 1
        topic0 = bytes.fromhex(flt["topics"][0][2:])
        topic1 = bytes.fromhex(flt["topics"][1][2:])
        r = self.registry
        out: List[Dict[str, Any]] = []
        if topic0 == EVENT_ANCHORED:
            for (run, size), root in r.root_at.items():
                if run == topic1:
                    out.append({"blockNumber": r.times[(run, size)], "data": abi_encode(
                        ["uint64", "bytes32", "bytes32"], [size, root, ZERO32])})
        elif topic0 == EVENT_ANCHORED_ONCE and topic1 in r.once:
            out.append({"blockNumber": r.once_times[topic1], "data": abi_encode(["bytes32"], [r.once[topic1]])})
        return out

    def get_block(self, number: int) -> Dict[str, Any]:
        self.requests += 1
        return {"timestamp": number}   # the mock uses block number == unix time of the anchor


# --------------------------------------------------------------------------------------------------
# Synthetic signed runs (tests and the tamper suite; real runs come from the harness)
# --------------------------------------------------------------------------------------------------

def synthetic_run(n_entries: int, seed: int = 0, split: str = "eval"):
    """A signed (mode B) board of ``n_entries`` valid entries. Returns ``(board, run_keys, anchor_key)``."""
    import hashlib as _h

    from antelligence.anchoring.rekor import AnchorKey
    from antelligence.kernel.board import Board, generate_run_keys

    keys = generate_run_keys()
    akey = AnchorKey()
    header = {
        "study": "v4.1", "arm": "sw-b", "case": f"SYN_{seed:05d}", "seed": seed, "split": split,
        "world_cfg_sha256": _h.sha256(b"world").hexdigest(), "harness_cfg_sha256": _h.sha256(b"harness").hexdigest(),
        "pubkeys": keys.pubkeys, "anchor_pubkey": akey.fingerprint, "code_sha": "0" * 40,
    }
    board = Board(header, mode="B", keys=keys.seeds)
    for i in range(n_entries):
        board.append(*_entry_args(i), tick=i // 4)
    return board, keys, akey


def _entry_args(i: int, variant: int = 0):
    squad = i % 8
    kind = i % 4
    if i == 0:
        return "task", "queen", {"objective": "find tumour core", "squads": list(range(8)), "until_tick": 200}
    if kind == 1:
        return "evidence", f"squad-{squad}", {
            "squad": squad, "pos_vox": [i % 64, (i * 3) % 64, (i * 5) % 64], "look_p_ppm": 400_000 + (i * 7919 + variant) % 500_000,
            "p_tumour_ppm": 300_000 + (i * 104729) % 600_000, "dose_ug": i * 10, "released_ug": i * 5,
        }
    if kind == 2:
        return "decision", f"squad-{squad}", {
            "squad": squad, "round": i // 8, "target": "ABCDEFSX"[i % 8], "release": i % 2, "amount_level": 1 + i % 5,
            "p_target_ppm": 500_000, "p_release_ppm": 600_000, "p_amount_ppm": 400_000, "backend": "mock",
            "deferred": 0, "router_pick": "mock", "brief_ids": [],
        }
    if kind == 3:
        return "knowledge", "perception", {
            "region_id": i, "centroid_dmm": [i * 20, 40, 60], "cls": ("WT", "TC", "ET")[i % 3], "p_tumour_ppm": 700_000,
            "volume_mm3": 1000 + i, "unc_ppm": 100_000, "source": "perception", "prov": "mock@r1",
        }
    return "alarm", "harness", {"kind": "other", "value_ppm": i, "note": f"tick note {i}"}
