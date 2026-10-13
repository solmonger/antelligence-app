"""Read-only access to ``AnchorRegistry`` for the outside verifier (no signer, no private key)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from web3 import Web3

ZERO32 = b"\x00" * 32

_VIEWS = {
    "deployer": ("deployer()", [], ["address"]),
    "runOwner": ("runOwner(bytes32)", ["bytes32"], ["address"]),
    "lastSize": ("lastSize(bytes32)", ["bytes32"], ["uint64"]),
    "anchorCount": ("anchorCount(bytes32)", ["bytes32"], ["uint64"]),
    "rootAt": ("rootAt(bytes32,uint64)", ["bytes32", "uint64"], ["bytes32"]),
    "onceDigestOf": ("onceDigestOf(bytes32)", ["bytes32"], ["bytes32"]),
}
SELECTORS: Dict[str, bytes] = {name: bytes(Web3.keccak(text=sig))[:4] for name, (sig, _, _) in _VIEWS.items()}
EVENT_ANCHORED = bytes(Web3.keccak(text="Anchored(bytes32,uint64,bytes32,bytes32,address)"))
EVENT_ANCHORED_ONCE = bytes(Web3.keccak(text="AnchoredOnce(bytes32,bytes32)"))


class ChainReader(Protocol):
    """What the verifier needs from the chain. ``None`` from a time method means "not available"."""

    def deployer(self) -> str: ...
    def run_owner(self, run_id: bytes) -> str: ...
    def last_size(self, run_id: bytes) -> int: ...
    def anchor_count(self, run_id: bytes) -> int: ...
    def root_at(self, run_id: bytes, size: int) -> bytes: ...
    def once_digest(self, single_shot_id: bytes) -> bytes: ...
    def anchor_time(self, run_id: bytes, size: int) -> Optional[int]: ...
    def once_time(self, single_shot_id: bytes) -> Optional[int]: ...
    @property
    def calls(self) -> int: ...
    @property
    def bytes_read(self) -> int: ...


def encode_view(name: str, *args: Any) -> bytes:
    _, in_types, _ = _VIEWS[name]
    return SELECTORS[name] + (abi_encode(in_types, list(args)) if in_types else b"")


def decode_view(name: str, data: bytes):
    _, _, out_types = _VIEWS[name]
    return abi_decode(out_types, data)[0]


class Web3ChainReader:
    """``eth_call`` / ``eth_getLogs`` reader pinned to one registry address and one expected deployer."""

    def __init__(self, w3, registry_address: str, deployer: str) -> None:
        self.w3 = w3
        self.address = Web3.to_checksum_address(registry_address)
        self.pinned_deployer = Web3.to_checksum_address(deployer)
        self._calls = 0
        self._bytes = 0

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def bytes_read(self) -> int:
        return self._bytes

    def _view(self, name: str, *args: Any):
        self._calls += 1
        out = bytes(self.w3.eth.call({"to": self.address, "data": "0x" + encode_view(name, *args).hex()}))
        self._bytes += len(out)
        return decode_view(name, out)

    def deployer(self) -> str:
        return Web3.to_checksum_address(self._view("deployer"))

    def run_owner(self, run_id: bytes) -> str:
        return Web3.to_checksum_address(self._view("runOwner", run_id))

    def last_size(self, run_id: bytes) -> int:
        return int(self._view("lastSize", run_id))

    def anchor_count(self, run_id: bytes) -> int:
        return int(self._view("anchorCount", run_id))

    def root_at(self, run_id: bytes, size: int) -> bytes:
        return bytes(self._view("rootAt", run_id, size))

    def once_digest(self, single_shot_id: bytes) -> bytes:
        return bytes(self._view("onceDigestOf", single_shot_id))

    def _block_time(self, block_number: int) -> int:
        self._calls += 1
        return int(self.w3.eth.get_block(block_number)["timestamp"])

    def _logs(self, topic0: bytes, topic1: bytes) -> List[Any]:
        self._calls += 1
        logs = self.w3.eth.get_logs({
            "address": self.address, "fromBlock": 0, "toBlock": "latest",
            "topics": ["0x" + topic0.hex(), "0x" + topic1.hex()],
        })
        self._bytes += sum(len(bytes(entry["data"])) + 64 for entry in logs)
        return list(logs)

    def anchor_time(self, run_id: bytes, size: int) -> Optional[int]:
        for entry in self._logs(EVENT_ANCHORED, run_id):
            (log_size, _root, _prev) = abi_decode(["uint64", "bytes32", "bytes32"], bytes(entry["data"]))
            if log_size == size:
                return self._block_time(int(entry["blockNumber"]))
        return None

    def once_time(self, single_shot_id: bytes) -> Optional[int]:
        logs = self._logs(EVENT_ANCHORED_ONCE, single_shot_id)
        return self._block_time(int(logs[0]["blockNumber"])) if logs else None
