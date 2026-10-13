"""In-memory EVM-ish provider for the v4.1 chain-client tests. No network, no real keys."""

from __future__ import annotations

import rlp
from eth_abi import decode as abi_decode
from eth_account import Account
from web3 import Web3
from web3.exceptions import TransactionNotFound

from chain.v4_1 import ABIS


def _sel(sig: str) -> bytes:
    return bytes(Web3.keccak(text=sig))[:4]


SEL = {
    "anchor": _sel("anchor(bytes32,uint64,bytes32,bytes32)"),
    "anchorOnce": _sel("anchorOnce(bytes32,bytes32)"),
    "postBatch": _sel("postBatch(bytes32,uint64,bytes32,bytes[])"),
    "report": _sel("report(bytes32,bytes32,bytes32,bool)"),
    "reported": _sel("reported(bytes32,bytes32,bytes32)"),
    "rootAt": _sel("rootAt(bytes32,uint64)"),
    "onceDigestOf": _sel("onceDigestOf(bytes32)"),
    "nextSeq": _sel("nextSeq(bytes32)"),
    "reputationOf": _sel("reputationOf(bytes32)"),
    "deployer": _sel("deployer()"),
}
assert set(ABIS) == {"AnchorRegistry", "BoardLog", "ReputationRegistry"}


class FakeChain:
    """Implements the slice of ``web3.eth`` the client uses, with a manually driven mempool."""

    def __init__(self, chain_id: int = 300, contract_addrs=()):
        self.chain_id = chain_id
        self.gas_price = 1_000
        self.code = {a.lower(): b"\x60\x80" for a in contract_addrs}
        self.mined_nonce: dict[str, int] = {}
        self.mempool: dict[str, dict] = {}      # tx hash -> {nonce, gas_price, sender, to, data}
        self.receipts: dict[str, dict] = {}
        self.drop_next = 0                      # silently drop this many broadcasts
        self.sent_log: list[dict] = []          # every accepted broadcast, in order
        self.max_mempool = 0
        self.block = 0
        # contract state (effects of mined txs)
        self.root_at: dict[tuple, bytes] = {}
        self.last_root: dict[bytes, bytes] = {}
        self.once: dict[bytes, bytes] = {}
        self.next_seq: dict[bytes, int] = {}
        self.rep: dict[bytes, int] = {}
        self.claims: set[tuple] = set()
        self.txs: dict[str, dict] = {}         # every tx the node accepted (mempool or mined)
        self.hide_receipts = False             # simulate a receipt index that lags behind blocks
        self.get_tx_error: Exception | None = None
        self.contract_deployer: str | None = None   # what ``deployer()`` returns for every contract (None = zero address)
        self.deployer_by_contract: dict[str, str] = {}   # lowercase address -> deployer(), overriding ``contract_deployer``
        self.eth = self

    # ---- web3.eth surface ---------------------------------------------------------------------
    def get_code(self, addr):
        return self.code.get(str(addr).lower(), b"")

    def get_transaction_count(self, addr, block="latest"):
        base = self.mined_nonce.get(str(addr).lower(), 0)
        if block == "latest":
            return base
        n = base
        pend = {t["nonce"] for t in self.mempool.values() if t["sender"] == str(addr).lower()}
        while n in pend:
            n += 1
        return n

    def get_transaction(self, h):
        if self.get_tx_error:
            raise self.get_tx_error
        if h in self.txs:
            return self.txs[h]
        raise TransactionNotFound(h)

    def get_transaction_receipt(self, h):
        if h in self.receipts and not self.hide_receipts:
            return self.receipts[h]
        raise TransactionNotFound(h)

    def estimate_gas(self, tx):
        ok, why = self._validate(tx["to"], bytes.fromhex(tx["data"][2:]))
        if not ok:
            raise ValueError(f"execution reverted: {why}")
        return 100_000

    def call(self, tx):
        data = bytes.fromhex(tx["data"][2:])
        s, args = data[:4], data[4:]
        if s == SEL["deployer"]:
            who = self.deployer_by_contract.get(str(tx["to"]).lower(), self.contract_deployer) or "0x" + "00" * 20
            return bytes(12) + bytes.fromhex(who[2:])
        if s == SEL["rootAt"]:
            run, size = abi_decode(["bytes32", "uint64"], args)
            return self.root_at.get((run, size), b"\x00" * 32)
        if s == SEL["onceDigestOf"]:
            (i,) = abi_decode(["bytes32"], args)
            return self.once.get(i, b"\x00" * 32)
        if s == SEL["nextSeq"]:
            (run,) = abi_decode(["bytes32"], args)
            return self.next_seq.get(run, 0).to_bytes(32, "big")
        if s == SEL["reported"]:
            run, a, cl = abi_decode(["bytes32", "bytes32", "bytes32"], args)
            return (1 if (run, a, cl) in self.claims else 0).to_bytes(32, "big")
        if s == SEL["reputationOf"]:
            (a,) = abi_decode(["bytes32"], args)
            return self.rep.get(a, 1_000_000).to_bytes(32, "big")
        raise ValueError("unknown view")

    def send_raw_transaction(self, raw):
        raw = bytes(raw)
        f = rlp.decode(raw)
        nonce, gas_price = int.from_bytes(f[0], "big"), int.from_bytes(f[1], "big")
        sender = Account.recover_transaction(raw).lower()
        h = Web3.keccak(raw).hex()
        h = h if h.startswith("0x") else "0x" + h
        if nonce < self.mined_nonce.get(sender, 0):
            raise ValueError("nonce too low")
        if h in self.mempool or h in self.receipts:
            raise ValueError("already known")
        for oh, o in list(self.mempool.items()):
            if o["sender"] == sender and o["nonce"] == nonce:
                if gas_price * 100 < o["gas_price"] * 110:
                    raise ValueError("replacement transaction underpriced")
                del self.mempool[oh]
        rec = {"nonce": nonce, "gas_price": gas_price, "sender": sender,
               "to": Web3.to_checksum_address(f[3]), "data": "0x" + f[5].hex(), "hash": h}
        self.sent_log.append(rec)
        if self.drop_next > 0:
            self.drop_next -= 1
            return h  # accepted by the RPC, then lost
        self.mempool[h] = rec
        self.txs[h] = rec
        self.max_mempool = max(self.max_mempool, len(self.mempool))
        return h

    # ---- mining -------------------------------------------------------------------------------
    def _validate(self, to, data):
        s, args = data[:4], data[4:]
        if s == SEL["anchor"]:
            run, size, root, prev = abi_decode(["bytes32", "uint64", "bytes32", "bytes32"], args)
            if prev != self.last_root.get(run, b"\x00" * 32):
                return False, "PrevRootMismatch"
        if s == SEL["anchorOnce"]:
            i, _ = abi_decode(["bytes32", "bytes32"], args)
            if i in self.once:
                return False, "AlreadyAnchored"
        if s == SEL["report"]:
            run, a, cl, _ = abi_decode(["bytes32", "bytes32", "bytes32", "bool"], args)
            if (run, a, cl) in self.claims:
                return False, "DuplicateReport"
        return True, ""

    def _apply(self, data):
        s, args = data[:4], data[4:]
        if s == SEL["anchor"]:
            run, size, root, _ = abi_decode(["bytes32", "uint64", "bytes32", "bytes32"], args)
            self.root_at[(run, size)] = root
            self.last_root[run] = root
        elif s == SEL["anchorOnce"]:
            i, d = abi_decode(["bytes32", "bytes32"], args)
            self.once[i] = d
        elif s == SEL["postBatch"]:
            run, first, _, entries = abi_decode(["bytes32", "uint64", "bytes32", "bytes[]"], args)
            self.next_seq[run] = first + len(entries)
        elif s == SEL["report"]:
            run, a, cl, contradicted = abi_decode(["bytes32", "bytes32", "bytes32", "bool"], args)
            self.claims.add((run, a, cl))
            r = self.rep.get(a, 1_000_000)
            self.rep[a] = r // 2 if contradicted else min(1_000_000, r + 100_000)

    def mine(self, max_txs: int | None = None) -> int:
        """Mine mempool txs whose nonce is next for their sender, lowest nonce first."""
        mined = 0
        progress = True
        while progress and (max_txs is None or mined < max_txs):
            progress = False
            for h, t in sorted(self.mempool.items(), key=lambda kv: kv[1]["nonce"]):
                if t["nonce"] != self.mined_nonce.get(t["sender"], 0):
                    continue
                self.block += 1
                ok, _ = self._validate(t["to"], bytes.fromhex(t["data"][2:]))
                if ok:
                    self._apply(bytes.fromhex(t["data"][2:]))
                self.receipts[h] = {"status": 1 if ok else 0, "blockNumber": self.block, "transactionHash": h}
                self.mined_nonce[t["sender"]] = t["nonce"] + 1
                del self.mempool[h]
                mined += 1
                progress = True
                break
        return mined
