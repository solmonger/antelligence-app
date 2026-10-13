"""Tamper suite engine (goal 4.4): attacks T1-T4 against verifier conditions B, OTS, R, C.

Threat model: the operator holds every key and can stamp, log and post anything it likes, at any time.
It cannot rewrite what OTS (Bitcoin), Rekor (transparency log) or the chain already hold.

* T1 rewrite   - change one entry, re-sign it, rebuild the hash chain, re-anchor wherever the system allows.
* T2 truncate  - drop the last k entries (k = 1, 10, 60) and re-anchor the shorter history.
* T3 equivocate- two different histories shown to two independent verifiers.
* T4 backdate  - swap in a different preregistration document.

What each verifier condition is handed: the exported board (attacker-controlled), the ``.ots`` files for
run anchors (attacker-controlled: nothing public says which stamps belong to a run), the public Rekor log
and the chain (shared, append-only), and for T4 the original preregistration ``.ots`` that was pushed to
the public PR when the preregistration was locked.

Everything here runs on mocks (:mod:`antelligence.anchoring.mocks`) or, with ``Web3Registry``, on a LOCAL
hardhat/anvil node whose chain id is 31337/1337. It never touches a real network or a private key.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from web3 import Web3

from antelligence.anchoring import mocks
from antelligence.anchoring.anchorer import Anchorer, SingleShotAnchorer, single_shot_id
from antelligence.anchoring.pins import TrustPins
from antelligence.anchoring.ots import OtsClient
from antelligence.anchoring.readers import ZERO32, Web3ChainReader
from antelligence.anchoring.rekor import AnchorKey, RekorClient
from antelligence.anchoring.verify import OutsideVerifier
from antelligence.experiments.stats import wilson_interval
from antelligence.kernel.board import Board, anchor_sizes
from antelligence.kernel.canonical import jcs_bytes

CONDITIONS = ("B", "OTS", "R", "C")
ATTACKS = ("T1", "T2", "T3", "T4")
LOCAL_CHAIN_IDS = {31337, 1337}
N_ENTRIES = 230          # 4 mid-run anchors + a distinct run-end anchor
TAIL_ENTRIES = 220       # mid-run snapshot: last anchor at 200, 20 entries unanchored
TRUNCATE_K = (1, 10, 60)
TAIL_TRUNCATE_K = (1, 10)
PREREG_DOC = b"v4.1 preregistration, locked before any evaluation episode\n"
FORGED_DOC = b"v4.1 preregistration, edited after seeing the results\n"


def expected_detects(attack: str, position: str, condition: str) -> bool:
    """The brief's table: B detects none; OTS only T4; R and C detect T1-T4 in the anchored prefix; the tail nothing."""
    if position.startswith("tail") or condition == "B":
        return False
    if condition == "OTS":
        return attack == "T4"
    return True


# --------------------------------------------------------------------------------------------------
# A lab: one set of outside systems
# --------------------------------------------------------------------------------------------------

class DirectChainMode:
    """Mode C without the queue: writes straight to a registry so a revert is visible as a write-time rejection."""

    def __init__(self, registry, sender: str) -> None:
        self.registry, self.sender = registry, sender
        self._last: Dict[str, bytes] = {}
        self._n = 0

    def submit(self, run_id: str, size: int, root: bytes) -> int:
        self.registry.anchor(self.sender, bytes.fromhex(run_id), size, root, self._last.get(run_id, ZERO32))
        self._last[run_id] = bytes(root)
        self._n += 1
        return self._n

    def submit_once(self, single_shot: bytes, digest: bytes) -> int:
        self.registry.anchor_once(self.sender, single_shot, digest)
        self._n += 1
        return self._n

    def confirmed(self, job: int) -> bool:
        return True

    def tx_hash(self, job: int) -> str:
        return f"direct-{job}"


class Web3Registry:
    """A real ``AnchorRegistry`` on a local node (unlocked accounts; no private key is read or held here)."""

    def __init__(self, w3, contract) -> None:
        if int(w3.eth.chain_id) not in LOCAL_CHAIN_IDS:
            raise RuntimeError("Web3Registry only talks to a local hardhat/anvil node (chain id 31337 or 1337)")
        self.w3, self.contract = w3, contract
        self.address = contract.address
        self.deployer_address = Web3.to_checksum_address(contract.functions.deployer().call())

    def _send(self, fn) -> None:
        from antelligence.anchoring.mocks import Revert

        try:
            tx = fn.transact({"from": self.deployer_address})
            receipt = self.w3.eth.wait_for_transaction_receipt(tx)
        except Exception as exc:
            raise Revert(str(exc)[:120]) from exc
        if int(receipt["status"]) != 1:
            raise Revert("status 0")

    def anchor(self, sender, run_id, size, root, prev_root):
        self._send(self.contract.functions.anchor(run_id, size, root, prev_root))

    def anchor_once(self, sender, single_shot, digest):
        self._send(self.contract.functions.anchorOnce(single_shot, digest))


def deploy_local_registry(w3, artifact: Path) -> Web3Registry:
    """Deploy ``AnchorRegistry`` from a compiled hardhat artifact onto a local node."""
    if int(w3.eth.chain_id) not in LOCAL_CHAIN_IDS:
        raise RuntimeError("refusing to deploy anywhere but a local node (chain id 31337 or 1337)")
    data = json.loads(Path(artifact).read_text())
    account = w3.eth.accounts[0]
    factory = w3.eth.contract(abi=data["abi"], bytecode=data["bytecode"])
    receipt = w3.eth.wait_for_transaction_receipt(factory.constructor().transact({"from": account}))
    return Web3Registry(w3, w3.eth.contract(address=receipt["contractAddress"], abi=data["abi"]))


class Lab:
    """Fresh OTS calendars, Rekor log and registry, plus the keys a verifier is allowed to know."""

    def __init__(self, registry_factory: Optional[Callable[[], Any]] = None) -> None:
        self._tick = itertools.count(1_700_000_000)
        clock = lambda: next(self._tick)  # noqa: E731  (strictly increasing, deterministic)
        self.ots_transport = mocks.MockOtsTransport(clock=clock)
        self.mock_rekor = mocks.MockRekor(clock=clock)
        self.rekor = RekorClient(self.mock_rekor, pinned_log_key_sha256=self.mock_rekor.log_key_sha256)
        if registry_factory is None:
            self.deployer = Web3.to_checksum_address("0x" + "d1" * 20)
            self.registry = mocks.MockRegistry(self.deployer, clock=clock)
            self._w3 = mocks.MockW3(self.registry)
            self.registry_address = mocks.MockW3.ADDRESS
        else:
            self.registry = registry_factory()
            self.deployer = self.registry.deployer_address
            self._w3 = self.registry.w3
            self.registry_address = self.registry.address
        self.study_key = AnchorKey()
        self._prereg_file: Optional[Path] = None

    def prereg_doc(self, base: bytes = PREREG_DOC) -> bytes:
        """The preregistration as locked: ``base`` plus the line that fixes this lab's study key."""
        return base + f"Study key fingerprint (sha256 of SPKI DER): `{self.study_key.fingerprint}`\n".encode()

    def prereg_path(self) -> Path:
        """The locked preregistration as a file (what ``--prereg`` points at), kept outside any exported run."""
        if self._prereg_file is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="v41-prereg-")
            self._prereg_file = Path(self._tmp.name) / "PREREGISTRATION.md"
            self._prereg_file.write_bytes(self.prereg_doc())
        return self._prereg_file

    def forged_prereg_doc(self) -> bytes:
        """What a swapping operator would produce: different text; it may keep or drop the study key line."""
        return self.prereg_doc(FORGED_DOC)

    def pins(self) -> TrustPins:
        """The pins a verifier would build from the deployments file at a commit and the locked preregistration."""
        return TrustPins(registry=Web3.to_checksum_address(self.registry_address), deployer=self.deployer,
                         code_hash=self.reader().code_hash(), study_key_sha256=self.study_key.fingerprint)

    def ots_client(self) -> OtsClient:
        return OtsClient(self.ots_transport, calendars=("https://cal-a.mock", "https://cal-b.mock"))

    def chain_mode(self) -> DirectChainMode:
        return DirectChainMode(self.registry, self.deployer)

    def reader(self) -> Web3ChainReader:
        return Web3ChainReader(self._w3, self.registry_address, self.deployer)

    def anchorer(self, board: Board, akey: AnchorKey, out_dir: Path) -> Anchorer:
        return Anchorer(board.run_id, {"OTS", "R", "C"}, out_dir, ots=self.ots_client(), rekor=self.rekor,
                        anchor_key=akey, chain=self.chain_mode())

    def single_shot(self, out_dir: Path) -> SingleShotAnchorer:
        return SingleShotAnchorer({"OTS", "R", "C"}, out_dir, ots=self.ots_client(), rekor=self.rekor,
                                  study_key=self.study_key, chain=self.chain_mode())

    def header_provider(self):
        return self.ots_transport.header

    def confirm_ots(self, directory: Path) -> None:
        """Bitcoin confirms the calendars' commitments, then every proof in ``directory`` is upgraded."""
        self.ots_transport.mine_block()
        client = self.ots_client()
        for path in sorted(Path(directory).glob("*.ots")):
            client.upgrade(path)


# --------------------------------------------------------------------------------------------------
# Attacks
# --------------------------------------------------------------------------------------------------

def _tweak(entry_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    p = json.loads(json.dumps(payload))
    if entry_type == "evidence":
        p["look_p_ppm"] = (p["look_p_ppm"] + 1) % 1_000_001
    elif entry_type == "decision":
        p["release"] = 1 - p["release"]
    elif entry_type == "knowledge":
        p["volume_mm3"] += 1
    elif entry_type == "task":
        p["until_tick"] += 1
    else:
        p["value_ppm"] += 1
    return p


def rebuild(board: Board, keys, *, rewrite_at: Optional[int] = None, keep: Optional[int] = None) -> Board:
    """The operator's tool: replay ``board`` with every key, optionally changing one entry or dropping the tail."""
    entries = board.entries()[: board.size() if keep is None else keep]
    out = Board(board.header, mode="B", keys=keys.seeds)
    id_map: Dict[str, str] = {}
    for e in entries:
        payload = _tweak(e.type, dict(e.payload)) if e.seq == rewrite_at else dict(e.payload)
        new = out.append(e.type, dict(e.author), payload, e.tick, cites=[id_map[c] for c in e.cites])
        id_map[e.id] = new.id
    return out


def _export(board: Board, akey: AnchorKey, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    board.export_jsonl(directory / f"{board.run_id}.jsonl")
    (directory / f"{board.run_id}.anchor_pub.pem").write_bytes(akey.pem)


def _positions(n: int) -> Dict[str, int]:
    """Seq of the entry an attack touches. 'last anchor' is the run-end anchor at size n."""
    return {"early": 5, "middle": n // 2, "before_last_anchor": n - 3}


@dataclass
class Outcome:
    attack: str
    position: str
    condition: str
    detected: bool
    when: str                  # "write" (rejected on write), "audit" (found by the verifier) or "none"
    wall_s: float
    bytes: int
    network_calls: int
    tail: Optional[int] = None
    audit: bool = False        # the verifier itself flags it (independent of any write-time rejection)


def _anchor_all(lab: Lab, board: Board, akey: AnchorKey, out: Path, sizes: Sequence[int]) -> Anchorer:
    anchorer = lab.anchorer(board, akey, out)
    for s in sizes:
        anchorer.anchor(board, s)
    return anchorer


def _verifier(lab: Lab, directory: Path, plan: Dict[str, Any], condition: str) -> OutsideVerifier:
    return OutsideVerifier(
        directory, plan,
        chain=lab.reader() if condition == "C" else None,
        rekor=lab.rekor if condition == "R" else None,
        ots_dir=directory, header_provider=lab.header_provider() if condition == "OTS" else None,
        conditions={condition}, pins=lab.pins(), prereg=lab.prereg_path(), study_pem=lab.study_key.pem,
    )


def _verify(lab: Lab, directory: Path, plan: Dict[str, Any], condition: str) -> Dict[str, Any]:
    """A real verifier run. Detection = verdict ``failed`` (``unverified`` is not a detection)."""
    if condition == "OTS":
        lab.confirm_ots(directory)
    return _verifier(lab, directory, plan, condition).verify()


def _plan(board: Board, akey: AnchorKey) -> Dict[str, Any]:
    return {"study": "v4.1", "runs": [{"run_id": board.run_id, "pubkeys": board.header["pubkeys"], "anchor_pubkey": akey.fingerprint}]}


def run_scenario(
    attack: str, position: str, *, seed: int, k: int = 0, tail: bool = False,
    registry_factory: Optional[Callable[[], Any]] = None, workdir: Optional[Path] = None,
    latency_sink: Optional[Dict[str, List[float]]] = None,
) -> List[Outcome]:
    """One honest run, one attack, verified under B, OTS, R and C. Returns one :class:`Outcome` per condition."""
    own_tmp = workdir is None
    root = Path(tempfile.mkdtemp(prefix="v41-tamper-")) if own_tmp else Path(workdir)
    try:
        lab = Lab(registry_factory)
        n = TAIL_ENTRIES if tail else N_ENTRIES
        board, keys, akey = mocks.synthetic_run(n, seed=seed)
        sizes = [s for s in anchor_sizes(n) if s % 50 == 0] if tail else anchor_sizes(n)
        honest_dir, attacked_dir = root / "honest", root / "attacked"
        _export(board, akey, honest_dir)
        honest_anchorer = _anchor_all(lab, board, akey, honest_dir, sizes)
        if latency_sink is not None:
            for mode, values in honest_anchorer.latency_s.items():
                latency_sink.setdefault(mode, []).extend(values)
        plan = _plan(board, akey)
        study = lab.single_shot(root / "study")        # the operator anchors prereg and run plan once, before the runs
        study.anchor("prereg", hashlib.sha256(lab.prereg_doc()).digest())
        study.anchor("runplan", hashlib.sha256(jcs_bytes(plan)).digest())
        pos_label = (f"tail_truncate_{k}" if attack == "T2" else "tail") if tail else position

        if attack == "T4":
            return _t4(lab, root, pos_label)

        if attack == "H":     # honest control
            return _collect(lab, honest_dir, plan, "H", "honest", None)
        if attack == "T2":
            forged = rebuild(board, keys, keep=n - k)
        else:                 # T1, T3
            j = (n - 10 + 3) if tail else _positions(n)[position]
            if tail:
                j = n - 5
            forged = rebuild(board, keys, rewrite_at=j)
        _export(forged, akey, attacked_dir)
        attacker = _reanchor(lab, board, forged, akey, sizes, honest_dir, attacked_dir, tail)
        write_rejected = any(e["mode"] == "C" for e in attacker.errors)
        if attack == "T3":
            return _collect(lab, attacked_dir, plan, attack, pos_label, write_rejected, other_dir=honest_dir)
        return _collect(lab, attacked_dir, plan, attack, pos_label, write_rejected)
    finally:
        if own_tmp:
            shutil.rmtree(root, ignore_errors=True)


def _reanchor(lab: Lab, honest: Board, forged: Board, akey: AnchorKey, honest_sizes: Sequence[int],
              honest_dir: Path, attacked_dir: Path, tail: bool) -> Anchorer:
    """Everything the operator can do to give ``forged`` the anchors a verifier expects.

    A (size, root) the honest run already anchored is reused as is (its ``.ots`` is copied; Rekor and the
    chain already hold it). Anything new is stamped, logged and posted, and the chain may reject it.
    """
    n = forged.size()
    needed = [s for s in anchor_sizes(n) if s % 50 == 0] if tail else anchor_sizes(n)
    anchorer = lab.anchorer(forged, akey, attacked_dir)
    for s in needed:
        reusable = s in honest_sizes and s <= honest.size() and forged.root(s) == honest.root(s)
        src = honest_dir / f"{honest.run_id}.{s}.ots"
        if reusable and src.exists():
            shutil.copy(src, attacked_dir / src.name)
        else:
            anchorer.anchor(forged, s)
    return anchorer


def _collect(lab: Lab, directory: Path, plan, attack: str, position: str, write_rejected: Optional[bool],
             other_dir: Optional[Path] = None) -> List[Outcome]:
    out: List[Outcome] = []
    for cond in CONDITIONS:
        report = _verify(lab, directory, plan, cond)
        detected = report["verdict"] == "failed"
        wall, nbytes, calls = report["metrics"]["wall_s"], report["metrics"]["bytes"], report["metrics"]["network_calls"]
        tail = next(iter(report["runs"].values()), {}).get("unanchored_tail") if report["runs"] else None
        if other_dir is not None:      # T3: the second verifier sees the other history
            r2 = _verify(lab, other_dir, plan, cond)
            detected = detected or r2["verdict"] == "failed"
            wall += r2["metrics"]["wall_s"]
            nbytes += r2["metrics"]["bytes"]
            calls += r2["metrics"]["network_calls"]
        rejected = bool(write_rejected) and cond == "C"
        when = "write" if rejected else ("audit" if detected else "none")
        out.append(Outcome(attack, position, cond, detected or rejected, when, wall, nbytes, calls, tail, audit=detected))
    return out


def _t4(lab: Lab, root: Path, position: str) -> List[Outcome]:
    public_dir, forged_dir = root / "public", root / "forged"
    prereg_path = root / "prereg.md"
    prereg_path.write_bytes(lab.prereg_doc())
    plan = {"study": "v4.1", "runs": []}
    plan_digest = hashlib.sha256(jcs_bytes(plan)).digest()
    honest = lab.single_shot(public_dir)
    honest.anchor("prereg", hashlib.sha256(lab.prereg_doc()).digest())
    honest.anchor("runplan", plan_digest)
    lab.confirm_ots(public_dir)
    original_ots = public_dir / "prereg.ots"          # pushed to the public PR when the prereg was locked, after upgrade
    # The operator swaps the document and anchors the new one wherever the systems allow.
    forged_path = root / "prereg-forged.md"
    forged_path.write_bytes(lab.forged_prereg_doc())
    attacker = lab.single_shot(forged_dir)
    attacker.anchor("prereg", hashlib.sha256(lab.forged_prereg_doc()).digest())
    write_rejected = any(e["mode"] == "C" for e in attacker.errors)
    out: List[Outcome] = []
    for cond in CONDITIONS:
        v = _verifier(lab, root, plan, cond)
        res = v.verify_study(forged_path, plan_digest, prereg_ots=original_ots, study_pem=lab.study_key.pem)
        detected = res["failed"]
        rejected = write_rejected and cond == "C"
        when = "write" if rejected else ("audit" if detected else "none")
        calls = (v.chain.calls if v.chain else 0) + (v.rekor.calls if v.rekor else 0)
        out.append(Outcome("T4", position, cond, detected or rejected, when, v.metrics.wall_s(),
                           v.metrics.file_bytes + (v.chain.bytes_read if v.chain else 0), calls, audit=detected))
    return out


# --------------------------------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------------------------------

def scenarios() -> List[Tuple[str, str, int, bool]]:
    """(attack, position, k, tail) for every row group of the matrix, plus the honest control."""
    rows: List[Tuple[str, str, int, bool]] = [("H", "-", 0, False)]
    for attack in ("T1", "T3"):
        rows += [(attack, p, 0, False) for p in ("early", "middle", "before_last_anchor")]
    rows += [("T2", f"truncate_{k}", k, False) for k in TRUNCATE_K]
    rows.append(("T4", "-", 0, False))
    rows += [("T1", "tail", 0, True), ("T3", "tail", 0, True)]
    rows += [("T2", f"truncate_{k}", k, True) for k in TAIL_TRUNCATE_K]
    return rows


def run_matrix(n_runs: int, registry_factory: Optional[Callable[[], Any]] = None, base_seed: int = 0) -> Dict[str, Any]:
    cells: Dict[Tuple[str, str, str], List[Outcome]] = {}
    latency: Dict[str, List[float]] = {}
    for attack, position, k, tail in scenarios():
        for i in range(n_runs):
            outs = run_scenario(attack, position, seed=base_seed + i, k=k, tail=tail,
                                registry_factory=registry_factory, latency_sink=latency)
            for o in outs:
                cells.setdefault((o.attack, o.position, o.condition), []).append(o)
    rows = []
    for (attack, position, cond), outs in sorted(cells.items()):
        trials, hits = len(outs), sum(o.detected for o in outs)
        rows.append({
            "attack": attack, "position": position, "condition": cond, "trials": trials, "detected": hits,
            "rate": hits / trials, "wilson": wilson_interval(hits, trials),
            "when": {w: sum(o.when == w for o in outs) for w in ("write", "audit", "none")},
            "audit_detected": sum(o.audit for o in outs),
            "mean_wall_s": sum(o.wall_s for o in outs) / trials,
            "mean_bytes": sum(o.bytes for o in outs) / trials,
            "mean_network_calls": sum(o.network_calls for o in outs) / trials,
            "unanchored_tail_entries": sorted({o.tail for o in outs if o.tail is not None}),
        })
    return {
        "runs_per_cell": n_runs, "rows": rows,
        "anchor_latency_s": {m: sum(v) / len(v) for m, v in latency.items() if v},
    }


def table_mismatches(matrix: Dict[str, Any]) -> List[str]:
    """Rows whose observed detection disagrees with :func:`expected_detects` (empty list = the brief's table holds)."""
    bad = []
    for r in matrix["rows"]:
        if r["attack"] == "H":
            want = 0.0
        elif r["position"].startswith("tail"):
            want = 0.0
        else:
            want = 1.0 if expected_detects(r["attack"], r["position"], r["condition"]) else 0.0
        if r["rate"] != want:
            bad.append(f"{r['attack']}/{r['position']}/{r['condition']}: expected {want}, got {r['rate']}")
    return bad
