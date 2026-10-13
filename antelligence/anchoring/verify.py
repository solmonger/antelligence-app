"""Outside verifier: public inputs only, no private key anywhere.

Inputs: the exported ``<run_id>.jsonl`` files, the anchored run plan (run ids, run public keys, anchor
public-key fingerprints), an ``AnchorRegistry`` reader pinned to a registry and deployer, ``.ots`` proofs,
a Rekor client, and the public PEM of each run's anchor key. See DESIGN-anchors.md.

Every check returns ``{"ok": bool, ...detail}``; a run passes when all enabled checks pass.
Conditions: ``B`` (signatures, ids, hash chain; always on), ``OTS``, ``R``, ``C``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from antelligence.anchoring.ots import HeaderProvider, verify_ots
from antelligence.anchoring.readers import ZERO32, ChainReader
from antelligence.anchoring.rekor import (
    RekorClient,
    RekorError,
    check_record,
    fingerprint_of,
    load_public_pem,
)
from antelligence.kernel.board import (
    ANCHOR_EVERY,
    ZERO_HASH,
    BoardError,
    Entry,
    anchor_digest,
    anchor_sizes,
    compute_id,
    merkle_root,
    run_id_of,
    validate_header,
    verify_sig,
)
from antelligence.kernel.canonical import jcs_bytes

CONDITIONS = ("B", "OTS", "R", "C")


def required_sizes(n: int) -> List[int]:
    """Anchors that must exist for ``n`` entries: every multiple of 50. The final size is optional (run end)."""
    return list(range(ANCHOR_EVERY, n + 1, ANCHOR_EVERY))


def _check(ok: bool, **detail: Any) -> Dict[str, Any]:
    return {"ok": bool(ok), **detail}


@dataclass
class Metrics:
    start: float = field(default_factory=time.perf_counter)
    file_bytes: int = 0

    def wall_s(self) -> float:
        return time.perf_counter() - self.start


class LoadedRun:
    def __init__(self, run_id: str, header: Dict[str, Any], entries: List[Entry], problems: List[str]):
        self.run_id, self.header, self.entries, self.problems = run_id, header, entries, problems
        self.ids = [bytes.fromhex(e.id) for e in entries]

    def root(self, size: int) -> bytes:
        return merkle_root(self.ids[:size])

    def digest(self, size: int) -> bytes:
        return anchor_digest(self.run_id, size, self.root(size))


def load_export(path: Path, metrics: Metrics) -> LoadedRun:
    raw = path.read_bytes()
    metrics.file_bytes += len(raw)
    lines = raw.decode("utf-8").splitlines()
    problems: List[str] = []
    header: Dict[str, Any] = {}
    entries: List[Entry] = []
    try:
        header = json.loads(lines[0])
        validate_header(header)
    except (IndexError, ValueError, BoardError) as exc:
        problems.append(f"header: {exc}")
    for i, line in enumerate(lines[1:]):
        if not line.strip():
            continue
        try:
            entries.append(Entry.from_dict(json.loads(line)))
        except (ValueError, BoardError, KeyError, TypeError) as exc:
            problems.append(f"entry line {i + 2}: unreadable ({exc})")
            break
    return LoadedRun(run_id_of(header) if header else "", header, entries, problems)


class OutsideVerifier:
    def __init__(
        self,
        export_dir: Union[str, Path],
        plan: Dict[str, Any],
        *,
        chain: Optional[ChainReader] = None,
        rekor: Optional[RekorClient] = None,
        ots_dir: Optional[Union[str, Path]] = None,
        header_provider: Optional[HeaderProvider] = None,
        conditions: Iterable[str] = ("B",),
        claims: Sequence[Dict[str, Any]] = (),
        pinned_deployer: Optional[str] = None,
    ) -> None:
        self.dir = Path(export_dir)
        self.plan = plan
        self.chain, self.rekor = chain, rekor
        self.ots_dir = Path(ots_dir) if ots_dir else self.dir
        self.header_provider = header_provider
        self.conditions = frozenset(conditions) | {"B"}
        unknown = self.conditions - set(CONDITIONS)
        if unknown:
            raise ValueError(f"unknown verifier conditions {sorted(unknown)}")
        if "C" in self.conditions and chain is None:
            raise ValueError("condition C needs a chain reader")
        if "R" in self.conditions and rekor is None:
            raise ValueError("condition R needs a rekor client")
        self.claims = list(claims)
        self.pinned_deployer = pinned_deployer
        self.metrics = Metrics()
        self._rekor_records: Dict[str, list] = {}
        self._rekor_bytes = 0

    # -- entry point ---------------------------------------------------------------------------
    def verify(self) -> Dict[str, Any]:
        self.metrics = Metrics()
        self._rekor_records, self._rekor_bytes = {}, 0
        report: Dict[str, Any] = {"conditions": sorted(self.conditions), "runs": {}, "problems": []}
        planned = {r["run_id"]: r for r in self.plan.get("runs", [])}
        present = {p.stem for p in self.dir.glob("*.jsonl")}
        for rid in sorted(set(planned) - present):
            report["problems"].append(f"planned run {rid} has no export")
        for rid in sorted(present - set(planned)):
            report["problems"].append(f"export {rid} is not in the anchored run plan")
        for rid in sorted(set(planned) & present):
            report["runs"][rid] = self._verify_run(rid, planned[rid])
        report["ok"] = not report["problems"] and all(r["ok"] for r in report["runs"].values())
        report["metrics"] = {
            "wall_s": round(self.metrics.wall_s(), 6),
            "bytes": self.metrics.file_bytes + self._net_bytes(),
            "network_calls": self._net_calls(),
        }
        return report

    def _net_calls(self) -> int:
        return (self.chain.calls if self.chain else 0) + (self.rekor.calls if self.rekor else 0)

    def _net_bytes(self) -> int:
        return (self.chain.bytes_read if self.chain else 0) + getattr(self, "_rekor_bytes", 0)

    # -- one run -------------------------------------------------------------------------------
    def _verify_run(self, rid: str, planned: Dict[str, Any]) -> Dict[str, Any]:
        run = load_export(self.dir / f"{rid}.jsonl", self.metrics)
        checks: Dict[str, Dict[str, Any]] = {}
        checks["header"] = self._header(run, rid, planned)
        checks["integrity"] = self._integrity(run)
        checks["signatures"] = self._signatures(run)
        tail: Dict[str, Any] = {}
        if "C" in self.conditions:
            checks["chain"] = self._chain(run, tail)
        if "R" in self.conditions:
            checks["rekor"] = self._rekor(run, planned, tail)
        if "OTS" in self.conditions:
            checks["ots"] = self._ots(run)
        checks["anchor_times"] = self._times(run)
        ok = all(c["ok"] for c in checks.values())
        return {"ok": ok, "entries": len(run.entries), "unanchored_tail": tail.get("tail"), "checks": checks}

    def _header(self, run: LoadedRun, rid: str, planned: Dict[str, Any]) -> Dict[str, Any]:
        problems = list(run.problems)
        if run.header:
            if run.run_id != rid:
                problems.append("run_id is not sha256(JCS(header))")
            if run.header.get("pubkeys") != planned.get("pubkeys"):
                problems.append("run public keys differ from the anchored run plan")
            if run.header.get("anchor_pubkey") != planned.get("anchor_pubkey"):
                problems.append("anchor public key differs from the anchored run plan")
        return _check(not problems, problems=problems)

    def _integrity(self, run: LoadedRun) -> Dict[str, Any]:
        problems: List[str] = []
        seen: set = set()
        prev = ZERO_HASH
        for i, e in enumerate(run.entries):
            if e.run_id != run.run_id:
                problems.append(f"entry {i}: run_id mismatch")
            if e.seq != i:
                problems.append(f"entry {i}: seq is {e.seq}")
            if e.prev_hash != prev:
                problems.append(f"entry {i}: prev_hash chain broken")
            if compute_id(e.body()) != e.id:
                problems.append(f"entry {i}: id does not match content")
            if any(c not in seen for c in e.cites):
                problems.append(f"entry {i}: cites an entry that is not earlier")
            seen.add(e.id)
            prev = e.id
        return _check(not problems, problems=problems[:20], count=len(problems))

    def _signatures(self, run: LoadedRun) -> Dict[str, Any]:
        problems: List[str] = []
        pubkeys = run.header.get("pubkeys", {}) if run.header else {}
        for i, e in enumerate(run.entries):
            pub = pubkeys.get(e.author["agent"])
            if e.sig is None or pub is None or not verify_sig(pub, e.id, e.sig):
                problems.append(f"entry {i}: bad or missing signature")
        return _check(not problems, problems=problems[:20], count=len(problems))

    # -- C -------------------------------------------------------------------------------------
    def _chain(self, run: LoadedRun, tail: Dict[str, Any]) -> Dict[str, Any]:
        rid, n = bytes.fromhex(run.run_id), len(run.entries)
        problems: List[str] = []
        deployer = self.chain.deployer()
        if self.pinned_deployer and deployer.lower() != self.pinned_deployer.lower():
            problems.append(f"registry deployer {deployer} is not the pinned deployer")
        last = self.chain.last_size(rid)
        if last == 0:
            return _check(False, problems=["run has no anchor on chain"])
        if self.chain.run_owner(rid).lower() != deployer.lower():
            problems.append("run owner is not the deployer")
        sizes = anchor_sizes(last)
        if self.chain.anchor_count(rid) != len(sizes):
            problems.append(f"chain holds {self.chain.anchor_count(rid)} anchors, the 50-grid up to {last} needs {len(sizes)}")
        if n < last:
            problems.append(f"export has {n} entries but the chain anchored size {last} (truncated)")
        for s in sizes:
            if s > n:
                continue
            onchain = self.chain.root_at(rid, s)
            if onchain == ZERO32:
                problems.append(f"size {s}: no root on chain")
            elif onchain != run.root(s):
                problems.append(f"size {s}: root on chain differs from the root recomputed from the export")
        tail["tail"] = n - last if n >= last else 0
        return _check(not problems, problems=problems[:20], last_size=last, tail=tail["tail"])

    # -- R -------------------------------------------------------------------------------------
    def _rekor(self, run: LoadedRun, planned: Dict[str, Any], tail: Dict[str, Any]) -> Dict[str, Any]:
        n = len(run.entries)
        pem_path = self.dir / f"{run.run_id}.anchor_pub.pem"
        problems: List[str] = []
        if not pem_path.exists():
            return _check(False, problems=["anchor public key PEM missing"])
        pem = pem_path.read_bytes()
        self.metrics.file_bytes += len(pem)
        try:
            fp = fingerprint_of(load_public_pem(pem))
        except (RekorError, ValueError) as exc:
            return _check(False, problems=[f"anchor public key unreadable: {exc}"])
        if fp != planned.get("anchor_pubkey"):
            return _check(False, problems=["anchor PEM does not match the anchored anchor_pubkey fingerprint"])
        records = self._rekor_records[run.run_id] = self.rekor.entries_for_key(pem)
        self._rekor_bytes = getattr(self, "_rekor_bytes", 0) + sum(r.raw_bytes for r in records)
        for r in records:
            for p in check_record(r):
                problems.append(f"rekor entry {r.uuid[:12]}: {p}")
        logged = {r.digest for r in records}
        required = {run.digest(s).hex(): s for s in required_sizes(n)}
        acceptable = set(required) | {run.digest(n).hex()} if n else set(required)
        for d, s in required.items():
            if d not in logged:
                problems.append(f"size {s}: digest not logged under the run's anchor key")
        extra = sorted(logged - acceptable)
        if extra:
            problems.append(f"{len(extra)} logged digest(s) under the run's anchor key match no prefix of this history (second history, rewrite or truncation)")
        last_logged = max([s for d, s in required.items() if d in logged] or [0])
        tail["tail"] = n - last_logged if run.digest(n).hex() not in logged else 0
        return _check(not problems, problems=problems[:20], rekor_entries=len(records),
                      required_anchors=len(required), unmatched_entries=len(extra), tail=tail["tail"])

    # -- OTS -----------------------------------------------------------------------------------
    def _ots(self, run: LoadedRun) -> Dict[str, Any]:
        n = len(run.entries)
        problems: List[str] = []
        pending = verified = 0
        required = required_sizes(n)
        for s in required + ([n] if n and n not in required else []):
            path = self.ots_dir / f"{run.run_id}.{s}.ots"
            if not path.exists():
                if s in required:
                    problems.append(f"size {s}: no .ots proof")
                continue                       # the run-end proof is optional on a mid-run snapshot
            self.metrics.file_bytes += path.stat().st_size
            res = verify_ots(path, run.digest(s), self.header_provider)
            if not res.ok:
                problems.append(f"size {s}: {res.error}")
            elif res.status == "pending":
                pending += 1
            else:
                verified += 1
        return _check(not problems, problems=problems[:20], bitcoin=verified, pending=pending)

    # -- anchor time before claims -------------------------------------------------------------
    def _anchor_times(self, run: LoadedRun) -> Dict[str, Dict[int, int]]:
        times: Dict[str, Dict[int, int]] = {}
        n = len(run.entries)
        if "C" in self.conditions:
            rid = bytes.fromhex(run.run_id)
            t = {s: self.chain.anchor_time(rid, s) for s in required_sizes(n)}
            times["C"] = {s: v for s, v in t.items() if v is not None}
        if "R" in self.conditions:
            by_digest = {run.digest(s).hex(): s for s in required_sizes(n)}
            times["R"] = {by_digest[r.digest]: r.integrated_time
                          for r in self._rekor_records.get(run.run_id, []) if r.digest in by_digest}
        if "OTS" in self.conditions and self.header_provider is not None:
            t_ots: Dict[int, int] = {}
            for s in required_sizes(n) + ([n] if n % ANCHOR_EVERY else []):
                path = self.ots_dir / f"{run.run_id}.{s}.ots"
                if path.exists():
                    res = verify_ots(path, run.digest(s), self.header_provider)
                    if res.ok and res.time is not None:
                        t_ots[s] = res.time
            times["OTS"] = t_ots
        return times

    def _times(self, run: LoadedRun) -> Dict[str, Any]:
        times = self._anchor_times(run)
        problems: List[str] = []
        for system, by_size in times.items():
            ordered = [by_size[s] for s in sorted(by_size)]
            if ordered != sorted(ordered):
                problems.append(f"{system}: anchor times are not monotone in size")
        for claim in self.claims:
            if claim.get("run_id", run.run_id) != run.run_id:
                continue
            covers, made = int(claim["covers_size"]), int(claim["made_at"])
            earliest = [min(t for s, t in by.items() if s >= covers) for by in times.values()
                        if any(s >= covers for s in by)]
            if not earliest:
                problems.append(f"claim {claim.get('id')}: no anchor covers size {covers}")
            elif min(earliest) > made:
                problems.append(f"claim {claim.get('id')}: made at {made}, before the earliest covering anchor ({min(earliest)})")
        return _check(not problems, problems=problems, systems=sorted(times))

    # -- study-level single-shot documents (preregistration, run plan) -------------------------
    def verify_study(
        self,
        prereg_doc: Union[str, Path],
        runplan_digest: bytes,
        *,
        prereg_ots: Optional[Union[str, Path]] = None,
        study_pem: Optional[bytes] = None,
        single_shot_ids: Optional[Dict[str, bytes]] = None,
    ) -> Dict[str, Any]:
        """The preregistration document in hand must be the one that was anchored (T4)."""
        from antelligence.anchoring.anchorer import single_shot_id

        ids = single_shot_ids or {"prereg": single_shot_id("prereg"), "runplan": single_shot_id("runplan")}
        doc = Path(prereg_doc).read_bytes()
        self.metrics.file_bytes += len(doc)
        prereg_digest = hashlib.sha256(doc).digest()
        checks: Dict[str, Dict[str, Any]] = {}
        if "OTS" in self.conditions:
            if prereg_ots is None:
                checks["ots"] = _check(False, problems=["no preregistration .ots proof supplied"])
            else:
                self.metrics.file_bytes += Path(prereg_ots).stat().st_size
                res = verify_ots(prereg_ots, prereg_digest, self.header_provider)
                checks["ots"] = _check(res.ok, problems=[res.error] if res.error else [], status=res.status)
        if "C" in self.conditions:
            problems = []
            if self.chain.once_digest(ids["prereg"]) != prereg_digest:
                problems.append("preregistration digest on chain differs from this document")
            if self.chain.once_digest(ids["runplan"]) != runplan_digest:
                problems.append("run plan digest on chain differs from this run plan")
            checks["chain"] = _check(not problems, problems=problems)
        if "R" in self.conditions:
            if study_pem is None:
                checks["rekor"] = _check(False, problems=["no study public key supplied"])
            else:
                records = self.rekor.entries_for_key(study_pem)
                self._rekor_bytes = getattr(self, "_rekor_bytes", 0) + sum(r.raw_bytes for r in records)
                logged = {r.digest for r in records}
                want = {prereg_digest.hex(), runplan_digest.hex()}
                problems = [f"digest {d[:12]} not logged" for d in sorted(want - logged)]
                if logged - want:
                    problems.append(f"{len(logged - want)} other digest(s) logged under the study key (second preregistration or plan)")
                for r in records:
                    problems += [f"rekor entry {r.uuid[:12]}: {p}" for p in check_record(r)]
                checks["rekor"] = _check(not problems, problems=problems)
        return {"ok": all(c["ok"] for c in checks.values()) if checks else True, "checks": checks}
