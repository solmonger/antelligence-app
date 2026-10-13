"""Outside verifier: public inputs only, no private key anywhere.

ADVERSARIAL CHECKLIST: every input and what authenticates it. An input with no row here is not trusted.

* deployments file   -> ``git show <40-hex commit>:path`` only (never the working tree, never a ref, replace refs off);
                        with ``--prereg-commit`` the commit must be an ancestor of it. Registry, deployer, code hash.
* preregistration    -> sha256 equals the ``anchorOnce`` digest on the PINNED registry; its single visible study-key
                        line (not in a comment or code block) must equal the pin. Without the chain it authenticates nothing.
* run plan           -> sha256(JCS) equals the ``anchorOnce`` runplan digest on chain AND/OR is logged under the pinned
                        study key on Rekor; any other digest under that key, a duplicate run_id or a malformed run fails.
* caller registry / deployer / study PEM -> assertions only: compared to the pins, a mismatch fails.
* chain reader       -> address, ``deployer()`` and keccak(code) all equal the pins before any anchor is read.
* run header         -> run_id == sha256(JCS(header)); pubkeys and anchor_pubkey must be PRESENT in the plan and equal it.
* entries            -> ids recomputed, hash chain, cites, every signature against the header keys (none skipped).
* Merkle roots       -> recomputed from the export; compared to chain roots, Rekor digests, OTS digests. Never read from a file.
* ``<run>.anchor_pub.pem`` -> sha256(SPKI DER) equals the plan's anchor_pubkey.
* Rekor entries      -> inclusion proof + SET against the PINNED log key + signature by the anchor key (``check_record``).
* ``<run>.anchors.json`` -> NOT trusted: each record's size/root/digest must equal the recomputed values, its rekor_uuid
                        must be a verified entry for that digest, and EVERY record with a rekor_uuid must carry a log-signed
                        ``rekor_checkpoint`` (missing -> unverified). Checkpoints are then consistency-checked against the live log.
* ``.ots`` proofs    -> digest recomputed per size; Bitcoin header hash, proof of work and difficulty floor checked locally.
* claims             -> must name a verified run and postdate the earliest covering anchor; an unknown run id fails.
* verdict            -> worst-of over every check, run and condition; ``failed`` > ``unverified`` > ``verified``;
                        no absent field, exception or option makes a check pass.

Inputs: the exported ``<run_id>.jsonl`` files, the anchored run plan (run ids, run public keys, anchor
public-key fingerprints), an ``AnchorRegistry`` reader pinned to a registry and deployer, ``.ots`` proofs,
a Rekor client, and the public PEM of each run's anchor key. See DESIGN-anchors.md.

Every check returns ``{"ok": bool, ...detail}``. The verifier FAILS CLOSED:

* a check that cannot reach a verdict (network error, no block-header source, no pinned log key, a calendar
  promise instead of a Bitcoin proof) is ``unverified`` (``ok`` False, ``unverified`` True, with ``error``);
* an enabled outside condition (OTS, R, C) that verified zero anchors for a run fails that run;
* condition ``B`` (signatures, ids, hash chain) cannot detect an operator who holds the keys, so a B-only
  report is never better than ``unverified``;
* the plan must be authenticated against the anchored run-plan digest, else the verdict is at most ``unverified``;
* **trust anchors are pinned** (:class:`~antelligence.anchoring.pins.TrustPins`): the registry address, deployer and
  code hash come from the deployments file at a commit, the study key from the preregistration whose digest is
  anchored on chain. A registry, deployer or study key supplied by the caller is only compared against those pins;
  a mismatch fails. The plan is authenticated only through a preregistration (``prereg=``) that is itself anchored
  on the pinned registry;
* the verdict is the **worst** over all checks and files: ``failed`` dominates ``unverified`` dominates ``verified``,
  and no option (missing header source, missing optional input) turns a ``failed`` into anything weaker.

Report ``verdict``: ``verified`` | ``unverified`` | ``failed``; ``ok`` is True only for ``verified``.
"""

from __future__ import annotations

import hashlib
import re
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from antelligence.anchoring.ots import HeaderProvider, verify_ots
from antelligence.anchoring.pins import PinError, TrustPins, study_key_in
from antelligence.anchoring.readers import ZERO32, ChainReader
from antelligence.anchoring.rekor import (
    RekorClient,
    RekorError,
    check_record,
    fingerprint_of,
    load_public_pem,
    parse_checkpoint,
    split_view_check,
    verify_checkpoint,
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
    """Anchors that must exist for ``n`` entries: every multiple of 50; for n < 50 the run-end anchor.

    On a longer run the final size is optional (a mid-run snapshot has no run-end anchor yet).
    """
    sizes = list(range(ANCHOR_EVERY, n + 1, ANCHOR_EVERY))
    return sizes or ([n] if n > 0 else [])


def _check(ok: bool, **detail: Any) -> Dict[str, Any]:
    return {"ok": bool(ok), **detail}


def _unverified(error: str, **detail: Any) -> Dict[str, Any]:
    return {"ok": False, "unverified": True, "error": error, "anchors_verified": 0, **detail}


def check_status(check: Dict[str, Any]) -> str:
    return "pass" if check["ok"] else ("unverified" if check.get("unverified") else "fail")


def _guard(fn):
    """Network and parse errors become an ``unverified`` check carrying the error string."""
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return _unverified(f"{type(exc).__name__}: {exc}")
    return wrapped


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


_RUN_EXPORT = re.compile(r"[0-9a-f]{64}\.jsonl")

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
        pins: Optional[TrustPins] = None,
        prereg: Optional[Union[str, Path]] = None,
        pinned_deployer: Optional[str] = None,
        pinned_registry: Optional[str] = None,
        study_pem: Optional[bytes] = None,
        require_final_anchor: bool = False,
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
        if "C" in self.conditions and pins is None:
            raise ValueError("condition C needs pins (registry, deployer, code hash from the deployments file at a commit): "
                             "no unpinned chain reads")
        if "R" in self.conditions and rekor is None:
            raise ValueError("condition R needs a rekor client")
        self.claims = list(claims)
        self.pins = pins
        self.prereg = Path(prereg) if prereg else None
        # What the caller asserts. These are NOT trust anchors: they are checked against ``pins`` and fail on mismatch.
        self.claimed_deployer, self.claimed_registry, self.study_pem = pinned_deployer, pinned_registry, study_pem
        # Completed runs must end on an anchor (tail == 0); mid-run snapshots keep the lenient default (goal 4.4).
        self.require_final_anchor = require_final_anchor
        self.metrics = Metrics()
        self._rekor_records: Dict[str, list] = {}
        self._rekor_bytes = 0
        self._rekor_proof_cps: Dict[str, list] = {}

    # -- pins ----------------------------------------------------------------------------------
    def _caller_pin_problems(self) -> List[str]:
        """Caller-supplied registry / deployer / study key that are not the pinned ones."""
        if self.pins is None:
            return []
        out: List[str] = []
        if self.claimed_registry and self.claimed_registry.lower() != self.pins.registry.lower():
            out.append("caller-supplied registry is not the pinned registry")
        if self.claimed_deployer and self.claimed_deployer.lower() != self.pins.deployer.lower():
            out.append("caller-supplied deployer is not the pinned deployer")
        if self.study_pem is not None:
            reason = self.pins.check_study_pem(self.study_pem)
            if reason:
                out.append(reason)
        return out

    def _chain_pin_problems(self) -> List[str]:
        """The chain reader must read the pinned registry, whose code and deployer are the pinned ones.

        Network errors raise (callers turn them into ``unverified``); a proven mismatch is a returned problem.
        """
        if self.pins is None or self.chain is None:
            return ["no pins: the chain reader cannot be trusted"]
        out: List[str] = []
        if str(getattr(self.chain, "address", "")).lower() != self.pins.registry.lower():
            out.append("chain reader is not pointed at the pinned registry address")
        if self.chain.deployer().lower() != self.pins.deployer.lower():
            out.append("registry deployer is not the pinned deployer")
        if self.chain.code_hash().lower() != self.pins.code_hash.lower():
            out.append("code at the registry address is not the pinned code hash")
        return out

    # -- entry point ---------------------------------------------------------------------------
    def verify(self) -> Dict[str, Any]:
        self.metrics = Metrics()
        self._rekor_records, self._rekor_bytes, self._rekor_proof_cps = {}, 0, {}
        report: Dict[str, Any] = {"conditions": sorted(self.conditions), "runs": {}, "problems": [], "unverified": []}
        report["problems"] += self._caller_pin_problems()
        planned = self._planned(report)
        if not planned:
            report["problems"].append("the run plan lists no runs: nothing to verify")
        # Only ``<64-hex run_id>.jsonl`` is a board export; any other *.jsonl (e.g. decisions.jsonl) is listed, never judged.
        files = sorted(self.dir.glob("*.jsonl"))
        present = {p.stem for p in files if _RUN_EXPORT.fullmatch(p.name)}
        report["ignored_files"] = [p.name for p in files if not _RUN_EXPORT.fullmatch(p.name)]
        for rid in sorted(set(planned) - present):
            report["problems"].append(f"planned run {rid} has no export")
        for rid in sorted(present - set(planned)):
            report["problems"].append(f"export {rid} is not in the anchored run plan")
        for rid in sorted(set(planned) & present):
            report["runs"][rid] = self._verify_run(rid, planned[rid])
        for claim in self.claims:
            if not isinstance(claim, dict) or claim.get("run_id") not in report["runs"]:
                label = claim.get("id") if isinstance(claim, dict) else repr(claim)
                report["problems"].append(f"claim {label} names no verified run")
        report["plan_authenticated"] = self._authenticate_plan(report)
        tails = {rid: r["unanchored_tail"] for rid, r in report["runs"].items()}
        report["unanchored_tail_by_run"] = tails
        report["unanchored_tail"] = max((t for t in tails.values() if t is not None), default=None)
        report["require_final_anchor"] = self.require_final_anchor
        statuses = [r["status"] for r in report["runs"].values()]
        if report["problems"] or "fail" in statuses:
            verdict = "failed"
        elif "unverified" in statuses or not report["plan_authenticated"] or self.conditions == {"B"}:
            verdict = "unverified"
            if self.conditions == {"B"}:
                report["unverified"].append("condition B alone cannot detect an operator who holds the keys")
            if not report["plan_authenticated"]:
                report["unverified"].append("run plan is not authenticated against an anchored run-plan digest")
        else:
            verdict = "verified"
        report["verdict"] = verdict
        report["ok"] = verdict == "verified"
        report["metrics"] = {
            "wall_s": round(self.metrics.wall_s(), 6),
            "bytes": self.metrics.file_bytes + self._net_bytes(),
            "network_calls": self._net_calls(),
        }
        return report

    def _planned(self, report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """run_id -> plan entry. Malformed or duplicate entries are problems, never silently dropped or overwritten."""
        planned: Dict[str, Dict[str, Any]] = {}
        runs = self.plan.get("runs") if isinstance(self.plan, dict) else None
        if not isinstance(runs, list):
            report["problems"].append("the run plan has no runs list")
            return planned
        for i, r in enumerate(runs):
            rid = r.get("run_id") if isinstance(r, dict) else None
            if not (isinstance(rid, str) and rid):
                report["problems"].append(f"run plan entry {i} has no run_id")
            elif rid in planned:
                report["problems"].append(f"run plan lists run {rid[:12]} more than once")
            elif not (isinstance(r.get("pubkeys"), dict) and isinstance(r.get("anchor_pubkey"), str) and r["anchor_pubkey"]):
                report["problems"].append(f"run plan entry for {rid[:12]} lacks pubkeys or anchor_pubkey")
            else:
                planned[rid] = r
        return planned

    def _authenticate_plan(self, report: Dict[str, Any]) -> bool:
        """True only if the plan in hand is the one an outside log holds AND the log it is read from is pinned.

        The chain of trust: pins (deployments file at a commit) -> registry -> ``anchorOnce`` digest of the
        preregistration document (``prereg=``) -> the study key fingerprint written in that document -> the
        Rekor entries signed by that key. A study key or registry the caller merely supplies authenticates nothing.
        """
        from antelligence.anchoring.anchorer import single_shot_id

        if self.pins is None or self.prereg is None:
            report["unverified"].append("run plan cannot be authenticated without --prereg and pins (deployments file at a commit)")
            return False
        try:
            doc = self.prereg.read_bytes()
            self.metrics.file_bytes += len(doc)
            doc_digest = hashlib.sha256(doc).digest()
            if study_key_in(doc) != self.pins.study_key_sha256:
                report["problems"].append("the preregistration states a different study key than the pins")
                return False
        except (OSError, PinError) as exc:
            report["problems"].append(f"preregistration unusable: {exc}")
            return False
        plan_digest = hashlib.sha256(jcs_bytes(self.plan)).digest()
        prereg_anchored = False
        plan_confirmed: List[bool] = []
        could_not_check = False
        if self.chain is not None:
            try:
                pin_problems = self._chain_pin_problems()
                report["problems"] += [p for p in pin_problems if p not in report["problems"]]
                if not pin_problems:
                    onchain_prereg = self.chain.once_digest(single_shot_id("prereg"))
                    prereg_anchored = onchain_prereg == doc_digest
                    if not prereg_anchored:
                        report["problems"].append("preregistration document does not match the digest anchored on chain")
                    onchain_plan = self.chain.once_digest(single_shot_id("runplan"))
                    plan_confirmed.append(onchain_plan == plan_digest)
                    if onchain_plan != plan_digest:
                        report["problems"].append("run plan does not match the run-plan digest on chain")
            except Exception as exc:
                could_not_check = True
                report["unverified"].append(f"plan check on chain: {type(exc).__name__}: {exc}")
        else:
            report["unverified"].append("preregistration digest not checked on chain: no chain reader")
        if self.rekor is not None and self.study_pem is not None and self.study_pem and not self.pins.check_study_pem(self.study_pem):
            try:
                key = self.rekor.log_key()
                records = self.rekor.entries_for_key(self.study_pem)
                self._rekor_bytes += sum(r.raw_bytes for r in records)
                good = {r.digest for r in records if not check_record(r, self.study_pem, key)}
                plan_confirmed.append(plan_digest.hex() in good)
                if plan_digest.hex() not in good:
                    report["problems"].append("run plan digest is not logged under the study Rekor key")
                # The study key is operator-held, so Rekor cannot stop a SECOND plan from being logged under it;
                # the verifier must see that exactly the preregistration and this plan are there.
                extra = {r.digest for r in records} - {doc_digest.hex(), plan_digest.hex()}
                if extra:
                    report["problems"].append(f"{len(extra)} other digest(s) logged under the study key (second run plan or preregistration)")
            except Exception as exc:
                could_not_check = True
                report["unverified"].append(f"plan check on Rekor: {type(exc).__name__}: {exc}")
        return prereg_anchored and bool(plan_confirmed) and all(plan_confirmed) and not could_not_check

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
            checks["chain"] = _guard(self._chain)(run, tail)
        if "R" in self.conditions:
            checks["rekor"] = _guard(self._rekor)(run, planned, tail)
        if "OTS" in self.conditions:
            checks["ots"] = _guard(self._ots)(run)
        checks["anchor_times"] = _guard(self._times)(run)
        stats = {check_status(c) for c in checks.values()}
        status = "fail" if "fail" in stats else ("unverified" if "unverified" in stats else "pass")
        anchors = {k: checks[k].get("anchors_verified", 0) for k in ("chain", "rekor", "ots") if k in checks}
        tail_n = tail.get("tail")
        if self.require_final_anchor and "fail" not in stats and (tail_n is None or tail_n > 0):
            what = "no anchor tail could be measured" if tail_n is None else f"{tail_n} entries after the last anchor"
            checks["final_anchor"] = _check(False, problems=[f"completed run is not finally anchored: {what}"], tail=tail_n)
            status = "fail"
        return {"ok": status == "pass", "status": status, "entries": len(run.entries),
                "unanchored_tail": tail.get("tail"), "anchors_verified": anchors, "checks": checks}

    def _header(self, run: LoadedRun, rid: str, planned: Dict[str, Any]) -> Dict[str, Any]:
        problems = list(run.problems)
        if run.header:
            if run.run_id != rid:
                problems.append("run_id is not sha256(JCS(header))")
            if "pubkeys" not in planned or run.header.get("pubkeys") != planned["pubkeys"]:
                problems.append("run public keys differ from the anchored run plan")
            if "anchor_pubkey" not in planned or run.header.get("anchor_pubkey") != planned["anchor_pubkey"]:
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
        deployer = self.pins.deployer
        problems += self._chain_pin_problems()
        if problems:                           # reading anchors from an unpinned registry would prove nothing
            return _check(False, problems=problems, anchors_verified=0)
        last = self.chain.last_size(rid)
        if last == 0:
            return _check(False, problems=["run has no anchor on chain"], anchors_verified=0)
        if self.chain.run_owner(rid).lower() != deployer.lower():
            problems.append("run owner is not the deployer")
        sizes = anchor_sizes(last)
        if self.chain.anchor_count(rid) != len(sizes):
            problems.append(f"chain holds {self.chain.anchor_count(rid)} anchors, the 50-grid up to {last} needs {len(sizes)}")
        if n < last:
            problems.append(f"export has {n} entries but the chain anchored size {last} (truncated)")
        verified = 0
        for s in sizes:
            if s > n:
                continue
            onchain = self.chain.root_at(rid, s)
            if onchain == ZERO32:
                problems.append(f"size {s}: no root on chain")
            elif onchain != run.root(s):
                problems.append(f"size {s}: root on chain differs from the root recomputed from the export")
            else:
                verified += 1
        if verified == 0:
            problems.append("0 anchors verified on chain")
        tail["tail"] = n - last if n >= last else 0
        return _check(not problems, problems=problems[:20], last_size=last, tail=tail["tail"], anchors_verified=verified)

    # -- R -------------------------------------------------------------------------------------
    def _rekor(self, run: LoadedRun, planned: Dict[str, Any], tail: Dict[str, Any]) -> Dict[str, Any]:
        n = len(run.entries)
        pem_path = self.dir / f"{run.run_id}.anchor_pub.pem"
        problems: List[str] = []
        if not pem_path.exists():
            return _check(False, problems=["anchor public key PEM missing"], anchors_verified=0)
        pem = pem_path.read_bytes()
        self.metrics.file_bytes += len(pem)
        try:
            fp = fingerprint_of(load_public_pem(pem))
        except (RekorError, ValueError) as exc:
            return _check(False, problems=[f"anchor public key unreadable: {exc}"], anchors_verified=0)
        if fp != planned.get("anchor_pubkey"):
            return _check(False, problems=["anchor PEM does not match the anchored anchor_pubkey fingerprint"], anchors_verified=0)
        log_key = self.rekor.log_key()          # pinned; failure here is "unverified", not a pass
        records = self._rekor_records[run.run_id] = self.rekor.entries_for_key(pem)
        self._rekor_bytes += sum(r.raw_bytes for r in records)
        good: List[Any] = []
        for r in records:
            found = check_record(r, pem, log_key)
            problems += [f"rekor entry {r.uuid[:12]}: {p}" for p in found]
            if not found:
                good.append(r)
        logged = {r.digest for r in good}
        required = {run.digest(s).hex(): s for s in required_sizes(n)}
        acceptable = set(required) | ({run.digest(n).hex()} if n else set())
        for d, s in required.items():
            if d not in logged:
                problems.append(f"size {s}: digest not logged (with a verified proof) under the run's anchor key")
        extra = sorted({r.digest for r in records} - acceptable)
        if extra:
            problems.append(f"{len(extra)} logged digest(s) under the run's anchor key match no prefix of this history (second history, rewrite or truncation)")
        verified = sum(1 for d in required if d in logged)
        if verified == 0:
            problems.append("0 anchors verified on Rekor")
        unverified = self._split_view(run, good, log_key, problems)
        last_logged = max([s for d, s in required.items() if d in logged] or [0])
        tail["tail"] = n - last_logged if run.digest(n).hex() not in logged else 0
        out = _check(not problems and not unverified, problems=problems[:20], rekor_entries=len(records),
                     anchors_verified=verified, required_anchors=len(required), unmatched_entries=len(extra), tail=tail["tail"])
        if unverified and not problems:      # a proven problem always wins over "could not check"
            out.update(unverified=True, error=unverified[0])
        return out

    def _split_view(self, run: LoadedRun, good: List[Any], log_key, problems: List[str]) -> List[str]:
        """Checkpoint consistency: what the log showed at anchor time (``anchors.json``) vs what it shows now.

        Appends proven inconsistencies to ``problems``; returns the reasons the check could not be completed.
        """
        archived: List[Any] = []
        unverified: List[str] = []
        path = self.dir / f"{run.run_id}.anchors.json"
        records: Any = []
        if path.exists():
            raw = path.read_bytes()
            self.metrics.file_bytes += len(raw)
            try:
                records = json.loads(raw)
            except ValueError as exc:
                problems.append(f"anchors.json unreadable: {exc}")
        if not isinstance(records, list):
            problems.append("anchors.json is not a list of anchor records")
            records = []
        by_uuid = {r.uuid: r for r in good}
        n = len(run.entries)
        seen_uuid: set = set()
        for rec in records:
            if not isinstance(rec, dict):
                problems.append("anchors.json holds a record that is not an object")
                continue
            size = rec.get("size")
            # anchors.json is operator-written: authenticate every record against the recomputed history
            if not (isinstance(size, int) and not isinstance(size, bool) and 0 < size <= n):
                problems.append(f"anchors.json record has an invalid size {size!r}")
                continue
            if rec.get("root") != run.root(size).hex() or rec.get("digest") != run.digest(size).hex():
                problems.append(f"size {size}: anchors.json root/digest differ from the root recomputed from the export")
                continue
            uuid = rec.get("rekor_uuid")
            if uuid is None:
                continue
            entry = by_uuid.get(uuid)
            if not isinstance(uuid, str) or uuid in seen_uuid:
                problems.append(f"size {size}: anchors.json names a missing or repeated rekor_uuid")
                continue
            seen_uuid.add(uuid)
            if entry is None:
                problems.append(f"size {size}: anchors.json rekor_uuid is not a verified entry under the run's anchor key")
                continue
            if entry.digest != run.digest(size).hex():
                problems.append(f"size {size}: the Rekor entry named in anchors.json logs a different digest")
                continue
            text = rec.get("rekor_checkpoint")
            if not text:
                unverified.append(f"size {size}: anchors.json record has a rekor_uuid but no rekor_checkpoint archived at anchor time")
                continue
            try:
                cp = verify_checkpoint(text, log_key)
            except RekorError as exc:
                problems.append(f"size {size}: archived checkpoint: {exc}")
                continue
            archived.append(cp)
            # Rekor is sharded: the global logIndex is NOT comparable with a checkpoint. Compare only within the
            # entry's own shard, using the shard-local index from its (verified) inclusion proof.
            if not entry.tree_id or cp.tree_id != entry.tree_id:
                problems.append(f"size {size}: the archived checkpoint is for tree {cp.tree_id!r}, the entry lives in tree {entry.tree_id!r}")
            elif entry.shard_index is None or cp.size <= entry.shard_index:
                problems.append(f"size {size}: the archived checkpoint (size {cp.size}) predates its own entry (shard index {entry.shard_index})")
            # the shard coordinates are REQUIRED in every record that names a Rekor entry
            for field, actual in (("rekor_tree_id", entry.tree_id), ("rekor_shard_index", entry.shard_index)):
                if rec.get(field) in (None, ""):
                    problems.append(f"size {size}: anchors.json record lacks {field}")
                elif rec[field] != actual:
                    problems.append(f"size {size}: anchors.json {field} {rec[field]!r} differs from the log's {actual!r}")
            # The tree size is the ANCHOR-TIME size (the log grows afterwards, so a fresh proof is larger): it must
            # equal the size of the archived, log-signed checkpoint and cannot exceed what the log shows now.
            ts = rec.get("rekor_tree_size")
            if ts in (None, ""):
                problems.append(f"size {size}: anchors.json record lacks rekor_tree_size")
            elif ts != cp.size or (entry.tree_size is not None and ts > entry.tree_size):
                problems.append(f"size {size}: anchors.json rekor_tree_size {ts!r} differs from the archived checkpoint size {cp.size} or exceeds the log's {entry.tree_size!r}")
        # every verified entry the run depends on needs its own archived view
        needed = {run.digest(s).hex() for s in required_sizes(n)}
        for r in good:
            if r.digest in needed and r.uuid not in seen_uuid:
                unverified.append(f"Rekor entry {r.uuid[:12]} has no authenticated anchors.json record with an archived checkpoint")
        if not archived:
            return ["no checkpoint archived at anchor time (anchors.json): a Rekor split view cannot be excluded"] + unverified
        seen = []
        for r in good:
            cp_text = (r.inclusion_proof or {}).get("checkpoint")
            if cp_text:
                seen.append(parse_checkpoint(cp_text))
        found, not_checked = split_view_check(self.rekor, archived, seen)
        problems += found
        return unverified + not_checked

    # -- OTS -----------------------------------------------------------------------------------
    def _ots(self, run: LoadedRun) -> Dict[str, Any]:
        # Without a header source nothing can be CONFIRMED, but a missing proof or a proof for another digest is
        # still provable, so those checks run regardless: a "failed" must never degrade to "unverified" here.
        n = len(run.entries)
        problems: List[str] = []
        unverified: List[str] = []
        verified = 0
        required = required_sizes(n)
        for s in required + ([n] if n and n not in required else []):
            path = self.ots_dir / f"{run.run_id}.{s}.ots"
            if not path.exists():
                if s in required:
                    problems.append(f"size {s}: no .ots proof")
                continue                       # the run-end proof is optional on a mid-run snapshot
            self.metrics.file_bytes += path.stat().st_size
            res = verify_ots(path, run.digest(s), self.header_provider)
            if res.ok:
                verified += 1
            elif res.status == "unverified":
                unverified.append(f"size {s}: {res.error}")
            else:
                problems.append(f"size {s}: {res.error}")
        if not problems and not unverified and verified == 0:
            problems.append("0 anchors verified by OTS")
        out = _check(not problems and not unverified, problems=problems[:20], anchors_verified=verified,
                     unverified_proofs=len(unverified))
        if unverified and not problems:
            out.update(unverified=True, error=unverified[0])
        return out

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
            # integratedTime counts only when a verified signed entry timestamp covers it
            times["R"] = {by_digest[r.digest]: r.integrated_time
                          for r in self._rekor_records.get(run.run_id, []) if r.digest in by_digest and r.time_trusted}
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
            if claim.get("run_id") != run.run_id:
                continue
            covers, made = int(claim["covers_size"]), int(claim["made_at"])
            earliest = [min(t for s, t in by.items() if s >= covers) for by in times.values()
                        if any(s >= covers for s in by)]
            if not earliest:
                problems.append(f"claim {claim.get('id')}: no anchor covers size {covers}")
            elif min(earliest) > made:
                problems.append(f"claim {claim.get('id')}: made at {made}, before the earliest covering anchor ({min(earliest)})")
        untrusted = sorted(
            {"R" for r in self._rekor_records.get(run.run_id, []) if not r.time_trusted}
        )
        return _check(not problems, problems=problems, systems=sorted(times), untrusted_time_systems=untrusted)

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
        """The preregistration document in hand must be the one that was anchored (T4).

        Returns ``{"ok", "failed", "unverified", "checks"}``; ``ok`` needs at least one outside condition to pass.
        """
        from antelligence.anchoring.anchorer import single_shot_id

        ids = single_shot_ids or {"prereg": single_shot_id("prereg"), "runplan": single_shot_id("runplan")}
        doc = Path(prereg_doc).read_bytes()
        self.metrics.file_bytes += len(doc)
        prereg_digest = hashlib.sha256(doc).digest()
        checks: Dict[str, Dict[str, Any]] = {}
        if "OTS" in self.conditions:
            checks["ots"] = _guard(self._study_ots)(prereg_ots, prereg_digest)
        if "C" in self.conditions:
            checks["chain"] = _guard(self._study_chain)(ids, prereg_digest, runplan_digest)
        if "R" in self.conditions:
            checks["rekor"] = _guard(self._study_rekor)(study_pem, prereg_digest, runplan_digest)
        if checks and self.pins is not None:
            checks["document"] = self._study_document(doc)
        statuses = {check_status(c) for c in checks.values()}
        failed = "fail" in statuses
        unverified = "unverified" in statuses or not checks
        return {"ok": not failed and not unverified, "failed": failed, "unverified": unverified, "checks": checks}

    def _study_document(self, doc: bytes) -> Dict[str, Any]:
        """The document in hand must fix the pinned study key (a swapped preregistration cannot keep it)."""
        try:
            ok = study_key_in(doc) == self.pins.study_key_sha256
            return _check(ok, problems=[] if ok else ["the preregistration states a different study key than the pins"])
        except PinError as exc:
            return _check(False, problems=[str(exc)])

    def _study_ots(self, prereg_ots, prereg_digest: bytes) -> Dict[str, Any]:
        if prereg_ots is None:
            return _check(False, problems=["no preregistration .ots proof supplied"], anchors_verified=0)
        self.metrics.file_bytes += Path(prereg_ots).stat().st_size
        res = verify_ots(prereg_ots, prereg_digest, self.header_provider)
        out = _check(res.ok, problems=[res.error] if res.error else [], status=res.status, anchors_verified=int(res.ok))
        if res.status == "unverified":
            out.update(unverified=True, error=res.error)
        return out

    def _study_chain(self, ids, prereg_digest: bytes, runplan_digest: bytes) -> Dict[str, Any]:
        problems = self._chain_pin_problems()
        if problems:
            return _check(False, problems=problems, anchors_verified=0)
        if self.chain.once_digest(ids["prereg"]) != prereg_digest:
            problems.append("preregistration digest on chain differs from this document")
        if self.chain.once_digest(ids["runplan"]) != runplan_digest:
            problems.append("run plan digest on chain differs from this run plan")
        return _check(not problems, problems=problems, anchors_verified=2 - len(problems))

    def _study_rekor(self, study_pem, prereg_digest: bytes, runplan_digest: bytes) -> Dict[str, Any]:
        if study_pem is None:
            return _check(False, problems=["no study public key supplied"], anchors_verified=0)
        if self.pins is None:
            return _check(False, problems=["no pins: the study key is not fixed by the preregistration"], anchors_verified=0)
        reason = self.pins.check_study_pem(study_pem)
        if reason:
            return _check(False, problems=[reason], anchors_verified=0)
        log_key = self.rekor.log_key()
        records = self.rekor.entries_for_key(study_pem)
        self._rekor_bytes += sum(r.raw_bytes for r in records)
        problems: List[str] = []
        good = set()
        for r in records:
            found = check_record(r, study_pem, log_key)
            problems += [f"rekor entry {r.uuid[:12]}: {p}" for p in found]
            if not found:
                good.add(r.digest)
        logged = {r.digest for r in records}
        want = {prereg_digest.hex(), runplan_digest.hex()}
        problems += [f"digest {d[:12]} not logged (with a verified proof)" for d in sorted(want - good)]
        if logged - want:
            problems.append(f"{len(logged - want)} other digest(s) logged under the study key (second preregistration or plan)")
        return _check(not problems, problems=problems, anchors_verified=len(want & good))
