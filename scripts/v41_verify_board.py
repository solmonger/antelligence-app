#!/usr/bin/env python
"""v4.1 outside verifier CLI. Public inputs only; this script never reads or accepts a private key.

  .venv/bin/python scripts/v41_verify_board.py --export-dir runs/ --plan runplan.json \\
      --rpc https://sepolia.era.zksync.dev --deployments-commit <40-hex commit> --prereg PREREGISTRATION.md \\
      --rekor-url https://rekor.sigstore.dev --ots-dir runs/ --conditions OTS,R,C --out report.json

Trust anchors are PINNED, never taken from the operator: the AnchorRegistry address, deployer and code hash are read
from blockchain/deployments/<network>.json at --deployments-commit (a full commit id in --repo), and the study key
fingerprint from the preregistration given as --prereg, whose digest is anchored on chain. --registry, --deployer
and --study-pem are optional assertions; any that differ from the pins fail the verdict. Without --prereg and
pins the run plan is unauthenticated and the verdict is at most "unverified".

Exit code 0 only for verdict "verified"; 1 for "failed"; 3 for "unverified" (could not reach a verdict,
including B-only, unauthenticated plan, no Bitcoin headers); 2 on a usage error.
Per run, <export-dir> holds <run_id>.jsonl and (for R) <run_id>.anchor_pub.pem; the OTS proofs are
<run_id>.<size>.ots in --ots-dir. Optional --claims is a JSON list of
{"id", "run_id", "covers_size", "made_at"} (unix seconds): each claim must postdate the earliest anchor covering it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.anchoring.verify import OutsideVerifier  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--plan", required=True, help="anchored run plan JSON: {study, runs:[{run_id, pubkeys, anchor_pubkey}]}")
    ap.add_argument("--conditions", default="B", help="comma list from B,OTS,R,C")
    ap.add_argument("--rpc", help="chain RPC URL (condition C)")
    ap.add_argument("--repo", default=str(REPO), help="git repository holding the deployments file")
    ap.add_argument("--deployments-commit", help="full 40-hex commit whose deployments file pins the registry (condition C)")
    ap.add_argument("--deployments-file", default=None, help="path inside the repo (default blockchain/deployments/zksync-era-sepolia.json)")
    ap.add_argument("--registry", help="assert the AnchorRegistry address (must equal the pin)")
    ap.add_argument("--deployer", help="assert the deployer address (must equal the pin)")
    ap.add_argument("--rekor-url", default="https://rekor.sigstore.dev")
    ap.add_argument("--ots-dir")
    ap.add_argument("--no-check-bitcoin", action="store_true",
                    help="do NOT fetch Bitcoin headers; OTS proofs then stay unverified (fail closed)")
    ap.add_argument("--rekor-log-key-sha256", help="override the pinned Rekor log key (sha256 of SPKI DER) for a private Rekor")
    ap.add_argument("--claims")
    ap.add_argument("--prereg", help="preregistration document: fixes the study key, authenticates the run plan, checks the study-level anchors")
    ap.add_argument("--prereg-ots", help="public .ots proof of the preregistration")
    ap.add_argument("--study-pem", help="public PEM of the study Rekor key (must be the key fixed in the preregistration)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    conditions = {c.strip().upper() for c in args.conditions.split(",") if c.strip()}
    chain = rekor = header_provider = pins = None
    if "C" in conditions and not (args.rpc and args.deployments_commit and args.prereg):
        ap.error("condition C needs --rpc, --deployments-commit and --prereg (trust anchors are pinned from public inputs)")
    if args.deployments_commit or args.prereg:
        from antelligence.anchoring import pins as pinmod

        if not (args.deployments_commit and args.prereg):
            ap.error("--deployments-commit and --prereg go together: the pins need both")
        try:
            pins = pinmod.pins_from_repo(args.repo, args.deployments_commit, args.prereg,
                                         args.deployments_file or pinmod.DEFAULT_DEPLOYMENTS)
        except (pinmod.PinError, OSError, ValueError) as exc:
            ap.error(f"cannot build the pins: {exc}")
    if args.rpc and pins is not None:
        from web3 import Web3

        from antelligence.anchoring.readers import Web3ChainReader

        chain = Web3ChainReader(Web3(Web3.HTTPProvider(args.rpc)), pins.registry, pins.deployer)
    if "R" in conditions:
        from antelligence.anchoring.rekor import REKOR_LOG_KEY_SHA256, HttpRekorTransport, RekorClient

        rekor = RekorClient(HttpRekorTransport(args.rekor_url), args.rekor_log_key_sha256 or REKOR_LOG_KEY_SHA256)
    if "OTS" in conditions and not args.no_check_bitcoin:
        from antelligence.anchoring.ots import BlockstreamHeaderProvider

        header_provider = BlockstreamHeaderProvider()

    plan = json.loads(Path(args.plan).read_text())
    claims = json.loads(Path(args.claims).read_text()) if args.claims else []
    study_pem = Path(args.study_pem).read_bytes() if args.study_pem else None
    verifier = OutsideVerifier(
        args.export_dir, plan, chain=chain, rekor=rekor, ots_dir=args.ots_dir, header_provider=header_provider,
        conditions=conditions, claims=claims, pins=pins, prereg=args.prereg,
        pinned_deployer=args.deployer, pinned_registry=args.registry, study_pem=study_pem,
    )
    report = verifier.verify()
    if args.prereg:
        import hashlib

        from antelligence.kernel.canonical import jcs_bytes

        report["study"] = verifier.verify_study(
            args.prereg, hashlib.sha256(jcs_bytes(plan)).digest(), prereg_ots=args.prereg_ots, study_pem=study_pem,
        )
        # failed dominates unverified dominates verified: a study-level finding can only make the verdict worse
        if report["study"]["failed"]:
            report["verdict"] = "failed"
        elif report["study"]["unverified"] and report["verdict"] == "verified":
            report["verdict"] = "unverified"
        report["ok"] = report["verdict"] == "verified"
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    return {"verified": 0, "failed": 1}.get(report["verdict"], 3)


if __name__ == "__main__":
    sys.exit(main())
