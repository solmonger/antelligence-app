#!/usr/bin/env python
"""v4.1 outside verifier CLI. Public inputs only; this script never reads or accepts a private key.

  .venv/bin/python scripts/v41_verify_board.py --export-dir runs/ --plan runplan.json \\
      --rpc https://sepolia.era.zksync.dev --registry 0x... --deployer 0x... \\
      --rekor-url https://rekor.sigstore.dev --ots-dir runs/ --conditions OTS,R,C --out report.json

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
    ap.add_argument("--registry", help="pinned AnchorRegistry address (condition C)")
    ap.add_argument("--deployer", help="pinned deployer address (condition C)")
    ap.add_argument("--rekor-url", default="https://rekor.sigstore.dev")
    ap.add_argument("--ots-dir")
    ap.add_argument("--no-check-bitcoin", action="store_true",
                    help="do NOT fetch Bitcoin headers; OTS proofs then stay unverified (fail closed)")
    ap.add_argument("--rekor-log-key-sha256", help="override the pinned Rekor log key (sha256 of SPKI DER) for a private Rekor")
    ap.add_argument("--claims")
    ap.add_argument("--prereg", help="preregistration document (checks the study-level anchors)")
    ap.add_argument("--prereg-ots", help="public .ots proof of the preregistration")
    ap.add_argument("--study-pem", help="public PEM of the study Rekor key")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    conditions = {c.strip().upper() for c in args.conditions.split(",") if c.strip()}
    chain = rekor = header_provider = None
    if "C" in conditions:
        if not (args.rpc and args.registry and args.deployer):
            ap.error("condition C needs --rpc, --registry and --deployer")
        from web3 import Web3

        from antelligence.anchoring.readers import Web3ChainReader

        chain = Web3ChainReader(Web3(Web3.HTTPProvider(args.rpc)), args.registry, args.deployer)
    if "R" in conditions:
        from antelligence.anchoring.rekor import HttpRekorTransport, RekorClient

        from antelligence.anchoring.rekor import REKOR_LOG_KEY_SHA256

        rekor = RekorClient(HttpRekorTransport(args.rekor_url), args.rekor_log_key_sha256 or REKOR_LOG_KEY_SHA256)
    if "OTS" in conditions and not args.no_check_bitcoin:
        from antelligence.anchoring.ots import BlockstreamHeaderProvider

        header_provider = BlockstreamHeaderProvider()

    plan = json.loads(Path(args.plan).read_text())
    claims = json.loads(Path(args.claims).read_text()) if args.claims else []
    verifier = OutsideVerifier(
        args.export_dir, plan, chain=chain, rekor=rekor, ots_dir=args.ots_dir, header_provider=header_provider,
        conditions=conditions, claims=claims, pinned_deployer=args.deployer, pinned_registry=args.registry,
        study_pem=Path(args.study_pem).read_bytes() if args.study_pem else None,
    )
    report = verifier.verify()
    if args.prereg:
        import hashlib

        from antelligence.kernel.canonical import jcs_bytes

        study_pem = Path(args.study_pem).read_bytes() if args.study_pem else None
        report["study"] = verifier.verify_study(
            args.prereg, hashlib.sha256(jcs_bytes(plan)).digest(), prereg_ots=args.prereg_ots, study_pem=study_pem,
        )
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
