#!/usr/bin/env python3
"""Antelligence Leaderboard — Track and rank simulation experiences.

Maintains a local SQLite index of submitted experiences and queries
the chain for attestation/verification status.

Subcommands:
  record  --metrics FILE --tx-hash HASH   Record a submission locally
  show    [--verified-only]               Display leaderboard
  sync                                    Update attestation counts from chain
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

DB_PATH = Path(__file__).parent.parent / "data" / "leaderboard.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            run_hash TEXT PRIMARY KEY,
            ipfs_cid TEXT NOT NULL,
            data_hash TEXT NOT NULL,
            score INTEGER NOT NULL,
            kill_rate REAL,
            strategy TEXT,
            n_nanobots INTEGER,
            tumor_radius REAL,
            seed INTEGER,
            steps INTEGER,
            tx_hash TEXT,
            block_number INTEGER,
            attestations INTEGER DEFAULT 0,
            verified INTEGER DEFAULT 0,
            submitted_at TEXT NOT NULL,
            synced_at TEXT
        )
    """)
    conn.commit()
    return conn


def cmd_record(args):
    """Record a local submission after on-chain TX."""
    path = Path(args.metrics)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    metrics = json.loads(path.read_text())

    # Compute hashes (same logic as antelligence_cli.py)
    from scripts.antelligence_cli import hash_metrics_file
    submission = hash_metrics_file(args.metrics)

    conn = init_db()
    try:
        conn.execute("""
            INSERT INTO submissions
            (run_hash, ipfs_cid, data_hash, score, kill_rate, strategy,
             n_nanobots, tumor_radius, seed, steps, tx_hash, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            submission["run_hash"],
            submission["ipfs_cid"],
            submission["data_hash"],
            submission["score"],
            metrics.get("kill_rate"),
            metrics.get("strategy_type", "pheromone-guided"),
            metrics.get("n_nanobots"),
            metrics.get("tumor_radius"),
            metrics.get("seed"),
            metrics.get("steps"),
            args.tx_hash,
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
        print(f"Recorded: {submission['run_hash'][:20]}... score={submission['score']}")
    except sqlite3.IntegrityError:
        print(f"Already recorded: {submission['run_hash'][:20]}...")
    conn.close()


def cmd_show(args):
    """Display the leaderboard."""
    conn = init_db()
    where = "WHERE verified = 1" if args.verified_only else ""
    rows = conn.execute(f"""
        SELECT run_hash, score, kill_rate, strategy, n_nanobots, seed,
               attestations, verified, tx_hash, submitted_at
        FROM submissions
        {where}
        ORDER BY score DESC
        LIMIT 20
    """).fetchall()
    conn.close()

    if not rows:
        print("No submissions yet. Use 'record' after submitting.")
        return

    print(f"{'Rank':>4}  {'Score':>6}  {'Kill%':>6}  {'Strategy':>18}  {'Bots':>4}  {'Att':>3}  {'V':>1}  {'Run Hash':>20}")
    print("-" * 80)
    for i, r in enumerate(rows, 1):
        v = "Y" if r["verified"] else "N"
        kr = f"{r['kill_rate']*100:.1f}%" if r["kill_rate"] else "?"
        print(f"{i:>4}  {r['score']:>6}  {kr:>6}  {r['strategy'] or '?':>18}  {r['n_nanobots'] or '?':>4}  {r['attestations']:>3}  {v:>1}  {r['run_hash'][:20]}...")

    print(f"\n{len(rows)} submissions. BaseScan: https://sepolia.basescan.org/address/0x22ECc5e4ddcCbAa44f508480e09eBD2640Dcd4e9")


def cmd_sync(args):
    """Sync attestation counts from chain."""
    from web3 import Web3
    from eth_account import Account

    rpc = os.getenv("BASE_SEPOLIA_RPC_URL")
    if not rpc:
        print("ERROR: BASE_SEPOLIA_RPC_URL not set", file=sys.stderr)
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(rpc))
    abi_path = Path(__file__).parent.parent / "blockchain" / "artifacts" / "contracts" / "ExperienceRegistry.sol" / "ExperienceRegistry.json"
    with open(abi_path) as f:
        abi = json.load(f)["abi"]
    registry = w3.eth.contract(
        address=w3.to_checksum_address("0x22ECc5e4ddcCbAa44f508480e09eBD2640Dcd4e9"),
        abi=abi,
    )

    conn = init_db()
    rows = conn.execute("SELECT run_hash FROM submissions").fetchall()

    updated = 0
    for row in rows:
        run_hash_bytes = w3.to_bytes(hexstr=row["run_hash"])
        exp = registry.functions.experiences(run_hash_bytes).call()
        attestations = exp[6]
        verified = 1 if exp[7] else 0

        conn.execute("""
            UPDATE submissions
            SET attestations = ?, verified = ?, synced_at = ?
            WHERE run_hash = ?
        """, (attestations, verified, datetime.now(timezone.utc).isoformat(), row["run_hash"]))
        updated += 1

    conn.commit()
    conn.close()
    print(f"Synced {updated} submissions from chain.")


def main():
    parser = argparse.ArgumentParser(description="Antelligence leaderboard")
    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record", help="Record a submission")
    p_record.add_argument("--metrics", required=True, help="Metrics JSON file")
    p_record.add_argument("--tx-hash", required=True, help="Transaction hash")
    p_record.set_defaults(func=cmd_record)

    p_show = sub.add_parser("show", help="Show leaderboard")
    p_show.add_argument("--verified-only", action="store_true")
    p_show.set_defaults(func=cmd_show)

    p_sync = sub.add_parser("sync", help="Sync from chain")
    p_sync.set_defaults(func=cmd_sync)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
