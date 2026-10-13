# v4.1 outside anchors: how a verifier detects a second history

Code: `antelligence/anchoring/` (ots, rekor, chain_modes, anchorer, verify, tamper, mocks, readers),
`scripts/v41_verify_board.py`, `scripts/v41_tamper.py`. Contract: `docs/v4.1/INTERFACES.md` sections 3, 6, 7.

## The problem

The operator holds every key (author keys, anchor key, wallet). A signed, hash-chained board (mode B)
therefore proves nothing against the operator: it can rewrite an entry, re-sign it, rebuild the chain and
hand over a perfectly valid second history. An outside log helps only if it holds something the operator
cannot change after the fact and that the verifier can find without being told where to look.

Each anchor commits to `digest = sha256(runId || uint64_be(size) || root)`. A second history differs from
the first in at least one root, so at the first differing anchor size it has a different digest.

## What each log gives a verifier

| Log | What the operator can do | What it cannot do | How the verifier finds a second history |
|---|---|---|---|
| **B** (signatures, hash chain) | anything | nothing | cannot |
| **OTS** (Bitcoin via calendars) | stamp any digest, any time; choose which `.ots` files to hand over | backdate a stamp; remove a stamp from Bitcoin | A `.ots` proves "this digest existed by block T". Nothing public says which stamps belong to a run, so the operator supplies the stamps for the history it shows. OTS catches a **backdated single document** only: the preregistration `.ots` is pushed to the public PR at lock time, so the verifier holds an independent copy and compares it to the document it is shown (T4). It gives anchor *time*, not uniqueness. |
| **R** (Rekor) | append more entries under the run's anchor key (it holds the key) | delete or edit an entry; hide an entry from search | The run's anchor public key is in the anchored run plan (as a sha256 fingerprint) and the PEM travels with the export. The verifier lists **every** entry under that key (`index/retrieve`, then `GET entries/{uuid}`) and compares the set of logged digests with the digests recomputed from the export it was given. A rewrite, a truncation or a second history leaves digests that match no prefix of the shown history: the log still holds the old ones. Detection is at **audit** time, because Rekor accepts the second history's entries too. |
| **C** (`AnchorRegistry`) | nothing past what the deployer key may write (still all of it) | overwrite an anchored `(runId, size)`, shrink or reorder anchors | The contract is write-once per `(runId, size)`, size-increasing and `prevRoot`-linked, so a second root for an anchored size **reverts at write time**. The verifier reads `lastSize`, `anchorCount` and `rootAt` through `eth_call`, with the registry address and the deployer pinned, and recomputes every root from the export. A shorter export than `lastSize` is a truncation. |
| **D** (`BoardLog`) | post entries | rewrite posted entries (contiguous `seq`, `prevHash` link) | Entries are posted as they are produced; included entries are what workers see. Verifying it is the same recomputation as C against the emitted `Entry` events. D is a worker-visibility mechanism in v4.1; the tamper suite covers B, OTS, R and C. |

### Why Rekor counts as "one history per planned run"

The run plan (anchored once, id `runplan`) fixes the run ids and anchor key fingerprints. For a planned
run the verifier requires: every multiple of 50 up to the export's length is logged under the run's key;
no logged digest is outside the export's prefix digests. A run with one history has exactly
`len(anchor_sizes(n))` entries. A rewrite at position j doubles the entries from the first anchor at or
after j; a truncation leaves the old final digest unmatched; an equivocation leaves the other history's
digests unmatched for **both** verifiers.

### Why the chain counts

`anchorCount(runId)` must equal the number of grid anchors up to `lastSize`; `runOwner` must be the pinned
deployer; each `rootAt` must equal the recomputed root. Because only the deployer can write and a root can
never change, a verifier that trusts the chain (not the operator) sees exactly one root per size. The
operator can, at most, refuse to anchor.

## Rekor key type: ECDSA P-256, not Ed25519

Checked on 2026-10-12 with read-only GETs of the public hashedrekord v0.0.1 schema and `entry.go`:
Rekor verifies an Ed25519 hashedrekord signature **only as Ed25519ph** (RFC 8032 pre-hashed;
`x509.NewSignatureWithOpts(..., options.WithED25519ph())`). The Python `cryptography` library has no
Ed25519ph, so per-run **anchor keys are ECDSA P-256** (INTERFACES section 2 asks for ed25519 for *author*
keys, which are unaffected). The signature is deterministic (RFC 6979) ECDSA-SHA256 over the 72-byte
anchor preimage, passed pre-hashed, so `data.hash.value` is exactly the anchor digest and a retried log
call is an identical entry (HTTP 409, not a duplicate).

INTERFACES keeps `anchor_pubkey` as 32 hex bytes: it is `sha256(SubjectPublicKeyInfo DER)` of the P-256
key. Rekor can only search by the full key, so the public PEM is published as `<run_id>.anchor_pub.pem`
next to the export, and the verifier checks its fingerprint against the anchored run plan before searching.

## What the verifier checks (`scripts/v41_verify_board.py`)

No private key, no `.env`, no signer import. Inputs: exported JSONL, run plan, RPC + pinned registry +
pinned deployer, `.ots` proofs, Rekor URL.

1. **header**: `run_id = sha256(JCS(header))`; header public keys and anchor key equal the plan's.
2. **integrity / signatures**: ids recomputed, `seq` contiguous, `prev_hash` chain, cites earlier, every
   signature valid under the header's keys.
3. **chain (C)**, **rekor (R)**, **ots (OTS)** as above, each against roots recomputed from the export.
4. **anchor times**: per system, times are monotone in size; optional `--claims` (a claim made at time t
   covering size s needs an anchor of size >= s no later than t).
5. **study** (`--prereg`): the preregistration in hand matches the public `.ots`, `onceDigestOf(prereg)`
   and the digests logged under the study Rekor key.

The report records wall time, bytes read and network calls. The size of the **unanchored tail** (entries
after the last anchor) is reported, not failed: that is exactly the window an operator can still rewrite.

## Fail-closed rules

Verdict is `verified`, `unverified` or `failed`; only `verified` is ok. OTS counts only with a Bitcoin
attestation checked against a block header (calendar promises, unknown attestations and a missing header
source are `unverified`). An enabled outside condition with zero verified anchors fails the run (for n < 50
the run-end anchor is required). B alone is at most `unverified`. Rekor needs a mandatory inclusion proof, an
entry key equal to the searched key, and a checkpoint signed by the log key pinned by sha256
(`REKOR_LOG_KEY_SHA256`, fetched once from `/api/v1/log/publicKey`); `integratedTime` is used only when the
signed entry timestamp verifies. Condition C requires a pinned deployer and registry. The plan must hash to
the anchored run-plan digest (`plan_authenticated`) or the verdict is capped at `unverified`. Network and RPC
errors give `unverified` with the error string.

## Limits, stated plainly

- Entries after the last anchor are not protected. Anchor often, and always at run end.
- Rekor's key is pinned by hash in code; a key rotation needs a code change (`--rekor-log-key-sha256` for private logs).
- OTS gives time, not uniqueness; its detection value is limited to the single published preregistration.
- The verifier trusts the RPC it is pointed at; pinning the deployer detects a wrong registry, not a lying node.
- A refusal to anchor (operator withholds) is visible only as a missing anchor.

## Tamper suite (`scripts/v41_tamper.py`)

T1 rewrite, T2 truncate (k = 1, 10, 60), T3 equivocation, T4 backdate, at positions early, middle and
before the last anchor, plus an unanchored-tail snapshot, under conditions B, OTS, R, C. The operator in
the suite holds every key and re-anchors wherever the system accepts it. Expected and observed (mocks, and
the real contract on a local hardhat node): B detects none; OTS detects only T4; R detects T1-T4 in the
anchored prefix (audit time); C detects T1-T4 in the anchored prefix (rejected at write time, and also found
at audit); the unanchored tail is detected by nothing. Detection rates carry Wilson intervals
(`antelligence/experiments/stats.py`). The honest control must never be flagged.
