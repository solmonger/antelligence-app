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
| **D** (`BoardLog`) | post entries, but only as the deployer (`postBatch` is deployer-only, like `anchor`; S2b) | rewrite posted entries (contiguous `seq`, `prevHash` link); squat a `runId` | Entries are posted as they are produced; included entries are what workers see. Verifying it is the same recomputation as C against the emitted `Entry` events. D is a worker-visibility mechanism in v4.1; the tamper suite covers B, OTS, R and C. |

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

## Trust anchors are pinned (S2b)

A verifier told "the registry is X, the deployer is Y, the study key is Z" by the party it audits has
verified nothing. `antelligence/anchoring/pins.py` therefore builds a `TrustPins` from public,
operator-independent inputs only:

| Pin | Source |
|---|---|
| `AnchorRegistry` address | `blockchain/deployments/<network>.json`, key `v4_1`, **read at a full 40-hex git commit** (`git show <commit>:<path>`; a branch name is refused because it moves) |
| deployer | the same record |
| code hash (keccak256 of the code the chain serves at the address) | the same record (`code_hash`, written by `deploy-zksync-v41.js` from `eth_getCode` right after deploy) |
| study key fingerprint (sha256 of the SPKI DER) | the line ``Study key fingerprint (sha256 of SPKI DER): `...` `` **inside the preregistration**, whose digest is `anchorOnce`'d on chain |

Anything a caller supplies on top (`--registry`, `--deployer`, `--study-pem`, a chain reader pointed
somewhere) is only *compared* with the pins and fails the verdict on a mismatch. Condition C refuses to
run without pins, and every chain read (per-run, plan, study) first checks address, deployer and code hash.

**Chain of trust for the run plan.** pins -> registry -> `onceDigestOf(prereg)` == sha256(`--prereg` document)
-> the study key written in that document -> the Rekor entries signed by that key. The plan is
`plan_authenticated` only if the preregistration is anchored on the pinned registry **and** the plan digest is
confirmed by every outside log available (chain `onceDigestOf(runplan)`; Rekor under the pinned study key).
Rekor alone cannot root the study key (the key and the entries both come from the operator), so a Rekor-only
or OTS-only verification stays `unverified` for the plan, never `verified`. Because the study key is
operator-held, Rekor cannot stop a *second* plan being logged under it; the verifier requires that exactly the
preregistration digest and the plan digest are logged under the study key.

## Rekor split view (S2b)

Rekor is one operator's log. It could show verifier A one tree and verifier B another; each tree would carry
valid inclusion proofs and validly signed checkpoints. The defence implemented is **checkpoint consistency**:

1. At anchor time the harness archives, per anchor, the signed checkpoint Rekor returned with the new entry
   (`rekor_checkpoint` in `<run_id>.anchors.json`).
2. At audit time the verifier verifies each archived checkpoint under the pinned log key, then collects every
   checkpoint it sees now: the one inside each inclusion proof and the log's current head
   (`GET /api/v1/log`, covering the active tree and frozen shards).
3. Per tree (checkpoint origin) these are sorted by size and each consecutive pair must be linked by an
   RFC 6962 consistency proof (`GET /api/v1/log/proof?firstSize&lastSize&treeID`), verified locally. Two roots
   for one size, or a proof that does not verify, is a `failed` split view. An archived checkpoint must also be
   at least as large as its own entry's index.
4. No archived checkpoint, or an unreachable log, makes the Rekor check `unverified` (never `verified`).

**Residual risk.** This detects a log that shows *this* verifier a history different from the one the harness
saw. It does not detect a log that shows the same fork to the harness and to every verifier (no witness gossip,
no third-party cosigned checkpoints, no monitor comparing independent views), and an operator can archive an old,
small checkpoint to weaken step 3 (steps 1-4 then reduce to the entry's own inclusion proof). Rekor alone is
therefore not a uniqueness guarantee; the chain (write-once roots) is, and Rekor's value here is detecting
rewrites by an operator who is not also the log. Independent witnesses would close the gap; v4.1 does not run any.

## Bitcoin headers are validated, not trusted (S2b)

`BlockstreamHeaderProvider` used to return whatever header an Esplora server sent, so a lying server could
confirm any `.ots` proof by choosing the merkle root. It now requires the header to hash to the block hash the
server named, to meet its own `nBits` target, and to carry real difficulty (target at or below the mainnet
proof-of-work limit and at least `MIN_DIFFICULTY` = 1e12 times harder than difficulty 1, so forging one costs
about 2^32 x 1e12 hashes). Residual: without a Bitcoin node it cannot check that the header is on the best
chain, or recompute difficulty retargets; point `--no-check-bitcoin` off only against a server you trust, or
cross-check the height with a second source.

**Residual risk of `MIN_DIFFICULTY` (S2c).** The floor is a constant, not a measurement. Header PoW is checked against the
header's own `nBits`, not against the true retargeted difficulty at that height, so a lying server can still serve a header
that clears the floor (1e12, about 2^72 hashes: out of reach for most, not for a state or a large miner) with a merkle root
it chose. The floor will also go stale: it must be raised if real difficulty rises far above it, or the margin shrinks.
The header is not shown to be on the best chain. Closing this needs a Bitcoin node or a second independent header source
checked against the same height; v4.1 has neither. A zero or overflowing `nBits` is rejected (`HeaderError`).

## What the verifier checks (`scripts/v41_verify_board.py`)

No private key, no `.env`, no signer import. Inputs: exported JSONL (with `<run_id>.anchors.json`), run plan,
RPC, `--deployments-commit` + `--prereg` (the pins), `.ots` proofs, Rekor URL.

1. **header**: `run_id = sha256(JCS(header))`; header public keys and anchor key equal the plan's.
2. **integrity / signatures**: ids recomputed, `seq` contiguous, `prev_hash` chain, cites earlier, every
   signature valid under the header's keys.
3. **chain (C)**, **rekor (R)**, **ots (OTS)** as above, each against roots recomputed from the export.
4. **anchor times**: per system, times are monotone in size; optional `--claims` (a claim made at time t
   covering size s needs an anchor of size >= s no later than t).
5. **study** (`--prereg`): the preregistration in hand fixes the pinned study key and matches the public
   `.ots`, `onceDigestOf(prereg)` and the digests logged under the study Rekor key.

The report records wall time, bytes read and network calls. The size of the **unanchored tail** (entries
after the last anchor) is reported, not failed: that is exactly the window an operator can still rewrite.

## Fail-closed rules

Verdict is `verified`, `unverified` or `failed`; only `verified` is ok. OTS counts only with a Bitcoin
attestation checked against a block header (calendar promises, unknown attestations and a missing header
source are `unverified`). An enabled outside condition with zero verified anchors fails the run (for n < 50
the run-end anchor is required). B alone is at most `unverified`. Rekor needs a mandatory inclusion proof, an
entry key equal to the searched key, and a checkpoint signed by the log key pinned by sha256
(`REKOR_LOG_KEY_SHA256`, fetched once from `/api/v1/log/publicKey`); `integratedTime` is used only when the
signed entry timestamp verifies. Condition C requires pins (registry, deployer, code hash). The plan must hash to
the anchored run-plan digest through the pinned preregistration (`plan_authenticated`, see above) or the verdict
is capped at `unverified`. Network and RPC errors give `unverified` with the error string.

**Failed dominates.** The verdict is the worst over all checks, files and attestations: a proof with one
provably bad Bitcoin attestation is `invalid` regardless of the order attestations are visited or whether another
header lookup failed first; a garbage `.ots` is `invalid` whatever the parser raises; a missing or wrong-digest
proof fails even when no header source is available (`--no-check-bitcoin` can only leave a *good* proof
unconfirmed); a study-level finding can only make the CLI verdict worse. A proven problem is never replaced by
"could not check".

## Limits, stated plainly

- Entries after the last anchor are not protected. Anchor often, and always at run end.
- Rekor's key is pinned by hash in code; a key rotation needs a code change (`--rekor-log-key-sha256` for private logs).
- OTS gives time, not uniqueness; its detection value is limited to the single published preregistration.
- The verifier trusts the RPC it is pointed at; the pins (address, deployer, code hash) detect a wrong or swapped registry, not a lying node. Cross-check with a second RPC for high-stakes audits.
- A refusal to anchor (operator withholds) is visible only as a missing anchor.

## Tamper suite (`scripts/v41_tamper.py`)

T1 rewrite, T2 truncate (k = 1, 10, 60), T3 equivocation, T4 backdate, at positions early, middle and
before the last anchor, plus an unanchored-tail snapshot, under conditions B, OTS, R, C. The operator in
the suite holds every key and re-anchors wherever the system accepts it. Expected and observed (mocks, and
the real contract on a local hardhat node): B detects none; OTS detects only T4; R detects T1-T4 in the
anchored prefix (audit time); C detects T1-T4 in the anchored prefix (rejected at write time, and also found
at audit); the unanchored tail is detected by nothing. Detection rates carry Wilson intervals
(`antelligence/experiments/stats.py`). The honest control must never be flagged.
