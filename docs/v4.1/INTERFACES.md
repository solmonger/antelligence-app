# Antelligence v4.1 — interfaces

This file is the contract that L3 (board), S1 (contracts and chain client), S2 (outside verifier),
S3 (workers) and S4 (queen) build against in parallel. Change it only through the lead, and log the
change. Every rule here has a test vector in `tests/v4_1/vectors/interfaces.json` (owned by L3).

## 1. Canonical JSON

- **RFC 8785 (JCS).** Keys are sorted by UTF-16 code unit, there is no whitespace, strings are UTF-8,
  and only the escapes JSON requires are used.
- **No floats in any board entry.** Every number in an entry is an integer in the range
  [-(2^53-1), 2^53-1]. Under that rule, JCS equals `antelligence.kernel.canonical.canonical_json`
  for ASCII keys, and the encoders on the Python, JS and Solidity sides agree byte for byte.
  The encoder rejects floats with a `CanonicalError`.
- **Fixed-point units** (the suffix is part of the field name):

| Suffix | Meaning | Example |
|---|---|---|
| `_ppm` | probability × 10^6, 0…1,000,000 | `p_tumour_ppm: 873000` |
| `_dmm` | length in 0.1 mm | `centroid_dmm: [1204, 873, 455]` |
| `_mm3` | volume in mm³ (integer) | `volume_mm3: 5120` |
| `_ug`  | drug mass in micro-units (1 unit = total drug / 10^6) | `amount_ug: 125000` |
| `_ms`  | milliseconds | `latency_ms: 412` |
| none   | plain integer: tick, seq, voxel index, count | `tick: 35` |

## 2. Entry

```json
{
  "run_id":   "<64 hex>",          // sha256(JCS(run header)), see §6
  "seq":      0,                   // 0,1,2,… contiguous per run
  "type":     "task|knowledge|evidence|route|decision|edit|alarm",
  "author":   {"agent": "squad-3|queen|perception|harness|router", "model": "<id or none>", "rev": "<revision or none>"},
  "cites":    ["<entry id>", …],   // sorted, unique; [] allowed
  "payload":  { … },               // typed, §4
  "tick":     35,
  "prev_hash":"<64 hex>",          // id of entry seq-1; 64 zeros for seq 0
  "id":       "<64 hex>",
  "sig":      "<128 hex> | null"   // mode B: ed25519; mode A: null
}
```

- **`id`** = lowercase hex sha256 of the JCS encoding of the entry **without** the `id` and `sig` keys.
- **`sig`** = ed25519 signature by the author's per-run key over the **32 raw bytes** of `id`
  (not over its hex text). Encoded as lowercase hex.
- **Author keys:** each author in a run (`perception`, `queen`, `harness`, `router`, `squad-0` … `squad-7`)
  has its own fresh ed25519 key for that run. The public keys are listed in the run header (§6).
  None of them is the wallet key.

## 3. Merkle root and anchor digest

- **Leaves:** entry ids `[0:size]`, each taken as 32 raw bytes.
- **Root:** the RFC 6962 / RFC 9162 Merkle Tree Hash:
  - `MTH({}) = sha256("")`;
  - `leaf(d) = sha256(0x00 ‖ d)`;
  - `node(l, r) = sha256(0x01 ‖ l ‖ r)`;
  - split at the largest power of two smaller than n.
- **Anchor digest:** `sha256(runId(32 bytes) ‖ uint64_be(size) ‖ root(32 bytes))`, 72 bytes in.
  This exact 32-byte value is what OTS stamps and Rekor logs (as a `hashedrekord` over the digest).
  The chain stores `(runId, size, root)` directly, and anyone can recompute the digest from it.
- **`prevRoot`** for `AnchorRegistry.anchor` is the root of the previous anchor for the run.
  For the first anchor it is `bytes32(0)`.
- **Anchor points:** after entries 50, 100, 150 … (size a multiple of 50), and at run end (final size).
  If the final size is a multiple of 50, the run end is the same anchor and is not repeated.
- **Single-shot ids** for `anchorOnce`: `keccak256("v4.1|prereg")` and `keccak256("v4.1|runplan")`.
  The digest is the sha256 of the document bytes (prereg) or of JCS(run plan) (runplan).

## 4. Typed payloads

Every payload has exactly the keys listed. Extra keys are a validation error.

| Type | Payload keys |
|---|---|
| `task` | `objective` (string ≤ 280 chars), `squads` (list of int), `until_tick` (int) |
| `knowledge` | `region_id` (int), `centroid_dmm` [x,y,z], `cls` ("WT"/"TC"/"ET"), `p_tumour_ppm`, `volume_mm3`, `unc_ppm`, `source` ("perception"/"local_look"), `prov` (string: model id + revision) |
| `evidence` | `squad` (int), `pos_vox` [x,y,z], `look_p_ppm` (local-look P(tumour)), `p_tumour_ppm` (perception at pos), `dose_ug` (local cumulative dose), `released_ug` |
| `route` | `squad` (int), `region_ids` (list, priority order), `priority` (1–5), `brief_ids` (list of entry ids, 6–12 or all if fewer) |
| `decision` | `squad`, `round`, `target` ("A"–"F","S","X"), `release` (0/1), `amount_level` (1–5), `p_target_ppm`, `p_release_ppm`, `p_amount_ppm`, `backend` (string), `deferred` (0/1), `router_pick` (string), `brief_ids` (list: the read receipt) |
| `edit` | `round` (int), `patch_sha256` (hex), `keys` (list of strings), `prediction` {`metric`: string, `delta_ppm`: int}, `kept` (0/1/null) |
| `alarm` | `kind` ("defer_rate"/"timeout"/"identity"/"other"), `value_ppm` (int), `note` (string ≤ 140) |

**On-chain (mode D) compact encoding** of an entry: `abi.encode(uint64 seq, uint8 typeCode,
bytes32 id, bytes32 prevHash, bytes payloadCompact)`. `typeCode`: task 0, knowledge 1, evidence 2,
route 3, decision 4, edit 5, alarm 6. `payloadCompact` is `abi.encode` of the integer fields of the
payload in the table order. Strings and lists of ids are **not** posted, because free text never goes
on-chain; the `id` commits to them.

## 5. Decision request and reply (S3 ↔ harness)

One call per squad-round answers all three questions.

**Request** (`antelligence.kernel.decision.DecisionRequest`, the existing type, extended only by adding fields):
- `state`: a string rendered from the state fields in goal §4.2 plus the brief. It never carries the truth mask.
- `questions`:
  - `target`: `choice`, options `A`…`F` (only the candidates present) plus `S`, `X`;
  - `release`: `noul`;
  - `amount`: `score`, options `1`…`5`.
- `images` (optional): the 64×64 FLAIR and T1c patches, for local look (d1 only).
- `brief_ids`: the entry ids in the brief, in brief order.

**Reply** (`DecisionResponse`):
- `answers[q].probabilities` for every label. Choice options are always in fixed label order;
  they are never shuffled in v4.1, because ties go to the earlier label.
- `cites`: the entry ids the worker says it used (SLM, FW). For d1 it is `[]`, and the read receipt
  is `brief_ids`.
- `identity`: `{served_model, revision, runtime, runtime_version}`. A mismatch against the pinned
  value raises `DecisionError`.
- `timing`: `{server_ms, network_ms, queue_ms}`; `usage`: `{tokens_in, tokens_out, cost_usd}`.
- `prob_method`: `"logprobs"` (vLLM/llama.cpp), `"system_one"` (d1) or `"verbalised"` (FW via Nous).

**Decision rule** (harness): `target` is the argmax, ties to the earlier label in the order A…F, S, X;
release when P(yes) ≥ 0.5; `amount` is the argmax level.

## 6. Run header and run plan

**Run header** (the first thing exported, not an entry):
`{"study":"v4.1","arm":…,"case":…,"seed":…,"split":"dev|eval","world_cfg_sha256":…,"harness_cfg_sha256":…,"pubkeys":{author: hex32,…},"anchor_pubkey":hex32,"code_sha":…}`.

`run_id = sha256(JCS(header))`. The case id is public (MSD), but it never goes on-chain; only the
`run_id` does.

**Run plan:** `{"study":"v4.1","runs":[{"run_id":…,"pubkeys":…,"anchor_pubkey":…},…]}`, sorted by
`run_id`, and anchored once with id `runplan`.

## 7. Export

`<run_id>.jsonl`: line 1 is the header, then one entry per line in `seq` order, each line in JCS.
`<run_id>.anchors.json`: a list of `{size, root, digest, ots_path?, rekor_uuid?, rekor_log_index?,
chain_tx?, l2_block?, l1_batch?}`.
