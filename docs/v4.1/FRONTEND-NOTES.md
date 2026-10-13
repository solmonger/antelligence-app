# v4.1 board and anchors for the 3D viewport (notes for Kashyap, PR #22)

What a v4.1 run leaves on disk, what each file holds, and how to read it. Contract: `docs/v4.1/INTERFACES.md`
(sections 2, 3, 4, 7). Producer: `antelligence/experiments/v41_harness.py` (`scripts/v41_run.py`).

## Where the files are

One folder per run: `<out_dir>/runs/<run_id>/` (`run_id` is 64 lowercase hex).

| File | Content |
|---|---|
| `<run_id>.jsonl` | the board export. Line 1 is the run header, then one entry per line in `seq` order |
| `<run_id>.anchors.json` | list of anchor records (only for anchored arms: SW-B and FW) |
| `decisions.jsonl` | the per-decision log, one JSON object per squad-round (latency, router pick, deferral, cites) |
| `run_log.json` | per-run numbers: wall time, ticks to 50% coverage, queen calls, board size, anchors, mode D latency |
| `cell.json` | the cell record (docs/v4.1/CELLS.md): coverage, off-target, defer rate, citations, ... |
| `bundle.json` | the content-addressed provenance bundle (`bundle_hash`; `extra.cell`, `extra.anchors`, `extra.board_root`) |

`SW-noBoard` has no shared board: its `.jsonl` holds only the queen's private log (task, route, knowledge, alarm) and
`run_log.shared_store` is `false`. Do not draw squad evidence or decisions for that arm. `S` and `S-prime` write a cell and
a bundle but no board.

## Board export (`<run_id>.jsonl`)

Every line is RFC 8785 canonical JSON, integers only.

Line 1, the header:
`{"study","arm","case","seed","split","world_cfg_sha256","harness_cfg_sha256","pubkeys":{author: hex32},"anchor_pubkey","code_sha"}`.
`arm` is `<arm>:calibration` for Phase 0 calibration runs. The case id is public (MSD) but never goes on chain.

Lines 2..n, one entry:

```json
{"run_id":"<64 hex>","seq":12,"type":"decision","author":{"agent":"squad-3","model":"<served model>","rev":"<revision>"},
 "cites":["<entry id>"],"payload":{...},"tick":35,"prev_hash":"<id of seq-1>","id":"<64 hex>","sig":"<128 hex>"}
```

`type` is one of `task knowledge evidence route decision edit alarm`. `author.agent` is `perception`, `queen`, `harness`,
`router` or `squad-0` .. `squad-7`. `sig` is `null` only on mode A boards (the queen's private log in the board-off arm).

Fields you will want for the viewport (all in the crop grid: 2 mm voxels, origin at the crop corner; `centroid_dmm = voxel * 20`):

| Type | Draw it as | Key payload fields |
|---|---|---|
| `knowledge` (tick 0, author `perception`) | a perceived region | `region_id`, `centroid_dmm`, `p_tumour_ppm`, `volume_mm3`, `unc_ppm` |
| `evidence` (every round, one per squad) | the squad's position and what it saw | `squad`, `pos_vox`, `look_p_ppm` (local look), `p_tumour_ppm` (perception), `dose_ug`, `released_ug` |
| `decision` (every round, one per squad) | the squad's choice | `squad`, `round`, `target` (`A`-`F` candidate, `S` stay, `X` explore), `release`, `amount_level` 1-5, `backend`, `router_pick`, `deferred`, `brief_ids` |
| `route` (queen step) | queen assignment | `squad`, `region_ids`, `priority`, `brief_ids` |
| `task` (queen step) | the current objective | `objective`, `until_tick` |
| `alarm` | a warning marker | `kind` (`defer_rate`, `timeout`, `identity`, `other`), `note` |

Arrows worth drawing: `cites` (entry to the entries it relied on), `brief_ids` (the read receipt: what the squad had in
front of it when it decided), `prev_hash` (the chain). A decision whose `cites` is empty used d1, which cites nothing; its
receipt is `brief_ids`. Units: `_ppm` is probability times 10^6, `_dmm` is 0.1 mm, `_ug` is micro-units of total drug.

Squad positions over time come from the `evidence` entries (one per squad per decision round, every 5 ticks, 24 rounds).
Positions between rounds are not logged; interpolate along the shortest path if you need smooth motion.

Byzantine arms (`SW-Byz-local`, `SW-Byz-chain`): two squads also post extra `evidence` with `look_p_ppm >= 900000` at places
perception scored below 0.1. Their entries are valid and signed; they stop appearing in other squads' `brief_ids` once the
author's reputation is under 0.3. The reputation trail is in `run_log.json` under `reputation` (author to ppm).

## Anchors (`<run_id>.anchors.json`)

A list, one record per anchor point: after entries 50, 100, 150, ... and at run end (a final size that is a multiple of 50
is anchored once).

```json
{"size":150,"root":"<merkle root>","digest":"<72-byte preimage sha256>","ots_path":"<run_id>.150.ots",
 "rekor_uuid":"...","rekor_log_index":123,"chain_tx":"0x..."}
```

`root` is the RFC 6962 Merkle root over entry ids `[0:size]`; `digest = sha256(runId || uint64_be(size) || root)` is what
OpenTimestamps stamps and Rekor logs. `chain_tx` appears only after the L2 confirms (mode C is asynchronous), so a fresh
file may lack it. Verifying an anchor against its entries is `scripts/v41_verify_board.py` (public inputs only).

Events to animate, in order, for one run: `knowledge` entries at tick 0, then per queen step (ticks 0, 30, 60, 90 by default)
a `task` and eight `route` entries, then per round eight `evidence` and eight `decision` entries; an anchor appears whenever
the entry count crosses a multiple of 50, and once at the end. `ots_path`, `rekor_uuid` and `chain_tx` fill in over time.

## Endpoints

There is **no v4.1 board endpoint in this branch**. What exists is the engine API in `antelligence/api/app.py` (prefix
`/engine`, local-only): `GET /runs/{run_id}`, `/runs/{run_id}/events`, `/runs/{run_id}/frames`, `POST /outbox/drain`. Those serve
the older engine runs, not v4.1 runs.

For the viewport, read the files above: serve `<out_dir>/runs/<run_id>/` as static files, or add a read-only route that
returns the same bytes:

* `GET /engine/v41/runs/{run_id}/board` -> `<run_id>.jsonl` (`application/x-ndjson`)
* `GET /engine/v41/runs/{run_id}/anchors` -> `<run_id>.anchors.json`
* `GET /engine/v41/runs/{run_id}/cell` -> `cell.json`

(proposed shape, not implemented; the lead decides before merge). The outbox bundle (`bundle.json`) can be published through the
existing `ProvenanceOutbox` (`LocalFilePublisher`) without a new endpoint.

Do not display `decisions.jsonl` fields `counterfactual`, `audit` or `sprime` in a public view: they exist only in calibration
and audit logs and `sprime` is derived from the truth mask (development cases only).
