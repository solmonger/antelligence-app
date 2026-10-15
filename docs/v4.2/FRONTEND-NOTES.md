# Frontend notes: colony run bundle (v4.2)

For: Kashyap, frontend owner.
Status: docs only. Written 2026-10-14.

## Sources

- Spec: `docs/v4.2/INTERFACES.md`. Sections 2 (marks), 3 (messages), 4 (need), 8 (name cards), 9 (notebooks), 12 (bundle), 14 (K1 notes).
- Real example: `/Volumes/WD_BLACK/antelligence-v4.2/runs/verdict1/S-talk/BRATS_016/seed3/bundle.json`.
  This is a development case (`"split": "dev"`), arm `S-talk`, seed 3, 120 ticks.
- Every field below is either in INTERFACES.md or was seen in that file. Anything else is in the Gaps section.
- Values are copied from the file. No MRI arrays and no voxel truth are shown here.

## Where the data sits

- Top level keys: `arm`, `case_id`, `extra`, `metrics`, `seed`, `split`.
- Colony data is under `extra.colony`.
- `extra.colony` keys: `action_log`, `arm`, `cards_after`, `cards_before`, `counters`, `marks`, `messages`, `metrics`, `need_per_zone`, `params`, `reading_logs`, `region_status`, `turns_skipped`.
- `metrics` appears twice with the same values.
- Keys are strings. Ants are `"0"` to `"7"` in `reading_logs`. Cards use `"ant-00"` to `"ant-07"`.

## (a) Marks

Source list: `extra.colony.marks`. One row per deposit, in time order.

Exact keys: `subtype` (always `"mark"`), `mark_kind`, `ant` (int, the author), `pos_vox` (`[x, y, z]`, ints), `amount_milli` (int), `reading_ppm` (int, 0 if no reading), `tick` (int).

Kind names: `dosed`, `dose_ok`, `slow_uptake`, `washout`, `tumour_here`, `not_tumour`, `need_more`.

Real rows:

```
{"amount_milli": 1000, "ant": 7, "mark_kind": "tumour_here", "pos_vox": [43, 40, 55], "reading_ppm": 704522, "subtype": "mark", "tick": 0}
{"amount_milli": 500, "ant": 7, "mark_kind": "dosed", "pos_vox": [43, 40, 55], "reading_ppm": 0, "subtype": "mark", "tick": 0}
```

Strength over time is not stored. Compute it from the rows:

- Per author, take that author's latest deposit within 4 mm of the query voxel.
- Strength is the sum of `amount x 0.5^((t - tick) / half_life)`.
- Half-lives are in `params.half_life`. The confirm radius is `params.confirm_radius_mm` (4.0).
- Strong means at least 2 authors and strength of at least 1.0 (INTERFACES 2.3).

Authors: each mark has one `ant`. The authors at a spot are the `ant` values of marks near it.

Decision rows in `action_log` also list their own emits in `params.emit`.

## (b) Messages

Source list: `extra.colony.messages`. Message dicts in send order.

Exact keys: `msg_id`, `type` (`"ask"` or `"answer"`), `from_ant`, `to_ant`, `tick`, `about` (`{"pos_vox", "region_id"}`, exactly one set), `question` (`"is_tumour"` in this file), `sender_reading` (float), `reply_to` (int or null), `answer` (`"yes"`, `"no"`, `"unknown"`, or null), `answerer_reading` (float or null), `note` (string, up to 120 characters).

Real ask (point-addressed):

```
{"about": {"pos_vox": [18, 6, 9], "region_id": null}, "answer": null, "answerer_reading": null, "from_ant": 0, "msg_id": 1, "note": "", "question": "is_tumour", "reply_to": null, "sender_reading": 0.429612, "tick": 0, "to_ant": 1, "type": "ask"}
```

Real answer:

```
{"about": {"pos_vox": [9, 46, 44], "region_id": null}, "answer": "yes", "answerer_reading": 0.804894, "from_ant": 5, "msg_id": 40, "note": "", "question": "is_tumour", "reply_to": 36, "sender_reading": 0.804894, "tick": 36, "to_ant": 4, "type": "answer"}
```

Real zone-addressed answer (msg 62): `"about": {"pos_vox": null, "region_id": 3}` with `"answer": "yes"`.

Notes:

- Answers usually come one tick after the ask. Example: ask at tick 0, answer at tick 1.
- `msg_id` runs 1 to 126 in this file. That is 63 asks and 63 answers, which matches `counters`.
- Every `note` is `""` in this file.
- Refused sends are counted in `counters` only. This file has no gaps in `msg_id`.

## (c) Name cards

Source: `extra.colony.cards_before` and `extra.colony.cards_after`. Each is a dict keyed by `ant-00` to `ant-07`.

Exact keys per card: `ant_id`, `policy`, `model`, `rev`, `thresholds` (`dose`, `explore`, `verify`), `in_run` (`claims`, `confirmed`, `contradicted`), `cross_run` (`claims`, `correct`, `checked`, `runs`).

Real card (`cards_after`, ant-06):

```
"ant-06": {"ant_id": "ant-06", "cross_run": {"checked": 0, "claims": 0, "correct": 0, "runs": 0}, "in_run": {"claims": 13, "confirmed": 7, "contradicted": 0}, "model": "script", "policy": "S-talk", "rev": "v4.2", "thresholds": {"dose": 0.45, "explore": 0.7000000000000003, "verify": 0.9000000000000006}}
```

Notes:

- Thresholds carry float noise (`0.7000000000000003`). Compare with a tolerance.
- `cards_before` is the reset state. Every threshold is 0.5 and every count is 0.
- `cross_run` is all zeros in this file.

The `band` is not in the bundle. Compute it from `in_run` using INTERFACES 8:

- `new` if fewer than 5 claims.
- `usually right` if confirmed / (confirmed + contradicted) is at least 0.7.
- `mixed` otherwise. Five or more claims with none resolved reads `mixed`.

Examples from this file:

- ant-00: 1 claim, so `new`.
- ant-06: 13 claims, 7 confirmed, 0 contradicted, so `usually right`.
- ant-02: 9 claims, none resolved, so `mixed`.
- ant-07: 15 claims, 1 confirmed, 0 contradicted. The rule does not say what to do here. See Gaps.

A claim is a judgement mark (`tumour_here`, `not_tumour`, `need_more`) or an answer.

## (d) Zone need

Source: `extra.colony.need_per_zone`. One row per zone per queen step.

Exact keys: `tick`, `region_id` (int), `need` (`"none"`, `"low"`, `"medium"`, `"high"`), `need_milli` (int), `delivered_milli` (int).

Real rows:

```
{"delivered_milli": 531, "need": "medium", "need_milli": 416, "region_id": 5, "tick": 30}
{"delivered_milli": 1000, "need": "none", "need_milli": 0, "region_id": 14, "tick": 60}
```

In this file there are 18 zones (`region_id` 1 to 18). The ticks are 0, 30, 60 and 90.

## What is NOT in the bundle

- No MRI arrays, no voxel truth, no TC or ET arrays, no scorer-side arrays (INTERFACES 13). Truth-side numbers appear only in `metrics`.
- No per-tick strength series. Compute it (see (a)).
- No per-spot author list. Compute it from the `ant` field.
- No lapse events. A lapse shows only as a missing answer and `counters.asks_lapsed`.
- No `band` on cards. Compute it (see (c)).
- No zone geometry. Zones appear only as `region_id`.
- No voxel size in mm. `params` has the radii in mm but not the voxel size.
- No notebook snapshots. The `notebook_self` observation exists in the live observation, not in `extra.colony`.
- No queen `colony_note` history.
- No status board entries (INTERFACES 4). Only the `region_status` dict is present.
- No refused rows. Refused marks and messages are counts only.
- No per-message effect results. Those come from replay runs (INTERFACES 12).

Also present but out of scope here: `reading_logs` (per ant, look/dose/uptake/washout rows), `turns_skipped`, `counters`, `params`.

## Gaps

1. Voxel size in mm is not in `params`. Without it you cannot turn the 4 mm and 6 mm radii into voxels.
2. INTERFACES does not give the cut points for `need` (none, low, medium, high) or how `need_milli` maps to them. This file shows: 0 is none, 52 is low, 251 to 590 is medium, 632 to 1000 is high. These are observations, not spec.
3. Lapse events are not recorded per message. INTERFACES 3.2 says unanswered asks lapse after 2 ticks, but no row shows it.
4. The `band` rule does not cover 1 to 4 resolved claims with 5 or more total claims. ant-07 is the case. A literal reading gives `usually right`. Confirm with the lead.
5. `metrics.coverage_at_queen` has a key `"120"`, but `need_per_zone` has no rows at tick 120.
6. INTERFACES 8 says cards reset at "section 9 run" and that "section 10 check" carries them. Those section numbers look wrong. Confirm with the lead.
7. Zones are defined in INTERFACES 14 as a sphere plus 6 mm around a centroid. The centroid is not in the bundle.
8. The `marks_near` fields `strong`, `disputed`, `distinct_authors` and `authors` are in the live observation (INTERFACES 7). None of them are bundled. Recompute them from `marks`.
