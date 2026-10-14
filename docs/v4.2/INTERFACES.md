# Antelligence v4.2 — interfaces

Fixed in Step 0, before any parallel work. Builders (W1, K1, K2, Q1, M1) code against this file. A change needs a
lead-signed note in `$V42/decisions/` and an edit here in the same commit. Goal section numbers are in brackets.

Everything v4.1 keeps its meaning (`docs/v4.1/INTERFACES.md` sections 1–3, 5–7). This file only adds.

## 0. Naming and identity

- One **ant** = one squad. `ant-NN` (two digits, `ant-00` … `ant-07`) is the stable name on cards. The world agent id stays
  `squad-N` and the board author stays `{"agent": "squad-N", ...}`, so `_AUTHOR` in `kernel/board.py` is **not** changed.
  Mapping: `ant-NN` ↔ `squad-N`. Board payloads carry the integer `ant` (= squad index).
- Ant integer index is used in every board payload; strings (`ant-03`) appear only in cards, notebooks and bundles.
- Zone = one perceived region (`Region.region_id`) plus everything within 6 mm of it [1].

## 1. Seeding: one rule for every random draw [2.7]

```
seed64(*parts) = int.from_bytes(sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big")
rng(*parts)    = numpy.random.default_rng(seed64(*parts))
```

`parts[0]` is always the stream name below. Cases are `case_id` strings; ints are decimal. Nothing else seeds anything.

| Stream | Tuple | Used for |
|---|---|---|
| `f1_field` | `("f1_field", case_id)` | SimLook uniform field (one per voxel, then smoothed) [5.1] |
| `f2_srcB` | `("f2_srcB", case_id)` | Source B poor-uptake field [5.2] |
| `f3_patch` | `("f3_patch", case_id)` | washout patches [5.3] |
| `uptake_noise` | `("uptake_noise", case_id, seed, ant, tick)` | uptake reading noise, sd 0.1 [5.2] |
| `washout_noise` | `("washout_noise", case_id, seed, ant, tick)` | washout reading noise, sd 0.01 [5.3] |
| `displace` | `("displace", case_id, seed, mark_index)` | S-marks-displaced destination [8] |
| `shuffle` | `("shuffle", case_id, seed, msg_id)` | S-talk-shuffled recipient [8] |
| `effect_sample` | `("effect_sample", arm, case_id, seed)` | which runs/items the message-effect replay samples [8] |
| `bootstrap` | `("bootstrap", comparison_name)` | case-clustered bootstrap, 10,000 resamples [8] |
| `tune_sample` | `("tune_sample", "grid")` | nothing random is expected here; the grid is enumerated in sorted order |

The fact name for the F1–F3 seed rule in [5] is `sha256(case_id ‖ fact name)`; it is the `parts` above (`"f1_field"` etc.).
Episode seeds (1, 2, 3 …) keep their v4.1 meaning (entry sites and the v4.1 draws). v4.2 draws never use the episode seed
unless the tuple above says so.

## 2. Marks [6.1]

### 2.1 Kinds

| Kind | Code | Placed by | Half-life (ticks) |
|---|---|---|---|
| `dosed` | 1 | world, on every release (carries `amount`) | 30 |
| `dose_ok` | 2 | world, own local dose ≥ D_t within a zone | 40 |
| `slow_uptake` | 3 | world, from the uptake reading | 60 |
| `washout` | 4 | world, from the washout reading | 60 |
| `tumour_here` | 5 | ant (judgement) | 40 |
| `not_tumour` | 6 | ant (judgement) | 80 |
| `need_more` | 7 | ant (judgement) | 15 |

Read radius 6 mm. Confirm radius 4 mm. Distances are millimetres (`voxel_mm * euclid_vox`). Changing a number needs a
decision note.

### 2.2 Field signal (in-memory, `SignalDraft` / `Signal`)

`kind` = the kind name above. `pos` = the voxel as 3 floats. `ttl` = `ceil(8 * half_life)` ticks (strength is negligible
after 8 half-lives and the signal then expires). `payload` (floats allowed in signals):

```
{"ant": int, "amount": float, "tick": int}
```

Visible from the next tick (kernel rule `emitted_at < tick`). Marks are scoped by arm (`scope = arm name`) so arms never
read each other.

### 2.3 Aggregation (our own query, not `GridField.intensity_at`)

For one `kind` at a query voxel `v`: take, **per author**, that author's latest deposit with `dist(v, pos) ≤ confirm_radius`.
`strength(kind, v, t) = Σ_authors amount × 0.5 ^ ((t − tick) / half_life[kind])`. Same author depositing again refreshes
its own age and adds nothing else. `distinct_authors` = number of authors counted. Set `SenseQuery.max_results` explicitly
(≥ 4096) when pulling candidates.

- **strong** = `distinct_authors ≥ 2` and `strength ≥ 1.0`.
- **disputed** = a `tumour_here` and a `not_tumour` both within `confirm_radius` of the same spot (a spot = a mark voxel).
- `amount` is 1.0 for judgement marks and automatic marks other than `dosed`. For `dosed` it is
  `released_ug / (total_drug / n_squads)` (the fraction of one squad's share).

### 2.4 Board entry (every deposit, every arm) [2.5]

Type `evidence`, payload with `subtype: "mark"` and exactly these keys (all integers, except the two enums):

```
{"subtype": "mark",              # enum: "mark"
 "mark_kind": "dosed",           # enum: the 7 kind names
 "ant": 3,                       # int, squad index
 "pos_vox": [x, y, z],           # int triple
 "amount_milli": 1000,           # int >= 0, round(amount * 1000)
 "reading_ppm": 0}               # int in [0, 1_000_000]; the sensor reading that triggered it (0 if none)
```

Compact encoding (mode C/D), integers in this order: `ant, pos_vox(3), kind_code, amount_milli, reading_ppm`.

## 3. Messages [6.2]

### 3.1 Schema (in-memory dataclass `Message`, JSON-plain in bundles)

```
msg_id: int            # per run, 1,2,3 ... in send order
type: "ask" | "answer"
from_ant: int
to_ant: int
tick: int              # tick of sending
about: {"region_id": int | None, "pos_vox": [x,y,z] | None}   # exactly one set
question: "is_tumour" | "dosed_enough" | "should_i_come"
sender_reading: float  # REQUIRED, the asker's own relevant reading in [0,1]
reply_to: int | None   # msg_id of the ask, only on answers
answer: "yes" | "no" | "unknown" | None     # only on answers
answerer_reading: float | None              # mean reading the answer was based on, only on answers
note: str              # <= 120 chars, default "", always quoted when shown, never followed
```

### 3.2 Rules (fixed numbers)

- Delivery next tick. An `ask` gives the receiver an **answer turn** at `tick + 1` (any tick, decision tick or not). Answer
  turns **never call `world.apply`** (it refuses non-decision ticks); they only post the answer.
- Unanswered asks lapse after 2 ticks. Answer rule: mean of the answerer's reading-log looks within 6 mm of `about`
  ≥ 0.5 → `yes`, else `no`; no readings → `unknown`. `dosed_enough` uses its dose readings (`dose_rel` ≥ 1 → `yes`).
  `should_i_come` answers `yes` iff the answerer has a `need_more` mark of its own within 6 mm of `about` or a look ≥ 0.5.
- Budget: ≤ 1 ask per ant per 2 decision rounds; ≤ 2 answers per ant per tick. Over-budget sends are refused and counted
  (`counters.asks_refused`, `counters.answers_refused`).
- Recipient (asks): (1) author of the most recent mark within the read radius of `about`; else (2) nearest ant whose
  reading log has an entry within 6 mm of `about`; else (3) nearest ant. Ties → lower ant id. Never self.

### 3.3 Board entry

Type `evidence`, `subtype: "message"`, exactly:

```
{"subtype": "message",
 "msg_type": "ask",              # enum: ask | answer
 "msg_id": 12,                   # int
 "from_ant": 3, "to_ant": 5,     # ints
 "about_pos_vox": [x,y,z],       # int triple; for a region, its centroid_vox
 "about_region": 7,              # int, -1 if none
 "question": 1,                  # int: 1 is_tumour, 2 dosed_enough, 3 should_i_come
 "sender_reading_ppm": 540000,   # int ppm
 "reply_to": -1,                 # int, -1 if none
 "answer": -1,                   # int: -1 none (ask), 0 no, 1 yes, 2 unknown
 "answerer_reading_ppm": -1,     # int ppm, -1 if none
 "note": ""}                     # str <= 120, NOT posted on chain (free text), the entry id commits to it
```

Compact encoding order: `msg_type_code(ask=1,answer=2), msg_id, from_ant, to_ant, about_pos_vox(3), about_region, question,
sender_reading_ppm, reply_to, answer, answerer_reading_ppm`.

## 4. Status, need, alarms (board) [6.3, 7]

Type `evidence` again (no new type code). Exact keys:

- `subtype: "status"`: `{"subtype","region_id": int,"status": "done"|"not_tumour","k": int,"tick_first": int}`.
  Compact: `region_id, status_code(done=1,not_tumour=2), k, tick_first`. Written once, when quorum is first met.
- `subtype: "need"`: `{"subtype","region_id": int,"need": "none"|"low"|"medium"|"high","need_milli": int,"delivered_milli": int}`.
  Compact: `region_id, need_code(none=0,low=1,medium=2,high=3), need_milli, delivered_milli`. One per zone per queen step.
- Queen alarms use the existing `alarm` type with `kind: "other"`, `value_ppm` = the number that tripped it, and `note`
  `"stall"` or `"dispute:<region_id>"`.

## 5. Board code changes (the only two) [2.5]

1. `kernel/board.py` `_PAYLOADS["evidence"]`: stays the six-key v4.1 dict. Add `_EVIDENCE_SUBTYPES: {"mark","message","status","need"}`
   with one exact-key spec each. `validate_payload("evidence", p)` dispatches on `p.get("subtype")`: absent → v4.1 six keys,
   byte-for-byte; present → that subtype's spec. Unknown subtype → `ValidationError`.
2. `anchoring/chain_modes.py` `_COMPACT_FIELDS`: add a second table keyed `("evidence", subtype)`; `payload_ints` picks it when
   `payload` has a `subtype`. The enum-to-code maps above live next to it. The contract type code stays "evidence" (no
   contract change; `BoardLog` reverts on codes above 6).

Test (`tests/v4_2/board/test_evidence_subtypes.py`): v4.1 evidence entries still validate, hash and encode identically
(golden id + compact bytes from a v4.1 fixture); each subtype round-trips; wrong key sets are refused.

## 6. Reading log and sensors [5, 6.6]

Sensor readings exist only at the ant's own voxel (uptake: about its own last release voxel). Per-ant `reading_log`, append-only:

```
{"tick": int, "pos_vox": [x,y,z], "kind": "look" | "dose" | "uptake" | "washout",
 "value": float,        # look: SimLook p in [0,1]; dose: local_dose_rel; uptake: ratio; washout: delta+noise
 "band": int | None,    # look only: scan-p band index 0..3 at that voxel (cuts 0.2, 0.5, 0.8)
 "word": str | None}    # uptake: "as expected"/"slow"; washout: "normal"/"fast"
```

`about` for uptake is the release voxel, stored in `pos_vox`. Held in the ant, serialised in the bundle, never visible to
other ants (only through a message answer or a mark).

## 7. Observation keys [6.6, 2.8]

`OBS_KEYS` (v4.1) is **unchanged**. Add in `worlds/tumor_mri/colony_obs.py`:

```
COLONY_OBS_KEYS = ("look", "uptake_reading", "washout_reading", "reading_log", "marks_near", "messages_in", "zone_need",
                   "colony_note", "region_status", "visited_self", "card_self", "cards_public", "notebook_self")
ALLOWED = {                       # per arm; keys outside the set are ABSENT from the observation, not empty
  "S":              (),
  "S-solo":         ("look","uptake_reading","washout_reading","reading_log","zone_need","colony_note","visited_self","card_self","notebook_self"),
  "S-marks":        S-solo + ("marks_near","region_status","cards_public"),
  "S-talk":         S-marks + ("messages_in",),
  "S-marks-displaced", "S-marks-perfect", "S-marks-quiet": same as S-marks,
  "S-talk-shuffled": same as S-talk,
  "S-prime":        ()              # the oracle reads the scorer and F1-F3 directly; it is not an ant arm
}
```

Barrier test: `set(obs) == set(OBS_KEYS) | set(ALLOWED[arm])` for every arm and every tick, and a second walk over every
queen input. In `S-solo` `zone_need` is the tick-0 value for the whole run (the queen sees no board). The existing test
(`set(obs) == set(OBS_KEYS)` with the colony off) passes unchanged.

Value shapes: `look` float|None; `uptake_reading`, `washout_reading` `{"value": float, "word": str, "about_vox": [..]}`|None;
`marks_near` list of `{"kind","pos_vox","strength","distinct_authors","strong","disputed","authors":[ant ints]}`;
`messages_in` list of Message dicts (note quoted); `zone_need` `{region_id: "none|low|medium|high"}`; `colony_note` str ≤ 200;
`region_status` `{region_id: "open|done|not_tumour"}`; `visited_self` `{"regions":[ids],"walked_count": int}`;
`card_self` / `cards_public` per section 8; `notebook_self` the frozen snapshot (section 9) as a list of lesson texts.

## 8. Name cards [6.4]

```
{"ant_id": "ant-03", "policy": "S-talk", "model": "script", "rev": "v4.2",
 "thresholds": {"dose": 0.5, "explore": 0.5, "verify": 0.5},
 "in_run": {"claims": int, "confirmed": int, "contradicted": int},
 "cross_run": {"claims": int, "correct": int, "checked": int, "runs": int}}
```

`public()` summary = `{"ant_id","policy","model","in_run","band"}` where band is `"new"` (< 5 claims), `"usually right"`
(confirm rate ≥ 0.7) or `"mixed"`. **Cards reset at the start of every section 9 run** (thresholds back to 0.5); only the
section 10 check carries them. Persist as `runs/<colony>/cards/<ant_id>.json` (canonical JSON). Future mapping to
`ReputationRegistry` is documented in the code docstring; nothing is written on chain.

A **claim** = a judgement mark (`tumour_here`, `not_tumour`, `need_more`) or an answer. Confirmed/contradicted in run =
another ant's later look (≥ 0.5 vs < 0.5 for tumour claims) within 4 mm of the claim spot. Cross-run checks use the scorer
on development cases only.

## 9. Notebooks [6.7]

```
{"ant_id": "ant-03", "cap_chars": 1500, "frozen": false, "run_index": 0,
 "lessons": [{"id": "L001", "text": "<=200 chars", "source_runs": [str], "uses": int, "effect": float,
              "status": "active" | "archived", "last_used_run": int, "pinned": false}]}
```

API: `add(text, source_runs)`, `patch(id, text)`, `remove(id)` (= archive), `snapshot()` (frozen copy, active lessons
only), `curate(run_index, stale_runs=10)` (archive unused ≥ `stale_runs`; pinned exempt; nothing is ever deleted),
`freeze()`. Writes that would exceed 1,500 characters of active text are refused, as are all writes while frozen.
Export format = agentskills.io lesson shape (`name` ≤ 64, `description` ≤ 1,024, body).

## 10. Job thresholds [6.8]

State per ant: `theta = {"dose": .5, "explore": .5, "verify": .5}`. `P_j = s_j^2 / (s_j^2 + theta_j^2)`, highest wins; ties
dose > verify > explore. Success → `theta_j = max(0.1, theta_j - 0.1)`; a turn in a job with no success →
`min(1.0, theta_j + 0.05)`. Stimuli and success tests exactly as goal 6.8.

## 11. Config keys: `config/v4_2/world.yaml` [5, 6]

All new keys under a `colony:` / `facts:` block. **Defaults are the off values.** With everything off the world loads the
v4.1 `world.yaml` unchanged.

```yaml
facts:
  f1_simlook: false         # SimLook on (injected through look_fn)
  f1_source: boards         # boards | local_d1 | log2000   (which fallback of 5.1 was used; recorded, never inferred)
  f2: false
  f2_source: A              # A | B
  u_core: 1.0               # {0.3, 0.5, 0.7} when on
  f2_share: 0.0             # {0.2, 0.35}, Source B only
  f3: false
  f3_decay: 0.0             # {0.04, 0.08}
  f3_share: 0.0             # {0.1, 0.2}
colony:
  enabled: false
  arm: S                    # S | S-solo | S-marks | S-talk | S-marks-displaced | S-talk-shuffled | S-marks-perfect | S-marks-quiet
  quiet_rule: false         # OFF in every verdict arm
  heartbeat_rounds: 3
  quorum_k: 2
  read_radius_mm: 6.0
  confirm_radius_mm: 4.0
  half_life: {dosed: 30, dose_ok: 40, slow_uptake: 60, washout: 60, tumour_here: 40, not_tumour: 80, need_more: 15}
  ask_every_rounds: 2
  answers_per_tick: 2
  ask_lapse_ticks: 2
  queen_every_ticks: 30
  notebook: {cap_chars: 1500, stale_runs: 10, frozen: true}
```

`WorldConfig.to_dict()` includes the new blocks only when something is on, so the v4.1 config hash is unchanged when off.

## 12. Run-bundle additions (colony runs only) [6.9]

`bundle.json` gains `extra.colony`:

```
{"arm": str, "marks": [ ...mark board payloads + tick ],
 "messages": [ ...Message dicts ], "reading_logs": {"0": [...], ...},
 "cards_before": {...}, "cards_after": {...}, "turns_skipped": {"ant-00": int, ...},
 "need_per_zone": [{"tick": int, "region_id": int, "need": str, "need_milli": int, "delivered_milli": int}],
 "action_log": [{"tick": int, "ant": int, "kind": "decision"|"answer"|"skipped", "action": str, "params": {...}}],
 "counters": {"marks": {kind: n}, "asks_sent": n, "asks_answered": n, "asks_lapsed": n, "asks_refused": n,
              "answers_refused": n, "quorum_events": n}}
```

Replay: `run_colony_episode(replay=bundle)` reproduces `metrics` exactly; a removal replay (`drop={"mark": i}` or
`{"message": msg_id}`) re-runs with that item absent and records whether the receiver's next action changed (message effect).

## 13. Hidden facts: physics hooks [5]

- `DrugField` gains optional `uptake: ndarray|None` (u per voxel, scorer-side) and `washout: ndarray|None` (δ per voxel).
  `dose += u * conc` (F2). `conc *= 1 - decay - δ(v)` (F3), per-voxel decayed-mass accounting incl. `decayed_in_region`.
  Both `None` ⇒ the v4.1 arithmetic path exactly.
- `store_adapter` keeps TC and ET as scorer-side arrays on `Scorer`, never in `Crop` observations or `Crop.truth`.
- `SimLook` is built by `worlds/tumor_mri/simlook.py` from a cell table `cells_v4_2.json` (counts + quantiles per cell,
  no MRI data), and injected as `look_fn(squad, pos, tick) -> float`. Identical at a voxel for every ant, tick, seed, process.
