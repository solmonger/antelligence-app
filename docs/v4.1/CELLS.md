# v4.1 cell schema

One JSON object per run (arm, case, seed). Validator: `antelligence/experiments/v41_cells.py`
(`validate_cell`, `validate_cells`). Every key below is required; "nullable" fields may be `null`.
Unknown keys are rejected, except an optional `error` string for failed runs (excluded by the analysis).
Scores are fractions in [0,1]; one "point" in the pre-registered margins is 0.01.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `run_id` | hex64 | no | sha256(JCS(run header)), INTERFACES 6 |
| `arm` | string | no | S, S-prime, SW-B, SW-noBoard, FW, SW-D, SW-noQueen, SW-staticQueen, SW-Bonsai, SW-Byz-local, SW-Byz-chain, SW-B-missingFLAIR |
| `case` | string | no | MSD case id (never goes on-chain) |
| `seed` | int | no | run seed |
| `split` | string | no | `dev` or `eval` |
| `score` | fraction | no | task score in [0,1] |
| `wall_s` | float >= 0 | no | wall time of the run, seconds |
| `ticks_to_50` | int | yes | ticks until 50% of the final score; null = never reached |
| `seconds_to_50` | float >= 0 | yes | wall seconds until 50% of the final score; null = never reached |
| `queen_calls` | int | no | queen model calls |
| `board_size` | int | no | entries on the board at run end |
| `anchors` | int | no | anchor points posted (INTERFACES 3) |
| `visibility_latency_ms` | int | yes | mode D write-to-visible latency; null outside mode D |
| `coverage` | fraction | no | fraction of tumour region visited |
| `off_target` | fraction | no | fraction of drug released off target |
| `defer_rate` | fraction | no | fraction of decisions deferred to the slow path |
| `citations` | int | no | cited board entries |
| `unread_frac` | fraction | yes | fraction of briefed entries not read; null without a board |
| `brief_age_median` | float >= 0 | yes | median age in ticks of briefed entries; null without a board |
| `router_top_share` | fraction | yes | share of routed calls going to the most used model; null without router |
| `perception_ece` | fraction | yes | perception expected calibration error; null if not measured |
| `cost_usd` | float >= 0 | no | paid API spend of the run |

Secondary measures S1 to S12 (see `v41_analysis.SECONDARY`) map to fields: S1 `wall_s`, S2 `ticks_to_50`,
S3 `seconds_to_50`, S4 `queen_calls`, S5 `board_size`, S6 `anchors`, S7 `visibility_latency_ms`,
S8 `coverage`, S9 normalised score (derived), S10 `off_target`, S11 `defer_rate`, S12 `citations`.
