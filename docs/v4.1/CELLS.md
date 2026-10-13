# v4.1 cell schema

One JSON object per run (arm, case, seed). Validator: `antelligence/experiments/v41_cells.py`
(`validate_cell`, `validate_cells`). Every key below is required; "nullable" fields may be `null`.
Unknown keys are rejected, except an optional `error` string for failed runs (excluded by the analysis).
Arms are the twelve in goal section 6 (`S`, `S-prime`, `SW-B`, `SW-noBoard`, `FW`, `SW-D`, `SW-noQueen`,
`SW-staticQueen`, `SW-Bonsai`, `SW-Byz-local`, `SW-Byz-chain`, `SW-B-missingFLAIR`).

The primary outcome is `coverage` (goal section 7), a fraction in [0,1]; one "coverage point" is 0.01.
The unit of analysis is the case: the mean of `coverage` over the case's seeds. Arms that share cases use
the same seeds, so a single-seed arm (FW, SW-D, ...) is paired with SW-B on the seeds it has.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `run_id` | hex64 | no | sha256(JCS(run header)), INTERFACES 6 |
| `arm` | string | no | one of ARMS |
| `case` | string | no | MSD case id (never goes on-chain) |
| `seed` | int | no | run seed |
| `split` | string | no | dev or eval |
| `wall_s` | float >= 0 | no | wall time of the run, seconds |
| `ticks_to_50` | int | yes | first tick at which absolute coverage reaches 0.5 (goal 4.7); null = never reached |
| `seconds_to_50` | float >= 0 | yes | wall seconds until 50% of the final score; null = never reached |
| `queen_calls` | int | no | queen model calls (retries of one call during a provider outage are not counted again) |
| `queen_fallbacks_provider` | int | no | queen steps that fell back to the S-rule because the provider failed for a reason other than an outage (HTTP 4xx other than 401/429, malformed response, wrong served model). **A run with any is invalid for fits and the sandbox keep rule** (`v41_fit` excludes it unless `--allow-provider-fallbacks`; the sandbox never caches it) |
| `queen_fallbacks_invalid_reply` | int | no | queen steps that fell back because the reply was invalid (not JSON, no valid route, length or over the 800-token output cap). Legitimate study behaviour |
| `queen_fallbacks_cap` | int | no | queen steps that fell back because the per-run call cap (6) or a call/token budget was spent. Legitimate study behaviour |
| `board_size` | int | no | entries on the board at run end |
| `anchors` | int | no | anchor points posted (INTERFACES 3) |
| `visibility_latency_ms` | int | yes | mode D: write to visible latency; null outside mode D |
| `coverage` | fraction [0,1] | no | primary outcome: fraction of tumour region covered, in [0,1]; 1 point = 0.01 |
| `off_target` | fraction [0,1] | no | fraction of drug released off target |
| `defer_rate` | fraction [0,1] | no | fraction of decisions deferred to the slow path |
| `citations` | int | no | cited board entries |
| `unread_frac` | fraction [0,1] | yes | fraction of briefed entries not read; null for arms without a board |
| `brief_age_median` | float >= 0 | yes | median age in ticks of briefed entries; null without a board |
| `router_top_share` | fraction [0,1] | yes | share of routed calls going to the most used model; null without router |
| `perception_ece` | fraction [0,1] | yes | perception expected calibration error; null if not measured |
| `cost_usd` | float >= 0 | no | paid API spend of the run, dollars |
| `tokens_in` | int | no | model tokens in, all paid and pod calls of the run |
| `tokens_out` | int | no | model tokens out |
| `pod_minutes` | float >= 0 | no | pod minutes attributed to the run |
| `gas` | int | yes | gas used on chain 300 by the run; null when nothing was posted on-chain |
| `alarms` | int | no | alarm entries written (defer-rate, timeout, identity, other) |
| `audit_conformal_coverage` | fraction [0,1] | yes | conformal coverage achieved on the 10% audit sample; null if no audit sample |
| `citation_validity` | fraction [0,1] | yes | fraction of cited entry ids that exist on the board and precede the decision |
| `uncited_decisions` | int | no | decisions that cite no entry |
| `perception_dice` | fraction [0,1] | yes | perception Dice against the truth mask; null if not measured |
| `locallook_ece` | fraction [0,1] | yes | d1 local-look ECE; null if not measured |
| `truth_outside_crop` | fraction [0,1] | yes | share of truth tumour voxels outside the perception crop |
| `decision_errors` | int | no | decisions that still failed after 3 retries (goal 8); the squad took the neutral fallback (target `S`, release no, amount 1), an `alarm` entry of kind `other` ("decision_error: ...") was written and the run continued |
| `look_errors` | int | no | d1 local looks that still failed after 3 retries; the squad decides with no look (`look = null`), an error row (`kind: "look"`) is written to `decisions.jsonl` and an `alarm` of kind `other` ("look_error: ..."). Not counted in `decision_errors` |
| `decision_error_share` | fraction [0,1] | no | `decision_errors` / decisions in the run |
| `decision_error_alarm` | int 0/1 | no | 1 when `decision_error_share` > 0.05; flags the run for the explained-null checklist, changes nothing else |
| `backends_kind` | `live` / `config` / `fake` | yes | where worker answers came from; evaluation analysis accepts only `live` (null only on error cells) |
| `router_kind` | `frozen` / `none` / `fake` | yes | evaluation analysis accepts only `frozen`; `none` is the Phase 0 bootstrap (development only) |
| `router_sha256` | hex64 | yes | sha256 of the frozen router; null unless `router_kind` is `frozen` |
| `look_temperature` | float | yes | temperature applied to d1's local-look probability; null only on error cells |
| `look_temperature_source` | str | yes | `frozen` (config/v4_1/look_temperature.json, sha256-verified) or `override`; null only on error cells |
| `identities` | comma list | yes | served-model identities seen in the run; `fake` marks fake backends and is rejected for analysis; null for scripted arms |

## Secondary outcomes S1 to S12 (goal section 7)

These are summaries of the fields above and of two external files. They are reported, never used to declare a win.

| ID | Definition | Source |
|---|---|---|
| S1 | SW-noQueen vs SW-B, coverage difference | `coverage` |
| S2 | SW-staticQueen vs SW-B | `coverage` |
| S3 | SW-Bonsai vs SW-B | `coverage` |
| S4 | SW-Byz-local vs SW-Byz-chain, and both vs SW-B | `coverage` |
| S5 | perception Dice, voxel ECE, d1 local-look ECE, truth share outside the crop | `perception_dice`, `perception_ece`, `locallook_ece`, `truth_outside_crop` |
| S6 | defer rate per arm, alarms, conformal coverage on the audit sample | `defer_rate`, `alarms`, `audit_conformal_coverage` |
| S7 | citation validity, uncited decisions, unread entries | `citation_validity`, `uncited_decisions`, `unread_frac` |
| S8 | off-target share and time to 50% coverage, all arms | `off_target`, `seconds_to_50`, `ticks_to_50` |
| S9 | (coverage - S)/(S' - S), ratio of means, only on cases with S' - S >= 0.15 | `coverage` |
| S10 | SW-B-missingFLAIR vs SW-B | `coverage` |
| S11 | cost per arm: dollars, tokens, pod-minutes, gas | `cost_usd`, `tokens_in`, `tokens_out`, `pod_minutes`, `gas` |
| S12 | the latency benchmark (goal 4.7) | `latency/results.json`, passed to `analyze(latency=...)` |

Comparison outcomes (S1 to S4, S10) report the mean paired case difference with a 95% BCa interval and no test.
Field outcomes (S5 to S8, S11) report the mean over runs per arm; nulls are excluded from the mean and counted in `n`.

## Explained-null checklist: exactly how each item is computed

Computed on SW-B runs. Every test is strictly "above".

| Item (goal 7 wording) | Value | Threshold | Basis |
|---|---|---|---|
| unread entries above 50% | `unread_frac` | 0.50 | mean of the per-run value over evaluation runs |
| median brief entry age above 30 ticks | `brief_age_median` | 30 | median over evaluation runs of the per-run median |
| deferral out of band in more than 25% of runs | share of runs | 0.25 | a run is out of band when `defer_rate` < 0.05 or > 0.40 (goal 4.5 alarm band) |
| router sent more than 80% of decisions to one model | `router_top_share` | 0.80 | mean of the per-run top-model share over evaluation runs |
| perception development ECE above 0.10 after the refit | `perception_ece` | 0.10 | mean over development-split runs (post-refit) |

The live-check defer band, 10% to 30% (goal 5.10), is a separate check (`live_check` in results.json) on the
mean development defer rate. It is not part of the explained-null checklist.

## Decision failure policy (goal 8: "retry provider errors up to 3 times with the same request, then mark them as errors")

Identical for every backend and arm (`V41Harness._call`). A backend call that raises `DecisionError` (provider error, reply
that fails validation: not JSON, wrong keys, an all-zero, NaN or negative distribution, ...) is retried up to 3 times with
the same request. On retries 2 and 3 a verbalised backend (`prob_method == "verbalised"`: FW, SLMs) also gets a one-line
corrective note naming the rejection reason; the note is not part of the request hash and is recorded in the decision log
(`notes`, `attempts`). Replies that fail validation are never cached. If the call still fails, only that DECISION is an
error: the decision log row has `error=true` and `error_reason`, `decision_errors` / `decision_error_share` go in the cell,
and `decision_error_alarm` = 1 when the share is above 5%. A `DecisionAbort` (a spent budget) is never absorbed and still
stops the run. The explained-null checklist reader should treat a flagged run as one whose decisions were partly neutral
fallbacks.

### Queen provider outages (decision 2026-10-13, mistake-2039)

A queen provider outage (HTTP 401, 429, 5xx or a transport error) is never an invalid reply. The in-episode queen and the
sandbox proposer send the SAME request again with exponential backoff (15 s, 30 s, 60 s, then 120 s between tries) for at
most 600 s in total (`--queen-max-wait-s`). The retries are the same queen call: they do not count against the 6-call cap
and each attempt still goes through the usage ledger and the Nous stop. Past the maximum wait `QueenUnavailable` (a
`DecisionAbort`) ends the run as an **error** cell, never a fallback. `v41_run.py run_many` then starts no new episode
(`EpisodeNotStarted`: no cell, no cache entry), lets in-flight ones finish or error, and the driver exits with code **75**
(`EXIT_QUEEN_UNAVAILABLE`, EX_TEMPFAIL; the sandbox driver too). Rerun later: the reply caches and the sandbox results cache
resume it. Fallbacks that remain are counted by cause in `queen_fallbacks_provider|invalid_reply|cap`, in the run log
(`queen_fallbacks`) and in the summary line; the batch end prints the totals. Never read "0 errors" as clean without them.

### Follow-ups to the policy

- **Local look.** The d1 look is under the same policy (3 retries, then `look = null`, a `kind: "look"` row in
  `decisions.jsonl`, an `alarm`). Looks are counted in `look_errors`, never in `decision_errors`.
- **Budget hard stops.** `BudgetRefused` (Nous 7.50, RunPod 9.50) and the SLM backend's run-budget trip are
  `DecisionAbort`: not retried, not an error decision, not a queen fallback. They end the run.
- **Billed failures.** A rejected reply that the provider charged for (Nous returns `usage`) is added to `cost_usd`,
  `tokens_in` and `tokens_out` and gets a usage-ledger row with its cost (so the budget stop sees it).
- **Replay.** A run's failed attempts (reason and billed usage) are in its decision log (`failures`) and in the bundle
  (`extra.attempts`, keyed by decision or look slot). `replay()` re-plays exactly those attempts, then lets the offline
  cache serve the stored reply, so retries, cost and error decisions reproduce. `CachedDecision` stores valid replies
  as `<hash>.json`; `<hash>.failures.jsonl` is a shared audit log and is never read back, so a stale failure from an
  earlier run cannot enter another run's replay. Usage billed on malformed or invalid replies (also inside the backend's
  own transport retries) is counted too.
- **Asymmetry.** The corrective note on retries 2 and 3 exists only for verbalised backends (FW and the SLMs). Logprob
  backends (d1, Qwen, Bonsai) get the identical request on every retry, and at temperature 0 a retry usually repeats the
  failure. FW therefore has more chances to recover a decision than the other arms. This is a difference in how failures
  are treated, not in the question asked.
- **Reporting.** `results.json` carries `decision_errors`: `decision_error_share` per arm (mean, run counts, look
  errors). If any arm's mean share is above 5%, it also carries a sensitivity analysis: P1 and P2 recomputed on the runs
  with no error decision (the run is the unit that can be excluded; coverage is not separable per decision). The
  sensitivity P2 is always exploratory. **P2 caveat:** SW-B and FW can have different error rates, and an error decision
  is a neutral fallback, so P2 must be read together with this block. It is a secondary: reported, never used to declare
  a win.
- **Internal retry multiplier.** `SLMBackend` and the HTTP backends retry transport/provider errors themselves (3 retries,
  with backoff). The harness policy sits on top, so one decision can make up to 4 x 4 = 16 physical calls in the worst
  case (each rejected-reply retry that is a transport error multiplies). Rejected replies (invalid JSON, zero sum) are not
  retried inside `SLMBackend`/`NousFWBackend` (they raise at once), so for them the maximum is 4 calls. Every physical call
  that returns a billed reply is in the usage ledger.
