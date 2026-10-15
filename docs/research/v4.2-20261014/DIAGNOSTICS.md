# A1 diagnostics: why sharing did not help (verdict attempt 1)

Analysis only. No run was repeated and no saved result was changed. Development cases only.
Goal sections 8 and 9.4 ask which of the world or the scripts is the limit. This note answers that and proposes changes. It implements none.

## How to read the cites

- `N` = `docs/research/v4.2-20261014/diagnostics_numbers.json`. It is written by `scripts/v42_diagnose.py`. Cites look like `N:shared_inputs.S-marks.per_run_mean`.
- `REPORT` = `docs/research/v4.2-20261014/attempt1-frozen/REPORT.md`. `EFFECT` = `.../attempt1-frozen/effect.json`. Both are byte-identical to the files in `runs/verdict1`.
- `B` = `runs/verdict1/<arm>/<case>/seed<3|4>/bundle.json` (decisions, readings, marks, need bands, final region status). `REC` = the `record.json` next to it.
- `SL` = `docs/research/v4.2-20261014/simlook/cells_v4_2.json` and `.../simlook/reliability.json`.
- All "per run" numbers are means over the 80 (case, seed) pairs of verdict 1: 40 dev cases, seeds 3 and 4.

## Method limits (read these first)

- Decisions, readings, marks, need bands and final region status are read straight from `B`. That part is exact.
- What an ant saw in `marks_near` and `region_status` is rebuilt by replaying the saved marks into the kernel's `MarkField` and `Quorum`. The replay reproduced the saved final region status and the saved mark count in all 80 runs of each sharing arm (`N:shared_inputs.*.per_run_mean.replay_status_ok` and `replay_marks_ok` both 1.0).
- A rebuilt dose-target ranking matched the logged target on 32.1 of 32.5 dose-zone turns per run in S-marks (98.7%) (`N:shared_inputs.S-marks.per_run_mean.dose_zone_pred_match`, `dose_zone_turns`).
- "Fired" means the rule's condition was true on a saved turn. It does not prove the action would have differed without it. The true removal test for single marks is already in `EFFECT`. I cite it, I did not redo it.
- The tumour mask is read offline only, to say where drug landed. Only aggregates are written.
- S (the v4.1 script) saved no decision log (`runs/verdict1/S/BRATS_016/seed3/bundle.json` has `action_log: []`). So S enters only through its run totals.

## Answer in six lines

1. The limit is both. The world comes first, the scripts second.
2. World: the close look (SimLook) is a worse sensor than the scan. AUROC 0.776 against 0.964 (`SL:reliability.json`, `auroc_overall`). The colony scripts trust it for releases. S trusts the scan.
3. About two thirds of the on-target gap to S is drug dumped on look false positives away from the tumour. About one third is drug never released (section 3).
4. Scripts: the explore job releases on one noisy look, and the verify job eats 45% of all turns (section 3).
5. Sharing is a narrow channel: it fires on a few turns per run and the decisions it changes are not the costly ones. Perfect marks cannot fix it, because the costly releases happen where no other ant is (section 2).
6. The world also offers little to share: the scan is 98% right where ants walk, and the F2 and F3 facts touch 13% and 6% of released drug (section 4).

## 1. How often do shared-input arms act differently from S-solo?

An action is the world-visible tuple: target label, release flag, amount level, goal override. Same case and seed are compared turn by turn (192 decision turns: 8 ants x 24 rounds). Up to the first difference the two runs are identical by construction.

| arm | pairs that ever differ | median first differing tick | pairs differing by tick 30 | turns that differ | target | release | amount | coverage change vs S-solo, mean (sd) |
|---|---|---|---|---|---|---|---|---|
| S-marks | 80 of 80 | 40 | 25% | 46.2% | 23.5% | 7.9% | 1.2% | +0.0093 (0.054) |
| S-talk | 80 of 80 | 37.5 | 31% | 46.4% | 23.4% | 8.1% | 1.2% | +0.0097 (0.055) |
| S-marks-perfect | 80 of 80 | 35 | 35% | 49.6% | 25.1% | 9.0% | 1.3% | +0.0031 (0.068) |
| S-marks-displaced | 80 of 80 | 40 | 26% | 37.4% | 18.5% | 6.1% | 0.8% | -0.0026 (0.049) |
| S-talk-shuffled | 80 of 80 | 40 | 25% | 46.3% | 23.4% | 8.0% | 1.2% | +0.0105 (0.056) |

Source: `N:divergence_from_solo.*`. The first differing tick is never before 25 for S-marks and S-talk-shuffled, and 5 for S-talk (`first_divergence_tick_min`). S-marks-quiet is not a verdict arm: it skips turns, so it differs from tick 15 (`N:divergence_from_solo.S-marks-quiet`).

What this says:

- Sharing changes many actions. Nearly half of all turns differ, and 90% of turns in the last 30 ticks (`share_turns_differ_by_tick_bin`, bin 90: S-marks 0.903). Most of that is drift: once two runs part, every later turn sits in a different state.
- The outcome does not move. The mean paired coverage change is under 0.011 for every arm, with a spread of 0.05 to 0.07 per pair. With 80 pairs the standard error is about 0.006 to 0.008. REPORT gives the same: median +0.0118 for S-marks, +0.0094 for S-talk.
- The displaced control moves just as much and in no better direction (-0.0026). So the content of the marks is not what changes outcomes.
- S-talk against S-marks: 30 of 80 pairs are identical on every turn, 8.2% of turns differ, mean coverage change +0.0004 (`N:divergence_other_pairs.S-talk_vs_S-marks`). Talking added almost nothing.
- S-talk-shuffled against S-marks: 67 of 80 pairs identical, 0.8% of turns differ (`N:divergence_other_pairs.S-talk-shuffled_vs_S-marks`). Shuffled asks are almost always ignored.
- S-marks-perfect against S-marks: 12 of 80 pairs identical, 31.7% of turns differ, mean coverage change -0.0062 (`N:divergence_other_pairs.S-marks-perfect_vs_S-marks`). Perfect marks changed actions and did not help.

What caused the first difference in S-marks (80 runs, `N:shared_inputs.S-marks.first_divergence_cause_counts`):

| first mechanism active on the first differing turn | runs |
|---|---|
| a `tumour_here` or `need_more` mark became a dose spot target | 27 |
| queen need bands changed the zone ranking (the queen reads `dosed` marks, S-solo's queen does not) | 17 |
| need bands differ but no ranking change was traced | 12 |
| explore aimed at a region others had marked | 16 |
| a closed region dropped from the dose targets | 2 |
| strong "dose already high" at the spot | 1 |
| foreign `slow_uptake` made a 75% release | 1 |
| none traced | 4 |

So 29 of 80 first differences (36%) come through the queen's need bands, not through `marks_near`. In S-solo the queen never sees the board and the need bands stay at their tick-0 values (`antelligence/kernel/colony/episode.py`, `_run_queen`). The fairness note in `colony_policies.py` names marks, region status and messages as the shared inputs. It does not name this one.

## 2. Which shared inputs were read, and how often did they change a target, release or amount?

Per run, 192 decision turns. Source: `N:shared_inputs.<arm>.per_run_mean` unless noted.

| shared input | S-marks | S-talk | S-marks-perfect |
|---|---|---|---|
| turns with a mark by another ant within 6 mm (`turns_foreign_mark_visible`) | 62.5 | 62.5 | 63.5 |
| turns with any mark within 6 mm (`turns_any_mark_visible`) | 86.9 | 87.1 | 86.2 |
| quorum events: zones closed (`quorum_events`) | 4.6 | 4.5 | 4.0 |
| median tick of the first closed zone (`N:shared_inputs.*.first_region_closed_tick_median`) | 51 | 51 | 41 |
| dose-zone turns where closed zones changed the top target (`dose_target_changed_by_status`), of about 32.5 dose-zone turns | 4.2 | 4.1 | 3.9 |
| dose-zone turns where the queen's need bands changed the top target (`dose_target_changed_by_queen_need`) | 5.3 | 5.1 | 4.7 |
| dose spot targets taken from another ant's `tumour_here` or `need_more` (`spot_targets_from_marks`) | 8.8 | 8.5 | 9.5 |
| ... of those on true tumour (`spot_targets_on_tumour`) | 7.9 | 7.6 | 9.5 |
| turns blocked by "dose already high" from a strong mark (`shared_dose_high`) | 0.24 | 0.18 | 0.13 |
| 75% releases made because of another ant's `slow_uptake` mark (`released_75pct_from_foreign_mark`) | 1.1 | 1.2 | 1.2 |
| releases blocked by a washout mark with no own fast reading (`washout_blocked_by_foreign_mark`) | 0.21 | 0.13 | 0.16 |
| explore turns aimed by another ant's mark (`explore_aimed_by_mark`) | 9.7 | 9.5 | 13.3 |

S-talk only, from the saved counters (`N:saved_counters_per_run.S-talk`): asks 75.7, answers used 24.9, releases changed by an answer 1.08, targets dropped by an answer 1.8. REPORT shows 63.5 asks sent, 63.4 answered and 12.2 refused per run.

Message-effect test already run on the verdict data (`EFFECT`): removing one mark changed the receiver's next action for 13.0% of 300 marks in S-marks, and for 8.7% of 300 items in S-talk. Spam rate (no action changed) was 87% and 68%. Mean end coverage change from removing one item: -0.0004 and -0.0001.

What this says:

- The channels work. Marks are visible on about a third of turns, and zones are closed by tick 51 in the middle run.
- The reading rules are narrow. Each rule that reads a mark fires on 0.2 to 10 of the 192 turns per run. The one with volume (spot targets, 8.8 per run) points at real tumour 90% of the time and still gives no gain.
- None of the rules lets a mark stop a release. The release rule reads only "dose already high" and the washout mark (`colony_policies.py`, `release_rule`). A `not_tumour` mark never blocks a release. It only removes a zone from the target list after two ants agree.

## 3. Why the colony arms trail S by about 0.3 coverage

S releases 99.8% of its drug. The colony arms release 82% to 86% (`N:arm_means.*.released_share_of_total`). The off-target share is 0.506 for S and 0.70 for the colony arms (`REPORT`, primary metrics).

### 3a. Where the drug went (mean per run, total drug 5153)

Each release was placed with the tumour mask: inside the tumour, within 4 mm of it, or farther than 4 mm.

| arm | inside tumour | within 4 mm | farther than 4 mm | never released | on-target mass at the end |
|---|---|---|---|---|---|
| S | not logged | not logged | not logged | 0.2% | 2805 |
| S-solo | 2172 (42.1%) | 382 (7.4%) | 1669 (32.4%) | 930 (18.0%) | 1402 |
| S-marks | 2263 (43.9%) | 418 (8.1%) | 1731 (33.6%) | 741 (14.4%) | 1496 |
| S-talk | 2273 (44.1%) | 406 (7.9%) | 1739 (33.8%) | 734 (14.3%) | 1499 |
| S-marks-perfect | 2245 (43.6%) | 412 (8.0%) | 1768 (34.3%) | 728 (14.1%) | 1478 |

Source: `N:release_anatomy_per_run.<arm>.mass_by_class_per_run` (`inside_tumour`, `within_4mm`, `far_from_tumour`), `N:arm_means.<arm>.on_target_mass`. Unreleased is total drug minus the three classes. My release accounting matches `released_total` of every run to within 4e-12 (`N:release_anatomy_per_run.*.max_abs_release_accounting_error_vs_released_total`).

Gap to S in on-target mass, split by cause. Drug released inside or within 4 mm of the tumour ends on target 54.9% to 55.9% of the time in the colony arms. S ends 54.5% on target overall. Because the two agree, I apply S's 54.5% to the two lost pools:

| arm | on-target gap to S | from releases farther than 4 mm | from drug never released | sum of the two |
|---|---|---|---|---|
| S-solo | 1403 | 910 (65%) | 507 (36%) | 101% |
| S-marks | 1309 | 944 (72%) | 404 (31%) | 103% |
| S-talk | 1306 | 949 (73%) | 401 (31%) | 103% |
| S-marks-perfect | 1327 | 965 (73%) | 397 (30%) | 103% |

This split is an estimate. It assumes S's tumour-side releases yield the same share as the colony arms'. The sums land within 3% of 100%, which supports it. Timing differences cannot be separated with this data (S has no log). Coverage at tick 30 is 0.128 for S and 0.008 for S-solo (`REPORT`, coverage at queen steps), so S also starts delivering earlier.

### 3b. The releases farther than 4 mm are look false positives

- Every release farther than 4 mm from the tumour was triggered by a look of 0.5 or more (`N:release_anatomy_per_run.S-solo.mass_by_class_per_run`: `look_false_positive_far` 1669 equals `far_from_tumour` 1669).
- The explore job caused most of the false-positive mass: 1740 of 2051 in S-solo (85%) (`fp_job_explore`, `look_false_positive`). The explore job releases on one look of 0.65 or more as it walks past.
- Timing of the false-positive mass in S-solo: 278 at tick 0, 908 in ticks 1 to 29, 865 from tick 30 (`fp_tick_*`). The first two pools are almost the same in every colony arm (278 at tick 0 in all four; 908 to 920 in ticks 1 to 29, `N:release_anatomy_per_run.*`), so sharing did not touch them.
- Releases farther than 4 mm from the tumour: 6.2 per run in S-solo (`N:release_anatomy_per_run.S-solo.per_run.releases_fp_far`).
- The sensor is the cause. At the places ants stood (15,360 looks per arm), the look and the scan compare as follows (`N:look_quality_totals.S-solo`):

| where the ant stood | looks | look at least 0.5 | scan at least 0.5 | look in the grey band 0.35 to 0.65 |
|---|---|---|---|---|
| inside the tumour | 9287 | 81.7% | 98.2% | 50.6% |
| outside, within 4 mm | 935 | 49.4% | 30.8% | 43.2% |
| outside, 4 to 10 mm | 856 | 61.0% | 5.5% | 45.9% |
| outside, more than 10 mm | 4282 | 34.1% | 2.2% | 72.9% |

- This matches the cell table that builds SimLook: far healthy tissue with a low scan value still reads 0.5 or more in 21.5% of draws (`SL:cells_v4_2.json`, cell `d0s0`, derived from the quantiles). The 4 to 10 mm band reads 0.5 or more in 63% of draws (cell `d1`).
- Every ant reads the same value at a voxel (`antelligence/worlds/tumor_mri/simlook.py`, docstring). A second look at the same place adds nothing.

### 3c. The never-released drug and the verify loop

- Turns by job in S-solo: verify 87.4, explore 59.2, dose 45.4 of 192 (`N:release_anatomy_per_run.S-solo.decision_turns_by_job`). Verify never releases. It takes 45% of turns. In S-marks it takes 51% (98.9 turns).
- 56% of all looks fall in the grey band, so a grey spot is almost always near (`N:look_quality_totals.S-solo`, sum of `look_grey_*` over 15,360 looks).
- While standing inside the tumour (116 of 192 turns per run), an S-solo ant releases on 17.6 turns (15%). The other turns: verify 52.6, dose already high 22.8, look under 0.5 13.8, payload empty 8.5, washout 0.8 (`N:release_anatomy_per_run.S-solo.turns_inside_tumour_per_run`).
- The look misses real tumour: inside the tumour the look is under 0.5 on 18.3% of turns, the scan on 1.8% (section 3b table).
- Release halving is not the cause. S halves the payload the same way (50% per release). The colony arms release 3.2 times per ant per run (25.6 releases per run, `per_run.releases`) and 0.35 ants per run never release at all (`ants_never_released`).
- Travel time was not measured separately. The mean first release is at tick 33 in S-solo (`first_release_tick_mean_over_ants_and_runs`).

### 3d. Same story in the redesigned worlds

R1 and R2 (stronger facts, less drug) repeat the pattern (`N:robustness_arm_means`):

| world | S | S-solo | S-marks | S-marks-perfect | S′ |
|---|---|---|---|---|---|
| R1 coverage | 0.441 | 0.198 | 0.200 | 0.199 | 0.585 |
| R1 drug released | 99.8% | 84.3% | 87.0% | 87.3% | 100% |
| R2 coverage | 0.341 | 0.136 | 0.138 | 0.137 | 0.447 |
| R2 drug released | 99.8% | 86.4% | 89.2% | 89.3% | 100% |

Stronger facts did not help because the loss is upstream of the facts.

## 4. Sensors, marks and beliefs: is there information worth sharing?

F2 and F3 readings (frozen world: `u_core` 0.5, `f3_decay` 0.04, `f3_share` 0.1; `N:frozen_facts`).

| item | per run | source |
|---|---|---|
| uptake readings (one per release, next turn) | 25.3 in S-solo, 27.7 in S-marks | `N:arm_means.*.facts_mean.uptake_readings` |
| ... of which slow | 5.2 and 5.9 | `...uptake_slow` |
| washout readings available (out of 192 turns) | 72.9 (38%) and 70.8 (37%) | `...washout_available`, `washout_available_share` |
| ... of which fast | 7.9 and 7.9 | `...washout_fast` |
| `slow_uptake` marks written | 5.9 in S-marks | `N:saved_counters_per_run.S-marks.marks_slow_uptake` |
| `washout` marks written | 7.9 in S-marks | `...marks_washout` |
| share of tumour voxels with slow uptake | 18.8% | `N:world_fact_shares_over_pairs.share_tumour_slow` |
| share of tumour voxels with fast washout | 9.8% | `N:world_fact_shares_over_pairs.share_tumour_fast_washout` |
| drug released onto slow tissue | 571 of 4223 (13.5%) in S-solo | `N:release_anatomy_per_run.S-solo.mass_by_class_per_run.on_slow_tissue_f2` |
| drug released onto fast-washout tissue | 263 of 4223 (6.2%) in S-solo | `...on_fast_washout_f3` |

So the sensors are available and do produce marks, about 14 per run. But the facts matter for at most 13% and 6% of released drug, and another ant used them on 1.1 and 0.2 turns per run (section 2).

Do ants hold different beliefs about the same zone? (`N:beliefs_and_overlap.S-solo.beliefs_per_run`, a zone means a perceived region plus 6 mm.)

- Ants looked inside 10.6 of 17.3 zones per run. 6.6 zones were looked at by two or more ants.
- In 2.0 of those 6.6 (30%) the ants' mean looks fall on opposite sides of 0.5. In 2.65 (40%) they differ by 0.2 or more.
- The disagreement is sensor noise, not private knowledge. SimLook is a fixed field: two ants at the same voxel read the same number. They differ only because they stood on different voxels.

Are the judgement marks right? (`N:judgement_mark_accuracy_per_run`, tumour mask used offline.)

| arm | `tumour_here` on true tumour | `tumour_here` placed where scan is under 0.5 | of those, on true tumour | `not_tumour` on true tumour (wrong) |
|---|---|---|---|---|
| S-solo | 33.8 of 43.4 (78%) | 8.9 | 0.21 (2%) | 4.5 of 5.9 (77%) |
| S-marks | 33.1 of 44.4 (74%) | 10.6 | 0.23 (2%) | 3.6 of 5.0 (72%) |
| S-marks-perfect | 37.6 of 37.6 (100%) | 0.26 | 0.26 | 0 of 11.8 |

- The marks that are meant to correct the scan (`tumour_here` where scan is low) are 98% wrong. They spread SimLook's false positives to other ants.
- S-marks-perfect writes correct marks. But 9.7 of its 11.8 `not_tumour` marks per run sit where the scan already says "not tumour" (`scan_below_half_not_tumour` 9.74 of 11.78). They add nothing the scan did not give.
- Place overlap: the nearest foreign mark is a median 12.0 mm away at a decision turn, and within 6 mm on 61.6 of 192 turns (`N:beliefs_and_overlap.S-marks.mark_overlap_per_run`, any age).
- Where it counts, there is nothing to read. Of 6.4 releases per run farther than 4 mm from the tumour in S-marks, 0.21 had any foreign mark within 6 mm and 0.013 had a foreign `not_tumour` mark. In S-marks-perfect: 0.14 and 0.10 of 6.65 (`N:shared_inputs.*.per_run_mean.releases_far_with_*`). These ants are alone, far from everything.

## 5. Ranked verdict and proposed changes

Ranking of what limits the result (largest first):

1. World: the close look is worse than the scan and the script trusts it. This explains about 65% to 73% of the on-target gap (section 3a) and makes shared judgement marks harmful or empty (section 4).
2. Scripts: the explore job releases on one look, and the verify job uses 45% of turns because most looks are grey. This explains the rest, about 30% to 36% (never-released drug) plus the false-positive share that the explore job causes (section 3b, 3c).
3. Both: sharing reaches few of the costly decisions. The costly releases happen alone, early, and far from the tumour, where no mark can reach (section 4). The shared inputs the scripts do read are the wrong ones to prevent that.

S-marks-perfect answers goal 9.4 this way. Perfect marks changed 49.6% of turns against S-solo (`N:divergence_from_solo.S-marks-perfect`) and left coverage flat (0.269 against 0.266, `REPORT`). The authors' own decisions still use SimLook, and those decisions carry the loss. So with this world and these scripts, better shared information cannot help. This is a joint limit. Neither fixing marks alone nor adding facts alone will move the verdict.

Proposed changes for the next iteration (not implemented):

1. **World: make the close look at least as informative as the scan.** Rebuild SimLook so a look refines the scan (for example, condition the cell on the scan value more finely, or combine the two) and keep far healthy tissue near 0 as the scan does. Evidence: look at least 0.5 on 34.1% of far-healthy turns against 2.2% for the scan (section 3b); AUROC 0.776 against 0.964 (`SL:reliability.json`); 65% to 73% of the on-target gap (section 3a). Check the fix on S-solo alone first: it should close most of the 0.59 against 0.27 gap before any sharing arm is run.
2. **Script: stop the explore job from releasing on a lone look.** Require scan 0.5 or more as well, or a mean of two looks 4 mm apart. Evidence: the explore job caused 1740 of 2051 false-positive mass in S-solo (85%) (section 3b). Same fix removes the mass that hits before tick 30.
3. **Script: bound the verify job.** Cap verify turns per spot, or define "grey" against the local scan value. Evidence: 45% of turns are verify, 56% of looks are grey, and an ant inside the tumour releases on 15% of turns (section 3c). This also frees the 14% to 18% of drug that is never released.
4. **World: give sharing something the scan does not know.** Add a scan error the close look can correct (a perceived region that is partly wrong, or a missed lesion), so a `tumour_here` or `not_tumour` mark carries new information. Evidence: the scan is right on 98.2% of in-tumour and 97.8% of far-healthy turns, and judgement marks that correct it were 2% right (section 4). F2 and F3 reach only 13.5% and 6.2% of released drug, so raising their strength alone (R1, R2) did not move any arm (section 3d).
5. **Script and world: let a mark or a board summary reach the first 30 ticks and the transit.** Options: ants enter in groups so marks are within 6 mm; a larger read radius for `not_tumour`; or a zone-level "healthy" status written at tick 0 from the scan. Also let the release rule read `not_tumour` marks. Evidence: 58% of false-positive mass in S-solo lands by tick 30 (1186 of 2051, section 3b), before the first zone closes (median tick 51), and only 3% of far releases had any foreign mark within 6 mm (sections 2 and 3b).

Order of work: 1 and 2 together, then 3, then check S-solo against S. Only then run 4 and 5 with a sharing arm. If S-solo still trails S by more than 0.05 after 1 to 3, sharing cannot show a win however it is built.

## Reproduce

```
HF_HUB_OFFLINE=1 PYTHONPATH=<worktree> /Volumes/WD_BLACK/antelligence-v4.2/venv/bin/python scripts/v42_diagnose.py \
  --run-dir /Volumes/WD_BLACK/antelligence-v4.2/runs/verdict1 \
  --cases-root /Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases \
  --extra-run-dir /Volumes/WD_BLACK/antelligence-v4.2/runs/verdict-R1 \
  --extra-run-dir /Volumes/WD_BLACK/antelligence-v4.2/runs/verdict-R2
```

It takes about 10 seconds, reads the run directories only, and writes `diagnostics_numbers.json` next to this file. Tests: `tests/test_v42_diagnose.py`.
