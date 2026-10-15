# v4.3 Phase 0 report (generated)

Generated from `results.json` by `scripts/v43_report.py`. Tables only. No interpretation here.

## Run info

| key | value |
|---|---|
| ant_rules | {"answer_rule": "v43", "card_min_resolved": 3, "dose_move_on": true, "explore_release_c": 0.6, "explore_scan_min": 0.5, "fold_can_permit": false, "fold_drop_c": 0.6, "foraging": true, "good_level": 3, "good_u": 0.7, "grey": [0.35, 0.65], "hold_max_turns": 2, "info_mm": 6.0, "middling_level": 2, "move_on_mm": 6.0, "not_tumour_c": 0.2, "not_tumour_scan_min": 0.5, "poor_u": 0.4, "probe_level": 1, "release_c": 0.6, "resolved_only": true, "theta_ceil": 1.0, "theta_floor": 0.1, "theta_step_down": 0.05, "theta_step_up": 0.05, "tumour_here_any_c": null, "tumour_here_c": 0.8, "tumour_here_scan_max": 0.5, "use_combined": true, "verify_per_spot_max": 2, "verify_share_max": 0.25, "version": "v43", "washout_release_c": 0.65, "zone_halve": 0.5} |
| ant_rules_sha | "957a0b3c51244eda82b90b080c43fc5c692e58260b83ca01086719147905a17f" |
| arms | ["S", "S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S'", "S-marks-perfect", "S-marks-quiet"] |
| cases | ["BRATS_016", "BRATS_406", "BRATS_183", "BRATS_238", "BRATS_193", "BRATS_464", "BRATS_479", "BRATS_303", "BRATS_420", "BRATS_337", "BRATS_056", "BRATS_441", "BRATS_373", "BRATS_462", "BRATS_113", "BRATS_375", "BRATS_205", "BRATS_246", "BRATS_468", "BRATS_368", "BRATS_034", "BRATS_356", "BRATS_120", "BRATS_389", "BRATS_343", "BRATS_483", "BRATS_287", "BRATS_328", "BRATS_286", "BRATS_069", "BRATS_167", "BRATS_067", "BRATS_355", "BRATS_297", "BRATS_149", "BRATS_161", "BRATS_290", "BRATS_423", "BRATS_374", "BRATS_101"] |
| ceiling_check | {"n_case_seeds": 80, "ok": true, "required": 0.95, "share_s_prime_ge_all": 1.0, "violations": []} |
| code_sha | "5923e49eea1adf31baef05109ebfb81e0dde98173a77fa82c859de40f5dddd32" |
| coefficients | "/Volumes/WD_BLACK/antelligence-v4.3/worktrees/v43-integration/docs/research/v4.3-20261015/combined/coefficients.json" |
| coefficients_sha256 | "6b124dc9ebb5187bc9de2ada489841a5a61e6225b51fc7882873e8292126f353" |
| config | "/Volumes/WD_BLACK/antelligence-v4.3/worktrees/v43-integration/config/v4_3/world.yaml" |
| config_sha256 | "1040e9f5d8c7e0fc573c3836ba9dbdedd582f59c0b70fea7a3439c0bf8a70bb2" |
| dry_overrides | {"ticks": null} |
| freeze | {"commit_now": "eb6c34441e7ca843cd2c7110e56e550f01f8d092", "commit_recorded": "e212d7e6cf874965bcbe82aa637e7488cd771db1", "file": "/Volumes/WD_BLACK/antelligence-v4.3/worktrees/v43-integration/docs/research/v4.3-20261015/freeze.json", "freeze_hash_now": "bac6746fe045144731788490558b0eb1dfabe44c4bdf1d116afde66335f7c69d", "freeze_hash_recorded": "bac6746fe045144731788490558b0eb1dfabe44c4bdf1d116afde66335f7c69d", "required": true} |
| frozen_facts | {"f1_simlook": true, "f1_source": "boards", "f2": true, "f2_model": "foraging", "f2_share": 0.0, "f2_source": "A", "f3": true, "f3_decay": 0.04, "f3_share": 0.1, "u_core": 1.0, "uptake_level": "mid"} |
| incomplete | [] |
| level | "mid" |
| n_runs | 720 |
| seeds | [5, 6] |
| source_dir | "/Volumes/WD_BLACK/antelligence-v4.3/runs/verdict-mid-s56" |
| study | "v4.3" |
| workers | 4 |

## Primary metrics (mean over runs)

The time-to-50% mean covers only runs that reached 50%; the last column counts the runs that never did.

| arm | runs | coverage | off-target share | time to 50% coverage | drug on non-tumour | runs never at 50% |
|---|---|---|---|---|---|---|
| S | 80 | 0.482 | 0.516 | 87.667 | 2393.668 | 41 |
| S-solo | 80 | 0.390 | 0.549 | 105.000 | 2125.435 | 74 |
| S-marks | 80 | 0.423 | 0.531 | 101.895 | 2152.798 | 61 |
| S-talk | 80 | 0.423 | 0.530 | 101.895 | 2155.001 | 61 |
| S-marks-displaced | 80 | 0.414 | 0.536 | 108.833 | 2158.907 | 62 |
| S-talk-shuffled | 80 | 0.423 | 0.531 | 101.895 | 2152.798 | 61 |
| S-marks-perfect | 80 | 0.423 | 0.531 | 103.474 | 2161.518 | 61 |
| S-marks-quiet | 80 | 0.416 | 0.530 | 101.688 | 2117.276 | 64 |
| S' | 80 | 0.633 | 0.443 | 80.042 | 1992.861 | 8 |

## Coverage at queen steps (mean)

| arm | t=0 | t=30 | t=60 | t=90 | t=120 |
|---|---|---|---|---|---|
| S | 0.000 | 0.095 | 0.351 | 0.447 | 0.482 |
| S-solo | 0.000 | 0.029 | 0.188 | 0.319 | 0.390 |
| S-marks | 0.000 | 0.030 | 0.198 | 0.340 | 0.423 |
| S-talk | 0.000 | 0.029 | 0.198 | 0.340 | 0.423 |
| S-marks-displaced | 0.000 | 0.030 | 0.197 | 0.337 | 0.414 |
| S-talk-shuffled | 0.000 | 0.030 | 0.198 | 0.340 | 0.423 |
| S-marks-perfect | 0.000 | 0.030 | 0.199 | 0.339 | 0.423 |
| S-marks-quiet | 0.000 | 0.027 | 0.189 | 0.334 | 0.416 |
| S' | 0.000 | 0.036 | 0.339 | 0.552 | 0.633 |

## Colony metrics (mean per run)

| arm | marks | share strong | share disputed | asks sent | answered | lapsed | refused | quorum events | turns skipped | honesty r |
|---|---|---|---|---|---|---|---|---|---|---|
| S-solo | 199.7 | 0.307 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 6.2 | 0.0 | n/a |
| S-marks | 196.8 | 0.268 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 6.2 | 0.0 | n/a |
| S-talk | 196.0 | 0.266 | 0.000 | 5.2 | 5.2 | 0.0 | 1.4 | 6.2 | 0.0 | n/a |
| S-marks-displaced | 197.3 | 0.047 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 1.4 | 0.0 | n/a |
| S-talk-shuffled | 196.8 | 0.268 | 0.000 | 5.2 | 5.2 | 0.0 | 1.4 | 6.2 | 0.0 | n/a |
| S-marks-perfect | 187.9 | 0.268 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 6.2 | 0.0 | n/a |
| S-marks-quiet | 194.0 | 0.274 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 6.1 | 6.1 | n/a |

### Marks by kind (mean per run)

| arm | dose_ok | dosed | good_uptake | need_more | not_tumour | slow_uptake | washout |
|---|---|---|---|---|---|---|---|
| S-solo | 73.8 | 62.0 | 26.0 | 9.0 | 0.0 | 11.8 | 17.1 |
| S-marks | 59.5 | 66.3 | 28.0 | 14.7 | 0.0 | 12.6 | 15.7 |
| S-talk | 59.5 | 66.2 | 28.0 | 14.6 | 0.0 | 12.6 | 15.2 |
| S-marks-displaced | 67.6 | 63.5 | 26.2 | 11.6 | 0.0 | 12.7 | 15.7 |
| S-talk-shuffled | 59.5 | 66.3 | 28.0 | 14.7 | 0.0 | 12.6 | 15.7 |
| S-marks-perfect | 58.8 | 67.3 | 25.1 | 14.7 | 0.0 | 9.4 | 12.6 |
| S-marks-quiet | 59.0 | 65.1 | 27.8 | 12.9 | 0.0 | 12.6 | 16.7 |

## Quiet-rule pair

| arm | mean coverage | mean turns skipped |
|---|---|---|
| S-marks | 0.423 | 0.0 |
| S-marks-quiet | 0.416 | 6.1 |

## Verdict statistics (coverage, paired by case and seed)

Case-clustered bootstrap, 10000 resamples, seeded. Holm over the two primary comparisons.

| comparison | pairs | median diff | 95% CI | p | Holm p | dev-A median | dev-B median | vs own control: median [CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| (a) S-marks - S-solo | 80 | 0.0293 | [0.0152, 0.0454] | 0.0002 | 0.0004 | 0.0292 | 0.0307 | 0.0104 [0.0002, 0.0155] | fail |
| (b) S-talk - S-solo | 80 | 0.0293 | [0.0151, 0.0452] | 0.0002 | 0.0004 | 0.0292 | 0.0307 | 0.0000 [0.0000, 0.0000] | fail |

| comparison | median >= 0.03 | Holm p < 0.05 | same sign dev-A/B | beats control |
|---|---|---|---|---|
| (a) S-marks - S-solo | False | True | True | True |
| (b) S-talk - S-solo | False | True | True | False |

Sharing helps scripts (at least one primary passes): **False**

### Secondary (no threshold)

| comparison | pairs | median diff | 95% CI | p |
|---|---|---|---|---|
| S-talk - S-marks | 80 | 0.0000 | [0.0000, 0.0000] | 1.0000 |

### Diagnostic: S-marks-perfect - S-solo (not a verdict arm)

| pairs | median diff | 95% CI | p |
|---|---|---|---|
| 80 | 0.0263 | [0.0170, 0.0406] | 0.0002 |

Source: `results.json`.

## Probes (mean per run)

A missing counter counts as 0 (the probe rule is a separate workstream; a run without it shows 0).

| arm | runs | probes | probes saved by others' uptake marks |
|---|---|---|---|
| S-solo | 80 | 28.46 | 0.00 |
| S-marks | 80 | 30.39 | 5.16 |
| S-talk | 80 | 30.35 | 5.22 |
| S-marks-displaced | 80 | 28.21 | 4.31 |
| S-talk-shuffled | 80 | 30.39 | 5.16 |
| S-marks-perfect | 80 | 31.12 | 4.30 |
| S-marks-quiet | 80 | 30.44 | 4.71 |

Source: `results.json`.

## Realised mean u per sub-region and level

Voxel-weighted over the development cases, after clipping to [0.05, 1.0].

| level | cases | rim | core | edema | whole tumour |
|---|---|---|---|---|---|
| low | 40 | 0.927 | 0.594 | 0.875 | 0.833 |
| mid | 40 | 0.852 | 0.389 | 0.751 | 0.704 |
| high | 40 | 0.775 | 0.190 | 0.630 | 0.577 |

Source: `docs/research/v4.3-20261015/uptake/realised_u.json`.
