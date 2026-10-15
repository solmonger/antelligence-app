# v4.2 Phase 0 report (generated)

Generated from `results.json` by `scripts/v42_report.py`. Tables only. No interpretation here.

## Run info

| key | value |
|---|---|
| arms | ["S", "S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S'", "S-marks-perfect", "S-marks-quiet"] |
| ceiling_check | {"n_case_seeds": 80, "ok": true, "required": 0.95, "share_s_prime_ge_all": 1.0, "violations": []} |
| config | "/Volumes/WD_BLACK/antelligence-v4.2/worktrees/v42-integration/config/v4_2/world.yaml" |
| dry_overrides | {"f2": null, "f3": null, "ticks": null} |
| frozen_facts | {"f1_simlook": true, "f1_source": "boards", "f2": true, "f2_share": 0.0, "f2_source": "A", "f3": true, "f3_decay": 0.04, "f3_share": 0.1, "u_core": 0.5} |
| incomplete | [] |
| n_runs | 720 |
| seeds | [3, 4] |
| source_dir | "/Volumes/WD_BLACK/antelligence-v4.2/runs/verdict1" |
| workers | 5 |

## Primary metrics (mean over runs)

The time-to-50% mean covers only runs that reached 50%; the last column counts the runs that never did.

| arm | runs | coverage | off-target share | time to 50% coverage | drug on non-tumour | runs never at 50% |
|---|---|---|---|---|---|---|
| S | 80 | 0.591 | 0.506 | 73.143 | 2336.965 | 10 |
| S-solo | 80 | 0.266 | 0.702 | 94.000 | 2820.821 | 79 |
| S-marks | 80 | 0.276 | 0.701 | 107.500 | 2915.805 | 78 |
| S-talk | 80 | 0.276 | 0.703 | 110.750 | 2920.190 | 76 |
| S-marks-displaced | 80 | 0.264 | 0.709 | 105.000 | 2872.501 | 78 |
| S-talk-shuffled | 80 | 0.277 | 0.702 | 107.500 | 2917.473 | 78 |
| S-marks-perfect | 80 | 0.269 | 0.710 | 109.250 | 2948.019 | 76 |
| S-marks-quiet | 80 | 0.274 | 0.699 | 116.000 | 2861.414 | 79 |
| S' | 80 | 0.788 | 0.458 | 69.359 | 2068.332 | 2 |

## Coverage at queen steps (mean)

| arm | t=0 | t=30 | t=60 | t=90 | t=120 |
|---|---|---|---|---|---|
| S | 0.000 | 0.128 | 0.444 | 0.552 | 0.591 |
| S-solo | 0.000 | 0.008 | 0.073 | 0.188 | 0.266 |
| S-marks | 0.000 | 0.008 | 0.073 | 0.189 | 0.276 |
| S-talk | 0.000 | 0.008 | 0.074 | 0.190 | 0.276 |
| S-marks-displaced | 0.000 | 0.008 | 0.070 | 0.185 | 0.264 |
| S-talk-shuffled | 0.000 | 0.008 | 0.073 | 0.189 | 0.277 |
| S-marks-perfect | 0.000 | 0.008 | 0.072 | 0.184 | 0.269 |
| S-marks-quiet | 0.000 | 0.008 | 0.074 | 0.192 | 0.274 |
| S' | 0.000 | 0.040 | 0.400 | 0.669 | 0.788 |

## Colony metrics (mean per run)

| arm | marks | share strong | share disputed | asks sent | answered | lapsed | refused | quorum events | turns skipped | honesty r |
|---|---|---|---|---|---|---|---|---|---|---|
| S-solo | 157.5 | 0.443 | 0.004 | 0.0 | 0.0 | 0.0 | 0.0 | 4.4 | 0.0 | n/a |
| S-marks | 149.7 | 0.354 | 0.004 | 0.0 | 0.0 | 0.0 | 0.0 | 4.6 | 0.0 | 0.243 |
| S-talk | 150.2 | 0.345 | 0.005 | 63.5 | 63.4 | 0.2 | 12.2 | 4.5 | 0.0 | 0.275 |
| S-marks-displaced | 151.9 | 0.050 | 0.004 | 0.0 | 0.0 | 0.0 | 0.0 | 2.5 | 0.0 | 0.138 |
| S-talk-shuffled | 150.3 | 0.355 | 0.004 | 63.1 | 63.1 | 0.0 | 12.3 | 4.6 | 0.0 | 0.240 |
| S-marks-perfect | 146.3 | 0.365 | 0.015 | 0.0 | 0.0 | 0.0 | 0.0 | 4.0 | 0.0 | n/a |
| S-marks-quiet | 142.6 | 0.369 | 0.006 | 0.0 | 0.0 | 0.0 | 0.0 | 4.9 | 8.2 | 0.083 |

### Marks by kind (mean per run)

| arm | dose_ok | dosed | need_more | not_tumour | slow_uptake | tumour_here | washout |
|---|---|---|---|---|---|---|---|
| S-solo | 52.7 | 25.6 | 16.9 | 5.9 | 5.2 | 43.4 | 7.9 |
| S-marks | 37.1 | 28.1 | 21.3 | 5.0 | 5.9 | 44.4 | 7.9 |
| S-talk | 37.6 | 28.2 | 21.6 | 5.2 | 6.0 | 43.8 | 7.8 |
| S-marks-displaced | 45.4 | 26.3 | 18.7 | 5.0 | 5.1 | 43.3 | 8.2 |
| S-talk-shuffled | 37.4 | 28.2 | 21.4 | 5.0 | 5.9 | 44.4 | 8.0 |
| S-marks-perfect | 33.8 | 28.2 | 23.1 | 11.8 | 6.0 | 37.5 | 5.9 |
| S-marks-quiet | 37.7 | 27.7 | 18.5 | 5.0 | 5.7 | 40.1 | 7.8 |

## Quiet-rule pair

| arm | mean coverage | mean turns skipped |
|---|---|---|
| S-marks | 0.276 | 0.0 |
| S-marks-quiet | 0.274 | 8.2 |

## Message effect (removal replay)

| arm | runs | items | effect rate | mean delta | median delta | spam rate | unapplied |
|---|---|---|---|---|---|---|---|
| S-marks | 10 | 300 | 0.130 | -0.0004 | 0.0000 | 0.870 | 0 |
| S-talk | 10 | 300 | 0.087 | -0.0001 | 0.0000 | 0.680 | 0 |

## Verdict statistics (coverage, paired by case and seed)

Case-clustered bootstrap, 10000 resamples, seeded. Holm over the two primary comparisons.

| comparison | pairs | median diff | 95% CI | p | Holm p | dev-A median | dev-B median | vs own control: median [CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| (a) S-marks - S-solo | 80 | 0.0118 | [0.0002, 0.0179] | 0.0382 | 0.0764 | 0.0104 | 0.0123 | 0.0074 [0.0000, 0.0182] | fail |
| (b) S-talk - S-solo | 80 | 0.0094 | [-0.0054, 0.0214] | 0.3792 | 0.3792 | 0.0119 | -0.0055 | 0.0000 [0.0000, 0.0000] | fail |

| comparison | median >= 0.03 | Holm p < 0.05 | same sign dev-A/B | beats control |
|---|---|---|---|---|
| (a) S-marks - S-solo | False | False | True | False |
| (b) S-talk - S-solo | False | False | False | False |

Sharing helps scripts (at least one primary passes): **False**

### Secondary (no threshold)

| comparison | pairs | median diff | 95% CI | p |
|---|---|---|---|---|
| S-talk - S-marks | 80 | 0.0000 | [0.0000, 0.0000] | 1.0000 |

### Diagnostic: S-marks-perfect - S-solo (not a verdict arm)

| pairs | median diff | 95% CI | p |
|---|---|---|---|
| 80 | 0.0041 | [-0.0016, 0.0187] | 0.2796 |
