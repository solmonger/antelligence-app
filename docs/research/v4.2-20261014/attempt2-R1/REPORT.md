# v4.2 Phase 0 report (generated)

Generated from `results.json` by `scripts/v42_report.py`. Tables only. No interpretation here.

## Run info

| key | value |
|---|---|
| arms | ["S", "S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S'", "S-marks-perfect", "S-marks-quiet"] |
| ceiling_check | {"n_case_seeds": 80, "ok": true, "required": 0.95, "share_s_prime_ge_all": 1.0, "violations": []} |
| config | "config/v4_2/world_R1.yaml" |
| dry_overrides | {"f2": null, "f3": null, "ticks": null} |
| frozen_facts | {"f1_simlook": true, "f1_source": "boards", "f2": true, "f2_share": 0.35, "f2_source": "A", "f3": true, "f3_decay": 0.08, "f3_share": 0.2, "u_core": 0.3} |
| incomplete | [] |
| n_runs | 720 |
| seeds | [3, 4] |
| source_dir | "/Volumes/WD_BLACK/antelligence-v4.2/runs/verdict-R1" |
| workers | 5 |

## Primary metrics (mean over runs)

The time-to-50% mean covers only runs that reached 50%; the last column counts the runs that never did.

| arm | runs | coverage | off-target share | time to 50% coverage | drug on non-tumour | runs never at 50% |
|---|---|---|---|---|---|---|
| S | 80 | 0.441 | 0.444 | 79.737 | 2032.242 | 61 |
| S-solo | 80 | 0.198 | 0.667 | n/a | 2745.951 | 80 |
| S-marks | 80 | 0.200 | 0.670 | n/a | 2820.975 | 80 |
| S-talk | 80 | 0.205 | 0.668 | n/a | 2837.068 | 80 |
| S-marks-displaced | 80 | 0.196 | 0.674 | n/a | 2797.463 | 80 |
| S-talk-shuffled | 80 | 0.202 | 0.670 | n/a | 2824.576 | 80 |
| S-marks-perfect | 80 | 0.199 | 0.677 | n/a | 2849.741 | 80 |
| S-marks-quiet | 80 | 0.198 | 0.666 | n/a | 2753.220 | 80 |
| S' | 80 | 0.585 | 0.367 | 83.734 | 1624.004 | 16 |

## Coverage at queen steps (mean)

| arm | t=0 | t=30 | t=60 | t=90 | t=120 |
|---|---|---|---|---|---|
| S | 0.000 | 0.110 | 0.352 | 0.421 | 0.441 |
| S-solo | 0.000 | 0.007 | 0.061 | 0.147 | 0.198 |
| S-marks | 0.000 | 0.007 | 0.061 | 0.146 | 0.200 |
| S-talk | 0.000 | 0.007 | 0.061 | 0.149 | 0.205 |
| S-marks-displaced | 0.000 | 0.007 | 0.059 | 0.144 | 0.196 |
| S-talk-shuffled | 0.000 | 0.007 | 0.062 | 0.147 | 0.202 |
| S-marks-perfect | 0.000 | 0.007 | 0.061 | 0.144 | 0.199 |
| S-marks-quiet | 0.000 | 0.007 | 0.061 | 0.148 | 0.198 |
| S' | 0.000 | 0.038 | 0.333 | 0.516 | 0.585 |

## Colony metrics (mean per run)

| arm | marks | share strong | share disputed | asks sent | answered | lapsed | refused | quorum events | turns skipped | honesty r |
|---|---|---|---|---|---|---|---|---|---|---|
| S-solo | 157.4 | 0.433 | 0.005 | 0.0 | 0.0 | 0.0 | 0.0 | 4.1 | 0.0 | n/a |
| S-marks | 150.7 | 0.351 | 0.004 | 0.0 | 0.0 | 0.0 | 0.0 | 4.1 | 0.0 | 0.213 |
| S-talk | 152.4 | 0.349 | 0.004 | 63.4 | 63.2 | 0.2 | 12.5 | 4.2 | 0.0 | 0.240 |
| S-marks-displaced | 151.2 | 0.044 | 0.003 | 0.0 | 0.0 | 0.0 | 0.0 | 2.5 | 0.0 | 0.075 |
| S-talk-shuffled | 151.3 | 0.351 | 0.004 | 63.2 | 63.2 | 0.0 | 12.6 | 4.0 | 0.0 | 0.210 |
| S-marks-perfect | 149.2 | 0.353 | 0.016 | 0.0 | 0.0 | 0.0 | 0.0 | 3.6 | 0.0 | n/a |
| S-marks-quiet | 143.4 | 0.369 | 0.005 | 0.0 | 0.0 | 0.0 | 0.0 | 4.3 | 7.8 | 0.047 |

### Marks by kind (mean per run)

| arm | dose_ok | dosed | need_more | not_tumour | slow_uptake | tumour_here | washout |
|---|---|---|---|---|---|---|---|
| S-solo | 45.1 | 27.0 | 20.7 | 6.0 | 6.2 | 43.4 | 9.1 |
| S-marks | 31.4 | 28.6 | 25.2 | 5.0 | 6.4 | 43.6 | 10.5 |
| S-talk | 32.2 | 28.9 | 25.6 | 5.1 | 6.6 | 43.7 | 10.3 |
| S-marks-displaced | 38.6 | 27.4 | 22.4 | 4.9 | 6.0 | 42.7 | 9.1 |
| S-talk-shuffled | 31.7 | 28.7 | 25.4 | 5.0 | 6.5 | 43.7 | 10.4 |
| S-marks-perfect | 29.7 | 28.8 | 26.1 | 11.7 | 6.4 | 36.8 | 9.7 |
| S-marks-quiet | 32.7 | 28.0 | 21.8 | 5.0 | 6.0 | 40.0 | 9.9 |

## Quiet-rule pair

| arm | mean coverage | mean turns skipped |
|---|---|---|
| S-marks | 0.200 | 0.0 |
| S-marks-quiet | 0.198 | 7.8 |

## Verdict statistics (coverage, paired by case and seed)

Case-clustered bootstrap, 10000 resamples, seeded. Holm over the two primary comparisons.

| comparison | pairs | median diff | 95% CI | p | Holm p | dev-A median | dev-B median | vs own control: median [CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| (a) S-marks - S-solo | 80 | 0.0040 | [-0.0043, 0.0119] | 0.3036 | 0.3036 | 0.0050 | 0.0033 | 0.0025 [-0.0043, 0.0115] | fail |
| (b) S-talk - S-solo | 80 | 0.0105 | [0.0016, 0.0194] | 0.0296 | 0.0592 | 0.0127 | 0.0049 | 0.0000 [0.0000, 0.0000] | fail |

| comparison | median >= 0.03 | Holm p < 0.05 | same sign dev-A/B | beats control |
|---|---|---|---|---|
| (a) S-marks - S-solo | False | False | True | False |
| (b) S-talk - S-solo | False | False | True | False |

Sharing helps scripts (at least one primary passes): **False**

### Secondary (no threshold)

| comparison | pairs | median diff | 95% CI | p |
|---|---|---|---|---|
| S-talk - S-marks | 80 | 0.0000 | [0.0000, 0.0008] | 1.0000 |

### Diagnostic: S-marks-perfect - S-solo (not a verdict arm)

| pairs | median diff | 95% CI | p |
|---|---|---|---|
| 80 | 0.0020 | [-0.0049, 0.0116] | 0.7361 |
