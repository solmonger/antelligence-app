# v4.2 Phase 0 report (generated)

Generated from `results.json` by `scripts/v42_report.py`. Tables only. No interpretation here.

## Run info

| key | value |
|---|---|
| arms | ["S", "S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S'", "S-marks-perfect", "S-marks-quiet"] |
| ceiling_check | {"n_case_seeds": 80, "ok": true, "required": 0.95, "share_s_prime_ge_all": 1.0, "violations": []} |
| config | "config/v4_2/world_R2.yaml" |
| dry_overrides | {"f2": null, "f3": null, "ticks": null} |
| frozen_facts | {"f1_simlook": true, "f1_source": "boards", "f2": true, "f2_share": 0.35, "f2_source": "A", "f3": true, "f3_decay": 0.08, "f3_share": 0.2, "u_core": 0.3} |
| incomplete | [] |
| n_runs | 720 |
| seeds | [3, 4] |
| source_dir | "/Volumes/WD_BLACK/antelligence-v4.2/runs/verdict-R2" |
| workers | 5 |

## Primary metrics (mean over runs)

The time-to-50% mean covers only runs that reached 50%; the last column counts the runs that never did.

| arm | runs | coverage | off-target share | time to 50% coverage | drug on non-tumour | runs never at 50% |
|---|---|---|---|---|---|---|
| S | 80 | 0.341 | 0.444 | 61.000 | 1524.181 | 76 |
| S-solo | 80 | 0.136 | 0.664 | n/a | 2103.424 | 80 |
| S-marks | 80 | 0.138 | 0.665 | n/a | 2150.054 | 80 |
| S-talk | 80 | 0.138 | 0.666 | n/a | 2160.746 | 80 |
| S-marks-displaced | 80 | 0.137 | 0.671 | n/a | 2151.452 | 80 |
| S-talk-shuffled | 80 | 0.138 | 0.666 | n/a | 2152.435 | 80 |
| S-marks-perfect | 80 | 0.137 | 0.675 | n/a | 2175.347 | 80 |
| S-marks-quiet | 80 | 0.138 | 0.660 | n/a | 2110.064 | 80 |
| S' | 80 | 0.446 | 0.351 | 92.273 | 1152.727 | 58 |

## Coverage at queen steps (mean)

| arm | t=0 | t=30 | t=60 | t=90 | t=120 |
|---|---|---|---|---|---|
| S | 0.000 | 0.076 | 0.267 | 0.324 | 0.341 |
| S-solo | 0.000 | 0.004 | 0.040 | 0.100 | 0.136 |
| S-marks | 0.000 | 0.004 | 0.040 | 0.098 | 0.138 |
| S-talk | 0.000 | 0.004 | 0.040 | 0.099 | 0.138 |
| S-marks-displaced | 0.000 | 0.004 | 0.039 | 0.100 | 0.137 |
| S-talk-shuffled | 0.000 | 0.004 | 0.040 | 0.099 | 0.138 |
| S-marks-perfect | 0.000 | 0.004 | 0.040 | 0.097 | 0.137 |
| S-marks-quiet | 0.000 | 0.004 | 0.040 | 0.098 | 0.138 |
| S' | 0.000 | 0.027 | 0.271 | 0.406 | 0.446 |

## Colony metrics (mean per run)

| arm | marks | share strong | share disputed | asks sent | answered | lapsed | refused | quorum events | turns skipped | honesty r |
|---|---|---|---|---|---|---|---|---|---|---|
| S-solo | 154.9 | 0.427 | 0.005 | 0.0 | 0.0 | 0.0 | 0.0 | 3.8 | 0.0 | n/a |
| S-marks | 148.5 | 0.340 | 0.005 | 0.0 | 0.0 | 0.0 | 0.0 | 3.6 | 0.0 | 0.194 |
| S-talk | 150.1 | 0.343 | 0.005 | 63.3 | 63.1 | 0.2 | 12.3 | 3.7 | 0.0 | 0.235 |
| S-marks-displaced | 148.9 | 0.041 | 0.003 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1 | 0.0 | 0.109 |
| S-talk-shuffled | 149.2 | 0.343 | 0.005 | 63.3 | 63.3 | 0.0 | 12.4 | 3.6 | 0.0 | 0.167 |
| S-marks-perfect | 146.6 | 0.344 | 0.016 | 0.0 | 0.0 | 0.0 | 0.0 | 3.2 | 0.0 | n/a |
| S-marks-quiet | 142.7 | 0.361 | 0.005 | 0.0 | 0.0 | 0.0 | 0.0 | 3.9 | 7.8 | 0.093 |

### Marks by kind (mean per run)

| arm | dose_ok | dosed | need_more | not_tumour | slow_uptake | tumour_here | washout |
|---|---|---|---|---|---|---|---|
| S-solo | 37.7 | 28.4 | 24.7 | 5.9 | 6.5 | 43.6 | 8.2 |
| S-marks | 24.8 | 30.2 | 29.8 | 5.2 | 6.7 | 43.5 | 8.3 |
| S-talk | 25.0 | 30.4 | 30.9 | 5.1 | 6.7 | 43.4 | 8.6 |
| S-marks-displaced | 29.3 | 29.4 | 28.8 | 4.7 | 6.3 | 42.6 | 7.8 |
| S-talk-shuffled | 25.1 | 30.2 | 30.2 | 5.2 | 6.7 | 43.5 | 8.3 |
| S-marks-perfect | 23.0 | 30.2 | 30.4 | 11.6 | 6.4 | 37.0 | 8.0 |
| S-marks-quiet | 26.9 | 29.6 | 25.4 | 4.9 | 6.3 | 40.5 | 9.1 |

## Quiet-rule pair

| arm | mean coverage | mean turns skipped |
|---|---|---|
| S-marks | 0.138 | 0.0 |
| S-marks-quiet | 0.138 | 7.8 |

## Verdict statistics (coverage, paired by case and seed)

Case-clustered bootstrap, 10000 resamples, seeded. Holm over the two primary comparisons.

| comparison | pairs | median diff | 95% CI | p | Holm p | dev-A median | dev-B median | vs own control: median [CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| (a) S-marks - S-solo | 80 | 0.0024 | [-0.0037, 0.0111] | 0.4666 | 0.5691 | 0.0024 | 0.0032 | 0.0021 [-0.0036, 0.0118] | fail |
| (b) S-talk - S-solo | 80 | 0.0032 | [-0.0041, 0.0122] | 0.2846 | 0.5691 | 0.0021 | 0.0057 | 0.0000 [0.0000, 0.0002] | fail |

| comparison | median >= 0.03 | Holm p < 0.05 | same sign dev-A/B | beats control |
|---|---|---|---|---|
| (a) S-marks - S-solo | False | False | True | False |
| (b) S-talk - S-solo | False | False | True | False |

Sharing helps scripts (at least one primary passes): **False**

### Secondary (no threshold)

| comparison | pairs | median diff | 95% CI | p |
|---|---|---|---|---|
| S-talk - S-marks | 80 | 0.0000 | [0.0000, 0.0013] | 1.0000 |

### Diagnostic: S-marks-perfect - S-solo (not a verdict arm)

| pairs | median diff | 95% CI | p |
|---|---|---|---|
| 80 | 0.0015 | [-0.0057, 0.0071] | 0.6077 |
