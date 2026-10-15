# v4.3 Phase 0 findings (hand-written)

Date: 2026-10-15. Cost: $0. Development cases only. No model run, no pod, no paid call, no chain write, no key.
Each number cites its file. Paths are under `docs/research/v4.3-20261015/` unless noted.
Code frozen at commit `e212d7e6cf874965bcbe82aa637e7488cd771db1` (`freeze.json`, freeze hash `bac6746fe045…`).

## 1. The short version

- The ant is fixed. In a world with no hidden facts, the colony ant alone now matches the simple script S (gate A passed).
- In the foraging world at the primary "mid" level, **the sharing verdict fails**. Both comparisons fail the v4.2 bar.
- The main sharing comparison (S-marks minus S-solo) is a close fail. Its median gain is 0.0293 and the bar is 0.03. It meets the other three conditions.
- Talking adds nothing measurable. S-talk equals S-marks.
- The sharing effect gets smaller as uptake gets patchier. That is the opposite of what the idea predicted.
- No replication was run, because nothing passed.

## 2. Gate A: fix the ant (F2 and F3 off)

Source: `gateA/r0/gate_a_summary.json`, `gateA/r1/gate_a_summary.json`, `gateA/r1-devB/gate_a_summary.json`.

| measurement | split, seed | S | S-solo | S-solo minus S | bar (>= -0.05) |
|---|---|---|---|---|---|
| 0 (first) | dev-A, 1 | 0.688 | 0.615 | -0.073 | fail |
| 1 (revision 1) | dev-A, 1 | 0.688 | 0.692 | +0.004 | pass |
| 1 (held out, run once) | dev-B, 1 | 0.685 | 0.710 | +0.025 | pass |

- v4.2's S-solo was 0.266 against S 0.591 (v4.2 `FINDINGS.md`). That world was different (F2 and F3 on, the raw look).
- Measurement 0 failed. The saved runs showed why. Off-target drug was no longer the problem (0.567 against 0.540). The ant left drug unspent: it released 76% of its drug and S released 99%. 153 of 240 ants stood at a dose-OK spot for 4 or more turns with payload left. (My analysis of `gateA/r0`; the decision note is in the vault: `decisions/2026-10-15-gateA-revision-1.md`.)
- Revision 1 is a job rule: when the dose job reads its own dose at or above D_t, the ant moves to the nearest un-dosed voxel in its zone. Revisions used: 1 of 2.
- S keeps v4.1's colony-wide visit flags. S-solo has per-ant flags. This favours S slightly.

### The combined estimate (A1)

Source: `combined/coefficients.json`, `combined/calibration.json`.

- It is a logistic fit on scan p, look and their product. Fit on 5,760 dev-A looks (30 cases). Coefficients: -0.604, 0.842, 0.484, 0.080.

| split | score | ECE | AUROC |
|---|---|---|---|
| dev-A (in sample) | combined | 0.014 | 0.969 |
| dev-A (in sample) | scan alone | 0.025 | 0.958 |
| dev-B (held out) | combined | 0.064 | 0.972 |
| dev-B (held out) | scan alone | 0.029 | 0.981 |

- **The combined estimate did not beat the scan on held-out dev-B.** The honest reading is "no demonstrated gain". dev-B has 10 cases and 1,920 looks on 397 unique voxels, so a 0.009 AUROC gap is probably noise. Any interval must resample by case.
- Circularity check on the gate-A looks (`gateA/circularity-devA.json`, `gateA/circularity-devB.json`): the combined estimate matches the scan (AUROC 0.966 against 0.960 on dev-A bundles; 0.984 against 0.982 on dev-B bundles). The raw look is far worse (0.764 and 0.749).
- What fixed the ant was therefore mostly the rules (release only on the estimate, verify bounds, thresholds, the move-on), not a better sensor.

## 3. Headroom

Source: `headroom/headroom.json`. Mid level, dev-A, seed 1, S-solo and S′ only. Report only.

- S′ beats S-solo by 0.10 or more on 96.7% of cases (29 of 30). Mean gap 0.245, median 0.2245.

## 4. The verdict at mid

Source: `verdict-mid/REPORT.md`, `verdict-mid/stats.json`. 40 dev cases, seeds 5 and 6, 720 episodes, one attempt.
S′ was at least as good as every arm on 80 of 80 case-seeds (`verdict-mid/verdict_checks.json`).

| Arm | Mean coverage |
|---|---|
| S | 0.482 |
| S-solo | 0.390 |
| S-marks | 0.423 |
| S-talk | 0.423 |
| S-marks-displaced | 0.414 |
| S-talk-shuffled | 0.423 |
| S-marks-perfect (diagnostic) | 0.423 |
| S-marks-quiet | 0.416 |
| S′ | 0.633 |

| Comparison | Median | 95% CI | Holm p | dev-A / dev-B median | Beats control | Result |
|---|---|---|---|---|---|---|
| S-marks minus S-solo | +0.0293 | [0.0152, 0.0454] | 0.0004 | +0.0292 / +0.0307 | yes (+0.0104 [0.0002, 0.0155]) | **fail** (median under 0.03) |
| S-talk minus S-solo | +0.0293 | [0.0151, 0.0452] | 0.0004 | +0.0292 / +0.0307 | no (difference 0) | **fail** (median and control) |

- The bar (v4.2 section 9.3, unchanged): median gain at least 0.03, Holm p under 0.05, same sign on dev-A and dev-B, beats its own control.
- S-marks missed the median by 0.0007. The goal says a fail is a fail and no redesigns. I did not touch anything and ran nothing again.
- The mean gain is 0.034 for both (`verdict-mid/stats.json`). The bar uses the median.
- S-talk minus S-marks: median 0.0000, interval [0.0000, 0.0000]. Talking changed almost nothing: 5.2 asks per run.
- **Replication on seeds 7 and 8 was not run.** The goal runs it only if a comparison passes.

## 5. The patchiness curve (secondary, no pass or fail)

Source: `curve/curve.md`, `curve/curve.json`. Arms S, S-solo, S-marks, S-talk, S′, same cases and seeds.

| level | S | S-solo | S-marks | S-talk | S′ | S-marks minus S-solo (mean, 95% CI) |
|---|---|---|---|---|---|---|
| low | 0.577 | 0.490 | 0.545 | 0.545 | 0.741 | +0.055 [0.040, 0.069] |
| mid | 0.482 | 0.390 | 0.423 | 0.423 | 0.633 | +0.034 [0.022, 0.046] |
| high | 0.375 | 0.286 | 0.307 | 0.308 | 0.496 | +0.022 [0.013, 0.030] |

- Realised mean u (voxel-weighted, 40 dev cases, `uptake/realised_u.json`): rim 0.93 / 0.85 / 0.78, core 0.59 / 0.39 / 0.19, edema 0.88 / 0.75 / 0.63 for low / mid / high.
- The sharing gain is positive at every level and largest when the tumour is least patchy. More patchiness did not make sharing more useful here.

## 6. Why (from the saved runs; an association, not an intervention)

Source: `diagnostics/diagnostics_numbers.json` (mid, 80 runs per arm).

- **Gap to S′ by cause** (coverage points, `gap_to_s_prime_by_cause`; each pool is its share of the drug times S′'s yield, so it is an estimate that assumes the colony arms' tumour-side yield equals S′'s):

| arm | gap | false-alarm releases | poor-patch releases | unreleased drug | remainder |
|---|---|---|---|---|---|
| S-solo | 0.244 | 0.017 | 0.062 | 0.098 | 0.067 |
| S-marks | 0.210 | 0.020 | 0.062 | 0.072 | 0.057 |

- False alarms are small now (they were about two thirds of the gap in v4.2).
- Poor-patch releases take about 12% of released drug in every arm (S-solo 12.8%, S-marks 12.3%). Sharing barely lowered it.
- Probes: S-solo 28.5 per run, S-marks 30.4. Others' uptake marks saved S-marks 5.2 probes per run.
- The uptake marks are accurate. Good-uptake marks: 24.8 of 28.0 sit on true u at or above 0.7 and none on true u below 0.4. Slow-uptake marks: 9.0 of 12.6 sit on true u below 0.4 and none above 0.7 (`uptake_mark_accuracy_per_run`).
- The marks are right, but the ants rarely act on them to change a costly decision. Removing one mark changed the receiver's next action 10.7% of the time (uptake marks 5.8%), and 89% of marks changed nothing (`effect/effect.json`). Mean coverage change from removing one item was 0.0002. For S-talk: 15.7% and 80%, mean 0.0006.
- **Perfect marks did not beat real marks** (0.423 both). So better mark content did not help.
- **The displaced control still beat S-solo** (0.414 against 0.390). So part of the sharing gain does not need correct mark positions. One candidate is the queen's zone need: in S-marks it differs from S-solo's fixed need on 136 of 192 turns per run (`shared_inputs`, `turns_need_differs_from_solo`), and displaced marks stay inside their zone. This is a hypothesis. No run separated it.
- The judgement marks (`tumour_here`, `not_tumour`) almost never fire under the v4.3 rules (0.0 `not_tumour` per run). So the "correct the scan" channel is silent, and the track-record honesty correlation is n/a (fewer than 3 usable ants).
- Sharing changes many actions (59.6% of turns differ from S-solo) and the coverage gain is small and about the same with shuffled or perfect marks. Most of that is drift after the first difference.

## 7. What broke or surprised us

- Gate A measurement 0 failed. Revision 1 fixed it (section 2).
- A full-suite run failed once from a test-order dependence (a tests/v4_2 test edits the arm registry). Fixed in the test only.
- An independent review found four bypasses in the freeze guard (an unguarded script, an incomplete hash, any config path accepted, untracked files not hashed). They were fixed and re-reviewed before the freeze.
- **Deviation from the freeze rule.** Before the freeze, colony arms ran on dev case BRATS_016 in tests: my own legacy-identity test (60 ticks, one seed) and three inherited v4.2 smoke tests (10 to 60 ticks, temp dirs). No result from them was read or used. The goal says tests use the phantom. My brief allowed the one real case. See the vault log, 2026-10-15 "DISCLOSURE".
- The combined estimate has no demonstrated gain over the scan on held-out data (section 2).
- On the phantom, the probe-and-hold rule lowered coverage (it costs turns). This was not tuned.

## 8. What this means for the next step

- The scripted ants do not yet earn a model-ant colony on this world. S-marks is 0.0007 short of the bar, so this is a near miss, not a clear no. I will not call it a pass.
- The ant is no longer the bottleneck. S′ still leads by 0.21 to 0.24 coverage points. About 13% (S-marks) to 18% (S-solo) of the drug is never released (total drug 5,153 per run, released 4,475 and 4,245: `diagnostics/diagnostics_numbers.json`, `poor_tissue_release`).
- Ideas for a later iteration (none implemented): make the judgement marks fire where the scan is wrong; run an ablation that fixes the queen's need to separate it from the marks; give the ants a rule that uses good-uptake marks to choose where to dose, not only whether to probe.
