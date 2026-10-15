# v4.2 Phase 0 findings (hand-written)

Date: 2026-10-14. Cost: $0. Dev cases only. No evaluation case was touched.
Each number cites its file (paths are under `docs/research/v4.2-20261014/` unless noted).

## 1. The short version

- Sharing did not help the scripted ants in this world. Both primary comparisons failed, three times.
- That means a model-ant colony should not be paid for yet, on this world and these scripts.
- The cause is not one thing. The close look (SimLook) is a worse sensor than the scan, and the scripts trust it. Even perfect marks changed half of all turns and did not move coverage.
- Throughput was never an engine problem. The slow part is waiting for the model servers.

## 2. Throughput (C0)

- v4.1 episodes spent 79% of their wall time waiting on model calls, 0% in server time, and 21% unlogged (`throughput/attribution.json`, set `calib1`).
- Our engine costs about 1.7 s of CPU per SW-B episode on fake backends. The target was under 30 s (`throughput/PROFILE.md`).
- With sleeping fakes at real latencies, 8 worker processes scale at 1.0 of linear. 16 episodes in one process scale at 0.90. The target was 0.7 (`throughput/profile.json`).
- One fix kept: a cached signing key. Metrics were identical on 10 episodes (`throughput/results/scripted-after.json`).
- "About tick 200" in v4.1 was a board-entry count, not a tick. Every finished episode ends at tick 120 (`throughput/results/ticks.json`).

## 3. Hidden facts and the stand-in look (W1)

- SimLook was built from 7,680 real d1 looks that v4.1 recorded on 40 dev cases. None were dropped (`simlook/cells_v4_2.json`, `simlook/reliability.json`).
- d1 reads high on healthy tissue. Overall AUROC of look against truth is 0.78, and the scan's is 0.96 (`simlook/reliability.json`). This matters for everything below.
- F2 (poor uptake) and F3 (washout) are off by default. Off reproduces v4.1 exactly (tests in `tests/v4_2/w1/`).

## 4. Tuning

- Stage 1 (12 points, v4.1 drug budget): no point met the median bands (`tuning/tune1_results.json`). S-solo medians were 0.12 to 0.18, S′ medians 0.40 to 0.60.
- Stage 2 (12 more points, drug budget 1.5 to 3 times larger): three points qualified. The rule picked point 4: `u_core` 0.5, `f3_decay` 0.04, `f3_share` 0.1, drug budget 1.5 times (`tuning/tune2_results.json`). Medians: S-solo 0.277, S′ 0.797.
- Two cautions. S-solo's median is close to the lower band edge of 0.25. Points 4 and 8 differ by 0.001 in median gap.
- The drug budget was tuned on dev-A seed 1 with S-solo and S′ only. No sharing arm ran during tuning.

## 5. Verdict

Seeds 3 and 4 on all 40 development cases (80 pairs). Source: `attempt1-frozen/REPORT.md`.

| Arm | Mean coverage |
|---|---|
| S | 0.591 |
| S-solo | 0.266 |
| S-marks | 0.276 |
| S-talk | 0.276 |
| S-marks-displaced | 0.264 |
| S-talk-shuffled | 0.277 |
| S-marks-perfect (diagnostic) | 0.269 |
| S-marks-quiet | 0.274 |
| S′ | 0.788 |

| Comparison | Median difference | 95% interval | Holm p | Result |
|---|---|---|---|---|
| S-marks − S-solo | +0.012 | [0.000, 0.018] | 0.076 | fail |
| S-talk − S-solo | +0.009 | [−0.005, 0.021] | 0.379 | fail |

- The bar was a median of at least 0.03 and a Holm p below 0.05. Neither came close on the median.
- S′ was at least as good as every other arm on all 80 pairs (`attempt1-frozen/verdict_checks.json`).
- Redesign R1 (stronger facts) also failed: +0.004 (Holm p 0.30) and +0.011 (Holm p 0.059) (`attempt2-R1/REPORT.md`).
- Redesign R2 (R1 and 25% less drug) also failed: +0.002 and +0.003 (Holm p 0.57 for both) (`attempt3-R2/REPORT.md`).
- Under R1 and R2 the tuning bands no longer held on dev-A seed 1: S-solo 0.195 and 0.148, S′ 0.579 and 0.459 (`band-checks/`). The goal said not to re-tune, so we did not.
- I read "25% less drug" as the frozen drug budget times 0.75. The goal does not say which base.
- No attempt passed, so there was nothing to replicate on seeds 5 and 6.

### What the controls say

- S-marks-displaced (0.264) and S-talk-shuffled (0.277) match their real versions. So mark content and message routing were not what moved anything.
- S-marks-perfect (0.269) did not beat S-solo (0.266). With this world and these scripts, better shared information was not shown to help.
- Removing one mark or message changed the receiver's next action 13% of the time for S-marks and 9% for S-talk. Spam rates were 87% and 68%. Mean coverage change was about zero (`attempt1-frozen/effect.json`).
- S-talk matched S-marks closely. Answers are used, but they rarely change a release.

## 6. Why (from the saved runs)

See `DIAGNOSTICS.md`. The numbers were rechecked by an independent reviewer. The wording is deliberately cautious, because no intervention was run.

- Perfect marks changed 49.6% of turns against S-solo and coverage stayed flat.
- Far healthy tissue reads look 0.5 or more on 34.1% of turns. The scan does so on 2.2%.
- About 65% to 73% of the drug that missed the tumour was released on those false positives. This is an association, and the split assumes S's yield.
- 98% of the marks that tried to correct the scan were wrong.
- The limit looks joint: the world and the scripts. No test separates them.

## 7. Learning check (stretch)

- Carrying job thresholds across 30 dev-A runs made coverage worse on dev-B: median −0.25, interval [−0.32, −0.21] (`learning/results.json`).
- The thresholds drifted to about 1.0, so the ants stopped acting. It is a plumbing check, not a finding.
- The notebook round trip (write, cap, archive, freeze, reload) works.

## 8. What broke or surprised us

- v4.1 settings had pre-allowed ten `scripts/v41_*.py` programs. We removed that allow-list in step 0.
- The v42 agent types were not registered at first. We used general-purpose agents with the role text pasted in.
- Our first colony scripts had an invented rule that dropped whole zones. It hid the true size of the sharing effect. A reviewer caught it and it was removed.
- S-talk's first version was almost inert. We changed how answers are used (decision note `2026-10-14-stalk-answer-consumer.md`) before any verdict run.
- A card band rule gives "usually right" after a single confirmed claim (`blockers.md`).

## 9. What Phase 1 needs

- Do not start with model ants on this world. Fix the close look first. Then check that S-solo comes within 0.05 of S before any sharing arm runs (`DIAGNOSTICS.md`, order of work).
- Give sharing something the scan lacks. Today every ant reads the same SimLook value at a voxel, so there is nothing private to share.
- Projection and tokens for a later Phase 1 are in `phase1/PROJECTION.md` and `phase1/TOKENS.md`.
