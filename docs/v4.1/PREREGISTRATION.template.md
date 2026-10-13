# Antelligence v4.1 preregistration

Code commit SHA: `$code_sha`

## Pinned inputs

| Input | Value |
|---|---|
| config/v4_1/world.yaml sha256 | `$world_sha256` |
| config/v4_1/harness.yaml sha256 | `$harness_sha256` |
| conformal thresholds sha256 | `$conformal_sha256` |
| split file sha256 | `$split_sha256` |
| router model | `$router_model` |

### Model revisions

$model_revisions

### Contracts (address, code hash)

$contracts

## Primary hypotheses

The unit is the case: the mean over its seeds. They are tested in a fixed sequence: P2 is tested only if P1 is significant. Both use alpha = 0.05.

- **P1, the board helps.** SW-B coverage > SW-noBoard coverage on 80 evaluation cases.
  - Test: two-sided Wilcoxon signed-rank, with a 95% BCa bootstrap interval of the mean paired difference.
  - P1 passes only if p < 0.05 **and** the mean difference is positive. A significant harm is reported as a finding, and it doesn't unlock P2.
- **P2, small matches frontier.** SW-B is non-inferior to FW on the 40 FW cases (seed 1), with a margin of -3 coverage points.
  - Test: one-sided 95% bootstrap lower bound of (SW-B - FW) > -3.
  - Implementation: the lower bound is the lower end of a two-sided 90% BCa bootstrap interval of the mean paired case difference. Coverage points are percentage points (0.01 of coverage).
  - Cost and seconds per decision are reported next to it.
  - If P1 is not significant, P2 is still reported, with its interval, as exploratory.
- **P3, verifiability (designed properties, measured).** A detection table by attack, position and verifier condition.
  - Expected:
    - **B:** detects no operator attack.
    - **OTS:** detects T4 only, because the preregistration's proof was published at the lock. It misses T1-T3: the operator can stamp new versions and withhold old proofs.
    - **R:** detects T1-T4 in the anchored prefix, at audit time, by finding extra entries in the run's digest series.
    - **C:** detects T1-T4 in the anchored prefix. The contract rejects forks at write time.
    - **Unanchored tail:** nothing detects it.
  - Any departure from these expectations is a finding (a bug, or a surprise).
  - The informative parts are the anchor latency, the cost, the tail size, and write-time rejection vs audit-time detection.
- **P4, chain as live bus.** SW-D vs SW-B on 12 cases: the coverage difference with a 95% interval, plus the overhead in wall-clock time to 50% coverage. No significance claim; 12 cases are too few.

Bootstrap: $bootstrap_resamples resamples, fixed seed $bootstrap_seed (scipy.stats.bootstrap).

## Secondary (reported, not used to declare a win)

- **S1:** SW-noQueen vs SW-B: the value of the queen.
- **S2:** SW-staticQueen vs SW-B: the value of the self-improvement loop.
- **S3:** SW-Bonsai vs SW-B on 40 cases.
- **S4:** SW-Byz-local vs SW-Byz-chain, and both against SW-B. The preregistered prior is no coverage difference between local and on-chain reputation.
- **S5:** perception Dice, voxel ECE, d1 local-look ECE, and the truth share outside the crop.
- **S6:** defer rate per arm, alarms, and conformal coverage achieved on the audit sample.
- **S7:** citation validity, uncited decisions, and unread entries.
- **S8:** off-target share and time to 50% coverage, for all arms.
- **S9:** normalised score (coverage - S)/(S' - S), as a ratio of means only, on cases where S' - S >= 0.15.
- **S10:** SW-B-missingFLAIR vs SW-B.
- **S11:** cost: dollars, tokens, pod-minutes and gas per arm.
- **S12:** the latency benchmark (4.7).

## Power

State the minimum detectable P1 difference from the development paired SD (Wilcoxon, ARE-adjusted normal approximation: (z(1-alpha/2) + z(power)) * SD / sqrt(n * 3/pi), power $power). Report a null as "not detected at this sample size", with its interval.

## Arms

| Arm | Workers | Queen | Board | Cases x seeds |
|---|---|---|---|---|
| S | none (script) | none | none | 80 x 2 |
| S' | none (oracle) | none | none | 80 x 2 |
| **SW-B** | d1-3B + Qwen3.5-4B + Qwen3.5-9B, routed | promoted harness | B, anchored to OTS, R, C | 80 x 2 |
| **SW-noBoard** | same | promoted harness | off (4.3) | 80 x 2 |
| **FW** | DeepSeek V4.1 Flash for every decision | promoted harness | B, anchored | 40 x 1 |
| **SW-D** | as SW-B | promoted harness | D (board on-chain) | 12 x 1 |
| SW-noQueen | as SW-B, router only; static assignments from S's rule | none | B | 40 x 1 |
| SW-staticQueen | as SW-B | base harness | B | 40 x 1 |
| SW-Bonsai | d1-3B + Qwen3.5-4B + Bonsai 8B | promoted harness | B | 40 x 1 |
| SW-Byz-local | as SW-B; 2 of 8 squads post false evidence | promoted harness | B, local reputation | 20 x 1 |
| SW-Byz-chain | as SW-Byz-local | promoted harness | B, plus `ReputationRegistry` | 20 x 1 |
| SW-B-missingFLAIR | as SW-B, degraded perception | promoted harness | B | 20 x 1 |

Seeds set the entry sites. Arms that share cases use the same seeds. Evaluation order is the split-file order; the 40-case, 20-case and 12-case subsets are the first cases in that order.

## Budgets

- **Nous (DeepSeek V4.1 Flash):** $$8, with a hard stop at $$7.50 by the ledger. Typical total about $$5.5.
- **RunPod:** $$10, with a hard stop at $$9.50, by the pod script's cost records. One L40S at a time.
- **Testnet ETH:** 0.04 ETH at most (expected about 0.025 ETH).
- Phase 0 sandbox: 4 rounds x 60 runs.

## Cut orders

**Nous cut order:**
1. sandbox rounds down to 3;
2. FW down to 30 cases;
3. SW-B-missingFLAIR, SW-Bonsai and SW-staticQueen down to 20 cases.

**RunPod cut order:**
1. SW-B-missingFLAIR;
2. SW-staticQueen;
3. SW-noQueen;
4. the SW-Byz arms down to 10 cases each;
5. SW-D down to 8 cases.

Never cut SW-B or SW-noBoard below 80 x 2, or FW below 30 x 1. If that still isn't enough, ask Eshaan at G1.

## Explained-null checklist (thresholds fixed now)

- the board was not read (unread entries above 50%);
- briefs were stale (median entry age above 30 ticks);
- deferral was outside its band (5% to 40% per run) in more than 25% of runs;
- the router sent more than 80% of decisions to one model;
- perception calibration failed (development ECE above 0.10 after the refit).

Each item's computation (mean or median, pooled or per run) is specified in `docs/v4.1/CELLS.md`.

## Changes after the lock

None, except cuts made for a budget hard stop. Those are logged as deviations, never added silently.
