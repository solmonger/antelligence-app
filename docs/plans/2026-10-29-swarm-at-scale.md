# Swarm intelligence at scale: what we learned, and what to build next

Date: 2026-10-29. Status: **proposal for review**. Nothing here is preregistered yet. No model call, no paid call,
no chain write and no key were used to write it. The one new measurement is the $0 scripted probe in section 5.

Base: `hermes/v4.31-a2-20261016` (the newest work). Every number below cites the file it comes from.

## 1. The short answer

Across every version so far one pattern holds: **the useful part of the swarm was never "more models
thinking together". It was the cheap, model-free layer that decides what gets through** (the merge, the
verifier, the admission rules). Teams of models did not beat their best member. Shared marks were accurate but
rarely changed a costly decision. The merge step, on the other hand, turned 0/19 plans into 16/19 for every model.

So the proposal is to make that layer the thing we scale and study:

> **D1 (top pick). Trusted gate, untrusted swarm.** Measure how many bad ants a swarm can absorb before it does
> worse than ants working alone (its *breakdown point*), and whether that number holds as the swarm grows from 8 to
> thousands of ants, with and without a model-free gate. Then find the decay rate at which one wrong mark stops
> spreading. This is cheap (a scripted ant decision costs about a quarter of a millisecond), it reuses most of the
> kernel, a null result is still a clear safety statement, and the same gate is what the ESP32 robots will need.

The other two directions, ranked:

- **D2. The diversity dividend.** Swarms can only beat their best member when their errors are independent. Build a
  world with a dial for how correlated the ants' errors are, check that a simple formula predicts the gain, then
  measure our real models on that dial. This explains the Oct 10 colony null and decides which SLMs are worth
  tuning.
- **D3. Hidden messages in marks.** Test whether ants can pass secret information to each other through where and
  how strongly they leave marks, and which engine rules (rounding, noise, rate limits) close that channel. Very new
  and very safety-relevant, but it needs D1's gate first.

## 2. What every past version tried and showed

### 2.1 The original app (`main`)

| Technique | What it did | Result |
|---|---|---|
| LLM ants foraging (`backend/simulation.py`) | ants call a model per move, rule fallback; optional queen | demo only, no measured comparison |
| Tumour nanobots (`backend/nanobot_simulation.py`, `biofvm.py`) | trail, alarm and recruitment pheromone fields; optional LLM queen | pheromones vs none: **0% improvement**, secretion was not wired (`SPRINT.md` phase 2) |
| On-chain provenance | 5 public fields per run | moved from Base Sepolia to **ZKsync Era Sepolia** (chain 300); all 4 contracts deployed, one end-to-end run at `proof_staged` (`docs/status/BLOCKERS.md`, `docs/status/e2e-zksync-era-sepolia.json`). `VISION.md`, `ARCHITECTURE.md` and `SPRINT.md` still say Base Sepolia (stale). |

### 2.2 DeSci paper, E1 to E15 (Sept, `docs/research/desci-paper-20260915/PAPER-v3.md`)

| Exp | Technique | Result |
|---|---|---|
| E1, E2 | memory + model-free verifier under silent drift | 2.6 to 6.9 times cheaper, **0 unsafe acts** (a cache without the verifier: 120 to 156 per cell); verifier 58/58 |
| E3 | bounded transport (admission) vs plain message passing | **blocked 6/6 attacks** |
| E5 | partial knowledge: gossip vs hive consult | both 48/48; hive used fewer messages but more bytes (226 KB vs 74 KB) |
| E7, E9, E10 | shared hive memory in binary foraging (hermes-4-405b, grok-4-fast) | sweep moves **−16%** (p = 0.012), replicated; communication added nothing on top; full spatial version 1/20 (capability floor) |
| E11 | swarm-proposed task DAGs | 3/20 vs solo 2/20 (p = 1.0); 63% of swarm nodes redundant |
| E13 | chain-prioritised foraging (glm-5.3-flash) | **−13%** moves (p < 0.001), success at ceiling |
| E12, E14 | partitioned DAGs, merge repair | voided by a set-up mistake (3 agents for 4 samples) |
| **E15** | partitioned DAGs + **deterministic merge** | solo 8/20, raw 0/20, **merged 16/20** (sign test p = 0.008) |

### 2.3 Workbench, W2 and the colony (Sept to Oct 10)

| Study | Technique | Result |
|---|---|---|
| Cloud study (Sept 9, `docs/FUGU_RESEARCH.md`) | vote, refine, evidence exchange on 20 Q&A items | **vote 16/20** best; exchange 8/20; 40/100 cells lost to format errors |
| W2 benchmark (`hermes/m0-w2-benchmark-20261010`) | 6 team protocols vs a single model, 2,831 cells | no team beat a single model; evidence exchange **−28 points**; planning: raw plans 0/19, **after the model-free merge 16/19 for every model** ("reflects the merge step, not model quality") |
| Colony "v3 Track 1" (**Oct 10**, `hermes/track1-colony-experiment-20261010`) | mixed colony with reputation, correlation discount, verified admission, vs best member at matched calls | **null**: finance −30 (local) and −16 (Haiku) points vs the best member; medical +4 (not significant). Cause: failed checks zeroed votes that were right 59% to 77% of the time; weak members (Phi 4%, Gemma 8%). Three copies of one model behaved like **1.1** independent voters, three different models like **2.05**. |

### 2.4 The v4 line (Oct 11 to 16)

| Version | Technique | Result |
|---|---|---|
| v4 pilot (`docs/research/v4-pilot-20261011/FINDINGS.md`) | decision-model workers + queen + escalation vs a strong model on every agent | unclear. Colouring: **workers with no queen did best** (G 1.223 vs DS-all 1.026); the queen and escalation each made things worse; reports 2.4% true; escalation fired 99%; Laya unusable; 9B beat 27B |
| v4.1 (`docs/research/v4.1-20261013/DESIGN-anchors.md`) | tamper-evident board: hash chain, Rekor, Bitcoin timestamps, write-once registry on ZKsync | built, fail-closed verifier; the chain is the only uniqueness guarantee |
| v4.2 (`docs/research/v4.2-20261014/FINDINGS.md`) | scripted ants sharing marks or talk in the tumour world | **fail** three times (+0.012 vs a 0.03 bar); perfect marks did not help; every ant read the same sensor, so nothing private to share; 79% of wall time spent waiting on model calls |
| v4.3 (`docs/research/v4.3-20261015/FINDINGS.md`) | fixed ant, foraging uptake marks, talk | **near miss**: +0.0293 vs the 0.03 bar; talk = marks; marks accurate but 89% changed nothing; displaced marks still helped (the queen's zone need is the suspect); gain *shrank* as the tumour got patchier |
| v4.31 (`docs/research/v4.31-20261016/`) | real-unit physics world | the gap between the script and the oracle is robust only at low thresholds; world science, not swarm science |

### 2.5 What is still unfinished

- v4.3's own next steps were never run: fix the queen's need to separate it from the marks; make judgement marks
  fire where the scan is wrong; use good-uptake marks to choose where to dose.
- Colony: soft-weighting failed checks instead of zeroing them (can be tested free on cached replies); a competence
  floor for members.
- Proof: the mock proof is still mock (`proof_staged`); the contract fix "needs a redeploy".
- Hive: queen succession, structured vs raw memory, never tested with models.

### 2.6 The lessons, in one list

1. **Model-free layers carry the gains** (E1 to E3, E15, W2 planning).
2. **Mixing models does not beat the best member at matched calls** (W2, Track 1, pilot). Copies of one model are
   barely more than one voter.
3. **Sharing helps only when someone holds information others lack**, and even accurate marks rarely change a costly
   decision (v4.2, v4.3).
4. **Central control can hurt** (queen in the pilot, possible anchoring).
5. **The engine is not the bottleneck; model latency is** (v4.2 throughput).

## 3. What 2025 to 2026 research says (and what to borrow)

| Finding | Source | What we borrow |
|---|---|---|
| Multi-agent setups gain on parallel tasks and lose 39% to 70% on sequential ones; independent agents amplified errors up to 17.2×, versus 4.4× with a validating orchestrator | Kim, Liu et al., Google, "Towards a Science of Scaling Agent Systems" (arXiv 2512.08296) | place each world on their axes; a validation bottleneck is what our merge is |
| Majority vote explains most of debate's gain; debate cannot raise expected accuracy unless updates lean toward correction | Choi et al., "Debate or Vote", NeurIPS 2025 (arXiv 2508.17536) | the theory behind our marks null: marks need a correction-biased rule |
| One model's own samples beat a mix of models | Li et al., "Rethinking Mixture-of-Agents" (arXiv 2502.00674) | "best SLM × N samples" becomes a required baseline |
| A router over 7 SLMs barely beats the best one, though the right answer is usually somewhere in the pool | Zhang, Yao, "OrchSLM" (arXiv 2609.13470) | the bottleneck is picking, not generating: spend the swarm on checking |
| Two LLMs agree on about 60% of the questions both get wrong | Kim et al., "Correlated Errors in LLMs", ICML 2025 (arXiv 2506.07962) | measure our pool's error correlation (D2) |
| Group diversity halves as groups grow from 3 to 7; leaders anchor | Chen et al., "Diversity Collapse" (arXiv 2604.18005) | test queen-visible vs queen-blind phases |
| LLM agents agree far more than humans; sampling from a self-generated list fixes it | Ballestero et al., "Strategic Algorithmic Monoculture" (arXiv 2604.09502) | let the engine supply the randomness |
| One wrong seed spreads into false consensus; risk ≈ susceptibility × graph spectral radius ÷ decay | Xie et al., "From Spark to Fire" (arXiv 2603.04474) | decay is our pheromone half-life: sweep it (D1 part B) |
| Byzantine-tolerant consensus claims beyond n/3 faults, but tested on 7 agents and 10 questions | Zheng et al. (arXiv 2511.10400) | test the claim at hundreds of ants |
| Agents reach valid consensus 41.6% of the time; 0% with 3 Byzantine agents at N = 16 | Berdoz et al., ETH (arXiv 2603.01213) | report safety (no wrong act) and liveness (got done) separately |
| Taxonomy of multi-agent risks: miscoordination, conflict, collusion; risk factors incl. network effects | Hammond et al., Cooperative AI Foundation (arXiv 2502.14143) | name the risk factor each experiment measures |
| Shared environments enable secret collusion | Schroeder de Witt et al. (arXiv 2505.02077) | marks are a shared environment (D3) |
| LLM swarms run thousands of times slower than rule swarms | Rahman, Schranz (arXiv 2506.14496) | keep rule ants; call models only at decision points |

Gaps no one has filled, which a cheap ant testbed can: a measured epidemic threshold for stigmergic errors;
Byzantine robustness at hundreds to thousands of agents; covert channels through a shared environment rather than
messages; a matched-compute account of *where* swarm value comes from; the bridge from LLM consensus safety to
physical robots.

## 4. Three directions

Each must be new, build on what we have, and be designed so a null is still worth reporting.

### D1. Trusted gate, untrusted swarm (top pick)

**Question.** How many bad ants can a swarm absorb before it does worse than the same ants working alone, does that
breakdown point hold as the swarm grows, and how much does a model-free gate raise it?

**Why it is new.** Byzantine results for LLM agents stop at 7 to 16 agents and test consensus on questions. Cascade
results test small graphs. Nobody has measured a breakdown point across three orders of magnitude of swarm size, in
worlds with exact ground truth, with the gate itself (not the models) as the thing under test.

**What it builds on.** `kernel/admission.py` (budgets, causality, citations), `kernel/verifier.py` (EvidenceGate),
the v4.1 tamper-evident board, the colouring liar variant, the v4.2/v4.3 displaced and shuffled controls, E3 (6/6
attacks blocked), E15 and W2 (the merge carries the gain). Today's probe (section 5) is its first measurement.

**Design (to preregister).**

- Worlds: colouring (fast, exact), E15 task DAGs (planning), the v4.3 tumour foraging world (marks).
- Factors: swarm size N ∈ {8, 64, 512, 4096}; bad fraction f ∈ {0, 1, 5, 10, 30%}; bad behaviour ∈ {chatter (random
  reports), steady liar (one consistent false report), adaptive (knows the gate rule and stays just under it),
  colluding group (several liars back each other)}; gate ∈ {none, invariant gate, soft reputation (Track 1's
  Beta-Bernoulli, weighting not zeroing), oracle}.
- Part B, cascades: inject one wrong mark and sweep mark half-life against the neighbour graph. Does a sharp
  threshold appear where Xie et al. predict?
- Primary outcome: breakdown point f* (the smallest f where honest score falls below the solo, no-sharing baseline),
  per N and gate. Report safety (wrong acts) and liveness (unfinished agents) separately.
- Baselines required: solo ants; the oracle; and, for any model arm, "best SLM × N samples".
- Models enter only after the scripted curves exist: replace the honest ants with SLM ants at N = 64 and ask whether
  model ants have a lower f* than scripts (do they believe liars more?).

**What a null still tells us.** If the gate does not raise f*, the merge's past gains were not about filtering bad
input. If f* falls as N grows even with the gate, cascades win at scale: a concrete warning for any agent system
sharing memory. If the adaptive liar beats every gate, we have measured the price of a smart adversary.

**Why it matters to AI safety.** It measures two named risk factors (network effects, destabilising dynamics) with
exact ground truth, and it tests the "untrusted agents, trusted monitor" idea from AI control at swarm scale.

### D2. The diversity dividend

**Question.** When can a swarm beat its best member, and do our models ever meet that condition?

**Why it is new.** Correlated-error work measures static benchmarks. Nobody has put a correlation dial into a swarm
world with exact truth and checked a prediction against it before running.

**What it builds on.** Track 1's independence measure (1.1 voters from three copies, 2.05 from three models), v4.2's
lesson that every ant read the same sensor, the colony aggregator, W2's cached cells (free to reuse).

**Design.** Give each ant a private noisy sensor whose errors correlate with others by a dial ρ. Preregister the gain
a correlated-voting formula predicts for each (accuracy, ρ). Run scripted ants on the dial (free). Then measure real
pairwise error correlation for our SLM pool on fresh items, predict the colony's result in advance, and run it at
matched calls against the best member and "best SLM × N samples". Add engine-supplied randomness (sample from a
generated list) as an arm.

**What a null still tells us.** If the formula predicts the scripts but our models sit at ρ near 1, the Oct 10 null is
explained and we know swarms of these models cannot win by voting: tune for independence, not accuracy. That makes
SLM steering part two concrete: steer for uncorrelated errors.

### D3. Hidden messages in marks

**Question.** Can ants pass secret information through where, when and how strongly they leave marks, and which
engine rules close that channel without hurting the task?

**Why it is new.** Collusion research studies messages. Marks are a shared environment, which Schroeder de Witt et
al. flag but nobody has measured.

**Design.** A sender and receiver pair try to pass bits through marks while doing the task. Measure channel capacity
(bits per tick) and task cost under engine rules: rounding mark strength, adding noise, rate limits, fixed deposit
times. Scripted encoders first (an upper bound), then SLMs prompted or trained to collude.

**What a null still tells us.** A channel that cannot be closed without hurting the task is a measured cost of
stigmergy; a cheap rule that closes it is a design rule.

### Ranking

| | New | Uses what we have | Cost to first result | Null still useful | Safety value | Rank |
|---|---|---|---|---|---|---|
| D1 gate and breakdown point | high | very high | $0 (scripts) | yes | high | **1** |
| D2 diversity dividend | medium | high | $0, then small model spend | yes | medium | 2 |
| D3 hidden messages in marks | very high | medium (needs D1's gate) | $0, then model spend | yes | very high | 3 |

## 5. First measurement: the liar probe ($0, done today)

`scripts/swarm_scale_probe.py` reuses the pilot colouring world and scripts and changes two things: a seeded share of
the ants are liars (each announces a random colour to every neighbour every round), and N runs from 8 to 2,048. Three
honest policies: S (the pilot's patient greedy), **gate** (S plus one model-free rule: stop hearing a neighbour whose
report has changed twice; honest ants commit once, so that breaks the protocol), and the oracle (told who lies).
Development seeds 100 to 102 only. Tests: `tests/swarm_scale/test_probe.py`. Raw rows:
`docs/research/swarm-scale-20261029/`.

Mean share of honest links coloured properly (3 seeds × 2 graph families per cell). "Stuck" = share of honest ants
that never chose a colour.

**8 rounds** (`probe_r8.json`):

| N | liars | S | gate | oracle | S stuck | gate stuck |
|---|---|---|---|---|---|---|
| 8 | 10% | 0.764 | **1.000** | 1.000 | 0.143 | 0.000 |
| 8 | 30% | 0.350 | **1.000** | 1.000 | 0.417 | 0.000 |
| 64 | 0% | 0.562 | 0.562 | 0.562 | 0.333 | 0.333 |
| 64 | 30% | 0.373 | **0.821** | 0.827 | 0.404 | 0.111 |
| 512 | 0% | 0.453 | 0.453 | 0.453 | 0.444 | 0.444 |
| 512 | 2% | 0.408 | 0.472 | 0.475 | 0.468 | 0.423 |
| 512 | 30% | 0.277 | **0.675** | 0.713 | 0.513 | 0.223 |
| 2048 | 0% | 0.469 | 0.469 | 0.469 | 0.436 | 0.436 |
| 2048 | 2% | 0.421 | 0.478 | 0.483 | 0.464 | 0.423 |
| 2048 | 30% | 0.238 | **0.699** | 0.739 | 0.538 | 0.199 |

**32 rounds** (`probe_r32.json`):

| N | liars | S | gate | oracle | S stuck | gate stuck |
|---|---|---|---|---|---|---|
| 64 | 0% | 0.932 | 0.932 | 0.932 | 0.052 | 0.052 |
| 64 | 10% | 0.699 | **0.995** | 0.995 | 0.215 | 0.003 |
| 512 | 0% | 0.645 | 0.645 | 0.645 | 0.288 | 0.288 |
| 512 | 10% | 0.535 | **0.767** | 0.767 | 0.356 | 0.173 |
| 512 | 30% | 0.481 | **0.980** | 0.983 | 0.357 | 0.012 |

What it shows (development seeds, scripts only, so a lead, not a result):

1. **One model-free rule recovers almost all of the oracle at every size.** At 2,048 ants with 30% liars the gate
   scores 0.699 against the oracle's 0.739 and plain S's 0.238. It is never told who lies.
2. **The gap between the gate and the oracle opens slowly as N grows** (0 at N = 8, 0.04 at N = 512 and 2,048 with 30%
   liars, 8 rounds). That widening is exactly the curve D1 must measure properly.
3. **The pilot's honest script does not scale, liars or not.** With no liars it drops from 1.000 at N = 8 to about
   0.45 at N ≥ 512 in 8 rounds, and still only 0.645 at N = 512 in 32 rounds, because "wait for every lower-id
   neighbour" builds waiting chains longer than the rounds; 29% to 44% of honest ants never act. This is a
   groundwork finding: **before any scale test, the honest protocol itself must be one whose speed does not depend
   on N** (for example a randomised local rule). Otherwise the scale curve measures waiting chains, not robustness.
4. A caution on reading the table: scores are over honest-to-honest links only, so removing liars also removes
   links, and a gated swarm with 30% liars can score above the same swarm with none. D1's breakdown point compares
   against a solo baseline on the same links for this reason.
5. **Scripted ants are cheap at scale**: about 0.24 ms per ant decision, flat from 8 to 2,048 ants. A 4,096-ant,
   32-round run is about half a minute on one core.

What it does **not** show: anything about model ants, adaptive or colluding liars (the chattering liar is the easiest
liar to catch), or the other worlds.

## 6. Build order if D1 is approved

1. Write the D1 preregistration (factors, f*, bars, controls) and freeze it with the v4.3 freeze guard.
2. Pick an honest protocol whose speed does not depend on N (section 5, point 3), and check on development seeds
   that with no liars it holds its score from 8 to 4,096 ants. Only then is a breakdown curve meaningful.
3. Generalise the probe into `antelligence/worlds/graph_coloring` (many liars as a world condition, the gate as a
   kernel admission rule, not a script trick), with tests.
4. Add the three adversaries (steady, adaptive, colluding) and the soft-reputation gate.
5. Port the liars to E15 DAGs and the v4.3 tumour marks.
6. Part B: the cascade sweep (one wrong mark, half-life × graph).
7. Only then: SLM honest ants at N = 64, with "best SLM × N samples" as the baseline.

## 7. Questions that decide the direction

1. Which result matters more to you: a safety result (how many bad ants a swarm survives) or a better tumour score?
2. Does "at scale" mean thousands of cheap simulated ants on one machine, or many real model calls?
3. Should the bad ants include ones that know the gate's rule and try to beat it?
