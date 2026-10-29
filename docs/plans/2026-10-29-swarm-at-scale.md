# A smarter swarm: think alone, fade smartly

Date: 2026-10-29 (revised twice the same day after Eshaan's steers). Status: **two ideas built and checked, for
review**. Nothing here is preregistered. No model call, no paid call, no chain write and no key were used. Every new
measurement is $0: scripted ants, or replays of model answers saved by earlier runs.

Base: `hermes/v4.31-a2-20261016` (the newest work). Every number below cites the file it comes from.

## 0. Built: smart fading memory and think alone, then compare (Eshaan's pick)

Eshaan picked ideas 4 and 5 and dropped anything with a checker or a teacher steering the swarm (section 0.1 lists
the earlier Antelligence runs that back that call). Both ideas are now built, with tests, and checked for free:

- `antelligence/kernel/fading.py`: a world-agnostic mark store where each mark has its own half-life. Under the smart
  rule a reader whose own view disagrees with a mark halves its half-life; one who agrees stretches it by a quarter.
  No checker: the feedback comes only from ordinary readers.
- `antelligence/worlds/rumour/`: a scripted world to test both. Ants walk a site graph, get rare, noisy private
  readings of one hidden yes/no fact (right 60% of the time), leave marks and read the marks where they stand.
  *Herd* ants say what they now believe (what they heard becomes what they say). *Think-alone* ants commit to their
  own reading first and only ever mark what they themselves saw; they still read and use the others' marks. Wrong
  information comes as a burst of strong wrong marks at 5% of the sites, or as 2% liars.
- `scripts/fading_alone_check.py` runs the grid; `scripts/alone_vs_seen_check.py` does the same comparison on saved
  real model answers. Results: `docs/research/swarm-scale-20261029/fading_alone_*.json`, `alone_vs_seen.json`.
  Tests: `tests/smarter_swarm/` (28 tests).

### What got smarter (scripted world, 10 development seeds per cell, half-life 10 unless noted)

Share of ants with the right answer at the end, no wrong marks planted:

| Ants | Each ant alone (no marks) | Herd, fixed fading (the old way) | Herd, smart fading | **Think alone**, fixed | Think alone + smart |
|---|---|---|---|---|---|
| 64 | 0.613 | 0.725 | 0.803 | **0.830** | 0.820 |
| 256 | 0.611 | 0.724 | 0.823 | **0.849** | 0.839 |
| 1,024 | 0.613 | 0.716 | 0.804 | **0.845** | 0.825 |
| 2,048 | 0.628 | 0.741 | 0.835 | **0.863** | 0.844 |

- **Think alone is the big win**: +0.10 to +0.13 over the herd at every size (all 10 seeds at 256, 1,024 and 2,048; 95%
  intervals well above 0). With longer-lived marks (half-life 20) the gap grows to +0.21; with noisier readings
  (55% right) it is +0.08. The one place it loses is very short-lived marks (half-life 5: −0.03), where relaying
  spreads news faster than the echo can build.
- **Smart fading rescues a herd** (+0.05 to +0.10 everywhere) but adds nothing on top of think-alone (−0.01 to
  −0.03): when marks only carry first-hand readings, cutting them short also throws away good evidence.
- **The swarm is now much smarter than any one ant**: 0.85 against 0.61 for an ant on its own, with no queen, no
  checker and no stronger model anywhere.

How much damage wrong information did (extra wrong share over the same run with nothing planted, averaged over the
60 ticks):

| Ants | Wrong marks | Herd, fixed | Herd, smart | Think alone, fixed | **Both together** |
|---|---|---|---|---|---|
| 256 | burst at 5% of sites | 0.034 | 0.033 | 0.015 | **0.009** |
| 256 | 2% liars | 0.030 | 0.032 | 0.015 | **0.010** |
| 1,024 | burst at 5% of sites | 0.034 | 0.034 | 0.014 | **0.008** |
| 1,024 | 2% liars | 0.032 | 0.035 | 0.015 | **0.009** |
| 2,048 | burst at 5% of sites | 0.033 | 0.034 | 0.014 | **0.008** |
| 2,048 | 2% liars | 0.034 | 0.037 | 0.017 | **0.012** |

- Think alone halves the damage; adding smart fading cuts it to about a quarter (burst) or a third (liars) of the
  herd's. Every one of the 10 seeds agrees at 256, 1,024 and 2,048.
- Smart fading kills planted wrong marks about twice as fast **only when the ants think alone** (median tick 13 to
  19 against 41 with fixed fading). Under herding, smart fading sometimes never kills them (1 to 5 runs in 10): the
  herd's own beliefs keep confirming the lie. That is the teacher problem in miniature: when the readers' views come
  from the crowd, the crowd's feedback protects the crowd's mistake.
- A single wrong mark among 256 sites did almost nothing in any arm (extra wrong under 0.002); the burst is the
  meaningful test.

### Think alone on real model answers (saved W2 runs, $0)

W2 ran the same three-agent teams on the same 60 questions and seeds, once blind (`independent_vote`: three answers,
majority wins) and once after reading each other's answers (`signal_board`: two rounds of revising). No checker in
either. Matched on question and seed:

| Model | Team right, blind | Team right, after seeing each other | Seeing helped / hurt | When one agent is wrong, a teammate is wrong too (blind → seen) |
|---|---|---|---|---|
| Qwen3.8-27B | 0.767 | 0.744 | 3 / 7 | 0.55 → **0.90** |
| Haiku | 0.883 | 0.822 | 0 / 11 | 0.87 → **0.97** |
| Sonnet | 0.650 | 0.600 | 0 / 3 | 1.00 → 1.00 |

- **Seeing each other's answers never paid**: 3 questions helped against 21 hurt across the three models (sign test
  p < 0.001). Most of the harm (18 of 21) came from the seeing runs breaking more often (more rounds, longer prompts,
  more format failures); where both runs finished, it was 2 helped and 3 hurt, so no measurable gain either way.
- **Seeing each other made the mistakes line up**: for Qwen, the chance that a teammate shares a wrong answer went
  from 0.55 to 0.90 (finance: 0.25 to 0.92). That is exactly what think-alone is meant to stop.
- The other W2 pair (`evidence_isolated` against `evidence_exchange`) shows the flip side: when each agent holds
  *different evidence*, sharing what you saw helps hugely (Qwen medical: 0.24 → 0.80 team right). But the isolated
  arm also failed format checks very often, so treat that pair as a lead. Together they say the same thing as the
  scripted world: **share what you saw, not what you concluded.**

### 0.1 Earlier runs where a teacher or checker made the swarm worse

- **v4 pilot, the queen** (`docs/research/v4-pilot-20261011/FINDINGS.md`): in colouring, workers with no queen scored
  G 1.223; adding the queen dropped them to 0.926, and the queen plus escalation to 0.613. The likely cause: the queen
  restricted whole groups and removed the move a worker needed.
- **Oct 10 colony, the checker** (`docs/research/colony-v2-20261010/FINDINGS.md` on `hermes/track1-colony-experiment-20261010`):
  verified admission zeroed every vote that failed its check, but those votes were still right 59% (Qwen) to 77%
  (Haiku) of the time. Turning the check off took the local colony's finance score from 48% to 76%.
- **v4 pilot, escalation and reports**: escalation fired on 99% of decisions, and only 2.4% of worker reports were
  true; the oversight loop was mostly noise.
- **v4.2 and v4.3, talk that could only veto**: asking a neighbour changed almost nothing (talk equalled marks), and
  98% of the marks that tried to correct the scan were wrong.
- **W2**: evidence exchange through a shared board cost 28 points against a single model, and (above) reading each
  other's answers never helped.
- **Research that agrees**: groups anchor on a leader's framing, with deference in about 61% of sessions and
  pushback under 1% ("Diversity Collapse", 2026); peers flip correct agents to wrong more often than the reverse
  ("Talk Isn't Always Cheap", 2025; "Herd Behavior", 2025).

One honest caveat: the gains that *did* come in the past came from a model-free rule, not a model: the deterministic
merge in E15 and W2 planning. Smart fading is that kind of rule. It has no judge; it only listens to the readers.

### What next

1. Make think-alone the default for every ant: ants mark only first-hand readings, never relays. Keep smart fading
   as a safety layer for wrong information, not for accuracy.
2. Port both into the colouring and foraging worlds, where the old marks were accurate but changed nothing, and check
   the gain survives a real task.
3. Then, and only then, model ants: small models that write down their own answer before they see the board, scored
   against "best model × N samples". That is also where SLM steering fits: steer for errors that do not line up.

## 1. The short answer

**The tumour use case is parked.** The brain-tumour shapes come from real scans (BraTS), but the drug spread, uptake
and nanobot behaviour are invented, and there are no treatment outcomes to check a score against (`data/README.md`,
`antelligence/worlds/tumor_mri/`). A better tumour score would prove little. The tumour world stays in the repo as
one test world among several, not the goal.

**The new goal is a smarter swarm.** Across every version so far one pattern holds: teams of models never beat their
best member, and the gains came from cheap, model-free rules (the merge, the verifier, the admission checks). So the
ideas below are about *rules that make the group smarter*, not about smarter members. Each is simple, each is new in
a specific way, and each has a free first test.

| # | Idea | In one line | Free first test | Status |
|---|---|---|---|---|
| 1 | **Earned trust** | each ant's vote or mark counts by its track record | saved colony answers; liar probe | **run** (section 4) |
| 2 | **Grow only for hard problems** | start with one ant; recruit more only when it is unsure | saved colony answers | **run** (section 4) |
| 3 | **Roles that emerge** | ants switch between scout, worker and checker by what is missing nearby | scripted colouring and foraging | dropped (has a checker) |
| 4 | **Memory that fades smartly** | marks fade fast when contradicted, slowly when confirmed | plant wrong marks, watch them spread | **built** (section 0) |
| 5 | **Think alone, then compare** | every ant commits before it sees the others and marks only what it saw | rumour world; saved W2 answers | **built** (section 0) |

**First recommendation (superseded by Eshaan's pick of ideas 4 and 5): ideas 1 and 2 together.** The free check on the saved
colony answers already points the right way: growing only on hard items matched the 3-call vote (within half a
point) with 40% fewer calls on finance and 63% fewer on medical, and earned trust removed all the damage weak
members did to the mixed vote. Neither beat the best single member, which matches every past result; what they win
is cost and safety, not raw accuracy. Ideas 3 to 5 are where a real *gain* over the best member could come from, and
idea 3 is the most ant-like.

## 2. Why these five

- Past colonies **always ran at full size**, even on easy items (W2, Track 1). Real ants recruit more workers only to
  good food. → idea 2.
- The Oct 10 colony lost because **failed checks zeroed votes** that were still right 59% to 77% of the time, and two
  weak members (Phi 4%, Gemma 8% on finance) were counted like the strong one. → idea 1.
- In the v4 pilot **the queen made workers worse**, and in colouring workers with no queen did best. Fixed central
  roles hurt. → idea 3 (roles chosen locally, not handed out).
- In v4.3 **marks were accurate but 89% changed nothing**, and research says shared updates only help when they lean
  toward correction ("Debate or Vote", NeurIPS 2025) and that fading decides whether one error spreads ("From Spark to
  Fire", 2026). → idea 4.
- **Three copies of one model behaved like 1.1 independent voters** (Track 1), and two LLMs agree on about 60% of the
  questions both get wrong (Kim et al., ICML 2025). A group cannot beat its best member while everyone is wrong
  together. → idea 5.

## 3. The five ideas

### Idea 1. Earned trust

- **What:** each ant's vote or mark is weighted by how often its past votes or marks turned out right. Nobody is
  zeroed; weak ants just count for less.
- **Why new:** LLM teams usually weight members equally or by a fixed prior. Here trust is earned inside the swarm,
  per ant and per task type, and checked against exact ground truth.
- **Free test:** (a) saved colony answers, weights from the development split (done, section 4.1); (b) the liar probe,
  where a flip gate is the simplest form of earned trust (done, section 4.2). Next: ants that are *sometimes* right,
  not pure liars, so trust has to be graded.

### Idea 2. Grow only for hard problems

- **What:** start with one ant. Recruit more only when its answer fails a cheap check or the first few disagree, like
  ants recruiting to a good food source.
- **Why new:** past colonies were always full size. This turns "does a team beat its best member at matched calls?"
  into "how few calls does the team need to match it?", a question the swarm can actually win.
- **Free test:** saved colony answers (done, section 4.1). Next: the same rule on the planning DAGs (E15), where the
  merge already gives a real team gain.

### Idea 3. Roles that emerge

- **What:** each ant has a simple threshold for each job. If many marks nearby are unchecked, it becomes a checker; if
  nothing new has been found, it scouts; otherwise it works. No queen hands out roles.
- **Why new:** LLM teams fix roles by hand. Here the role mix appears from local need, and the checker role targets the
  real bottleneck the research found: picking the right answer, not generating one ("OrchSLM", 2026).
- **Free test:** scripted colouring and foraging ants at 8 to 2,048: fixed role mix against threshold switching,
  against an oracle mix. The kernel already has the pieces (`antelligence/kernel/colony/thresholds.py`, `quorum.py`).

### Idea 4. Memory that fades smartly

- **What:** a mark fades faster each time another ant contradicts it and slower each time one confirms it.
- **Why new:** every past version used one fixed half-life per mark kind. Research gives the reason this matters (an
  error spreads when sharing outruns fading) but nobody has tuned fading per mark.
- **Free test:** plant one wrong mark in a scripted swarm and measure how far it spreads and how long it lives under
  fixed fading against smart fading, across neighbour graphs.

### Idea 5. Think alone, then compare

- **What:** every ant writes down its own answer before it sees anyone else's. Where the ants share a model, the engine,
  not the model, supplies the variety (each ant draws from its own list of options).
- **Why new:** it attacks "everyone is wrong together" head on. A 2026 study found that sampling from a self-made list
  brought same-model agreement down to the human level (4%); nobody has tested it inside a swarm with exact truth.
- **Free test:** a scripted world with a dial for how alike the ants' errors are. Check that a simple formula predicts
  the team's gain at each dial setting, then place our real models on the dial using their saved answers.

### The earlier safety ideas

The first version of this plan led with safety directions (how many bad ants a swarm survives; whether ants can hide
messages in marks). They are not dropped: idea 1 is the same machinery, and the liar probe in 4.2 is its first test.
They come back once the swarm itself is smarter.

## 4. Free checks already run

### 4.1 Earned trust and growing on hard items, on saved model answers

`scripts/cached_colony_check.py` replays the Oct 10 colony run's saved answers (branch
`hermes/track1-colony-experiment-20261010`: Qwen3.8-27B, Phi-4-mini, Gemma 4 E4B and Haiku; 50 finance and 50
medical items; 4 samples each). Each item is replayed 4 times, each time with a different sample first. Trust weights
come only from the development split. Results: `docs/research/swarm-scale-20261029/cached_colony.json`.

| Policy | Finance accuracy | Finance calls/item | Medical accuracy | Medical calls/item |
|---|---|---|---|---|
| best member once (Qwen) | 0.765 | 1.00 | 0.825 | 1.00 |
| best member ×3, vote | 0.780 | 3.00 | 0.830 | 3.00 |
| mixed colony (Qwen, Phi, Gemma), equal votes | 0.745 | 3.00 | 0.800 | 3.00 |
| mixed colony, **earned trust** | 0.765 | 3.00 | 0.825 | 3.00 |
| **grow on hard**: Qwen, +2 Qwen only if the check fails | **0.785** | **1.82** | 0.825 | **1.11** |
| grow on hard: Qwen, ask Haiku only if the check fails | 0.885 | 1.41 | 0.825 | 1.05 |
| Haiku once | 0.880 | 1.00 | 0.820 | 1.00 |

What it shows: earned trust brings the mixed colony back up to its best member (the weak members' votes stop
hurting); growing on hard items lands within half a point of the 3-call vote (above it on finance, below it on medical)
at 40% to 63% fewer calls. What it does not show: any
policy clearly above the best single model. Asking Haiku on hard items lands where Haiku alone lands, so on these
questions it is just "use the stronger model". Rotations of one item are not independent and there are only 50 items
per domain, so these are leads, not results.

### 4.2 Earned trust at scale: the liar probe

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
   liars, 8 rounds). That widening is the curve a proper scale test must measure.
3. **The pilot's honest script does not scale, liars or not.** With no liars it drops from 1.000 at N = 8 to about
   0.45 at N ≥ 512 in 8 rounds, and still only 0.645 at N = 512 in 32 rounds, because "wait for every lower-id
   neighbour" builds waiting chains longer than the rounds; 29% to 44% of honest ants never act. This is a
   groundwork finding: **before any scale test, the honest protocol itself must be one whose speed does not depend
   on N** (for example a randomised local rule). Otherwise the scale curve measures waiting chains, not robustness.
4. A caution on reading the table: scores are over honest-to-honest links only, so removing liars also removes
   links, and a gated swarm with 30% liars can score above the same swarm with none. A proper test compares against a solo
   baseline on the same links for this reason.
5. **Scripted ants are cheap at scale**: about 0.24 ms per ant decision, flat from 8 to 2,048 ants. A 4,096-ant,
   32-round run is about half a minute on one core.

What it does **not** show: anything about model ants, adaptive or colluding liars (the chattering liar is the easiest
liar to catch), or the other worlds.

## 5. What every past version tried and showed

### 5.1 The original app (`main`)

| Technique | What it did | Result |
|---|---|---|
| LLM ants foraging (`backend/simulation.py`) | ants call a model per move, rule fallback; optional queen | demo only, no measured comparison |
| Tumour nanobots (`backend/nanobot_simulation.py`, `biofvm.py`) | trail, alarm and recruitment pheromone fields; optional LLM queen | pheromones vs none: **0% improvement**, secretion was not wired (`SPRINT.md` phase 2) |
| On-chain provenance | 5 public fields per run | moved from Base Sepolia to **ZKsync Era Sepolia** (chain 300); all 4 contracts deployed, one end-to-end run at `proof_staged` (`docs/status/BLOCKERS.md`, `docs/status/e2e-zksync-era-sepolia.json`). `VISION.md`, `ARCHITECTURE.md` and `SPRINT.md` still say Base Sepolia (stale). |

### 5.2 DeSci paper, E1 to E15 (Sept, `docs/research/desci-paper-20260915/PAPER-v3.md`)

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

### 5.3 Workbench, W2 and the colony (Sept to Oct 10)

| Study | Technique | Result |
|---|---|---|
| Cloud study (Sept 9, `docs/FUGU_RESEARCH.md`) | vote, refine, evidence exchange on 20 Q&A items | **vote 16/20** best; exchange 8/20; 40/100 cells lost to format errors |
| W2 benchmark (`hermes/m0-w2-benchmark-20261010`) | 6 team protocols vs a single model, 2,831 cells | no team beat a single model; evidence exchange **−28 points**; planning: raw plans 0/19, **after the model-free merge 16/19 for every model** ("reflects the merge step, not model quality") |
| Colony "v3 Track 1" (**Oct 10**, `hermes/track1-colony-experiment-20261010`) | mixed colony with reputation, correlation discount, verified admission, vs best member at matched calls | **null**: finance −30 (local) and −16 (Haiku) points vs the best member; medical +4 (not significant). Cause: failed checks zeroed votes that were right 59% to 77% of the time; weak members (Phi 4%, Gemma 8%). Three copies of one model behaved like **1.1** independent voters, three different models like **2.05**. |

### 5.4 The v4 line (Oct 11 to 16)

| Version | Technique | Result |
|---|---|---|
| v4 pilot (`docs/research/v4-pilot-20261011/FINDINGS.md`) | decision-model workers + queen + escalation vs a strong model on every agent | unclear. Colouring: **workers with no queen did best** (G 1.223 vs DS-all 1.026); the queen and escalation each made things worse; reports 2.4% true; escalation fired 99%; Laya unusable; 9B beat 27B |
| v4.1 (`docs/research/v4.1-20261013/DESIGN-anchors.md`) | tamper-evident board: hash chain, Rekor, Bitcoin timestamps, write-once registry on ZKsync | built, fail-closed verifier; the chain is the only uniqueness guarantee |
| v4.2 (`docs/research/v4.2-20261014/FINDINGS.md`) | scripted ants sharing marks or talk in the tumour world | **fail** three times (+0.012 vs a 0.03 bar); perfect marks did not help; every ant read the same sensor, so nothing private to share; 79% of wall time spent waiting on model calls |
| v4.3 (`docs/research/v4.3-20261015/FINDINGS.md`) | fixed ant, foraging uptake marks, talk | **near miss**: +0.0293 vs the 0.03 bar; talk = marks; marks accurate but 89% changed nothing; displaced marks still helped (the queen's zone need is the suspect); gain *shrank* as the tumour got patchier |
| v4.31 (`docs/research/v4.31-20261016/`) | real-unit physics world | the gap between the script and the oracle is robust only at low thresholds; world science, not swarm science |

### 5.5 What is still unfinished

- v4.3's own next steps were never run: fix the queen's need to separate it from the marks; make judgement marks
  fire where the scan is wrong; use good-uptake marks to choose where to dose.
- Colony: soft-weighting failed checks instead of zeroing them (can be tested free on cached replies); a competence
  floor for members.
- Proof: the mock proof is still mock (`proof_staged`); the contract fix "needs a redeploy".
- Hive: queen succession, structured vs raw memory, never tested with models.

### 5.6 The lessons, in one list

1. **Model-free layers carry the gains** (E1 to E3, E15, W2 planning).
2. **Mixing models does not beat the best member at matched calls** (W2, Track 1, pilot). Copies of one model are
   barely more than one voter.
3. **Sharing helps only when someone holds information others lack**, and even accurate marks rarely change a costly
   decision (v4.2, v4.3).
4. **Central control can hurt** (queen in the pilot, possible anchoring).
5. **The engine is not the bottleneck; model latency is** (v4.2 throughput).

## 6. What 2025 to 2026 research says (and what to borrow)

| Finding | Source | What we borrow |
|---|---|---|
| Multi-agent setups gain on parallel tasks and lose 39% to 70% on sequential ones; independent agents amplified errors up to 17.2×, versus 4.4× with a validating orchestrator | Kim, Liu et al., Google, "Towards a Science of Scaling Agent Systems" (arXiv 2512.08296) | place each world on their axes; a validation bottleneck is what our merge is |
| Majority vote explains most of debate's gain; debate cannot raise expected accuracy unless updates lean toward correction | Choi et al., "Debate or Vote", NeurIPS 2025 (arXiv 2508.17536) | the theory behind our marks null: marks need a correction-biased rule |
| One model's own samples beat a mix of models | Li et al., "Rethinking Mixture-of-Agents" (arXiv 2502.00674) | "best SLM × N samples" becomes a required baseline |
| A router over 7 SLMs barely beats the best one, though the right answer is usually somewhere in the pool | Zhang, Yao, "OrchSLM" (arXiv 2609.13470) | the bottleneck is picking, not generating: spend the swarm on checking |
| Two LLMs agree on about 60% of the questions both get wrong | Kim et al., "Correlated Errors in LLMs", ICML 2025 (arXiv 2506.07962) | measure our pool's error correlation (idea 5) |
| Group diversity halves as groups grow from 3 to 7; leaders anchor | Chen et al., "Diversity Collapse" (arXiv 2604.18005) | test queen-visible vs queen-blind phases |
| LLM agents agree far more than humans; sampling from a self-generated list fixes it | Ballestero et al., "Strategic Algorithmic Monoculture" (arXiv 2604.09502) | let the engine supply the randomness |
| One wrong seed spreads into false consensus; risk ≈ susceptibility × graph spectral radius ÷ decay | Xie et al., "From Spark to Fire" (arXiv 2603.04474) | decay is our pheromone half-life (idea 4) |
| Byzantine-tolerant consensus claims beyond n/3 faults, but tested on 7 agents and 10 questions | Zheng et al. (arXiv 2511.10400) | test the claim at hundreds of ants |
| Agents reach valid consensus 41.6% of the time; 0% with 3 Byzantine agents at N = 16 | Berdoz et al., ETH (arXiv 2603.01213) | report safety (no wrong act) and liveness (got done) separately |
| Taxonomy of multi-agent risks: miscoordination, conflict, collusion; risk factors incl. network effects | Hammond et al., Cooperative AI Foundation (arXiv 2502.14143) | name the risk factor each experiment measures |
| Shared environments enable secret collusion | Schroeder de Witt et al. (arXiv 2505.02077) | marks are a shared environment (later safety work) |
| LLM swarms run thousands of times slower than rule swarms | Rahman, Schranz (arXiv 2506.14496) | keep rule ants; call models only at decision points |

Gaps no one has filled, which a cheap ant testbed can: a measured epidemic threshold for stigmergic errors;
Byzantine robustness at hundreds to thousands of agents; covert channels through a shared environment rather than
messages; a matched-compute account of *where* swarm value comes from; the bridge from LLM consensus safety to
physical robots.


## 7. Next steps once you pick

1. Write a short preregistration for the picked idea (what counts as a win, on which worlds, against which
   baselines: best member once, best member × N, and the oracle).
2. Fix the groundwork the liar probe exposed: an honest ant rule whose speed does not depend on swarm size.
3. Build the idea as a kernel rule (not a one-off script), with tests, and run it scripted on colouring, foraging and
   the planning DAGs at 8 to 2,048 ants.
4. Only then add model ants, small first, with "best model × N samples" as the bar to beat.
5. SLM tuning and steering come after, aimed at whatever the swarm turns out to need (for example, steering for
   errors that do not line up, if idea 5 shows that is the limit).

## 8. Questions

1. Which idea do you want built first? My pick is 1 and 2 together; 3 is the most ant-like.
2. Should the test bed be questions and answers (where we have saved model answers) or grid worlds (where ants move
   and leave marks)?
