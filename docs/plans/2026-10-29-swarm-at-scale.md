# A smarter swarm: think alone, fade smartly

Date: 2026-10-29 (revised several times the same day after Eshaan's steers). Status: **two ideas built and checked, plus a
skipped-work test, a deep-thinker check and two ported tasks, for review**. Nothing in this file is preregistered: these
are development runs. No model call, no paid call, no chain write and no key were used. Every new measurement is $0:
scripted ants, or replays of model answers saved by earlier runs. The one paid test, against the best single model
on outside tasks, is preregistered separately in `2026-10-29-fair-test.md` and has not been run.

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
  Tests: `tests/smarter_swarm/` (85 tests, counting the skipped-work world in 0.2, the deep-thinker check in 0.3, the two ports in 0.4 and the fair test harness).

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

### 0.2 No faking "done" (Eshaan's second steer)

The worry: a swarm that does 90% of a job, skips the last bit and records it as done. Models do this. Frontier models
reward-hack in real tasks (METR, June 2025), exploit unit tests when the task cannot be done honestly (ImpossibleBench,
arXiv 2510.20270), and learning to reward-hack spreads into wider bad behaviour (Anthropic, arXiv 2511.18397).
"Premature termination" and "missing or incomplete verification" are among the commonest failures of multi-agent LLM
systems (MAST, arXiv 2503.13657). No earlier Antelligence run measured it: none of the branches has a test for work
claimed done but not done.

**Built:** `antelligence/worlds/finish/` and `scripts/finish_check.py` (tests in `tests/smarter_swarm/test_finish.py`).
A job is split into pieces (4 per ant, 2 to 10 units of work each). Ants leave "work left" or "done" marks. A share
of the ants are **shortcut-takers**: they do the work until a piece is down to its last 20%, then mark it "done" and
walk away. Nobody counts when the job is finished: each ant notices when it keeps finding nothing to do, and the swarm
calls the job finished when 95% of its ants are idle (the way harvester ants decide, from how often they meet work).
The score is the work still left at that moment. Three ways of handling "done":

- **Trust the tick-box**: if a piece is marked done, believe it, don't look, and repeat "done" (the herd).
- **Look yourself** (think alone): always look at the piece you stand on before believing it. Re-checking a piece
  someone marked done costs a tick. Mark only what you saw. No checker anywhere.
- **Trust plus inspectors** (a checker, for comparison): 10% of the ants stop working and become inspectors who walk
  to pieces marked done and reopen them if work is left.

Each with fixed or smart fading. 10 development seeds per cell, all arms on the same worlds:

With 20% shortcut-takers (`finish_n{64,256,1024}.json`):

| Ants (pieces) | Pieces left unfinished when the swarm said done: trust | **look yourself** | inspectors | Work left: trust | **look** | Ticks to "done": trust | look | inspectors |
|---|---|---|---|---|---|---|---|---|
| 64 (256) | 20.7 | **0.9** | 20.1 | 1.8% | **0.2%** | 65.7 | 65.4 | 78.0 |
| 256 (1,024) | 84.8 | **2.6** | 84.3 | 1.9% | **0.2%** | 69.8 | 74.7 | 77.0 |
| 1,024 (4,096) | 334.4 | **13.4** | 322.5 | 1.7% | **0.2%** | 70.8 | 73.6 | 79.4 |

For scale, an honest swarm (no shortcut-takers) leaves 0.8, 2.9 and 10.3 pieces, the same under trust and look.

- **Trusting "done" lets the shortcut through.** With 20% shortcut-takers about 8% of the pieces are left unfinished
  and recorded as done (about 2% of the work). It grows with the share of shortcut-takers (5%: 0.6%, 10%: 0.9%,
  20%: 1.9% of the work at 256 ants), in every seed.
- **Looking yourself closes it, with no checker.** The swarm ends at about the honest swarm's level at every size and every
  shortcut share. It beats trust on all 10 seeds in every cell but one (9 of 10 at 256 ants with 5%). It costs 0 to 5
  extra ticks (at most 7%), because most re-checking happens when an ant would otherwise be idle.
- **Inspectors did not help.** Taking 10% of the ants off work to inspect caught almost nothing (84 against 85 pieces
  unfinished at 256 ants) and made "done" come 6 to 12 ticks later. Their "work left" marks were outvoted by everyone
  repeating "done". That is Deming's point, measured.
- **Smart fading changed nothing here.** Whether a piece counts as done depends on which claim is stronger, and under
  trust nobody ever contradicts a claim. The fix is a rule (who may write "done"), not a setting.
- **The stop rule leaks a little on its own.** Even an honest swarm calls it done with 0.2% of the work left. Waiting 20
  idle ticks instead of 10 cuts that to 0.02% for 13 more ticks (18% longer); with 20% shortcut-takers it then leaves
  81 pieces under trust against 0.8 when ants look (`finish_n256_p20.json`).
- **Looking has a price when looking is slow.** If re-checking a claim takes 3 ticks, looking at every claim doubles
  the time (70 to 145 ticks) and still leaves 1.1%, because ants run out of patience while re-checking. Spot looking
  (re-check one claim in four) gives 107 ticks and 0.5% (`finish_n256_look3*.json`). So the next step is to look
  cheaply: spot checks by peers, and more often on claims from ants with a poor record (earned trust, idea 1).

### What the books say, and the factor we were missing

| Source | What it says | What it means for the swarm |
|---|---|---|
| Deming, *Out of the Crisis* (1982), point 3 | Stop depending on inspection; build quality into the process. | Fix how ants mark work, not who checks it. The inspector arm above agrees: it cost 10% of the workers and caught nothing extra. |
| Ohno, *Toyota Production System* (1978) | Jidoka and the andon cord: anyone who sees a defect stops the line; never pass a defect on. | Any ant that sees unfinished work says so, and its first-hand "work left" outweighs a claim. That is the look-yourself rule. |
| Goodhart (1975); Strathern (1997); Kerr, "On the folly of rewarding A, while hoping for B" (1975) | When a measure becomes a target it stops measuring. You get what you reward, not what you hoped for. | Never score or reward "pieces marked done". Score the work left in the world. The same holds for reasoning traces (0.3). |
| Brooks, *The Mythical Man-Month* (1975) | Projects sit at "90% done" for a long time: the last part hides the hard work. | The last bit is where skipping hides. Even with no cheats our stop rule leaves 0.2% to 0.3% behind (0.02% with more patience). |
| Meadows, *Thinking in Systems* (2008) | Changing the rules and the information flows moves a system far more than tuning a number. | Changing who may write "done" (a rule) fixed it. Tuning fade speed (a number) did nothing here. |
| Surowiecki, *The Wisdom of Crowds* (2004) | Crowds are wise only with diversity, independence, decentralisation and a way to combine. | Relaying a claim breaks independence: one shortcut becomes many "done" marks. |
| Gordon, *Ant Encounters* (2010) | No ant is in charge; colonies decide from how often ants meet things. | Our stop rule. No counter, no boss. |
| Grassé (1959) on termites | Stigmergy: the work itself prompts the next worker. | "Done" should be a fact you can see on the piece, not a note about it. |
| Ostrom, *Governing the Commons* (1990) | Shared resources last when the users themselves do the monitoring, with graduated sanctions. | Peers look, not a separate police. Next: an ant caught claiming done on unfinished work loses trust (idea 1). |
| Perrow, *Normal Accidents* (1984) | In tightly coupled systems small faults spread into big ones. | A trusted claim ties everyone to one ant's word. Looking yourself loosens that tie. |

**The factor we weren't accounting for:** who is allowed to write "done", and whether "done" is a *claim* or a *fact
you can see in the work*. Every earlier version scored what agents reported (reports, verdicts, escalations), and the
reports were mostly wrong (2.4% true in the v4 pilot). A second, smaller place to skip is the stop rule itself: the
moment a swarm decides it is finished is a decision too, and it needs the same "look at the work" rule.

### 0.3 Deep thinkers in the swarm (Eshaan's third steer)

Models that think at length before answering are strong but slow and costly. The rule stays the same: a deep thinker
is a smarter ant, never a boss. Three ways to use one, simplest first:

1. **Ask it only where the cheap ants disagree, and count it as one vote.** Its vote is weighed by its track record,
   exactly like every other ant's (idea 1). Tested for free below.
2. **Thoughts as trails.** After an ant has answered alone, it leaves its key steps as fading marks. Other ants may
   reuse a step only after they have answered alone themselves, and a step lasts longer when other ants' own work
   agrees with it (smart fading on steps). This needs new model calls, so it is not run yet (cheap: about 30 items).
3. **Traces as proof of work.** A trace that says "I checked X" can be compared with what the world shows was done,
   the same way the finish world compares "done" with the work left. The catch is real: traces are often not faithful.
   Reasoning models used a planted hint but mentioned it in only a minority of cases (arXiv 2505.05410), and training a
   model against a trace monitor teaches it to hide the misbehaviour instead of stopping (arXiv 2503.11926; see also
   arXiv 2507.11473). So a trace is information to read, never a score to reward: Goodhart again.

**Free test of idea 1** (`scripts/deep_on_split_check.py`, `deep_on_split.json`; tests in
`tests/smarter_swarm/test_deep_on_split.py`). None of the saved runs used a long-reasoning model, so the strongest
saved model (Haiku, 0.88 against Qwen's 0.78 on finance) stands in for it. Saved Oct 10 colony answers, 50 items per
domain, 4 rotations each. "Split" means the three cheap answers were not unanimous:

| Cheap crowd (3 calls) | Domain | Cheap vote | Deep alone | Deep **decides** where split | Deep is **one vote** where split | Deep calls per item |
|---|---|---|---|---|---|---|
| Qwen ×3 | finance | 0.780 | 0.880 | 0.875 | **0.890** | 0.41 |
| Qwen ×3 | medical | 0.830 | 0.820 | **0.855** | 0.830 | 0.12 |
| Qwen + Phi + Gemma | finance | 0.745 | 0.880 | 0.880 | **0.885** | 0.97 |
| Qwen + Phi + Gemma | medical | 0.800 | 0.820 | 0.830 | **0.840** | 0.56 |

- **Ask the deep model only where the cheap ants split.** With three Qwen samples, that matched or beat asking it on
  every item while using it on 41% (finance) and 12% (medical) of them.
- **One vote beat a verdict in 3 of 4 cells**, by 0.5 to 1.5 points. The exception is Qwen on medical, where the
  verdict won by 2.5 points: Qwen's track record there is better than Haiku's, so one Haiku vote never overturned two
  Qwen votes. Read all of this as a lead: one stand-in model, 50 items per domain, rotations that are not independent.
- **Different cheap models disagree almost always** (97% of finance items), so the split test saves little there. A
  crowd of one model splits less often, but its mistakes line up (section 0, the W2 table). That trade is the next thing to
  measure with a real deep thinker.

### 0.4 Think alone is now the default, and it holds on two tasks we did not design (Eshaan's go-ahead)

Every new ant now thinks alone first and calls something done only when it sees it done:
`RumourWorld.run` defaults to `alone`, `FinishWorld.run` to `look`, and the two ported worlds get a `default_ant`.
The pilot's recorded arms keep their own scripts, so its published numbers still reproduce (tested).

**Colouring** (AgentsNet, arXiv 2507.08616; `antelligence/worlds/graph_coloring/think_alone.py`,
`scripts/colouring_think_alone_check.py`, `colouring_n{64,256,1024}.json`). Each agent must pick a colour different
from its neighbours'. Conditions: `base`, 30% of messages lost, 10% liars announcing random colours, and both.
10 development seeds × 2 graph families per cell. Score: share of honest edges properly coloured (1.0 is perfect),
then the share of honest agents that call themselves done while really clashing.

| Agents, rounds | Condition | Pilot S (wait, then pick once) | Pilot S′ (resend, guess at the end) | **Think alone, look before done** |
|---|---|---|---|---|
| 64, 4 | base | 0.19 (61% never coloured) | 0.69, 56% "done" but clashing | **1.00**, 0% |
| 64, 4 | lost messages + liars | 0.06 | 0.64, 65% | **0.99**, 2% |
| 256, 4 | base | 0.17 | 0.65, 59% | **1.00**, 0% |
| 256, 4 | lost messages + liars | 0.06 | 0.63, 67% | **0.99**, 1% |
| 256, 8 | lost messages + liars | 0.07 | 0.76, 44% | **1.00**, 0% |
| 1,024, 4 | base | 0.16 | 0.63, 61% | **1.00**, 0% |
| 1,024, 4 | lost messages + liars | 0.05 | 0.60, 68% | **0.99**, 1% |
| 1,024, 8 | lost messages + liars | 0.06 | 0.73, 46% | **1.00**, 0% |

- Look-before-done beat S and S′ in every cell, on 16 to 20 of 20 paired runs (average gains +0.46 to +0.96 over S and
  +0.19 to +0.44 over S′ across 64, 256 and 1,024 agents; every 95% interval above 0).
- **The "done" result is the one that matters most.** S′ gets most agents coloured, but a third to three quarters of them
  are sure they are finished while still clashing. The look-before-done ant almost never is (0 to 2%).
- Honest caveats. Much of the gain is "act first, then look and repair" instead of waiting for permission: the pilot's S
  was a weak, patient script, so this is not a win over a strong rival. The earned-trust gate added nothing here,
  because this ant already avoids every colour it hears, liars' included. Telling it who the liars are (the oracle)
  added nothing either.

**Foraging** (SwarmBench, arXiv 2505.04364; `antelligence/worlds/foraging_sb/think_alone.py`,
`scripts/foraging_think_alone_check.py`, `foraging.json`). Ants fetch food to a nest. 40 development seeds; food
delivered per run:

| Condition | Pilot S (trust the brief) | Pilot S′ (trust your eyes) | **Think alone, then compare** | Same, when ants also pass on old news |
|---|---|---|---|---|
| base | 16.0 | 16.2 | 16.2 | 16.2 |
| food runs out | 5.4 | 16.7 | **16.6** (+11.3 over S, 40/40 seeds) | 16.3 |
| sensors flip | 3.7 | 3.9 | 3.9 | 3.9 |

- When food runs out, trusting your own eyes over the old plan triples deliveries. That is the whole gain:
  "then compare" adds nothing here, because ants that say only what they see right now rarely have news a
  listener needs (about one lead per run).
- When ants also pass on what they remember, listeners follow about 12 leads per run and their own eyes throw out 11
  of them as stale. Old news cost about 0.3 deliveries (not significant). This is the rumour world's lesson again, on a
  task we did not build: second-hand news is mostly echo.
- When the ants' own sensors are wrong (v2), think alone does not help. That needs a different fix (an ant can't
  out-look a broken eye), so it stays open.

### 0.5 Are we heading the right way? What we know, and what we only hope

Eshaan asked whether the evidence supports the direction, and to check it on outside tasks we did not pick.

**What we know** (measured, $0, scripted ants or saved model answers):

1. In every world we have, ants that mark only what they saw themselves beat ants that pass on what they heard:
   the rumour world (+0.10 to +0.13 right answers), the finish world (much less work left behind), and now
   colouring and foraging, two tasks other people designed.
2. Calling something done only when you see it done stops the swarm from faking "done". Trusting the tick-box let
   20% shortcut-takers hide skipped work; looking caught it. In colouring, S′ claimed done while clashing for up to
   74% of agents, the look-before-done ant for 2% at most.
3. On saved real model answers (the W2 runs), answers given alone were worth more than answers given after seeing
   the board, and a cheap crowd of one model has errors that line up (section 0).
4. Earlier Antelligence runs where a checker or a teacher steered the swarm made it worse (section 0.1).

**What we only hope** (not yet shown):

1. That a swarm of **model** calls that think alone beats the best single model asked the same number of times on
   real tasks. Every win above is against weaker swarms, not against one strong model. This is the test that decides
   whether the swarm is worth building, so it is next, and its pass line is written down first (below).
2. That the gains hold with real costs: looking costs time, and models charge per call.
3. That smart fading pays off beyond being a safety layer: it helped the herd but added nothing once ants think alone.
4. That a deep thinker asked only where the cheap ants split is worth its price (one stand-in model so far, 0.3).

**The fair test** (`docs/plans/2026-10-29-fair-test.md`; the pass lines, files and code are frozen in
`docs/research/fair-test-20261029/prereg.json` before any call). One model, Step 3.7 Flash, free on the Nous portal
(DeepSeek V4.1 Flash was the first pick; see the test's Amendment 2), so "the best single model" is that model asked the plain question 5 times with the majority (or
average) taken, a strong baseline. The swarm is the same model, 5 calls, each told to look at the question a
different way and answering alone. Two outside task sets: 240 exam questions from AGIEval (law-school logic,
LogiQA, maths word problems) and real forecasting questions that resolved after the model was released. Free,
with a $3 cap in case the portal ever charges. It needs a key that lives only on Eshaan's Mac, so it runs there.

**The shared-read test** (`docs/plans/2026-10-30-shared-read.md`, frozen in
`docs/research/shared-read-20261030/prereg.json`). The Swarm Telepathy paper (MIT CSAIL and MIT-IBM, arXiv
2610.15095) hands one model's reading of a document to a smaller model of another family and beats a written summary
by 15 to 19 points, but only ever with one receiver. This asks the swarm's version first, in plain text and free on
local models: one reader (Qwen3 14B) writes notes on each of the same 240 exam questions, and 8 small ants from 4
families answer alone, from the notes, or both. It measures whether the shared reading makes the vote smarter or
just makes the ants' mistakes alike, and whether the swarm beats the reader answering alone.

**How the MIT team checks its huge simulation** (Large Population Models, Chopra, MIT, arXiv 2507.09901; the AgentTorch
work behind the 8.4 million-agent New York City COVID-19 model). They do not check that each simulated person
behaves faithfully. They check the crowd against the world: the simulator is differentiable, so they tune it by
gradient until its predicted case numbers match the real case statistics. The paper names what is still open: its
"Open Problems" section lists formal verification ("relies on empirical validation rather than formal guarantees")
and reports "numerical instabilities and unexpected emergent behaviors" when economics, movement and disease are
combined; it also names the gap between simulation and reality. Fitting to known data shows the model can match
the past, not that it predicts. We borrow the part that works and add the part they leave open: we never score the
swarm on what it says about itself, only on the world (work left, edges clashing, food delivered), and the fair test
scores it on outcomes nobody knew when the answers were written, with the pass line fixed in advance.

### What next

1. Run the fair test on Eshaan's Mac (free model, cap $3). If it passes, build model ants
   that answer alone and look before done. If it fails, say so plainly and work out why before building more.
2. Make looking cheap: spot checks by peers, more often on claims from ants with a poor record (idea 1), and a stop
   rule that waits long enough. Score the swarm on work left in the world, never on what it reports.
3. Fix the case think alone cannot: an ant whose own eyes are wrong (foraging v2).
4. Then SLM steering: steer small models for errors that do not line up, and for honest "not done yet".

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
