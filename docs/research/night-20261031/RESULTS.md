# Night run (2026-10-31 prompt, ran 2026-11-01): results

Short answer: **the reader's notes are still the one real win, and nothing beats the plain reader yet.**

- Field notebook ants: no clear change on fresh exam questions (pooled over sets 1 and 2, 320 questions).
- The reader's notes lift the ants by +8.1 points (range +4.2 to +12.1, 480 questions), but the ants with notes stay
  6.2 points under the reader alone.
- Routing each kind of question to its best setup: unclear on the one new set it was checked on (+4.4, range 0 to
  +9.4). Set 3's check failed, so it is still unchecked.
- Jev is not better than the reader overall on new questions.

The run did not finish cleanly. The external drive WD_BLACK, which holds the model cache on Eshaan's Mac, froze at
18:57, and after that Python could not start for a while. Three stages failed (below). The hung main process then
finished its report step on its own at 20:03, once file reads recovered, and printed the block below. Eshaan pasted it
into Claude Duet on 2026-11-02. The numbers are copied from it exactly; they were not re-checked here, because the run
folder lives on the Mac.

## The block, as pasted

```
NIGHT RUN RESULT (paste this back to Claude in Duet)
stages: route done; set2:make done; set2:swarm done; set1:notebook done; set3:make done; set3:swarm done; pool failed; report done; set2:jev done; set2:d1 done; set2:score done; route-check:set2 done; set2:notebook done; set3:jev done; set3:d1 done; set3:score done; set3:notebook failed; route-check:set3 failed
route set1: reader 76.9%; in-sample 88.1%; split-half 84.6% vs reader +7.7% (moves +3.8% to +11.2% with the split); routes {"arith": "reader", "assign": "reader", "order": "jev:read", "syllogism": "jev:dm1"}
route check set2: routed 85.0% vs reader 80.6% +4.4% (range +0.0% to +9.4%); vs single +5.6% (range +0.0% to +11.2%); VERDICT unclear: routing may help, but the range is too wide to call
notebook pooled n 320: nb_notes 69.4% vs both 70.3% -0.9% (range -5.6% to +3.8%); nb 68.8% vs alone 64.7%; reader 78.8%; copy wrong reader both 34.1% -> nb_notes 28.0% +6.1% (range -3.6% to +15.9%); VERDICT {"notebook_helps": "no clear change", "stops_copying_reader": "no clear change", "notebook_beats_reader": "no"}
pooled n 480 (set1, set2, set3): jev:dm1 vs reader -3.1% (range -7.7% to +1.5%); jev:read vs reader -0.4% (range -4.8% to +4.2%); notes8 vs alone8 +8.1% (range +4.2% to +12.1%); notes8 vs reader -6.2% (range -10.8% to -1.7%); both8 vs reader -9.2% (range -14.0% to -4.4%); jev:vote vs notes8 -4.0% (range -7.7% to +0.0%)
```

## Which stages failed, and what that means

| Stage | Why it failed | What's missing |
|---|---|---|
| `set3:notebook` | It stalled while loading Granite (ant model 3 of 4) from the frozen drive; the run stopped it (exit −15). | The notebook numbers cover sets 1 and 2 only (320 questions, not 480). |
| `route-check:set3` | Python could not start (exit 1). | Routing has one held-out set, not two. |
| `pool` | Same (exit 1). | Nothing lost: `report` runs the same pooling itself, and it succeeded at 20:03. |

Everything else finished: the routing picks on set 1, sets 2 and 3 (swarm, Jev, d1-3B, scoring), the routing check on
set 2, and the notebook rounds on sets 1 and 2. `REPORT.md` was written on the Mac; it is not on the fork (the Mac
can't push).

## How each verdict follows the frozen lines

From `docs/plans/2026-10-31-night-run.md`, frozen before any real call:

- **Routing helps** needs at least +5 points with the range above 0 on new questions. Set 2: +4.4, range starting at
  +0.0. Just short on both counts, so **unclear**. On set 1 the split-half estimate was +7.7 (it moves +3.8 to +11.2
  with the split), so the held-out gain came in lower than the in-set estimate, as expected.
- **The notebook helps** needs `nb_notes` at least +3 points over `both` with the range above 0: it was −0.9 (range
  −5.6 to +3.8). **No clear change.**
- **Ants stop copying a wrong reader** needs the copy rate to drop at least 5 points with the range above 0: it dropped
  6.1 points (34.1% to 28.0%), but the range runs from −3.6 to +15.9. **No clear change**, though it leans the right way.
- **The notebook beats the reader** needs the best notebook setup's range above the reader: the reader got 78.8% on
  these questions, against 69.4% at best. **No.**

## What helped, what didn't, what next

- Helped: the reader's notes. They lift the ants by about 8 points on new questions, and the range is clearly above 0.
- Didn't help: the field notebook. Reading every other model's work once added nothing on top of the notes (−0.9).
  It may cut copying of the reader's wrong answers (34% to 28%), but that isn't proven.
- Didn't help: the ants as a group. With notes they are still about 6 points under the reader alone, and question +
  notes is 9 points under.
- Not settled: routing by kind (+4.4 on one new set, just under the bar). Sending syllogisms and ordering puzzles to Jev
  is the part that might be real.
- Jev: about even with the reader on new questions (−3.1 alone, −0.4 reading the ants), so its AGIEval lead was memory.
- Next, cheap: run the set 3 routing check on the Mac (free, seconds) for a second held-out test.
- Next, bigger: the swarm's gap to the reader is the real problem. Ideas that keep the reader's notes and change what
  the ants do with them are where the evidence points, not more ways of mixing their answers.

## Run notes

- Time: about 7 hours of work, 12:31 to 19:26 (`state.json` says 24,997 s). Each swarm took 114 to 119 minutes; each
  notebook round about 55 to 58 minutes, faster than the plan's budget (130 and 90).
- Memory: no memory stops. Lowest free memory was 42% on the first attempt; during the main run it stayed at 57% or
  more. Swap stayed flat at about 5.5 GB.
- fseventsd: 16 GB at the start, mostly compressed. That blocked the first attempt until Eshaan ran
  `sudo killall fseventsd`; it stayed near 0 afterwards. The script logged 0.41 GB because it only reads resident size,
  not compressed size. The memory watch should read the compressed size too.
- Jev: $0.0144 for the night, under the 5-cent cap.
- Services on the Mac (embedding on :8091, RAG on :8094) were turned back on afterwards. The night run's own MLX server
  also uses :8091, so it looks like the embedding service was paused for the run (inferred, not confirmed). A future
  run should use a port nothing else on the Mac uses.
- The `antelligence-night` worktree is still in place, with nothing committed or pushed from the Mac.
- Claude Code logged the incident in Eshaan's vault: `000-inbox/2026-11-01-night-run-wdblack-hang-desktop-eintr.md`.

## The one step left (optional)

Once `~/Desktop` reads work, this adds the set 3 routing check and rewrites `REPORT.md` with it. It's free and takes
seconds:

```
cd ~/Desktop/research/antelligence-night && uv run python scripts/night_run.py route-check --set set3 && uv run python scripts/night_run.py report
```

## Before the next unattended run

This is the second time the WD_BLACK drive has caused trouble (the first was a 5-day outage in April). Eshaan's
suggestion, recorded here: **move the model cache to the internal disk before the next unattended run.**
