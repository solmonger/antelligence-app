# Night run (2026-10-31 prompt, ran 2026-11-01): partial results

Status: **did not finish.** The external drive WD_BLACK, which holds the model cache on Eshaan's Mac, froze at 18:57.
After that, programs started from the Mac's Claude Code session could not read files under `~/Desktop` ("interrupted
system call"), and Python could not start. So the last steps never ran: there is no `NIGHT RUN RESULT` block and no
`REPORT.md`.

Everything below is copied from what Claude Code on the Mac reported back (pasted into Claude Duet on 2026-11-02). The
numbers were not re-checked here, because the run folder lives on the Mac. The "What helped, what didn't, what next"
section of the report is **not written yet**: it waits on the pooled notebook numbers.

## What finished and was saved

- Routing picks from set 1 (frozen before any new question existed).
- Sets 2 and 3: swarm (reader + 8 ants), Jev, d1-3B and scoring.
- Routing test on set 2 (the held-out test).
- Notebook rounds on sets 1 and 2.

## What is missing, and why

| Missing | Why |
|---|---|
| Set 3 notebook round | It stalled while loading Granite (ant model 3 of 4) from the frozen drive. Claude Code stopped that step, which its own run had started. |
| Routing check on set 3 | Python could not start after about 19:20. |
| Pooling (`pool`) | Same. Pooling is the step that scores the notebook, so **there is no notebook result yet**, not even for sets 1 and 2. |
| `REPORT.md` and the result block | Same. The main `night_run.py` process (PID 74737) was left hung in its report step; it may still finish on its own if file reads recover. |

Claude Code did not rerun, because a rerun could not start Python either.

## Numbers so far (as reported)

Routing on new questions, set 2 (routes frozen on set 1; routed minus the reader alone, 95% range):

| | Share right |
|---|---|
| Routed by kind | 85.0% |
| Reader alone | 80.6% |
| Gain | +4.4 points, range 0 to +9.4 |
| Frozen reading | **unclear**: it may help, but the range is too wide to call |

The frozen line for "helps" is at least +5 points with the range above 0, so +4.4 with a range starting at 0 falls just
short. The set 3 check would add a second held-out set.

Other numbers:

- The reader is still the best single setup: 80.6% on set 2 and 82.5% on set 3.
- On syllogisms Jev did better: 98% on set 2 and 100% on set 3. That is the gap routing tries to use.
- Jev's lead is gone on new questions in both sets, so its old AGIEval lead was likely memory (as in the first fresh
  run, `docs/research/fresh-20261031/RESULTS.md`).
- The reader's notes lift the ants: +10.6 points on set 2 (range +3.8 to +17.5) and +3.8 on set 3 (range −2.5 to +10).
- The ants still lose to the reader in every arm.

Plain read: no setup beats the plain reader yet. Routing by kind came close on one new set (+4.4) but didn't clear the
frozen bar. The notebook question (do ants stop copying the reader's mistakes?) has no answer yet.

## Run notes

- Time: about 7 hours of work, 12:31 to 19:26 (`state.json` says 24,997 s). Each swarm took 114 to 119 minutes; each
  notebook round about 55 to 58 minutes. Faster than the plan's budget (130 and 90).
- Memory: no memory stops. Lowest free memory was 42% on the first attempt; during the main run it stayed at 57% or
  more. Swap stayed flat at about 5.5 GB.
- fseventsd: 16 GB at the start, mostly compressed. That blocked the first attempt until Eshaan ran
  `sudo killall fseventsd`; it stayed near 0 afterwards. The script logged 0.41 GB because it only sees resident size,
  not compressed size. (Worth fixing in the memory watch: it should read the compressed size too.)
- Jev: $0.0144 for the night, under the 5-cent cap.
- Services on the Mac (embedding on :8091, RAG on :8094) were turned back on afterwards. The worktree
  `antelligence-night` is still in place, with nothing committed or pushed.
- Claude Code logged the incident in Eshaan's vault: `000-inbox/2026-11-01-night-run-wdblack-hang-desktop-eintr.md`.

Note on port 8091: the night run starts its own MLX server on :8091, and Claude Code's report says the Mac's embedding
service answers on :8091 when it's on. The run's start check needs 8091 free, so it looks like the embedding service
was paused for the run (the report says "the pause is cleared"). Inferred, not confirmed. Either way, any rerun of
`all` needs 8091 free again, and a future run should use a port nothing else on the Mac uses (the port is part of the
frozen plan, so it can't be changed for this run).

## How to finish

Once the drive and `~/Desktop` reads work again (probably a reboot or reconnecting the drive):

Quick finish, a few minutes, free. Adds the set 3 routing check, then pools and writes the report. The notebook result
then covers sets 1 and 2 only:

```
cd ~/Desktop/research/antelligence-night && uv run python scripts/night_run.py route-check --set set3 && uv run python scripts/night_run.py pool && uv run python scripts/night_run.py report
```

(Eshaan's command was `pool` then `report`; the `route-check --set set3` step is added in front because `pool` and
`report` only read a routing check that already exists, and set 3's never ran.)

Full finish, about an hour, free (Jev is already done for every set). Rerunning the same `all` command skips every
finished stage, redoes only the set 3 notebook round (its cache keeps what was done), then the set 3 routing check,
pool and report. Two things to know first:

- The time budget counts the ~417 minutes already used (`state.json` keeps them), and the set 3 notebook round only
  starts with at least 265 minutes left. So it needs `--hours 12`, not a small number.
- Ports 11434, 8091 and 8520 must be free, so the embedding service on 8091 has to be paused again for the rerun.

```
nohup uv run python scripts/night_run.py all --hours 12 --mlx-server <mlx path> --d1-python <d1 path> --agieval <dir> > night2.log 2>&1 &
```

Then paste back the `NIGHT RUN RESULT` block and the "What helped, what didn't, what next" section gets written from it.

## Before the next unattended run

This is the second time the WD_BLACK drive has caused trouble (the first was a 5-day outage in April). Eshaan's
suggestion, recorded here: **move the model cache to the internal disk before the next unattended run.**
