# Learning plumbing check (v4.2, goal section 10)

**This is a plumbing check, not a finding.** The "learning" here is a script nudging its own job thresholds (a number goes
down a little when a job works, up a little when it does not) and a file being saved and loaded. No model is involved and
nothing was learned in the sense of a model keeping a lesson. Do not cite the number below as evidence that ants learn.

What was run (development cases only, scripts only, no model, no network, seed and config as stated):

1. One colony of 8 S-marks ants ran the 30 dev-A cases in split order, seed 1, frozen `config/v4_2/world.yaml`. After each
   run the 8 name cards were saved and the next run loaded them back; only the job thresholds were carried. Nothing else was.
   The v4.2 rule that cards reset every run is unchanged everywhere else; carrying is an explicit option of
   `scripts/v42_learning.py` only.
2. The 10 dev-B cases, seeds 3 and 4: S-marks started from those carried thresholds versus S-marks with fresh cards, paired by
   (case, seed). Every carried run starts from the same trained cards. Descriptive only; not a verdict, not part of Holm.
3. A notebook round trip (write, cap, archive, freeze, reload) is in `results.json` under `notebook_round_trip`.

Result (descriptive): median paired coverage difference, carried minus fresh, 20 pairs over 10 cases: -0.2485 (95% case-clustered bootstrap interval -0.3227 to -0.2052, 10000 resamples). Notebook round trip ok: True.

Plain reading: carrying thresholds made coverage clearly WORSE here. Over 30 runs the thresholds only drift up (a turn without
success adds 0.05, a success removes 0.1, and most turns do not succeed), so by the end most ants sit near 1.0 and act less
readily. That says the plumbing works (cards saved, loaded, applied, measured) and that this threshold rule does not improve
with carrying; it says nothing about models learning lessons.

Files: `results.json` (everything, per-run threshold trajectory included), `colony_cards/cards/` (final saved name cards),
`notebook_ant-00.json` (the demo notebook). Full run output also in /Volumes/WD_BLACK/antelligence-v4.2/runs/learning/.
