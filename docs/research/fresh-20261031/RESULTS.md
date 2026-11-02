# Fresh-question test, run 1 (2026-10-31): results

Plan, frozen before any call: `docs/plans/2026-10-31-fresh-questions.md` and `prereg.json` here. Eshaan ran it on
the Mac Studio with `CLAUDE_CODE_PROMPT.md`. It ran to the end, and nothing was committed or pushed from the Mac. The
raw output is in `score.out` in the `antelligence-fresh` worktree on the Mac. The block below is copied exactly as he
pasted it back.

## Short answer

- **Jev's run-1 lead was likely memory.** Run 1 had Jev alone 18.3 points ahead of the local reader on public AGIEval
  questions. On 160 new questions it is 1.9 points *behind* (75.0% vs 76.9%, range −10.6 to +6.9). The frozen rule
  says "gone", because the whole range is under +10.
- **The letter check is unclear.** Moving the letters cost Jev almost nothing (+0.8, range −2.1 to +3.8), so it isn't
  matching remembered letters. The options alone scored 6.7 points above chance (range +1.1 to +12.4). That is mostly
  the logic group (37% vs 22% chance), and it sits under the frozen "memorised" line of 25. It is also just over the
  "no sign" line of 10, which is why the verdict is "unclear".
- **The reader's notes still help the ants.** Notes vs ants alone is +10.0 (range +2.5 to +17.5), and the ants don't
  think alike. That repeats run 1's +7.9.
- **The swarm still doesn't beat the reader alone** (notes 74.4% vs reader 76.9%). The best arm overall is Jev reading
  the ants (79.4%). That is 2.5 points over the reader, but the range crosses zero (−5.6 to +10.6), so it isn't a real
  win.
- **It depends on the kind of question.** On syllogisms, Jev alone got 100% and the ants with both helps got 80%,
  against the reader's 62%. On word problems (100%) and assignment puzzles (60%), the reader is best and Jev is weak
  (72% and 40%). This is a pattern to test, not a result: it was not in the frozen plan.
- **d1-3B adds nothing**: it is below the reader on every arm (alone −15.6, range −25 to −6.3).

Caution: these questions turned out easier for the reader than AGIEval (76.9% vs 63.3%, and 100% on word problems).
The frozen plan compares leads, not raw scores, for exactly this reason. Still, a code-made set this regular may favour
some models over others in ways AGIEval doesn't (plan, "Limits").

## The output, as pasted

```
FRESH TEST RESULT (paste this back to Claude in Duet)
questions: 160 new (order 40, assign 40, arith 40, syllogism 40), seed 8655006865359881267
swarm: usable 93%; reader alone 76.9%; votes alone 64.4%, notes 74.4%, both 72.5%; notes vs alone +10.0% [0.025, 0.175]
  swarm verdict: {"shared_read_helps": "yes", "ants_think_alike": "no", "swarm_beats_reader_alone": "no"}
  by task (reader / alone / notes / both): arith 100%/98%/92%/100%; assign 60%/30%/52%/50%; order 85%/57%/75%/60%; syllogism 62%/72%/78%/80%
  run: 4160 new calls, 262 unusable, 121 min
arms on new questions (run 1 on AGIEval in brackets):
  jev:read        79.4%   [77.5%]
  reader          76.9%   [63.3%]
  jev:dm1         75.0%   [81.7%]
  notes8          74.4%   [57.5%]
  both8           72.5%
  jev:vote        70.0%   [53.8%]
  d1-3b:vote      66.2%   [48.8%]
  alone8          64.4%
  d1-3b:read      64.4%
  d1-3b:dm1       61.3%   [47.1%]
  jev: alone vs reader -1.9% [-0.1062, 0.0688]; vote vs ants +5.6% [0.0125, 0.1]; vote vs reader -6.9% [-0.15, 0.0125]; read vs reader +2.5% [-0.0563, 0.1062]; vote vs alone8 +5.6% [0.0125, 0.1]; read vs alone8 +15.0% [0.075, 0.2313]
  d1-3b: alone vs reader -15.6% [-0.25, -0.0625]; vote vs ants +1.9% [-0.0187, 0.0625]; vote vs reader -10.6% [-0.1938, -0.0187]; read vs reader -12.5% [-0.2188, -0.0312]; vote vs alone8 +1.9% [-0.0187, 0.0625]; read vs alone8 +0.0% [-0.0875, 0.0875]
  by task: order reader 85%/jev:dm1 88%/d1-3b:dm1 68%; assign reader 60%/jev:dm1 40%/d1-3b:dm1 38%; arith reader 100%/jev:dm1 72%/d1-3b:dm1 50%; syllogism reader 62%/jev:dm1 100%/d1-3b:dm1 90%
VERDICT, Jev's lead over the reader (run 1: +18.3%): gone: on new questions Jev is not clearly better than the reader, so run 1's lead was likely memory
cost jev: 320 calls (0 failed), $0.0072, median 0.152s
cost d1-3b: 320 calls (0 failed), $0.0000, median 0.137s
letter check on the 240 AGIEval questions: n 240, chance 21.2%; relabel 80.8%, blind 27.9%, dm1 81.7%
  moving the letters cost +0.8% [-0.0208, 0.0375]
  options alone, above chance: +6.7% [0.0106, 0.1242]
  by group: logic chance 22%/relabel 89%/blind 37%/dm1 88%; puzzles chance 20%/relabel 72%/blind 19%/dm1 75%
  LETTER CHECK VERDICT: unclear: some sign of memory, under the frozen lines; cost $0.0106
```

The verdicts follow the frozen lines:

- Lead "gone": the range's top is +6.9, under +10.
- Letter check "unclear": the drop's range top is +3.8, under 5, so the drop alone is clean. But the options-alone
  range top is +12.4, which is over 10, and +6.7 is under 25, so it is neither clean nor memorised.

## How the run went (from the Mac's report)

- The swarm took 121 minutes for 4,160 local calls; 262 answers were unusable, so 93% were usable. That passes the 90%
  guard.
- Memory pressure stayed normal. The lowest free memory was 42%, and swap grew 454 MB (limit 2 GB). The d1-3B run's
  lowest free memory was 55%.
- Nothing was skipped. Jev cost $0.0178 in all ($0.0072 + $0.0106) against the 2.5-cent cap. With run 1's $0.0223, the
  total is $0.0401.
- The first two swarm attempts stopped at the memory guard. The cause was the macOS file watcher `fseventsd` holding
  22 GB. After Eshaan restarted it, free memory went from 8 GB to 24 GB. The Mac session wrote this up in his vault.
  Worth checking before any long local run.
- The local chat shim `com.openclaw.chat-shim` uses port 8090. With Eshaan's OK, it was stopped for the run and
  brought back up afterwards. Ports 11434 and 8520 were left free. A later run could use a different `--mlx-port`
  instead.
- The MLX server was `mlx_lm.server` from `antelligence-dmaddon/.venv-optiq` (mlx-lm 0.32.2, the pinned version). Run
  1's record doesn't say which one it used.

## What's next (not built)

A free re-score of the saved answers that sends each kind of question to its best arm. This needs Eshaan's yes. It
would also need fresh questions to check it, because picking the best arm on these same 160 would overfit.
