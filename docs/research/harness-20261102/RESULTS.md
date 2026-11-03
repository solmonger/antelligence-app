# Harness v0, first Mac run (2026-11-02): results

Eshaan ran `CLAUDE_CODE_PROMPT.md` on the Mac Studio with local models only ($0). Nothing was committed or pushed
from the Mac; the run's files stay in `docs/research/harness-20261102/run` in the `antelligence-harness` worktree there.
The block below is copied exactly as he pasted it back.

**These are dev numbers.** Code-made puzzles and saved dev answers only. The real test set is picked last and kept out
of this repo, so nothing here proves the harness works.

## Short answer

- **On the saved night-run answers, the harness beat the reader by 6.9 points** (89.4% vs 82.5%, range +3.1 to
  +10.6) on set 3, which the learner never saw. The whole gain is one move it found by itself from sets 1 and 2: send
  syllogism puzzles to Jev (72.5% to 100%) and everything else to the reader.
- **That matches the fresh test.** Run 1 of the fresh-question test saw Jev get 100% on syllogisms and called it "a
  pattern to test, not a result". This is a second, separate set where it held.
- **Two cautions.** The fresh test also found the options alone scored above chance on logic puzzles (37% vs 22%), so
  part of Jev's syllogism edge may come from how the puzzle maker writes options. And Jev is a paid hosted model.
- **It is faster but uses more tokens:** time x0.80 (Jev answers in 0.2s), tokens x1.11 (Jev reads about 500 tokens a
  question, the reader about 378). So it is not yet a win on all three.
- **The live smoke run was too small to learn from.** 12 puzzles is under the 8 per kind the learner needs before it
  trusts a pattern, so it fell back to the reader, and the live planner used only the reader. Jev is off in live runs
  because it costs money, so the syllogism move can't happen live yet.
- **46 of 342 calls came back unusable.** The 30 code-checker calls can't be unusable, so this is 46 of 312 answer
  calls, about 15%. The fresh test had 6% (262 of 4,160). A blank answer counts as wrong, so this pulls the ants down.
  The cause isn't in the result block.
- **The code checker settled 4 of 12 puzzles and was right on all 4.** When it settles, it is right. The other 8 used
  all three tries without settling. Why isn't in the result block either.

## What to fix next

1. Run `scripts/harness.py diagnose` on the Mac (no model calls; `DIAGNOSE_PROMPT.md`). It shows which models gave the
   unusable replies and why (cut off, or the wrong form), and replays every code-checker try from the saved programs to
   show what tripped it.
2. Unusable replies: fix the cause it shows. If replies are cut off, give them more room; if the form is wrong, add one
   short re-ask for just the letter.
3. Code checker: give it a worked example of the program shape (list every possible case, keep the ones that fit the
   rules, then test each option the way the question asks), and when it narrows to two options, let a vote pick between
   those two instead of throwing the result away.
4. Tokens: try the code checker on syllogisms. They are plain set logic, which code handles well, it is free, and it
   doesn't need Jev. If it matches Jev there, the harness can beat the reader on accuracy, tokens and time together.
5. Check that the syllogism puzzle maker doesn't give away the answer in how it writes the options.
6. A bigger live sweep (about 10 puzzles per kind) so the live learner has enough to work with.

The real test set still comes last.

## Diagnose (same day)

Eshaan ran `DIAGNOSE_PROMPT.md` (no model calls). What it showed, in plain words:

- **Most failures were replies that ran out of room.** The reader model (Gemma 12B) hit its length limit 25 times,
  mostly while writing code-checker programs. The provider then threw the whole reply away, so the checker saw nothing
  and said "did not come back as a program". That, not bad logic, is why 8 of 12 puzzles didn't settle. (The replay
  says "a try was missing" because a thrown-away reply was never saved.) Granite ran out of room 4 times. Those calls
  were not billed either, so the token counts were a little low.
- **The smallest ant (LFM2.5 1.2B) gave the answer as the option's value instead of its letter** 13 times, like
  `"answer": "15"` when an option reads 15. Those were thrown away as blank.
- **A few replies refused to pick**: "None" or "None of the above" (Gemma E2B twice, Gemma 12B once).
- Blanks were mostly on word problems (12 of 75) and assignment puzzles (8 of 75).

```
HARNESS DIAGNOSE (paste this back to Claude in Duet)
saved calls: 346, failed: 47
failed replies by model:
  LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit: 13 of 72; Unusable (stop stop) x13; reply length up to 45 tokens
    e.g. ...{"reason": "Calculate combined rates and solve step by step.", "answer": "15"}
    e.g. ...{"reason": "Calculate total salt step by step, then compute the percentage.", "answer": "35"}
  ibm-granite/granite-4.2-3b-q4-mlx: 4 of 72; ProviderTruncated (stop -) x4
    e.g. ...generation did not finish normally (length)
    e.g. ...generation did not finish normally (length)
  mlx-community/Qwen3.5-4B-MLX-4bit: 1 of 72; Unusable (stop stop) x1; reply length up to 242 tokens
    e.g. ...g. D: PJ wrong. E: PJ wrong. Only A has PJ. Check A: V(3), T(4) ok. R(6), L(5) ok. V/R not adj ok. All good."}
  mlx-community/gemma-4-12B-it-4bit: 27 of 58; ProviderTruncated (stop -) x25, Unusable (stop stop) x2; reply length up to 132 tokens
    e.g. ...generation did not finish normally (length)
    e.g. ...anes' is a subset of that. However, none of the options A-E follow logically.", "answer": "None of the above"}
  mlx-community/gemma-4-e2b-it-4bit: 2 of 72; Unusable (stop stop) x2; reply length up to 59 tokens
    e.g. ... Tuesday, and Pell does not wear Grey. This limits Pell's day and color possibilities.", "answer": "None"} ```
    e.g. ...ell cannot be on Thursday or Tuesday, and Quinn/Lio/Jory have sequential constraints.", "answer": "None" } ```
blank answers by tool (of 12 puzzles): ant0:notes 1, ant1:notes 1, ant2:notes 1, ant4:both 1, ant5:both 2, ant5:notes 1, ant6:alone 3, ant6:both 2, ant6:notes 2, ant7:alone 1, ant7:both 3, ant7:notes 2, reader 2
blank answers by kind: arith 12/75; assign 8/75; order 1/75; syllogism 1/75
code checker, try by try (replayed from the saved programs):
  order (answer C): right; did not come back as a program in a ```python block | settled on C (a try was missing from the saved calls)
  assign (answer E): unsettled; did not come back as a program in a ```python block | did not come back as a program in a ```python block | did not come back as a program in a ```python block (a try was missing from the saved calls)
  arith (answer A): right; did not come back as a program in a ```python block | settled on A (a try was missing from the saved calls)
  syllogism (answer A): unsettled; did not come back as a program in a ```python block | did not come back as a program in a ```python block | did not come back as a program in a ```python block (a try was missing from the saved calls)
  order (answer A): unsettled; did not come back as a program in a ```python block | did not come back as a program in a ```python block | did not come back as a program in a ```python block (a try was missing from the saved calls)
  assign (answer E): unsettled; did not come back as a program in a ```python block | did not come back as a program in a ```python block | did not come back as a program in a ```python block (a try was missing from the saved calls)
  arith (answer C): unsettled; did not come back as a program in a ```python block | marked 0 options as correct (none), but exactly one option is correct, so a rule or the question's ask is coded wrong | marked 0 options as correct (none), but exactly one option is correct, so a rule or the question's ask is coded wrong (a try was missing from the saved calls)
  syllogism (answer B): right; settled on B
  order (answer A): unsettled; did not come back as a program in a ```python block | did not come back as a program in a ```python block | did not come back as a program in a ```python block (a try was missing from the saved calls)
  assign (answer C): unsettled; did not come back as a program in a ```python block | did not come back as a program in a ```python block | did not come back as a program in a ```python block (a try was missing from the saved calls)
  arith (answer C): right; settled on C
  syllogism (answer D): unsettled; did not come back as a program in a ```python block | did not come back as a program in a ```python block | did not come back as a program in a ```python block (a try was missing from the saved calls)
average tokens a call by tool: ant0 508, ant1 514, ant2 548, ant3 533, ant4 506, ant5 545, ant6 453, ant7 455, code 995, notes 582, reader 419
```

## Fixes (same day, on PR 27)

1. A reply that runs out of room now comes back with its text, is billed, and is marked "cut off", instead of being
   thrown away.
2. When an answer names no option (cut off, "None", or the wrong form), the harness asks once more, briefly: pick
   exactly one, reply with just the letter (30 tokens at most). Both calls are billed.
3. An answer given as an option's text or number ("15" for "15 hours") counts as that option's letter, but only when
   exactly one option matches. "None of the above" and a number two options share stay blank. The same reading is used
   on the saved night-run replies, for free.
4. The code checker's ask is shorter and clearer: under 60 lines, no comments, a three-step shape (list every case,
   keep the ones that fit the rules, test each option the way the question asks), and fractions instead of decimals
   for numbers. When a program is cut off, the next try is told to write a much shorter one.
5. When the code checker narrows a puzzle down to a few options without settling it, the ants' vote is taken among
   those options only.
6. Diagnose now shows cut-off replies as cut off.

Check: `CHECK_PROMPT.md` reruns the same 12 puzzles (seed 258388501). It reuses every saved call that still fits and
only asks again where a fix changes the call.

## Check run (same day, same 12 puzzles)

Eshaan ran `CHECK_PROMPT.md`. It finished first time: no stalls, no retries, both servers it started shut down.

- **Blank answers fell from 22 to 5** (of 300 answer steps), and unusable replies from 46 to 8. The short re-ask was
  used 14 times and fixed 9.
- **The reader went from 75.0% to 83.3% on these 12 puzzles**, because its cut-off replies now count. It also costs
  more than it looked: 892 tokens a question, not 419, because cut-off replies are now billed instead of hidden.
- **The code checker is still stuck at 4 of 12.** Nearly every try runs to the 2,000-token limit, even when told to
  write less, and it costs about 6,000 tokens a puzzle (15 times the reader's old figure). Asking for shorter programs
  didn't work.
- The live harness tied the reader (83.3%), as expected with only 12 puzzles.

The two counts the Mac flagged were both diagnose bugs. The result block was right.

- **Re-asks (9 of 14 fixed vs 8 fixed and 4 blank):** diagnose read only the last part of each note. Two re-asked
  answers also had a "load-time outlier" note added after the re-ask, so they were counted under that instead. That is
  14 in all, and 9 fixed, as the result said.
- **392 calls and 50 failures:** diagnose counted every call in the saved log, including the first run's calls. The
  24 thrown-away code programs from that run had a different prompt and were never asked again. This run's own calls
  were fewer and had far fewer failures.

```
HARNESS RESULT (paste this back to Claude in Duet)
import: set1 160q 100.0% priced 9 loads, 86 blank answers read from their saved reply; set2 160q 100.0% priced 10 loads, 86 blank answers read from their saved reply; set3 160q 100.0% priced 10 loads, 84 blank answers read from their saved reply
saved night-run answers: learned on set1+set2 (320), scored on set3 (160); 26 recipes compared
  harness 89.4% vs reader 82.5%: +6.9 points (range +3.1 to +10.6); tokens x1.108, time x0.799, size-weighted tokens x0.819; VERDICT more accurate, but not cheaper on both tokens and time
  policy: default reader; by kind of request {"rules+categorical:must": "dm:jev:dm1"}
  top recipes by value: reader 82.5%, 378 tok, 8.1s; dm:jev:dm1 80.6%, 500 tok, 0.2s; seq:alone:m4>reader 87.5%, 2061 tok, 15.4s; seq:alone:m3>reader 84.4%, 2067 tok, 14.4s; dm:d1-3b:dm1 68.1%, 194 tok, 0.1s
  on the accuracy/tokens/time frontier: harness, reader, dm:jev:dm1, dm:d1-3b:dm1
  by dev kind (reader vs harness): arith 97.5% vs 97.5%; assign 72.5% vs 72.5%; order 87.5% vs 87.5%; syllogism 72.5% vs 100.0%
smoke sweep: 12 new dev puzzles (seed 258388501), 27 tools, 357 calls in 27.4 min wall; 3 load-time outliers counted at the median; 8 unusable replies; 9/14 blank answers fixed by a short re-ask; code checker settled 4/12, right on 4
smoke (tiny, k-fold): 3-fold on 12 questions (each routed by a policy that never saw it); 34 recipes compared
  harness 83.3% vs reader 83.3%: +0.0 points (range +0.0 to +0.0); tokens x1.0, time x1.0, size-weighted tokens x1.0; VERDICT ties it on accuracy, not cheaper on both
  policy: default reader; by kind of request {}
  top recipes by value: reader 83.3%, 892 tok, 13.5s; seq:alone:m3>reader 75.0%, 2233 tok, 14.3s; seq:notes:m4>reader 83.3%, 5235 tok, 32.5s; seq:alone:m4>reader 75.0%, 2847 tok, 17.5s; seq:notes:m3 75.0%, 4286 tok, 26.3s
  on the accuracy/tokens/time frontier: harness, reader, seq:alone:m3, seq:alone:m2
  by dev kind (reader vs harness): arith 100.0% vs 100.0%; assign 33.3% vs 33.3%; order 100.0% vs 100.0%; syllogism 100.0% vs 100.0%
planner live on 4 more new puzzles: accuracy 100.0%, 403 tokens and 5.2s a question (model loads included, 5s wall); recipes used ['harness:reader']
waits for a stalled drive or server: 0 (0s)
DEV NUMBERS ONLY: code-made puzzles and saved dev answers, not the final test. They prove nothing yet.

HARNESS DIAGNOSE (paste this back to Claude in Duet)
saved calls: 392, failed: 50
failed replies by model:
  LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit: 15 of 79; Unusable (stop stop) x13, Unusable (stop length) x2; reply length up to 45 tokens
  ibm-granite/granite-4.2-3b-q4-mlx: 2 of 74; Unusable (stop length) x2; reply length up to 1200 tokens
  mlx-community/Qwen3.5-4B-MLX-4bit: 1 of 73; Unusable (stop stop) x1; reply length up to 242 tokens
  mlx-community/gemma-4-12B-it-4bit: 28 of 92; ProviderTruncated (stop -) x24, Unusable (stop length) x2, Unusable (stop stop) x2; reply length up to 4000 tokens
  mlx-community/gemma-4-e2b-it-4bit: 4 of 74; Unusable (stop stop) x4; reply length up to 59 tokens
blank answers by tool (of 12 puzzles): ant0:notes 1, ant1:notes 1, ant6:notes 1, ant7:both 1, reader 1
blank answers by kind: arith 1/75; assign 4/75; order 0/75; syllogism 0/75
code checker, try by try: every unsettled puzzle was "cut off at 2000 tokens" on all three tries; the 4 settled ones
  settled after one cut-off or on the first try (full lines in the Mac's harness2.log)
short re-asks for the letter: fixed by a short re-ask for the letter x8, still blank after a short re-ask for the letter x4, load-time outlier counted at the median (107s) x1, load-time outlier counted at the median (111s) x1
average tokens a call by tool: ant0 525, ant1 531, ant2 571, ant3 533, ant4 589, ant5 666, ant6 497, ant7 519, code 6079, notes 582, reader 892
```

## Changes after the check run

1. **The code checker is benched.** It is off by default in live runs; `--code` turns it back on. When it is on, it
   gives up after two cut-off programs instead of paying for a third. Diagnose now shows how each cut-off program
   starts and ends, so a later try can see whether the model writes prose, loops, or just long code.
2. **Diagnose counts only this run's calls.** It replays the run's own sweep against the saved log and counts only
   the calls that replay asks for. It also says how many saved answers the replay matched. Re-asks are counted the
   same way as in the result block.
3. **The night-run replay now bills cut-off replies.** The night run's provider threw away replies that ran out of
   room, so the replay billed them as 0 tokens and 0 seconds. They are now billed at their length limit, with the
   usual prompt and time for that many tokens. This changes the token and time ratios, so `REPLAY_PROMPT.md` reruns
   the free part (no models, about 2 minutes) together with the fixed diagnose.

## Free rescore (same day)

Eshaan ran `REPLAY_PROMPT.md` (no models). Both counts now agree, and the replay of the smoke run matched every saved
answer (324 of 324).

- **The night-run win holds: +6.9 points** (89.4% vs 82.5%, range +3.1 to +10.6), from the same one move (syllogisms
  to Jev).
- **Counting cut-off replies honestly almost closes the token gap.** About 90 cut-off replies a set had been billed at
  0. The reader's real cost is 588 tokens and 20 seconds a question, not 378 and 8. The harness is now 2.5% over the
  reader on tokens (was 11%) and 16% faster.
- **The code checker's real problem is a loop, not length.** Every cut-off program starts as proper Python and then
  repeats a comment line ("# Let's check each option.") until it hits the limit. That is a known way small models get
  stuck at low temperature. Asking for shorter programs can't fix it. A repetition penalty or a different coding model
  might, so it stays benched until there is a run to test that.

```
HARNESS RESULT (paste this back to Claude in Duet)
import: set1 160q 100.0% priced 9 loads, 86 blank answers read from their saved reply, 93 cut-off replies billed at their limit; set2 160q 100.0% priced 8 loads, 86 blank answers read from their saved reply, 80 cut-off replies billed at their limit; set3 160q 100.0% priced 10 loads, 84 blank answers read from their saved reply, 91 cut-off replies billed at their limit
saved night-run answers: learned on set1+set2 (320), scored on set3 (160); 26 recipes compared
  harness 89.4% vs reader 82.5%: +6.9 points (range +3.1 to +10.6); tokens x1.025, time x0.844, size-weighted tokens x0.839; VERDICT more accurate, but not cheaper on both tokens and time
  policy: default reader; by kind of request {"rules+categorical:must": "dm:jev:dm1"}
  top recipes by value: dm:jev:dm1 80.6%, 500 tok, 0.2s; reader 82.5%, 588 tok, 20.0s; seq:alone:m4>reader 87.5%, 2281 tok, 24.7s; seq:alone:m3>reader 84.4%, 2261 tok, 22.2s; dm:d1-3b:dm1 68.1%, 194 tok, 0.1s
  on the accuracy/tokens/time frontier: harness, reader, dm:jev:dm1, dm:d1-3b:dm1
  by dev kind (reader vs harness): arith 97.5% vs 97.5%; assign 72.5% vs 72.5%; order 87.5% vs 87.5%; syllogism 72.5% vs 100.0%
DEV NUMBERS ONLY: code-made puzzles and saved dev answers, not the final test. They prove nothing yet.

HARNESS DIAGNOSE (paste this back to Claude in Duet)
this run's calls: 349 (of 392 in the saved log; the rest are from earlier runs), failed: 25; replay matched 324/324 saved answers
blank answers by tool (of 12 puzzles): ant0:notes 1, ant1:notes 1, ant6:notes 1, ant7:both 1, reader 1
blank answers by kind: arith 1/75; assign 4/75; order 0/75; syllogism 0/75
short re-asks for the letter: 14, fixed 9, still blank 5
code checker (cut-off programs, how they end): "...# Let's check each option. # Let's check each option. # Let's
  check each o", "...# Let's write the code. # Let's write the code. # Let's write th", "...# Let's check the
  logic: # If Day 0 is Orange, then Day 1 is", "...} # Type 2: {F, Q} # Type 3: {Q} # Type 4: {Y} # Type 5: {Q, Y}"
  (full block in the thread; every program starts as Python)
average tokens a call by tool: ant0 525, ant1 531, ant2 571, ant3 533, ant4 589, ant5 666, ant6 497, ant7 519, code 6079, notes 582, reader 892
```

## Cost-tuned learner

The learner weighs accuracy against cost with a small fixed weight (0.02), so it nearly always prefers accuracy. It
now also tries a range of cost weights (0.02 to 1.0), each judged by 5-fold on the training sets only, and keeps the
one that is cheaper than the reader on both tokens and time with the best accuracy. If none is cheaper on both, it
keeps 0.02 (tightened later; see "Tuner that only trusts savings that hold up"). The test set is scored once, with
the weight it picked, so the result stays honest. `mac` prints this as
an extra "cost-tuned" part after the usual one.

## First cost-tuned rescore (same day)

The tuner kept 0.02, the same policy as before, so nothing changed on the test set: still +6.9 points but tokens
x1.025. On the training sets 0.02 looked cheaper (tokens x0.99), and the rule took that at face value. A saving of 1%
on 320 questions is inside the noise, and on new questions it came out 2.5% dearer.

The training rows show where the real trade is. At 0.05 the harness gives up about one point on the training sets
(84.4% against 85.3%) for 14% fewer tokens and 60% less time. Past 0.1 it starts losing to the reader on accuracy.

```
import: set1 160q 100.0% priced 9 loads, 86 blank answers read from their saved reply, 93 cut-off replies billed at their limit; set2 160q 100.0% priced 8 loads, 86 blank answers read from their saved reply, 80 cut-off replies billed at their limit; set3 160q 100.0% priced 10 loads, 84 blank answers read from their saved reply, 91 cut-off replies billed at their limit
saved night-run answers: learned on set1+set2 (320), scored on set3 (160); 26 recipes compared
  harness 89.4% vs reader 82.5%: +6.9 points (range +3.1 to +10.6); tokens x1.025, time x0.844, size-weighted tokens x0.839; VERDICT more accurate, but not cheaper on both tokens and time
  policy: default reader; by kind of request {"rules+categorical:must": "dm:jev:dm1"}
  top recipes by value: dm:jev:dm1 80.6%, 500 tok, 0.2s; reader 82.5%, 588 tok, 20.0s; seq:alone:m4>reader 87.5%, 2281 tok, 24.7s; seq:alone:m3>reader 84.4%, 2261 tok, 22.2s; dm:d1-3b:dm1 68.1%, 194 tok, 0.1s
  on the accuracy/tokens/time frontier: harness, reader, dm:jev:dm1, dm:d1-3b:dm1
  by dev kind (reader vs harness): arith 97.5% vs 97.5%; assign 72.5% vs 72.5%; order 87.5% vs 87.5%; syllogism 72.5% vs 100.0%
cost-tuned: the cost weight was picked by 5-fold on set1+set2 only (the test set unseen); picked 0.02
  tried on the training sets: 0.02: 85.3%, tokens x0.99, time x0.723; 0.05: 84.4%, tokens x0.861, time x0.401; 0.1: 82.2%, tokens x0.704, time x0.16; 0.2: 80.0%, tokens x0.647, time x0.16; 0.3: 76.6%, tokens x0.508, time x0.145; 0.5: 72.2%, tokens x0.435, time x0.12; 1.0: 57.8%, tokens x0.324, time x0.005
  harness 89.4% vs reader 82.5%: +6.9 points (range +3.1 to +10.6); tokens x1.025, time x0.844, size-weighted tokens x0.839; VERDICT more accurate, but not cheaper on both tokens and time
  policy: default reader; by kind of request {"rules+categorical:must": "dm:jev:dm1"}
DEV NUMBERS ONLY: code-made puzzles and saved dev answers, not the final test. They prove nothing yet.
```

## Tuner that only trusts savings that hold up

What changed (commit after 41f20a0):

* The scorecard now gives a 95% range for the tokens and time ratios too, by resampling questions in pairs, the same
  way the accuracy gap gets its range.
* The tuner picks, in this order: the most accurate weight that beats the reader on all three *with confidence*
  (accuracy range above 0, tokens and time ranges under 1); else the most accurate weight that is surely cheaper on
  both and not surely less accurate; else 0.02. A ratio of x0.99 whose range reaches past 1 no longer counts.
* The grid is finer between 0.02 and 0.2 (0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0).
* The block prints each weight's gain and ranges, why the weight was picked, and tokens and seconds per dev kind for
  the reader and the harness, so we can see where the tokens go.

An honest caveat: this rule was changed after seeing set 3's first tuned result. The rule itself uses no set 3
numbers, but set 3 has now been looked at twice, so it is a weaker check than a fresh set. That is fine for dev
work. It is exactly why the real test set stays hidden until last.

## Second cost-tuned rescore: beats the reader on all three (same day)

With the tightened rule the tuner picked 0.05 ("beats the reader on all three with confidence" on the training
sets). On set 3 the harness scores 90.0% against the reader's 82.5% (+7.5 points, range +1.9 to +13.1), with tokens
x0.786 (range x0.642 to x0.982) and time x0.209 (range x0.135 to x0.341). All three ranges are on the right side.

The policy flipped: the default is now Jev, and the reader keeps three kinds of request (numbers, "could" order
puzzles, and "must" rule puzzles). Where it came from, by dev kind:

* syllogism: 72.5% to 100%, about the same tokens, 12.7s to 0.2s (Jev).
* assign: 72.5% to 77.5%, 1151 to 658 tokens, 44.7s to 4.6s (Jev on some of them).
* order: 87.5% to 85.0% (one puzzle lost), same cost.
* arith: unchanged.

Jev's price in the ledger is $0.042 per million input tokens, so about $0.00002 a question. The dollars don't move
the picture, but Jev is a paid outside service and is still off in live runs.

Caveats, plainly:

* The rule was tightened after seeing set 3's first tuned result, so set 3 has been looked at twice. The rule uses no
  set 3 numbers, but a fresh set is the real check.
* Most of the gain is syllogisms, and the syllogism maker can be shortcut (next section).

```
import: set1 160q 100.0% priced 9 loads, 86 blank answers read from their saved reply, 93 cut-off replies billed at their limit; set2 160q 100.0% priced 8 loads, 86 blank answers read from their saved reply, 80 cut-off replies billed at their limit; set3 160q 100.0% priced 10 loads, 84 blank answers read from their saved reply, 91 cut-off replies billed at their limit
saved night-run answers: learned on set1+set2 (320), scored on set3 (160); 26 recipes compared
  harness 89.4% vs reader 82.5%: +6.9 points (range +3.1 to +10.6); tokens x1.025 (range x0.931 to x1.095), time x0.844 (range x0.684 to x0.94), size-weighted tokens x0.839; VERDICT more accurate, but not cheaper on both tokens and time
  policy: default reader; by kind of request {"rules+categorical:must": "dm:jev:dm1"}
  top recipes by value: dm:jev:dm1 80.6%, 500 tok, 0.2s; reader 82.5%, 588 tok, 20.0s; seq:alone:m4>reader 87.5%, 2281 tok, 24.7s; seq:alone:m3>reader 84.4%, 2261 tok, 22.2s; dm:d1-3b:dm1 68.1%, 194 tok, 0.1s
  on the accuracy/tokens/time frontier: harness, reader, dm:jev:dm1, dm:d1-3b:dm1
  by dev kind (reader vs harness): arith 97.5% vs 97.5%; assign 72.5% vs 72.5%; order 87.5% vs 87.5%; syllogism 72.5% vs 100.0%
  cost by dev kind (reader vs harness, tokens/seconds): arith 263/8.4s vs 263/8.4s; assign 1151/44.7s vs 1151/44.7s; order 512/14.4s vs 512/14.4s; syllogism 427/12.7s vs 485/0.2s
cost-tuned: the cost weight was picked by 5-fold on set1+set2 only (the test set unseen); picked 0.05 (beats the reader on all three with confidence)
  tried on the training sets (points vs reader):
    0.02: 85.3% (+6.6, range +3.1 to +10.0), tokens x0.99 (range x0.904 to x1.071), time x0.723 (range x0.582 to x0.849)
    0.03: 85.3% (+6.6, range +2.8 to +10.3), tokens x0.919 (range x0.821 to x1.024), time x0.569 (range x0.418 to x0.715)
    0.05: 84.4% (+5.6, range +1.2 to +10.0), tokens x0.861 (range x0.753 to x0.983), time x0.401 (range x0.265 to x0.549)
    0.07: 83.4% (+4.7, range +0.3 to +9.1), tokens x0.822 (range x0.713 to x0.948), time x0.348 (range x0.221 to x0.486)
    0.1: 82.2% (+3.4, range -1.6 to +8.1), tokens x0.704 (range x0.601 to x0.832), time x0.16 (range x0.112 to x0.231)
    0.15: 80.6% (+1.9, range -3.1 to +6.9), tokens x0.676 (range x0.575 to x0.798), time x0.16 (range x0.112 to x0.231)
    0.2: 80.0% (+1.2, range -3.8 to +6.2), tokens x0.647 (range x0.548 to x0.765), time x0.16 (range x0.112 to x0.231)
    0.3: 76.6% (-2.2, range -7.5 to +2.8), tokens x0.508 (range x0.429 to x0.606), time x0.145 (range x0.101 to x0.213)
    0.5: 72.2% (-6.6, range -12.2 to -0.9), tokens x0.435 (range x0.366 to x0.518), time x0.12 (range x0.08 to x0.18)
    1.0: 57.8% (-20.9, range -27.5 to -14.4), tokens x0.324 (range x0.277 to x0.38), time x0.005 (range x0.004 to x0.007)
  harness 90.0% vs reader 82.5%: +7.5 points (range +1.9 to +13.1); tokens x0.786 (range x0.642 to x0.982), time x0.209 (range x0.135 to x0.341), size-weighted tokens x0.369; VERDICT beats it on all three
  policy: default dm:jev:dm1; by kind of request {"num": "reader", "rules+order:could": "reader", "rules:must": "reader"}
  by dev kind (reader vs harness): arith 97.5% vs 97.5%; assign 72.5% vs 77.5%; order 87.5% vs 85.0%; syllogism 72.5% vs 100.0%
  cost by dev kind (reader vs harness, tokens/seconds): arith 263/8.4s vs 263/8.4s; assign 1151/44.7s vs 658/4.6s; order 512/14.4s vs 512/14.4s; syllogism 427/12.7s vs 485/0.2s
DEV NUMBERS ONLY: code-made puzzles and saved dev answers, not the final test. They prove nothing yet.
```

## The syllogism maker can be shortcut, and a harder one

Most of the harness's gain is syllogisms, where Jev scores 100% and the reader 72.5%. So I checked whether those
puzzles can be answered without logic. `antelligence/harness/devsets.py` has a `shortcut_report`: no-logic guessers
learned on 1,000 puzzles and scored on 1,000 others (chance is 20%).

| maker | options only (which form tends to be right) | surface lookup (form, word places in the premises) |
|---|---|---|
| fresh-question maker (the one the night sets used) | 32.8% | 60.8% |
| balanced maker (new) | 20.0% | 26.4% |

So with the old maker, a lookup with no logic at all gets about 61%, close to the reader's 72.5%. Jev's 100% may be
partly real logic and partly these patterns. We can't tell from the saved sets.

The balanced maker uses the same premises and the same brute-force check, but picks wrong options that look like the
right one: same form, same pairing in the premises, word-place counts within one. `fresh_questions.py` is untouched
(the fresh test's preregistration pins its hash). New dev puzzles (`make_dev`, used by `mac` and `fresh`) now use the
balanced maker; the saved night sets are as they were.

## Fresh check (built, waiting on a Mac run)

`scripts/harness.py fresh` runs the frozen tuned policy (`replay/saved/policy-tuned.json`, learned on set 1 and
set 2 before these puzzles existed) and the reader alone on brand-new puzzles:

* 10 per kind with the balanced maker, every kind;
* 10 old-maker syllogisms, so the two makers can be compared the same day with the same models;
* 10 reworded syllogisms (added Nov 3): balanced, and written in other words. "All A are B" becomes "Each one of
  the A is one of the B", "Some A are not B" becomes "At least one of the A is not one of the B", and the opening
  line and the question are new too. None of their sentences appears in the old wording. If the gain holds on
  these, it isn't tied to how our maker phrases things.

It runs only the reader and the tools the policy uses (Jev, capped by `--jev-cap`), so it is quick and costs well
under a cent. It prints accuracy, tokens and time with ranges, by kind, and which recipes were used. This is the
honest answer to the caveat above: nothing in it was tuned on these puzzles. Prompt: `FRESH_PROMPT.md`.

## The first result block

```
HARNESS RESULT (paste this back to Claude in Duet)
import: set1 160q 100.0% priced 9 loads; set2 160q 100.0% priced 10 loads; set3 160q 100.0% priced 10 loads
saved night-run answers: learned on set1+set2 (320), scored on set3 (160); 26 recipes compared
  harness 89.4% vs reader 82.5%: +6.9 points (range +3.1 to +10.6); tokens x1.108, time x0.799, size-weighted tokens x0.819; VERDICT more accurate, but not cheaper on both tokens and time
  policy: default reader; by kind of request {"rules+categorical:must": "dm:jev:dm1"}
  top recipes by value: reader 82.5%, 378 tok, 8.1s; dm:jev:dm1 80.6%, 500 tok, 0.2s; seq:alone:m4>reader 88.1%, 2067 tok, 15.5s; seq:alone:m3>reader 84.4%, 2068 tok, 14.4s; dm:d1-3b:dm1 68.1%, 194 tok, 0.1s
  on the accuracy/tokens/time frontier: harness, reader, dm:jev:dm1, dm:d1-3b:dm1
  by dev kind (reader vs harness): arith 97.5% vs 97.5%; assign 72.5% vs 72.5%; order 87.5% vs 87.5%; syllogism 72.5% vs 100.0%
smoke sweep: 12 new dev puzzles (seed 258388501), 27 tools, 342 calls in 33.2 min wall; 1 load-time outliers counted at the median; 46 unusable replies; code checker settled 4/12, right on 4
smoke (tiny, k-fold): 3-fold on 12 questions (each routed by a policy that never saw it); 34 recipes compared
  harness 75.0% vs reader 75.0%: +0.0 points (range +0.0 to +0.0); tokens x1.0, time x1.0, size-weighted tokens x1.0; VERDICT ties it on accuracy, not cheaper on both
  policy: default reader; by kind of request {}
  top recipes by value: reader 75.0%, 419 tok, 12.0s; seq:alone:m3>reader 75.0%, 2235 tok, 14.4s; seq:alone:m4>reader 75.0%, 2370 tok, 16.0s; code>reader 75.0%, 1323 tok, 49.2s; reader+check 75.0%, 1413 tok, 54.2s
  on the accuracy/tokens/time frontier: harness, reader, seq:alone:m2, seq:alone:m2>code
  by dev kind (reader vs harness): arith 100.0% vs 100.0%; assign 33.3% vs 33.3%; order 100.0% vs 100.0%; syllogism 66.7% vs 66.7%
planner live on 4 more new puzzles: accuracy 75.0%, 314 tokens and 3.9s a question (model loads included, 16s wall); recipes used ['harness:reader']
waits for a stalled drive or server: 0 (0s)
DEV NUMBERS ONLY: code-made puzzles and saved dev answers, not the final test. They prove nothing yet.
```
