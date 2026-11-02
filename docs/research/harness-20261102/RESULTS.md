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

## The block

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
