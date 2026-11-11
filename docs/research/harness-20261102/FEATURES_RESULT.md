# Borrowed parts: the overnight result (2026-11-02)

Plan: [../../plans/2026-11-02-borrowed-parts.md](../../plans/2026-11-02-borrowed-parts.md). Mac prompt:
[FEATURES_PROMPT.md](FEATURES_PROMPT.md). The block Eshaan pasted back is at the end, unchanged.

Dev numbers only: the night run's saved answers, learned on sets 1 and 2, set 3 scored once. The hidden final test set
was not used.

## Verdict

No switch beat today's harness on set 3. Today's harness stays the best: 90.0% against the reader's 82.5%, with
x0.786 the tokens and x0.209 the time.

That is not the same as "the switches are useless". Three of them (stop, gate, cache) were never used on a single
test question by today's learner, so those rows say nothing about whether they help. Only the router really changed
the picks, and it made them worse.

## Did each switch actually run?

| switch | used on a test question? | how we know | what it means |
|---|---|---|---|
| stop | **no**, with today's learner; **a few**, with the router and the cache count both on | Every row with stop but no router is identical to today's harness, to the third decimal, in accuracy, tokens, time and all three ranges. Two different sets of picks can't land on the same numbers everywhere. With router + cache, adding stop moves tokens from x0.79 to x0.814 at the same 85.0%. | The learner kept the reader everywhere and never chose to ask ants one at a time. Untested, not failed. Where it did run (with the router), it cost tokens and gained nothing. |
| gate | **no**, with today's learner; **a few**, with the router | Same as stop: every gate row without the router equals today's harness. With the router, adding gate moves 85.0% to 84.4% (one answer of 160) and tokens x0.79 to x0.793. | Untested with today's learner. With the router it ran on a handful of questions and lost one answer. |
| cache | it changed the bill only where the router picked repeat asks | With today's learner, fresh tokens equal full tokens (x0.786 both): the picked routes (the reader, and Jev for one kind of question) never ask the same model twice on a question, so there is no shared start to discount. With router + stop, fresh tokens are x0.805 against x0.814. | It had nothing to work on. The server's own count also showed this code's estimate was too generous (below). |
| router | **yes**, on every question | Its rows differ from today's on every number. | It cost 5 points (85.0% against 90.0%) and was slower (x0.348 time). A clear loss on this data. |
| prompt | ran, kept the original | Ten instructions tried on 30 training questions with 4 ants. | Nothing shorter held its accuracy, so the set 3 check was skipped by design. |

Why today's learner never picked stop or gate: those recipes start with the small ants, and the ants are much less
right than the reader. Even when an ant is at least 95% sure of its letter (2489 of its 3817 answers), it is right
only 64% of the time, well under the reader's 82.5%. The cost weight the learner tuned on sets 1 and 2 (0.05) puts
accuracy first, so starting with ants never paid.

The exact counts are in the saved run on the Mac (`features/free/features.json` and `features/gate/features.json`,
the `used` field of each row). That folder isn't on the branch, so the table above is read from the pasted numbers.
[RECHECK_PROMPT.md](RECHECK_PROMPT.md) prints those counts and reruns the scoring with the new "switch used on"
column. It calls no model.

## The other findings

- **Ants are overconfident.** Sure 0.95 to 1.00: 64% right. Sure 0.85 to 0.95: 30%. Under 0.5: 20%. The odds do sort
  answers, but even the surest bin is far below the reader. Rescaling the odds (temperature) keeps the same order, so
  it can't fix a gate. A gate could only work on a model that is as right as the reader when it is very sure. The
  new `calib` stage checks exactly that, per model, on sets 1 and 2, then on set 3.
- **The server reuses less of a shared prompt start than this code estimated.** Measured: 37% on the second way of
  looking at a question, against this code's 80% estimate. So the cache switch's saving was overstated, and it still
  changed nothing.
- **The probe's four calls all showed 0 reused tokens**, even the exact repeat, while the full sweep measured 37%.
  Why isn't known: the exact repeat should have reused its whole start. It doesn't change the results above, which
  use this code's estimate and the sweep's measured share.
- **Answer odds work** on this server (mlx-lm 0.32.2): 100% of 3817 ant answers had them.

## What changed in the code with this result

- `bw.switch_use` counts, per mix, on how many test questions a stop recipe was picked, a gate recipe was picked, or
  the cache count lowered the picked route's bill. The table has a new last column, "switch used on (test
  questions)", so a switch that never ran shows as `stop 0 of 160` and can't be mistaken for one that failed.
- A new `calib` stage (no model calls): for each ant model asked alone, how often it is very sure (0.95 or more), how
  often it is right then, and how often the reader is right on those same questions. A model passes only if it is
  within 2 points of the reader there, on 30 or more answers. Picked on sets 1 and 2, checked once on set 3.
- `--confsweep PATH` reuses a saved answer-odds sweep from another folder, so the rescore can write to a new folder and
  leave last night's results alone.

## The pasted block (unchanged)

```
BORROWED PARTS RESULT (paste this back to Claude in Duet)
saved answers: set1 160q 100% priced; set2 160q 100% priced; set3 160q 100% priced
FREE (saved night answers, no model calls): learned on 320 training questions, scored once on 160 test questions; switches available: stop, router, cache (gate needs the answer-odds sweep)
  reader: 82.5%, 588 tokens, 20.0s a question
  mix | test accuracy (points vs reader, range) | tokens vs reader (range) | fresh tokens vs reader | time vs reader (range) | verdict
  harness (today) | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +stop | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +router | 85.0% (+2.5, -3.1 to +8.8) | x0.79 (0.641 to 0.984) | x0.79 | x0.348 (0.181 to 0.551) | ties it on accuracy, and is cheaper on both
  harness +cache | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +stop +router | 85.0% (+2.5, -3.1 to +8.8) | x0.79 (0.641 to 0.984) | x0.79 | x0.348 (0.181 to 0.551) | ties it on accuracy, and is cheaper on both
  harness +stop +cache | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +router +cache | 85.0% (+2.5, -3.1 to +8.8) | x0.79 (0.641 to 0.984) | x0.79 | x0.348 (0.181 to 0.551) | ties it on accuracy, and is cheaper on both
  harness +stop +router +cache | 85.0% (+2.5, -3.1 to +8.8) | x0.79 (0.641 to 0.984) | x0.79 | x0.348 (0.181 to 0.551) | ties it on accuracy, and is cheaper on both
  best mix: harness +stop (picked on the training sets only: beats the reader on all three with confidence, 84.4% on them by 5-fold)
PROBE: mlx-lm 0.32.2, model LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit; answer odds readable: yes; prefix cache counted: no
  A: question 1, way 1: 241 prompt tokens, 0 reused, 3.67s, answer D, sure 0.697218
  B: question 1, way 2: 247 prompt tokens, 0 reused, 0.17s, answer B, sure 0.141642
  C: A again: 241 prompt tokens, 0 reused, 0.16s, answer D, sure 0.697218
  D: question 2, way 1: 297 prompt tokens, 0 reused, 0.2s, answer A, sure 0.882474
CONFSWEEP: 3988 new ant calls (171 unusable) in 1.2 h; 0 load-time outliers counted at the median; answer odds on 100.0% of 3817 ant answers
  how sure vs how often right: 0.00-0.50: 434 answers, 20% right; 0.50-0.70: 281 answers, 23% right; 0.70-0.85: 269 answers, 28% right; 0.85-0.95: 343 answers, 30% right; 0.95-1.00: 2489 answers, 64% right
  prompt start reused by the server: second way of looking at a question 37% (this code's estimate 80%); first ask of a question 9% (start shared with the question before); server reported it on 100% of calls
GATE (saved answers with the new ant answers in, all four switches): learned on 320 training questions, scored once on 160 test questions; switches available: stop, gate, router, cache
  reader: 82.5%, 588 tokens, 20.0s a question
  mix | test accuracy (points vs reader, range) | tokens vs reader (range) | fresh tokens vs reader | time vs reader (range) | verdict
  harness (today) | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +stop | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +gate | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +router | 85.0% (+2.5, -3.1 to +8.8) | x0.79 (0.641 to 0.984) | x0.79 | x0.348 (0.181 to 0.551) | ties it on accuracy, and is cheaper on both
  harness +cache | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +stop +gate | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +stop +router | 85.0% (+2.5, -3.1 to +8.8) | x0.79 (0.641 to 0.984) | x0.79 | x0.348 (0.181 to 0.551) | ties it on accuracy, and is cheaper on both
  harness +stop +cache | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +gate +router | 84.4% (+1.9, -3.8 to +8.1) | x0.793 (0.644 to 0.983) | x0.793 | x0.346 (0.18 to 0.546) | ties it on accuracy, and is cheaper on both
  harness +gate +cache | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +router +cache | 85.0% (+2.5, -3.1 to +8.8) | x0.79 (0.641 to 0.984) | x0.79 | x0.348 (0.181 to 0.551) | ties it on accuracy, and is cheaper on both
  harness +stop +gate +router | 84.4% (+1.9, -3.8 to +8.1) | x0.793 (0.644 to 0.983) | x0.793 | x0.346 (0.18 to 0.546) | ties it on accuracy, and is cheaper on both
  harness +stop +gate +cache | 90.0% (+7.5, +1.9 to +13.1) | x0.786 (0.642 to 0.982) | x0.786 | x0.209 (0.135 to 0.341) | beats it on all three
  harness +stop +router +cache | 85.0% (+2.5, -3.1 to +8.8) | x0.814 (0.655 to 1.018) | x0.805 | x0.352 (0.186 to 0.553) | ties it on accuracy, and is cheaper on both
  harness +gate +router +cache | 84.4% (+1.9, -3.8 to +8.1) | x0.801 (0.65 to 0.993) | x0.798 | x0.346 (0.18 to 0.546) | ties it on accuracy, and is cheaper on both
  harness +stop +gate +router +cache | 84.4% (+1.9, -3.8 to +8.1) | x0.825 (0.664 to 1.033) | x0.812 | x0.351 (0.184 to 0.549) | ties it on accuracy, and is cheaper on both
  best mix: harness +stop (picked on the training sets only: beats the reader on all three with confidence, 84.4% on them by 5-fold)
PROMPT: tried 10 ant instructions on 30 training questions with 4 ants (one per model): original 48.3%, 368 tokens an answer; picked [original] 48.3%, 368 tokens
  picked instructions: You answer one question on your own. You cannot look anything up. Reply with one JSON object and nothing else. | Reply as JSON: {"reason": "<at most 60 words>", "answer": "<one letter>"}
  nothing shorter held its accuracy on the training questions, so the set 3 check was skipped
DEV NUMBERS ONLY: code-made puzzles and saved dev answers, not the final test. They prove nothing yet.
Set 3 is the night run's third set, scored once after everything was picked on sets 1 and 2; the hidden final test set was not used.
```
