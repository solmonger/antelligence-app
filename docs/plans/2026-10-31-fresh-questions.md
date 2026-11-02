# Fresh questions: is Jev's 81.7% reasoning or memory?

Date: 2026-10-31. Status: **frozen before any real call** (`docs/research/fresh-20261031/prereg.json`). Code:
`antelligence/experiments/fresh_questions.py` (the question maker), `antelligence/experiments/dm_leak.py` (the
letter check), `scripts/fresh_test.py`, `scripts/mlx_standin.py`. Tests: `tests/smarter_swarm/test_fresh.py`. Mac
prompt: `docs/research/fresh-20261031/CLAUDE_CODE_PROMPT.md`.

## Why

In run 1 (`docs/research/dm-addon-20261030/RESULTS.md`), Jev alone got 81.7% of 240 AGIEval exam questions right,
18.3 points above the local reader (Gemma 4 12B), and 75% on the LSAT puzzles and AQuA-RAT maths. AGIEval is public,
so Jev may have seen those questions. Eshaan's ask: check with questions that can't be public.

## Two checks

### 1. New questions, made on the Mac at run time

`make` draws a seed when it runs and writes 160 questions, 40 of each kind. None exists before the run.

| Kind | Like | What it is |
|---|---|---|
| `order` | LSAT analytical reasoning | six things in a row under 4 to 6 rules; "which could be the order" or "which must be true" |
| `assign` | LSAT analytical reasoning | four people, each with one colour and one day; clues leave one solution; "which must be true" |
| `arith` | AQuA-RAT | a short word problem (rates, percents, speed, mixtures, ages, remainders, interest) with a whole-number answer; wrong options are the usual slips |
| `syllogism` | LSAT logical reasoning, LogiQA | statements about made-up kinds of thing; "which must also be true" |

- Every question has five options, exactly one right. That is checked by brute force over every case before
  anything is written, and the tests check it again by other means.
- Names are random letters (never A to E), random people's names and made-up words, so nothing can be recognised.
- A syllogism answer is right only if it follows whether or not every kind is assumed to exist, so the answer never
  hangs on that reading.
- Answers go to their own file; only `score` opens it.

The same arms as run 1 then answer them, with the same harness and the same models (run 1's `models.json`, with one
change: parser 2, so Gemma 4 E2B's LaTeX doesn't make its answers unusable):

- the reader alone; the 8 ants alone, with the reader's notes, and both (`scripts/shared_read.py`'s own code);
- Jev and d1-3B alone (`dm1`), reading the ants' answers (`read`), and as one more voter (`vote`)
  (`antelligence/experiments/dm_addon.py`'s own code).

No decision model is asked unless at least 90% of the ants' answers are usable. Run 1's fair-test source went through
with every ant empty, and this guard is the fix.

### 2. Moving the letters on the 240 AGIEval questions

The same questions Jev got 81.7% on, asked again with one change each:

- **relabel**: every option moved to a different letter, by a fixed derangement per question, and the answer mapped
  back. A model that reasons doesn't care. A model that remembers "the answer is C" now picks a wrong option.
- **blind**: the options alone, with no passage and no question. Nobody can reason to the answer from these, so a
  score far above chance means the options were remembered. AQuA-RAT's maths options can also give the answer away
  by how they look, so this one is read by group.

## Frozen reading

**Jev's lead over the reader on new questions** (Jev alone minus the reader alone, paired bootstrap 95% range):

| Reading | Rule |
|---|---|
| holds | lead at least +10 points and its range above 0 |
| gone (run 1's lead was likely memory) | the whole range below +10 points |
| unclear | anything else |

The lead, not Jev's raw score, is what's compared, because the new questions may be harder or easier than AGIEval
for everyone. The reader is the yardstick that stays the same. The reader may have memorised AGIEval too, which would
make a lead that holds a conservative result.

**Letter check:**

| Reading | Rule |
|---|---|
| likely memorised | moving the letters costs at least 10 points (range above 0), or the options alone score at least 25 points above chance (range above 0) |
| no sign of memory | the drop's range is under 5 points and the options-alone range is under 10 points above chance |
| unclear | anything else |
| invalid | under 90% of the answers were usable |

Everything else (ants with notes, Jev reading the ants, Jev as a voter, per-kind scores) is reported next to run 1's
AGIEval numbers, not judged.

## Cost and time

- Jev: about 320 calls on the new questions (cap $0.01) and 480 for the letter check (cap $0.015). With run 1's
  $0.0223, that stays under 5 cents in all. The estimates printed are low by about 1.75 times, as run 1 showed, and
  the caps allow for that.
- Local models: 160 questions times 26 calls, free. One model in memory at a time. `scripts/mlx_standin.py` lets
  the harness unload each MLX model before the next one loads, as run 1's stand-in did.
- d1-3B: free, through `scripts/v41_d1_server.py`.

## Limits

- 160 questions give a 95% range of about ±7 points on a share right; good enough to tell an 18-point lead from none.
- These are made by code, so they are more regular than human-written exam questions. A model can be good at
  AGIEval for real and still worse on these, or the reverse; that is why the reader is the yardstick.
- If Jev was trained on puzzles made like these, it could still have an edge that isn't reasoning. Nothing here can
  rule that out.
