# Shared-read test, run 1: results

Run on Eshaan's Mac Studio (36 GB) on 2026-10-30, all 240 questions, free and local. The plan, frozen before any
call: `docs/plans/2026-10-30-shared-read.md`. How the run differed from that plan (MLX instead of Ollama, the fallback
reader, one ant swapped) is under "How run 1 ran" there. The freeze it ran under: `run1-prereg.json`. Its meta
file: `run1-meta.json`.

## The answer

The reader's notes **help** the ants: the vote goes from 47.5% to 55.4% (+7.9 points, 95% range +2.1 to +14.2).
They also make the ants **fail together**: the alike-mistakes measure rises by 0.055, just over the 0.05 line, with
the interval just above 0. And the swarm **does not beat the reader on its own**: the reader scores 63.3%, ahead of
every swarm arm. In the plan's own terms this is "helps but alike" and "doesn't beat the reader alone". The reader
does the thinking and the ants echo it.

So the cache version of SwaT, which needs a rented GPU to train its translator, is not worth paying for now. The plan
made it worth it only if the notes helped without making the ants alike and the swarm beat the reader. Neither
happened, and a richer shared cache is likely to make the ants more alike than text does.

## Score, as `score` printed it

```
240 questions; usable answers 93%; notes that leaked an option 0%
Reader (mlx-community/gemma-4-12B-it-4bit) answering alone: 63.3%
  ants alone: vote 47.5%, average ant 39.3%, mistakes alike +0.26, same wrong pick 42%
  ants notes: vote 57.5%, average ant 46.2%, mistakes alike +0.35, same wrong pick 50%
  ants both : vote 55.4%, average ant 43.4%, mistakes alike +0.31, same wrong pick 43%
Notes + question vs alone (vote): +7.9% (95% range +2.1% to +14.2%)
Rise in alike mistakes: +0.055 (95% range +0.002 to +0.108)
Lift per family (score with notes / score alone): gemma 1.194, qwen 1.12, granite 1.17, lfm 0.824
Verdict: {"shared_read_helps": "yes", "ants_think_alike": "yes", "swarm_beats_reader_alone": "no"}
```

`score` reported no change since the freeze. (The "lift per family" line uses the `both` arm.)

## What else the numbers say

- **Notes alone beat the passage.** Ants that never saw the passage, only the reader's 200 words, voted 57.5%. Ants
  that read the full question themselves voted 47.5%. Adding the passage back to the notes (`both`) did slightly
  worse than notes alone (55.4%). Small models seem to do better with the reader's digest than with the raw text.
  `score` gives no interval for notes against both, so read this as a lead, not a finding.
- **The notes copy the reader's mistakes too.** When two ants are both wrong, they pick the same wrong letter 50% of
  the time with notes, against 42% alone. Guessing blind would give about 25%.
- **The smallest ant got worse with the notes.** LFM2.5 1.2B scored 0.82 times its own score alone. The other three
  families gained 12% to 19%.
- **The reader's lead is, if anything, bigger than shown.** 25 of its 240 answers looped until the token cap and
  count as wrong. On the 215 it did answer, it was right about 71% of the time.
- **The 60-question quick pass was misleading.** There, the alike measure didn't move (0.347 to 0.348) and the reader
  scored 70%. The rise only showed on all 240. That is why the verdict waits for the full run.

## What it means for Antelligence

- **Keep think-alone as the default.** Giving every ant the same input made their mistakes alike again. That matches
  the earlier finding that ants who see each other fail together (`docs/plans/2026-10-29-swarm-at-scale.md`).
- **Small ants don't add to a bigger model on reasoning questions.** The plan's own rule for this outcome: put
  effort into the reader rather than the ants, and try the reader as one voice among the ants rather than feeding
  every ant.
- **Don't train a SwaT translator for the swarm yet.** No GPU spend is needed or proposed.

## Free next checks, from the answers already on the Mac

Neither asks a model anything new. Both run in seconds in `antelligence-fairtest`, once this branch is pulled there:

```
uv run python scripts/shared_read.py explore
```

`explore` reads other swarm shapes off run 1's answers: the reader as one voter among the 8 ants, half the ants given
the notes (averaged over all 70 ways to pick the half), and each arm without the weakest family. It is exploratory,
because the shapes were picked after seeing this result, so it points to the next test rather than settling
anything.

```
uv run python scripts/shared_read.py run --parser 2 --offline --seed-cache docs/research/shared-read-20261030/run/cache.jsonl
uv run python scripts/shared_read.py score --parser 2
```

This re-reads run 1's own replies with parser 2, which also reads Gemma 4 E2B's LaTeX answers. It asks no model
(`--offline`), writes to `run-p2/` and leaves `run/` alone, and `score` labels the result as a re-read, not the
preregistered score. It shows whether those unusable answers changed the picture.

## Limits

- One run, one reader (Gemma 4 12B) and one mix of four small ants, on short exam questions rather than the long
  documents where the paper's gains are.
- Text notes, not a KV cache: this tests whether a shared reading herds a swarm, not SwaT's translator.
- 7% of answers were unusable, mostly Gemma 4 E2B's LaTeX answers and the reader's looped answers. That is above the
  90% line, but not nothing. The parser 2 re-read above measures how much it mattered.
