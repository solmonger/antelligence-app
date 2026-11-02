# Real forecasts on the Mac, free: can the ants do something the reader can't?

Date: 2026-11-02. Status: **frozen, not run yet.** Everything here was written and frozen before any model call.
Harness: `antelligence/experiments/forecast_local.py`. Script: `scripts/forecast_local.py`. Tests:
`tests/smarter_swarm/test_forecast_local.py`. Fingerprint: `docs/research/forecast-local-20261102/prereg.json`.
Mac prompt: `docs/research/forecast-local-20261102/CLAUDE_CODE_PROMPT.md`.

## Why this, and why now

- The fair test (`docs/plans/2026-10-29-fair-test.md`) never ran. Every try on the Nous portal was refused for
  credit, so it has no score.
- Three local runs since then all say the same thing on exam questions: the swarm doesn't beat one good reader. The
  runs were shared-read, fresh questions and the night run (`docs/research/night-20261031/RESULTS.md` on
  `claude/project-thread-fly2s6`). The reader's notes lift the ants by about 8 points, but the ants still finish
  about 6 points under the reader.
- So the open question is what ants can do that the reader can't. Exams reward one strong reasoner. Forecasting is
  the one place where averaging many different minds is a proven win: the "wisdom of crowds", and an ensemble of 12
  LLMs matching a human forecasting crowd (Schoenegger et al., "Wisdom of the Silicon Crowd", 2024). If ants can't
  add anything here either, that is a strong answer too.
- The fair test's 101 real forecasts are ready and frozen, and the night run's local models cost nothing. This runs
  that set on the Mac with no Nous call at all.

## What stays exactly as frozen

From the fair test, unchanged:
- The 101 forecasts and their answer file.
- The prompts, the five forecast lenses, the JSON answer, the parser, combining by mean probability clipped to
  0.01–0.99, temperature 0.7 and the per-call seeds.
- The pass line: a Brier gain of at least 0.005 with the 95% range above 0.
- The rule that under 90% usable answers means "invalid run".

## What changes

- **Models.** These are the night run's reader and four ants, unchanged
  (`docs/research/forecast-local-20261102/models.json`). They are 4-bit MLX builds served one at a time by
  `mlx_lm.server` on `127.0.0.1`, with thinking off:

  | Role | Model | Released | Source |
  |---|---|---|---|
  | reader | Gemma 4 12B | 2026-06-03 | Google's Gemma releases page |
  | ant | Gemma 4 E2B | 2026-03-31 | Google's Gemma releases page |
  | ant | Qwen 3.5 4B | 2026-03-02 (one page says Feb 24; the later day is kept) | llm-stats.com, apxml.com |
  | ant | Granite 4.2 3B | 2026-08-25 | datanorth.ai, llmreference.com |
  | ant | LFM2.5 1.2B Instruct | 2026-01-20 | openrouter.ai |

- **Reply length.** Up to 800 tokens per reply, not 4,000. Thinking is off, so a JSON answer is about 100 tokens;
  800 only cuts off loops. A cut-off reply counts as unusable.
- **The model cache must be on the internal disk.** The WD_BLACK drive that held it froze on Nov 1. The script
  refuses a cache under `/Volumes/`. It starts `mlx_lm.server` with `HF_HUB_CACHE` set to that folder and downloads
  turned off, so a missing model fails at once rather than downloading somewhere else.
- **Port 8611.** Nothing else on the Mac uses it; the night run clashed with the embedding service on 8091.

## Why memory can't explain a win

- **The reader came out on June 3, a month before the first forecast resolved (July 3).** So it cannot have read any
  outcome.
- **Granite 4.2 3B came out on August 25, after 34 of the forecasts resolved.** So any test that uses the ants is
  scored only on the 67 forecasts that resolved on or after August 25 (`LATE_FROM`). On those, no model can have read
  the outcome. The test file checks that `LATE_FROM` equals the newest release day in the model file.

## The two tests (each gets its own verdict)

**Test A: lenses.** This is the fair test's own primary, with the local reader as the model.
- `swarm5`: the reader through the 5 lenses, each answering alone, averaged.
- Against `single5`: the reader's plain prompt sampled 5 times, averaged.
- Scored on all 101 forecasts.

**Test B: a mixed crowd.** This tests what the ants add.
- `mix5`: the reader's first plain answer plus one plain answer from each of the 4 ants, averaged.
- Against the same `single5`.
- Scored on the 67 forecasts resolved on or after 2026-08-25.

Both sides of each test use 5 calls per question, so neither side gets more tries. `mix5` and `single5` share the
reader's first answer. That is on purpose: the question is whether swapping four of the reader's extra samples for
four different small models helps.

**Verdict rules.** Each test passes when its gain is at least 0.005 and the low end of its 95% range is above 0. A
test whose answers are under 90% usable is an "invalid run" and gets no verdict.

There are two tests, so the chance that one passes by luck alone is about twice that of one test. Read a single
narrow pass with that in mind.

**Secondaries.** These carry no verdict:
- Every arm's Brier score on all 101 and on the late 67.
- `mix5` against `single5` on all 101.
- `seen10` against `single10` (does reading each other help).
- The 4 ants alone against `single5`, on the late 67.
- Each model's Brier score from one answer.
- Usable share per model.
- Results by field.
- The market crowd's Brier score on the same questions. The market's price comes from shortly before the market
  closed, so it is a yardstick, not a fair rival.

## What each result would mean

- **B passes, A doesn't.** Different small models add something one big model can't get by asking itself again. That
  points the swarm at mixed crowds on judgment calls, not on exams.
- **A passes.** Lenses help even with one model, the fair test's original hope.
- **Neither passes.** Even on forecasting, the ants add nothing over the reader. Then the swarm's value has to come
  from what the ants can do that a single call can't do at all: watching many things at once, or acting in a world.
  It won't come from answering better.
- **Invalid run.** The plumbing failed and nothing was measured. The report says why.

## Size and time

- 101 forecasts: 2,020 reader calls (10 plain, 5 lenses and 5 second-round lenses each) and 404 ant calls. $0.
- **Time (an estimate).** The fresh-question run averaged about 1.75 seconds per local call, mostly small ants. The
  12B reader is slower, so this should take roughly 2 to 4 hours, plus a few minutes per model load.
- One model is in memory at a time, and the reader is about 10 GB.

## How it runs

1. `copy-models` copies the five models from wherever the cache is now to `~/antelligence-models/hub`. It only
   copies, keeps the cache's links, skips a model already copied whole, and stops if the internal disk would keep
   under 15 GB free.
2. `check` makes no model call. It confirms that:
   - the plan is unchanged;
   - the cache is internal and holds all five models;
   - the port is free;
   - there is room on disk and in memory;
   - `mlx_lm.server` exists.
3. `run --live` loads the reader once for all its calls, then each ant once.
   - It caches every reply, so a rerun resumes, and a model whose calls are all cached is never loaded.
   - It stops after 25 unusable replies in a row, or when memory is short, and says why.
   - It never opens the answer file.
4. `score` reads the answers and prints both verdicts and a block to paste back.

The Mac prompt adds the house rules: no push, no spending, don't touch other processes, and one model at a time.
