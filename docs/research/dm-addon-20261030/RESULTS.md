# Decision models on top of the ants: results of run 1

Ran on Eshaan's Mac Studio, 2026-10-30, 16:50 to 17:01 New York time, from `CLAUDE_CODE_PROMPT.md` (by way of the
shared-read branch's `OVERNIGHT_PROMPT.md`). Numbers below are copied from the run's own SUMMARY.md as pasted back on
2026-10-31. The run's commit (`afb726c`, branch `claude/overnight-20261030`) is only on the Mac: the Mac's push is
turned off on purpose, so the raw files are not on the fork.

## Shared-read source: valid (240 exam questions, 8 small ants)

| Arm | Share right |
|---|---|
| Jev alone | 81.7% |
| Jev reading the ants' answers | 77.5% |
| Reader alone (Gemma 4 12B) | 63.3% |
| Ants with notes | 57.5% |
| Jev as a ninth voter | 53.8% |
| d1-3B as a ninth voter | 48.8% |
| d1-3B alone | 47.1% |

- Jev alone beats the local reader by 18 points. Jev reading the ants' answers is +14.2% over the reader (95% range
  +7.5% to +20.8%).
- Jev as one more voter gains only +6% over the plain ants vote (range +2.5% to +10%): the 8 ants outvote it.
- d1-3B adds nothing over the ants (+1.25%, range -1.25% to +4.2%).
- Jev scores 88% on logic questions and 75% on puzzles.
- **Caveat, and the main open question:** AGIEval is public, so any model may have seen these questions. 75% on LSAT
  analytical puzzles and AQuA-RAT is very high for a model that only picks from a list. See "Next" below.

## Fair-test source: INVALID

The fair test's saved live run (`rows_live.jsonl` in the Mac's fair-test folder) holds only empty answers for every
ant, with 0 tokens; its own `result_live.json` shows 0.0 accuracy and Brier 0.25. The export went through ("reasons
found 0, missing 0"), the scorer reported "ants usable 0%", and the verdict is **invalid run**. No ant comparison
from it means anything. The primary verdict of the preregistration is therefore not yet tested.

Only the decision models' own answers there are real:

- Exams: Jev alone 81.7%, d1-3B alone 47.1%.
- Forecasts (Brier, lower is better): Jev 0.277, d1-3B 0.277. The market crowd gets 0.216; a coin flip is 0.25.
  **Neither decision model beats a coin flip on the forecasts.**

(The export should have refused a run whose ants are all empty. The add-on stays frozen as it ran; the fresh test's
script refuses a swarm with under 90% usable ant answers before any decision-model call.)

## Skipped

- d1-2B: `optiq serve` is a plain OpenAI-style server. It answers `POST /v1/systemone` with HTTP 404, so the smoke
  test failed. Not run.
- Perplexity Decider Flash: its model card needs a CUDA GPU and a private Hugging Face repo with a token. There is no
  Apple Silicon runtime. Not run.

## Cost, time, memory

- Jev: $0.0223 in all, under the 5-cent cap. Its shared-read run cost $0.0124 against a $0.0071 estimate, so the
  estimate runs low by about 1.75 times.
- Jev: 23 s (shared-read) + 22 s (fair-test), 479 + 439 calls, none failed.
- d1-3B: 84 s + 64 s, 479 + 439 calls, none failed. Server start with the model download took about 3 minutes.
- Memory pressure normal throughout; lowest free memory 49% (d1-3B loaded); swap flat at 2008.81 MB. One model
  server at a time, each stopped afterwards.

## What it means

- On public exam questions, Jev alone is the best answerer we have tried, by a wide margin, and cheaply. Adding it
  as one voter among many weak ants wastes it; letting it read the ants does not beat it alone.
- Whether that 81.7% is reasoning or memory is unknown. That decides whether decision models belong at the centre of
  the swarm.
- On real forecasts, neither decision model has any skill yet.

## Next

The fresh-question test (`docs/plans/2026-10-31-fresh-questions.md`): the same arms on puzzles made by code on the
Mac at run time, with new names and numbers each run, so no model can have seen them; plus a letter-moving probe on
the same 240 AGIEval questions.
