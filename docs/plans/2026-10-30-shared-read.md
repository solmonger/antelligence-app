# The shared-read test: does one bigger reader help a swarm of small ants? (preregistered)

Date: 2026-10-30. Status: **written and frozen before any model call.** Nothing has been measured against a real
model yet. The question and answer files, harness, script, this plan and the pass lines are fingerprinted in
`docs/research/shared-read-20261030/prereg.json`; `score` warns, and marks the result as a new test, if any of them
changed since.
Harness: `antelligence/experiments/shared_read.py`. Script: `scripts/shared_read.py`. Tests:
`tests/smarter_swarm/test_shared_read.py`. Runs free, on local open models on Eshaan's Mac Studio (36 GB).

## Why

The **Swarm Telepathy** paper (SwaT, [arXiv 2610.15095](https://arxiv.org/abs/2610.15095), MIT CSAIL and the
MIT-IBM Computing Research Lab, 10 Oct 2026; code at github.com/Ar-Kareem/SwaT) trains a ~300M translator that
hands one frozen model's working memory (its KV cache) to a model from another family, so the receiver never reads
the document. What we checked in the paper itself:

- Against a written summary handed between the same two models, it scores **+14.9, +19.0 and +18.2 points** on 7
  long-document QA sets (QuALITY, QASPER, HotpotQA, MuSiQue, SQuAD, NarrativeQA, LongHealth; Table 3), about 4x
  faster to first token, and 8x to 45x fewer FLOPs than re-reading (Table 6).
- Against the receiver **reading the full text itself**, it keeps a bit over 90% to a bit over 100% of that score
  (Table 2). So the win is over summaries, not over reading everything.
- "Hire the thinker, rent the talker": Granite 3.3 8B reading for SmolLM3 3B gave 143% of SmolLM3's own score
  (mean over 17 benchmarks). That is one pair; the other two big-to-small pairs gave 104.6% and 103.0%.
- The "3x over captions" image result is one pair (Gemma 3 4B to Llama 3 8B: 39.6 vs 13.3). CMMLU: Llama 3 8B goes
  from 51.2 to 65.1 against Qwen 3 4B's 69.7, which is the claimed three quarters of the gap.
- Limits the paper states: one trained translator per model pair (n(n-1) for n models); one handoff at a time, never
  many receivers or merged messages; the cache "may contain information beyond what is explicit in the text"; it
  does worse on tasks that need fine image detail. Within one family, a plain dense baseline (73.5) slightly beat
  SwaT (72.3) (Table 7), so its edge is across families.
- The repo has training code, no trained translators, no license file (so nothing can be copied into Antelligence),
  and its configs assume two CUDA GPUs.

The paper never gives one reading to **many** receivers. Antelligence's own results say a swarm wins when ants think
alone, and that seeing each other makes their mistakes alike (correlated errors 0.55 to 0.90 on W2;
`docs/plans/2026-10-29-swarm-at-scale.md`). So the question for us comes before any translator: **if every ant gets
the same reader's notes, does the swarm get smarter, or do the ants just start failing together?** This test asks it
in plain text, which is the summary handoff the paper beats. If text notes already help without making ants alike,
the cache version is worth paying for. If text notes make ants alike, a richer shared cache is likely to do so more.

## Set-up (fixed now)

- **Questions:** the fair test's frozen 240 AGIEval multiple-choice questions (LSAT analytical, LSAT logical,
  LogiQA, AQuA maths; 60 each), files and hashes in `docs/research/fair-test-20261029/`. Answers are opened only by
  `score`.
- **Reader:** Qwen3 14B (`qwen3:14b`, ~9 GB at 4-bit), answering without hidden reasoning (`/no_think`), temperature
  0.7, fixed seed per call. Per question it (a) writes notes of at most 200 words: what is asked, the key facts, rules
  or numbers, and traps, and is told never to say or hint by letter which option is right; and (b) answers the
  question itself once (the "thinker alone" line).
- **Ants:** 4 small models from 4 families, each with the fair test's first two exam lenses ("work it out directly",
  "rule out what you can"), so 8 ants: `qwen3:4b` (no hidden reasoning), `llama3.2:3b`, `gemma3:4b`,
  `phi4-mini:3.8b`. Temperature 0.7, fixed seed per call, the fair test's JSON answer format.
- **Arms** (every ant, every question; ants never see each other):
  - `alone`: the full question. This is the fair test's own lens prompt, unchanged.
  - `notes`: the reader's notes in place of the passage, then the question and options (the receiver never reads
    the document, as in the paper). AQuA items have no passage, so there `notes` and `both` differ only in order.
  - `both`: the full question plus the reader's notes.
- **Swarm answer:** plain vote of the 8 ants (ties to the first tied letter given), as in the fair test.
- **Order:** all reader calls first, then each ant model's calls in one block, so one or two models are in memory
  at a time (peak about 14 GB).
- **Calls:** 240 x (2 reader + 8 ants x 3 arms) = 6,240. Free.

## What is measured

- Vote accuracy per arm, and the reader answering alone.
- Average single-ant accuracy per arm, and per family: the "rent the talker" lift (score with notes / score alone).
- **How alike the mistakes are:** mean over ant pairs of the correlation of their wrong-answer records across
  questions (0 = they fail on unrelated questions, 1 = they always fail together). Also the share of both-wrong ant
  pairs that picked the same wrong letter (about 0.25 if they guessed blindly among the 4 wrong ones).
- Leaks: notes that name an option by letter despite the instruction are flagged by a fixed pattern. Results are
  reported with and without them.

## Pass lines (fixed before any call)

95% intervals are paired bootstraps over questions (10,000 draws for accuracy, 2,000 for the alike-mistakes rise).

1. **The shared read helps** if the `both` vote beats the `alone` vote by at least **2 points** with the interval
   above 0. It **hurts** if the interval is below 0.
2. **The ants think alike** if the alike-mistakes measure rises from `alone` to `both` by at least **0.05** with
   the interval above 0.
3. **The swarm beats the reader alone** if the `both` vote beats the reader's own answer by at least **2 points**
   with the interval above 0.
4. **No verdict** if under 90% of answers are usable. If over 10% of notes leak an option, the verdict carries a
   warning and the no-leak block is the fair read.

## What each outcome means for Antelligence

- **Helps, not alike, beats the reader:** the "one reader, many cheap ants" shape works in text. Next: rent one
  CUDA GPU for under a day and train a real SwaT translator (the repo's default pair, Granite 3.3 8B to SmolLM3 3B,
  needs under 50M tokens of English text), then rerun with the cache in place of the notes. That spends money, so it
  waits on Eshaan's yes.
- **Helps but alike:** the reader is doing the thinking and the ants are echoing it. Keep the reader as one voice
  among the ants rather than feeding every ant, and test giving notes to only some ants.
- **Doesn't beat the reader alone:** the swarm adds nothing over just asking the bigger model. Better to spend
  effort on the reader than on ants.
- **No change or hurts:** don't build the cache version for the swarm; keep think-alone.

## Limits, said up front

- This passes **text notes**, not a cache. It tests the swarm question (does a shared reading herd the ants), not
  SwaT's translator.
- These are short exam questions (about 460 characters of passage and question on average), not the long documents
  where the paper's gains are; the reader helps here by explaining rules, not by saving reading time.
- One reader, four ant models, one run. A different mix could differ.

## How to run (on the Mac)

```
ollama pull qwen3:14b && ollama pull qwen3:4b && ollama pull llama3.2:3b && ollama pull gemma3:4b && ollama pull phi4-mini:3.8b
uv run python scripts/shared_read.py check
uv run python scripts/shared_read.py bench
uv run python scripts/shared_read.py run --limit 60
uv run python scripts/shared_read.py run
uv run python scripts/shared_read.py score
```

`bench` puts 2 questions through everything, shows the reader's first notes, and prints seconds per call on this
machine with a time estimate for the full run. `run` resumes where it stopped and never asks a finished call again;
it stops by itself, saying why, after 8 unusable replies in a row or when a model isn't pulled. The verdict is only
taken from the full 240; `--limit 60` is a quick look. Output goes to `docs/research/shared-read-20261030/run/`.
