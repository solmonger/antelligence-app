# The shared-read test: does one bigger reader help a swarm of small ants? (preregistered)

Date: 2026-10-30. Status: **run 1 done** (all 240 questions, 2026-10-30). Verdict: the shared read helps, the ants
think alike, and the swarm does not beat the reader alone. Numbers and what they mean:
`docs/research/shared-read-20261030/RESULTS.md`. How run 1 differed from the set-up below is under "How run 1 ran".
Everything above "How run 1 ran" was written and frozen before any model call. The question and answer files, harness, script, this plan and the pass lines are fingerprinted in
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
- **Models** live in `docs/research/shared-read-20261030/models.json` (part of the freeze), all open weights on
  Ollama, the newest that fit a 36 GB Mac one at a time (checked on ollama.com on 2026-10-30):
  - **Reader:** Qwen3.8 27B (`qwen3.8:27b`, Aug 2026, ~18 GB at 4-bit; the only Qwen3.8 size on Ollama), with its
    hidden reasoning asked off (`reasoning_effort: none`). Fallback if it won't fit or is too slow: Gemma 4 12B
    (`gemma4:12b`, ~8 GB). Per question it (a) writes notes of at most 200 words: what is asked, the key facts,
    rules or numbers, and traps, and is told never to say or hint by letter which option is right; and (b) answers
    the question itself once (the "thinker alone" line).
  - **Ants:** 4 small models from 4 families, each with the fair test's first two exam lenses ("work it out
    directly", "rule out what you can"), so 8 ants: Gemma 4 E2B (`gemma4:e2b`), Qwen3.5 4B (`qwen3.5:4b`; same
    family as the reader on purpose, to see if a same-family ant gains more), IBM Granite 4.2 3B (`granite4.2:3b`)
    and Liquid LFM3 1.7B (`lfm3:1.7b`, a non-transformer hybrid). No 2026 Llama or Phi model this small is on Ollama.
  - All at temperature 0.7, a fixed seed per call, hidden reasoning off where a model has it, the fair test's JSON
    answer format. If `bench` shows a model still thinking, the way it is turned off may be changed in the model
    file before the first real run, then re-frozen; that is the only allowed change.
- **Arms** (every ant, every question; ants never see each other):
  - `alone`: the full question. This is the fair test's own lens prompt, unchanged.
  - `notes`: the reader's notes in place of the passage, then the question and options (the receiver never reads
    the document, as in the paper). AQuA items have no passage, so there `notes` and `both` differ only in order.
  - `both`: the full question plus the reader's notes.
- **Swarm answer:** plain vote of the 8 ants (ties to the first tied letter given), as in the fair test.
- **Order:** all reader calls first, then each ant model's calls in one block. On Ollama each model is unloaded when
  its block ends, and the next is loaded only if there is room for it plus 6 GB, so one model is in memory at a
  time (peak about 21 GB, the reader).
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
ollama pull qwen3.8:27b && ollama pull gemma4:e2b && ollama pull qwen3.5:4b && ollama pull granite4.2:3b && ollama pull lfm3:1.7b
uv run python scripts/shared_read.py check
uv run python scripts/shared_read.py bench
uv run python scripts/shared_read.py run --limit 60
uv run python scripts/shared_read.py run
uv run python scripts/shared_read.py score
```

Or paste `docs/research/shared-read-20261030/CLAUDE_CODE_PROMPT.md` into Claude Code on the Mac and it does all of
this, watching memory as it goes.

`bench` puts 2 questions through everything, shows the reader's first notes, prints seconds and tokens per call on
this machine with a time estimate for the full run, and stops with a warning if a model is still thinking. `run` resumes where it stopped and never asks a finished call again;
it stops by itself, saying why, after 8 unusable replies in a row, when a model isn't pulled, or when memory is
too tight to load the next model. The verdict is only
taken from the full 240; `--limit 60` is a quick look. Output goes to `docs/research/shared-read-20261030/run/`.

## How run 1 ran (2026-10-30)

Claude Code ran it on Eshaan's Mac Studio from `docs/research/shared-read-20261030/CLAUDE_CODE_PROMPT.md`. What
differed from the set-up above, all allowed by it:

- **No Ollama on the Mac, so MLX.** Every model was a 4-bit MLX build from Hugging Face, served by `mlx_lm.server`
  0.32.2. `models.json` now holds those names, so it records what ran; `run1-meta.json` is the run's own meta file.
- **The Ollama-shaped stand-in on port 11434.** `mlx_lm.server` can't unload a model, so the memory guard saw the
  last model still in memory at every model change and stopped the run. The fix was a small stand-in on port 11434,
  where the script expects Ollama. It passes `/v1` calls through to `mlx_lm.server`, and when the script sends its
  unload call (`POST /api/generate` with `keep_alive: 0`), it restarts `mlx_lm.server`, so the memory really comes
  back before the guard checks for the next model. The stand-in lives on the Mac, not in this repo. Any server can
  join the test this way: answer `/v1/chat/completions` and `/v1/models`, and free the memory on the unload call.
- **Thinking off through the chat template**: `"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}`
  for every model, since MLX ignores Ollama's `reasoning_effort`. `bench` showed no model still thinking.
- **Reader: Gemma 4 12B** (`mlx-community/gemma-4-12B-it-4bit`), the preregistered fallback. The guard saw 19.6 to
  22.9 GB free against about 23 GB needed for Qwen3.8 27B.
- **One ant swapped: LFM2.5 1.2B Instruct for LFM3 1.7B.** No released mlx-lm runs LFM3 yet. LFM2.5 2.6B was
  passed over because its thinking can't be turned off.
- `models.json` was changed and frozen again before any scored call, as the Models section allows;
  `run1-prereg.json` is that freeze as rebuilt here, and `score` on the Mac reported the plan unchanged.
- **The 8-in-a-row stop was raised to 100 for the run only.** Gemma 4 E2B often writes LaTeX such as `\cap` inside
  its JSON answer, which the frozen parser rejects even when the letter is plain. Those replies still count as
  unusable. Usable answers: 93%, above the 90% line.
- 25 of the reader's own answers looped until the token cap, and were asked again on every resume (about 20 minutes
  each time) with the same result. They count as wrong for the reader.
- Time: about 4 hours 10 minutes in all. Memory: lowest free 28%, never critical; swap flat at 2 GB.

## After run 1: changes for the next run

None of these change run 1's answers or score. `run1-prereg.json` keeps run 1's freeze; `prereg.json` is frozen
again for the next run.

- **Parser 2.** It also reads a JSON answer with LaTeX in it (backslashes that aren't JSON escapes are read as plain
  backslashes), and as a last resort takes the `"answer"` letter when every `"answer"` key in the reply agrees. It is
  chosen in the model file (`"parser": 2`) or with `--parser 2`; without either, a run uses parser 1, as run 1 did.
  A parser 2 run writes to `run-p2/`, never over `run/`. `run --parser 2 --offline --seed-cache
  docs/research/shared-read-20261030/run/cache.jsonl` re-reads run 1's own replies and asks no model, which shows
  how much the LaTeX losses mattered; `score` labels that as a re-read.
- **Kept unusable replies are used again on a rerun** instead of being asked again (each call has a fixed seed, so
  asking again mostly returns the same reply). `--retry-unusable` asks them again.
- **`--max-fail-streak`** sets the in-a-row stop (default 8), and `meta.json` records it, with the parser.
- **`explore`** reads other swarm shapes off a finished run's answers, with no new call: the reader as one voter
  among the ants, half the ants given the notes (averaged over all 70 ways to pick the half), and each arm without
  the weakest ant family. It is exploratory, since the shapes were picked after seeing run 1, and writes
  `explore.json` next to `score.json`.
