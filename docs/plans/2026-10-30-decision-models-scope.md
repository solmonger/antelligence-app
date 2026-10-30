# Decision models plus LLMs: scope and first test

Date: 2026-10-30. Status: **plan only. Nothing built, nothing spent.**

Eshaan's idea: Antelligence should prove that a mix of decision models (models that only pick from a fixed list and
give a probability for each option, such as Jev) and LLMs, in a well-built harness, gives the best accuracy. This
doc checks the two outside pieces he named, adds what our own past runs already show, and turns the idea into a
claim we can prove or disprove cheaply.

## 1. What the outside pieces really are

### Switch (github.com/osolmaz/switch)

- **What it is.** A small TypeScript framework, MIT licence, npm `@osolmaz/switch` 0.1.1, written between 25 and 30
  October 2026 by Onur Solmaz. Code turns the state of a task into typed questions (yes/no, pick one, score), the
  decision model answers with probabilities, and code moves to the next state. It loops until the code asks nothing.
- **Models.** Any classifier that the `pi-ai` library lists: Jev (TypeSafe's hosted API), open decision models served
  by llama.cpp at `/v1/systemone`, and ordinary chat models used as classifiers through their next-token
  probabilities.
- **Examples.** A rubric judge for evals, chess, and WikiRace.
- **Its own result** (its `docs/2026-10-25-first-legal-judge-run.md`, author's numbers, private data, not held out):
  Jev as a rubric judge agreed with the reference on 94.6% to 95.0% of 1,288 items, level with DeepSeek V4.1 Flash
  (94.6%) and above GPT-6 Luna (93.5%), for about $0.13 in total. The README itself says Jev is weak at chess and not
  yet good enough to replace the strongest chat models as a judge.
- **What we can reuse.** The licence allows copying, but we do not need its code: Antelligence is Python and already
  has the same plumbing (below). What is worth taking is the shape: the task's code writes the menu, a "stop" option
  is always on the menu, every step is saved so a run can be replayed without the model, and the model's view is cut
  to a token budget.

### Harness² ("harness2 by Google")

- **It exists.** github.com/google-research/harness2, Apache 2.0, paper arXiv 2610.14224 (9 October 2026). The README
  says it is not an officially supported Google product.
- **What it does.** It keeps one model frozen and rewrites the harness around it for each task: system prompt,
  guardrails, memory, tool guidance, subagents, skills, scripts and plugins. It proposes edits, reruns the same task
  with each one, and combines what worked by comparing the runs, with no answer key. Lessons carry over to later tasks.
  Model weights never change.
- **Reported gains.** With GLM-5 and Gemini 3.5 Flash, up to 11.5 points on WorkBuddy-Bench, 4.5 on JobBench and 11.8
  on LAB, more with a bigger rerun budget. These are long professional tasks run through OpenCode or Codex.
- **What it does not claim.** Nothing about hundreds to millions of agents, swarms, or beating every benchmark. Its
  "parallel recursion" means trying several candidate harnesses on one task, not many agents.
- **Fit for us.** Each task needs several full reruns plus improver sessions, too costly for a $10 a month budget
  today. One idea is worth borrowing later: improving the ants' "ways of looking" prompts automatically, by comparing
  runs, with no checker.

## 2. What our own runs already show

Antelligence has mixed a decision model with an LLM before. The v4 pilot (11 October 2026,
`docs/research/v4-pilot-20261011/FINDINGS.md`) ran Jev ants and DeepSeek ants in the colouring and foraging worlds:

- **Plain Jev ants beat all-DeepSeek ants** in colouring: score 1.223 against 1.026, at about half the time per
  decision (1.48 s against 2.89 s). That is the strongest hint so far that decision models belong in the swarm.
- **"Ask the LLM when Jev is unsure" failed.** It fired on 98.9% of decisions, because Jev was far less sure on live
  states than on the practice states the threshold was set on. So the mix needs a better rule, set on live data.
- **A queen steering the ants made Jev worse,** which matches the later finding that checkers and teachers hurt.
- **Laya (0.4B) was not usable:** 0.41 right against 0.70 for always picking the most common answer.

The code is already there: `antelligence/kernel/decision.py` follows the same System One request format as Switch,
and `antelligence/providers/decision_backends.py` has Jev, Laya, a chat model used as a classifier (log
probabilities), caching, a spend ledger and escalation.

## 3. The claim

Not "best on every task". This one:

> On questions with a fixed set of answers, a swarm that mixes decision-model ants and LLM ants, each thinking alone
> and combined by a fixed rule, gives better answers per dollar than the same LLM asked the same number of times.

It is measured on two axes, always reported together:

1. **Quality:** share right on exams; Brier score on forecasts (lower is better, 0.25 is a coin flip).
2. **Cost per answer:** dollars and seconds, from the ledger.

A mix **wins** when it is better on one axis and no worse on the other, on questions it was not tuned on, with a 95%
interval that leaves out zero. Exact pass lines get frozen in a preregistration before any call, as the fair test
did.

## 4. Where to test it (questions we already have)

1. **The 101 real forecasts** (`docs/research/fair-test-20261029/forecast_*.jsonl`). Yes/no with a probability is
   exactly what decision models are built for, and every question resolved after the LLM was released. The market's
   own crowd probability is the yardstick to aim at.
2. **The 240 exam questions, split in two.** Reading and logic (LSAT logical reasoning, LogiQA: 120) and puzzles and
   maths (LSAT analytical reasoning, AQuA-RAT: 120). A model that reads once and picks should hold up on the first
   half and fail alone on the second, where it needs the LLM ants' reasons. The split shows which part earns its place
   where.
3. **Later: the colouring world.** Many small picks in a row, the Switch shape, and the one place Jev already won.

## 5. Who competes

Same LLM as the fair test (Step 3.7 Flash, free route) and the same questions.

| Arm | What it is |
|---|---|
| `llm1` | the LLM asked once |
| `single5` | the LLM asked 5 times, majority or mean (the fair test's baseline) |
| `swarm5` | PR 21's 5 ants thinking alone, each looking a different way |
| `dm1` | the decision model alone, reading the question once |
| `mix-vote` | `swarm5` plus the decision model as a 6th ant, same combine rule |
| `mix-read` | the decision model reads the question and the 5 ants' answers and reasons, then picks |
| `mix-cheap` | the decision model first; the LLM ants are asked only when it is unsure |

None of these lets anything talk back to the ants, so nothing steers the swarm. `mix-read` is the closest to a
checker, since one model weighs the others' answers; it is a combine rule, not a teacher, but if that crosses the "no
checker" line it gets dropped. The `mix-cheap` threshold is set on one half of the questions (fixed seed) and scored
on the other half, which is the fix for the v4 escalation failure.

## 6. The first test: a decision model on saved answers

Once the fair test finishes on the Mac, its cache holds every ant's answer and reason for all 341 questions. Every new
arm above except `llm1` then needs **decision-model calls only, and no new LLM calls**.

- **Model, one of:**
  - **Jev** through the existing `JevBackend` and Eshaan's TypeSafe key (used in v4). Questions average about 220
    tokens (exams) and 90 (forecasts); with the 5 ants' reasons, about 650. All arms together come to well under a
    million input tokens, at $0.042 per million: **under 5 cents**.
  - **Kev-4B or lev** (4B, Apache 2.0, a 3 GB file), free and local through llama.cpp's `/v1/systemone`, once the Mac
    is free of the shared-read test. No published accuracy numbers exist for these yet, so this also tells us whether
    they are any good.
- **Freeze first.** A separate preregistration (the fair test stays frozen as it is), with the pass lines, the seed
  that splits the questions, and the prompt text.
- **What it shows.** Whether a decision model adds anything on top of the LLM ants, on which kinds of question, and
  at what cost. If it adds nothing on the forecasts, its best case, the scope is wrong, and we learn that for pennies.
  If it helps, the next step is the colouring world and then training a small open decision model on the swarm's own
  saved answers, which is where this meets the plan to fine-tune small models.

## 7. What this needs

- The fair test to finish on the Mac (in progress).
- A choice of decision model for the first test: Jev (under 5 cents, his key) or local Kev-4B (free, after the
  shared-read test).
