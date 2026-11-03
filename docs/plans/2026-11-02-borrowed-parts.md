# Borrowed parts as switches (2026-11-02)

Eshaan, after the DeepSeek harness check: "we can eval each of them as features and see what works best maybe?"

This adds the five parts that check ranked as worth borrowing. Each is a switch on the harness (draft PR 27), so each
can be scored alone, and every mix of them can be scored too, on the same saved work.

## The five switches

| switch | borrowed from | what it does here | what it needs |
|---|---|---|---|
| **stop** | Adaptive-Consistency (Aggarwal et al., 2023) | Ask ants one at a time and stop once a small odds check says more answers are unlikely to change the vote. Today's rule is "stop when one answer is k votes ahead". | the saved answers |
| **gate** | DeepConf (Fu et al., 2025), AutoMix, FrugalGPT | Read how sure a small ant was of its letter (its chance for that letter at the answer). If it is sure, stop there; if not, ask more ants or the reader. A second form weighs each ant's vote by how sure it was. | a new run of the ants with answer odds turned on |
| **router** | RouteLLM (Ong et al., 2024), its similarity-weighted router | Pick the recipe for a question from the most similar past questions (by their words), not from a coarse kind of question. | the saved answers |
| **cache** | DeepSeek Harness's cost story | Count only the prompt tokens a model really had to read. The same question to the same model with another way of looking starts the same, and a server with a prefix cache reads that start once. The learner then saves on that count, so it can learn that a second look from the same model is cheap. | the saved answers (an estimate); the new run (the server's own count, to check the estimate) |
| **prompt** | GEPA, as Hermes Agent's self-evolution uses it | A small loop asks the reader to rewrite the ants' fixed instructions shorter, scores each rewrite, and keeps the shortest one whose accuracy holds. | the local models, a few hours |

Code: `antelligence/harness/borrowed.py` (the first four and the scoring), `antelligence/harness/promptopt.py` (the
prompt search), `antelligence/providers/logprobs.py` (answer odds and cache counts from a server's reply),
`OddsRunner` in `antelligence/harness/tools.py` (keeps those next to the call cache), `scripts/harness_features.py`
(the commands), `tests/harness/test_borrowed.py`.

## How each is scored (the same way the harness is scored today)

1. Every recipe (today's menu, plus the stop rule's and the gate's recipes when those switches are on) is replayed on
   the saved work of all 480 night-run questions. Each gets two bills: every prompt token, and only the fresh ones.
2. For each mix of switches, the cost weight is tuned and the routes are learned on **set 1 and set 2 only**, with 5
   folds inside them, by the same rule as today's tuned learner.
3. The best mix is picked on sets 1 and 2 only.
4. Set 3 is scored once, for every mix, so the table shows each switch alone, every mix, and today's harness, all
   against the reader: accuracy (with a range), tokens, fresh tokens and time.

The hidden final test set is not touched.

## What runs where

- **Free, no model calls**: stop, router, cache on the saved answers (`free` stage).
- **On the Mac, local models only, no cost**: a quick probe of the server, a new run of the 8 ants alone on all 480
  saved questions with answer odds on (`confsweep`), the free scoring again with the gate added (`gate`), and the
  prompt search (`promptopt`). One command runs them all and writes one block to paste back:
  `docs/research/harness-20261102/FEATURES_PROMPT.md`.

## Things to know

- **Answer odds need a newer mlx_lm.server.** Version 0.32 sends each token's top alternatives with their text.
  Older versions send token ids only, so the letter's chance can't be read. The probe says which it is. If the first
  server has none, the run can try a second one (`--odds-server`), and otherwise it skips the gate and says so.
- **The cache count is an estimate on the saved answers.** It counts the shared start of prompts the same model was
  asked on the same question, and nothing across questions, so it leans low. The new run also logs the server's own
  count (`cached_tokens`), and the result compares the two.
- **Fresh tokens change the token bill, not the time.** A cached start is also quicker to read, but the time column
  still shows the measured time.
- **The new ant answers replace the old ones in the gate stage**, so the gate table's "harness (today)" row is today's
  harness on those new answers. Compare within a table.
- **The prompt search is small**: 30 training questions, one ant per model, 3 rounds of 2 rewrites. A rewrite must
  keep the JSON reply with one letter and may not name a letter. Its pick is checked once on set 3 with all 8 ants,
  then inside the harness with the routes already learned.
- `shared_read.py` is fingerprinted by the shared-read test's frozen plan, so it is not changed: the new odds and
  cache counts are kept in a side file next to each call cache (`cache.jsonl.extras.jsonl`).

Dev numbers only: these sets are saved dev answers, not the final test.
