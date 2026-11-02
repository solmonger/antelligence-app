"""The harness: one front door that decides, per request, which models and tools to call (v0, 2026-11-02).

Eshaan's goal, in his words: "build a harness like hermes or harness2 which can cater swarm of slms, llms, decision
models, etc. and wire things in and call the things needed by itself while improving overtime and beating all scores
in terms of token consumption, accuracy, latency". The harness is the product. The swarm is one tool in it.

Four parts, each small and plain:

* **Toolbox** (:mod:`.tools`): every worker behind one call, ``run(tool_id, task, ctx) -> Step``. A step says what
  the worker answered and what it cost: tokens in and out, seconds, dollars. Workers today: the reader (one bigger
  local LLM), the reader's notes, each small ant model (alone, from the notes, or both), a code solver that writes a
  short program to check every option against the puzzle's rules and is sent back to fix it when the check fails,
  and decision models (Jev, Liquid d1) when they are switched on.
* **Recipes** (:mod:`.recipes`): small programs over the toolbox, such as "ask the reader once", "ask cheap ants one
  by one and stop when they agree, else ask the reader" (a FrugalGPT-style cascade), or "write a checking program
  first". The same recipe code runs live and on saved steps, so a recipe can be scored on logged work for free.
* **Scorecard** (:mod:`.scorecard`): every request's answer, whether it was right, and its total tokens, seconds and
  dollars, compared with "always ask one model" on all three at once.
* **Learner and planner** (:mod:`.learner`, :mod:`.planner`): the learner reads the scorecard and learns which
  recipe wins for which kind of request (a lookup over a few plain features of the text, shrunk toward the overall
  winner so a handful of questions can't fool it). The planner is that lookup, so choosing costs nothing.

Ideas borrowed from proven work: FrugalGPT (Chen, Zaharia and Zou, 2023): cascades scored on cached replies; RouteLLM
(Ong et al., 2024): a learned router judged against "always the strong model"; early-stopping self-consistency
(Aggarwal et al., 2023; Li et al., 2024): stop sampling once the vote can't change; program-aided answers (PAL, Gao et
al., 2022; Logic-LM, Pan et al., 2023): let a program check the rules.

The test-set rule (Eshaan, 2026-11-02): the final test set is chosen last, modelled on how big labs evaluate, and
kept unseen so nobody tunes to it. Until then the harness is only ever developed on code-made puzzles
(:mod:`antelligence.experiments.fresh_questions`) and on the night run's saved answers. **Numbers on those prove
nothing about real questions.**
"""
