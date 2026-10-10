# Antelligence 48-hour sprint report (2026-10-08 → 2026-10-10)

Hermes started the sprint. Claude (Opus 5.5) resumed it on 2026-10-09 and finished it. Every item below was checked against a live artifact: a file, a PR, an on-chain transaction or command output. Branches are `hermes/*` on the fork `solmonger/antelligence-app`. All PRs target the fork's `main`. None are merged, and nothing was pushed to upstream.

## Done

| Item | Result | Evidence |
|---|---|---|
| Fork sync | Upstream PR #3 merged into a sync branch; waiting on the operator to merge | PR #15 (`hermes/sync-20261008`) |
| W0 truth pass | Test suite: 905 passed, 2 skipped (data-gated), 0 failed. SPRINT.md reconciled. | `docs/status/2026-10-08-truth-pass.md`, PR #16 |
| W1 chain layer | Moved to ZKsync Era Sepolia (chain 300) and all 4 contracts deployed; `eth_getCode` is non-empty for each: FoodToken `0xA2B8…4330`, ColonyMemory `0x5e20…9363`, TumorIntel `0x9044…e317`, ExperienceRegistry `0x2d77…9c3b`. End-to-end run at trust tier `proof_staged` (mock proof): [tx 0xf0cf…150f](https://sepolia.explorer.zksync.io/tx/0xf0cf992598bd4324808270ebb426f44a0beac7fa874ade35563ee71d3b94150f), status 1, replay passed. Privacy audit and chain-options note written. | `blockchain/deployments/zksync-era-sepolia.json`, `docs/status/2026-10-08-chain-redeploy.md`, `docs/research/chain-options-20261008.md`, PR #18 |
| W2 benchmark | Preregistered before any eval call. 2,831 cells: local Qwen3.8-27B (Q3_K) vs Claude Haiku 5.5 and Sonnet 5.5. Includes `results.csv`, a REPORT.md with Wilson lower bounds and paired tests, a token-cap sensitivity run and an exploratory lenient re-score. | `docs/research/slm-vs-frontier-20261008/REPORT.md`, PR #17 (head b02543e) |

## What W2 found (plain words)

- **Teamwork did not make the small model better.** Qwen alone got 78.9% on medical and finance questions. Its best team setup, three agents voting, got 76.7%. Sharing evidence between agents scored worse, at 50.6%, a significant drop.
- **Small team vs big model alone, under the preregistered strict scoring:** no reliable difference. Sonnet scored only 68.3%, but mostly because it writes its working before the answer and the strict scorer throws those answers out.
- **Same answers, re-read leniently (exploratory):** Sonnet alone reaches 91.7% and clearly beats the small-model team at 75.0%.
- **Planning tasks:** after merging, a team of planners solved 16 of 19 tasks with every model. A single planner solved 2 (Qwen) to 10 (Sonnet). This is the one setting where coordination clearly helped.
- **Foraging gave no usable signal,** because replies were cut off at the 256-token limit.

## What failed and why

- **Crash.** At 14:06 on 10-09, a Hermes client disconnect killed the local model server and queue. 42 cells were lost. They were quarantined, re-run cleanly, and recorded as deviation D1. Long jobs now run detached with `setsid`.
- **Scoring artifacts, not model ability.**
  - One formatting slip voids a whole multi-call answer.
  - The "isolated evidence" control fails because of its majority-vote rule.
  - A hidden 3-citation limit is enforced but never shown to the agents.
  - All of these are written up in the limitations section of REPORT.md.
- **Token-cap check stopped early.** The operator stopped the local Qwen part of the token-cap check at 452 of 540 cells, so that comparison uses seeds 0–1 only.
- **Two security slips this session.**
  - A GitHub token embedded in the git remote URL was printed. It has been revoked (401) and removed from the config, and the operator re-logged in.
  - A 40-hex fragment of the old deployer key was printed. That key is retired and logged in the vault.

## Blocked

Nothing is blocked. The W1 gas blocker was resolved once a funded key was supplied (`docs/status/BLOCKERS.md`). The remaining items are review and merge decisions for the operator: PRs #15–#18 and this report's PR.

## Usage

- **Testnet gas:** 0.000217 ETH on ZKsync Era Sepolia, covering 4 deploys and the end-to-end run, from deployer `0xF822…96cE`. The operator also moved 0.01 testnet ETH to the retired deployer; that transfer was not sprint work. No mainnet use.
- **Frontier models:** all billed to **Nous credits**, not the Claude subscription. 9,398 calls cost $10.14 against a $12 cap:
  - Preregistered run: $7.55.
  - Token-cap sensitivity check: $2.59.
  - The Claude subscription was used for a single smoke-test call.
- **Local model:** 7,075 calls to Qwen on this Mac, $0 in API cost.

## Next three steps

1. **Merge PR #15 first**, then review #16, #17 and #18. Until #15 lands, their diffs show 140 extra sync commits.
2. **Fix the scoring harness, then re-run.**
   - Lenient JSON parsing, with the format failure rate reported separately.
   - State the citation limit in the prompt.
   - A plurality-vote control.
   - A larger token budget for foraging.

   Then re-run the swarm comparison under a new preregistration. The current strict result mostly measures format compliance.
3. **Replace the mock proof with a real one** (trust tier `verified_onchain`), and keep on-chain data limited to hashes, commitments and run IDs, as the privacy audit recommends.
