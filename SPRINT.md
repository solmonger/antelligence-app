# Antelligence v2 — Sprint Roadmap

This file guides the dev-sprint-pipeline. The pipeline reads this to determine what to build next.
Work on the FIRST unchecked `[ ]` task in the CURRENT phase. Do NOT skip ahead to later phases.
Do NOT create new Solidity contracts unless the current phase requires it.
Do NOT write more tests for completed contracts — move forward.

## Phase 0 — Blockchain Foundation [COMPLETE]
- [x] ExperienceRegistry.sol — on-chain run registry with validator management
- [x] TumorIntel.sol — tumor intelligence contract with priority updates
- [x] AdvancedTumorIntel.sol — enhanced queries and analytics
- [x] ColonyMemory.sol — shared colony memory contract
- [x] FoodToken.sol — ERC20 resource token
- [x] Hardhat test suites for all contracts
- [x] GitHub Actions CI pipeline

## Phase 1 — Python Simulation Core [COMPLETE]
Goal: Get the existing Python simulation running with tests that pass in CI.
The simulation code already exists in `backend/` — fix and extend it, don't rewrite from scratch.

- [x] Fix imports: replace `openai`/`google.generativeai`/`mistralai` with LiteLLM client (use `requests` to call `http://host.orb.internal:4000/v1/chat/completions`)
- [x] Make `backend/biofvm.py` independently testable: add unit tests in `tests/test_biofvm.py` for substrate diffusion, decay, mass conservation
- [x] Make `backend/tumor_environment.py` independently testable: add unit tests in `tests/test_tumor_env.py` for voxel grid initialization and oxygen gradients
- [x] Make `backend/nanobot_simulation.py` independently testable: add unit tests in `tests/test_nanobot.py` for nanobot movement, chemotaxis, drug delivery
- [x] Add `pytest.ini` or `pyproject.toml` with test configuration and CI integration
- [x] Update GitHub Actions to run both `npx hardhat test` AND `pytest` in CI

## Phase 2 — Pheromone System Enhancement [COMPLETE]
Goal: Implement and validate the pheromone signaling system for decentralized coordination.

- [x] Implement trail pheromone field in `backend/biofvm.py` with secretion, diffusion, and exponential decay
- [x] Implement alarm pheromone field with higher diffusion rate and faster decay
- [x] Implement recruitment pheromone for zone exploration signaling
- [x] Add chemotaxis logic to nanobots: follow trail gradient, avoid alarm zones
- [x] Unit tests: pheromone decay half-life, chemotaxis directionality, mass conservation
- [x] Baseline comparison: bots without pheromones vs with pheromones (automated benchmark script)

## Phase 3 — Blockchain Integration [COMPLETE]
Goal: Connect the Python simulation to the Solidity contracts for provenance.

- [x] IPFS pinning utility: hash simulation artifacts, pin to IPFS, return CID
- [x] Verification CLI: `python3 scripts/antelligence_cli.py verify --run-hash HASH` fetches from chain, checks data integrity
- [x] Deploy ExperienceRegistry + TumorIntel to Base Sepolia testnet (deployment script + verified contract)
- [x] Submission CLI: `python3 scripts/antelligence_cli.py submit --metrics FILE` posts run hash + CID + score
- [x] Leaderboard service: SQLite tracker + on-chain sync, ranks by score with attestation status

## Phase 4 — LLM Hierarchy (Queen/Worker) [CURRENT]
Goal: Add the Queen agent for strategic coordination with measurable improvement over baseline.

- [x] Queen policy wrapper: episodic planner that adjusts bot parameters every K simulation steps
- [x] Worker agent parameters: exploration bias, trail secretion rate, alarm sensitivity (configurable)
- [ ] Queen uses LiteLLM (qwen3.5-35b or deepseek-chat) for strategic decisions (blocked: LiteLLM billing)
- [x] Evaluation harness: compare queen-guided vs fixed-policy across seeds and patients
- [ ] Success gate: queen policy must improve kill rate ≥10% vs fixed baseline (≥5 seeds, ≥3 patients) — FAIL at current scale (direct targeting dominates); needs larger domain or fewer bots

## Phase 5 — Experiment Ops & Evaluation
Goal: Automated experiment sweeps with reproducible, on-chain-attested results.

- [x] Batch runner: YAML config for seed × parameter grid sweeps (`scripts/batch_runner.py`)
- [x] Metrics collection: kill rate, deliveries, drug amount, elapsed time per run
- [x] Report generator: Markdown with tables, parameters, seed stats, summary statistics
- [ ] Attestation bot: re-runs k% of submissions for reproducibility spot-checks
- [ ] Streamlit dashboard: live leaderboard from on-chain data
