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

## Phase 1 — Python Simulation Core [CURRENT]
Goal: Get the existing Python simulation running with tests that pass in CI.
The simulation code already exists in `backend/` — fix and extend it, don't rewrite from scratch.

- [x] Fix imports: replace `openai`/`google.generativeai`/`mistralai` with LiteLLM client (use `requests` to call `http://host.orb.internal:4000/v1/chat/completions`)
- [x] Make `backend/biofvm.py` independently testable: add unit tests in `tests/test_biofvm.py` for substrate diffusion, decay, mass conservation (21 tests, all passing)
- [x] Make `backend/tumor_environment.py` independently testable: add unit tests in `tests/test_tumor_env.py` for voxel grid initialization and oxygen gradients (22 tests, all passing)
- [x] Make `backend/nanobot_simulation.py` independently testable: add unit tests in `tests/test_nanobot.py` for nanobot movement, chemotaxis, drug delivery (17 tests, all passing)
- [x] Add `pytest.ini` or `pyproject.toml` with test configuration and CI integration
- [x] Update GitHub Actions to run both `npx hardhat test` AND `pytest` in CI

## Phase 2 — Pheromone System Enhancement
Goal: Implement and validate the pheromone signaling system for decentralized coordination.

- [x] Implement trail pheromone field in `backend/biofvm.py` with secretion, diffusion, and exponential decay (D=1e-6, t½≈10min)
- [x] Implement alarm pheromone field with higher diffusion rate and faster decay (D=5e-6, t½≈3min)
- [x] Implement recruitment pheromone for zone exploration signaling (D=2e-6, t½≈7min)
- [x] Add chemotaxis logic to nanobots: follow trail gradient, avoid alarm zones (wired to trail_pheromone, alarm_pheromone, recruitment_pheromone substrates)
- [x] Unit tests: pheromone decay half-life, chemotaxis directionality, mass conservation (7 pheromone tests in test_biofvm.py)
- [x] Baseline comparison: bots without pheromones vs with pheromones (benchmark_pheromones.py — 0% improvement, expected: secretion logic not yet wired)

## Phase 3 — Blockchain Integration
Goal: Connect the Python simulation to the Solidity contracts for provenance.

- [x] IPFS pinning utility: hash simulation artifacts, pin to IPFS, return CID (backend/chain/ipfs.py, 11 tests, supports Pinata/local/dry-run)
- [x] Verification CLI: `python3 -m chain.verify <run_hash>` fetches CID, recomputes metrics, checks tolerance (8 tests)
- [x] Deploy ExperienceRegistry + TumorIntel to Base Sepolia testnet — TumorIntel at `0xd1cfa5b9994e06cc18a21dc18fb9d20a3c02238b`, SP1 Gateway wired
- [x] Submission CLI: `python3 -m chain.submit` creates attestation bundle (IPFS + on-chain data), ready for ZK proof submission (3 tests)
- [x] Leaderboard service: reads on-chain events, ranks policies by attested performance (6 tests, CLI with table + JSON output)

## Phase 4 — LLM Hierarchy (Queen/Worker)
Goal: Add the Queen agent for strategic coordination with measurable improvement over baseline.

- [x] Queen policy wrapper: episodic planner that adjusts bot parameters every K simulation steps (heuristic adaptation)
- [x] Worker agent parameters: exploration bias, trail secretion rate, alarm sensitivity (configurable, applied to workers each episode)
- [ ] Queen uses LiteLLM (qwen3.5-35b or deepseek-chat) for strategic decisions
- [ ] Evaluation harness: compare queen-guided vs fixed-policy across seeds and patients
- [ ] Success gate: queen policy must improve kill rate ≥10% vs fixed baseline (≥5 seeds, ≥3 patients)

## Phase 5 — Experiment Ops & Evaluation
Goal: Automated experiment sweeps with reproducible, on-chain-attested results.

- [ ] Batch runner: YAML config for seed × parameter grid sweeps
- [ ] Metrics collection: tumor-kill %, toxicity proxy, time-to-control, runtime/cost per run
- [ ] Report generator: HTML/Markdown with plots, parameters, seed stats, links to Sepolia tx + IPFS CID
- [ ] Attestation bot: re-runs k% of submissions for reproducibility spot-checks
- [ ] Streamlit dashboard: live leaderboard from on-chain data
