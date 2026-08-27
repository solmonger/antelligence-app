# Antelligence — ARCHITECTURE.md

> Tech stack, folder structure, and conventions. Read every loop. Know the project before you start.

## Repository Layout

```
antelligence-app/
├── backend/
│   ├── chain/           # Blockchain integration
│   │   ├── config.py    # Chain metadata, RPC endpoints
│   │   ├── deploy.py    # Contract deployment (DO NOT run unattended)
│   │   ├── leaderboard.py  # Leaderboard aggregation
│   │   ├── proof_adapter.py  # Mock/staged/real proof interface
│   │   ├── proof_lifecycle.py  # Proof state machine
│   │   ├── proof_spec.py  # Proof bundle schema
│   │   ├── submit.py   # On-chain submission
│   │   └── verify.py   # Verification logic
│   ├── api_server.py    # FastAPI server (port 8001)
│   ├── cli.py           # CLI entry points
│   ├── config.py        # App-level configuration
│   ├── nanobot.py       # Nanobot agent logic
│   ├── run_store.py     # Simulation run persistence (SQLite)
│   ├── simulation_replay.py  # Deterministic replay
│   └── tumor_simulation.py   # Core simulation engine
├── blockchain/          # Solidity + Hardhat
│   ├── contracts/
│   │   └── TumorIntel.sol  # On-chain public values (5 fields)
│   ├── hardhat.config.js
│   └── test/
├── tests/               # pytest suite (228+ tests)
│   ├── test_api_server.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_leaderboard.py
│   ├── test_nanobot.py
│   ├── test_proof_adapter.py
│   ├── test_submit.py
│   ├── test_verify.py
│   └── ...
├── frontend/            # React/Vite UI
│   └── src/
├── docs/                # Specs, release notes, plans
├── VISION.md            # What success looks like
├── RULES.md             # Agent guardrails
├── README.md            # Quick start
└── pyproject.toml       # Python project config (uv)
```

## Key Technologies

| Layer | Technology | Notes |
|-------|-----------|-------|
| Simulation | Python (numpy) | Deterministic, seedable |
| API | FastAPI | Port 8001, local-only |
| Frontend | React + Vite | Dev server, not production |
| Blockchain | Solidity + Hardhat | Base Sepolia testnet |
| Contracts | TumorIntel.sol | 5 public values on-chain |
| Proof staging | Python proof_spec | mock → staged → verified_onchain |
| Persistence | SQLite | `data/api_runs.sqlite3` |
| Package mgmt | uv | `uv sync --extra test` |

## Chain Configuration

- **Canonical chain**: Base Sepolia (chain ID 84532)
- **Exploratory chains**: Somnia testnet (50311), Somnia mainnet (50312)
- **TumorIntel**: `0x925b455175eF932a9a0239090a94E593224CD8AB` (Base Sepolia)
- **ExperienceRegistry**: `0x58A78E337ce3D948A39475f05Ca1A2c30274CADE` (Base Sepolia)
- **ColonyMemory**: `0x914D72b9d49ED4Bb46FA553a01fEbbd5EEf481fA` (Base Sepolia)
- **FoodToken**: `0x7310fb01b393459d2f8Ab15AD4a66F5380200869` (Base Sepolia)

## On-Chain vs Off-Chain

| Data | Location | Reason |
|------|----------|--------|
| config_hash | On-chain | Public commitment |
| kill_rate_bps | On-chain | Key metric |
| nanobot_count | On-chain | Configuration |
| tumor_radius | On-chain | Configuration |
| steps | On-chain | Configuration |
| Simulation details | Off-chain (proof bundle) | Privacy + size |
| Agent trajectories | Off-chain | Size |
| Patient parameters | Off-chain (never on-chain) | Privacy |
| Proof bytes | Off-chain (staged) → On-chain (verified) | Cost |

## Build & Test Commands

```bash
# Install
uv sync --extra test

# Run tests
uv run --extra test pytest -q

# Run simulation
uv run antelligence simulate --steps 100 --bots 10

# Run API
uv run antelligence-api

# Compile contracts
cd blockchain && npx hardhat compile
```

## Ports

- `8001`: FastAPI server (local only)
- `5173`: Vite dev server (frontend)

## Conventions

- **Python**: 3.11+, type hints encouraged
- **Tests**: pytest, one test file per module
- **Imports**: absolute within `backend/`
- **Config**: environment variables, not hardcoded secrets
- **Proof honesty**: never claim verified when staged
- **TDD**: test first, code second, refactor third
