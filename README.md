# Antelligence

Antelligence is a DeSci swarm-intelligence product for tumor simulation, nanobot coordination, and verifiable research provenance. It models how small agents coordinate through local signals, records run provenance for later verification, and stages a proof pipeline that can evolve from mock artifacts to real cryptographic checks.

## What it does

- Runs a tumor simulation with configurable nanobot count, grid size, steps, seed, and pheromone parameters.
- Exposes a small simulation API for local apps and demos.
- Ships a CLI for single runs, benchmarks, and leaderboard reads.
- Records provenance for Base Sepolia workflows and proof artifacts.
- Distinguishes staged proof bundles from cryptographically accepted verification.

## Current product surface

### CLI

The Python package exposes these entry points:

- `antelligence`
- `antelligence-api`

Common commands:

```bash
uv run antelligence simulate --steps 100 --bots 10 --output out/run.json
uv run antelligence benchmark --runs 5 --steps 50 --bots 5 --output out/benchmark.json
uv run antelligence leaderboard --limit 10
uv run antelligence-api
```

### API

The FastAPI service is defined in `backend/api_server.py` and currently exposes:

- `POST /simulate`
- `GET /runs/{run_id}`
- `GET /health`

Example local flow:

```bash
curl -X POST http://127.0.0.1:8001/simulate \
  -H 'content-type: application/json' \
  -d '{"num_bots": 8, "grid_size": 40, "steps": 60, "seed": 7}'

curl http://127.0.0.1:8001/runs/{run_id}
curl http://127.0.0.1:8001/health
```

Run lookups are durable across process restarts. By default the API stores completed runs in `data/api_runs.sqlite3`, and you can point it at a different local database with `ANTELLIGENCE_RUN_DB=/path/to/runs.sqlite3`.

### Proof and provenance

Antelligence already carries proof-shaped artifacts, but the product is explicit about what is and is not cryptographically real yet.

- `proof_origin=mock` means the proof bytes are staging placeholders.
- `proof_ok=false` must remain false for mock or staged artifacts.
- `trust_tier=proof_staged` means a proof bundle exists, not that it has been cryptographically accepted.
- `verified_onchain` is the only state that should be treated as cryptographically accepted.

The current on-chain transport is intentionally small and Base Sepolia oriented:

- `config_hash`
- `kill_rate_bps`
- `nanobot_count`
- `tumor_radius`
- `steps`

Richer provenance lives in the backend proof artifact alongside that tuple so replay, proof generation, and contract verification can stay pinned to the same run identity.

## Quick start

### 1. Install Python dependencies

Using `uv`:

```bash
uv sync --extra test
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

### 2. Run a local simulation from the CLI

```bash
uv run antelligence simulate --steps 25 --bots 6 --output out/example-run.json
```

This writes a JSON artifact containing:

- `config`
- `metrics`

### 3. Run the API

```bash
uv run antelligence-api
```

The API listens on port `8001` by default when started through the packaged entry point.

### 4. Optional frontend

If you want the frontend dev server:

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

## Base Sepolia scope

Blockchain-facing provenance is currently scoped to Base Sepolia test workflows. The repo includes chain utilities for submission, verification, leaderboard reads, and verifier administration, but the public contract is conservative: staged artifacts are not marketed as finished cryptographic proof.

## Testing

Run the narrow test suite you need while keeping the `uv` cache in a writable directory:

```bash
mkdir -p /private/tmp/uv-cache
UV_CACHE_DIR=/private/tmp/uv-cache uv run --extra test pytest -q
```

## Repository layout

- `backend/` - simulation engine, CLI, API, proof helpers, and chain integration
- `tests/` - Python test suite
- `frontend/` - local UI
- `blockchain/` - smart contracts and Hardhat project
- `docs/` - release notes, plans, and proof specification

## Status

This repo is being tightened toward a public-ready DeSci release. The current emphasis is simulation correctness, provenance clarity, proof lifecycle staging, and consistent local interfaces.
