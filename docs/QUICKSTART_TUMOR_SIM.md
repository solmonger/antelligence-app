# Quick Start: Antelligence Tumor Simulation

This guide matches the current public product surface: the packaged CLI, the FastAPI service, and the staged proof/provenance model.

## 1. Install dependencies

From the repository root:

```bash
uv sync --extra test
```

If you are not using `uv`, the fallback flow is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

## 2. Run a local simulation from the CLI

Write one simulation artifact:

```bash
uv run antelligence simulate --steps 25 --bots 6 --output out/example-run.json
```

Run a small benchmark:

```bash
uv run antelligence benchmark --runs 3 --steps 25 --bots 6 --output out/benchmark.json
```

Read the on-chain-oriented leaderboard view:

```bash
uv run antelligence leaderboard --limit 10
```

The CLI output stores the current canonical config keys:

- `num_bots`
- `steps`
- `grid_size`
- `seed`

## 3. Start the local API

Use the packaged entry point:

```bash
uv run antelligence-api
```

The API listens on port `8001` by default.

## 4. Check the API surface

Health check:

```bash
curl http://127.0.0.1:8001/health
```

Run a simulation:

```bash
curl -X POST http://127.0.0.1:8001/simulate \
  -H 'content-type: application/json' \
  -d '{
    "num_bots": 8,
    "grid_size": 40,
    "steps": 60,
    "seed": 7
  }'
```

Example response:

```json
{
  "run_id": "replace-with-real-run-id",
  "status": "completed",
  "metrics": {
    "kill_rate": 0.12
  }
}
```

Fetch a stored run:

```bash
curl http://127.0.0.1:8001/runs/{run_id}
```

Run lookups survive process restarts. By default the API stores completed runs in `data/api_runs.sqlite3`, and you can point it at another local database with `ANTELLIGENCE_RUN_DB=/path/to/runs.sqlite3`.

Current public endpoints:

- `POST /simulate`
- `GET /runs/{run_id}`
- `GET /health`

## 5. Understand the proof and provenance state

Antelligence is explicit about proof maturity. Current public-facing guidance:

- `proof_origin=mock` means proof bytes are placeholders.
- `proof_ok=false` means the run has not been cryptographically accepted.
- `trust_tier=proof_staged` means a proof bundle exists, but verification is still staged.
- `verified_onchain` is the only state that should be treated as cryptographically accepted.

The current blockchain-facing provenance flow is scoped to Base Sepolia. Keep public language conservative until the verifier path is fully real, not theater in a lab coat.

## 6. Run narrow verification

Use the narrowest useful checks while you work:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run --extra test pytest tests/test_readme_public_readiness.py tests/test_quickstart_public_readiness.py -q
```

If you are changing backend behavior, extend the test command to the affected test file instead of defaulting to the full suite.

## 7. Optional frontend

If you want the local UI:

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

## Troubleshooting

If `uv run antelligence-api` fails, confirm you installed the Python dependencies from the repo root instead of launching an older ad hoc server entry point.

If `GET /runs/{run_id}` returns `404`, the `run_id` was never stored or you are querying a different local database.

If leaderboard reads fail locally, the CLI should degrade gracefully in offline mode rather than pretending chain data exists.

## More information

- Main overview: `README.md`
- Technical backend details: `docs/TUMOR_SIMULATION_README.md`
- Proof lifecycle notes: `docs/plans/proof-spec-v1.md`
