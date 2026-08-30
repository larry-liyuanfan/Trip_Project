# Repository Agent Guidelines

## Scope and authority

This file applies to the whole repository. Direct user instructions take precedence. A nested
`AGENTS.md` may add rules only for its subtree.

The repository is in handoff mode. Do not recreate weekly plans, weekly reports, chat prompts,
meeting transcripts, approval workflows, or future-week roadmaps. Historical process documents
remain available in Git history; the current tree should stay focused on runnable code and the
final release.

Read these files before changing the project:

1. `README.md`
2. `configs/releases/qwen3_vl_system_final_v1.json`
3. `docs/model_handoff.md`
4. `reports/project_summary.md`
5. `reports/final_delivery_status.md`

Use machine-readable records under `experiments/` only when reproducing an existing experiment.
They are evidence, not current requirements.

## Current release

The formal release is `trip-qwen3-vl-8b-week8-final-v1`. Keep its release configuration,
four-layer handoff package, prompts, schemas, and recorded hashes immutable. Create a new
versioned release for any behavioral change; never overwrite historical results or model assets.

The current product evidence is model-generated silver, not human visual ground truth. Do not
claim human visual accuracy, supported price-range accuracy, strict dialogue research-gate
success, or independent business retrieval relevance unless new evidence explicitly supports it.

## Project structure

- `src/api/`: FastAPI routes and typed boundary schemas.
- `src/inference/`: release loading, prompts, model runtime, validation, and retry logic.
- `src/retrieval/`: CLIP, Milvus, keyword, and hybrid retrieval.
- `src/planning/`: itinerary planning helpers.
- `src/data/`: data ingestion, validation, and transformations.
- `src/evaluation/`: metrics and error analysis.
- `src/training/`: reproducible training and evaluation utilities.
- `configs/`: release, model, prompt, schema, and experiment configuration.
- `scripts/`: command-line entry points and operational tools.
- `tests/`: `unittest` coverage.
- `docs/`: current technical references and handoff instructions.
- `reports/`: final status and project summary only.

Keep route handlers thin and business logic in the matching `src/` package. Use Pydantic models at
API boundaries. Prefer configuration and structured parsers over hard-coded values or ad hoc text
processing. Do not introduce machine-specific absolute paths.

## Development and verification

Use Python 3, 4-space indentation, descriptive `snake_case`, UTF-8, and concise comments only for
non-obvious logic. Tests use `unittest`; add focused failure-path coverage for behavior changes.

Run from the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/tripctl.py validate
python scripts/verify_final_delivery.py outputs/releases/trip-qwen3-vl-8b-week8-final-v1
docker compose -f docker/system/docker-compose.yml --env-file docker/system/.env.example config --quiet
git diff --check
```

Never claim a check passed unless it was run in the current worktree. Report unavailable live GPU,
model, or Milvus checks accurately.

## Data and security

Do not commit secrets, `.env` files, model weights, adapters, model caches, raw Yelp data, generated
datasets, vector databases, or run outputs. Preserve source counts, rejection reasons, hashes, and
data identities when processing datasets. Never fabricate labels, metrics, model outputs, or human
review decisions.

The Git-external final package is stored at
`outputs/releases/trip-qwen3-vl-8b-week8-final-v1`. It is the only model handoff directory that
should remain locally. Verify its manifest before use.

## Git branches

- `dev`: active development and integration.
- `stg`: verified stable candidate; promote from `dev` only after tests and package checks pass.
- `main`: final submission branch; promote from `stg` only for an approved delivery.

Develop on `dev` or a short-lived `feature/*` branch. Do not commit directly to `main`. Keep commits
coherent, do not stage unrelated changes, and remove merged temporary branches and worktrees.
