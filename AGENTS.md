# Repository Agent Guidelines

## Scope and Precedence

This file defines repository-wide instructions for coding agents and applies to the entire repository. A nested `AGENTS.md` may add or override rules only for its own subtree. Direct user instructions take precedence over this file.

Treat this file as version-controlled project configuration. Keep it focused on stable engineering constraints; put week-specific requirements, machine-specific commands, measured results, and delivery evidence in the project documents listed below.

## Sources of Truth

Use the following files according to their responsibility:

- `README.md`: current runnable workflow, dependency groups, and service commands.
- `docs/requirements.md`: scoped weekly goals, deliverables, acceptance criteria, and non-goals.
- `docs/decisions.md`: accepted architectural and workflow decisions.
- `docs/weekly_log.md`: concise progress timeline and verification summary.
- `docs/weekly_delivery.md`: completed delivery checklists and evidence.
- `docs/experiments.md` and `experiments/`: experiment setup, results, and failures.

Do not duplicate detailed procedures or measured statistics in this file. If documents disagree, prefer the most specific accepted decision or current requirement, then reconcile the stale document as part of the task when authorized.

For a new weekly internship requirement, read `README.md`, `docs/requirements.md`, `docs/weekly_log.md`, `docs/decisions.md`, and `docs/experiments.md` before planning. During planning, do not modify code unless the user explicitly requests implementation.

## Mentor Authority and Multi-Chat Workflow

The mentor's latest requirement, as relayed by the user and recorded in `docs/requirements.md`, is the only authority for the current weekly scope. Use this precedence order when instructions conflict:

1. The user's latest direct instruction or mentor clarification.
2. The closest applicable `AGENTS.md`.
3. The current-week requirement in `docs/requirements.md`.
4. Accepted decisions in `docs/decisions.md`.
5. `README.md`, weekly records, and experiment records.
6. Historical agent drafts, which are non-authoritative.

The current Project Control chat owns requirement interpretation, scope decisions, and phase approval. A separate Execution chat implements only the approved current-week phase. A separate Review and Report chat first performs an independent read-only review; it may update delivery documentation only after Project Control approves the review findings. Review must return code defects to the Execution chat instead of refactoring code itself.

For Week 3, implement only the mentor scope recorded in `docs/requirements.md`: three manually annotated OTA evaluation sets, evaluation/training isolation, reproducible baseline metrics and batch testing, four-layer standardized prompts, three JSON Schemas, and a concise evidence-based report. Do not add a second annotator, adjudication workflow, UI, training pipeline, mandatory paired Prompt experiment, bootstrap analysis, or other unrequested infrastructure.

One human annotator is sufficient. Model outputs or deterministic suggestions may assist inspection but must not replace the annotator's own labels. Unknown values remain valid when the image does not support a field and must be reported rather than guessed. Validate readable inputs, required JSON fields, required scenario coverage, and evaluation/training isolation without inventing exact gold-label or source-type quotas. The after-sales set must include both public-scene and business-synthetic samples. This data-labeling rule is separate from the Review and Report chat's independent review of code and delivery evidence.

The Week 3 v1 annotations, manifests, Prompts, Schemas, and runs remain immutable. The user confirmed the mentor's 2026-07-22 authorization for a separately versioned `week3_evaluation_v2`: reuse product labels unchanged, let the existing annotator label the 70 replacement after-sales rows, and supplement only `style_preferences` for the 100 itinerary rows while inheriting their other labels unchanged. That scoped v2 work is complete. Do not create generated gold, require a second reviewer or adjudication, reopen unrelated labels, or start another dataset version without a new direct mentor or user instruction.

Run eligibility and delivery acceptance are distinct. Sampling strata may establish intended candidate coverage but must not be presented as human-gold coverage. Evidence-supported unknown or empty values remain valid and reduce metric support rather than forcing relabeling. Existing traceability, immutable run records, Schema checks, evaluation exclusion checks, mock/dry/live modes, metric aggregation, and error export support the mentor deliverables but are not separate release gates. The approved `baseline_semantic_coding_v1` track supplies gold-independent deterministic lexical metrics for the minimal baseline while preserving its original JSON/Schema results; do not replace it with manual coding, an LLM judge, or gold-derived rules.

Every Execution and Review chat must begin by reading the applicable `AGENTS.md`, `README.md`, `docs/requirements.md`, `docs/decisions.md`, `docs/weekly_log.md`, `docs/weekly_delivery.md`, and `docs/experiments.md`, then inspect the branch, working tree, recent commits, and relevant implementation. If the sources conflict, stop and report the conflict to Project Control rather than choosing a direction.

Agents must not invent future-week plans, roadmaps, product directions, schedules, stretch tasks, or deliverables that the mentor has not requested. They may decompose the current approved requirement into verifiable phases, but may not expand its scope. When a new mentor requirement supersedes an agent draft, discard the draft. Keep chat prompts, agent plans, personal configuration, and meeting transcripts local and out of Git.

## Project Structure

Core Python code lives under `src/`:

- `src/api/`: FastAPI application and route handlers.
- `src/inference/`: VLM clients, schemas, and prompt handling.
- `src/retrieval/`: keyword, embedding, hybrid retrieval, and index building.
- `src/planning/`: preference parsing and itinerary generation.
- `src/data/`: dataset ingestion, validation, alignment, and transformation.
- `src/evaluation/`: metrics and error analysis.

Tests belong in `tests/`, configuration in `configs/`, Docker assets in `docker/`, runnable utilities in `scripts/`, experiment records in `experiments/`, and lightweight samples in `data/samples/`. Yelp raw and generated data stays below `data/yelp/` and must remain ignored.

Keep FastAPI route handlers thin. Put domain behavior in the matching `src/` package, and expose typed Pydantic request and response schemas at API boundaries.

## Environment and Commands

Run commands from the repository root unless a document explicitly says otherwise.

Create the environment and install the default API and data dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Install a narrower dependency group when the task permits:

```bash
pip install -r requirements-api.txt
pip install -r requirements-data.txt
pip install -r requirements-llm.txt
```

Do not install `requirements-llm.txt` in native Windows Python by default. Prefer the documented Docker or WSL2 path for vLLM and CUDA workloads. Use `requirements-clip.txt` only through the dedicated CLIP Docker image; do not merge its GPU dependencies into the API or data environments.

Run the API and standard verification commands with:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
python -m unittest discover -s tests -v
python scripts/test_health.py
python scripts/test_image_understanding.py
docker compose -f docker/docker-compose.yml up --build
```

GPU services and one-off data jobs may have exclusive-memory requirements. Follow the current commands and runtime constraints in `README.md` and the relevant requirement or delivery document instead of copying them into this file.

## Coding Style

Use Python 3 style, 4-space indentation, descriptive `snake_case` names, and clear module boundaries. Use UTF-8 for source, Markdown, YAML, CSV, and JSONL files, especially where Chinese text appears.

Prefer configuration and structured parsers over hard-coded values or ad hoc string processing. Do not introduce machine-specific absolute paths. Data locations, limits, thresholds, output formats, model names, devices, batch sizes, and worker counts must be configurable when they vary by environment or experiment.

Add short comments for non-obvious logic, data assumptions, parsing and filtering rules, alignment quality, resource bounds, and error handling. Avoid comments that merely restate the code.

Prefer concise Simplified Chinese for new business-logic comments and explanatory notes. Keep code identifiers, public API names, protocol fields, library terms, commands, and existing English conventions unchanged when translation would reduce clarity or compatibility.

Keep changes scoped to the request. Preserve existing user changes in a dirty worktree and do not reformat, rewrite, delete, or revert unrelated files. Do not add abstractions or dependencies without a concrete need.

## Testing and Verification

Tests use the standard `unittest` framework. Name files `test_*.py` and methods `test_<behavior>`. Add focused coverage for changed behavior, including failure and fallback paths that do not require a live vLLM server.

Before committing code changes, run:

```bash
python -m unittest discover -s tests -v
```

Also run task-specific smoke tests, data validators, or Docker checks when the changed surface requires them. Never claim that a command passed unless it was run in the current worktree. If a required check cannot run, report the exact command, reason, and remaining risk.

Documentation-only changes do not require invented tests, but they must be checked for valid paths, commands, internal consistency, and an accurate Git diff.

## Data Pipeline Invariants

Treat `data/yelp/raw/`, `data/yelp/interim/`, and `data/yelp/processed/` as raw, intermediate, and generated layers. Do not commit raw archives, extracted images, generated tables, model caches, or reports containing fabricated measurements. Commit only code, lightweight configuration, schemas, tests, and documentation needed to reproduce outputs.

Process large JSONL sources incrementally and write bounded chunks. Do not materialize full review, image, or similarity datasets in memory. Keep schemas stable across chunks, serialize nested fields deliberately, and bound per-business aggregation to downstream needs.

Cleaning and validation must be explicit and auditable:

- Preserve source and accepted row counts and record rejection reasons.
- Reject invalid reviews by documented identifier, text, length, and content rules.
- Record image existence and readability status; exclude invalid images from downstream pairs without silently dropping their validation records.
- Declare join keys and quality levels for alignment artifacts.
- Compute statistics from actual generated outputs; never invent counts, ratios, scores, or completion claims.

Strong image-caption pairs require the same `photo_id`, a readable image, and a non-empty native caption. Image-business attribute pairs join on `business_id` and use standardized natural-language descriptions with dimension labels. Weak image-review alignment is business-level, bounded, and reported with coverage.

Semantic denoising is optional. When run, record model, threshold, batch limits, device, row-level scores, and before/after statistics. When skipped or unavailable, write an explicit skipped status instead of implying completion.

## Documentation and Experiments

Update documentation when behavior, commands, schemas, configuration, or accepted decisions change. Keep `README.md` focused on what can be run now, and keep historical evidence in weekly and experiment records.

Use Simplified Chinese as the primary language for mentor-facing reports, requirement and decision explanations, weekly logs, delivery records, experiment narratives, and human-readable run summaries. Tables, metric names, paths, commands, configuration keys, Schema fields, model names, and machine-readable log values may remain in English. Do not translate stable interfaces or rewrite unrelated historical records solely to change their language.

Record model-serving, prompt, dataset, retrieval, or evaluation changes in the appropriate experiment files. Each experiment should identify the date, Git commit, model and backend, configuration, dataset version, command, metrics, failures, and next action. Use only observed results.

For weekly review work, map every delivered change to the current requirement. Include reproducible commands, verification results, expected outputs, known limitations, and documentation updates. Do not commit personal schedules, agent scratch plans, chat transcripts, or internship reflections.

## Git and Delivery

Use `feature/* -> dev -> stg -> main` as the normal promotion flow:

- Develop on `dev` or a focused `feature/*` branch, never directly on `main`.
- Promote only verified weekly deliverables from `dev` to `stg`.
- Promote `stg` to `main` only for a milestone or mentor-confirmed stable version.

This is currently a single-developer repository, so lightweight direct commits to `dev` are acceptable when the change is coherent and verified. Keep commits small and use short imperative messages. Do not stage unrelated working-tree changes.

Before promoting to `stg`, review the changed-file list, confirm no raw or generated data is included, run the relevant tests and validators, update the weekly log and delivery evidence, state known limitations, and propose the weekly tag. Create the tag only after the verified commit is pushed successfully.

Pull requests or promotion summaries should include a concise change summary, verification commands and results, and any affected configs, datasets, schemas, or experiment records.

## Security and Resource Safety

Never commit secrets, API keys, credentials, private endpoints, model weights, local virtual environments, caches, or licensed raw datasets. Keep `.env` files local and provide sanitized examples when configuration documentation is needed.

Before changing or running GPU services, verify the documented CUDA, container runtime, model compatibility, and memory assumptions. Do not run competing GPU workloads concurrently when the documented workflow requires exclusive access.

Treat an acquired GPU allocation as a scarce execution asset. For long-running training, keep the Slurm allocation owner or supervisor separate from the training child process so a recoverable application failure does not automatically terminate the allocation. Save automatic, resumable checkpoints at bounded intervals and before risky transitions. On a recoverable failure, keep the allocation alive for a bounded diagnosis and repair window, preserve logs and immutable outputs, apply only a versioned and verified fix, and resume from the latest valid checkpoint in the same allocation. Do not release the GPU merely because the training child exited. Stop the allocation only for data-identity mismatch, memory or hardware safety risk, output-contamination risk, an unrecoverable infrastructure failure, or the requested walltime limit. If the scheduler has already reclaimed the allocation, record that accurately instead of describing it as an intentional release.

Request the smallest realistic GPU walltime that covers the measured workload plus an evidence-based safety buffer. Estimate it from pilot throughput, dataset size, epochs, evaluation and checkpoint overhead, and startup or reload cost; update later requests from observed runtimes. The purpose of a tighter walltime is faster scheduling and better backfill eligibility, not usage-based billing. A shorter request may improve schedulability but is not a guaranteed priority boost. Do not default to a multi-day maximum such as 72 hours when evidence supports a materially shorter request, and do not choose an unrealistically short limit that is likely to kill a healthy run before a checkpoint can complete.

Avoid destructive Git, filesystem, Docker, and data operations unless the user explicitly requests them and the target has been verified. Prefer reversible, scoped operations and preserve reproducibility evidence.

## Local Infrastructure Context

Before operating project-specific cloud servers, read `.agents/server.local.md` when it exists. Treat it as local, non-authoritative runtime context: verify live state before any cost-incurring, destructive, networking, or security-group change. Never copy private-key contents, API keys, passwords, or other secrets from local infrastructure files into tracked files, logs, chat output, or commits.

## Single-Operator Human Review

When the user explicitly declares that only one human operator is available, never invent a second annotator or independent reviewer. Use the accepted single-operator workflow: the operator explicitly confirms self-review while saving each correction, then performs deterministic blind second-pass cross-review and core audit only for selected samples in distinct review sessions. Unsampled records may be accepted after real human correction plus inline self-review. Model output, automatic validation, or an agent action must never supply the human confirmation, reviewer identity, review session, or acceptance decision.
