# Weekly Log

Use this file for concise weekly progress records. Keep detailed mentor-facing prose in `docs/internship_weekly_summary.md` when needed.

## Week 1: Engineering Foundation and Data Preparation

### Completed

- Created the project scaffold for an OTA multimodal search and travel planning system.
- Added FastAPI endpoints for health, image understanding, visual search, and travel planning.
- Added deterministic fallback behavior so local tests do not require a live vLLM server.
- Added Docker and vLLM serving configuration for local GPU smoke tests.
- Added sample POI, review, and image data under `data/samples/`.
- Added Yelp Open Dataset preparation logic and tests.
- Created experiment logs, results table, and failure-case tracking under `experiments/`.

### Verification

```bash
python -m unittest discover -s tests
python scripts/test_health.py
python scripts/test_image_understanding.py
```

### Notes

- Use smaller VLM models for local 8GB GPU validation when larger Qwen-VL variants are not stable.
- Keep raw datasets and generated large outputs out of Git.

### Review and Report - 2026-07-09

Changed files reviewed:

- New review/control documents: `AGENTS.md`, `docs/requirements.md`, `docs/weekly_log.md`, `docs/decisions.md`, `docs/experiments.md`, `docs/internship_weekly_summary.md`.
- Local raw Yelp archives were present under `data/` and must remain uncommitted.

Verification completed:

```bash
python -m unittest discover -s tests -v
python scripts/test_health.py
python scripts/test_image_understanding.py
```

Results:

- Unit tests passed: 9/9.
- Health smoke test passed and returned the OTA multimodal service metadata.
- Image-understanding smoke test passed and returned structured JSON output from the running API.

Review findings:

- Week 1 acceptance is mostly satisfied for scaffold, API fallback behavior, smoke tests, Docker/vLLM documentation, sample data, and Yelp JSONL subset preparation.
- Raw Yelp archives were detected at `data/Yelp-JSON.zip` and `data/Yelp-Photos.zip`; `.gitignore` now ignores `data/Yelp-*.zip`, but the files should not be staged or committed.
- Current Yelp preparation code expects extracted raw JSON/metadata files under `data/yelp/raw/`; it does not currently implement direct extraction from the official zip/tar archives. Any report language should avoid claiming archive extraction support unless that code is added.
- Live VLM JSON parsing is usable for the current smoke result, but malformed or truncated model JSON remains a known future hardening task.

## Weekly Log Template

```markdown
## Week X: Task Name

### Completed

- Item completed

### Verification

- `command` should pass and produce expected output.

### Issues / Risks

- Risk or blocker

### Next Steps

- Next action
```
