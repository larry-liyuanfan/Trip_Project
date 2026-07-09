# Weekly Requirements

This file is the source of truth for mentor-provided weekly requirements, deliverables, and acceptance criteria. Add new weeks at the top or below the current active week, then keep implementation scoped to that section.

## Week 1: Engineering Foundation and Data Preparation

### Background

Build the reproducible foundation for an OTA multimodal VLM search and travel planning project.

### Goals

- Establish the repository structure, local setup, Docker layout, and FastAPI scaffold.
- Validate vLLM-compatible image understanding through a deterministic fallback and local smoke tests.
- Prepare a Yelp Open Dataset subset workflow for OTA-style POI, review, and image metadata.
- Create repeatable experiment records.

### Non-goals

- Do not build a full UI.
- Do not commit raw Yelp archives, extracted images, model weights, or local environment files.
- Do not require live GPU inference for every local test.

### Deliverables

- `src/` modules for API, inference, retrieval, planning, data, and evaluation.
- `docker/` service definitions and vLLM launch script.
- `data/samples/` mock catalog, reviews, and image.
- Experiment records under `experiments/`.
- Supporting docs under `docs/`.

### Acceptance Criteria

- `python -m unittest discover -s tests` passes.
- `python scripts/test_health.py` verifies the health endpoint.
- `python scripts/test_image_understanding.py` returns structured image-understanding output.
- Yelp subset preparation can generate POI, review, and multimodal JSONL outputs from local raw files.

### Risks / Questions

- Local GPU memory may require smaller VLM models for smoke tests.
- vLLM image, CUDA, and NVIDIA driver versions must remain compatible.
- Live model JSON output may be malformed and needs parser hardening in later weeks.

## New Weekly Requirement Template

```markdown
## Week X: Task Name

### Background

### Goals

- Goal 1

### Non-goals

- Out-of-scope item

### Deliverables

- Deliverable 1

### Acceptance Criteria

- Verification command and expected result

### Risks / Questions

- Open issue or technical risk
```
