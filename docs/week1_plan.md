# Week 1 Plan

## Goal

Build the reproducible engineering foundation for a VLM-based OTA multimodal search system.

## P0 Checklist

- [x] Initialize Git repository.
- [x] Create repository structure.
- [x] Add README draft.
- [x] Add Dockerfile and docker-compose draft.
- [x] Add vLLM launch script.
- [x] Add FastAPI business API scaffold.
- [x] Add image-understanding fallback path for local smoke tests.
- [x] Add experiment log and results templates.
- [ ] Start real vLLM service on GPU.
- [ ] Verify live image input inference.
- [ ] Record first real model experiment with Git commit.

## P1 Checklist

- [x] Add structured JSON schema for image understanding.
- [x] Add sample POI catalog and reviews.
- [x] Add model selection document.
- [x] Add API design document.
- [ ] Add 10-20 real sample images.
- [ ] Add multi-image live vLLM test.

## Acceptance Criteria

- API starts locally with deterministic fallback.
- `/health` returns service metadata.
- `/v1/image-understanding` returns structured fields.
- Experiment files exist and define reproducible logging fields.
