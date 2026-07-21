# Technical Decisions

Record decisions that affect architecture, reproducibility, model serving, data handling, branching, or review scope.

## ADR-001: Keep API Tests Independent from Live vLLM

- **Date**: 2026-07-09
- **Status**: Accepted
- **Decision**: Maintain deterministic fallback responses for local image-understanding tests when live vLLM is not configured.
- **Reason**: Contributors can run core tests without GPU access, model downloads, or container startup.
- **Consequence**: Live model behavior must be validated separately through smoke tests and experiment records.

## ADR-002: Store Raw and Generated Yelp Data Outside Git

- **Date**: 2026-07-09
- **Status**: Accepted
- **Decision**: Keep raw Yelp archives, extracted images, and generated large subsets ignored locally.
- **Reason**: These files are large, external, and may have distribution restrictions.
- **Consequence**: Dataset preparation must be reproducible from documented commands and local source files.

## ADR-003: Use Experiment Files for Reproducibility

- **Date**: 2026-07-09
- **Status**: Accepted
- **Decision**: Track experiment commands, parameters, outcomes, and failures in `experiments/` and summarize them in `docs/experiments.md`.
- **Reason**: Weekly mentor review needs clear evidence of what was run and what changed.
- **Consequence**: Model, prompt, data, and serving changes should update experiment documentation before review.

## ADR-004: Use `dev`, `stg`, and `main` for Weekly Delivery

- **Date**: 2026-07-09
- **Status**: Accepted
- **Decision**: Use `feature/* -> dev -> stg -> main` as the promotion flow. Daily work happens on `dev` or `feature/*`; verified weekly deliverables promote to `stg`; milestone or mentor-confirmed stable versions promote from `stg` to `main`.
- **Reason**: This separates active development and experiments from mentor-reviewed weekly deliverables and milestone-level stable code.
- **Consequence**: Before merging into `stg`, provide a changed-files summary, verification commands and results, expected outputs, known limitations, updated documentation, and a proposed weekly tag such as `week2-yelp-data-processing`.

## ADR-005: Build Week 2 Yelp Processing as a Config-Driven Offline Pipeline

- **Date**: 2026-07-09
- **Status**: Accepted
- **Decision**: Add a reusable offline pipeline for Yelp JSONL parsing, local image validation, multimodal alignment generation, optional CLIP denoising, and report generation.
- **Reason**: Weekly review needs reproducible data preparation artifacts without requiring live VLM serving, GPU access, or committed raw Yelp files.
- **Consequence**: Raw and generated data stay under ignored `data/yelp/` paths; scripts must tolerate missing optional CLIP and Parquet dependencies while documenting the fallback.

## ADR-006: Keep One Canonical Weekly Delivery Record

- **Date**: 2026-07-11
- **Status**: Accepted
- **Decision**: Append each completed week to `docs/weekly_delivery.md`; keep `docs/weekly_log.md` as a concise timeline and avoid separate plan/delivery files per week.
- **Reason**: Separate Week 1 and Week 2 files drifted across branches and obscured earlier completed work.
- **Consequence**: Checklist state is finalized on `dev` before promotion, then inherited unchanged by `stg` and `main` through merge-based promotion.

## ADR-007: Freeze the Week 3 v1 Evaluation Labels

- **Date**: 2026-07-21
- **Status**: Accepted
- **Decision**: Keep the existing `week3_evaluation_v1` manifests and completed run artifacts immutable. Do not create `week3_gold_v2`, reopen the annotation UI, request supplemental labels, or perform v2 rescoring. Treat evidence-supported `unknown` values as completed labels rather than omissions. Keep itinerary image-style preference, after-sales facility-damage, and baseline natural-language semantic metrics `PENDING` where the frozen evidence does not support them.
- **Reason**: Project Control approved the frozen-v1 route after reviewing the annotation audit, historical UI backups, corrected product-facility statistics, and run provenance. The 100 empty itinerary style arrays are recorded as a probable historical field-exposure or serialization defect and are not attributed to the annotator.
- **Consequence**: Week 3 remains `PARTIAL`; reports must preserve support counts and limitations without modifying gold labels, rerunning equivalent inference, or converting sampling metadata into gold coverage.

## Decision Template

```markdown
## ADR-XXX: Title

- **Date**: YYYY-MM-DD
- **Status**: Proposed | Accepted | Superseded
- **Decision**:
- **Reason**:
- **Consequence**:
```
