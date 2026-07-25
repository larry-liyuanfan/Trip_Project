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
- **Status**: Superseded
- **Decision**: Keep the existing `week3_evaluation_v1` manifests and completed run artifacts immutable. Do not create `week3_gold_v2`, reopen the annotation UI, request supplemental labels, or perform v2 rescoring. Treat evidence-supported `unknown` values as completed labels rather than omissions. Keep itinerary image-style preference, after-sales facility-damage, and baseline natural-language semantic metrics `PENDING` where the frozen evidence does not support them.
- **Reason**: Project Control approved the frozen-v1 route after reviewing the annotation audit, historical UI backups, corrected product-facility statistics, and run provenance. The 100 empty itinerary style arrays are recorded as a probable historical field-exposure or serialization defect and are not attributed to the annotator.
- **Consequence**: Week 3 remains `PARTIAL`; reports must preserve support counts and limitations without modifying gold labels, rerunning equivalent inference, or converting sampling metadata into gold coverage.

## ADR-008: Build an Isolated Curated Week 3 v2 Evaluation Set

- **Date**: 2026-07-22
- **Status**: Accepted
- **Decision**: Preserve every v1 manifest, Prompt, Schema, and run as immutable history. Build `week3_evaluation_v2` in new ignored manifests and registry files, reuse all 200 product labels, retain 80 evidence-supported after-sales labels, replace 70 low-evidence after-sales candidates, and reopen the 100 itinerary pairs only to capture the previously unavailable image-style field. Use one human annotator; deterministic suggestions never become gold automatically.
- **Reason**: The mentor explicitly prioritized removal of low-quality images so the zero-shot baseline can serve as a fair reference for later comparison. The frozen-v1 set has unsupported facility-damage and itinerary-style dimensions.
- **Consequence**: V2 full baseline and standardized runs cannot begin until the 70 after-sales replacements and 100 itinerary style supplements are complete. Existing itinerary non-style labels are inherited, not re-entered. Standardized v2 may use a separately versioned bounded itinerary Schema and output-type skeleton to improve raw JSON/Schema compliance, while strict format compliance remains separate from semantic quality.

## ADR-009: Score Minimal-Baseline Semantics with a Gold-Independent Lexical Track

- **Date**: 2026-07-25
- **Status**: Accepted
- **Decision**: Use `baseline_semantic_coding_v1` to convert the immutable `baseline_minimal_v1` raw text into predictions before loading human gold. The encoder accepts only the scenario, raw output, a fixed versioned codebook derived from existing Schemas and annotation definitions, and general text normalization. Gold is joined only in the scoring stage.
- **Reason**: The mentor requires baseline business metrics, while the minimal Prompt intentionally produces unconstrained natural language. A deterministic lexical track measures supported semantics without changing the Prompt, rerunning inference, adding manual output coding, or treating JSON failure as semantic failure.
- **Consequence**: Store the lexical metrics as an independent scoring track with explicit support counts and codebook SHA-256. Preserve baseline JSON/Schema rates, raw output, latency, and run provenance. Do not compare this track to the standardized strict-structured track as if their difference were a pure Prompt effect.

## ADR-010: Select Week 4 Prompts Only from Fixed Tested Candidates

- **Date**: 2026-07-25
- **Status**: Accepted
- **Decision**: Use fixed v2-gold examples and a fixed disjoint pilot to compare `standardized_v2`, 4-shot, and 7-shot. Select by the checked-in weighted business, JSON, Schema, token, and latency score, then run only the per-scenario winner on the full v2 set.
- **Reason**: This meets the mentor's bounded Prompt-optimization requirement without changing gold, inventing labels, or expanding the candidate search.
- **Consequence**: `standardized_v2` is the winner for all three scenarios only among these tested candidates. Week 3 artifacts stay immutable, and cross-track baseline differences are descriptive rather than causal.

## ADR-011: Keep Milvus and CLIP Isolated from Business Inference

- **Date**: 2026-07-25
- **Status**: Accepted
- **Decision**: Run fixed-version Milvus standalone with a separate PyMilvus dependency group and store normalized 512-dimensional `openai/clip-vit-base-patch32` image vectors. Keep Qwen2-VL on its existing vLLM inference interface and never treat it as an embedding endpoint.
- **Reason**: The mentor requires real vector CRUD without changing the existing API/data/vLLM dependency groups or exceeding the local 8 GB GPU boundary.
- **Consequence**: vLLM is stopped before CLIP runs; generated vectors and volumes stay ignored. Retrieval supports only the fixed scalar-filter whitelist and HNSW/COSINE parameters from configuration.

## Decision Template

```markdown
## ADR-XXX: Title

- **Date**: YYYY-MM-DD
- **Status**: Proposed | Accepted | Superseded
- **Decision**:
- **Reason**:
- **Consequence**:
```
