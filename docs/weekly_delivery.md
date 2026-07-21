# Weekly Delivery Record

This is the canonical weekly delivery document. Each week keeps its original
scope, acceptance checklist, measured results, and known limitations. New
weeks append sections instead of replacing earlier delivery history.

## Week 1: OTA Multimodal VLM Engineering Foundation

### Objective

Build a reproducible local foundation for an OTA multimodal search system:
Dockerized model serving, a FastAPI image-understanding API, a small Yelp data
sample, deterministic fallback behavior, and experiment records.

### Completion Checklist

- [x] Initialize the Git repository and project directory structure.
- [x] Add the README, Dockerfile, Docker Compose configuration, and vLLM launcher.
- [x] Add the FastAPI scaffold and deterministic image-understanding fallback.
- [x] Add structured image-understanding request and response schemas.
- [x] Add sample POI, review, and image data.
- [x] Add model-selection and API-design documentation.
- [x] Start a real vLLM service on the local NVIDIA GPU.
- [x] Verify live single-image inference through the API.
- [x] Record the first real model experiment with its Git commit and runtime parameters.
- [x] Prepare more than 10 local Yelp subset images.
- [x] Add a multi-image live test as a stretch item.

### Delivered Artifacts

- API routes: `/health`, `/v1/image-understanding`, `/v1/visual-search`, and `/v1/travel-planning`.
- Docker services for the API and `vllm/vllm-openai:v0.8.5`.
- Stable local smoke model: `Qwen/Qwen2-VL-2B-Instruct` on an 8GB GPU.
- Yelp sample outputs: 200 businesses, 1,000 reviews, and 581 multimodal items.
- Experiment records in `experiments/experiment_log.md`, `experiments/results.csv`, and `experiments/failure_cases.md`.

### Verification and Limitations

- The Week 1 unit-test baseline passed 9 tests.
- Single-image live inference completed through the vLLM OpenAI-compatible endpoint.
- Multi-image input reached the live model, but the small model sometimes returned truncated or malformed JSON. This was recorded as a non-blocking stretch limitation.
- Qwen2.5-VL-3B was not used for the stable smoke path because model loading and profiling exceeded the comfortable margin of the local 8GB GPU.

## Week 2: Yelp Multimodal Dataset Processing

### Objective

Build and verify a full-data Yelp pipeline for archive extraction, streaming
parsing, image validation, strong/medium/weak multimodal alignment, CLIP
semantic denoising, and mentor-facing reporting.

### Completion Checklist

- [x] Extract all 5 core Yelp JSON files, photo metadata, local photos, and official documentation.
- [x] Normalize raw, interim, processed, validation, log, and report directories.
- [x] Stream all business, review, and photo metadata records without loading the full sources into memory.
- [x] Filter invalid reviews and record every rejection reason.
- [x] Validate every referenced local image for existence and readability.
- [x] Build non-empty-caption strong pairs joined by `photo_id`.
- [x] Build image-business attribute pairs joined by `business_id`.
- [x] Build bounded business-level weak image-review groups.
- [x] Run CLIP image-review scoring in an isolated GPU Docker task.
- [x] Validate output files, schemas, counts, image paths, and Parquet storage.
- [x] Generate the detailed Yelp processing report.

### Measured Full-Run Results

| Metric | Result |
| --- | ---: |
| Businesses parsed | 150,346 |
| Raw review rows | 6,990,280 |
| Valid review rows | 6,989,830 |
| Photo metadata rows | 200,100 |
| Valid local images | 199,994 |
| Missing local images | 0 |
| Corrupted/unreadable images | 106 |
| Covered cities | 1,416 |
| Strong non-empty-caption pairs | 96,733 |
| Medium image-business pairs | 199,994 |
| Weak business groups | 36,673 |
| CLIP candidates scored | 555,459 |
| CLIP pairs retained at threshold 0.25 | 131,146 |

### Reproducible Commands

```bash
pip install -r requirements-data.txt
python scripts/parse_yelp_json.py --config configs/data_processing.yaml
python scripts/build_yelp_alignment.py --config configs/data_processing.yaml
docker compose -f docker/docker-compose.yml stop vllm
docker compose -f docker/docker-compose.yml --profile data run --rm clip-denoising
python scripts/generate_yelp_report.py --config configs/data_processing.yaml
python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml
python -m unittest discover -s tests -v
```

### Delivered Artifacts and Limits

- `requirements-data.txt` remains independent of vLLM, torch, and CLIP.
- `requirements-clip.txt` and `docker/Dockerfile.clip` define the isolated CLIP runtime.
- Large raw files, generated Parquet files, images, and model weights remain ignored.
- Full review processing uses bounded chunk writes; image validation uses bounded worker batches.
- CLIP used `openai/clip-vit-base-patch32` on CUDA. vLLM must be stopped first because both workloads cannot safely share the local 8GB GPU.
- The mentor-facing output is `reports/yelp_multimodal_data_processing_report_part1.md`.

## Week 3: Zero-Shot Evaluation Framework

### Objective and Status

Build an auditable three-scenario zero-shot evaluation framework with stable
data contracts, multimodal prompts, strict structured output, a
configuration-driven runner, metrics, and reporting. Local implementation and
the frozen human-authored manifests and real run artifacts are validated. The
delivery remains `PARTIAL` because baseline semantic metrics and several
mentor-required gold dimensions are unsupported by the frozen data.

### Completion Checklist

- [x] Implement manifest inputs, image SHA-256 validation, exclusion tracking, duplicate rejection, and local initialization.
- [x] Implement baseline and standardized multimodal requests without changing the three baseline prompt texts.
- [x] Expose the complete Schema contract and enforce scenario-specific images, bounded evidence, and itinerary structure.
- [x] Implement strict JSON handling, pre-run registry validation, scene ownership validation, runner metadata, and error separation.
- [x] Implement completed-run metadata consistency checks and explicit failed-run rejection, metrics, summaries, and error export.
- [x] Implement deterministic non-gold annotation suggestions and the human annotation application gate.
- [x] Restore and validate the frozen human-authored manifests without relabeling.
- [x] Validate completed full `baseline_minimal_v1` run `week3_baseline_full_20260721_003`.
- [x] Validate the optional `standardized_v1` run on the identical frozen set.
- [x] Generate an evidence-backed status report with unsupported metrics marked `PENDING`.
- [x] Record Project Control's frozen-v1 decision: no v2 dataset, annotation reopening, supplemental labels, or v2 rescoring.
- [x] Receive Project Control approval of the final actual diff and evidence boundary.

Current verification restores all 450 completed annotations and 450 exclusion
rows. Both named completed runs pass artifact validation; no equivalent live
requests were repeated. The final repository suite passes 203 Python tests.

### Evaluation Data Counts

| Scenario | target_count | candidate_count | annotated_count | validated_count | tested_count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product understanding | 200 | 200 | 200 | 200 | 200 |
| After-sales issue recognition | 150 | 150 | 150 | 150 | 150 |
| Itinerary constraint understanding | 100 | 100 | 100 | 100 | 100 |

The local Yelp source data and Week 2 processed artifacts exist. The exclusion
registry contains 450 candidates. `tested_count` is bound to the completed
baseline run. Unknown and empty semantic fields remain frozen limitations and
reduce metric support rather than release eligibility.

Product `price_range=unknown` is an allowed evidence-based result, and product
`visible_facilities` is non-empty for 128 samples and empty for 72. The 100
empty itinerary `style_preferences` arrays are recorded as a probable
historical UI field-exposure or serialization defect without blaming the
annotator. Itinerary style, after-sales facility-damage, and baseline
natural-language semantic metrics remain `PENDING`; the Week 3 status is
`PARTIAL`.

### Verification Evidence and Boundaries

- Synthetic/mock framework verification: PASS，不属于真实模型 baseline，不计入 tested_count。
- `stage3_dry_run_20260713_001`: `baseline_minimal_v1`, `selected_count=0`, `record_count=0`.
- `stage3_dry_run_20260713_002`: `baseline_minimal_v1`, `selected_count=0`, `record_count=0`.
- Both dry-runs belong to Stage 3 and validate only the zero-selection framework path.
- 2026-07-14 `/v1/models` 探测成功，返回 `Qwen/Qwen2-VL-2B-Instruct`；未发送 Week 3 图片请求，未产生模型输出或延迟指标。
- Runs `week3_baseline_full_20260721_003` and `week3_standardized_full_20260721_001` each retain 450 records and pass restored-manifest provenance validation.
- Historical comparison `week3_prompt_pair_strict_20260721_001` remains optional traceability evidence and is not a Week 3 completion gate.
- Baseline semantic task metrics are `PENDING`; invalid natural-language JSON is not treated as semantic zero.
- The current status and data defects are documented in `reports/week3_zero_shot_baseline_report.md`.

The standalone status report is
`reports/week3_zero_shot_baseline_report.md`.

### Review boundary

Project Control approved one complete Week 3 commit and push to `dev` for the
frozen-v1 `PARTIAL` delivery. The delivery must not be promoted to `stg`,
merged to `stg`, tagged, or expanded into follow-up work.

## Promotion Rule

Weekly work is implemented and verified on `dev`, promoted unchanged to `stg`
for mentor review, and promoted from `stg` to `main` only after approval. A
completed checklist is updated on `dev` before promotion so all downstream
branches inherit the same delivery state.
