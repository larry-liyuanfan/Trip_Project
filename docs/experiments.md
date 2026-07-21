# Experiment Notes

## Week 1 Serving and API Baseline

- Runtime: `vllm/vllm-openai:v0.8.5` with `Qwen/Qwen2-VL-2B-Instruct` on the local 8GB NVIDIA GPU.
- Verified `/health`, `/v1/models`, and live single-image `/v1/image-understanding` requests.
- Deterministic fallback remained available when the live service was absent.
- Yelp sample preparation produced 200 businesses, 1,000 reviews, and 581 multimodal items.
- Multi-image structured JSON quality remained a stretch limitation and was recorded in `experiments/failure_cases.md`.

## Week 2 Data Processing Baseline

- Dataset source: local Yelp Open Dataset files under `data/yelp/raw/`.
- Config: `configs/data_processing.yaml`.
- Base pipeline does not require GPU, CLIP, live vLLM, or `requirements-llm.txt`.
- Week 2 data-only dependency install: `pip install -r requirements-data.txt`.
- CLIP denoising runs through the dedicated `clip-denoising` Docker profile and records a model, device, candidate, retention, and similarity summary in `data/yelp/processed/clip_denoising_summary.json`.
- Generated large outputs remain under ignored `data/yelp/` directories.

Record future full-dataset runs with command, Git commit, raw file availability, output counts, runtime, and any skipped dependency notes.

### Full local Week 2 run on 2026-07-10

- Command sequence: parse, alignment, CLIP denoising, report generation with `configs/data_processing.yaml`.
- Review processing limit: none.
- Parsed rows: 150346 businesses, 6989830 valid reviews from 6990280 raw review lines, 200100 photo metadata records.
- Raw extraction: 5 core JSON files, `photos.json`, `photos/`, and official documentation/ToS files are present under `data/yelp/raw/`.
- Image validation: 199994 valid images, 0 missing images, 106 corrupted/unreadable images.
- Alignment rows: 96733 strong non-empty-caption image pairs, 199994 medium image-business pairs, 36673 weak business-level groups.
- Data quality statistics: city count, valid image ratio, photo label distribution, caption length statistics, weak-alignment category coverage, and denoising before/after counts are recorded in `dataset_statistics.json` and the report.
- CLIP denoising: completed on 2026-07-10 with `openai/clip-vit-base-patch32` on CUDA (RTX 4070 Laptop GPU). Input: 36,673 weak groups and 555,459 candidates; retained: 131,146 at threshold 0.25; similarity min/mean/max: 0.0226 / 0.2210 / 0.4199.
- Storage behavior: real Parquet files were written because a pyarrow Parquet engine is available locally.
- Output validation: `scripts/validate_week2_pipeline.py` confirmed expected files, columns, image paths, report counts, and Parquet format.
- Full-run robustness: business nested attributes/hours use stable JSON storage for chunk-compatible Parquet schemas; review statistics use running sums/counts; image validation runs in bounded parallel batches.

## Week 3 Evaluation Readiness Record (historical, superseded)

- Date: 2026-07-14.
- Git state: HEAD `06005fa` with a dirty Week 3 worktree.
- Overall Week 3 status: `PARTIAL`.
- Engineering status: Stage 1–4 implementation and verification complete.
- Real evaluation status: Stage 5 baseline and standardized comparison `PENDING`.
- Synthetic/mock framework verification: PASS，不属于真实模型 baseline，不计入 tested_count。
- 2026-07-14 `/v1/models` 探测成功，返回 `Qwen/Qwen2-VL-2B-Instruct`；未发送 Week 3 图片请求，未产生模型输出或延迟指标。

Evaluation data counts:

| Scenario | target_count | candidate_count | annotated_count | validated_count | tested_count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product understanding | 200 | 200 | 200 | 200 | 0 |
| After-sales issue recognition | 150 | 150 | 150 | 150 | 0 |
| Itinerary constraint understanding | 100 | 100 | 100 | 100 | 0 |

This historical count snapshot used the earlier eligibility rule and was later
invalidated by Project Control's gold-coverage review.

The two existing Stage 3 dry-runs use `baseline_minimal_v1`; each has
`selected_count=0` and `record_count=0`. They verify framework behavior only.
No Stage 5 run directory or score artifact exists.

All real product-understanding, after-sales, itinerary, JSON/Schema, OCR, and
latency metrics are `PENDING`. No typical model failure case or capability
weakness is inferred because no Week 3 image request or model output exists.

The local Yelp source data and processed artifacts exist, including 200,100
photo metadata rows and 199,994 image-business attribute pairs. This readiness
snapshot is historical; the approved frozen-label restoration below supersedes
its run-eligibility interpretation.

### Historical incomplete items at readiness time

At that readiness snapshot, the remaining mentor-approved Week 3 scope was the
full real baseline, standardized comparison, scoring, and evidence-backed
reporting. Project Control later rejected the underlying test-set coverage, so
those historical runs do not satisfy the current incomplete items.

## Week 3 Full Evaluation Record

- Date: 2026-07-21.
- Git state: dirty Week 3 worktree on `dev`; no commit or promotion performed.
- Model/backend: `Qwen/Qwen2-VL-2B-Instruct`, vLLM, temperature 0.1, top_p 0.9, max_tokens 512.
- Dataset: frozen `week3_evaluation_v1`, 450 completed and structurally valid samples; exclusion count 450.
- Baseline: `week3_baseline_full_20260721_003`, completed/live/full, 450/450 records.
- Standardized: `week3_standardized_full_20260721_001`, completed/live/full, 450/450 records.
- Pairing: identical selected-sample hash `5d244771ae4acd9eca46ad3937394232733d2526f2dde2255774ed2dcf9e96a7` and identical non-Prompt artifacts/model settings.
- Comparison: `week3_prompt_pair_strict_20260721_001`, 450 paired rows, 2,000 bootstrap iterations.
- Baseline strict JSON compliance: 0% in all scenes because the mentor-required minimal Prompt has no format instruction.
- Standardized JSON compliance: product 68.5%, after-sales 98.0%, itinerary 28.0%.
- Standardized Schema pass: product 29.5%, after-sales 2.0%, itinerary 0.0%.
- Project Control approved reuse of the frozen annotations without relabeling. The runs are valid raw-output, format, and latency evidence; unsupported semantic metrics and frozen-gold limitations remain explicit, so Week 3 stays `PARTIAL`.

## Week 3 data-quality correction on 2026-07-21

- Current status: `PARTIAL`; valid baseline and standardized comparison are `PENDING`.
- Corrected eligibility excludes rejected records, pending PII state, product `unknown`, non-target after-sales labels, and empty itinerary styles.
- Rebuilt after-sales candidates with deterministic v3 evidence and exact `38/38/37/37` strata. All 150 are pending human annotation; no gold label was generated automatically.
- Current counts are product `200/200/200/110/0`, after-sales `150/150/0/0/0`, itinerary `100/100/100/0/0`.
- Minimal-baseline unparsed task metrics are `PENDING`; only format compliance and latency remain directly measurable without an approved deterministic parser or human coding protocol.

This correction route was superseded by the approved frozen-label restoration;
the v3 pending candidates remain backed up and are not the active run-bound
manifest.

## Week 3 frozen-label restoration on 2026-07-21

- Restored after-sales manifest SHA-256 `e1fdfc1b77db6519b311a6f846f4ff02df336e34661d841c1a5a42c725dc8a6e`; product and itinerary hashes remain `cd85ce2926b3c9adee85c95dc166edd3b9905a844d4b9dd8fe76c224e133dd15` and `584e2725459a88d48925077fe28239c77860f64b039fd410ed9199a0c6909fa8`.
- Rebuilt exclusion registry hash `1430478f2af28c63025d017a806c3e8924900a168b39ca756eac8b0d776465c3`.
- After-sales sources: public Yelp 76, business synthetic 74. Audit binding: 150/150 annotation payload hashes and annotators match.
- Baseline and standardized run validators both return `status=ok` with 450 selected/persisted records and tested counts 200/150/100.
- Baseline JSON and Schema rates are 0% for all scenarios; semantic task metrics are `PENDING` with support count 0. Mean latency is 3463/2139/3508 ms for product/after-sales/itinerary.
- Standardized JSON rates are 68.5%/98.0%/28.0%; Schema rates are 29.5%/2.0%/0.0%. Scalar metrics use only known gold: product category 20.0% over 110, product price 13.0% over 100, after-sales issue and severity 0% over 82 each.
- Frozen limitations: product category unknown 90 and price unknown 100 (valid no-direct-evidence labels, not automatically missing); product `visible_facilities` is non-empty for 128 and empty for 72; after-sales issue unknown 68 and facility-damage gold 0; itinerary style preferences empty 100, with retained evidence indicating a probable historical annotation-UI field exposure or serialization defect rather than confirmed annotator omission.
- Final scope decision: Project Control selected frozen v1. No v2 manifest, annotation-UI reopening, supplemental annotation, or v2 rescoring is part of this delivery; unsupported itinerary-style, facility-damage, and baseline semantic metrics remain `PENDING`, and Week 3 remains `PARTIAL`.
