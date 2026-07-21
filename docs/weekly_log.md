# Weekly Log

## 2026-07-06: Week 1 OTA Multimodal VLM Foundation

- Added the repository structure, FastAPI application, deterministic fallback, retrieval baseline, and planning baseline.
- Added Dockerized API and vLLM services and verified live single-image inference with `Qwen/Qwen2-VL-2B-Instruct`.
- Added the Yelp sample preparation workflow and generated 200 businesses, 1,000 reviews, and 581 multimodal items.
- Added experiment logs, results, failure cases, API design, and model-selection notes.
- Added a multi-image live test as a stretch item; malformed small-model JSON remains a documented non-blocking limitation.
- Completed checklist and evidence are retained in `docs/weekly_delivery.md`.

## 2026-07-10: Week 2 Yelp Multimodal Dataset Pipeline

- Added a config-driven Yelp processing pipeline under `configs/data_processing.yaml`.
- Added line-by-line JSONL reading for business, review, and photo metadata.
- Added chunked table writing for business, review, photo, and image-index outputs; nested business fields are serialized to a stable Parquet schema.
- Added bounded parallel image validation for missing, valid, and unreadable local images.
- Added strong, medium, and weak alignment builders; strong pairs require both a valid image and non-empty caption.
- Added operational CLIP denoising with an explicit skipped status when disabled or unavailable.
- Added report generation for `reports/yelp_multimodal_data_processing_report_part1.md`.
- Added focused unit tests in `tests/test_yelp_data_pipeline.py`.
- Split dependencies so Week 2 data processing uses `requirements-data.txt` and does not require native Windows vLLM installation.
- Extended Yelp archive extraction to cover the 5 core JSON files, `photos.json`, full photo extraction, and official documentation/ToS files under `data/yelp/raw/docs/`.

Verification commands:

```bash
python -m unittest discover -s tests -v
python scripts/parse_yelp_json.py --config configs/data_processing.yaml
python scripts/build_yelp_alignment.py --config configs/data_processing.yaml
docker compose -f docker/docker-compose.yml stop vllm
docker compose -f docker/docker-compose.yml --profile data run --rm clip-denoising
python scripts/generate_yelp_report.py --config configs/data_processing.yaml
python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml
```

Verification results on 2026-07-10:

- Unit tests: 43 tests passed.
- Parse command: 150346 businesses, 6989830 valid reviews, 200100 photo metadata rows, 199994 valid local images.
- Image validation: 0 missing local images, 106 corrupted/unreadable images.
- Alignment command: 96733 strong caption pairs, 199994 medium pairs, 36673 weak business-level groups.
- Data quality statistics: city count, valid image ratio, label distribution, caption length statistics, weak-alignment category coverage, and denoising before/after counts are included in dataset statistics and the report.
- CLIP denoising: completed in a dedicated CUDA Docker task with `openai/clip-vit-base-patch32`; 555,459 candidates scored and 131,146 retained at threshold 0.25.
- Report generation: wrote 10 sections to `reports/yelp_multimodal_data_processing_report_part1.md`.
- Output validation: all expected files, required columns, alignment image paths, report counts, and storage format checks passed.
- Local storage note: `pyarrow` is available in the current environment, so real Parquet files were written.
- Scale note: the current implementation is verified on the full Yelp business/review/photo metadata files and the fully extracted local photo folder.

## 2026-07-14: Week 3 Zero-Shot Evaluation Framework (PARTIAL)

阶段 1–4 工程实现完成并通过验证；阶段 5 真实跑测为 PENDING。

- Stage 1 added auditable scenario manifests, top-level inputs, image SHA-256 validation, exclusion tracking, cross-scenario duplicate rejection, and explicit local initialization.
- Stage 2 added baseline and standardized multimodal request rendering, full Schema exposure, scenario image-count checks, bounded evidence fields, and strict structured itinerary output.
- Stage 3 added the configuration-driven runner, strict JSON handling, pre-run manifest/exclusion validation, source-scene ownership checks, and distinct mock-fixture/live-request errors.
- Stage 4 added completed-run metadata consistency checks and explicit failed-run rejection, metrics, scoring summaries, and error export paths.
- Human annotation export now supports deterministic, non-gold source/rule suggestions under packet-only context; no VLM output is used and suggestions are removed before manifest application.
- Synthetic/mock framework verification: PASS，不属于真实模型 baseline，不计入 tested_count。
- 2026-07-14 `/v1/models` 探测成功，返回 `Qwen/Qwen2-VL-2B-Instruct`；未发送 Week 3 图片请求，未产生模型输出或延迟指标。
- The two existing Stage 3 dry-runs, `stage3_dry_run_20260713_001` and `stage3_dry_run_20260713_002`, both use `baseline_minimal_v1` with `selected_count=0` and `record_count=0`. They validate the framework only and are not real baseline results.
- Configuration validation passed. The full repository test suite passed 180 of 180 tests.

Current evaluation data counts (2026-07-21 validation snapshot):

| Scenario | target_count | candidate_count | annotated_count | validated_count | tested_count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product understanding | 200 | 200 | 200 | 200 | 0 |
| After-sales issue recognition | 150 | 150 | 150 | 150 | 0 |
| Itinerary constraint understanding | 100 | 100 | 100 | 100 | 0 |

This snapshot is historical and is superseded by the 2026-07-21 Project Control
data-quality correction below.

The local Yelp source data and processed outputs exist, including 200,100
photo metadata rows and 199,994 image-business attribute pairs. The historical
450-record annotation and run snapshot below was later rejected by Project
Control because its human-gold coverage was insufficient. It must not be read
as the current validated dataset or accepted baseline.

### Historical incomplete items at that snapshot

- Historical note: the real `baseline_minimal_v1` baseline had not yet run on an accepted validated set.
- Historical note: the standardized comparison had not yet run on that same accepted set.
- Historical note: evidence-backed scoring and reporting remained pending.

## 2026-07-17: After-sales synthetic evidence quality correction

- During human annotation, the v1 business-synthetic closure/delay cards were found to use an undersized default font and visually distracting decorative blocks. The content was technically present but was not suitable evidence for reliable human annotation.
- Replaced all 74 affected pending samples with deterministic v2 evidence cards across four document-style layouts. The exact source text, sample/source/group IDs, strata, and human workflow state were preserved; only image fingerprints and synthetic recipe provenance changed.
- The refresh refused to proceed unless every target remained unannotated, unreviewed, and free of target drafts. It staged outputs, backed up v1 artifacts, used atomic replacement with rollback, rebuilt the exclusion registry and annotation packets, and revalidated perceptual independence.
- Completed refresh run `20260717_after_sales_v2`; the ignored audit is at `data/eval/logs/after_sales_synthetic_refresh_v2.json` and the ignored backup is at `data/eval/backups/synthetic-evidence-v1-20260717_after_sales_v2/`.
- The historical local annotation helper was used during data preparation; it is not part of the approved Week 3 Git deliverable.
- Post-refresh verification passed 86 focused tests and 230 full-suite tests. The validator reported 450 exclusion rows; the three scenario rows were 200/200/18/0/0, 150/150/0/0/0, and 100/100/0/0/0 for target/candidate/annotated/validated/tested.

## 2026-07-21: Historical full evaluation (test set later rejected)

- At run time, all 450 candidates were treated as passing the earlier single-annotator release gate; Project Control later rejected that eligibility claim.
- Completed full baseline run `week3_baseline_full_20260721_003`: 450 selected, 450 persisted, 450 JSON parse failures under the intentionally unconstrained baseline Prompt.
- Completed same-set standardized run `week3_standardized_full_20260721_001`: 450 selected and persisted; 138 JSON parse failures, 250 Schema failures, and 62 Schema-valid outputs.
- Both runs use sample-set hash `5d244771ae4acd9eca46ad3937394232733d2526f2dde2255774ed2dcf9e96a7` and identical non-Prompt artifacts/model settings.
- Generated strict paired comparison `week3_prompt_pair_strict_20260721_001` over 450 samples with 2,000 bootstrap iterations.
- Standardized JSON compliance reached 68.5% / 98.0% / 28.0% for product, after-sales, and itinerary; Schema pass reached 29.5% / 2.0% / 0.0%.
- Run-bound validation reports target/candidate/annotated/validated/tested of 200/200/200/200/200, 150/150/150/150/150, and 100/100/100/100/100, with 450 exclusion rows.
- The 224 Python and 22 JavaScript checks described the superseded pre-review diff; the UI tests and delivery files were subsequently removed to comply with the Week 3 non-UI scope.
- Baseline natural-language outputs retain format and latency measurements, while unparsed semantic task metrics are now `PENDING` rather than numeric zero.
- Project Control rejected the test-set coverage. Week 3 is `PARTIAL`; no commit, push, tag, or `stg` promotion has been performed.

## 2026-07-21: Project Control data-quality correction

- Added release rejection for legacy rejected records, pending PII status, invalid product categories, non-target after-sales labels, and empty itinerary style preferences.
- Added full-run human-gold quota validation independent of candidate `sampling_stratum`.
- Corrected current counts to product `200/200/200/110/0`, after-sales `150/150/0/0/0`, and itinerary `100/100/100/0/0` after rebuilding the invalid after-sales set as pending candidates.
- Replaced the 150 after-sales candidates with deterministic v3 project-owned evidence covering `38/38/37/37`; old manifest and registry are retained under ignored backup storage.
- Removed the browser annotation UI and JavaScript tests from the Week 3 Git delivery. JSONL packet export/application remains available.
- Historical 450-record runs remain immutable traceability evidence but no longer count as valid baseline, comparison, or `tested_count`.
- Current verification passed 201 Python tests; the full-run probe failed at the corrected gold gate before creating a run directory.

## 2026-07-21: Approved frozen-label restoration

- Project Control superseded the recuration route: existing human annotations are frozen and no new annotation, relabeling, second review, or manual semantic coding is required.
- Verified backup SHA-256 `e1fdfc1b77db6519b311a6f846f4ff02df336e34661d841c1a5a42c725dc8a6e`, restored the 150 completed after-sales records, and rebuilt the 450-row exclusion registry to hash `1430478f2af28c63025d017a806c3e8924900a168b39ca756eac8b0d776465c3`.
- The restored after-sales set contains 76 public Yelp and 74 business-synthetic samples; all 150 annotation payload hashes and annotators match the retained audit records.
- Simplified run eligibility to completed human annotation, valid/readable file and structure, and non-rejection. `unknown`, empty semantic fields, and pending compatibility metadata are reported limitations rather than global run blockers.
- Both completed real runs pass manifest, exclusion, Prompt, Schema, sample-set, metadata, and result-count validation. The baseline run supplies tested counts of 200/150/100 without repeating live inference.
- Recomputed local scores from immutable raw outputs. All unparsed baseline semantic metrics are `PENDING`; standardized scalar metrics exclude `unknown` gold and expose support counts.
- Week 3 remains `PARTIAL` because frozen gold has no facility-damage labels, no itinerary style-preference labels, and incomplete scalar support; sampling strata are reported separately and never treated as gold labels.
- Final verification passed 203 Python tests, both run-bound validators, configuration validation, CSV structure checks, and `git diff --check` (CRLF warnings only).

## 2026-07-21: Annotation-field diagnostic correction

- Corrected the product facility diagnostic to use the contract field `visible_facilities`; 128/200 annotations contain at least one visible facility and 72/200 contain a valid empty array. The earlier ad hoc `core_facilities` lookup was not a manifest field and its 200-empty conclusion was invalid.
- Clarified that product `price_range=unknown` is an allowed completed label when the image has no direct price evidence; it must not be described as missing annotation or automatically reopened for labeling.
- Checked all itinerary backups and submission audits. All 100 style arrays were empty at submission and all payload hashes still match, excluding later manifest overwrite as the cause.
- The retained design specifies 15 style choices, but the final recoverable vocabulary was compiled after the annotation submissions and the contemporaneous frontend assets were not retained. Recorded a probable historical field-exposure or serialization defect, not annotator omission; no product annotation was reopened and no semantic gold was generated.

## 2026-07-21: Project Control frozen-v1 final route

- Project Control completed route review and selected immutable `week3_evaluation_v1` for final delivery review.
- Product annotation remains closed; valid `unknown` values are not omissions. No `week3_gold_v2`, annotation-UI repair/reopening, supplemental annotation, or v2 rescoring is authorized.
- Itinerary image-style preference, after-sales facility-damage, and baseline natural-language semantic metrics remain `PENDING` according to actual support. Week 3 remains `PARTIAL`.
- This entry records a final scope boundary, not a future plan. Frozen manifests, raw runs, baseline/standardized Prompt assets, and Schema v1 remain unchanged.
