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
configuration-driven runner, metrics, and reporting. The v2 human-authored
manifests, full baseline, full standardized run, scores, and comparison are
validated. The approved `baseline_semantic_coding_v1` track now supplies the
previously missing baseline business metrics without changing the raw run.
Week 3 is `READY / COMPLETED`.

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
- [x] Record the historical Project Control frozen-v1 decision (superseded on 2026-07-22 by the mentor-authorized v2 recuration route).
- [x] Receive Project Control approval of the final actual diff and evidence boundary.
- [x] Generate and verify gold-independent deterministic baseline semantic metrics with explicit support counts.

Current v2 verification contains 450 completed annotations and 450 exclusion
rows. Both v2 completed runs pass artifact validation and bind tested counts to
their 450 persisted records. The final repository test result is recorded in
the verification evidence below.

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
`visible_facilities` is non-empty for 128 samples and empty for 72. The v2
itinerary style supplements and representative after-sales replacements are
complete. Evidence-supported `unknown` gold values remain valid and reduce
only the corresponding metric support. Week 3 is `READY / COMPLETED`.

### Verification Evidence and Boundaries

- Synthetic/mock framework verification: PASS，不属于真实模型 baseline，不计入 tested_count。
- `stage3_dry_run_20260713_001`: `baseline_minimal_v1`, `selected_count=0`, `record_count=0`.
- `stage3_dry_run_20260713_002`: `baseline_minimal_v1`, `selected_count=0`, `record_count=0`.
- Both dry-runs belong to Stage 3 and validate only the zero-selection framework path.
- 2026-07-14 `/v1/models` 探测成功，返回 `Qwen/Qwen2-VL-2B-Instruct`；未发送 Week 3 图片请求，未产生模型输出或延迟指标。
- Runs `week3_baseline_full_20260721_003` and `week3_standardized_full_20260721_001` each retain 450 records and pass restored-manifest provenance validation.
- Historical comparison `week3_prompt_pair_strict_20260721_001` remains optional traceability evidence and is not a Week 3 completion gate.
- Baseline semantic task metrics are stored under `baseline_semantic_coding_v1`; invalid natural-language JSON remains a separate 0% format result.
- The current status and data defects are documented in `reports/week3_zero_shot_baseline_report.md`.
- Full unit suite: 226/226 passed on 2026-07-25.
- Standalone v2 validation: `status=ok`, exclusion count 450, target/candidate/annotated/validated counts 200/150/100.
- Baseline and standardized v2 run-bound validation: both `status=ok`, with tested counts 200/150/100.
- Semantic score read-only verification: 450 rows, scenario counts 200/150/100, explicit support columns present, strict JSON serialization valid, and baseline JSON/Schema values unchanged at 0%.

The standalone status report is
`reports/week3_zero_shot_baseline_report.md`.

### Review boundary

Project Control approved the historical frozen-v1 `PARTIAL` commit on `dev`.
After the mentor-authorized v2 work, the user approved the final
`baseline_semantic_coding_v1` completion task and explicitly authorized safe
promotion through `dev`, `stg`, and `main` after all gates pass. No tag is
created.

### Active v2 recuration evidence (2026-07-22)

| Scenario | target_count | candidate_count | annotated_count | validated_count | tested_count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product understanding | 200 | 200 | 200 | 200 | 200 |
| After-sales issue recognition | 150 | 150 | 150 | 150 | 150 |
| Itinerary constraint understanding | 100 | 100 | 100 | 100 | 100 |

- [x] Preserve v1 manifests, runs, Prompts, and Schemas.
- [x] Create isolated v2 candidates, registry, preparation log, and annotation packets.
- [x] Remove 70 low-evidence after-sales rows and restore all four candidate strata with public and synthetic sources.
- [x] Expose only the previously unavailable itinerary-style choices and inherit all v1 constraint fields unchanged.
- [x] Add standardized v2 response contracts and verify representative product, after-sales, and itinerary JSON/Schema paths.
- [x] Replace the 70 abstract after-sales candidates with representative evidence, then complete their submissions.
- [x] Complete all 100 itinerary style-only supplements with audit/reconciliation hash coverage.
- [x] Run and sign full same-set v2 baseline and standardized evaluations.
- [x] Score actual v2 outputs and replace interim counts with run-bound evidence.
- [x] Generate the 450-row paired comparison and evidence-backed report.
- [x] Preserve the strict format track and add the independent deterministic baseline semantic score.

All mentor-required Week 3 baseline metrics now have persisted values and
support counts. No numeric semantic score is derived from JSON failure.

## Week 4 交付：Prompt 优化与 Milvus

状态：`READY / COMPLETED`，交付位于 `dev`。不进入 `stg`，不打标签。

### 完成清单

- [x] 保持 Week 3 manifest、金标、baseline、输出、Prompt、Schema 和评分不变。
- [x] 三个场景各固定选择 5 个正例和 2 个边界例。
- [x] 构建并实测 `standardized_v2`、4-shot v2、7-shot v2 pilot；
  三组均无模型请求错误。
- [x] 使用业务质量、格式、token 和延迟选择每场景胜出版本。
- [x] 只对三个胜出版本执行 450 条 Week 3 v2 全量跑测。
- [x] 保存原始输出、哈希、token、延迟、评分和 bad case。
- [x] 提供不修复内容的 JSON/Schema 格式兜底和定向测试。
- [x] 部署带持久化、健康检查和资源限制的固定版本 Milvus standalone。
- [x] 实现并验证批量入库、单条新增、过滤检索、删除和索引构建。
- [x] 生成 20 条真实 CLIP 向量并完成 CRUD 与性能测量。
- [x] 移除跟踪文件中的明文凭据，提交脱敏环境模板并完成本地凭据轮换。
- [x] 修复 LF/CRLF 运行绑定问题，新增 Week 4 统一只读证据验证。
- [x] runner/验证器拒绝模型请求失败；删除跨评分轨道业务差值。
- [x] 通过 244 个单元测试和全部交付验证。

实测 Prompt 和 Milvus 结果记录在
`reports/week4_prompt_optimization_report.md`,
`reports/week4_bad_cases.md` 和
`reports/week4_milvus_deployment_performance_report.md`。生成的运行、
向量和数据库 volumes 均保持忽略。

### 修复后验证证据

- `python -m unittest discover -s tests -v`：244/244 通过。
- Week 3 v2 数据验证和 baseline/standardized 两个 run-bound 验证均为
  `status=ok`，tested_count 为 200/150/100。
- `python scripts/validate_week4_delivery.py`：3 个有效 pilot 共 45 条且
  请求错误为 0；全量 450 条、score 450 条和 bad case 269 条全部通过。
- baseline token 未记录，明确为 `PENDING`；baseline 与 Week 4 业务指标
  使用不同预测编码轨道，不计算 `business_quality_delta`。
- Compose 脱敏展开通过；凭据轮换后 etcd、MinIO、Milvus 均 healthy，
  19 条逻辑可见向量保持可查询。
- 基准脚本会在既有输出或非空集合上写入前失败，并以 Milvus `count(*)`
  记录插入和删除后的真实可见行数。

## Promotion Rule

Weekly work is implemented and verified on `dev`, promoted unchanged to `stg`
for mentor review, and promoted from `stg` to `main` only after approval. A
completed checklist is updated on `dev` before promotion so all downstream
branches inherit the same delivery state.
