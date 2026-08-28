# Weekly Delivery Record

This is the canonical weekly delivery document. Each week keeps its original
scope, acceptance checklist, measured results, and known limitations. New
weeks append sections instead of replacing earlier delivery history.

### 2026-08-19 Week 6 训练后质量审计与整理

- [x] 核验 Week 6 训练、门禁、冻结评测、归档与 `stg` 证据完整。
- [x] 如实区分工程完成度和业务效果，未把低支持或弱业务指标描述为优秀。
- [x] 使用一手资料完成前沿方法评审，并把方法映射到商品、售后、行程的实际弱项。
- [x] 接受 ADR-030 和非污染门禁；现有冻结集不再用于调参。
- [x] 更新报告索引、requirements、decisions、experiments、weekly log、README。
- [ ] 新 development/test 锁、后续 SFT/约束解码/DPO 消融尚未执行，因此没有新指标。

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

状态：`READY FOR MENTOR REVIEW`。Milvus 与 Prompt 交付均完成；新
Few-Shot 证据来自独立 development 人工金标池。旧 test-gold 运行保留为
历史证据，不参与新选择。

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
- [x] 新增不覆盖旧评分的 450 对共同确定性语义评分与 paired bootstrap。
- [x] 旧 Few-Shot pilot 明确标为 test-gold contamination 历史证据。
- [x] 建立 36 条独立 `week4_demo_dev_v1` 人工金标并验证与 450 条最终
  evaluation 在样本、来源、图片和来源组四层隔离。
- [x] 使用 selection v2 重跑 45 条无偏 pilot；商品胜出 4-shot，售后和
  行程胜出 `standardized_v2`。
- [x] 只按 pilot 胜出映射执行新的 450 条全量，不用全量结果反向改选。

实测 Prompt 和 Milvus 结果记录在
`reports/week4_prompt_optimization_report.md`,
`reports/week4_bad_cases.md` 和
`reports/week4_milvus_deployment_performance_report.md`。生成的运行、
向量和数据库 volumes 均保持忽略。

### 修复后验证证据

- `python -m unittest discover -s tests -v`：248/248 通过。
- Week 3 v2 数据验证和 baseline/standardized 两个 run-bound 验证均为
  `status=ok`，tested_count 为 200/150/100。
- `python scripts/validate_week4_delivery.py`：3 个有效 pilot 共 45 条且
  请求错误为 0；全量 450 条、score 450 条和 bad case 376 条全部通过。
- baseline token 未记录，明确为 `PENDING`；原词法轨道与原结构化轨道
  仍不直接相减。新增共同语义轨道包含 450 对预测、38 个聚合指标和
  2,000 次 paired bootstrap。
- Compose 脱敏展开通过；凭据轮换后 etcd、MinIO、Milvus 均 healthy，
  19 条逻辑可见向量保持可查询。
- 基准脚本会在既有输出或非空集合上写入前失败，并以 Milvus `count(*)`
  记录插入和删除后的真实可见行数。

2026-07-26 本次复核中，使用临时脱敏环境变量执行 Compose 配置展开通过；
本次已启动 Docker daemon 和固定 Qwen2-VL vLLM，仅执行 Prompt 推理，
未与 CLIP 并发。Milvus、MinIO、etcd 当前均 healthy；正式集合 19 条逻辑
可见行，临时集合 CRUD 为插入 1、命中 1、删除后 0。历史性能产物未改写。

## Week 1-4 Qwen3.7 整体交付整理

- [x] 云端模型配置固定为阿里云百炼 `qwen3.7-plus`，关闭 thinking。
- [x] Week 1 真实图片 API smoke 通过，且禁止请求失败时静默 fallback。
- [x] Week 2 数据与 CLIP 结果确认为模型无关项，不重复制造运行数据。
- [x] Week 3 baseline、standardized 和 Week 4 winner 均完成 Qwen3.7
  450/450 全量运行，请求错误为 0。
- [x] 行程 `standardized_v4` 完成 100/100，JSON/Schema 均为 100%。
- [x] 保留 Milvus 真实 CRUD、HNSW 和小规模性能证据。
- [x] 新增总报告与报告索引，明确纯模型变化、Prompt 联合修复和已知限制。
- [x] stg 推广范围固定到 Week 1-4，不包含后续 Week 5 实现。

总报告：`reports/week1_to_week4_qwen37_overall_report.md`。
报告索引：`reports/README.md`。

### 整理后验证证据

- `python -m unittest discover -s tests -v`：262/262 通过。
- `python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml`：
  `status=ok`，无 errors 或 warnings。
- Qwen3.7 baseline、standardized 和行程 v4 三个 run-bound 验证均为
  `status=ok`；exclusion count 为 450。
- `python scripts/validate_week4_delivery.py --config
  configs/evaluation_week4_qwen37_plus_aliyun.yaml`：`status=ok`，pilot 45、
  full/score/common-semantic paired 均为 450，请求错误为 0。

## Week 5 交付：数据集标注与质检

状态：`PARTIAL / WAITING FOR REQUIRED HUMAN INPUT`。候选池、隔离、Schema、
标注配置、预标注/人工修正/三级质检/对话脚本已交付；真实模型预标注和人工工作
尚未发生。

### 实际交付与数量

- [x] 三场景标注规范与当前 accepted 推理 Schema 对齐。
- [x] 多模态 JSONL 标注配置和字段自动校验规则。
- [x] 样本池构建、两版 exclusion manifest 隔离和重复图片拒绝。
- [x] 商品 50,000、售后 20,000、行程 10,000 候选池实际生成并验证。
- [x] Qwen3.7 最优 Prompt 映射、批量预标注、断点续跑和失败记录。
- [x] 人工 revision、自审、同场景交叉互审及确定性抽检能力（原 10%/5% 方案已由
  2026-08-10 单人最小人工方案替代）。
- [x] 三类多轮对话 Schema、生成、结构校验和人工五项质检。
- [x] 实际导出商品/售后/行程 50,000/20,000/10,000 条本地人工包；包内明确标记
  预标注缺失，生成文件保持 Git 忽略。
- [ ] 模型预标注：商品真实 smoke 3/3 完成、失败 0；其余全量仍在执行范围内。
- [ ] 人工修正与三级质检：0；未收到必要人工输入。
- [ ] 最终合格单轮：商品/售后/行程均 0。
- [ ] 对话候选与最终合格：均 0；不能使用评估集绕过合格单轮前置条件。

### 隔离与分布证据

- 80,000 个唯一 `sample_id` 和 80,000 个唯一图片 SHA-256。
- exclusion 共加载唯一 source/image 520、group 520、constraint template 12；
  最终冲突 0。
- 商品：酒店/景点/餐饮 200/800/49,000。
- 售后：公开/合成 5,552/14,448，四类问题路由各 5,000。
- 行程：四类人群各 2,500；预算 3,336/3,332/3,332；天数
  3,336/3,336/3,328。

详细规范和实测报告分别为 `docs/week5_annotation_guidelines.md` 与
`reports/week5_dataset_quality_report.md`。大型生成产物保持 Git 忽略。

验证：Week 5 定向测试 15/15、完整 `unittest` 285/285、样本池/隔离校验、三个
新增 JSON 配置解析和 `git diff --check` 均通过。

### 2026-08-09 增量证据

- [x] Project Control 六项裁决已同步到 requirement、decision、配置、Schema、代码和测试。
- [x] workflow v2 sidecar 绑定现有 80,000 条不可变候选，三场景数量为
  50,000/20,000/10,000，初始人工状态均为 `awaiting_human_annotation`。
- [x] 候选池全量图片、哈希、重复与冻结评测集隔离复核返回 `status=ok`；
  唯一 sample/image 均为 80,000。
- [x] 行程配对 pilot 使用新 run ID 完成 30×2=60 请求；保留 60 份原始输出和
  2 条 Schema 失败，选中 `standardized_v4`，估算计算费 CNY 6.09。
- [x] 从胜出 Prompt 导出 30 条版本化行程人工任务，状态全部为
  `awaiting_human_annotation`，未填写或伪造 annotator/reviewer。
- [ ] 三场景全量预标注未获授权；当前只有历史小 pilot 与本次 30 条行程配对 pilot。
- [ ] 真实人工修正、自审、交叉互审、核心抽检与最终 accepted 均为 0。
- [ ] 多轮对话候选和人工 accepted 均为 0；未执行未获授权的批量生成。

### 2026-08-10 单人质检增量证据

- [x] 全局规则、Week 5 requirement、accepted decision、操作规范、标注工具配置、
  数据集配置和 README 已统一为单人最小人工模式。
- [x] 人工修正提交必须包含真实 `annotator`、`corrected_at`、独立
  `review_session_id` 和显式 `self_review_confirmed=true`；保存时生成同一 revision
  的真实内联自审记录，不允许模型或 Agent 自动确认。
- [x] 商品交叉复核/核心抽检为 1%/0.5%，售后和行程为 2%/1%；核心集合固定嵌套于
  交叉复核集合，未抽中的样本在真实修正和内联自审后无需额外重复审核。
- [x] 对现有 80,000 个候选实际计算出的盲复核/核心抽检集合为商品 516/259、售后
  399/219、行程 190/94；共 1,677 次额外阶段操作，均尚未计为人工完成。
- [x] 同一操作者可执行后续复核，但必须使用不同审核会话；程序拒绝未抽中样本、
  会话复用、缺失人工身份和跳级状态。
- [x] `export-quality` 只导出当前阶段准备完成、确定性选中且尚未记录的任务，避免
  人工筛选和重复审核；输出包不预填人工决定、身份、时间或会话 ID。
- [x] 活动 GPU 预标注配置保持不变；新增独立的
  `configs/week5_dataset_qwen3_vl_4b_single_operator.json` 仅用于后处理。
- [x] Week 5 定向测试 19/19、完整 `unittest` 289/289、Week 3 v1/v2 隔离验证、
  配置解析和 `git diff --check` 通过；活动 run 配置 SHA-256 与 manifest 一致。
- [ ] 真实人工修正、自审、盲二次复核、核心抽检与 accepted 数量仍为 0；规则精简
  不改变“必须有真实人工确认”的事实边界。

### 2026-08-10 低于 500 次质检与预标注进度证据

- [x] 当前单人质检比例调整为商品 0.2%/0.05%、售后和行程 0.5%/0.1%。
- [x] 对现有候选的确定性实算集合为商品 112/26、售后 102/21、行程 53/7，合计
  321 次额外阶段操作，满足低于 500 次的要求；它们尚未计为人工完成。
- [x] 全量预标注 run B 实际成功商品 8,140，未解决失败 8；商品完成 16.28%，全池
  完成 10.175%，售后和行程全量完成数均为 0。
- [ ] run B 当前未运行：checkpoint 停在 2026-08-10 02:10:24（悉尼时间），本机
  Python/SSH 进程与回环隧道均不存在。产物可按原 manifest 哈希安全 resume。

### 2026-08-12 自动恢复增量证据

- [x] 根因确认为本机 SSH 隧道退出；远端 vLLM 健康，run 数据与 manifest 未损坏。
- [x] `scripts/supervise_week5_preannotation.ps1` 支持 keepalive、端点健康检查、互斥
  防重、隐藏后台运行、自动重连和相同 run ID 安全 resume。
- [x] 主流程不重复成功或反复重试 terminal bad case；仅在全池完成后做一次失败清理。
- [x] 守护进程恢复后 checkpoint 实际继续增长，首轮观测新增 48/48 成功且连续失败 0。
- [x] 已建立每 30 分钟本地 heartbeat 状态检查；全量完成前不自动停止或释放 ECS。
- [x] PowerShell 语法、守护流程定向测试 2/2、Week 5 联合定向测试 21/21、完整
  `unittest` 291/291 和 `git diff --check` 通过。

### 2026-08-12 ECS 常驻迁移增量证据

- [x] 80,000 条候选和 80,000 张唯一引用图片已同步，远端缺失 0；三份候选 manifest
  与活动 run manifest 哈希一致。
- [x] 远端 Week 5 定向测试 22/22、候选池与隔离验证 `status=ok`、回环端点验证通过。
- [x] 本地在 checkpoint 15,190 安全停止；断点 JSONL 逐行有效并绑定原 run identity。
- [x] systemd 服务已启用并从同一 run 恢复，成功数 15,166→15,197、checkpoint
  15,190→15,209；历史 raw 已用不覆盖模式补传。
- [x] 本地 supervisor、runner 和 SSH 隧道已停止，禁止与 ECS 服务并行写入。
- [ ] 全量预标注仍在运行；真实人工修正、三级质检和最终 accepted 未因迁移而完成。

### 2026-08-14 Week 5 闭环与 Week 6 数据锁定证据

- [x] 最终 merge 覆盖 80,000 个唯一候选：成功 79,936、最终失败 64，成功/失败互斥且
  无缺失；商品/售后/行程成功为 49,957/19,991/9,988。
- [x] 44 条不可读输入保持最终 `input_error`；其余失败为 Schema 19、JSON 解析 1，
  未伪造成功或人工结果。
- [x] 标注台已加载全量 Schema-valid 预标注，并原样保留 27 条真实人工修订。
- [x] Week 6 训练/验证数据版本及 manifest/split SHA-256 已锁定；模型预标注权重 0.5，
  仅 27 条真人修订权重 1.0；冻结 Week 3 样本未进入训练或验证。
- [x] 六份锁定 JSONL 通过流式数据契约校验；完整 unittest 312/312、Week 5 候选池与
  隔离验证通过。
- [ ] Week 6 GPU pilot、正式训练和参数锁定后的 Week 3 最终评估尚未完成。

### 2026-08-18 Week 6 固定训练链终态与专项优化增量

- [x] `29312210`、`29312212`、`29312214`、`29312215`、`29312217` 均
  `COMPLETED 0:0`，代码和 Spartan checkout 固定为
  `3d6bc81df8c4afd496e1e78d41c6b4bfa07c7bf4`。
- [x] 三场景 `run_summary` 均为 completed、adapter-only、磁盘回载已验证；最佳
  checkpoint 为商品 5930、售后 2856、行程 1620，并完成 Spartan、本地 E 盘和
  `trip-api-sg` 三处 SHA-256 归档核验。
- [x] 对行程锁完成确定性业务结构审计；原 validation 全通过率 `0/450`，因此没有把
  `eval_loss=0.005681941285729408` 误报为业务效果优秀。
- [x] 构建不可覆盖派生 silver 锁 `week6_itinerary_structural_repair_20260818_v1`；
  train/validation 为 `9538/450`，结构审计全部通过，未改写原锁、外部事实或人工身份。
- [x] 增加现有 adapter 基线评估、候选 provenance 校验和同集非回退比较门禁；本地
  定向测试 31/31、Python 编译、Slurm shell 语法及差异检查通过。
- [x] 派生 validation 基线、候选训练与同集业务 comparison 已取得终态；候选九项
  结构计数 64/64 且门禁无回退。随后才执行冻结 Week 3 最终评测，未反向用于调参。

### 2026-08-19 Week 6 专项与最终评测交付证据

- [x] refinement `29375367` 完成并回载 `checkpoint-540`；固定 64 条候选九项结构
  计数均为 64/64，comparison `29412603` 无回退原因并正式晋级。
- [x] 冻结 `week3_evaluation_v2` validator `29418805` 为 450/450；CPU preflight
  `29418839` 的 77 项测试、离线模型缓存及三个 adapter SHA-256 全部通过。
- [x] 商品/售后/行程最终评测 `29418875`/`29419327`/`29422130` 均
  `COMPLETED 0:0`，严格串行且未提交竞争 GPU 作业；样本分别为 200/150/100。
- [x] JSON/Schema 分别为商品 100%/100%、售后 100%/96.67%、行程 95%/85%；
  业务指标和已知局限已写入 `reports/week6_qlora_quality_report.md`，没有用冻结结果
  继续调参。
- [x] 最终代码完整测试 370/370 通过，`dev` 与 `stg` 已同步收尾提交。版本化增量归档
  `week6_quality_closeout_20260819_524a30c` 已在 Spartan、本地 E 盘和
  `trip-api-sg` 三处保存；三处均按 `SHA256SUMS.final` 校验 69 个交付文件通过，
  清单本身 SHA-256 为
  `8c1ac916409d2446bd0b80f2a70ebec92747ca25df93b00a5ebf997b75856b7c`。
- [x] 依据 2026-08-14 最新单人预算决策，将人工验收限制为三场景各 100 条，并在每个
  队列内固定包含 10 条盲复核候选与 3 条核心抽检候选；另预留 100 次自动对话候选
  人工验收，总预算 439 次，未把其余 silver 记录改写为人工完成。
- [x] 当前真实人工修订和内联自审为商品/售后/行程 100/100/100；预算内首轮队列
  已清零。三场景各完成 10 次盲复核和 3 次核心抽检，review session 互异且记录有效。
- [ ] 多轮对话候选及人工 accepted 尚未完成；候选生成和人工验收数量必须分开报告。
- [x] 三场景各 100 条 canonical annotation 均通过 Schema；中文展示镜像各 100 条，
  以 canonical SHA-256 绑定且不覆盖训练数据。
- [x] 对话生成支持严格 run identity 和显式 `--resume`，Spartan sbatch 通过语法检查；
  目标为 10,000 条自动候选。唯一作业 `29226849` 已提交到 `gpu-l40s`，首次状态为
  `PENDING (Priority)`；尚未产出候选，不能计为完成或 accepted。
- [x] 300 条人工/QC 归档已在 Spartan 项目目录校验并安装；旧 27 条全部被新记录
  包含，安装前副本可恢复，六个 JSONL 的 SHA-256 与本地一致。
- [x] 本轮完整 `unittest` 319/319、定向测试 7/7、`bash -n` 和
  `git diff --check` 通过。
- [x] 修复 vLLM 容器 `python3` 入口，并将容器 HOME、XDG 与 FlashInfer 缓存绑定到
  Trip 专属 GPFS，避免使用已满的 Spartan home。
- [x] 历史失败 `29114276`、`29116649`、`29116828` 均保留为不可覆盖证据；未生成
  benchmark 结果，也未提交剩余分片。
- [x] `29116943` 已验证 vLLM health 200；随后发现并补齐缺失的 36 条 Week 4
  development manifests 与 36 张引用图片，容器内校验三场景各 12 条通过。
- [ ] 唯一替代 benchmark `29117353` 当前在 `gpu-l40s` 等待资源；须实际完成并通过
  身份、哈希、成功率和吞吐核验后才能提交剩余分片。

### 2026-08-12 展示部署与提交身份增量证据

- [x] `trip-api-sg:/opt/trip-display/20260812a` 独立部署完成；部署包 SHA-256 为
  `404e7a681bdf35a839de56298568960a950203a21d9f7ae61b7dac4fdbe8a81d`。
- [x] `ota-trip-display-api` 在 `127.0.0.1:8010` healthy，`/v1/project-status` 和 Week 5
  静态报告可读；原 `ota-trip-api` 在 `127.0.0.1:8000` 仍 healthy。
- [x] 未安装 CUDA、vLLM 或模型权重，未开放公网展示端口，未覆盖原服务。
- [ ] Spartan Slurm 尚未提交：门户只读核验的当前身份为第三方账户 `yzhang3504`，不满足
  ADR-020 的代理提交要求；project、quota、scratch 和 partition 仍待账户所有者或用户
  自有身份核验。

> 后续用户已更正账户归属：`yzhang3504` 为本人持有并授权 Trip_Project 使用。允许代理
> 核验和提交，但仍须先完成 project/quota/scratch/partition 实测，并只操作新建项目目录
> 和本项目 job ID。上方条目保留为授权更正前的事实快照。

- [x] 实测 account/project=`punim2936`、QOS=`publicgpu`、project GPFS 可写且约余
  93 GiB；home quota 已满，因此所有 Trip 文件均进入新建专属 project 目录。
- [x] 依据实测待排数量只选择 `gpu-l40s`；benchmark job `29109265` 已提交并处于
  `PD(Resources)`，估计启动 `2026-08-12T20:27:34`。没有提交 H100/A100 重复作业。

### 2026-08-12 Spartan 接管增量证据

- [x] 删除已失效的 A10 两小时监控；未释放实例或云盘。
- [x] 从本地真实 15,166 条成功恢复点生成不可覆盖 Spartan migration；100 条
  benchmark 加 4 个互斥分片覆盖剩余 64,834 条，连同恢复点合计 80,000。
- [x] 增加分区只读检查、H100/A100/L40S 单分区提交模板、回环 vLLM、checkpoint、
  状态统计和严格合并验证。
- [x] 增加 Week 6 Qwen3-VL-8B NF4 QLoRA 配置、锁定数据契约、环境检查和最多
  10-step 的小样本 pilot 入口。
- [x] 增加包月 CPU ECS 结果展示 Compose 和只读状态 API；不承担 GPU 推理。
- [x] Spartan account=`punim2936`、project GPFS、Python/Apptainer 模块和提交身份已
  实测；旧 benchmark `29109265` 因模块依赖失败后，修复版环境作业 `29114275` 与唯一
  L40S benchmark `29114276` 已提交；环境作业已 `COMPLETED 0:0`，benchmark 当前
  在 `spartan-gpgpu006` 上 `RUNNING`。
- [ ] Week 5 全量预标注仍未完成；人工修正、三级质检、最终 accepted 与对话 accepted
  仍为 0。
- [ ] Week 6 GPU pilot 和正式训练均未运行，不能计为训练完成。
- [x] 验证：完整 unittest 299/299；Week 5 候选/隔离、Week 3 v1/v2、Slurm shell
  语法、展示 Compose 展开和 `git diff --check` 通过。本机 Week 6 环境如实返回缺少
  GPU 训练依赖，未计为 Spartan 环境通过。
- [x] 新增 project-scoped Python 3.11 venv 安装作业；环境及全部缓存仅写入 Trip 专属
  版本目录。job `29114275` 完成，`pip check` 和关键包导入通过。GPFS 实测可用
  93 GiB；未使用 home 或其他成员目录。

### 2026-08-12 ECS 常驻迁移增量证据

- [x] 80,000 条候选和 80,000 张唯一引用图片已同步，远端缺失 0；三份候选 manifest
  与活动 run manifest 哈希一致。
- [x] 远端 Week 5 定向测试 22/22、候选池与隔离验证 `status=ok`、回环端点验证通过。
- [x] 本地在 checkpoint 15,190 安全停止；断点 JSONL 逐行有效并绑定原 run identity。
- [x] systemd 服务已启用并从同一 run 恢复，成功数 15,166→15,197、checkpoint
  15,190→15,209；历史 raw 已用不覆盖模式补传。
- [x] 本地 supervisor、runner 和 SSH 隧道已停止，禁止与 ECS 服务并行写入。
- [ ] 全量预标注仍在运行；真实人工修正、三级质检和最终 accepted 未因迁移而完成。

### 2026-08-16 Week 5 多轮对话最终交付证据

- [x] 权威合并 run `week5_dialogues_merged_10000_20260816_522b4af` 包含 10,000 个
  唯一对话，索引 0–9999，三场景 3334/3333/3333，消息数 8–12。
- [x] Schema、严格角色交替、图片引用、配置与 qualified 集合哈希通过；
  duplicate/conflict/missing 均为 0。
- [x] candidates/manifest SHA-256 分别为
  `7e00f326fc1b2896a6efcc5c2f6c1f67ffdb728501ba3eb9ba65efdb28265d99` 与
  `02795c8df44ca564dcd873974c5bcb6939c41bf38bee2f6c1f550d7916669556`，本地同步
  JSONL、gzip 与 manifest 哈希和远端一致。
- [x] 固定 100 条人工队列全部完成：100 个队列内唯一 ID，reviewer 非空，五项
  checks 完整，100 条 decision 均为 `pass`；人工验收 JSONL SHA-256 为
  `eb3a6f436a78389e919b86d3756fc2208265bac7f4420158dc597d5bc4682e54`。
- [x] 仅 100 条抽样对话计为人工 accepted；其余 9,900 条未被伪装成人工验收。
- [x] 干净 checkout 完整 unittest 330/330、Week 5 `validate-pools`（80,000 个唯一
  sample/image，
  `status=ok`）和 `git diff --check` 通过。
- [x] `report --dialogue-run-id week5_dialogues_merged_10000_20260816_522b4af`
  从权威 run-scoped 目录实测返回候选 10,000、人工校验 100、最终合格 100；裸
  `report` 会拒绝运行，避免旧固定目录再次静默报告为 0。
- [x] 已跟踪测试实际依赖的 4 份 Qwen3-VL-4B 脱敏配置/示例；干净 checkout 不再
  依赖开发机未提交文件。
- [x] Week 5 按 ADR-023 的单人预算口径闭环；未运行 Week 6 pilot、训练或评估。

## Promotion Rule

Weekly work is implemented and verified on `dev`, promoted unchanged to `stg`
for mentor review, and promoted from `stg` to `main` only after approval. A
completed checklist is updated on `dev` before promotion so all downstream
branches inherit the same delivery state.

## Week 7 Delivery Status（2026-08-21）

状态：`CORE_AUTOMATED_ACCEPTED / CORRECTED_DIALOGUE_DEV_HUMAN_COMPARISON_COMPLETED / V3_TEST_DIALOGUE_INVALID`。
三个核心场景自动门禁与一次性 final-test 已完成；v3 对话缺陷保留为历史事实，新的
development-only 修复队列及 Week 6 routed 配对人工评分均已完成。

- [x] 从指定 Week 6 终态提交创建隔离分支，旧工作树保持不变。
- [x] v3 train/development/test 锁通过 sample/source/image/group/template 五维隔离；
  test 未运行，对话父任务在三场景间为 150/150/150、8/8/8、8/8/8。
- [x] 固定实际混合比例：核心场景各 760、通用正则 9%、多轮对话 15%，工具调用占
  对话 10%。
- [x] 固定 QLoRA/SFT、结构感知截断、完整 development 生成评估、checkpoint 和早停实现。
- [x] 固定 Schema constrained decoding 的 format-only 对照实现，禁止语义提升结论。
- [x] 对话 24 条人工队列保持空白，未由 Agent 代填。
- [x] DPO 初始门禁为 `SKIPPED`；真实双模型评分完成后确定性派生并审计 16 对，锁为
  10 train/6 validation。唯一 job `29491859` 完成，但 validation 0.3333/-0.00981
  未通过 0.5/>0 门禁；新 adapter 未选用、未重试、未触碰 test。
- [x] 完整 unittest 428/428、compileall、锁验证、十份 shell 语法和 diff 检查通过。
- [x] 终态机器优先对抗审计 11/11 个反事实被拒绝；完整 unittest 更新为 431/431。
- [x] 作废 v1/v2、失败构建与临时文件 408,127,632 字节进入回收站；v3 和全部有效证据保留。
- [x] Week 6 adapters 与零样本的完整 114 条 development 基线已生成并哈希绑定。
- [x] Schema 自由/受约束实际对照完成；constrained primary 90/90 请求失败，free
  fallback 90/90 成功，生产模式锁定 free，未宣称语义提升。
- [x] 多任务训练在 step 151 早停并正常完成，四个 checkpoint/raw/metrics 已保存和哈希
  绑定；v3 训练时历史最高综合分 step 76 为 0.869412。
- [x] 独立 protocol-v4 公平重评仅重跑 development 评估，继续绑定 v3 配置、数据锁和
  checkpoints，未创建 v4 训练或 v4 数据。attempt 1 作业 `29449140` 被取消，不完整
  产物排除于评分；attempt 2 作业 `29449999` 为 `COMPLETED`。
- [x] protocol-v4 候选 step 38/76/113/151 的综合分为
  0.258513/0.723404/0.746154/0.733077，全局延迟比为
  1.6312/1.3221/1.3155/1.4894；该历史 attempt 保持 `eligible_count=0`。
- [x] protocol-v5 继续绑定 v3 数据/训练身份，在同一 allocation 统一 BF16/static cache/
  compile/warm-up/gold-support。有效 job `29456896` 完整结束；step
  38/76/113/151 延迟比为 1.0405/0.9417/0.8609/0.8809，4/4 eligible，选择 step 151。
- [x] 参数锁绑定 checkpoint-151 与完整 v5 runtime；唯一 final-test job `29459265`
  `COMPLETED 0:0`，marker 和 7 个 artifact hash 通过，`all_passed=true`。
- [x] test 统一模型商品/售后/行程 composite 为 0.153846/1.000000/0.996667，
  Week 6 路由基线为 0.056410/0.100000/0.028333，zero-shot 为
  0.076923/0.100000/0.050000；所有场景、支持、JSON/Schema、延迟和失败率门禁通过。
- [x] test 对话历史自动输出为格式合规率 1.0、字符串包含式上下文召回率 0.878472；
  后续审计确认该 scorer 不检测 assistant/user 语义顺序，不能作为真实多轮能力结论。
- [x] 标注台对固定队列执行上下文完整性门禁，24/24 标记为
  `BLOCKED_INVALID_SOURCE_CONTEXT`，前后端均禁止保存无效人工分数并提供逐条错位说明。
- [x] 新建 `week7_dialogue_review_20260821_v2`，24 条均为正确 user→assistant 语义顺序，
  5–8 轮各 6 条、图片仅首次用户轮；未修改 v3 锁、旧 raw 或 test marker。
- [x] checkpoint-151 修复队列 GPU run `29479822` 为 `COMPLETED 0:0`，24/24 成功、
  失败 0；两次 PENDING 取消和两次真实失败均已记录，未伪装为成功。
- [x] 标注台在评分前为 `READY_FOR_REAL_HUMAN_INPUT`，无效上下文 0/24，具有逐轮对齐
  与四维扣分辅助；后端要求真实身份和本人自审。
- [x] corrected development 对话人工四维完成 24/24：同一真实操作者和会话、自审 24，
  最终 24 `pass`；四维均分 4.541667/4.625000/4.500000/4.708333。Agent 未代填分数。
- [x] 人工结果 SHA `bdec2d18...af932`，26 条 append-only 记录含 2 次真实修订；原记录
  保持在忽略目录，Git 仅提交聚合与哈希证据。
- [x] Week 6 三 adapter 在同一 corrected development 上按 8/8/8 路由；job `29491047`
  完成 24/24、失败 0，raw SHA `c3effb6d...318e59`，未读取 test。
- [x] Week 6 routed 对话的配对人工四维评分完成 24/24：总均值 4.56250，结果 SHA
  `af3721d2...d49f93`；multitask 总均值 4.59375，配对差 +0.03125、10 胜/7 平/7 负。
  评分由同一真实操作者重新输入，未复制 multitask 分数；小差异不作显著提升结论。
- [x] 代码提交 `64a5a7a`、final runtime 修复 `8619b76` 与对话修复提交已推送；本次
  人工聚合证据和状态修复由收尾提交推送。终态对抗审计允许实现进入 `dev` 集成，
  但因 v3 test 对话构造缺陷不可逆，完整 Week 7 能力结论仍未通过；不进入 `stg`，
  不打标签。

### Corrected-dialogue v4 执行增量（2026-08-22）

- [x] 接受 ADR-031，保留 v3 失效历史证据，新建 v4 全量 train/development/test
  身份与独立 run IDs。
- [x] v4 锁 3000/114/114 实际构建并验证；训练比例 9% 通用正则、15%
  对话，三分区五维冲突 0。
- [x] v4 test 与 v3 完整 3228 行 identity manifest 五维重叠 0；test 仍
  `LOCKED_UNCONSUMED`。
- [x] 所有 assistant span 参与 SFT loss；development/test 实现逐轮生成和机器
  sequential 门禁，未冒充人工评分。
- [x] 实现严格 checkpoint selector 和一次性 corrected-dialogue test runner，包含原子
  marker、全候选证据重算与 Week 6 routed/zero-shot 对比。
- [x] v4 定向测试 10/10、Week 7 测试 71/71、完整 unittest 441/441、Python
  语法和两份 Slurm shell 语法通过。
- [x] attempt 1 `29504508` 在首次逐轮 development 评估因 assistant content block
  类型错误失败，无 checkpoint/raw/metrics；失败证据已保留且未消费 test。
- [x] attempt 2 前置修复验收：两条真实逐轮生成路径的 strict block 回归、v4
  13/13、Week 7 74/74、完整 unittest 445/445、数据锁、Slurm 语法和 diff 检查通过。
- [x] `29505375` 在完整 step-38 checkpoint 后无主堆栈地退出；同身份受控恢复
  job `29506065` 从该 checkpoint 提交，失败与恢复证据均完整保留。
- [x] `29506065` 在 step 76 暴露明确磁盘配额根因；已只回收可再生成的 26 GiB
  container cache、1.2 GiB pip cache 与 0-byte partial，冻结资产未动。恢复 job
  `29506362` 仍从完整 checkpoint-38 继续，未改变 config/data/run identity。
- [x] job `29506362` `COMPLETED 0:0`；六个 development checkpoint 及 raw/metrics
  全部写出并由 run summary 哈希绑定，step 151 综合分最高（0.833980），step 226
  因 patience=2 早停。
- [x] `select-v4-checkpoint` 已实际重算六个候选并正确拒绝：0/6 通过全部自动门禁，
  因此未生成 selection。corrected-dialogue v4 test 按规则标记 `SKIPPED_GATE_FAILED`，
  consumption marker 不存在且 test policy 仍 `LOCKED_UNCONSUMED`，没有提交或重跑
  test，也没有 v4 Week 6/zero-shot test 对比。
- [ ] v4 完整验收与 `dev` 快进未完成：真实原因是 development 自动门禁 FAIL，
  不是缺少人工输入或未授权资源；本周不绕过门禁、不重训、不进入 `stg`、不打标签。

### Corrected-dialogue v4 fix1 执行增量（2026-08-23）

- [x] 新 config/data/run identity 已锁定并推送：commit `6bb5322`，fix1 canonical
  lock SHA-256 `7f66795c...9a7`，train/development/test=3000/114/114；v3、首版 v4
  与 fix1 三分区五维隔离复验通过，test 未消费。
- [x] 实际训练配比为商品/售后/行程 600/840/840、通用正则 270（9%）、多轮对话
  450（15%）；工具调用、gold-plus-anchor、metric-support 和 loss multiplier 均在运行前
  固定，未进行原地动态调参。
- [x] 首 job `29526506` 的错误 HF cache 环境失败如实保留；同 config/data/run/git
  identity 的安全恢复 job `29526965` 在新非覆盖目录 `COMPLETED 0:0`，02:43:24，
  六个 development checkpoint/raw/metrics 和 run summary 全部写出并哈希绑定。
- [x] best checkpoint-151 weighted composite=0.764049，核心三场景
  0.153846/0.970000/1.000000，对话 automatic=0.877630、失败率=0；最终 adapter
  SHA-256 `b42aeeb...5131bc`。
- [x] selector 实际执行并写出不可覆盖阻断证据：0/6 eligible，状态
  `BLOCKED_NO_ELIGIBLE_CHECKPOINT`，文件 SHA-256 `782e92ab...cc104`；最佳候选仅
  sequential coverage 0.725585 未达 0.75。
- [x] fix1 one-shot test 正确标记 `SKIPPED_GATE_FAILED`：没有提交 test job，
  marker 不存在，`test_read=false`，没有生成 Week 6/zero-shot fix1 test 指标。
- [x] DPO 保持既有唯一一次 validation 门禁失败后关闭，没有第二次消融或 adapter 晋级。
- [x] fix1 定向 26/26、Week 7 79/79、完整 unittest 450/450、config、数据锁、五维
  隔离、两份 v4 Slurm shell 语法和 diff 检查全部通过。
- [ ] fix1 完整验收与 `dev` 快进未完成，真实原因是 development 自动门禁 FAIL；
  不绕过门禁、不进入 `stg`、不打标签。

### Corrected-dialogue v4 fix2 修复增量（2026-08-24）

- [x] 保留 fix1 FAIL/test-unconsumed 证据不可变；接受 ADR-032，不降低既有阈值。
- [x] 修复嵌套结构顶层全等、silver caption 逐字 hard-gate 和训练/selector 目标错位；
  新协议与旧 fix1 由 config/data/run identity 隔离。
- [x] fresh fix2 锁 3000/114/114、五维隔离 PASS；v3、首版 v4、fix1 身份均排除，
  canonical SHA-256 `86a43601...e5b14b`；test 仅消费一次。
- [x] GPU 训练 job `29540085` 完成；selector PASS 并锁定 checkpoint-226（5/8 合格）。
- [x] 唯一 one-shot test job `29544969` 完成，24 条/角色；自动最终门禁 `FAIL`，10 项
  绝对阈值未达。没有重跑、没有人工代填、没有以相对基线提升替代绝对门禁。
- [x] 终态定向 54/54、完整 unittest 454/454、数据锁/五维隔离、config loader、两份
  Slurm shell 语法和 `git diff --check` PASS。
- [ ] fix2 模型验收未通过，真实原因是一次性 test 结果未达预注册门禁；不进入 `stg`、
  不打标签，DPO 保持关闭。
- [x] 长期分支仅 `dev`、`stg`、`main`，临时分支已清理。
- [x] 系统收敛代码：生产模式 fail-closed、三场景/对话/视觉检索接口、严格 `/ready`、
  统一 Compose、release manifest 与 `tripctl`。
- [x] Week 5 v2 候选池和 64 条修复队列实际生成；80,000 sample/source/image SHA 唯一，
  与冻结评测五维冲突 0；历史人工 accepted 不变。
- [x] fresh 修复锁 1,980/168/120，五维跨 split 冲突 0，test 未消费。
- [x] 1,000 张真实图片 CLIP/Milvus 基准完成：Recall@10=1.0，平均/P95
  2.2355/2.4097 ms；当前工作树和全新 checkout 完整测试均为 482/482。
- [ ] Week 5 64 条 Qwen3-VL 修复、Prompt pilot、继续 SFT、fresh test、四场景模型
  smoke 和 OSS 下载复验尚未完成；因此只允许代码进入 `dev`，不得进入 `stg`。
- [x] Prompt pilot 已完成：商品/售后/行程分别选择 compact/evidence/current，选择记录
  保持不可覆盖，未使用 fresh test 反向选 Prompt。
- [x] Week 5 v2 64 条修复全部 Schema-valid；最终 80,000/80,000 silver 结果 SHA-256
  为 `86b0a158...5926`，历史人工 accepted 统计未改变。
- [x] continuation SFT job `29562078` `COMPLETED 0:0`，`04:48:36`；patience=2
  早停并回载最佳 checkpoint-87，最终 adapter SHA-256 `c2fbb5c7...eaa2a`。
- [x] 同一 development 集完成候选、旧 unified、zero-shot 和 Week 6 routed 对比；总体
  加权为 0.920725/0.750034/0.084010/0.061806。不可覆盖开发门禁为 `PASS`，失败项 0。
- [x] 唯一一次 fresh test job `29569338` 在 A100 上 `COMPLETED 0:0`，`00:42:49`；
  120/120、失败率 0，三场景 JSON/Schema 1.0，对话自动综合 0.973330。
- [x] fresh-test raw/metrics 与单次消费标记哈希一致；final gate `PASS`、失败项 0，
  SHA-256 `9574b05b...a77d`。发布配置已绑定 checkpoint-87 adapter。
- [x] Compose 已增加 fail-closed `retrieval-init`，只接受 Milvus 空集合或完整 1,000 条
  状态；API 等待初始化成功。四层最终本地包含 12 份 evidence，全部哈希复验通过。
- [x] 真实四场景生产模型 smoke job `29571134` 在 A100 20 GB MIG 上
  `COMPLETED 0:0`；三场景首轮 Schema-valid，对话一次纠错后达到 `DIALOGUE_BETA`。
- [x] 当前工作树和全新 checkout 完整 unittest 均为 513/513；Compose 配置、
  `tripctl validate/doctor`、tracked secret scan 和 `git diff --check` 通过。
- [x] 导师确认交接不要求 Spartan 或 OSS 留存；唯一 59.9 MB 本地交接包通过四层、
  adapter、release config、final gate、真实 smoke 和 Milvus 基准复验，失败项 0。
- [x] 清理 21 个 ignored 目标并释放 71,735,466,519 字节；Yelp、模型缓存、中间输出、
  checkpoint 和迁移目录已删除，唯一交接包和凭据均受保护。
- [x] 新增接手文档、本地验证命令和导师可选择的 Week 8 优化方向；不将候选方向冒充
  已确定任务。最终完整 unittest 与全新 checkout 更新为 514/514。

## Week 8 交付（2026-08-26）

- [x] 临时 feature 分支和独立 worktree；主工作树既有改动未触碰。
- [x] 本地、Spartan、阿里云资产审计及官方 Yelp JSON/Photos 重建。
- [x] 版本化 fresh source、拒绝清单、human/silver 身份和 source/group/image 历史隔离。
- [x] 商品 v4 `400/60/60` 锁、错误切片统计和五维跨 split 隔离。
- [x] 三 Prompt 同 development 比较、不可覆盖 selection 和明确失败候选。
- [x] Prompt 门禁通过；按预注册规则跳过不必要的 continuation SFT。
- [x] Week 8 商品最终 test 唯一消费、marker/comparison/raw/metrics 哈希证据。
- [x] 对话首轮路由 v1/v2/v3 真实模型比较；v2 部分改善，剩余失败如实保留。
- [x] 固定输入冷启动/P50/P95/token/s/显存/失败率与质量一致性基准。
- [x] 检索 1,000 原图补齐、独立 query/index、development selection 和唯一 final test。
- [x] `configs/releases/qwen3_vl_system_week8_v4.json` 合并候选身份；正式 release 未改写。
- [x] 定向测试和完整 unittest `561/561 PASS`。
- [x] Week 8 报告、README、weekly log、delivery 和 experiments 更新。
- [ ] 对话仍有 `25%` 固定样本失败率；原因是模型尾部路由仍可能输出单任务附加键。
- [ ] 商品 known-price support 为 `0`；原因是合法隔离池没有可靠可见价位证据。
- [ ] 纯延迟优化只有 `0.24%` mean 改善；不以质量回退或量化换取更大数字。
- [ ] 阿里云历史数据盘未能认证读取；未进行任何云资源或安全组变更。

### Week 8 全自动扩展交付（2026-08-27）

- [x] 将后续人工标注、人工复核和人工验收数量固定为 `0`；新增数据全部显式标记为
  `programmatic_silver`，没有把自动标签伪装为 human gold。
- [x] official Yelp fresh source v3：6,000 候选、1,291 validated、1,000 selected；
  source/group/image 历史重叠 `0/0/0`，manifest SHA-256 `5c538740...a6e`。
- [x] 商品 v7 锁 train/development/test=`400/60/60`，五维隔离 PASS；候选选择期间 test
  保持未消费，内部 lock SHA-256 `321bea49...b0301`。
- [x] fresh development 三 Prompt 比较完成；`week8_product_field_check_v1` 以
  `0.836536` 对 current `0.782941` 锁定，JSON/Schema `1/1`、失败率 `0`、字段支持不变。
- [x] 可观察证据 Schema、确定性映射、hard-slice silver lock 与 continuation SFT 实现；
  human annotation/review/acceptance 均为 `0`，最终 test 未纳入训练或 development。
- [x] 对话预算/天数/城市/偏好/节奏确定性更新与安全 fallback；5 条固定样本三键合规、
  状态召回/值准确率/精确率/整状态准确率均为 `1`，纠错率和失败率均为 `0`。
- [x] 真实 600x400 图片重复基准；选择 384 token cap，5/5 输出完全一致，mean/P95
  `5006.81/5028.50→5000.55/5009.02 ms`。未选择 mean 变慢的图片 cap + cache 候选。
- [x] Milvus Lite `hybrid_weighted` development selection 与唯一 final；NDCG@10
  `0.125654→0.564459`、Recall@10 `0.018090→0.142734`，无 fallback、失败率 `0`。
- [x] release candidate 更新为 `configs/releases/qwen3_vl_system_week8_v7.json`；正式
  release、`dev/stg/main` 和历史冻结包未覆盖。
- [x] 两阶段基线与 SFT 首个 10% development 实际执行；checkpoint-5 adapter-only
  回载通过但因 composite `0.369804`、failure `0.683333` 被拒绝，未选为最终 adapter。
- [x] v7 商品 final test 唯一消费并完成：composite `0.819003→0.857729`，JSON/Schema
  `1/1`、失败率 `0`；comparison/marker 绑定 SHA-256 `5dc83953...f3829`。
- [x] release v7 四场景真实 smoke job `29638236` 通过；三任务首轮 Schema-valid，
  dialogue 无模型纠错达到 `DIALOGUE_BETA`，证据 SHA-256 `086133ec...85030`。
- [x] 终态完整 unittest `594/594 PASS`；远端锁复验、唯一 test consumption marker、
  release/adapter/smoke 哈希均一致；`compileall`、`git diff --check`、tracked secret
  signature scan 和 tracked large-file scan 均为 `PASS`。

### Week 8 剩余优化续行交付（2026-08-27）

- [x] 新增 v8 development-only Prompt overlay；绑定 v7 development 锁、禁用 final，并在
  test 读取前 fail-closed。
- [x] 实测两个额外商品 Prompt；composite `0.701144/0.703235` 均低于现有
  `0.836536`，按真实结果拒绝，最终 Prompt/adapter/release 不变。
- [x] 自动审计未消费 official source、历史图片哈希、native caption 与 OCR；新增数据的
  human annotation/review/acceptance 均为 `0`，没有把 metadata 价位写成视觉标签。
- [x] 审计确认 post-hash 安全候选仅 8，确认视觉价格 tier 正支持为 0；据此不启动预期发生
  unknown/空标签塌缩的 continuation SFT。
- [x] 商品 prepared-input cache 完成固定输入重复基准；质量一致但 mean/P95 回退，开关
  保持默认关闭，未改 release。
- [x] 检索 metadata cache 改为容量 512 的 LRU；真实 Milvus Lite 质量/支持/过滤/引用/
  失败率完全一致，稳态 P95 改善 12.55%，并记录预计算、命中、淘汰和内存成本。
- [x] 新 Prompt、商品 cache 和检索 cache 仅使用 development-only 身份；未读取或生成新
  final，v7 已消费 final marker 未修改。
- [x] 续行定向 `76/76`、完整 unittest `609/609 PASS`；compileall、Slurm shell、diff、
  tracked secret/large-file scan 与 release/adapter 哈希复验通过。
- [ ] 商品业态/风格残余回退和视觉价位正支持仍未解决；原因是无人工条件下合法新正样本
  不足，不虚构 gold 或模型自评正确性。
- [ ] 检索 LRU 尚未进入正式 API/release；当前证据只锁定真实 Milvus Lite development
  候选，生产接入仍需绑定 metadata 生命周期、预热与失效配置。

### 全项目复审修复交付（2026-08-27）

- [x] 更新最新复审要求；读取项目约束和历史证据，保留主工作树未提交改动。
- [x] 新增商品引用语义审计、两阶段 Schema 可见 Prompt/映射/纠错修复与失败零分协议。
- [x] 视觉 SFT 拒绝 metadata 代理或 known/unknown 矛盾目标；训练内存 backend 初始化修复。
- [x] 修复 VLM/CLIP 并发加载、失败后的半初始化、HTTP 图片缓存、API 生产示例兜底问题。
- [x] 修复对话重复修改、局部否定、部分更新、合法取消、非法数值和状态漏键评分。
- [x] development 不读 test 标签；final 对完整 selection 及五维身份进行校验；诊断重计分
  必须校验原输出 SHA，写入新目录，不能覆盖冻结证据。
- [x] 完整 unittest `654/654 PASS`、新增三组定向 `45/45 PASS`；历史 release、adapter
  与 test 保持原样。
- [x] 两轮商品 GPU 诊断及哈希校验重计分完成；旧证据链请求失败 `17/60→0/60`，基座
  解码消融失败 `33/60→2/60`。保留所有失败和字段支持，不将银标匹配解释为视觉准确率。
- [x] 修正 benchmark 图片身份：原 64×64 cafe 占位图只作连通性证据；新真实照片 job
  `29666837` 锁定 SHA/533×400 尺寸。三场景 Schema/对话契约通过，10 条对话状态
  exact `10/10`，首轮格式 `100%`，失败 `0`，包括再次修复的取消预算和非法负天数。
- [x] 真实照片重复延迟各 5 次：512/384 输出上限 mean `3894.280/3901.957 ms`，
  P95 `3943.654/3926.002 ms`，输出/tokens/Schema 一致。未证明明确速度收益，未改 release。
- [ ] 视觉能力整体提升未由现有 metadata 银标证明；固定四图反例与完整 development
  的审计限制已明确披露，不能标记“商品理解问题全部解决”。
- [ ] 对话一般问答/推荐完成度尚无验证；固定契约与状态更新成功不能替代这一指标。
- [ ] 真实 smoke 的行程输出仍复述模板占位内容；Schema 合规不能作为实质行程质量通过。

### c01b732 审查修复交付（2026-08-28）

- [x] 九项工程缺陷修复：新银标协议、语义选优资格、实际任务分派、行程业务检查、金额
  完整解析、轮次图片、生产查询检索、场景 422、统一 release 解析与 Compose 入口。
- [x] 25 条新增回归和完整 679 条 unittest 通过；没有通过删除失败反例“修好”测试。
- [x] 新 caption-only 诊断版本保留全部 60 样本，五维隔离通过；正支持 3/0/3/0 明确披露，
  旧/新引用均禁止视觉选优，未消费 final，不虚构 gold。
- [x] 真 Milvus Lite 与生产路由完成 5 个固定查询验证；请求条件影响结果、过滤正确、
  无匹配返回空集。旧参考 metadata NDCG 不作为该路径的相关性提升依据。
- [x] 真实 GPU job 29667548：商品对话执行模型；错误行程进入一次纠错，仍失败则明确
  NOT_COMPLETED。技术 PASS/业务 FAIL 分列，原始失败证据完整保留。
- [x] 固定实图商品重复 5 次：Schema 100%、失败 0、输出一致；MIG P50/P95
  4684.040/4702.216 ms，每次 713/57 token。不认定跨硬件提速，未改发布参数。
- [x] 旧模型交接包和 release/adapter 哈希复验通过，Compose 仅静态验证，未重启生产服务。
- [ ] 商品不可见设施猜测、可靠视觉评测支持、行程实际质量仍未解决；工程拦截不是模型语义修复。
- [ ] 检索未建模条件、独立相关性证据与 LRU 生产化、商品稳定提速仍未完成；不安排人工标注。

结论继续为 PARTIAL，不晋级、不合并 dev/stg/main、不打标签。详见商品报告第 13 节。

### 持续候选修复（2026-08-28，独立图像 silver）

- [x] 独立图像教师 60/60，全部 model_generated_silver，人工工作 0；不发送 merchant
  metadata、旧 target 或候选输出。商品原 development 样本与正支持不删减。
- [x] 正式/观察 v1/观察 v2 三组各 60 条真实推理及 raw 哈希复算；观察 v2 在所有指定
  字段不回退的条件下 composite 0.463794→0.745493，仅取得 development 候选资格。
- [x] 商品观察协议与逐场景 adapter 开关接入真实服务；保留售后正式 adapter。
- [x] 真实业务复审发现并保护日期、地点占位、活动截止/交通/必去禁去和虚构引用错误；
  尚待最新探针验证，不将旧探针 PASS 沿用为当前业务通过。
- [x] 发布运行层补齐缺失模块与配置，隔离导入与缺依赖反例通过；完整 unittest 729 条通过。
- [ ] 最新全链路探针、真实 null 模板的最终身份、候选锁、一次 final 与最终候选结论。

最新核验补充：生产探针 v3、准确 v8 release smoke、实际检索及对话分派已 PASS。v2 最终
集已消费且因 1/100 无效教师参考记为 INVALID_REFERENCE，不能晋级；正式及候选各 100
次推理无请求失败不等于语义验收通过。新的 development 纠错可靠性验证 60/60 有效，
固定选优参考未变。新增否定证据、候选身份绑定及验收重放回归后，全量 743 条测试通过。

### 2026-08-28 最终交付：v9 自动 silver 可晋级候选

- [x] 九项工程缺陷及后续复审反例修复，商品短事实协议和准确 adapter 分派接入服务。
- [x] 固定 development 60 条选优；新 100 条最终身份与所有历史、development 隔离，
  human=0、silver 身份真实，前次失败 final 永久排除且原样保留。
- [x] v9 单次最终及原始输出复算 PASS：composite 0.429365→0.736721，风格/设施 F1
  0.318182/0.163462→0.636364/0.751220；支持 57/51/0 样本（风格/设施/价位）。
- [x] JSON/Schema 100%/100%、失败 0/100；价位 N/A 不伪造支持。观察输出增加，平均
  延迟 4925.687→6016.703 ms，不能称为性能提升。
- [x] 真实商品/售后/行程/对话 smoke，2/3/5 日业务检查，实际检索/推荐闭环，24 次固定
  图像重复基准；缓存质量不变，但收益不显著，下游行程仍可能一次纠错。
- [x] 定向与全量 746 条 unittest、五维/图片哈希、release/adapter、准确运行层隔离导入、
  全量交接证据校验 PASS。Compose 仅静态检查，没有部署服务。
- [x] 新候选四层包与验收记录位于 `outputs/releases/trip-qwen3-vl-8b-week8-visual-silver-v9-rc1`。

本结论只覆盖自动 silver 候选验收，不是人工标注准确率或正式发布。价位、稀疏类别/多主体
支持、标签误差、延迟和检索未建模条件仍有限制；正式模型未替换，不合并长期分支或打标签。

补充交接 review：Spartan 同包复验 PASS；校验器改为检查必需 API 端点，而非依赖不同
FastAPI 版本的内部路由对象计数。新增缺路由反例后，全量 747/747 通过；原始 746 条
包内日志、模型/评分锁与最终数据均保持不变。

### 2026-08-28 v9 后续优化（不替换已验收候选）

- [x] 新增紧凑证据协议、重复键/推断证据防护、校验幂等性及 incumbent 非回退比较。
- [x] 原 60 条 development 四组实测及 raw 重放：v9 结果逐条复现；新紧凑 v4/v5
  因误报、请求失败及字段回退均拒绝，原输出和支持数保留。
- [x] 修复未应用检索条件被错误删除和英语复数业态解析；最新真实 Milvus 10 查询、
  4 对话状态检查通过，未支持条件明确未完成，不宣称图像相关性提升。
- [x] 定向 44/44、完整 unittest 769/769，原 v9 配置与四层归档 SHA 不变，隔离运行层导入通过。
- [x] 等价简洁 Schema 追加 development 完成（job `29697591`，18:39）：综合分
  0.666554、失败 2/60，相对同场 v9 0.759287、0/60 回退，拒绝替换；两组累计 420 请求。
- [x] 完整切片、支持数、延迟/token、失败历史和复核命令见商品报告 15.4–15.6。
- [ ] 新商品方案未取得超越 v9 的语义/稳定速度收益；价位无正支持，稀疏类别和风格/设施
  错误仍待改善。保持 v9，不为通过而调低指标或重新消费 final。

本轮没有新增人工工作、训练或最终 test；检索修复是 feature 代码及真实探针结果，未
重新打包或发布 v9。详见商品报告第 15 节，原候选质量结论仍只对应原冻结交接包。

### 2026-08-28 持续自主商品优化（执行中）

- [x] 完整 60 图的风格复查实测，失败结果保留；两种全面扩展方案均不替换 v9。
- [x] 独立版本修复 4 条 silver 风格依据，其余 56 条继承；style 支持 34 图/44 标签，
  非风格字段与五维身份不变，human=0。v9/候选用同一修订参考重算，不横比旧参考分数。
- [x] 实际像素上限修复与 CPU 视觉 token 验证；候选 raw 锁、独立教师范围拒绝和实际
  配置打包/隔离导入修复。最新完整 827 条测试通过。
- [x] 定点假设复查和有效像素上限的三组 development 对照已完成；前者无收益、后者
  有质量下降，均拒绝。新增明确非场所证据弃权的配对作业 `29705434` 完成（13:35）。
- [x] 同一修订参考 60 图：style F1 0.630435→0.637363、composite 0.754617→0.756926，
  其余有支持字段不退；固定 v10 为待验候选。支持 category 39、style 34 图/44 标签、
  facility 37 图/78 标签、price 0（N/A），silver 权重 0.5，human=0。完整测试 834 条通过。
- [x] v10 真实商品/对话、约束行程、弃权分支、检索及三组重复延迟通过；24 次同标签且无失败。
- [x] v10 新单次 final（GPU `29705792`，28:20）已执行并保留真实失败：Schema 99%、
  失败 1/100，未晋级。教师 100 条有效，raw 重放一致；失败图片不替换、集合不复用。
- [ ] 新可晋级候选尚未形成。仅使用已有 development 检查通用纠错，不用最终集调参。
- [ ] 未取得无损加速；当前质量收益伴随约 1.4% 的 development 平均延迟增加。原 v9 不变。
- [x] development 历史错误的有界纠错实测与 raw/输入血缘重放；新逻辑 1/34 失败，拒绝。
- [x] 修复旧 revision 比较允许综合提升掩盖单字段回退的问题；全量 852 条测试通过。
- [x] Schema 字段注释完整 development 已执行并因质量回退拒绝，未进入最终测试。

最新补充：884 条 unittest 通过；共享格式解码器的 CPU 契约/格式检查通过，真实纠错
诊断时限中断后仅续跑剩余 12 条，两组 34/34 且重放通过，旧结果和中断证据保留。
新纠错均值 29.245 秒，不能宣称提速；完整 60 图保留风格收益，但补充纠错切片出现
设施回退，统一约束不晋级。已实现仅食品矛盾触发的更小改动，真实验证与最终验收未完成。
新解码候选的纠错原始重放及分切片语义不退已接入自动验收；冻结 v9 交接包再次通过。
针对性方案已通过全部 34 条双组纠错及逐错误语义不退，跨主机重放一致；固定 60 图
质量对比也已通过，选定 v11 待验；style F1 0.630435→0.637363，其他支持字段不退。
仍待新真实业务/重复性能/单次 final 和交接包验证，此处不是新晋级结论。

### 2026-08-29 v11 验收进展

- [x] 准确 v11 release 的真实商品/售后/行程/对话及风格弃权、业务约束检查通过。
- [x] 三种缓存各8次，24条 raw 重放及延迟/token复算通过；无失败/标签变化，无实质提速。
- [x] 生产检索10查询/4对话通过；未支持条件不伪装完成。
- [x] 新100图五维隔离、候选/数据锁及原adapter/v9/v10冻结哈希检查通过，human=0。
- [x] 新单次final `29708885` 已完成并保留真实失败：三组无请求失败，但候选业态45/56
  低于同场正式47/56，验收FAIL；跨主机raw复算一致。全量884/884、定向51/51通过。
- [ ] 新的可晋级候选及交接包。v11不晋级，回到原development改进类别；final v5不复用。
- [x] 独立主体复查及字段不变/矛盾失败/原始重放回归实现，898条完整测试通过。
- [x] 原固定60图v15双组真实对比及跨主机重放；类别32/39→30/39，保留失败，不晋级。
- [ ] 收窄至可见功能矛盾的v2/v16实测及后续独立验收；没有新通过候选。
- [x] 无标签归档身份池、锁和历史隔离回归实现；全量906/906、冻结v9交接复验通过。
- [ ] 新身份池实际提取与独立100图final。仅完成实现不能算最终商品质量通过。
- [x] v16真实双组60图、跨主机原始重放及对v11的development revision非回退检查通过。
- [x] 无标签池实际4000候选→1039独立可读身份，拒绝2961，所有历史含final v5排除。
- [x] 新真实服务主体复查/风格弃权组合检查，完整910/910、原缺陷64/64回归通过。
- [ ] v12准确配置的纠错实测、真实业务/检索/重复延迟、新100图final及交接包；未晋级。
- [x] v12纠错34条双组、分切片语义非回退及跨主机重放；10查询/4对话生产检索通过。
- [x] 新100图无标签身份锁及五维隔离、本地/Spartan来源依赖复验；运行层独立导入通过。
- [x] 修正正式Prompt测试对环境默认值的依赖；v12环境完整910/910（v63），v62失败保留。
- [ ] 真实业务/24次性能探针完成、新单次final与可晋级交接包；仍不作完整通过声明。
