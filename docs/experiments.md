# Experiment Notes

## 2026-08-02：Week 5 候选池与隔离验证

- Git 基线：`72be7ce` 加本次未提交工作区；数据版本
  `week5_instruction_candidates_v1`。
- 输入：本地 Yelp business/photo/weak-pair Parquet；排除接口为 Week 3 v1/v2
  exclusion manifest。
- 命令：`python scripts/manage_week5_dataset.py build-pools` 与
  `python scripts/manage_week5_dataset.py validate-pools`。
- 实际候选：商品 50,000、售后 20,000、行程 10,000；跨场景唯一图片 SHA-256
  为 80,000，评估 source/hash/group/template 冲突为 0。
- 售后公开图片关键词路由在严格来源组排除后得到 5,552 条；用 14,448 条独立
  Week 5 项目自有业务合成凭证补足四类各 5,000。路由和严重度提示不是金标。
- Qwen3.7 映射为商品/售后 `fewshot_4_v2`、行程 `standardized_v4`。通过
  `trip-api-sg` SSH 别名从 ECS 进程内临时读取密钥，商品 smoke 3/3 Schema
  合规、失败 0；密钥未写入本地文件、运行记录或 Git。
- 人工修正、三级质检、最终合格和多轮对话均为 0；原因是未收到必要人工输入。

## 2026-08-02：Qwen3.7-Plus 前期任务重跑

- 模型/后端：`qwen3.7-plus`，阿里云百炼新加坡 OpenAI-compatible API，
  thinking disabled。
- 数据：不可变 `week3_evaluation_v2`，商品/售后/行程 200/150/100。
- Week 3：baseline `_002` 与 standardized `_001` 均完成 450/450，
  请求错误 0；baseline `_001` 的单条 ReadTimeout 仅作为失败证据保留。
- Week 4 pilot：商品和售后选择 `fewshot_4_v2`，行程选择
  `standardized_v2`；winner full 完成 450/450，请求错误 0。
- 共同语义比较：450 对、38 个聚合指标、2,000 次 bootstrap。
- 主要结果：winner 商品/售后/行程 JSON 合规率 100%/100%/33%，
  Schema 98.5%/100%/33%；行程格式和约束仍是主要短板。
- 完整报告：`reports/qwen37_previous_weeks_rerun_report.md`。

## 2026-08-02：Qwen3.7-Plus 行程规划修复

- 根因：旧行程输出中 67/100 条精确达到 1280 completion tokens 并截断。
- 修复：版本化 `standardized_v4`、紧凑字段、约束原文保留、英文枚举约束，
  行程专用 `max_tokens=2560`。
- 最终 run：`itinerary_qwen37_repair_v4_full_20260802_001`，100/100 JSON，
  100/100 Schema，请求错误和 token 上限命中均为 0。
- 业务指标：约束识别 89.95%，硬/软约束 F1 96.33%/85.67%，约束检查
  覆盖率 94%，行程要素完整度 100%。
- 报告：`reports/qwen37_itinerary_repair_report.md`。

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

## Week 3 v2 recuration and response-format probes on 2026-07-22

- The mentor-authorized low-quality-image route supersedes the frozen-v1-only scope while preserving all v1 artifacts unchanged.
- Prepared `week3_evaluation_v2`: product 200 completed; after-sales 80 completed plus 70 pending replacements; itinerary 4 completed plus 96 pending style-only supplements; exclusion count 450.
- After-sales candidates contain public Yelp and business-synthetic sources and exact intended candidate strata `38/38/37/37`. Candidate strata are not reported as human gold until submitted.
- Local model: `Qwen/Qwen2-VL-2B-Instruct` through vLLM on `localhost:8001`; temperature 0.1, top_p 0.9, repetition_penalty 1.05, max_tokens 1280.
- Strict `json_schema` response mode for the nested itinerary contract timed out at 180 seconds and was rejected for full evaluation.
- `json_object` plus full Schema and a final type skeleton produced JSON- and Schema-valid representative product, after-sales, two-day itinerary, and four-day itinerary outputs. The four-day probe generated only one itinerary day, so semantic constraint metrics remain independent from Schema pass.
- These probes validate request/format behavior only. They are not baseline results, do not increment `tested_count`, and do not support model capability conclusions.
- Annotation inspection rejected the first hygiene/facility replacement batch because its abstract synthetic diagrams were not representative business evidence. Three early submissions were invalidated with retained hashes. The after-sales UI is paused until those images are replaced.
- All original itinerary non-style annotations were found intact. The corrected UI inherits them server-side and collects only `style_preferences`; four submitted style supplements are preserved.

## Week 3 v2 full evaluation on 2026-07-24

- Dataset `week3_evaluation_v2` validates at product/after-sales/itinerary counts `200/200/200/200/200`, `150/150/150/150/150`, and `100/100/100/100/100`; exclusion count is 450.
- Runs `week3_v2_baseline_full_20260724_001` and `week3_v2_standardized_full_20260724_001` are completed/live/full with 450/450 persisted records, no request errors, and identical selected-sample SHA-256 `3e900e64bb345df35343c8f14bfb1f8310ae597a57e4a4d9585bc01173ad648c`.
- Baseline JSON/Schema rates are 0% for all scenarios. All semantic task metrics are `PENDING` because the required minimal Prompt produced unparsed natural language.
- Standardized v2 JSON/Schema rates are product 79%/75%, after-sales 96.67%/96%, and itinerary 90%/88%. Persisted strict semantic aggregates and support counts are reported in `reports/week3_zero_shot_baseline_report.md`.
- Comparison `week3_v2_prompt_pair_20260724_001` contains 450 paired rows and 2,000 bootstrap iterations. Only comparable format, Schema, and latency metrics are used; no baseline semantic score is inferred.
- Week 3 remains `PARTIAL` because the baseline natural-language semantic track remains unsupported, not because data collection or live inference is incomplete.

## Week 3 deterministic baseline semantic score on 2026-07-25

- Source run: `week3_v2_baseline_full_20260724_001`; no new inference was sent.
- Score ID: `week3_v2_baseline_full_20260724_001__baseline_semantic_coding_v1`.
- Coding version: `baseline_semantic_coding_v1`; codebook SHA-256 `563dc0747f92b6ccaa37466045cb0e74229787824013d59a5f6f26261bb033a6`.
- Prediction inputs are limited to scenario, raw output, fixed codebook, and normalization. Annotation, sampling stratum, source metadata, suggestions, and standardized output are unavailable to the encoder.
- Product: category accuracy 45.45% (support 110), price accuracy 2.00% (support 100), style macro/micro F1 28.28%/16.53% (support 200), facility macro/micro F1 53.22%/46.41% (support 200), label completeness 23.50% (support 169).
- After-sales: issue accuracy 60.00% (support 150), severity accuracy 0.00% (support 150), key-information macro/micro F1 29.67%/21.02% (support 150), OCR recall 14.22% (support 75).
- Itinerary: constraint recognition 0.00% (support 100), hard/soft constraint macro F1 0.00%/0.00% (support 100), itinerary-element completeness 77.20% (support 100), element macro/micro F1 71.09%/71.81% (support 100).
- Baseline JSON/Schema compliance remains 0%/0% in all scenarios. The lexical and strict structured tracks are shown separately and are not used for a causal Prompt-effect claim.
- Verification: 226/226 unit tests passed; standalone v2 validation and both run-bound validators returned `status=ok`; the 450-row semantic score passed strict JSON, support, run-ID, codebook-hash, and scenario-count checks.
- Status: `READY / COMPLETED`.

## 2026-07-25：Week 4 Prompt pilot 与 Milvus standalone

- 数据集：不可变的 `week3_evaluation_v2`；每场景固定 5 个正例、
  2 个边界例和 5 个不重叠 pilot 样本。
- 模型/后端：`Qwen/Qwen2-VL-2B-Instruct`、vLLM；temperature 0.1、
  top-p 0.9、repetition penalty 1.05、max tokens 1280。
- 旧候选 `fewshot_4_v1`、`fewshot_7_v1` 的行程请求因上下文超过
  4096-token 上限而返回 HTTP 400，保留为无效运行证据。
- 版本化候选 `fewshot_4_v2`、`fewshot_7_v2` 保持模型、生成参数和示例
  数量不变，只压缩重复 Schema 与行程展开；两组均完成 15/15，
  `model_request_error_count=0`。
- 有效 pilot 中，商品、售后、行程均由 `standardized_v2` 胜出，选择分数
  为 0.3280、0.5967、0.4775。新增 Few-Shot 未超过控制组。
- 全量运行 `week4_winners_full_20260725_001` 完成 450/450；
  样本哈希为 `3e900e64...ad648c`。
- 全量结构化轨道的商品、售后、行程 JSON/Schema 分别为
  77.5%/75.5%、96.67%/96.67%、90.0%/87.0%。baseline 词法业务轨道
  与结构化业务轨道不计算差值；baseline token 未记录，明确为 `PENDING`。
- Milvus 2.6.20、PyMilvus 2.6.16、etcd 3.5.18 和固定 MinIO
  部署 healthy；十字段 Schema、HNSW/COSINE 和 8 个标量索引已验证。
- 20 条真实 CUDA CLIP 向量完成 CRUD。HNSW 构建 5.6621 s，
  10 次查询平均/P95 为 7.7982/10.7236 ms，Recall@5 为 1.0000。

### 中央审查问题及修复

- 审查发现 Windows CRLF 使两个 Week 3 run-bound 原始字节哈希失败。
  新增 `.gitattributes`，provenance 对文本换行归一化并兼容既有
  LF/CRLF 历史哈希；两个验证现均为 `status=ok`。
- 移除跟踪配置中的 Milvus/MinIO 明文凭据，改为环境变量和脱敏
  `.env.example`；本地容器已使用新随机凭据重建，19 条可见向量保留。
- Milvus 基准现在要求输出不存在且集合物理行数为 0；插入/删除后的数量
  来自 `count(*)`，不再由输入长度推断。
- runner 在任何模型请求失败时将运行标为 `failed`；统一验证器同时拒绝
  运行记录或候选摘要中的请求错误。
- v2 比较产物删除跨轨道 `business_quality_delta`，只比较同口径格式、
  延迟和 token 可用性；全套 244 个测试通过，状态为
  `READY / COMPLETED`。

## 2026-07-26：共同语义评分与 Few-Shot 设计复核

- Git 基线：`d6e1b8c`；模型运行未重跑，使用冻结的
  `week3_v2_baseline_full_20260724_001` 和
  `week4_winners_full_20260725_001` 原始输出。
- 数据/模型：`week3_evaluation_v2`，450 对相同 sample，SHA-256
  `3e900e64...ad648c`；`Qwen/Qwen2-VL-2B-Instruct`、vLLM 和原生成
  参数保持不变。
- 命令：`python scripts/compare_week4_common_semantics.py`。
- 评分：两边均使用 `BaselineSemanticCoder.encode`、
  `baseline_semantic_coding_v1` codebook
  `563dc074...033a6`、同一人工金标与指标函数；bootstrap 2,000 次，
  seed `20260726`。
- 主要结果：商品 category/price delta +32.73/+13.00 pp，style/facility
  macro F1 +3.93/-19.88 pp；售后 issue/severity +18.00/0.00 pp，
  key-information F1/OCR recall +0.67/-9.78 pp；行程 constraint
  recognition 0.00 pp，element completeness -55.40 pp。
- 局限：固定词法 codebook 原为 baseline 自然语言设计，对 JSON 标点分隔
  的枚举值、改写和隐含约束识别有限；共同轨道只支持该编码器下的成对解释。
- Few-Shot 示例来自最终 test gold。现有 v2 pilot 请求有效，但只作描述性
  证据；不构造未授权的新 demo/dev 数据，不重跑模型。
- 验证：245/245 单元测试通过；Week 3 v2 数据、baseline/standardized
  run-bound 和 Week 4 统一验证均为 `status=ok`。Compose 配置展开通过；
  当前容器状态复核因 Docker daemon 未运行而 `PENDING`，历史 Milvus
  CRUD/性能证据未改写。

## 2026-07-26：独立 demo/dev Few-Shot 重跑

- Git 基线：`abb689a` 加本次未提交工作区；数据版本
  `week4_demo_dev_v1`（development 36 条人工金标）与
  `week3_evaluation_v2`（evaluation 450 条）。
- 模型/后端：`Qwen/Qwen2-VL-2B-Instruct`、vLLM 0.8.5；
  temperature 0.1、top-p 0.9、repetition penalty 1.05、
  max tokens 1280；未运行 CLIP。
- 隔离：完整 development 池与最终 evaluation 在 sample_id、source_id、
  image SHA-256 和 group_id 上无交集；selection v2 选择 21 个示例。
- Pilot：`standardized_v2`、4-shot、7-shot 各 15 条，全部完成且请求
  错误为 0。综合分胜出为商品 4-shot、售后 zero-shot、行程 zero-shot。
- 全量运行 `week4_winners_full_20260726_002` 完成 450/450，样本
  SHA-256 `3e900e64...ad648c`。
- 全量 JSON/Schema：商品 82.0%/20.5%，售后 96.67%/96.67%，行程
  91.0%/88.0%。商品结果显示小 pilot 选择方差，负结果不触发反向选模。
- 同轨比较：`week4_common_semantic_coding_v1_20260726_003`，450 对、
  38 个指标、bootstrap 2,000 次；bad case v5 共 376 条。
- 验证：`python scripts/validate_week4_delivery.py` 返回 `status=ok`。

## 2026-08-09：Qwen3-VL-4B Week 5 行程配对 pilot

- 运行：`week5_itinerary_prompt_pair_4b_20260809_a`；模型
  `Qwen/Qwen3-VL-4B-Instruct`、vLLM 0.11.0、A10，回环模型端点经 SSH 隧道访问。
- 数据：现有不可变行程候选池前 30 条唯一样本；候选 manifest SHA-256
  `4072260173f0b25cf7d5d63ab694f0849b351a483f42e4c39b9a99c5b9a17e75`。
- 请求：`fewshot_4_v2` 与 `standardized_v4` 各 30 次，共 60 次，无请求失败、无重试；
  原始输出 60 份，Schema 失败 2 份且均来自 `fewshot_4_v2`。
- 结果：Schema 合规率 93.33% 对 100%；平均总 token 2,171.7 对 1,128.3；
  平均延迟 21,353.77 ms 对 18,523.86 ms。业务质量无人工分数，按裁决的结构性
  并列规则选择 `standardized_v4`。
- 成本：推理进程 1,197.05 秒，估算 CNY 6.09；未运行 80,000 条全量预标注、
  对话批量生成或任何训练任务。
