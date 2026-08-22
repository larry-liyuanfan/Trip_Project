# Experiment Notes

## 2026-08-19：Week 6 训练后方法审计（未运行新训练）

- 输入仅为已锁定的 Week 6 汇总指标、错误分类和一手技术资料；没有对冻结
  `week3_evaluation_v2` 重新生成、筛选或计分，也没有提交 GPU 作业。
- 审计结论：商品设施/风格/价位、售后严重度/关键信息、行程约束泛化是主要语义短板；
  Schema 约束解码可单独改善格式，但不能替代语义学习。
- 下一版本按 ADR-030 先建立新的非冻结 development/test 锁，再依次比较错误切片 SFT、
  grammar-constrained decoding 和有质量门禁的多模态 DPO 小规模消融。
- 论文依据、适用边界和预注册门禁见
  `reports/week6_post_training_improvement_review.md`。本记录不包含新的效果声明。

## 2026-08-19：Week 6 行程专项晋级与冻结三场景最终评测

- 行程基线 `29356991` 在固定 64 条派生 validation 上 JSON/Schema 64/64，但全项
  通过、必需元素完整均为 0；天数/日序/约束原文/constraint check 覆盖分别为
  11/11/9/54。refinement `29375367` 完成 1791 steps/3 epochs，最低
  `eval_loss=0.0013725368` 位于 `checkpoint-540`，adapter 回载 SHA-256 为
  `7ab168a0f7073f2fad3369c028f744585362a0668f77c024098d9b27d92c9a6a`。
- 候选评测 `29408124` 的九项结构计数均为 64/64；CPU comparison `29412603`
  返回 `status=passed`、无回退原因，故锁定候选并停止调参。
- 恢复冻结 `week3_evaluation_v2` 后，validator `29418805` 确认 450/450，CPU
  preflight `29418839` 完成 77 项测试、HF cache 和三个 adapter 哈希。三个最终 GPU
  作业严格串行：商品 `29418875`、售后 `29419327`、行程 `29422130`，均
  `COMPLETED 0:0`，样本 200/150/100。
- 最终 JSON/Schema：商品 100%/100%，售后 100%/96.67%，行程 95%/85%。商品
  业态准确率 86.36%、价位 46.00%；售后问题类型 86.67%、严重度 34.67%；行程
  约束识别 30.33%、约束覆盖 48.83%、要素完整度 85.00%。完整指标与局限见
  `reports/week6_qlora_quality_report.md`。冻结结果只用于最终报告，没有继续调参。

## 2026-08-18：Week 6 行程专项误差切片与结构修复准备

- Git 基线：`3d6bc81df8c4afd496e1e78d41c6b4bfa07c7bf4` 加本次未提交工作区；已完成的
  行程训练 job `29312217` 为 `COMPLETED 0:0`，best eval loss
  `0.005681941285729408`、best checkpoint `checkpoint-1620`，adapter 已通过磁盘回载。
- 原始锁 `week6_week5_final_human300_20260817_v4` 的行程目标虽然全部 JSON/Schema
  合规，但业务结构审计仅 train `3/9538`、validation `0/450` 全项通过。train 中
  行程天数匹配 `1681/9538`、约束原文精确覆盖 `1661/9538`、必需行程要素完整
  `8/9538`；validation 分别为 `61/450`、`82/450`、`0/450`。因此低 validation
  loss 只说明模型拟合了原目标，不能证明行程业务效果优秀，也不应继续在同一错误目标上
  增加 epoch。
- 新建不可覆盖的派生 silver 锁
  `week6_itinerary_structural_repair_20260818_v1`：保留图片证据字段，只按输入中明确的
  天数、约束原文和模板规则修复结构，不推断外部地点；所有派生行均标记
  `model_preannotation`、权重 `0.5`，不继承真人身份。train `9538/9538`、validation
  `450/450` 通过同一确定性审计。
- 新锁内部 manifest SHA-256 为
  `c492148d0127fb0c557985fab8aefad80f2e8dd9ac787ef5f84b4d0355a0e31c`，split
  SHA-256 为
  `561580d06215e026c0e24ade8d349c3b4ba950dfc8d9dc50754b1f399bdca5b6`；原 v4 锁未修改。
- 第二阶段配置要求从已完成行程 adapter 继续训练，并绑定
  `adapter_model.safetensors` SHA-256
  `18c5dfad0a423945f19b0d1ea863e82bda3934634aa4b5922023c3421ba114ac`，禁止意外从基座
  重训或混用 adapter。新增 supervisor 将 allocation 与训练子进程分离，支持在同一
  allocation 内从最新有效 checkpoint 恢复；walltime 为基于本轮 `04:26:45` 实测的
  紧凑 `08:00:00`，不是默认 72 小时。
- 本地验证：行程原锁和修复锁四次审计均完成；Week 6 定向测试 31/31、完整 unittest
  362/362、`py_compile`、3 份 Slurm 脚本 `bash -n` 和 `git diff --check` 通过。
- 新增同集 comparison gate：绑定 evaluation input、sample ID、dataset lock 与生成参数；
  候选必须增加全通过样本且九项结构计数不回退。候选 adapter 还必须由完成的 refinement
  `run_summary` 证明来自固定初始 adapter、相同数据版本且 adapter-only 可回载。
- 状态：`PREPARED`，尚未提交新的 GPU 作业。下一步先在新锁 validation 的固定子集上
  评估现有 adapter，取得真实业务基线后才决定是否启动第二阶段训练；冻结 Week 3 评测
  不用于调参。

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

## 2026-08-12：Week 5 全量预标注隧道故障恢复

- 运行：`week5_full_preannotation_qwen3_vl_4b_20260809_b`，模型
  `Qwen/Qwen3-VL-4B-Instruct`，vLLM 0.11.0，A10，SSH 回环隧道。
- 故障：本机隧道退出后出现连续连接拒绝；runner 在 21 次连续请求失败时触发保护
  停止。停止前唯一成功商品 13,477，未解决失败 22。
- 远端核验：`/v1/models` 返回目标模型，说明 vLLM 健康；故障边界在本地隧道。
- 恢复：使用 `scripts/supervise_week5_preannotation.ps1` 重建带 keepalive 的隧道，
  隐藏启动同一 run ID 的 `--resume`，成功结果不重复请求。
- 恢复证据：checkpoint 从池索引 13,498 推进到 13,546，本进程 48/48 成功、连续
  请求失败 0；全量运行仍在进行，不能记录为完成。

## 2026-08-12：Week 5 全量预标注 ECS 原生续跑

- 代码提交：`a9a8d99`；运行仍为
  `week5_full_preannotation_qwen3_vl_4b_20260809_b`，未创建替代 run。
- 部署输入：80,000 条候选、80,000 张唯一图片、图片缺失 0；迁移 payload SHA-256
  `735b64305cf84790937ae94e63057f8ffcddb07d08ec812a0bd075c03e389bbc`。
- 身份校验：canonical config SHA-256
  `dd94313dfa0dbd070e11270ec70157ea60c5e9162ca4a56e76ca461c46d05484`，三份候选
  manifest 哈希与历史 run manifest 完全一致。
- 切换点：本地 checkpoint 15,190；`results/attempts/failures` 分别为
  15,166/15,329/44，均逐行 JSON 合法。服务器恢复后首次确认点为成功 15,197、
  checkpoint 15,209、连续请求失败 0。
- 运行方式：systemd 常驻，vLLM 仍只通过服务器回环地址访问；本地 supervisor、runner
  和 SSH 隧道已停止。迁移不代表全量预标注、人工标注或质检完成。

## 2026-08-20：Week 7 v3 development、Schema 与统一训练

- 权威配置：`configs/week7/qwen3_vl_8b_multitask_context_v3.json`，SHA-256
  `d77d9f10b551f30c599572e974fba2c3c2af087f37ed35e93b9dc7ac2dc105fa`；数据锁
  `week7_fresh_multitask_context_20260820_v3`，SHA-256
  `8af2e2d13c22fb641fc7344b1e56e5827aa78b1ebde653c6e55c83b36d20504d`。详细计数与
  隔离证据见 `experiments/week7_data_lock_20260820_v3.json`。
- v2 对话 train/development/test 分别为商品父任务 450/24/24，售后和行程均为 0；GPU
  作业 `29431992` 已取消并排除。该失败不修改既有锁，v3 以全新身份重建。
- development 作业 `29433880`–`29433884` 完成，Week 6 adapters 与 zero-shot 的
  raw/metrics SHA、支持数及重算值见
  `experiments/week7_development_baselines_20260820_v3.json`。
- Schema 最终作业 `29434316` 完成；free/constrained 各 90 条。free JSON 合规率
  98.89%；constrained primary 失败率 100%，free fallback 失败率 0%，配对延迟比
  1.0181。只选 free，且格式结果不得解释为语义提升。完整哈希与失败/取消尝试见
  `experiments/week7_schema_decoding_20260820_v3.json`。
- 统一训练作业 `29434317` 使用一张 L40S，运行身份
  `week7_multitask_context_sft_20260820_v3`，step 151 早停并正常完成。step 76/113
  综合分最高为 0.869412，但全局延迟比分别为 1.4276/1.4506，四个 checkpoint 还都
  存在商品支持数不足；selector 返回 `BLOCKED_NO_ELIGIBLE_CHECKPOINT`。详见
  `experiments/week7_multitask_training_20260820_v3.json`；未创建 parameter-lock，
  final-test 未运行。
- DPO 门禁见 `experiments/week7_dpo_gate_20260820_v3.json`：真实审核偏好对为 0，
  因此 `SKIPPED`。该提交当时完整 unittest 401/401，compileall、五份 Slurm `bash -n`、锁验证和
  diff 检查通过。

## 2026-08-20：Week 7 evaluation protocol-v4 公平重评

- 复审确认 v3 selector 的商品支持数与延迟比较存在口径污染：无 gold 可评证据的
  Schema-invalid 输出被错误计为支持，训练内 development 推理又使用
  `use_cache=false`，不能与独立加载且启用 cache 的 Week 6 基线直接比较。提交
  `833d41a` 增加独立 `evaluation_protocol_v4.json`；它绑定既有 v3 config、数据锁、
  development 与四个 checkpoint，不是新数据锁、新训练或新模型。
- 首次公平重评 job `29449140` 在 A100 preempt 节点运行 00:48:55；完成 Week 6
  baseline 角色后确认 90 分钟不足并显式取消。该目录没有 `protocol_summary.json`，
  partial 产物保留但不进入任何聚合或 selection。
- 第二次独立作业 `29449999` 在 L40S 节点运行 01:19:44，状态 `COMPLETED 0:0`。
  同一 allocation 依次完成 Week 6 routed adapters、step 38/76/113/151 与 zero-shot；
  protocol summary SHA-256 为
  `6990bda69463d9d9df65082c39d9d53733e176d4988ea6342eb170fde7c960f3`。
- 公平 Week 6 全 development 平均延迟为 5727.70 ms。step 38/76/113/151 的综合分
  分别为 0.258513/0.723404/0.746154/0.733077，失败率均为 0；延迟比分别为
  1.6312/1.3221/1.3155/1.4894，全部超过预注册 1.25。step 38 还未通过行程任务与
  格式门禁；其余三个候选仅被全局延迟阻断。
- selector SHA-256
  `39f76e9992eb6cb88f095200b3378646c44a8dab18e64acde7e986813de4cb5b`
  返回 `BLOCKED_NO_ELIGIBLE_CHECKPOINT`，candidate/eligible=4/0。未创建
  parameter-lock，正式 test 未读取且一次性额度未消费；人工对话项仍为
  `PENDING_REAL_HUMAN_INPUT`，DPO 仍因无真实审核偏好对而 `SKIPPED`。完整机器证据见
  `experiments/week7_evaluation_protocol_20260820_v4.json`。
- protocol-v4 修复提交当时的完整 unittest 为 412/412；compileall、Week 7 Slurm `bash -n`、数据锁
  验证和 `git diff --check` 均通过。本节记录的是 development 阻断结论，不能替代
  Week 6/统一模型/zero-shot 的一次性正式 test 三方比较。

## 2026-08-21：Week 7 evaluation protocol-v5 与一次性 test

- `configs/week7/evaluation_protocol_v5.json` 继续绑定 v3 config SHA
  `d77d9f10...105fa`、数据锁 SHA `8af2e2d1...20504d`、114 条 development 与四个
  既有 checkpoint；仅把公平推理口径锁为 BF16、static KV cache、Transformers
  compile、32-token warm-up、CUDA 同步计时和 gold-evaluable support。没有新训练、
  新数据锁或 test 调参。
- job `29452655` 因 `/data/gpfs` 全局 100% 满在 zero-shot 阶段 `FAILED 1:0`；前五角色
  完整但整个 attempt 排除。job `29456882` 因提交器提前创建输出目录，在 13 秒内由
  不可覆盖门禁拒绝，未执行推理。job `29456896` 改用 home 输出与节点本地 compile
  cache，在 L40S 上 `COMPLETED 0:0`，耗时 01:28:19；protocol summary SHA-256 为
  `08e9e49067a5aabba623139603607d00f51daf7fd041b4b184cc2ad468c3a351`。
- development step 38/76/113/151 的综合分为
  0.074359/0.642718/0.645237/0.740904，平均延迟为
  8689.47/7863.99/7188.87/7356.58 ms，相对同 allocation Week 6 基线为
  1.0405/0.9417/0.8609/0.8809，失败率均为 0。selector 4/4 eligible，按最高综合分
  选择 checkpoint-151；selection SHA-256 为 `68bfbedb...ca50`。
- 提交 `8619b76` 修复 final runner 的 NF4/dynamic/旧支持口径不一致，参数锁完整绑定
  v5 runtime；canonical lock SHA-256 为 `1b3f3ffa...adef`。唯一 final-test job
  `29459265` 在 L40S 上 `COMPLETED 0:0`，耗时 00:40:50；marker 为 COMPLETED、无
  failure history，7 个 artifact hash 全部复验通过。
- test 统一模型/Week 6/zero-shot 加权综合分为 0.744987/0.061840/0.075577，平均延迟
  为 7173.16/8250.70/4788.49 ms，失败率均为 0。统一模型商品/售后/行程 composite
  为 0.153846/1.000000/0.996667，三场景任务、支持、JSON/Schema、全局延迟和失败率
  门禁全部通过。对话自动格式/上下文召回为 1.0/0.878472；人工四维继续
  `PENDING_REAL_HUMAN_INPUT`，DPO 因 0 条真实审核偏好对保持 `SKIPPED`。
- 完整机器证据见 `experiments/week7_final_evaluation_20260821_v5.json`。当前完整
  unittest 414/414，远端定向 22/22，compileall、shell 语法与 diff 检查通过。

## 2026-08-21：Week 7 dialogue context integrity audit

- 触发：真实用户在人工评分界面发现 assistant 回复与 user 问题不对应。只读代码与固定
  development 队列复核确认构造顺序为 assistant→对应 user，而非 user→assistant；
  development 24/24 命中，配置声明的 train/test 影响数为 450/24。
- 影响：checkpoint-151 raw 是在错误历史上下文上生成的，不能通过 UI 重排后计分；历史
  格式合规和字符串包含式 context recall 不检测语义轮次或末轮相关性，因此不再用于真实
  多轮能力结论。三个核心场景的独立自动指标保持原始事实。
- 门禁：标注台绑定 v3 数据锁 `8af2e2d1...20504d`、队列 SHA `3a94eed4...f156`、
  development SHA `3c775c14...0e04` 和 raw SHA `aee27cf1...7090`，24/24 阻断；后端
  同样拒绝保存，人工记录仍为 0。不得改写 v3 锁、旧 raw 或 test marker。
- 机器审计见 `experiments/week7_dialogue_context_audit_20260821_v1.json`；完整 unittest
  418/418、compileall 和 `git diff --check` 通过。

## 2026-08-21：Week 7 corrected dialogue development review v2

- 独立身份 `week7_dialogue_review_20260821_v2` 仅派生新的 development 对话和空白人工
  队列，绑定 v3 锁与 checkpoint-151；24 条 5–8 轮按 6/6/6/6 分布，首轮及每个
  follow-up 均为 user→具体 assistant 回答，图片仅首次用户轮，未读取 test。
- 失败证据不合并：`29479321`/`29479416` 在运行前取消；`29479456` 因错误 HF cache
  路径失败；`29479500` 因 home quota 在落盘时失败且无 raw。有效 job `29479822` 在
  `gpu-a100-preempt`/`spartan-gpgpu098` 完成，24/24 成功、失败 0。
- 新 raw/metrics SHA 为 `9cb8cafc...cd162`/`4bf4a5dd...9e57b`；自动格式 0.875、字符串
  context recall 0.583333，只作标注辅助。标注台完成 0/24、无效上下文 0/24，状态
  `READY_FOR_REAL_HUMAN_INPUT`；机器证据见
  `experiments/week7_dialogue_repair_20260821_v2.json`，完整 unittest 422/422。

## 2026-08-22：Week 7 corrected dialogue 单人人工评估完成

- 真实单人操作者在同一 session 完成固定队列 24/24，26 条 append-only 记录包含 2 次
  revision=2；最终 24 条均有非空 reviewer、本人自审与完整四维分数，决定均为 `pass`。
- 四维均分为历史图片指代 4.541667、需求迭代 4.625000、上下文承接 4.500000、逻辑
  连贯 4.708333，未加权均值 4.59375。结果 SHA 为 `bdec2d18...af932`。
- 原人工 JSONL 留在忽略目录；机器聚合见
  `experiments/week7_dialogue_human_review_20260822_v2.json`。这是 corrected development
  人工证据，不重开 v3 test；没有产生 chosen/rejected 偏好对，DPO 继续 `SKIPPED`。

## 2026-08-22：Week 7 corrected dialogue Week 6 routed baseline

- 配置 `configs/week7/dialogue_comparison_v1.json` 绑定 corrected development、三个冻结
  Week 6 adapter SHA 与 8/8/8 路由；范围仅 development，test 未读取。
- Spartan job `29491047` 为 `COMPLETED 0:0`，L40S 单卡耗时 00:09:10。24/24 唯一
  sample、失败 0，格式合规率 1.0、context recall 0.555556、平均/中位延迟
  15449.04/13493.49 ms；raw SHA `c3effb6d...318e59`。
- 真实单人操作者完成 Week 6 routed 24/24，25 条 append-only 记录含 1 次修订；最终
  均为 `pass`，四维总均值 4.56250，结果 SHA `af3721d2...d49f93`。与 multitask 的
  描述性配对差为 +0.03125，样本级 10 胜/7 平/7 负；四维差分别为
  -0.125/+0.291667/-0.041667/0，不宣称统计显著。
- 机器证据见 `experiments/week7_dialogue_comparison_20260822_v1.json`。两轮评分由同一
  真实操作者在同一 session identity 完成，Agent 未代填。它们不是明确审核的
  chosen/rejected 偏好对，DPO 继续 `SKIPPED`。

## 2026-08-22：Week 7 audited mDPO-style v1

- 用户要求在不伪造人工身份的前提下继续对抗审查。两组真实四维评分按总分非平局、chosen
  各维 ≥4、chosen JSON、视觉证据命中、来源一致和无生成失败筛选，得到 16 对；7 个
  平局和 1 个 chosen 非 JSON 被拒绝，反转 chosen/rejected 探针 16/16 被拒绝。
- `week7_preference_pairs_20260822_v1` 锁 SHA `a29736fc...d1f38`，train/validation
  为 10/6，六个场景×chosen-role strata 各留 1 对 validation。human pair choice 并非
  显式二选一，来源准确记为真实逐输出评分的确定性派生，Agent 审计不替代人工评分。
- 单次 mDPO-style job `29491859` 在 `gpu-a100-preempt`/`spartan-gpgpu098` 完成，
  5 个 optimizer updates，train 为 0.8/+0.01861，validation 为 0.3333/-0.00981，
  未通过 0.5/>0 门禁。adapter `3791896e...39b64` 不选用，不做第二次 DPO 或 test。
- 完整机器证据见 `experiments/week7_mdpo_20260822_v1.json`。该结果是已执行但门禁失败，
  不能写成模型提升；生产/交付选择仍为 checkpoint-151。

## 2026-08-22：Week 7 终态对抗审计

- `python scripts/run_week7.py adversarial-audit` 以机器证据和跨文件身份为主要判断，真人
  development 四维评分只作辅助证据；Agent 不替换人工身份、分数或验收决定。
- 主动注入基线替换、五维碰撞、比例漂移、Schema 语义洗白、test 重跑、支持数删除、
  对话缺陷洗白、repair 读取 test、Agent 冒充人工、失败 DPO 晋级和 DPO 读取 test，
  11/11 个反事实均被拒绝。
- 保守结论为 `PASS_WITH_KNOWN_IMMUTABLE_LIMITATION`：核心自动门禁和 corrected dialogue
  development 通过，DPO adapter 拒绝；实现可以进入 `dev` 集成，但 v3 test 对话仍无效，
  因此不得宣称完整 Week 7 test 对话能力，也不得晋级 `stg`。
- 本地目录清理将 408,127,632 字节作废 v1/v2、失败构建和临时文件移入回收站，保留 v3
  锁/归档、修复 raw、真人记录、偏好锁和 DPO 证据。机器证据见
  `experiments/week7_adversarial_completion_audit_20260822_v1.json`。

## 2026-08-22：Week 7 corrected-dialogue v4 数据锁与执行链

- 配置：`configs/week7/qwen3_vl_8b_multitask_context_v4.json`，SHA-256
  `e5b76008e504e0775b62506acbeba3e38438cf14851493be512aa4325fd89b7c`。
- 数据：`week7_corrected_multitask_context_20260822_v4`；命令为
  `python scripts/manage_week7.py --config configs/week7/qwen3_vl_8b_multitask_context_v4.json build-lock --source-project-root E:\Project\Trip_Project`。
  实际锁 SHA-256 为 `000a2e57620428034da27e03ba3c92483e9c147032166ad273ed089fbb97c9fa`。
- 实测身份：train/development/test=3000/114/114；训练三核心场景各 760、
  general 270（9%）、dialogue 450（15%）；分区内重复和跨分区五维冲突为 0。
- 对已消费 v3 完整 identity manifest（3228 行）重算 v4 test 隔离，
  sample/source/image/group/template 重叠均为 0。`validate-lock` 实测 PASS 且
  `test_consumed=false`。
- 训练实现修正为 all-assistant-span SFT；评测实现修正为逐 assistant 轮生成，
  不使用金标中间回复 teacher forcing。自动门禁增加 task value accuracy、
  sequential turn coverage 和 sequential turn failure rate；已有真人结果只作辅助描述。
- 实测 v4 定向测试 10/10、全部 Week 7 测试 71/71、完整 unittest 441/441、
  Slurm shell 语法检查 2/2 通过。GPU 训练、selector 和一次性 test 仍为
  `PENDING`；未生成新指标。
- Git 提交 `d14a1292360619d47672acdeee09e88dfd408840` 已推送。Spartan 作业
  `29504508`使用 `gpu-l40s`、单 L40S、16 CPU、128 GiB，运行目录
  `work/week7_multitask_v4/run_d14a129`；当前为 `PENDING(Resources)`，尚无
  loss、checkpoint、development raw 或模型指标。test marker 未消费。
- attempt 1 终态：job `29504508`，L40S，`FAILED 1:0`，实际运行 00:18:30。
  在 step 38 首次评估时，`week7_qlora._generate_record` 将生成文本作为裸字符串
  追加到 processor-normalized conversation，Transformers 4.57.1 遍历 content block 时报
  `TypeError: string indices must be integers, not 'str'`。日志中已观测的 loss 为
  1.0281/0.6484/0.3572（step 10/20/30），但无 checkpoint 和可评分指标。
- 修复只规范化生成 assistant content block 并增加 raw/normalized text 可逆转换；
  config SHA、dataset lock SHA、run ID、数据比例与训练超参数均不变。失败目录
  `work/week7_multitask_v4/run_d14a129` 保留，不作为 checkpoint 或完成 run。
- 修复验证：strict fake processor 分别实际运行训练 development 与 final-test
  逐轮生成，旧裸字符串回填会确定失败；修复后 v4 13/13、Week 7 74/74、完整
  unittest 445/445，v4 锁和两份 Slurm shell 语法均 PASS。Week 7 JSON 与 Spartan
  shell 通过 `.gitattributes` 固定 LF，消除 Windows CRLF 对哈希和 `bash -n` 的影响。
- attempt 2：执行提交 `c002a78`，job `29505375`，初始状态 `PENDING(Resources)`；
  输出 `work/week7_multitask_v4/run_c002a78_attempt2`、日志
  `work/week7_multitask_v4/logs_c002a78_attempt2`。config `e5b76008...`、canonical
  lock `000a2e57...`、run ID 和全部训练超参数与 attempt 1 相同；未声明 checkpoint
  resume，因为 attempt 1 无 checkpoint。
- attempt 2 首次进程终态：`FAILED 1:0`，00:27:51，MaxRSS 约 19.6 GB；step 38
  development 114/114 完成，raw SHA `7c41d30...`、metrics SHA `6b7b37...`，
  checkpoint adapter SHA `f425646...`。weighted composite 0.339991、failure 0；
  商品/售后/行程 composite 0.564706/0.52/0.045，对话 automatic 0.231308。
  日志在 step 39 后只有 `Exception ignored in: <object repr() failed>`，无可归因主堆栈；
  作业非超时、非 Slurm OOM，checkpoint 文件完整且 test marker absent。
- 恢复：job `29506065` 从同输出目录 `checkpoint-38` 提交；远端仍为 clean
  `c002a78`，config SHA、lock SHA、run ID 未变。此次仅使用显式
  `--resume-from-checkpoint` 恢复能力；若相同故障重复，不再提交下一次恢复。
- `29506065` 终态 `FAILED 1:0`，00:34:51，MaxRSS 约 20.1 GB；训练越过原退出点
  并到 step 76，随后写 development artifact 时明确报 `[Errno 122] Disk quota exceeded`。
  step-76 仅创建 0-byte raw，无可评分证据。只删除项目内可再生成的 container cache
  26 GiB、pip cache 1.2 GiB 和该 partial；step-38 adapter/optimizer/raw/metrics 哈希
  均保持。GPFS `df` 从约 0.9 GiB 可用恢复为约 28 GiB。
- quota 修复后的同身份恢复 job 为 `29506362`，仍使用 clean `c002a78`、相同
  config/data/run ID 和 `checkpoint-38`；未变更任何训练超参数或已评分数据。
- `29506362` 终态 `COMPLETED 0:0`，02:37:50；global step 226，因 step 188/226
  连续两次未超过 step 151 最优综合分而按 patience=2 早停。训练 loss 0.112032，
  峰值 GPU allocated/reserved 为 34,112,156,160/40,852,520,960 bytes。run summary
  SHA-256 为 `5af980efc851e2e0c15d96ea13853e3728fa194618fcd737ea976e3926e2e5a5`；
  最终 adapter 与 best checkpoint-151 adapter SHA-256 均为
  `296ad3f362e559738b55d93e2164f549631994138f5acaed72d8b4b3b48d9d86`。
- 六个 development 候选的 weighted composite 为 step 38/76/113/151/188/226 =
  0.339991/0.787641/0.806420/0.833980/0.832830/0.817248；全程 overall/dialogue/
  sequential failure rate 均为 0，但 0/6 通过全部预注册自动门禁。
- 最佳 step 151 的格式、context recall、context-state value、task-result key/value、
  sequential coverage、automatic composite 分别为
  0.833333/0.777778/0.697917/0.833333/0.741319/0.733081/0.778585，对应阈值
  0.95/0.85/0.75/0.95/0.75/0.75/0.85，全部 FAIL。最后 step 226 相应值为
  0.833333/0.715278/0.687500/0.833333/0.741319/0.711618/0.765102，也未通过。
- selector 在空的新目标路径上实际执行并返回
  `no v4 checkpoint passed the automatic development gate`；按不可覆盖实现没有写出
  selection。corrected-dialogue v4 test 因门禁失败标记 `SKIPPED_GATE_FAILED`，test
  保持 `LOCKED_UNCONSUMED`，没有 Week 6 routed/zero-shot v4 test 指标。DPO 仍保持
  已执行一次、validation FAIL、adapter 拒绝且关闭，不因 v4 结果重开。

## 2026-08-17：Week 6 最终数据锁与 QLoRA pilot 前置验证

- Git 基线：`068b40c` 加本次 Week 6 未提交工作区；模型固定
  `Qwen/Qwen3-VL-8B-Instruct`，NF4 double quant、bf16、LoRA
  r/alpha/dropout=`16/32/0.05`，单 GPU batch 1、梯度累积 16。
- 数据源：Week 5 Spartan 最终 merge 79,936 成功、64 最终失败；三场景各 100 条
  最新人工修订。100 条人工验收对话仅作为 Week 5 证据，不进入单场景训练锁。
- 失败 v2：silver 商品标签含 Schema 允许但人工词表未允许的自由标签，错误复用人工
  校验器导致锁定停止；修复为人工修订执行 Schema+受控词表、silver 仅执行 Schema。
- 失败 v3：Windows 绝对路径虽已转为项目相对 `file://`，但 Transformers 4.57.1
  `apply_chat_template` 只自动加载 `type=image`，且本地加载器不识别 `file://`。
  该版本未用于训练。
- 活动 v4：`week6_week5_final_human300_20260817_v4`，全部视觉项为
  `type=image/path=<project-relative-path>`，路径均受项目根目录约束且文件存在。
  manifest/split SHA-256 为 `0b8d9f96...adf0e`/`450abbe7...cc0`；六份 JSONL
  计数与数据契约验证通过。
- 命令：`python scripts/prepare_week6_data.py --config
  configs/week6/qwen3_vl_8b_qlora_final300_v4.json`；验证使用完整 unittest、
  `validate-pools`、Week 3 v2 validator、六份 `validate-data`、`bash -n` 和
  `git diff --check`。
- 结果：完整 `unittest` 337/337；Week 5/Week 3 隔离验证 `status=ok`；GPU 环境、
  4bit 加载、反向传播、显存和 adapter checkpoint 仍为 `PENDING`，不得称为训练完成。
- 运行提交：L40S job `29296577`，run ID
  `week6_after_sales_8b_qlora_pilot_20260817_0b3f755_a`；32/32 样本、10 steps 上限、
  Slurm time limit 2 小时。截至 01:16 AEST 状态为 `PENDING(Resources)`，未取消或
  重提，暂无 GPU 指标。
- 自动链准备：用户直接批准 pilot 成功后继续。新增有限 loss/显存/checkpoint/adapter
  回载 gate，正式训练使用三场景任务级并行而非模型跨卡切分；全量 JSONL 使用 byte
  offset dataset，避免加载整份文件到内存。
- 数据传输：全量锁压缩包 14,293,390 bytes，SHA-256
  `1b8dc1ca792f977dfe6b448b1f8604ab6c82020321bbba49d46e2d68da6c322e`，编号分片重组后
  远端哈希一致并已展开。全量锁引用 79,937 个唯一图片，manifest SHA-256
  `1afd768a1996a7ebd7004e1ef2fcdcff60ad7a54ce4161efe6673c4e0a27e5a7`，本地全部存在；
  远端图片审计和缺失补传仍待执行，不能据此声明正式数据已就绪。
- 工程验证：完整 `unittest` 343/343，`py_compile`、五份 Week 6 Slurm shell
  `bash -n` 与 `git diff --check` 通过。
- 远端数据审计：首轮 array/merge `29297594`/`29297595` 检出 missing=15,129、
  size_mismatch=1；补传归档 562,031,601 bytes，SHA-256
  `62fe6e80ecc0bfa0cbd08a0b082fef193121248c8e0d32f71d884566ac5151e0`。复审
  `29297871`/`29297872` 覆盖 79,937 项，failures=0、status=ok。
- pilot 终态：`29296577`，02:00:30–02:00:41 AEST，`FAILED 1:0`。环境为 torch
  `2.13.0+cu130`，L40S 节点驱动报告最高 CUDA 12.8；在 `torch.cuda.get_device_name`
  前初始化失败。另有 bnb 0.50.0 缺少 kernels 提示。没有模型下载、loss、显存、
  checkpoint 或 adapter 结果。
- 修复：新增 `requirements-training-spartan-cu128.txt` 和不可覆盖的新 venv setup，
  固定 torch `2.8.0+cu128`、Transformers/PEFT/bnb `4.57.1/0.17.1/0.47.0`、
  kernels `0.11.7`。首次环境作业 `29297982` 因 GPFS 配额失败；清理容量缓存后的
  恢复作业 `29305189` 再次失败，明确定位为共享 project inode `489K/489K`，失败点
  是写入训练无关的 `pandas/tests` 文件。
- 用户明确授权后仅删除可重建缓存和两个失败/过期 venv；GPFS 可用容量由约 36 GiB
  增至 76 GiB，inode 由 223 个可用恢复为约 68K，4B hub 模型、数据、代码、日志和
  训练结果均保留。setup 改为只安装 `requirements-training-spartan-cu128.txt` 并设置
  `PIP_NO_CACHE_DIR=1`；定向测试 12/12、完整 unittest 346/346、`bash -n` 和
  `git diff --check` 通过。
- 精简环境作业 `29305905` 在 `spartan-bm178` 用时 1:55，`COMPLETED 0:0`；
  `pip check` 无破损依赖，版本为 torch `2.8.0+cu128`/CUDA `12.8`、Transformers
  `4.57.1`、PEFT `0.17.1`、bnb `0.47.0`、kernels `0.11.7`，结束时约余 37K inode。
  新 pilot `29305985` 依赖 gate `29306001`，正式商品/售后/行程作业分别为
  `29306002`/`29306003`/`29306004`，仅在前序成功时放行。四个错误完整提交哈希的
  依赖作业 `29305986`–`29305989` 已在运行前取消，未取消或重复提交 pilot。
- pilot `29305985` 实际运行 1:20 后 `FAILED 1:0`；CUDA、依赖版本和 32/32 数据验证
  均通过，8B 四个 checkpoint shard 已下载并加载，随后 `AutoProcessor` 因缺少
  torchvision 在训练 step 前失败。下游 `29306001`–`29306004` 全部自动取消，未执行
  正式训练。按 PyTorch 官方 CUDA 12.8 组合补充 torchvision `0.23.0+cu128`，并将其
  加入环境门禁；受控现有 venv 修复和新 pilot 仍待远端验证。
- torchvision 修复作业 `29309546` 在 23 秒内 `COMPLETED 0:0`，安装
  `0.23.0+cu128` 后 `pip check` 和 torch/CUDA/torchvision 导入通过，约余 37K inode。
  修复后链为 pilot `29309556`、gate `29309557`、正式商品/售后/行程
  `29309558`/`29309559`/`29309560`，绑定提交
  `fd7b806ae6e064454a5dcd91a07e3ea9b002e92c`；截至 14:50 AEST，pilot 为
  `PENDING(Priority)`，预计 2026-08-18 06:00 启动。

## 2026-08-16：Week 5 多轮对话并行生成、合并与人工抽样验收

- 模型/后端：`Qwen/Qwen3-VL-4B-Instruct`、Spartan vLLM 0.11；活动主作业保留，
  额外数组任务使用独立 run、输出、日志和临时目录，客户端并发上限为 4。
- 代码提交：并行生成 `306527e`、单源快照 `ae11be6`、多源快照 `522b4af`、验收
  队列 `f93e7ef`、图片展示修复 `48108a7`。
- 生成结果：不可变主前缀 4,000 条，四片各 1,500 条，合并 run
  `week5_dialogues_merged_10000_20260816_522b4af` 为 10,000 个唯一 ID；场景
  3334/3333/3333，消息数 8–12，缺失、重复和冲突均为 0。
- 哈希：candidates
  `7e00f326fc1b2896a6efcc5c2f6c1f67ffdb728501ba3eb9ba65efdb28265d99`；manifest
  `02795c8df44ca564dcd873974c5bcb6939c41bf38bee2f6c1f550d7916669556`；人工队列
  `45c34b558456577d5eaaf9b74cf04a8766b0160ec05935a181131db66134634e`。
- 人工结果：`Larry Fan` 实际完成固定队列 100/100，五项检查完整，全部决定为
  `pass`；验证 JSONL SHA-256 为
  `eb3a6f436a78389e919b86d3756fc2208265bac7f4420158dc597d5bc4682e54`。其余 9,900
  条保持未人工验收候选。
- 验证：完整 unittest 329/329；Week 5 `validate-pools` 返回 `status=ok`，确认
  80,000 个唯一 sample ID 和图片 SHA-256；`git diff --check` 通过。
- 结论：Week 5 在批准的抽样人工预算内闭环。本实验没有执行 Week 6 训练。

## 2026-08-14：Week 5 Spartan 最终合并与 Week 6 数据锁定

- Git 基线：`824530e` 加本次未提交工作区；Spartan merge job `29190753`，打包 job
  `29190774`，均为 `COMPLETED 0:0`。
- 合并：商品/售后/行程成功 49,957/19,991/9,988，总成功 79,936；最终失败 64，
  与 80,000 候选全集严格闭合。归档 SHA-256 为
  `a9ae67cb677bb940c94197e692ba1ce85671a83cba9e5fb070b012dfaa43abee`。
- 数据锁定：sample ID SHA-256 阈值切分，seed `20260814`，validation 5%；manifest
  SHA-256 `877c16d8ee79d9b0601fe9b6a5f531dfcbd81bb7e16f3fbd6e2526b760d62198`，split
  SHA-256 `7ec02ed629a4b434dae39c5eb32ff783ab7fafdde8ac151e4124b34a294fc018`。
- 标签策略：真实人工修订 27 条、权重 1.0；其余 79,909 条模型预标注显式标记为
  silver、权重 0.5。没有自动创建人工审核身份或金标状态。
- 验证：六份训练/验证 JSONL 全部通过数据契约；Week 5 pools 与冻结评测隔离通过；
  完整 unittest 312/312。本机环境因未安装 torch/transformers/PEFT/bitsandbytes
  返回 `missing_dependencies`，不作为 Spartan GPU pilot 通过证据。

## 2026-08-12：Spartan vLLM 容器启动修复

- `29114276`：`FAILED 4:0`，容器无 `python`、仅有 `/usr/bin/python3`；没有模型请求。
- `29116649`：改用 `python3` 后进入 vLLM 0.11.0 初始化，随后 FlashInfer 因写入已满的
  `/home/yzhang3504/.cache/flashinfer` 而 `FAILED 4:0`；没有模型请求。
- `29116828`：仅通过环境变量覆盖 HOME 不足，Apptainer 保留宿主 HOME，复现同一失败；
  因此没有模型请求或结果。
- 修复提交：`300ca04`（容器入口与版本化日志）、`270b8ba`（项目内 cache 变量）、
  `3600a7b`（Apptainer `--home` 强制绑定和可写预检）。
- 登录节点使用缓存镜像实测容器 HOME 为
  `/data/gpfs/projects/punim2936/Trip_Project_yzhang3504/20260812a/huggingface/runtime-home`。
  作业 `29116943` 已使 vLLM health 返回 200，证明容器与缓存问题修复；随后因远端缺少
  Week 4 development few-shot manifest，在首个模型请求前 `FAILED 1:0`。
- 依赖恢复归档含 4 个 JSONL 和 36 张引用图片，共 40 个文件、1,151,658 字节，
  SHA-256 `216b458546cbcc61e326c56f3b38517f1f06a96c9f7cdbec85078bee469ba0ff`；
  以不覆盖模式解压后，容器内实际加载商品/售后/行程各 12 条并通过图片字节哈希校验。
- 新作业 `29117353` 已提交，当前 `PD(Resources)`；没有完成吞吐或成功率，不反推全量时间。
- 验证：Spartan 定向 `unittest` 3/3、完整 `unittest` 300/300、`bash -n` 和
  `git diff --check` 通过。

## 2026-08-12：Spartan benchmark 失败诊断与修复重提

- 历史 job `29109265`：`gpu-l40s`、16 秒、`FAILED 1:0`。stderr 明确为
  `Apptainer/1.3.3` 缺少 `GCCcore/11.3.0` 前置模块；没有模型请求或 benchmark
  结果，不能计为 pilot。
- 修复：`scripts/spartan/week5_job.sbatch` 先加载 `GCCcore/11.3.0`；远端仓库更新到
  `a6107bd`。新增 CPU 作业创建 `Python/3.11.3` 虚拟环境并安装基础和 Week 6 依赖。
- 新提交：环境 job `29114275` 于 3 分 48 秒后 `COMPLETED 0:0`，Python 环境实际为
  torch `2.13.0+cu130`、transformers `5.15.0`，`pip check` 和 accelerate/
  bitsandbytes/peft/torch/transformers 导入通过。L40S benchmark job `29114276` 于
  20:40:58 AEST 在 `spartan-gpgpu006` 启动，当前仍在等待回环 vLLM 健康；尚无
  吞吐、成功率、checkpoint 或模型结果，不提交剩余分片。
- 存储：`/data/gpfs` 467/375/93 GiB（总/已用/可用），Trip 版本目录 99 MiB；全部
  新文件和缓存限定在 Trip 专属目录，未访问或修改其他成员项目内容。

## 2026-08-12：`trip-api-sg` CPU 展示部署

- 输入：明确获批的 `src/`、`data/samples/`、Dockerfile、CPU 展示 Compose、
  `requirements-api.txt`、状态 JSON 和 Week 5 质量报告。
- 版本目录：`/opt/trip-display/20260812a`；上传归档 SHA-256：
  `404e7a681bdf35a839de56298568960a950203a21d9f7ae61b7dac4fdbe8a81d`。
- 实测：`ota-trip-display-api` healthy，绑定 `127.0.0.1:8010`；health、
  `/v1/project-status` 和静态报告均成功。原 `ota-trip-api` 的 `127.0.0.1:8000` health
  同时成功。
- 资源边界：CPU-only；没有 CUDA、vLLM、模型权重或实时 LoRA 推理，没有安全组或公网
  端口变更。
- Spartan 核验：门户显示当前登录为第三方账户 `yzhang3504`。为遵守 ADR-020，未代替
  账户所有者运行 quota/scratch 命令或提交 Slurm；本次没有 GPU benchmark 结果。

> 身份更正：用户随后确认 `yzhang3504` 为本人持有并授权本项目使用。后续允许 Agent
> 代理核验和提交，但密码不落盘，且只允许使用新建的 Trip_Project 专属目录和本项目
> job ID；本段原记录仍表示更正前未提交作业的真实状态。

- 后续实测：account/project=`punim2936`，QOS 包含 `publicgpu`；home 51.2 GiB quota 已满，
  project GPFS 可写且约余 93 GiB。新建隔离目录
  `/data/gpfs/projects/punim2936/Trip_Project_yzhang3504/20260812a`。
- 公共分区待排观测：A100 约 280、H100 约 80、L40S 约 8。按最短总时间策略只提交
  L40S benchmark job `29109265`；状态 `PD(Resources)`，预计启动
  `2026-08-12T20:27:34`。运行时采用项目缓存内的 Apptainer 1.3.3 + vLLM 0.11.0。

## 2026-08-12：Spartan migration 准备（未运行 GPU）

- 原因：用户报告 A10 因欠费停机，活动 run 无法访问；未释放实例和数据盘。
- 输入：不可变 80,000 条候选；本地 run B 可验证成功 15,166，全部为商品场景。
- 命令：`python scripts/manage_spartan_migration.py prepare ... --shard-count 4
  --benchmark-count 100`。
- 产物：`week5_spartan_migration_20260812_a`，benchmark 商品/售后/行程为
  49/36/15；4 个分片分别为 16,093、16,106、16,169、16,366，总计 64,734。
  加上 benchmark 和恢复点后覆盖 80,000 个唯一候选。
- 模型边界：Week 5 仍为 `Qwen/Qwen3-VL-4B-Instruct`；Week 6 配置固定 8B、NF4、
  double quant、bf16 和 LoRA 16/32/0.05。
- 运行状态：仅生成本地 manifest 和作业模板，Spartan project/quota/queue 未核验，
  未提交 Slurm、未产生 GPU 吞吐、训练损失、费用或新模型输出。
- 工程验证：新增定向测试 7/7、完整 unittest 299/299；Week 5 pools 和 Week 3 v1/v2
  验证为 `status=ok`，Git for Windows Bash 对 4 个 Slurm shell 文件执行 `bash -n`
  通过。展示 Compose 使用脱敏 `DISPLAY_DATA_DIR` 展开通过。

## 2026-08-12：Week 5 全量预标注 ECS 原生续跑

- 代码提交：`a9a8d99`；运行仍为
  `week5_full_preannotation_qwen3_vl_4b_20260809_b`，未创建替代 run。
- 部署输入：80,000 条候选、80,000 张唯一图片、图片缺失 0；迁移 payload SHA-256
  `735b64305cf84790937ae94e63057f8ffcddb07d08ec812a0bd075c03e389bbc`。
- 身份校验：canonical config SHA-256
  `dd94313dfa0dbd070e11270ec70157ea60c5e9162ca4a56e76ca461c46d05484`，三份候选
  manifest 哈希与历史 run manifest 完全一致。
- 切换点：本地 checkpoint 15,190；`results/attempts/failures` 分别为
  15,166/15,329/44，均逐行 JSON 合法。服务器恢复后首次确认点为成功 15,197、
  checkpoint 15,209、连续请求失败 0。
- 运行方式：systemd 常驻，vLLM 仍只通过服务器回环地址访问；本地 supervisor、runner
  和 SSH 隧道已停止。迁移不代表全量预标注、人工标注或质检完成。
