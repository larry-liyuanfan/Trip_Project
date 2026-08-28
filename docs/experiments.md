# Experiment Notes

## 2026-08-24：Week 7 v4 fix2 门禁对齐与 fresh 数据锁

- 历史输入：fix1 step 151 为 12/13 门禁通过，只有旧 sequential coverage
  0.725585 < 0.75；这只用于定位缺陷，没有据此降低阈值、选择 fix2 样本或重算 fix1。
- 实现：`gate_aligned_v2` 将嵌套 JSON 从顶层对象全等改为叶子值准确率；programmatic
  silver 自由文本 evidence 不再充当 hard-gate 的逐字视觉金标；新增 protocol coverage
  和 semantic accuracy 独立支持数。训练目标改为 gate-first feasibility，PASS 候选优先，
  未 PASS 候选由最弱门禁进度主导，最终仍按原 weighted composite/最早 step 裁决。
- 数据：`week7_corrected_multitask_context_20260824_v4_fix2`，3000/114/114；train 为
  商品/售后/行程 600/840/840、通用 270（9%）、对话 450（15%）。canonical lock
  SHA-256 `86a4360142c2517e46460cefc575131940989aa8129eca236c68eaaf71e5b14b`；
  train/development/test SHA-256 为 `cc21a001...07ced`、`b157eace...025a4`、
  `1c79407f...c8ede`。五维跨 split 冲突 0，v3/首版 v4/fix1 全量身份排除。
- 训练：Spartan job `29540085` `COMPLETED 0:0`，03:35:20，step 301 早停；train loss
  0.160256，峰值 allocated/reserved GPU memory 为 15,166,590,464/25,071,452,160 bytes。
  selector 对 8 个 checkpoint 重算，5 个通过门禁，锁定 step 226；development weighted/core/
  dialogue automatic 为 0.796113/0.746154/0.995949，失败率 0。
- 一次性 test：job `29544969` `COMPLETED 0:0`，24 条/角色，最终 comparison SHA-256
  `047d48bd40db7e06110063687e2fdb3b52801e856ae438fdaec02980b8a68e00`，consumption marker
  SHA-256 `3c9370b40f137d521f853f7534e57f57f50588a8ef938530fb9510e6ef50067b`。
  multitask/Week 6 routed/zero-shot 自动综合分为 0.793399/0.152144/0.174505；multitask
  格式/上下文召回/上下文值/失败率为 0.916667/0.750000/0.719444/0.041667。最终门禁
  `FAIL`（10 项绝对阈值失败），不重跑、不重新调参；人工评估未执行。DPO 保持一次
  validation FAIL 后关闭。
- 终态工程验证：fix2 定向 54/54、完整 unittest 454/454、`validate-lock`、config loader、
  两份 v4 Slurm `bash -n` 与 `git diff --check` PASS。immutable data-lock 摘要保留训练前
  `test_consumed=false`；实际单次消费由上述独立 marker 记录，未改写数据锁。

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

## 2026-08-23：Week 7 corrected-dialogue v4 fix1 gate repair

- 执行身份：Git `6bb5322f9f0b1daa3004bab27c0884c4bd6971fd`；config SHA-256
  `42ac8657bf21dd0887ab53acbce68e0ab074aa5c5c9e0044b802d2f4a3003de6`；dataset
  canonical lock SHA-256 `7f66795c69f8cb35cafa712e7847155708a662b88d069824b60706f6903ea9a7`；
  run ID `week7_fix1_multitask_context_sft_20260823_v4`。config、data、run 与首版 v4
  完全分离，后者只作为排除 manifest 与历史证据。
- 锁：train/development/test=3000/114/114；train 为商品 600、售后 840、行程 840、
  general 270（9%）、dialogue 450（15%）。identity manifest 文件 SHA-256
  `f3ae7c2f...d6d54`，train/development/test 文件 SHA-256 分别为
  `a5be73aa...94323`/`b7e79e56...be612`/`a4098010...dd60`；三分区五维冲突 0，
  且排除 Week 3、Week 6、v3 与首版 v4 已消费身份。
- 方法：Qwen3-VL-8B、NF4 4bit、r=16/alpha=32/dropout=0.08，attention 与视觉
  projection LoRA，lr=1.5e-4、weight decay=0.03、max grad norm=1、gradient
  checkpointing、effective batch=16。训练样本 loss multiplier 为商品/售后/行程/
  对话/general=0.8/1.1/1.1/1/1；对话使用显式工具请求、gold-plus-anchor 和
  3072 max-new-token，全部在运行前锁定。
- job `29526506` 因 HF_HOME 错指仅约 35 MiB runtime-home 失败。修复只把 HF_HOME
  指向已确认的 25 GiB `huggingface/hub` cache，并使用非覆盖输出
  `run_6bb5322_attempt2`；config/data/run/git identity 不变。恢复 job `29526965`
  为 L40S、16 CPU、128 GiB，`COMPLETED 0:0`，耗时 02:43:24，没有再次操作失败 job。
- 训练 global step=226/计划 376，step 188 与 226 连续未超过 step 151，故按
  patience=2 早停。train loss=0.182206；峰值 GPU allocated/reserved 为
  15,191,208,448/31,545,360,384 bytes。run identity 文件 SHA-256 `03663dad...69dbf`，
  run summary SHA-256 `6d5400fd...491d0`，best/final adapter SHA-256
  `b42aeeb...5131bc`。

| fix1 step | weighted | core weighted | dialogue automatic | format | context recall | sequential coverage | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 38 | 0.353427 | 0.391269 | 0.202059 | 0.708333 | 0.371528 | 0.334551 | FAIL |
| 76 | 0.729860 | 0.735654 | 0.706683 | 0.833333 | 0.826389 | 0.711825 | FAIL |
| 113 | 0.751086 | 0.725154 | 0.854817 | 0.958333 | 0.833333 | 0.724458 | FAIL |
| 151 | 0.764049 | 0.735654 | 0.877630 | 1.000000 | 0.854167 | 0.725585 | FAIL |
| 188 | 0.753292 | 0.725154 | 0.865844 | 0.958333 | 0.840278 | 0.733360 | FAIL |
| 226 | 0.752986 | 0.725154 | 0.864316 | 1.000000 | 0.854167 | 0.733084 | FAIL |

- 每次 development 支持数为商品/售后/行程各 30、对话 24。最佳 step 151 的三场景
  composite=0.153846/0.970000/1.000000，context-state value=0.791667、task key/value=
  0.962384/0.820023、initial stable=0.931548、anchor=1、tool protocol=1、overall/
  dialogue/sequential failure=0，平均延迟 11,503.48 ms。唯一未通过项为 sequential
  coverage 0.725585 < 0.75。
- selector 重算六个候选后写出 `BLOCKED_NO_ELIGIBLE_CHECKPOINT`；evidence 文件
  SHA-256 `782e92ab673c8628861af8e1eb6247454f3c8c9f608c9888899ae3eec64cc104`，内部
  canonical selection SHA-256 `e2069003...c3572d`，eligible_count=0、selected=null、
  `test_read=false`。六份 development metrics SHA-256 依次为
  `51bb3b4f...5c1d`、`2cedb294...2659`、`e3b7dd18...874c`、`cc222f31...2f3c`、
  `38dbd617...6edf`、`7ad6b83c...3a21`。
- fix1 one-shot test 未提交，marker 不存在；因此没有 fix1 Week 6 routed/zero-shot test
  对比，不能以 development 或历史 v3 test 替代。DPO 仍为既有一次 validation FAIL
  后关闭；本实验不快进 `dev`、不进入 `stg`、不打标签。
- 代码与证据复验为 fix1 定向 26/26、Week 7 79/79、完整 unittest 450/450；config
  解析、fix1 lock/five-dimension isolation、两份 v4 Slurm `bash -n` 和
  `git diff --check` 均 PASS。

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
## 2026-08-24：系统收敛修复与 CLIP/Milvus 实测

- Git 基线：`dev` 提交 `2e44125`，模型为
  `Qwen/Qwen3-VL-8B-Instruct` + Week 7 unified adapter；本地未加载该 adapter。
- fresh 数据：`system_repair_fresh_multitask_20260824_v1`，train/development/test
  1,980/168/120，lock SHA-256
  `15fe49114dcfd54019b742a4e551114c8c50d4cc54f09479ed4f0abfba3f8366`，五维隔离 PASS，
  test 未消费。
- Week 5 v2：80,000 候选、历史成功 79,936、修复队列 64、替换 44，五维评测冲突 0；
  状态 `AWAITING_MODEL_REPAIR`。
- CLIP：`openai/clip-vit-base-patch32`，本地 RTX 4070 CUDA，1,000 个真实 OTA 图片，
  512 维且全部 L2 normalized。
- Milvus：`2.6.20`，HNSW/COSINE，M=16、efConstruction=128、ef=64；索引 4.6205 秒，
  100 个查询平均/P95 2.2355/2.4097 ms，Recall@10=1.0，CRUD 全部通过。
- 验证：定向 25/25、当前工作树和全新 checkout 的完整 unittest 均 482/482，配置校验和
  `git diff --check` 通过；3 个依赖 ignored Week 7 产物的测试已改成自包含 fixture。
- 未运行：Qwen3-VL Prompt pilot、64 条 Week 5 修复、继续 SFT、fresh test 和 OSS 上传。
  未生成或声称任何新模型提升。

## 2026-08-25：系统修复 Prompt、Week 5 与 continuation SFT

- Prompt pilot：run `system_repair_prompt_pilot_20260824_v8` 在固定 development 集比较
  current、compact 和 evidence 三候选；商品/售后/行程分别选择 compact/evidence/current。
- Week 5 修复：Spartan job `29560346` 完成 64/64，最终合并产物 80,000/80,000
  Schema-valid；SHA-256 `86b0a158567da3e3b683fd73476d51f1608ad6f59ae5219e7f52354180ff5926`。
  新结果均为 silver，人工 accepted 统计不变。
- continuation SFT：job `29562078`，`Qwen/Qwen3-VL-8B-Instruct`，初始 adapter
  checkpoint-226 SHA-256 `ccc6062f...5f24ee`，单个 L40S，学习率 `5e-5`，最多 1 epoch，
  development 约每 10% step 评估，patience=2。作业 `COMPLETED 0:0`，`04:48:36`；
  step 100/112 连续未提升后回载 checkpoint-87。
- checkpoint-87：总体加权 0.920725、核心三场景加权 0.905382，商品/售后/行程
  0.716146/1.000000/1.000000，
  对话自动综合 0.982097，失败率 0；adapter SHA-256 `c2fbb5c7...eaa2a`，回载通过。
- 同集比较：job `29565493` 完成旧 unified 与 zero-shot 后暴露单场景汇总和失败前原始
  输出持久化缺陷；修复后 job `29567157` 完成 Week 6 routed。候选/旧 unified/zero-shot/
  Week 6 routed 总体加权分别为 0.920725/0.750034/0.084010/0.061806。
- 不可覆盖开发门禁 `system_repair_development_gate_20260825_v4` 为 `PASS`，失败项 0，
  允许消费一次 fresh test；门禁 SHA-256 `e7ba5bc7...0402`。最终测试 job `29569338`
  在 A100 上 `COMPLETED 0:0`，耗时 `00:42:49`，120/120、失败率 0。
- fresh test：总体/核心三场景加权 0.936170/0.926880；商品/售后/行程
  0.780639/1.000000/1.000000；三场景 JSON/Schema 均 1.0；对话自动综合 0.973330。
  商品风格/设施/价位支持为 25/30/5，没有通过删除困难样本降低支持数。
- raw/metrics SHA-256 为 `34446498...eb19`/`853bd67e...1018`，与 completed 单次消费
  标记一致。不可覆盖 final gate 为 `PASS`，失败项 0，SHA-256 `9574b05b...a77d`。
- 发布配置已绑定 checkpoint-87。生产 smoke 的前序失败依次定位输入上下文误传和
  arbitrary-object 对话状态与 `lm-format-enforcer` 不兼容；失败原始输出保持不可覆盖。
- 最终 smoke job `29571134` 在 A100 20 GB MIG 上 `COMPLETED 0:0`，耗时 `00:01:22`。
  商品/售后/行程首轮 Schema-valid；对话经一次模型级纠错后达到 `DIALOGUE_BETA`。
  成功结果 SHA-256 `a256c64a...8f32`，绑定 adapter `c2fbb5c7...eaa2a` 和发布配置。
- 最终本地私有包 runtime/adapter/retrieval/evidence SHA-256 分别为
  `ae61fb86...0f72`、`f74c0787...619d`、`3cdb98f4...1a15b`、`3ab0c024...2a7`；
  evidence 12 份。OSS 尚未上传，未进入 `stg`。
- 导师随后明确只要求模型可交接，不要求 Spartan、OSS 或逐周全量运行数据留存。
  `verify_model_handoff.py` 对唯一交接包复验 `PASS`；该结论只验证已封装身份和证据，
  不生成新模型指标。
- 清理 21 个 ignored 目标、释放 71,735,466,519 字节；保留唯一约 59.9 MB 交接包。
  历史报告与哈希结论不改写，原始大数据和中间 checkpoint 不再作为接手依赖。

## 2026-08-26：Week 8 商品理解、对话、延迟与检索

### 商品 fresh source 与数据锁

- Git：source build `5b97a2c`，正式 v4 锁代码/config `995f43d`。
- 官方重建：download `29627585`、source rebuild `29627942`，均 `COMPLETED`。
- 正式 source build `29628987`：3,000 candidates、851 validated、800 selected；
  historical hash/unreadable/internal duplicate 拒绝 `2140/6/3`；manifest SHA-256
  `582f7e47...ce195`。
- 正式 v4 lock `29629630`：train/dev/test=`400/60/60`，五维隔离 PASS，lock SHA-256
  `49d238b0...11e7f`，全部 `programmatic_silver`、human=0。
- 失败证据：`29628510` 缺 pydantic、`29628573` 类别配额不符合实际源、`29628676`
  暴露历史图片哈希冲突、`29628863` 暴露不可读图、`29628924` 暴露合法候选短缺；均在
  正式锁或 test 前失败，未消费 test。`29629051`/`29629506` 暴露切片支持设计短缺；
  v3 中间锁 test 保持未消费。

### 商品 Prompt development 与 final

- 模型/adapter：正式 `Qwen/Qwen3-VL-8B-Instruct` + checkpoint-87，adapter SHA-256
  `c2fbb5c7...eaa2a`；max_new_tokens=384，deterministic。
- dev job `29632502`，L40S，`00:07:21`：current/field-check/evidence composite
  `0.766765/0.815131/0.698464`。field-check 通过严格提高、JSON/Schema/失败/支持非回退
  门禁并锁定；selection SHA-256 `db60824a...5c90e`。SFT=`SKIPPED_NOT_NEEDED`。
- 唯一 final job `29632815`，L40S，`00:04:45`：composite
  `0.804239→0.861085`，comparison SHA-256 `2d01ec7a...944a7`，marker=`COMPLETED`。
  没有根据 test 继续调参或重跑。

### 对话与延迟

- jobs `29627793/29628024/29628215` 分别对应 v1/v2/v3。v2 最优：首轮三键合规
  `0.5`、纠错 `0.5`、上下文召回/值准确率 `0.8182/0.8182`、失败 `0.25`；v3 严格
  schema 产生未终止字符串，失败 `1.0`，拒绝。
- 固定 L40S 商品 latency：current/bounded mean `1907.79/1903.28 ms`，P95
  `1918.50/1914.19 ms`，5/5 exact match，tokens 不变。冷启动 `24.67 s`，峰值
  allocated/reserved `6.69/8.43 GB`。结论：无实质延迟提升。

### 检索

- 正式图像 overlay job `29628014`，1,000/1,000；development `29628015` 锁定
  metadata rerank；唯一 final `29628157` 完成。
- final CLIP→rerank：NDCG@10 `0.125654→0.506740`、Recall@10
  `0.018090→0.133046`、过滤正确率 `1`、失败率 `0`、可追溯率 `1`；mean latency
  `1.3768→1.5697 ms`。这是离线银标基准，不代表 Milvus 网络路径。

## 2026-08-27：Week 8 全自动扩展实验

### 商品 fresh v7 与 Prompt

- 身份：`Qwen/Qwen3-VL-8B-Instruct` revision `0c351d...`，正式 checkpoint-87
  adapter SHA-256 `c2fbb5c7...eaa2a`；新数据和 target 全部为 `programmatic_silver`，
  human annotation/review/acceptance=`0/0/0`。
- source build jobs `29637053/29637170`：v2 因 post-hash 酒店支持不足 fail-closed；v3
  根据实际合法上限完成 1,000 条 source，餐饮/景点/酒店=`992/7/1`，manifest SHA-256
  `5c538740...a6e`。v7 lock job `29637462` 完成 `400/60/60`，五维隔离 PASS，test
  初始为 `LOCKED_UNCONSUMED`。
- Prompt development job `29637779`，A100 20 GB MIG，`00:14:40`。current/field-check/
  evidence composite=`0.782941/0.836536/0.740866`；field-check 的业态、风格 micro-F1、
  设施 micro-F1、完整性分别为 `0.916667/0.734375/0.820809/0.803333`，JSON/Schema
  `1/1`、失败率 `0`。selection SHA-256 `35abf1b6...c4fae6`，test 未消费。

### 可观察证据与 continuation SFT 负实验

- hard-slice lock job `29637171`：train/dev=`400/60`、test 不包含且未访问、lock SHA-256
  `cdd56c66...0401e`。caption proxy 很稀疏：train unknown category/style/facility/price
  `385/400/387/400`，dev `58/60/58/60`，因此不具备替代商品正标签的支持。
- 两阶段 development 首个 15 分钟 job `29637294` 超时且无结果目录；以同一 identity、
  45 分钟恢复的 job `29637921` `COMPLETED 0:0`，`00:19:16`。composite `0.352974`、
  evidence Schema pass `0.266667`、failure `0.733333`，明确拒绝。
- continuation SFT job `29637514` 从正式 adapter 继续，LoRA r/alpha/dropout=`16/32/0.08`、
  LR `1e-5`、silver weight `0.5`、最多 1 epoch。首个 10% checkpoint-5 composite
  `0.369804`、failure `0.683333`；同一 hard-slice development 的未训练两阶段基线为
  `0.352974/0.733333`，改善不足以解决高失败率。结合 silver target 支持方向错配，在
  step 10、第二次评测前主动停止，未读取 final test；不与其他数据身份的 Prompt 分数作
  同口径比较。
- checkpoint-5 adapter SHA-256 `a94f9f75...e2249`；CPU adapter-only 回载 `PASS`，292
  个 LoRA tensor，结构与正式 adapter 一致。该 checkpoint 仅作失败证据，最终 adapter
  继续选择正式 checkpoint-87。

### 对话、商品延迟与检索

- runtime v7 job `29637886`，固定 5 条 dialogue + 600x400 商品真实图片。确定性候选的
  三键合规、状态召回/值准确率/精确率/整状态准确率=`1/1/1/1/1`，纠错/失败/fallback
  `0/0/0`；current 分别为合规 `0.4`、召回/值准确 `0.5/0.5`、纠错/失败 `0.6/0.6`。
- 商品 current→selected release：mean/P50/P95
  `5006.81/5006.33/5028.50→5000.55/5002.16/5009.02 ms`，tokens 相同、Schema `1`、
  failure `0`、5/5 exact match。v6 图片 cap + cache 虽有 `5/6` cache hit，但 mean
  `4871.60→4874.44 ms`，因此不进入 v7 release。
- retrieval v3 development job `29636996` 锁定真实 Milvus Lite `hybrid_weighted`；唯一
  final job `29637070`。final NDCG@10/Recall@10
  `0.125654/0.018090→0.564459/0.142734`，P95 `12.756→14.905 ms`，无 offline
  fallback、失败率 `0`、过滤正确率/可追溯率 `1/1`。

### 单次商品 final 与 release smoke

- 唯一 final job `29638144`，A100 20 GB MIG，`00:09:58`。current→field-check：
  composite `0.819003→0.857729`、业态 `0.966667→0.950000`、风格 micro-F1
  `0.755906→0.753846`、设施 micro-F1 `0.695652→0.834286`、label completeness
  `0.749167→0.819722`、price unknown `0.033333→1.0`、unknown exact
  `0→0.033333`。JSON/Schema `1/1`、failure `0`，metric support
  business/style/facility/known-price=`60/60/60/0`，price-unknown=`60`。
- mean/P50/P95 `4701.94/4713.56/4993.48→4609.50/4606.05/4733.06 ms`；input/output
  tokens `40215/3544→39975/3467`。comparison SHA-256 `5dc83953...f3829`，marker
  `COMPLETED`；没有重跑或 test 后调参。
- release smoke job `29638236`，A100 20 GB MIG，`00:01:07`，状态 `PASS`。商品/售后/
  行程首轮 Schema-valid；对话 `DETERMINISTIC_CONTRACT`、无 fallback/attempt，达到
  `DIALOGUE_BETA`。smoke SHA-256 `086133ec...85030`，release config SHA-256
  `9defb3e7...ef749`，adapter SHA-256 `c2fbb5c7...eaa2a`。
- 终态复验：完整 unittest `594/594 PASS`；远端 v7 lock validator 为 `PASS`，唯一
  `test_consumption.json` 为 `COMPLETED`，其 comparison SHA-256 与 final 输出一致。
  Python `compileall`、`git diff --check`、tracked secret signature scan 和大于 10 MiB
  的 tracked file scan 均为 `PASS`。

## 2026-08-27：Week 8 剩余优化 development 实验

### 商品 Prompt refinement v8

- Git commit：`40d12c7`；job `29643869`，A100 80 GB PCIe MIG 1g.20gb，
  `COMPLETED 0:0`，`00:15:11`。
- 配置：`configs/week8/product_prompt_refinement_v8_development.json`；绑定 v7 lock
  SHA-256 `321bea49...b0301`，test policy 为 `DISABLED_DEVELOPMENT_ONLY`。
- current/field-check-v2/uncertainty composite=
  `0.836536/0.701144/0.703235`；业态=`0.916667/0.900000/0.950000`，风格 micro-F1=
  `0.734375/0.676471/0.724409`，设施 micro-F1=`0.820809/0.388060/0.356757`。
  三者 JSON/Schema=`1/1`、failure=`0`、known-price support=`0`。
- 两个候选均失败于 `composite_not_strictly_above_current_release`；selection SHA-256
  `110d3630...aef8a`。没有 final 路径、test read 或消费标记。

### 商品 silver/OCR source audit v8

- Git commit：`7bde26f`；CPU job `29643962`，`COMPLETED 0:0`，`00:01:12`，
  MaxRSS `388264K`。
- 审计结果/candidate manifest SHA-256=
  `b425ab81...9e29`/`6ca17cc5...a32d`；human annotation/review/acceptance=`0/0/0`。
- pre/post historical-image-hash candidates=`45/8`；确认可见 amount/tier/正 price-range
  支持=`0/0/0`。未使用 v7 的 480 张图均为 restaurant，caption style/facility=
  `0/12`，metadata price=319 仅作非视觉信息。
- 结论：现有未消费数据不支持另一次完整 continuation SFT；保持 checkpoint-87，未读取
  v7 final rows/outputs。

### 商品 prepared-input cache v8

- Git commit：`40d12c7`；job `29643870`，A100 MIG，`COMPLETED 0:0`，`00:02:45`。
- 固定图片 SHA-256 `90595c2b...f542`，每侧 10 次。current/cache mean/P50/P95=
  `4845.46/4839.64/4877.90` / `4868.88/4858.77/4920.32 ms`。
- cache hit/miss=`11/1`，10/10 exact、input/output tokens=`7370/590`、Schema/failure=
  `1/0`；性能回退，候选拒绝。证据 SHA-256 `83e8b2ce...1161`。

### 检索有界 metadata LRU v5

- 前置 v4 job `29643904` 首次比较 pool100/50/25。pool100 cache 保持
  NDCG@10/Recall@10=`0.584776/0.172498`，P95 `10.2815→9.2886 ms`；pool50/25 质量
  回退。由于 v4 没有显式容量以及预计算/内存证据，它只保留为中间实验，随后以新 v5
  identity 修复，不覆盖 v4 输出。
- job `29644063`，4 CPU，`COMPLETED 0:0`，`00:00:47`；真实 backend
  `milvus_lite_flat_cosine`，offline fallback=false。
- config canonical SHA-256 `95a82cc9...acd5`；lock index/development/final=
  `582/127/0`，排除历史 v3 query 291，五维隔离 PASS，lock SHA-256
  `a5fdf0a1...fbc9`。
- pool100 uncached/LRU512 的 NDCG@10=`0.584776/0.584776`、Recall@10=
  `0.172498/0.172498`、support=102、filter/trace/source/failure=`1/1/1/0`。
  mean/P50/P95=`9.6001/9.2823/9.6339→8.3079/8.1101/8.4247 ms`。
- 预计算 `1602.23 ms`、tracemalloc peak `22,991,100 B`、最终 entries/capacity=
  `393/512`、evictions=0；稳态 hit/miss=`2484/0`。metrics/results/refs/selection SHA-256=
  `1686ad20...0388`/`8f2090bf...17e3f`/`b45ac704...3053`/`61b6d8e4...8150`。
- 结论：锁定 development 候选 `hybrid_weighted_pool100_lru512`，但不将其描述为已进入
  正式 API/release，未执行 final。
- 续行终态验证：相关定向 `76/76`、完整 unittest `609/609 PASS`；compileall、三份新增
  Slurm 脚本 `bash -n`、`git diff --check`、tracked secret/large-file scan 均为 `PASS`。
  release config/正式 adapter SHA-256 复算为
  `9defb3e7...ef749`/`c2fbb5c7...eaa2a`。

## 2026-08-27：全项目复审与商品证据诊断

### 固定 development 与数据口径

- `configs/week8/product_review_v1.json` 使用现有 v7 development 全部 60 条，未读取已消费
  final 标签；数据锁 `321bea495df6e53813d79caa93fcd3478391ecf0b613f972500f7463224b0301`。
- 自动审计发现 metadata 代理 60/60、业态 known/unknown 矛盾 56/60、混合风格/设施
  provenance 错误 60/60。全部为 `programmatic_silver`，human annotation/review/
  acceptance=`0/0/0`。因此匹配分不能解释为视觉准确率。
- 固定索引 0/15/30/45 的自动图像定性检查发现不可见停车场仍被输出为 `parking`。
  该检查不产生 gold、不改变 target、不用于估计总体视觉准确率。

### 运行与重计分协议

- 首轮代码 `a099f3f`，Spartan job `29664584`；四组依次为现有商品 Prompt、旧证据链、
  Schema 可见/负证据约束链、同底座禁用 adapter 消融。正式 checkpoint-87 权重未改动。
- 首轮保存的原始输出和指标均不覆盖。复审发现失败占位 JSON 会取得部分分数后，新增
  `week8_product_failure_zero_credit_v2`：失败样本留在分母但不给格式或语义分，原始输出
  保持不变。使用 `scripts/review_week8_product.py --rescore-dir ... --output-dir ...`
  校验原 SHA 后另存结果；重计分不再次调用模型。
- 首轮基座受约束输出出现字符串中途终止，因此新增 `configs/week8/product_review_v2.json`
  和 job `29666004`。同一 development、底座、Prompt 和 256-token 证据预算，只取消
  生成时的约束解码；完整 JSON/Schema 后校验与重试保留。另测当前商品 release，以免
  把跨硬件的耗时变化解释为优化效果。
- job `29666004` 依赖首轮成功结束，随后才在原项目目录将 feature checkout 快进到
  `f129ea8`；不与首轮共享 GPU 并发推理。申请 25 分钟，覆盖 120 条推理、四场景 smoke、
  10 条对话对照与固定图片重复基准及加载余量。
- 完整数值与任务终态在商品报告第 12 节记录；本次不训练、不运行新 final、不修改正式
  release，不把格式修复或银标得分变化写成已验证的视觉能力提升。

### 观测结果与再次修复

- 两作业均 `COMPLETED 0:0`，耗时分别 `47:37`、`13:40`，硬件均为 NVIDIA A100
  80GB PCIe，torch `2.8.0+cu128`。不是 MIG 基准，不将跨运行加载时间用于提速结论。
- 四组完整 development 的修正 composite 依次为 `0.836046/0.587549/0.694199/0.269641`，
  请求失败 `0/17/0/33`；Schema 可见契约消除本轮 adapter 证据链失败，但仍猜测不可见
  设施、银标设施 F1 回退。该诊断不获选为发布 Prompt。
- 基座取消约束解码后 composite `0.510065`、失败 `2/60`，原始 JSON syntax `100%`，
  Schema `96.6667%`。两次剩余失败是重复观察事实，不是 JSON 中文字符串截断。模型
  观察事实仍存在错读；无整体视觉准确率提升结论。
- 重计分使用 `f69797e`，原 raw SHA 全部校验，`new_model_requests=0`。v1/v2 新 summary
  SHA 分别 `add2379dc9f2d23b88390882bc42d5d931d831e491799a8614cf870140df8dd3`、
  `259b9ce9b97193095cb7803e1cb0ecb10b4da8b92d0fc2ad4da2d9c699ff23a6`。
- 10 条 runtime v2 发现取消预算未置 null、非法负天数被 fallback 改成正数；`f58707c`
  增加取消解析、被拒字段保护及部分失败回复。45 项新增定向和 654 项全量测试通过。
- 默认 cafe 图实际是 64×64 图形占位图；前两作业 smoke/固定输入重复延迟保留为连通性
  证据，商品 60 条 development 对照使用的是真实图片，不受此问题影响。
- 最终真实图片 job `29666837` 在同一项目目录、代码 `f58707c` 运行，申请 15 分钟，
  实际 3 分钟 `COMPLETED 0:0`。`configs/week8/runtime_review_v3.json` 固定原 10 条
  对话及 533×400 development 图片，SHA `4522e1aa84ef6f0800b2b138068f56db88e8096a622ef1f842e652b9024cf6d8`。
- 真实图片对话 current/candidate 首轮格式 `0.9/1.0`、纠错 `0.1/0`、失败 `0.1/0`、状态
  exact `0.4/1.0`、状态值正确率 `0.8/1.0`（support=25）。candidate 仅儿童变更使用
  1 次语义 fallback。此结果只证明状态/契约，不能代表推荐任务完成。
- 商品延迟加载 `21434.521 ms`，512/384 上限各 5 次 mean `3894.280/3901.957 ms`，
  P50 `3902.640/3908.000 ms`，P95 `3943.654/3926.002 ms`；tokens 均 `3565/285`，
  Schema `1/1`、failure `0/0`、输出 exact `5/5`。未证明稳定提速；既有停车场猜测仍在。
- 真实照片 smoke SHA `b6532cf4f2cbc15db604537909e2da95f31222fc68713e472677a4bc6f8d0734`；
  runtime SHA `0701c1e7299c8c3e0c90b241273d4602f555bb409af76c285363ee393e6742a4`。
  smoke 的商品/售后/行程 Schema 均通过，对话走确定性契约；行程仍复述模板，不能将
  `PASS` 写成业务语义全部正确。最终保留 v7 RC Prompt 和 checkpoint-87，不安排人工工作。

## 2026-08-28：Week 8 九项审查修复验证

- 执行代码：`327f764`；配置 `configs/week8/audit_repair_v1.json`，SHA
  `2b30dccebddab76e5d82766987de1ea78d57b9a824684c79e157d9e2c607ea67`。
- 数据：原 v7 development 60 条，原生 caption parquet；另存 `caption_evidence_v2`，
  human=0，silver=60，原图片/身份不变。五维隔离 PASS，train/test 只读身份，不读 test 标签。
- 命令：`python scripts/audit_week8_labels.py`、`python scripts/verify_week8_retrieval_routing.py`、
  `python scripts/verify_week8_runtime_repairs.py`，均使用上述配置与同一 Spartan 项目目录。
- 标签审计：旧 parking 58→caption 支持 0，业态/风格/设施/价位正支持 60/53/60/0→3/0/3/0。
  全样本保留。原三 Prompt raw 的重计分只是引用敏感性诊断，不是新模型改善；三者均返回
  `DIAGNOSTIC_ONLY_INVALID_REFERENCES`，无新 Prompt/adapter 选择，无新 final。
- 检索：真实隔离 Milvus Lite FLAT + 生产路由，复用发布 1,000 CLIP 向量，固定 5 查询
  返回数 5/5/0/5/5、过滤正确、查询变化改变结果；没有读取 query gold 用于排序。
  未运行新 CLIP 编码、未证明图片相关性提升、未重启正式服务。
- GPU：job `29667548`，Qwen3-VL-8B-Instruct NF4 + checkpoint-87，A100 MIG 1g.20gb，
  15 分钟 walltime，实际 2:46，COMPLETED 0:0。技术 smoke PASS、业务 smoke FAIL。
  两日行程纠错后仍只一天；对话行程纠错后四天；均被业务检查拒绝，没有伪报成功。
- 商品：对话已调用真实模型 1 次；5 次固定实图输出一致，仍猜测 parking。冷启动
  36987.748 ms，mean/P50/P95=4686.114/4684.040/4702.216 ms，input/output 总量
  3565/285，商品 Schema=1、失败=0，峰值 allocated=8143745536 B。不与整卡做速度增益比较。
- 标签/检索/运行时 summary SHA 分别为 `960e0d8e...688d3c`、`14ff41ef...cba468`、
  `c81af248...9f2cb1`；完整哈希、字段指标与局限见商品报告第 13 节。
- 处置：保留失败、模型和候选历史身份，交付工程修复但状态维持 PARTIAL。没有扩大训练、
  人工标注或后续周计划。CLI 候选 quality 字段误判在最终本地复验中修正，实际配置 SHA 不变。

### 2026-08-28 Prompt 契约与 adapter 消融（执行中）

- 起点 `ac84265`：发现旧行程 Prompt 强制证据 null、禁止规划地点并提供占位骨架，
  与业务验收冲突。新增 `week8_itinerary_actionable_v1` 和 `week8_product_visual_facts_v3`，
  不覆盖旧 Prompt；前者允许明确标为建议的规划，后者禁止从商家/图片类型猜测设施或价格。
- 配置 `configs/week8/contract_ablation_v1.json`：固定既有 development 的 6 张图片与
  2 个文字请求，比对旧 adapter、修复 Prompt + adapter、修复 Prompt + base。脚本保留
  原始响应、token、延迟、Prompt/脚本/adapter/数据哈希；消融共用模型锁，不改权重。
- 命令：`python scripts/review_week8_contracts.py --config configs/week8/contract_ablation_v1.json`。
  计划使用已验证的 A100 MIG 20GB，15 分钟 walltime；此处未声称模型结果通过。
- 本地定向测试 4/4、完整 unittest 683/683 通过；不读取 final，无人工标签或自动晋级。

- 第一轮 job `29684303`（执行 `0402a93`）已完成：三组商品均 6/6 Schema 通过，旧/新
  Prompt + adapter 仍输出元数据和无依据价位；base 图像事实明显更具体，但仍有业态与
  unknown 声明错误。base 两条行程旧检查为 1/2，复审发现所谓通过项有 18:30 超时，
  按新增活动级检查实为未通过；不能把 1/2 写成业务改善。
- 新增逐标签可见事实协议 `product_visual_observation_v1`，将观察与确定性字段映射
  分开，保留所有原始尝试；价格无比较口径时仅保留 OCR，价位为 unknown。还移除纠错
  的四天天数截断与占位骨架，提供可配置无强制解码纠错（仍执行完整 Schema 校验）。
- 阿里云现有 qwen3.7-plus API 已实测可用，两个独立图像教师 pilot 分别 6/6 与 4/6
  通过，后者两条超过 80 字符；全部是 model_generated_silver，非人工真值，不选优。
  全 60 条 development 教师运行使用一次格式纠错、并发 2，保存失败、不删除样本。
- 配置 `contract_ablation_v2.json` 对商品观察协议及修复后的行程继续实测；结果尚待运行。
- 第二轮 job `29684709` 已完成，6 张商品均首轮通过，mean 约 5.5 秒；食品特写不再
  猜场所设施，仍有工业场所业态错误与风格漏识别。行程 1/2 通过当前检查，但复审又发现
  无输入日期时生成了日期、等义约束被逐字匹配拒绝；仍不能认定完整业务通过。
- 独立 qwen3.7-plus 教师全 60 条 development 已完成（11 条使用一次纠错、最终失败 0），
  style/facility 正支持 34/37（标签 42/78），已知业态支持 39、unknown 业态 21，价位 N/A。
  原始输出 SHA `19a5eeb588158deb991724868b8c14fd2386af7dccae3d895520b8baef9ad194`。
- 完整 development 配置 `contract_ablation_v3.json` 预先绑定教师 raw/manifest 哈希，对比
  正式 Prompt + adapter、观察 v1 + base、观察 v2 + base。评分包含无正标签图片的误报，
  从原始生成重建结果，保留全部样本；任何参考污染、指标非有限数或字段回退均不可选。
  此协议仅产生 development 候选，不宣称人工视觉准确率、统计显著性或已可晋级。
- 本地完整 unittest 704/704 通过（30.709 秒）。
- 全量 job `29684981` 完成：三组 60/60 结构通过，独立视觉 silver 对比在
  `outputs/week8/review/week8_contract_comparison_20260828_v3/comparison.json`。
  `observation_enhanced_base` 已知业态 0.820513（正式同值），风格 P/R/F1
  0.696970/0.547619/0.613333，设施 0.824324/0.782051/0.802632，unknown 0.920833。
  支持 60 样本、业态 39、风格 34 样本/42 标签、设施 37 样本/78 标签、价位 0。
  综合分 0.463794→0.745493；mean/P50/P95 4723.828/4717.584/5015.134→
  6574.683/5714.439/11249.417 ms，input/output 均值 682.8/59.167→923.8/98.717。
  观察 v1 因风格回退不选；v2 仅通过 development 选择，不是已晋级或人工真值提升。
- 生产探针 `candidate_runtime_probe_v1.json` 绑定观察 v2、base 商品/行程/对话及原售后
  adapter，检查两/三/五日行程、商品和图片比较对话、三种缓存各八次。所有产物新建，
  不读取最终 test；探针并非正式 release 切换。
- 探针 v1（job `29685503`，执行 `ee8a5ee`）结束：四场景、2/3/5 日请求及图片比较均
  通过当时检查；复审对话结果仍有“某文化空间”，按新增具体地点规则不接受此 PASS。
  商品三种缓存各 8 次标签完全一致、失败 0；mean 未缓存/CPU 缓存/GPU prepared 缓存
  9112.760/9086.717/9091.707 ms。CPU 缓存仅约 0.29% 改善，不夸大为显著提速。
- 探针 v2 保留同图同请求，改用行程 v3 和具体地点验证；新最终集预先固定 100 个未消费
  身份、按 seed/source_id 排序，不看标签挑样本。数据封存与候选锁定分离；任何最终角色
  启动即写 consumed 标记，评分器拒绝将 final 用于 development 选优。
- 完整 unittest 724/724 通过（24.750 秒），包含五维排除、检索 group_id 规范化、一次性
  消费、源文件篡改和 final/development 阶段隔离测试。
- 最终 v2 的候选锁定执行 `40a0f34`，GPU job `29693616`：正式/候选各 100 次成功；教师
  99/100 有效，1 次在两次生成后仍为食品特写与场所设施矛盾。原评分器拒绝该参考，退出 1，
  未产生语义对比分数。所有已消费身份永久排除，保留 failure 与 raw，不通过补标获得 PASS。
- 执行 `1eddfd2` 登记 development-only `contract_ablation_v4.json` 和教师可靠性 v4：前者
  仍对比原 60 条及固定教师 v3 raw，仅缩短可观察事实；后者只测有界自纠错可靠性，不用于
  候选选择。原观察语义/银标身份不变，新增尝试失败也不能生成有效 target。全量 733 测试通过。
- 教师可靠性 v4 完成：60/60 有效，尝试分布 48/11/1（一次/两次/三次），额外 13 次；
  不是新的选优参考。追加明确否定证据反例修复，并对固定教师 v3 与可靠性 v4 各 60 条
  原始观察重放，target 全部不变。未读取旧最终语义分数，也未修改旧最终参考。
- job `29693799` 的观察 v3 与同组正式比较：composite 0.463794→0.759287，style F1
  0.313725→0.644444、facility F1 0.257143→0.812903；全部 60 样本与支持数保持不变。
  相对观察 v2 综合分 +0.013794，但 style precision 降低且 P50 变慢，不能称为纯性能胜出。
- 执行 `6fb9133`：v9 probe job `29694606` PASS，8:55，24 次标签一致且失败 0；CPU 缓存
  mean/P50/P95 9221.647/9219.868/9233.382 ms，仅约 0.12% mean 差异。数据/检索 job
  `29694607` 完成；新 final 明确排除已消费 v2，数据锁、候选锁和全部命令见商品报告 14.7。
- 锁定 v9 后仅一次启动 final teacher 与 GPU job `29694824`。教师 100 条有效，118 次
  原始尝试全部保留，无人工参与；最终模型指标待配对推理完成后一次评分，不作中途选优。
- v9 final 已完成 PASS：job `29694824` 实用 18:39，正式和候选均 100 次成功，候选仅
  1 条使用预定纠错。正支持业态 62、风格 57 样本/77 标签、设施 51 样本/106 标签、价位 0；
  各有支持字段不回退，composite 0.429365→0.736721。mean/P50/P95
  4925.687/4902.253/5664.163→6016.703/5965.712/10297.217 ms，input/output 均值
  681.73/60.04→1005.38/87.20。单次最终不再用于任何方案选择。
- 从 raw 重放生成 `promotion_acceptance.json`；新四层包验证配置、adapter、全部图像、
  教师/配对 raw、最终质量与隔离 API 导入均 PASS。全量 746 条测试通过（22.729 秒）。
  所有产物、哈希和限制见商品报告 14.8—14.9；无新增训练，不把 silver 当作 human。
- 交接补充：Spartan FastAPI 0.141.1 用不同内部路由表示，route_count=6（本地 14），
  但 OpenAPI 路径均为 10 个。增加七个必需业务/健康端点的明确检查和缺路由反例；本地
  原包复验 PASS，全量 747/747（24.750 秒）。只加强交接检查，没有修改归档、模型、
  Prompt、评分器或 final 结果；新验证记录不覆盖旧记录。

### 2026-08-28：v9 后商品证据精简 development 对照

- 新配置 `configs/week8/contract_ablation_v5.json`，完整原 60 条 development、原教师 v3
  silver 及其哈希不变；同场比较 formal、v9、紧凑 label→fact 对象、紧凑对象加语义边界。
  不读取已消费 final，不新增人工工作，不覆盖 v9 配置/adapter/交接包。
- 新 `product_visual_observation_v4` wire protocol 仅去掉重复的 label/fact 键名，仍保留
  全部正标签、逐标签简短事实、十事实上限和 unknown 价位。新增重复 JSON 键及
  “菜单暗示有座位”等推断证据的拒绝路径，原 v1/v2/v3 协议行为保持不变。
- 新 incumbent 比较器要求所有有支持语义字段均不低于同场 v9；提速还须 mean 至少 5%、
  输出 token 至少 10% 改善且 P50/P95 不回退。只有 development 资格，不自动替换发布。
- 定向 35/35，全量 760/760（25.814 秒）通过；真实模型结果尚未产生，不提前声明提升。
  按上轮 60 条约 6–7 分钟，四组申请 30 分钟 A100 MIG，沿用既有模型缓存与环境。
- 首次 job `29697329` 未取得新配置，在模型加载前失败且无输出目录；按原脚本保留
  180 秒诊断窗口后退出。修正为对 `FETCH_HEAD` 快进并核验实际提交，60 条图片/数据
  预检 PASS，恢复 job `29697351` 与原作业串行，不重复消费任何成功请求。
- 本地二次 review 修复紧凑校验重复调用改变 wire representation，以及检索冲突条件被
  错误删除的问题；全量 767/767（25.855 秒）通过。GPU 仍执行固定 `1764324`，不热改
  其目录；扩展检索使用项目目录内独立校验 worktree，不产生新模型或人工标签。
- CPU probe job `29697455` 在独立 worktree 缺少 `data/yelp` 图片引用时退出，尚未创建
  Milvus 集合或发出查询；保留其日志和空输出目录。补齐既有图片目录的只读用途引用后，
  用独立 `candidate_retrieval_probe_v4.json` 重试，v3 不覆盖。
- development 新紧凑协议在 `0009` 两次都生成 `seating: No visible seating`，被原有
  否定证据校验正确拒绝。追加 `contract_ablation_v6.json`：只在新对象协议里把模型看到的
  Schema 换成等价 `propertyNames` 表达，明确允许 `{}`、禁止填入缺席属性；内部仍用展开
  Schema 完整校验，不降低失败/语义标准。该追加比较仍使用原 60 条和固定教师，不接触 final。
- 扩展检索 v4 job `29697507`（`8afd53c`）通过 8 查询、4 对话状态检查；原索引/CLIP
  向量未变。进一步修复英语复数业态未被解析的问题，`restaurants/hotels/cafes/museums/parks`
  按完整词映射原类别，继续保留否定处理；新 v5 探针增加两条英语查询，不修改旧探针结果。
- 完整 `contract_ablation_v5` job `29697351`（执行 `1764324`）完成，26:10：formal/v9/
  紧凑 v4/语义 v5 各 60 请求，后三者综合分 0.759287/0.743747/0.648588，失败 0/1/2。
  两个修订均回退，不选；v9 与上轮原公开输出 60/60 一致。
- `contract_ablation_v6` job `29697591`（执行 `fb49de6`）完成，18:39，formal/v9/Schema
  v6 各 60 请求；v6 综合分 0.666554、失败 2/60，同场 v9 0.759287、0/60。
  v6 mean 6794.864 ms 高于 v9 6424.501 ms，input/output 1176.65/98.78 高于
  995.80/94.90，仍不选。短 Schema 不等于端到端提速，所有数据支持不变、price N/A。
- 两组原始输出经 `score_week8_visual_silver.py` 重放，并用 `compare_week8_incumbent.py`
  比较同场 v9，均为 KEEP_V9_CANDIDATE。完整分字段、切片、身份 SHA 和复核命令见商品
  报告第 15 节。420 请求是同一 60 张图上的多组对照，不作为 420 个独立样本。
- 英语查询扩展 `candidate_retrieval_probe_v5` job `29698776`（执行 `06f1b48`）完成，
  9 秒；原 Milvus Lite/1,000 向量上 10 查询、4 对话状态检查 PASS。无法支持的安静/
  多价位/多业态对话明确未完成，不将该通过解释为这些条件已实现或图片相关性提高。
- 最终全量 769/769（23.892 秒）、定向 44/44 通过；保存原始失败与独立运行身份。
  本轮没有新 teacher、final、SFT、候选打包或部署，继续保留 v9。

### 2026-08-28：自主持续优化，风格专项二阶段复查

- `contract_ablation_v7.json` 仍绑定原 60 条 development、教师 v3 raw 与原评测口径。
  formal/v9/独立风格替换/仅补充风格四组，不读取已消费 final，不新增人工或 teacher。
- `product_observation_v7.json` 在非食品主体上独立重看图像、仅替换 style_evidence；
  `product_observation_v8.json` 只在已有风格的场景追加遗漏标签。两个配置的首阶段与 v9
  Prompt 完全一致；主阶段的业态、设施、价位不可被二阶段覆盖，证据与 unknown 派生字段
  由完整 mapper 重新计算。复查失败不使用首阶段结果掩盖失败。
- 新协议保存全部阶段、纠错、token 和耗时；评分从完整 raw 顺序重放，不能只信末次
  style JSON 或已汇总结果。重复键/标签、推断事实和十个独立事实的上限仍严格检查。
- 定向 54/54 通过（0.045 秒）。四组约 240 主请求加 70–90 次风格复查；依据前轮
  26:10 和新增阶段预算申请 38 分钟 A100 MIG。该时长包含冷加载与失败诊断余量，
  不并发其他 GPU 作业。质量结果尚未产生，不提前宣称达成优化目标。
- 完整 unittest 786/786（49.368 秒）通过，日志独立保存为
  `outputs/week8/review/week8_full_unittest_20260828_v29.log`；旧协议及本轮各失败路径均覆盖。
- 执行提交 `151a8f2`，GPU job `29704676`，38 分钟 A100 MIG。提交前预检发现 Spartan
  不存在原教师副本，未启动 GPU；从本地原样补传 raw/identity 后 SHA、60 条与五维隔离
  检查通过再提交。不是新 teacher，不改标签或消费 final。
- 检查固定 Transformers 4.57.1 源码发现 FastImageProcessor 的像素边界由 `size` 或
  成对 min/max 生成，原 runtime 单独设置 `max_pixels` 可能未生效。新增兼容 fast/legacy
  的边界设置器和实际 processor CPU 探针，None 对 v9 保持 no-op；31 条定向回归通过。
  该修复尚未改变任何已锁定 release 的像素参数，也不据此宣称商品提速或质量提升。
- 像素边界修复后的完整 unittest 792/792（40.222 秒）通过，日志
  `outputs/week8/review/week8_full_unittest_20260828_v30.log`，SHA
  `4f98caa3781ff59db5ab1e5985deba6e3183cac855cbf0e5c0d8a81407c149c8`。
- CPU job `29704717`（独立校验 worktree，`6f3e31f`）22 秒完成：真实处理器确认旧
  max_pixels=131072 单属性设置不生效，仍为 208896 像素/204 visual tokens/1030 input
  tokens；修复后分别为 119808/117/943，65536 上限为 55296/54/880。未加载模型权重、
  未使用 GPU、未评估质量，不把单次预处理时间写成 VLM 加速结果。
  summary SHA `ba73c6effe87e7368e19d909855b2e75a32d6680658dfcd9048c68616dd812fb`。
- GPU `29704676` 当前等待资源，保持唯一提交；调度器 dry-run 显示同长度 L40S 作业
  预计更晚，未创建替代作业。Spartan 主代码仍固定 `151a8f2`，不把本地像素修复热改进
  已登记的风格对照目录；其原像素参数不变。
- 追加范围校验 replay `week8_style_scope_development_20260828_v1`：从冻结 v9 development
  raw 检出 3 条非场所风格依据（饮品、衣服），保留所有 raw 与样本。其中 1 个 casual
  标签虽匹配 silver，但其依据是外套；直接过滤使 style precision 0.604167→0.622222、
  recall 0.690476→0.666667、F1 0.644444→0.643678，composite
  0.759287→0.759031。按原规则拒绝，不当作优化完成。
- 该 replay 是未提交工作树上的确定性诊断（基于 `6f3e31f`），不是新 GPU 性能或最终
  结果；其脚本、规则配置和实现随本次提交留存，原始失败产物不覆盖。后续 replay 会
  同时记录 dirty 状态与源码 SHA。不允许以此脚本模拟需要新模型调用的候选。
- 因此新增 `product_observation_scope_review_v1.json`：只有既有事实明确越出场所范围
  才触发真实图像风格复查，保留其他场景的 v9 输出；复查仍只替换风格，不能靠继续删除
  不合范围的新事实伪造成功。69 条定向通过；候选尚待真实模型验证。
- 完整 unittest 807/807（23.518 秒）通过。继续审查发现原教师也存在把衣物/食品作为
  场所风格依据的记录；范围词表还需区分 seating 等真实场所上下文，不能把仅有词命中
  当成最终标签判断。保留原参考，先做版本化自动范围审计/修订，修订前旧匹配分只作诊断。
  GPU `29704676` 已进入 RUNNING，推理不接收参考标签，因此继续保存完整结果；后续
  必须将 v9 与候选在同一可靠参考下比较，不能按已知有问题的旧分数锁定。
- `product_observation_scope_v2` 补充 seating/sofa 等一般场所上下文词，保留已运行 v1。
  完整 60 条只读审计定位 4 条明确范围错误；使用独立 qwen3.7-plus 重看这些图片的风格，
  不发送旧参考、候选、样本 ID 或商家 metadata。其余 56 条逐字段继承，失败不得删除样本；
  四个有值身份和不适用的模板身份原样核对。新参考保持 silver/0.5/human=0，支持变化单列。
  配置 `visual_teacher_style_revision_v1.json`、修订工具与 11 个反例已实现；定向 43/43、
  完整 818/818（35.422 秒）通过。此处仅记录修订工具验证，尚未记录实际教师结果。
- GPU `29704676` 完成全部四组 60 图；formal/v9/add-only 各 60 个请求通过，独立替换
  59/60。后者存在真实失败，不具备晋级条件。四份 raw 已下载且 SHA 与远端 summary 一致；
  没有消费新 final、训练或改写 v9 发布包，等待统一修订参考后再比较语义结果。
- 独立教师修订实际完成：4 次请求均成功，4 条风格改用场所可见证据，保留其余 56 条。
  style 支持为 34 图/44 标签（原 34/42），其他支持不变；无人工、无候选输入。
  新 raw SHA `29a34f8aff360286c1e4053c0e53e24fe143ab877538d7bf848314cf1f9a51aa`。
- 使用 `score_week8_reference_revision.py` 对所有原始输出重放，核验五维身份、旧/新标签
  血缘、非风格字段继承及 60 条范围审计。新参考下 v9 composite 0.754617、style F1
  0.630435；独立替换 0.707098/0.490323、add-only 0.717134/0.517986，均不晋级。
  GPU `29704676` 实际 30:19、退出 0；失败请求及误报完整保留，不按新旧参考分差宣传收益。
- 下一组 `contract_ablation_v8` 仅比较同场 v9、越界风格假设定点复查、实际 131072 像素
  上限。定点复查只修复或否定被标记的原假设，不扩展其他风格；有效的原标签和其他字段
  原样保留。新对照使用完整 60 图，正式 adapter 的历史 raw 只作同参考质量基线，不作
  同场速度对照。像素实验用执行锁内的临时参数，异常也恢复，缓存键包含有效尺寸。
- 此实现定向 83/83、完整 824/824（30.278 秒）通过；v8 尚未运行，不声称新候选成立。
- v8 GPU `29705244` 在 `fffb6b1` 上运行，申请 26 分钟（前一组实测吞吐估算三组约
  20 分钟加启动/纠错余量）。Spartan 主目录不热更新；所有新验证代码先在本地测试。
- 候选锁新增修订参考 raw 重算与 incumbent 改善检查；实际像素配置和复查/范围实现也
  纳入哈希绑定。最终配对推理临时应用各组像素参数，避免把候选参数泄漏到基线。
  教师路径发现越界证据时必须重新观察或失败，不能用推理过滤器自动删除参考标签。
- 修复发布打包只包含 observation v1–v3 的限制：自动加入选中配置，且拒绝越界路径、
  密钥和非配置目录。新范围复查配置在隔离归档中加载通过；打包定向 17/17、最新全量
  827/827（27.813 秒），日志 `week8_full_unittest_20260828_v35.log`，SHA
  `ec52e19f259c108c249b14ecaedac196727290c521bd06e5cc532c15cd7d87ac`。
  跟踪文件扫描未发现密钥模式或大于 5 MiB 文件；主工作树仍为 dev/34 项既有改动。
- 独立教师范围协议在完整 development 完成 60/60，有 5 次纠错，raw SHA
  `c92289f1b34d74d13c782b1eecb385bdd8ca8bcdb9b62347fee4f8f4ccbaab67`；这是协议可靠性
  验证，不作为候选选择参考。随后增加餐具审计发现其中 1 条仍用塑料盘/调料瓶支持 casual；
  更严格重放失败，保留 `week8_teacher_scope_v3_replay_20260828_v1.log`，不虚报通过。
  正在使用的四条修订参考按同一餐具范围复查无新增错误，60 条评分参考不再改变。
- v8 的定点模型复查把鸡尾酒事实改写成扭纹玻璃杯，仍非场所装修；因此新增独立 v2
  推理配置：餐具/衣物仅支持物件事实，不能支持场所风格，结构有效但证据仍越界时按明确
  unknown 策略弃权并保留 `style_evidence_abstentions` 和全部 raw。重复键/标签、推断句、
  长度及未请求标签仍失败，旧 reject 配置行为不改；教师从不采用推理弃权过滤参考标签。
  玻璃幕墙、吊灯、门窗等场所上下文保留，未删样本或减少参考支持。
- `contract_ablation_v9` 准备按同一 60 图/44 style 标签比较 v9 与该弃权修复；尚未运行。
  教师另用 `visual_teacher_scope_reliability_v2` 验证显式餐具边界，不新增人工。
  定向 52/52、完整 831/831（27.820 秒）通过；商品质量仍以实际完整对照为准。
- v8 实际 job `29705244` 19:58、退出 0，三组均 60/60。定点事实重写标签分完全等于
  v9（composite 0.754617），平均延迟比 1.013055，无改进；有效 131072 上限 composite
  0.715351、style F1 0.533333、facility F1 0.792208，虽平均快约 2.13% 仍因质量回退拒绝。
- 弃权修复完整对照 `29705434` 已在 `a878a0c` 上运行，按前组双组约 14 分钟申请
  17 分钟；不采用低像素，不改参考。修订参考远端完整 raw/五维重放通过。身份只读检查
  显示排除历史训练/dev/已消费 final 后仍有 228 张可用原图，未建立或运行新 final。
- 明确餐具边界的独立教师协议再测 60/60、64 次请求，raw SHA
  `6a62f136ec921464008cae3c5ca00e9d85ce080913790624ee1eeed6763e198b`。此为协议可靠性
  证据，不作为新的选择参考；正式选优仍使用四条修订后的 60 图/44 style 标签。
- 新 final 执行/复核/交接能够保留第三组 incumbent：原正式模型严格提升规则不变，另需
  不低于当前 v9。综合状态不会在 incumbent 失败时只显示正式基线 PASS。定向 18 条、
  全量 833/833（27.936 秒）通过，日志 v37 SHA
  `6635fecf86071f5d6228c5b3a7ff354ad06ffa4cdf3cc11fdff6c4ecbe466e11`。
  原 v9 四层 SHA、显式 release 校验及隔离运行层导入再次通过；没有读取旧 test 输出调参。
- `29705434` 正常完成，实际 13:35、退出 0，固定 60 图两组各 60/60。修订参考下
  v9→scope_repair_v2 的 style TP/FP/FN 为 29/19/15→29/18/15，style F1
  0.630435→0.637363、composite 0.754617→0.756926、unknown 0.920833→0.925。
  其余有支持字段不退，JSON/Schema 100%，失败 0；平均延迟 6453.674→6544.186 ms，
  输入 token 995.8→1020.1、输出 94.9→96.25。只消除一条非场所 classy 误报，收益有限。
  raw SHA（v9/候选）为 `d04f6166e19d7c65083fe6c2324cda205fa6587280f87505f052d4f3ac3247cb` /
  `0f1253d96d4d592bd5a88c170343e4ed6f06e3ace3d2297ca90573674e00ffef`；同参考重放
  `week8_contract_comparison_20260828_v9_style_revision_v1/comparison.json` SHA
  `cd71580aa8db3b5f39b822b634850d398d5e08bb7325a2981c4db8d186c65d0e`。
- 固定 `qwen3_vl_system_week8_v10.json`，不改变首阶段 Prompt、模型、像素及 adapter 路由；
  新增 runtime_probe_v5（含真实弃权分支）、retrieval_probe_v6、visual_final_v4 和独立
  teacher_final_v2 配置。新 final 保留 v9 第三组和原正式模型严格提升规则。尚未创建或
  消费新 holdout，所有新验收均待执行；全量 834/834（27.759 秒），日志 v39。
- v10 runtime 作业 `29705563` 采用 `3072446`，按 v9 8:55 实测加一条范围探针申请
  12 分钟。当前商品/售后/行程 smoke、三条多日约束行程、商品对话、两图比较及真实
  非场所弃权分支通过，重复缓存延迟仍执行中。原 GPU venv 不修改。
- 检索 `29705565` 因误选 GPU venv 缺 FastAPI 在导入时失败（没有创建输出）；保留日志，
  改用已有 data/API venv 的 `29705571` 8 秒完成，10 查询/4 对话状态全部通过，不宣称
  新图像相关性或所有文字约束都支持。Compose 缺运行环境变量时明确失败；提供仅供静态
  验证的值后 config --quiet 通过，没有启动服务。v10 隔离归档导入及七必需端点通过。
- CPU `29705564` 5 秒完成新 100 图无标签 holdout，候选从 228 未用身份固定抽取；
  本地全图哈希及远端从历史身份重新推导选择均通过，五维重叠均为 0，模板为 N/A。
  数据锁 `8f3044e1362d90232d0631c7795bde62f3bc76aa4c16a778bc9c4d7fe9dfeb10`。
  尚未建立 candidate_lock、生成新 final 标签或运行最终候选。只读重放首次临时命令导入
  模块位置写错，失败日志 v1 保留；更正导入后 v2 通过，数据无更改。
- 错误切片工具支持显式 baseline observation，v9 与新候选均按各自 raw 重放；完整
  60 图风格错误样本 25→24，总语义错误 36→36（有重叠字段），不夸大修复范围。
  定向 65/65 和新增切片 2/2；全量 835/835（28.904 秒），日志 v40 SHA
  `6ffab2c97532ad725e25c49064e90c27a1a5cf6b7c2c911620f24dee9e62d478`。
  完整分支 diff --check、密钥模式和大文件扫描通过，主 dev 仍保留 34 项既有改动。
