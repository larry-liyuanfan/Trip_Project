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

## 2026-07-22: Mentor-authorized v2 recuration in progress

- The mentor's low-quality-image clarification superseded the frozen-v1-only route. All v1 manifests, runs, Prompt assets, and Schemas remain immutable historical evidence.
- Prepared isolated `week3_evaluation_v2` manifests and a 450-row v2 exclusion registry. Product reuses 200 completed labels. After-sales retains 80 supported labels and exposes 70 replacement rows for annotation with candidate strata `38/38/37/37`. Itinerary exposes 100 rows for style-preference capture.
- Current v2 counts are product `200/200/200/200/0`, after-sales `150/150/80/80/0`, itinerary `100/100/100/100/0` for target/candidate/annotated/validated/tested.
- The local single-annotator station is restricted to the approved v2 supplements. Suggestions are deterministic hints and are never submitted automatically as gold.
- Standardized v2 preserves the three baseline sentences, uses explicit multimodal parts, full Schema exposure, JSON-object response mode, and scenario type skeletons. A bounded `itinerary_planning_v2` Schema was added without changing v1.
- Live format probes on the local Qwen2-VL-2B endpoint produced JSON- and Schema-valid product, after-sales, two-day itinerary, and four-day itinerary responses after the type-skeleton change. The four-day response remained semantically incomplete (`itinerary` contained one day), confirming that Schema pass and task quality must be reported separately.
- Strict guided JSON was rejected as an operational strategy: the itinerary request exceeded a 180-second timeout. Raw-output postprocessing was not used to inflate Schema pass.
- A workflow correction confirmed that all 100 v1 itinerary constraint annotations remain intact. The v2 station now exposes only style choices and merges the untouched v1 fields server-side. Four early itinerary submissions were reconciled to exact v1 non-style fields while preserving their human style choices.
- Three early after-sales submissions on abstract synthetic diagrams were invalidated with an audit log. The after-sales tab is paused; its 70 replacements require more representative evidence before annotation resumes.
- All 100 itinerary style supplements are complete: 96 payload hashes match the current annotations directly and 4 match the recorded v1-field reconciliation hashes; one evidence-supported empty style array is valid.
- Full v2 baseline, standardized run, scoring, and comparison remain blocked only by the 70 after-sales replacements. Week 3 remains `PARTIAL`.
- Generated an eight-image photorealistic after-sales pilot (four hygiene, four facility-damage) for visual quality approval. The pilot is ignored local data and has not entered the manifest or gold labels.

## 2026-07-24: Week 3 v2 full baseline and standardized comparison

- Replaced the 70 low-evidence after-sales candidates with visually reviewed photorealistic evidence and completed their human annotations. Final v2 counts are product `200/200/200/200/200`, after-sales `150/150/150/150/150`, and itinerary `100/100/100/100/100`.
- Completed live full baseline `week3_v2_baseline_full_20260724_001` and standardized run `week3_v2_standardized_full_20260724_001`; each persisted 450/450 records with no model request error and the same sample-set SHA-256 `3e900e64bb345df35343c8f14bfb1f8310ae597a57e4a4d9585bc01173ad648c`.
- The baseline kept all three `baseline_minimal_v1` sentences unchanged. All 450 responses were natural language: JSON and Schema compliance are 0%, while unparsed classification, OCR, and constraint metrics remain `PENDING` rather than numeric zero.
- Standardized v2 JSON/Schema rates are product 79%/75%, after-sales 96.67%/96%, and itinerary 90%/88%. Strict semantic metrics and their support counts are recorded in the baseline report.
- Generated paired comparison `week3_v2_prompt_pair_20260724_001` over 450 identical samples with 2,000 bootstrap iterations. Comparison is limited to format, Schema, and latency because baseline semantic output has not been deterministically parsed.
- Week 3 remains `PARTIAL` solely because the mentor-required minimal baseline's natural-language semantic task metrics are unsupported by the current reproducible scoring track.

## 2026-07-25: Week 3 deterministic baseline semantic scoring

- Added `baseline_semantic_coding_v1`, a fixed lexical codebook and encoder whose prediction interface accepts only `scenario` and `raw_output`; the codebook is loaded independently and human gold is joined only after all predictions are complete.
- Generated immutable score `week3_v2_baseline_full_20260724_001__baseline_semantic_coding_v1` for the existing 450-record baseline without any model request, Prompt change, Schema change, manifest change, or relabeling.
- Preserved baseline JSON and Schema compliance at 0% for all scenarios. Semantic metrics are stored on a separate track with per-metric support counts and codebook SHA-256 `563dc0747f92b6ccaa37466045cb0e74229787824013d59a5f6f26261bb033a6`.
- Moved the earlier gold-leaking semantic score and incompatible cross-track comparison unchanged into ignored quarantine with a hash disposition record.
- Final verification passed 226 Python tests, standalone v2 validation, both run-bound validators, semantic score integrity checks, and Git diff/boundary checks.
- Week 3 status is `READY / COMPLETED`; the deterministic lexical method and its limitations are documented without attributing cross-track differences solely to Prompt behavior.

## 2026-07-25：Week 4 Prompt 优化与 Milvus

- 保持全部 Week 3 产物不变；每场景从 v2 金标固定选择 5 个正例和
  2 个边界例，构建 4-shot（3+1）和 7-shot（5+2）。
- 旧 Few-Shot v1 行程请求因上下文超限全部返回 HTTP 400，不能作为有效
  候选。新增版本化 v2 删除重复长上下文后，4-shot 和 7-shot 均完成
  15/15，模型请求错误为 0。
- 有效 pilot 中 `standardized_v2` 在三个场景均胜出；选择分数为商品
  0.3280、售后 0.5967、行程 0.4775。新增 Few-Shot 候选未超过控制组，
  没有产生新的胜出 Prompt。
- 全量胜出运行 `week4_winners_full_20260725_001` 保持 450/450。
  baseline 词法业务轨道与 Week 4 结构化业务轨道不再计算差值；同口径
  JSON/Schema 和延迟单独比较，baseline token 明确为 `PENDING`。
- 导出真实 bad case：分类错误 86、字段/Schema 错误 7、格式错误 67、
  严重等级错误 105、约束遗漏 100；类别允许重叠。
- 格式兜底只去除可选围栏、解析 JSON、执行现有 Schema 校验，不补字段、
  不修改枚举、不猜标签、不重试模型。
- Milvus 2.6.20、etcd 和 MinIO 均 healthy；固定十字段集合、
  HNSW/COSINE 和 8 个标量索引均已验证。
- 停止 vLLM 后，用 CUDA CLIP 编码 20 张真实 Yelp 图片。HNSW 构建
  5.6621 s，10 次查询平均/P95 为 7.7982/10.7236 ms，
  Recall@5 为 1.0000。
- 中央审查发现的两个阻断项已修复：Milvus/MinIO 凭据改为本机环境变量并
  完成实际轮换；评估文本哈希统一换行且兼容既有 LF/CRLF 运行记录。
- 修复后 244 个单元测试通过；Week 3 v2 数据验证、两个 run-bound
  验证、Week 4 统一只读验证、Compose 展开和容器健康检查均通过。
- Week 4 状态恢复为 `READY / COMPLETED`；不进入 `stg`，不打标签。

## 2026-07-26：Week 4 共同语义评分与证据边界修正

- 保持 Week 3 baseline、原始输出和 `baseline_semantic_coding_v1` 原评分
  不变，新增 `week4_common_semantic_coding_v1_20260726_001`。
- baseline 与 Week 4 winner 的 450 对原始输出均使用同一个
  `BaselineSemanticCoder.encode`、同一 codebook 和同一指标函数；预测
  完成后才连接金标，并执行 2,000 次 paired bootstrap。
- 共同轨道实测：商品业态 +32.73 pp、价位 +13.00 pp、设施 macro F1
  -19.88 pp；售后分类 +18.00 pp、OCR recall -9.78 pp；行程要素完整度
  -55.40 pp。该结果只解释固定词法编码轨道，不替代人工语义编码。
- 明确现有 Few-Shot 示例取自最终 test gold，pilot 降级为描述性证据；
  `standardized_v2` 仍是三场景描述性最高分且不含示例，全量运行不受该
  污染直接影响。
- Week 3 过程状态文件已标为历史快照并链接后续 `READY / COMPLETED`
  正式报告；Week 4 bad case 增补真实金标、预测、错误原因和字段检查方向。
- Prompt 部分因缺少独立 demo/dev pool 保持 `PARTIAL`；Milvus 保持
  `READY`。不新增标注或数据集版本，不重跑模型。

## 2026-07-26：独立 demo/dev Few-Shot 完成

- 用户后续直接授权完成独立 demo/dev pool，取代此前“禁止新增”的临时
  限制；Week 3 v1/v2 数据、Prompt、Schema、运行和评分保持不变。
- 建立 `week4_demo_dev_v1`：三场景各 12 条、共 36 条人工金标，
  split=`development`；与 450 条最终 evaluation 在 sample/source/image/
  group 四层无重叠。
- selection v2 从每场景 development 金标固定选择 5 正例 + 2 边界例；
  三组新 pilot 共 45/45 请求完成，模型请求错误 0。
- 固定综合分胜出：商品 `fewshot_4_v2`，售后与行程
  `standardized_v2`。新的混合 winner 全量
  `week4_winners_full_20260726_002` 完成 450/450。
- 全量 JSON/Schema：商品 82.0%/20.5%，售后 96.67%/96.67%，行程
  91.0%/88.0%。商品 4-shot 的低 Schema 率作为 pilot 方差与负结果保留，
  不用全量结果反向改选。
- common comparison `_003` 完成 450 对同编码器评分、38 个聚合指标和
  2,000 次 bootstrap；v5 bad case 共 376 条。
- Week 4 状态更新为 `READY FOR MENTOR REVIEW`。

## 2026-08-02：Week 5 数据集标注与质检工程交付（人工阶段未完成）

- 新增三场景标注规范、accepted Schema 映射、多模态 JSONL 标注配置、人工 revision
  和三级质检规则；售后/行程确定性抽检 10%，商品抽检 5%。
- 从本地 Yelp 数据实际构建 80,000 条互不重复候选：商品 50,000、售后 20,000、
  行程 10,000；调用 Week 3 v1/v2 exclusion manifest 后，source、图片 SHA-256、
  来源组和约束模板冲突均为 0。
- 商品层为酒店 200、景点 800、餐饮 49,000；评估来源组排除后仅剩 258 张可用
  酒店图片，未伪造均衡分布。
- 售后为公开 Yelp 5,552、项目自有业务合成 14,448，四个问题路由各 5,000；
  路由和严重度只是抽样提示，不是人工金标。
- 行程四类人群各 2,500，预算和 2/3/4 天近似均衡。
- 新增 Qwen3.7 批量预标注、断点续跑、失败记录和重复执行保护；商品/售后复用
  `fewshot_4_v2`，行程复用 `standardized_v4`。
- 通过全局 SSH 别名从 ECS 进程内临时读取百炼密钥，商品真实 smoke 预标注 3/3
  Schema 合规、失败 0；密钥未落盘。没有 Week 5 人工输入，人工修正、三级质检、
  最终合格和对话候选/合格数量均为 0。

## 2026-08-03：Week 1-4 Qwen3.7 整体报告与 stg 整理

- 以 Week 1-4 最后验证提交 `83c67f0` 为边界，排除后续 Week 5 实现。
- 新增 `reports/week1_to_week4_qwen37_overall_report.md`，统一整理云端模型
  迁移、Week 2 不变项、Week 3 重跑、Week 4 Prompt 重选、行程 v4 修复和
  Milvus 原型结果。
- 新增 `reports/README.md` 作为报告索引；历史报告和过程状态保持原路径，
  不移动、不覆盖冻结运行产物。
- 总报告明确区分同 Prompt 模型对比与 Prompt/预算联合修复，避免错误归因。

## 2026-08-09：Week 5 冲突裁决落地与受限 pilot

- 记录 Project Control 六项裁决，新增 workflow v2 sidecar 和
  `multimodal_dialogue_v2`，保留候选池、历史 pilot 与 dialogue v1 不变。
- workflow v2 sidecar 实际绑定商品/售后/行程 50,000/20,000/10,000 条候选，
  初始人工状态全部为 `awaiting_human_annotation`；候选池全量校验返回
  `status=ok`，80,000 个唯一 sample 和图片 SHA-256。
- 新增不可覆盖 run 目录、配置/候选/输入/请求哈希、原始输出、attempt、checkpoint、
  failures 和严格 resume 元数据校验。
- 完成 30 条行程样本的双 Prompt pilot，共 60 请求；`fewshot_4_v2` Schema 28/30，
  `standardized_v4` 30/30，请求失败均为 0，最终按既定规则选择
  `standardized_v4`。实测推理 1,197.05 秒，估算 CNY 6.09。
- 未执行全量预标注、全量对话生成或训练；真实人工修正、三级质检、最终合格和
  人工合格对话数量仍均为 0。

## 2026-08-10：Week 5 单人最小人工质检

- 用户确认仅由一名真实操作者完成 Week 5 人工环节；不得虚构第二位标注人或独立
  审核人。人工修正保存时同步显式确认自审，不再安排一次重复的全量自审遍历。
- 交叉互审改为同一操作者在不同 `review_session_id` 中进行的盲二次复核；商品按
  1% 抽取，售后和行程按 2% 抽取。核心抽检分别按 0.5% 和 1% 抽取，且使用同一
  SHA-256 选择值，保证核心抽检是交叉复核的固定子集。
- 未抽中二次复核的样本只有在真实人工修正和内联自审均完成后才可 accepted；模型、
  校验器和 Agent 均不能填写人工身份、确认、审核意见或通过状态。
- 新增 `configs/week5_dataset_qwen3_vl_4b_single_operator.json` 供预标注后的人工处理、
  质检和报告使用；正在运行的全量预标注仍绑定原配置和 manifest 哈希，不作修改。
- 新增 `export-quality`，仅导出确定性选中、前序已通过且当前 revision 尚未处理的
  二次复核或核心抽检任务；同阶段重复导入和审核会话复用会被拒绝。
- 按现有 80,000 个真实 `sample_id` 计算，额外盲复核/核心抽检固定集合为商品
  516/259、售后 399/219、行程 190/94，共 1,677 次额外阶段操作。它们是待处理
  集合，不是已完成人工数量；当前真实人工修正和质检数量仍为 0。
- Week 5 定向测试 19/19、完整 `unittest` 289/289、Week 3 v1/v2 隔离验证、三个
  单人配置 JSON 解析和 `git diff --check` 均通过；活动 run 配置哈希核对一致。

## 2026-08-10：额外质检压缩与全量预标注进度复核

- 用户要求把额外人工质检压到 500 次以下。当前比例更新为商品 0.2%/0.05%、售后
  和行程 0.5%/0.1%；对不可变 80,000 个候选实算为商品 112/26、售后 102/21、
  行程 53/7，合计 321 次盲复核/核心抽检操作。上一版 1,677 次方案已被替代。
- 全量 run `week5_full_preannotation_qwen3_vl_4b_20260809_b` 实际保留商品成功
  8,140、未解决失败 8（Schema 3、请求 5），已覆盖池索引 0–8,147；商品进度
  16.28%，全体 80,000 候选进度 10.175%。售后和行程全量成功数仍为 0。
- checkpoint 最后更新时间为 2026-08-10 02:10:24（悉尼时间）；复核时本机没有
  Python/SSH 运行进程且回环隧道端口关闭，因此 run 当前为 `partial / not running`，
  不能报告为仍在持续执行。人工修正、三级质检和 accepted 仍均为 0。

## 2026-08-12：Week 5 全量预标注自动恢复与持续监控

- 确认第二次中断由本机 SSH 隧道退出引起；远端 Qwen3-VL-4B vLLM 端点保持健康，
  checkpoint 在连续 21 次连接拒绝后按既有保护阈值停止，候选、manifest 和成功结果
  未损坏。
- 新增 Windows 守护脚本，使用 SSH keepalive、全局互斥锁、端点健康检查和隐藏后台
  runner；连接中断后自动重建隧道并以同一 run ID `--resume`，不重复成功样本。
- 主流程跳过已知失败继续全池；全池结束后只执行一次 `--retry-failures` 清理，避免
  永久 Schema bad case 在每次重连时反复消耗请求。
- 2026-08-12 00:13（悉尼时间）恢复成功；首次复核 checkpoint 从池索引 13,498
  推进至 13,546，新增 48/48 成功、连续请求失败为 0。另建立每 30 分钟 heartbeat
  监控，用于检查进度、发现 supervisor 消失并安全恢复，不停止或释放 ECS。
- PowerShell 语法检查、守护流程定向测试 2/2、Week 5 联合定向测试 21/21、完整
  `unittest` 291/291 和 `git diff --check` 均通过。

## 2026-08-12：Week 5 预标注迁移到 ECS 常驻执行

- 在不中断本地 runner 的阶段预同步 80,000 条候选、80,000 张唯一引用图片和冻结评测
  依赖；迁移归档 81,109 个条目、2,975,730,176 字节，远端 SHA-256 一致。
- 远端复核三份候选 manifest 与 run manifest 一致，图片缺失 0；Week 5 定向测试
  22/22、`validate-pools` 返回 `status=ok`，运行端点仅为 `127.0.0.1:8001`。
- 在池索引 15,190 停止本地 supervisor、runner 和 SSH 隧道；三份 JSONL 逐行合法，
  随后以原 run ID、原 canonical config SHA-256 和 `--resume --retry-failures` 恢复。
- systemd 服务启用并实际将成功结果从 15,166 推进到 15,197、checkpoint 推进到
  15,209，连续请求失败归零；历史 raw 记录随后以不覆盖模式补传。
- 当前预标注仍未完成，人工修正、质检和 accepted 数量没有因此增加；本地电脑不再是
  运行依赖，且禁止本地与 ECS 同时写入该 run。

## 2026-08-12：Spartan 存储复核、环境与 benchmark 重提

- 旧 L40S benchmark `29109265` 并非仍在排队：`sacct` 实测其于 18:50:23 AEST
  以 `FAILED 1:0` 结束，日志显示 `Apptainer/1.3.3` 缺少
  `GCCcore/11.3.0` 前置模块。修复提交 `1bdb419` 已同步到远端。
- project GPFS 实测总量 467 GiB、已用 375 GiB、可用 93 GiB；Trip 版本目录为
  99 MiB。“500 GB”是共享文件系统总量，不是 Trip 独享空间。home quota 已满，
  后续只使用 `/data/gpfs/projects/punim2936/Trip_Project_yzhang3504/20260812a`。
- 新增项目范围 Python 3.11 venv 作业，环境路径固定为
  `envs/trip-week5-week6-py311`，pip/XDG/HF/tmp 缓存均留在 Trip 根目录。
- 新作业已提交。环境安装 `29114275` 于 20:40:23–20:44:11 AEST 运行并
  `COMPLETED 0:0`；`pip check` 无破损依赖，关键包导入成功。唯一 L40S benchmark
  `29114276` 于 20:40:58 在 `spartan-gpgpu006` 启动，当前为 `RUNNING`；没有提交
  H100/A100 重复竞争作业。两小时 heartbeat 已改为追踪这两个 job。

## 2026-08-12：包月展示部署与 Spartan 身份核验

- 用户明确批准上传 `src/`、轻量样例、CPU Docker 资产、展示状态 JSON 和 Week 5
  质量报告。部署包 SHA-256 为
  `404e7a681bdf35a839de56298568960a950203a21d9f7ae61b7dac4fdbe8a81d`。
- 已在 `trip-api-sg:/opt/trip-display/20260812a` 启动独立容器
  `ota-trip-display-api`，仅绑定 `127.0.0.1:8010`；health、状态 API 和静态报告均返回
  成功。原 `ota-trip-api` 的 `127.0.0.1:8000` health 仍成功。
- Spartan Open OnDemand 只读核验显示当前登录身份为 `yzhang3504`。该身份属于用户声明
  的第三方账户，不满足 ADR-020 的代理提交边界；未读取第三方 quota/scratch，未提交、
  修改或取消任何 Slurm 作业。project、quota、scratch 和可用 GPU partition 仍须由账户
  所有者或用户自己的 Spartan 身份核验。

### 身份授权更正

- 用户随后最新确认 `yzhang3504` 为本人持有并授权本项目使用的账户，允许 Agent 代理
  核验并提交。仍强制使用新建的 Trip_Project 专属目录，只管理本项目 job ID，不触碰
  账户内既有文件、作业或进程；密码不落盘。
- 实测 account/project 为 `punim2936`，QOS 包含 `publicgpu`；home 51.2 GiB quota 已满，
  project GPFS 可写且约余 93 GiB。已创建唯一目录
  `/data/gpfs/projects/punim2936/Trip_Project_yzhang3504/20260812a`。
- GPU 待排观测为 A100 约 280、H100 约 80、L40S 约 8，故只提交 L40S benchmark。
  job `29109265` 已进入 `PD(Resources)`，Slurm 估计启动时间为
  `2026-08-12T20:27:34`；未提交重复竞争作业。

## 2026-08-12：A10 停机与 Spartan 迁移工程

- 用户报告按量 A10 因欠费停机；旧两小时 heartbeat 监控已删除。没有停止、释放、
  格式化或删除实例/云盘。
- 当前本地可独立验证恢复点为商品成功 15,166；远端最后一次监控约 36,615 只能作为
  未复核历史线索。完整候选池仍为 50,000/20,000/10,000。
- 新增 `week5_spartan_migration_v1`：实际生成 migration
  `week5_spartan_migration_20260812_a`，100 条 benchmark 按商品/售后/行程
  49/36/15 分布；其余 64,734 条确定性拆为 4 个互斥分片。基线、benchmark、分片
  覆盖合计 80,000，且不续写 A10 历史 run。
- 新增 H100/A100/L40S 队列检查、benchmark、array shard、状态和不可覆盖合并工具。
  Spartan project、quota、scratch 和提交身份仍未核验，因此本次没有声称作业已排队。
- 新增 Week 6 Qwen3-VL-8B QLoRA 配置、数据锁定契约、环境检查、小样本训练入口和
  Spartan pilot 模板；正式训练尚未运行。
- 新增 `trip-api-sg` CPU 展示 Compose 和 `/v1/project-status`，只消费预计算结果和
  静态报告；未部署 CUDA、vLLM 或模型权重。
- 当前验证：新增定向测试 7/7、Week 5 联合定向测试 27/27、完整 unittest 299/299；
  Week 5 候选/隔离、Week 3 v1/v2、四份 Slurm shell 语法、展示 Compose 展开和
  `git diff --check` 均通过。

## 2026-08-12：Week 5 预标注迁移到 ECS 常驻执行

- 在不中断本地 runner 的阶段预同步 80,000 条候选、80,000 张唯一引用图片和冻结评测
  依赖；迁移归档 81,109 个条目、2,975,730,176 字节，远端 SHA-256 一致。
- 远端复核三份候选 manifest 与 run manifest 一致，图片缺失 0；Week 5 定向测试
  22/22、`validate-pools` 返回 `status=ok`，运行端点仅为 `127.0.0.1:8001`。
- 在池索引 15,190 停止本地 supervisor、runner 和 SSH 隧道；三份 JSONL 逐行合法，
  随后以原 run ID、原 canonical config SHA-256 和 `--resume --retry-failures` 恢复。
- systemd 服务启用并实际将成功结果从 15,166 推进到 15,197、checkpoint 推进到
  15,209，连续请求失败归零；历史 raw 记录随后以不覆盖模式补传。
- 当前预标注仍未完成，人工修正、质检和 accepted 数量没有因此增加；本地电脑不再是
  运行依赖，且禁止本地与 ECS 同时写入该 run。
