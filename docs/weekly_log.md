# Weekly Log

## 2026-08-24：Week 7 fix2 门禁和分支治理修复

- 审计确认 fix1 selector 本身正确；失败同时来自模型错误和评分/训练目标错位。嵌套 JSON
  顶层全等会放大局部错误，主观 caption 逐字匹配不是可靠视觉金标；early stopping 的
  weighted composite 又与最终 13 项硬门禁不一致。
- 新增 `gate_aligned_v2`：稳定结构按叶子值计分，自由文本 evidence 不进入 hard-gate
  逐字比较，sequential protocol 与 semantic 分开；训练使用 hard-gate-first objective，
  阈值未按 fix1 结果下调，旧 raw/metrics/selector 未重算或覆盖。
- 新建 fresh fix2 锁 3000/114/114，配比 600/840/840 + 270（9%）+ 450（15%），
  canonical SHA-256 `86a43601...e5b14b`；五维跨分区冲突 0，v3、首版 v4、fix1
  身份均排除；该状态是训练前锁定证据，后续唯一 test 已按同一锁消费。
- Spartan 训练 job `29540085` `COMPLETED 0:0`（03:35:20），step 301 按 patience=2
  早停；selector 在 5 个 development 合格 checkpoint 中锁定 step 226，weighted/core
  composite 为 0.796113/0.746154，selection SHA-256 `cba44b4f...d0c3ac8`。
- 唯一 corrected-dialogue test job `29544969` `COMPLETED 0:0`（00:42:43），24 条/角色、
  自动评估且未执行人工评分。multitask/Week 6 routed/zero-shot 自动综合分为
  0.793399/0.152144/0.174505；最终绝对门禁 `FAIL`，10 项未达阈值，test 已消费且不得重跑。
- ADR-004/ADR-032 明确仅 `dev`、`stg`、`main` 为长期分支；临时分支已清理，未进入
  `stg`、未打标签。DPO 保持既有一次 validation FAIL 后关闭。
- 终态复验：fix2 定向 54/54、完整 unittest 454/454、数据锁与五维隔离、config loader、
  两份 v4 Slurm shell 语法和 `git diff --check` 全部通过。

## 2026-08-19: Week 6 训练后质量审计与项目收束

- 区分“工程闭环完成”与“业务指标优秀”，保留三场冻结终态指标和已知局限不变。
- 基于 Qwen2.5-VL/Qwen3-VL、JSONSchemaBench、mDPO、HDPO 和 JEST 一手资料，形成
  与三个场景弱项对应的下一版本方案，并明确不直接外推论文收益。
- 接受 ADR-030：后续提升必须使用新的非冻结 development/test 身份锁；现有 Week 3 v2
  不再参与调参，未运行新训练时不声明指标上升。
- 新增 `reports/week6_post_training_improvement_review.md`，并精简为报告索引、需求、
  决策、实验记录和 README 各自承担单一职责。

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

## 2026-08-20：Week 7 v3 锁、development 基线与 GPU 执行

- 从 Week 6 终态 `132779b0f6d2929ce1cdbed18e62adf3ef9edd18` 建立
  `codex/week7-multitask-context`；旧 `agent/portfolio-positioning` 工作树保持不变。
- 活动锁 `week7_fresh_multitask_context_20260820_v3` 为 3000/114/114，五维跨分区碰撞
  为 0，锁 SHA-256 为 `8af2e2d13c22fb641fc7344b1e56e5827aa78b1ebde653c6e55c83b36d20504d`。
  train 核心场景各 760、通用正则 270（9%）、对话 450（15%），对话父场景精确
  150/150/150，工具调用 45/450。24 条人工队列仍待真实用户填写。
- v2 对话错误地全部继承商品父任务；作业 `29431992` 在 step 151 development 评估时
  取消，已完成的 38/76/113 checkpoint 全部排除。v3 未修改 v2 锁。
- v3 development 并行作业 `29433880`–`29433884` 全部完成。Week 6 路由 adapters 的
  114 条综合分 0.064270、失败率 0%、平均延迟 5830.99 ms；零样本综合分 0.070147、
  失败率 0%、平均延迟 1961.06 ms；对话均按 8/8/8 覆盖三场景。
- Schema 作业 `29434316` 完成。free JSON 合规率 98.89%；constrained primary 90/90
  被服务端拒绝，单独执行的 free fallback 90/90 成功；延迟比 1.0181。只选择 free，
  不将格式结果解释为语义提升。
- 训练作业 `29434317` 在 step 151 按 patience=2 早停并正常完成；step 76 综合分最高
  为 0.869412，但四个 checkpoint 均超 1.25× 全局延迟门禁且商品支持数不足。selector
  返回 `BLOCKED_NO_ELIGIBLE_CHECKPOINT`，参数锁未创建、test 未读取。DPO 因真实审核
  偏好对为 0 记为 `SKIPPED`。
- 证据链与崩溃恢复修复提交当时，完整 unittest 401/401、compileall、五份 Slurm `bash -n`、
  数据锁验证和 `git diff --check` 通过；提交 `bb6ecfe` 已推送。
- 后续独立审查确认原 selector 的两个口径问题：商品 `label_completeness` 支持数会因
  Schema-invalid 输出被置 0 而虚增，且被列入 `unknown_fields` 的空 style/facility
  仍被误作可评估 gold；训练内候选与 Week 6 基线的 cache/runtime 及 GPU allocation
  也不一致。新增的 `evaluation_protocol_v4` 仅修正评估，不创建 v4 模型或数据锁；它
  继续绑定 v3 config、数据锁、四个既有 checkpoint 和原始 development evidence，按
  gold evaluability 在聚合前固定支持集合，并在同一 allocation 中统一 cache、同步和
  计时范围。
- protocol-v4 首次作业 `29449140` 为 `CANCELLED`，仅写出部分 Week 6 baseline 角色
  文件且没有 protocol summary，因此全部排除。attempt2 作业 `29449999` 在单张 L40S
  上以 `COMPLETED 0:0` 结束，用时 `01:19:44`；protocol summary SHA-256 为
  `6990bda69463d9d9df65082c39d9d53733e176d4988ea6342eb170fde7c960f3`。
- 重算后 Week 6 路由基线全 development 平均延迟为 5727.70 ms。step
  38/76/113/151 的延迟比分别为 1.6312/1.3221/1.3155/1.4894，gold-support 综合分
  分别为 0.258513/0.723404/0.746154/0.733077；step 76/113/151 的三场景
  task/format/support 门禁均通过，step 38 另有行程 task/format 回退，但四个候选均未
  通过全局 1.25 延迟门禁。selector 仍返回 `BLOCKED_NO_ELIGIBLE_CHECKPOINT`，未创建
  parameter lock，test 未读取；该 protocol-v4 提交当时的完整 `unittest` 为 412/412。

## 2026-08-21：Week 7 protocol-v5、参数锁与一次性最终评测

- protocol-v4 的语义门禁已通过但输出长度差异使全局延迟仍不可接受。提交 `64a5a7a`
  新增独立 protocol-v5：继续绑定 v3 config/data/checkpoints，只把所有比较角色统一为
  BF16、static KV cache、Transformers compile、32-token warm-up、CUDA 同步计时和
  gold-evaluable 支持口径；未重训、未重切分、未读取 test。
- v5 attempt1 job `29452655` 在完成 Week 6 与四个 checkpoint 后因项目盘全局 100%
  满而 `FAILED 1:0`，zero-shot raw 为 0 字节，整个 attempt 排除。`29456882` 因提交器
  预建输出目录而在 13 秒内被不可覆盖门禁拒绝，未执行模型推理。新 attempt
  `29456896` 改用 home 持久化与节点本地 compile cache，在 L40S 上以
  `COMPLETED 0:0` 运行 01:28:19，六个角色和 protocol summary 完整。
- v5 development 中 step 38/76/113/151 综合分为
  0.074359/0.642718/0.645237/0.740904，全局延迟比为
  1.0405/0.9417/0.8609/0.8809，失败率均为 0。selector 以 4/4 eligible 选择
  checkpoint-151；selection SHA-256 为
  `68bfbedbc3b61494daf6fdf0911486d60339b181479e05406aa0c0434dc2ca50`。
- 审计发现旧 final runner 仍固定 NF4/dynamic/旧支持口径；提交 `8619b76` 修复为参数锁
  哈希绑定并复用完整 v5 runtime。完整 unittest 414/414、远端定向 22/22、compileall
  和 diff 检查通过。parameter lock canonical SHA-256 为
  `1b3f3ffafc2f549ca29034fcee505e346bcb70bc8ce974adcdbb83ad6d38adef`。
- 唯一 final-test job `29459265` 在 L40S 上运行 00:40:50，状态 `COMPLETED 0:0`；
  test marker 绑定 test SHA-256 `2137eaf46e927366e4991b04306d138f2217eb82d8e7b8cba17f82a282aa2d99`
  并验证 7 个 artifact hash。统一模型 test 综合分 0.744987，商品/售后/行程为
  0.153846/1.000000/0.996667，平均延迟 7173.16 ms、失败率 0；Week 6 路由基线综合
  0.061840、延迟 8250.70 ms，zero-shot 综合 0.075577、延迟 4788.49 ms。三场景、
  支持数、格式、全局延迟和失败率门禁全部通过。
- 对话 test 自动指标为格式 1.0、上下文召回 0.878472；人工四维仍为
  `PENDING_REAL_HUMAN_INPUT`。真实审核偏好对仍为 0，DPO 保持 `SKIPPED`。旧工作树、
  Week 6 终态、Week 3 v2 与历史失败 evidence 均未修改。

## 2026-08-21：Week 7 对话上下文与人工标注台审计

- 用户在固定人工队列首条发现回复语义顺序异常。代码核验确认 `_dialogue_row` 对每个
  follow-up 先追加 assistant、再追加其对应 user；初始任务也没有先获得实质 assistant
  结果。该路径用于 v3 train/development/test 的 450/24/24 条对话。
- development 固定队列只读检查为 24/24 命中 `assistant_precedes_its_prompt`；人工结果
  文件不存在，真实评分仍为 0。v3 数据锁、checkpoint、raw 和一次性 test marker 均未
  修改，也未在页面重排后复用旧 raw。
- 本地标注台增加上下文完整性门禁：逐条展示提前出现的 assistant 及其本应跟随的 user，
  前端禁用四维控件，后端拒绝绕过提交；当前状态固定为
  `BLOCKED_INVALID_SOURCE_CONTEXT`。标注辅助只解释问题，不代填或建议具体分数。
- 历史对话格式/字符串包含式上下文召回保留为原始输出，但不能证明真实多轮连贯性；
  三个核心场景的独立 task/Schema/支持/延迟/失败率结果不因该顺序审计被重写。恢复对话
  验收需要新数据身份和对应的新模型输出，不能修改 v3 锁。完整 unittest 418/418。

## 2026-08-21：Week 7 对话 development 修复与标注台恢复

- 新建 development-only `week7_dialogue_review_20260821_v2`，绑定 v3 数据锁、原
  development/queue SHA 和 checkpoint-151 adapter；不读 test、不训练、不改变 final-test
  结论。24 条均按 user→具体 assistant 回答构造，5/6/7/8 轮各 6 条，首轮任务 24/24
  获得具体 JSON 回答，旧提前回复命中 0，图片仅首次用户轮 24/24。
- GPU 调度和失败如实保留：初次无分区提交被调度器拒绝，`29479321`/`29479416` 在
  运行前取消；`29479456` 因离线
  cache 路径错误 12 秒失败且未加载模型；`29479500` 完成生成后因 home quota 落盘失败，
  未产生 raw；最终 `29479822` 在 A100 上 `COMPLETED 0:0`，耗时 00:11:14。
- 有效 run 24/24 成功、失败 0，raw SHA 为 `9cb8cafc...cd162`；development 自动格式
  合规率 0.875、字符串包含式 context recall 0.583333，仅作自动辅助，不替代人工四维。
- 标注台状态为 `READY_FOR_REAL_HUMAN_INPUT`、完成 0/24、无效上下文 0/24，控件已启用；
  虚假自审探针返回 HTTP 400，人工结果文件仍不存在。当前完整 unittest 422/422。

## 2026-08-22：Week 7 corrected dialogue 单人人工四维完成

- 真实单人操作者在一个 review session 内完成 corrected development 固定队列 24/24；
  26 条 append-only 记录对应 24 个唯一样本，其中 2 条 revision=2。全部记录绑定同一
  dataset/queue/development/raw SHA，最终 24 条 reviewer 非空、自审确认和四维分数完整。
- 最终决定 24 `pass`、0 `rework`、0 `reject`。历史图片指代/需求迭代/上下文承接/
  逻辑连贯均分为 4.541667/4.625000/4.500000/4.708333，四维未加权均值 4.59375。
- 人工结果 SHA 为 `bdec2d18...af932`；原记录保持在忽略目录，聚合证据见
  `experiments/week7_dialogue_human_review_20260822_v2.json`。该 development 人工结果
  不改写已消费且存在构造缺陷的 v3 test 对话；DPO 仍因 0 条审核偏好对 `SKIPPED`。

## 2026-08-22：Week 7 corrected dialogue 的 Week 6 人工对比入口

- 新增 development-only `dialogue_comparison_v1`，绑定同一 corrected 24 条及 Week 6
  三个冻结 adapter SHA；按父场景商品/售后/行程 8/8/8 路由，不读取 test。
- Spartan job `29491047` 在 `gpu-l40s`/`spartan-gpgpu003` 完成，耗时 00:09:10、
  exit 0:0；24/24 唯一样本、失败 0。格式合规率 1.0、自动 context recall 0.555556，
  raw SHA `c3effb6d...318e59`。自动值不代替人工四维结论。
- 标注台已切换到 Week 6 routed 输出，状态 `READY_FOR_REAL_HUMAN_INPUT`、0/24、无效
  上下文 0/24。该真实评分用于与已完成的 multitask 24 条配对比较，Agent 未代填。
- 同一真实操作者完成 Week 6 routed 24/24；25 条 append-only 记录含 1 次 revision=2，
  最终均为 `pass`、自审 24。四维均分 4.666667/4.333333/4.541667/4.708333，总均值
  4.56250，结果 SHA `af3721d2...d49f93`。
- 与 multitask 配对后，总均值差 +0.03125，样本级 10 胜/7 平/7 负；图片指代、需求
  调整、上下文承接、逻辑连贯差值为 -0.125/+0.291667/-0.041667/0。该同 session
  单人结果仅作描述性人工对比，不宣称统计显著。

## 2026-08-22：Week 7 单次 mDPO-style 消融

- 两组真实四维评分经确定性偏好派生和 Agent 对抗审计产生 16 对：7 个总分平局和 1 个
  chosen 非 JSON 被拒绝；16/16 反转探针被拒绝。按场景×chosen 来源各留 1 对，锁为
  10 train/6 validation；Agent 审计没有冒充真人评分或显式 pair choice。
- job `29491859` 在 A100 上 `COMPLETED 0:0`、耗时 00:01:46，执行 5 次 optimizer
  update；LoRA 梯度范数 0.03499–0.17151，确认并非空跑。train 准确率/平均 margin 为
  0.8/+0.01861，但 validation 仅 0.3333/-0.00981，未通过 0.5/>0 门禁。
- 新 adapter SHA `3791896e...39b64` 不被选用；门禁后不再运行核心 development 生成、
  不重试调参、不读取 test。checkpoint-151 继续作为选择结果。

## 2026-08-22：Week 7 终态对抗审计与目录收敛

- 新增确定性终态审计和 3 个定向测试；11/11 个证据篡改/门禁绕过反事实全部被拒绝。
  机器证据为主要判断，既有真人 development 评分为辅助证据，Agent 没有冒充人工。
- 审计允许当前实现进入 `dev` 集成，但完整能力结论仍为
  `FAIL_KNOWN_V3_TEST_DIALOGUE_INVALID`；不重跑 test，不选择验证回退的 DPO adapter，
  不进入 `stg`。
- 将作废 v1/v2 锁和传输包、失败构建及临时预检脚本共 408,127,632 字节移入 Windows
  回收站。v3 锁/归档、修复对话 raw、真人评分、偏好锁和 mDPO 证据均保留。
- 完整 unittest 431/431、v3 数据锁复验、10 份 Week 7 Slurm shell 语法和差异检查通过。

## 2026-08-22：Week 7 corrected-dialogue v4 全量修正开始

- 用户直接 supersede v3 对话 test 不可重开的旧约束，接受 ADR-031。v3 全部
  产物保持不变；新建分支 `codex/week7-dialogue-correction-v4`。
- 新锁 `week7_corrected_multitask_context_20260822_v4` 已本地实际构建：
  train/development/test 为 3000/114/114，训练比例为三核心场景各 760、通用
  270（9%）、对话 450（15%）。三分区五维冲突为 0。
- v4 test 另外排除已消费 v3 完整 identity manifest 的 3228 行；
  sample/source/image/group/template 五维重叠均为 0。锁 SHA-256 为
  `000a2e57620428034da27e03ba3c92483e9c147032166ad273ed089fbb97c9fa`，test 仍是
  `LOCKED_UNCONSUMED`。
- 修正数据构造的 user→assistant 对齐、5–8 轮、首轮图片和最终结构化目标；
  SFT 改为监督所有 assistant token span。development/test 改为逐 assistant 轮生成，
  每轮使用模型自己的前序回复，不再 teacher-force 金标中间答案。
- 自动 development selector 与一次性 corrected-dialogue test runner 已实现，绑定
  config/data/training/checkpoint/raw/metrics 哈希，重算全候选门禁并原子消费 test。
  新的人工输入不再是 v4 前置；历史 24/24 真人结果只作辅助描述。
- 当前实测 v4 定向 10/10、全部 Week 7 测试 71/71、完整 unittest 441/441
  通过。GPU 训练、checkpoint
  选择和新 test 尚未运行，不记录任何新模型指标。
- 实现提交 `d14a129` 已推送到 `origin/codex/week7-dialogue-correction-v4`；
  Spartan 同一提交和 v4 锁复验 PASS。唯一 L40S 训练作业 `29504508` 已提交，
  当前 `PENDING(Resources)`，Slurm 估计 2026-08-22 20:59:57 AEST 启动；未提交
  A100/H100 竞争副本，未消费 test marker。
- 远端 GPFS 在删除已验证传输压缩包后实测剩余约 992 MiB/8784 inodes；
  v4 锁已解包且验证，训练期持续监控空间。本地传输包、不完整首次锁和
  空临时目录已进入回收站，权威 v4 锁保留。
- 训练 attempt 1 `29504508` 于 step 38 进入首次逐轮 development 评估后
  `FAILED 1:0`。根因为生成的 assistant 文本以裸字符串追加回 Qwen 多模态消息，
  processor 要求标准 text content block，因此报 `TypeError: string indices must be integers`。
  该 attempt 无 checkpoint、无 development raw/metrics，test 未消费；训练 loss 日志如实保留。
- 恢复修复将生成回复统一写为 `[{"type":"text","text":...}]`，并将规范化
  assistant content 可逆还原为评分文本。配置、数据锁、run ID 和超参数不变；
  失败 attempt 不覆盖，修复通过新增回归后才允许 attempt 2。
- 修复后两条逐轮生成路径均由 strict content-block 回归覆盖；v4 定向 13/13、
  Week 7 74/74、完整 unittest 445/445、v4 数据锁复验、两份 v4 Slurm shell
  语法和 `git diff --check` 均通过。`.gitattributes` 固定 Week 7 config 与
  Spartan shell 为 LF，避免 Windows checkout 改变哈希或破坏 shell 解析。
- 修复提交 `c002a78` 已推送隔离分支；attempt 2 job `29505375` 于
  2026-08-22 17:35:01 AEST 提交到 `gpu-l40s`，输出目录
  `work/week7_multitask_v4/run_c002a78_attempt2`。config SHA、canonical lock SHA、
  `week7_multitask_context_sft_20260822_v4` run ID 均未改变，初始状态
  `PENDING(Resources)`；旧 attempt 与 v4 test 未消费状态保持不变。
- job `29505375` 在 step 38 完成首次 development 评估并产出完整 checkpoint 后，
  于 step 39 以 `FAILED 1:0` 终止；日志未给出主异常堆栈，仅留下 tqdm 退出期
  `Exception ignored`。可核验证据为 114/114 raw、metrics SHA
  `6b7b37...`、adapter SHA `f425646...`，failure rate 0；首次综合分 0.339991，
  商品/售后/行程为 0.564706/0.52/0.045，对话 automatic 0.231308。
- 排除超时、CPU OOM、test 消费及不完整 checkpoint 后，按既有恢复契约从同目录
  `checkpoint-38` 做一次受控 resume；config/data/run/git identity 不变，resume job
  `29506065` 已提交。若同点再次失败将停止，不继续盲重试。
- `29506065` 成功越过 step 39，但在完成 step-76 生成后写评估文件时明确报
  `[Errno 122] Disk quota exceeded`；仅留下 0-byte partial，未形成第二个有效
  evaluation/checkpoint。审计确认项目内可再生成的 Apptainer/container cache 26 GiB
  与 pip cache 1.2 GiB，不属于数据锁、模型、checkpoint 或报告；已删除缓存内容及
  该 0-byte partial，保留 step-38 全部证据，GPFS 可用空间由约 0.9 GiB 恢复到
  约 28 GiB。随后提交同身份 quota-recovered resume job `29506362`。
- `29506362` 终态 `COMPLETED 0:0`，运行 02:37:50，从相同 `checkpoint-38` 恢复并
  完成 step 38/76/113/151/188/226 六次 development 评估；连续两次未改善后在
  step 226 按 patience=2 早停。run summary SHA-256 为
  `5af980efc851e2e0c15d96ea13853e3728fa194618fcd737ea976e3926e2e5a5`，最佳综合分
  0.833980 位于 step 151，最终 adapter SHA-256 为
  `296ad3f362e559738b55d93e2164f549631994138f5acaed72d8b4b3b48d9d86`。
- 不可覆盖 selector 已实际执行并以 `no v4 checkpoint passed the automatic development
  gate` 拒绝 selection。step 151 虽为综合分最高，但格式/上下文召回/上下文值/任务键/
  任务值/逐轮覆盖/automatic composite 均低于预注册阈值；step 226 也未通过。
  selection 文件与 v4 test consumption marker 均不存在，test policy 仍为
  `LOCKED_UNCONSUMED`；按一次性 test 规则未提交任何 test job，未生成 Week 6/zero-shot
  的 v4 test 对比，也未快进 `dev`。

## 2026-08-23：Week 7 v4 fix1 独立身份闭环

- 用户直接授权的 gate repair 以新提交 `6bb5322`、新配置
  `qwen3_vl_8b_multitask_context_v4_fix1.json` 和新身份
  `week7_corrected_multitask_context_20260823_v4_fix1` 执行；v3 与首版 v4 的数据、
  raw、checkpoint、报告和失败结论均未改写。
- fix1 锁为 train/development/test=3000/114/114，canonical SHA-256
  `7f66795c69f8cb35cafa712e7847155708a662b88d069824b60706f6903ea9a7`；训练实际
  商品/售后/行程=600/840/840、通用正则=270（9%）、对话=450（15%）。三分区五维
  冲突为 0，并额外排除 v3 与首版 v4 identity manifest；test 未消费。
- fix1 锁定显式工具请求、gold-plus-anchor 对话评分、3072-token 生成上限、
  evaluation protocol v4 支持口径和场景 loss multiplier 0.8/1.1/1.1；所有调整只依据
  development，以新 config/data/run identity 实施，未读取旧或新 test。
- 首次 job `29526506` 因 `HF_HOME` 错指仅约 35 MiB 的 runtime-home 而失败；没有把该
  环境故障冒充训练结论。安全恢复保持 config/data/run/git identity 不变，改用真实
  25 GiB Hugging Face cache 和非覆盖目录 `run_6bb5322_attempt2`；job `29526965`
  `COMPLETED 0:0`，耗时 02:43:24，L40S 单卡，step 226 按 patience=2 早停。
- 六个 development weighted composite（38/76/113/151/188/226）为
  0.353427/0.729860/0.751086/0.764049/0.753292/0.752986；best/final adapter 为
  checkpoint-151，SHA-256 `b42aeeb...5131bc`。run summary SHA-256 为
  `6d5400fd...491d0`，train loss 0.182206，峰值 allocated/reserved 显存为
  15,191,208,448/31,545,360,384 bytes。
- step 151 的商品/售后/行程 composite 为 0.153846/0.970000/1.000000，对话格式、
  context recall、context-state value、task key/value、sequential coverage、automatic
  composite 为 1.0/0.854167/0.791667/0.962384/0.820023/0.725585/0.877630；总体失败率
  0，平均延迟 11,503.48 ms。支持数为核心场景各 30、对话 24。
- 不可覆盖 selector 实际重算 6 个候选，0/6 通过全部门禁；step 151 仅
  sequential coverage 0.725585 < 0.75，其他候选也至少一项失败。阻断证据状态为
  `BLOCKED_NO_ELIGIBLE_CHECKPOINT`，文件 SHA-256 `782e92ab...cc104`，selected
  checkpoint 为 null、`test_read=false`。
- 因 development 门禁失败，唯一 fix1 one-shot test 未提交且不得重跑；没有 fix1
  Week 6 routed/zero-shot test 对比。DPO 保持既有一次 validation 失败后关闭；分支不
  快进 `dev`，不进入 `stg`，不打标签。
- 终态本地复验：fix1 定向 26/26、全部 Week 7 79/79、完整 unittest 450/450；fix1
  config、数据锁/五维隔离、两份 v4 Slurm shell 语法和 `git diff --check` 均 PASS。

## 2026-08-17：Week 6 最终数据锁与 8B pilot 准备

- 独立核验 Week 5 最终单轮闭环为 79,936 成功、64 最终失败，三场景最新人工修订
  各 100 条；权威 10,000 条对话及 100 条人工验收文件哈希与记录一致，但不自动
  混入本次单场景训练。
- 历史 Week 6 v1 锁仅含商品/售后/行程 10/8/9 条人工修订，保持不可变。v2 在错误
  地将人工受控词表校验用于 silver 后停止；v3 发现仍使用 OpenAI `image_url`，不能
  由 Transformers 4.57.1 自动加载为视觉输入，均保留为失败证据。
- 活动数据锁 `week6_week5_final_human300_20260817_v4` 使用原生
  `type=image/path=<project-relative-path>`。训练/验证数量为商品 47,428/2,529、
  售后 19,026/965、行程 9,538/450；人工修订在训练/验证中分别为 94/6、97/3、
  94/6。manifest SHA-256 为
  `0b8d9f96b1237b16fc40f510916d6fb07178dfcbb13ab21647480de4cf7adf0e`，split
  SHA-256 为 `450abbe7abd5d1c2dc4a585fc474378cf9771b738077fadfe0ebb603f0df0cc0`。
- 增加固定训练超参数门禁、运行时 LoRA 目标层和基座冻结检查、断点恢复入口、
  adapter-only 重载验证、显存/耗时/Slurm 元数据记录及版本化 config/run ID 要求。
- 本地完整 `unittest` 337/337、Week 5 五维隔离、Week 3 v2 验证、六份 v4 数据契约、
  Slurm shell 语法和 `git diff --check` 均通过；GPU pilot 尚未运行。
- 提交 L40S pilot job `29296577`，32 条训练/32 条验证、最多 10 steps；截至
  01:16 AEST 为 `PENDING(Resources)`，保留原作业和队列位置。用户随后批准成功后
  通过确定性 gate 自动推进三场景正式训练，并要求任务级分片并行。
- 新增 pilot gate、正式 `train-full`、三场景独立单卡模板、流式 JSONL offset dataset
  和图片 manifest/audit array。全量锁归档已在 Spartan 逐分片重组并通过 SHA-256；
  79,937 个唯一引用图片的本地 manifest SHA-256 为
  `1afd768a1996a7ebd7004e1ef2fcdcff60ad7a54ce4161efe6673c4e0a27e5a7`。远端当前源图
  50,423 张，正式训练前仍须完成分片审计并只补齐实际缺失文件。
- 更新后完整 `unittest` 343/343、Python 编译、五份 Slurm shell 语法及
  `git diff --check` 通过；尚无 pilot loss、显存或训练完成结果。
- 图片首轮 CPU array `29297594`/merge `29297595` 完整检查 79,937 项，发现
  15,129 缺失和 1 个大小不符。仅补传这 15,130 项；562,031,601-byte 归档 SHA-256
  为 `62fe6e80ecc0bfa0cbd08a0b082fef193121248c8e0d32f71d884566ac5151e0`。大小不符的
  few-shot 拼图先移到审计目录备份。复审 `29297871`/`29297872` 为 79,937/79,937、
  failures=0、`status=ok`。
- pilot `29296577` 于 02:00:30 获得 L40S，但在 11 秒环境检查阶段 `FAILED 1:0`：
  原 venv 为 torch `2.13.0+cu130`，节点驱动仅支持 CUDA 12.8；同时 bnb 0.50.0
  提示缺少 kernels。没有模型下载或训练 step。新增不可变 CUDA 12.8 venv 安装链，
  成功后只重提一次 pilot。
- 环境作业 `29297982` 与 `29305189` 均因共享 GPFS 配额失败；第二次失败已将根因
  从容量收窄到 project inode `489K/489K`。经用户逐项确认，只清理可重建缓存以及
  失败/过期 venv；清理后文件系统约余 76 GiB、68K inode，项目数据、代码、日志、
  训练结果和 4B hub 模型未删除。
- CUDA 12.8 setup 收窄为训练专用依赖并关闭 pip 下载缓存，不再安装 API/data 聚合
  依赖。当前本地验证为定向 12/12、完整 unittest 346/346、`bash -n` 与
  `git diff --check` 通过。
- 精简环境作业 `29305905` 已 `COMPLETED 0:0`，`pip check` 和固定版本导入通过。
  新 pilot `29305985`、gate `29306001`、正式商品/售后/行程
  `29306002`/`29306003`/`29306004` 已构成严格 `afterok` 链；截至 12:20 AEST，
  pilot 为 `PENDING(Priority)`，Slurm 预计 22:40 启动。错误哈希依赖
  `29305986`–`29305989` 已在运行前取消，pilot 没有重复提交。
- pilot `29305985` 后续实际运行 1:20 并 `FAILED 1:0`：环境 gate、32/32 数据验证和
  8B 权重加载均成功，但 `AutoProcessor` 缺少 torchvision；没有训练 step、loss、
  checkpoint 或 adapter。下游 `29306001`–`29306004` 自动取消。修复将官方匹配的
  torchvision `0.23.0+cu128` 加入 setup、环境门禁及受控 venv 修复脚本；本地定向
  12/12、完整 unittest 346/346、Python 编译、两份 shell 语法和差异检查通过。
- 修复作业 `29309546` 已 `COMPLETED 0:0`，torchvision 导入和 `pip check` 通过。
  修复后 pilot/gate/正式三场景为 `29309556`/`29309557`/
  `29309558`–`29309560`；截至 14:50 AEST，pilot 为 `PENDING(Priority)`，Slurm
  预计 2026-08-18 06:00 启动，其他作业保持依赖关闭。

## 2026-08-19：Week 6 专项晋级与独立最终评测

- 行程 refinement `29375367`、候选评测 `29408124` 和 comparison `29412603`
  均完成；固定 64 条候选九项结构计数全为 64/64，comparison 无回退原因，获胜
  adapter SHA-256 为
  `7ab168a0f7073f2fad3369c028f744585362a0668f77c024098d9b27d92c9a6a`。
- 参数锁定后恢复并验证冻结 `week3_evaluation_v2` 450/450；CPU preflight
  `29418839` 的 77 项测试、HF cache 和三 adapter 哈希通过。
- 严格串行完成商品/售后/行程最终评测 `29418875`/`29419327`/`29422130`，样本
  200/150/100，JSON 为 100%/100%/95%，Schema 为 100%/96.67%/85%。三作业均
  `COMPLETED 0:0`，冻结结果未用于新一轮调参。
- 结论：训练、专项门禁和独立评测闭环完成；格式质量总体稳定，但商品多标签、售后
  严重度/关键信息、行程约束泛化仍是已记录局限，不能笼统声称业务效果优秀。
- 收尾提交 `524a30c` 的完整 unittest 370/370 与差异检查通过，并已推广到 `dev`、
  `stg`。adapter-only、配置、训练摘要、专项门禁、最终评测、日志、报告和源代码快照
  已写入版本化归档 `week6_quality_closeout_20260819_524a30c`；Spartan、本地 E 盘及
  `trip-api-sg` 三处均按 `SHA256SUMS.final` 校验通过，清单 SHA-256 为
  `8c1ac916409d2446bd0b80f2a70ebec92747ca25df93b00a5ebf997b75856b7c`。

## 2026-08-18：Week 6 固定链终态与行程专项质量门禁

- 消息规范化修复提交 `3d6bc81df8c4afd496e1e78d41c6b4bfa07c7bf4` 对应唯一固定链
  pilot/gate/商品/售后/行程 `29312210`/`29312212`/`29312214`/`29312215`/
  `29312217` 均为 `COMPLETED 0:0`；三场景 `run_summary` 均完成 adapter-only
  保存和磁盘回载。
- 最佳 validation loss/checkpoint 分别为商品 `0.2234927862882614`/
  `checkpoint-5930`、售后 `0.008334202691912651`/`checkpoint-2856`、行程
  `0.005681941285729408`/`checkpoint-1620`。三个最佳 checkpoint adapter 已以
  SHA-256 覆盖清单备份到 Spartan 专属 GPFS、本地 E 盘和 `trip-api-sg`。
- 行程原锁结构审计显示 validation 全项通过 `0/450`，因此上述 loss 只作为目标拟合
  指标，不作为业务优秀结论。派生 silver 修复锁保持原锁不可变，train/validation
  `9538/450` 条均通过确定性结构审计，派生目标统一权重 `0.5` 且不继承人工身份。
- 增加同一输入 SHA、样本 ID、生成参数和数据锁身份的 adapter 对照门禁。候选必须增加
  全项通过样本，且 JSON/Schema、天数、约束覆盖和必需元素等检查均不回退，才允许
  作为更优结果；当前状态仍是 `BASELINE EVALUATION PENDING`，未把准备工作写成效果提升。

## 2026-08-16：Week 5 多轮对话与人工验收最终完成

- 保留原 L40S 作业的同时，按 ADR-024 使用独立输出的四个确定性互斥分片；每片
  1,500 条。对主 run 的索引 0–3999 生成不可变 4,000 条前缀快照，随后显式合并为
  `week5_dialogues_merged_10000_20260816_522b4af`。
- 权威 run 含 10,000 个唯一 `dialogue_id`，索引 0–9999，三场景分布
  3334/3333/3333，消息数 8–12；严格角色交替、图片字段、Schema、配置和 qualified
  集合哈希均通过，duplicate/conflict/missing 均为 0。
- 候选/manifest SHA-256 分别为
  `7e00f326fc1b2896a6efcc5c2f6c1f67ffdb728501ba3eb9ba65efdb28265d99` 与
  `02795c8df44ca564dcd873974c5bcb6939c41bf38bee2f6c1f550d7916669556`；本地 JSONL、
  gzip 和 manifest 与远端一致。
- 固定 100 条人工验收队列 SHA-256 为
  `45c34b558456577d5eaaf9b74cf04a8766b0160ec05935a181131db66134634e`。本人实际完成
  100/100；记录 ID 唯一且全部属于队列，reviewer 均为 `Larry Fan`，五项 checks
  完整，100 条 decision 均为 `pass`，人工验收 JSONL SHA-256 为
  `eb3a6f436a78389e919b86d3756fc2208265bac7f4420158dc597d5bc4682e54`。
- 只将抽样通过的 100 条计为人工 accepted，其余 9,900 条保持未人工验收候选。
  Week 5 按批准的单人预算内口径完成；干净 checkout 完整 unittest 330/330、Week 5
  `validate-pools`（80,000 个唯一 sample/image，`status=ok`）和 `git diff --check`
  通过。本轮未执行 Week 6。
- 修复最终汇总固定读取旧 `dialogues/` 目录的问题；`report` 现在必须显式指定权威
  `--dialogue-run-id`，并从 run-scoped 目录读取 10,000 条候选和 100 条人工验收。
  同时修复 v2 `turns` 的平均轮次统计，更新最终质量报告与机器可读当前状态。
- 推广前干净 checkout 发现 4 份已被测试引用的 Qwen3-VL-4B 脱敏配置仍未跟踪；补齐
  这些轻量文件后重新验证，避免依赖开发机未提交文件产生假阳性。

## 2026-08-14：Week 5 最终闭环与 Week 6 数据锁定

- Spartan merge job `29190753` 为 `COMPLETED 0:0`；下载归档 SHA-256
  `a9ae67cb677bb940c94197e692ba1ce85671a83cba9e5fb070b012dfaa43abee`。
- 最终覆盖为 79,936 条 Schema-valid 成功与 64 条最终失败，共 80,000 个唯一候选；
  失败含 44 条不可读输入、19 条 Schema 错误和 1 条 JSON 解析错误。44 条不可读输入
  保持 `input_error`，未请求模型或替换候选池。
- 全量预标注已同步到本地标注台，保留真人完成的商品/售后/行程 10/8/9 条修订；
  自动化没有增加人工身份、自审、交叉复核、核心抽检或 accepted。
- Week 6 锁定版本为 `week6_week5_spartan_merge_20260814_8cbfd8d_v1`；manifest SHA-256
  `877c16d8ee79d9b0601fe9b6a5f531dfcbd81bb7e16f3fbd6e2526b760d62198`，split SHA-256
  `7ec02ed629a4b434dae39c5eb32ff783ab7fafdde8ac151e4124b34a294fc018`。
- 训练/验证计数分别为商品 47,393/2,564、售后 19,039/952、行程 9,502/486；
  27 条真人修订权重 1.0，其余 79,909 条 silver 权重 0.5。六份 JSONL 均通过流式
  `validate-data`；本机未安装 GPU 训练依赖，因此环境检查如实为 `missing_dependencies`。
- 当前完整 unittest 为 312/312，Week 5 `validate-pools` 返回 `status=ok`。

## 2026-08-14：Week 5 单人预算内抽样验收

- 用户确认全量人工修订和 0.9 万条人工对话验收在单人条件下不可执行，继续遵守此前
  已记录的 3 小时、全部操作低于 500 次约束，不得美化或伪造结果。
- 标注台改为每场景 100 条确定性人工验证队列，保留既有真实修订商品 10、售后 8、
  行程 9 条；当前剩余首轮任务为 90/92/91，共 273 条。
- 每场景队列固定包含 10 条现行 SHA-256 盲复核候选和其中 3 条核心抽检候选；另从
  自动生成的 10,000 条对话候选中固定抽取 100 条人工验收。完整人工预算为
  300 + 30 + 9 + 100 = 439 次。
- 未进入队列的 79,636 条 Schema-valid 预标注继续保持 silver；自动生成的对话只能
  作为候选，未经本人检查不能计为人工 accepted。
- 标注台已在 `http://127.0.0.1:8095` 重启；定向测试 5/5 和配置 JSON 校验通过。

## 2026-08-15：Week 5 预算内人工验收完成与对话链路准备

- 本人 Larry Fan 已完成商品/售后/行程各 100 条真实人工修订和内联自审；三个场景
  各完成 10 条确定性盲二次复核与其中 3 条核心抽检。共 300 条修订、300 次自审、
  30 次盲复核、9 次核心抽检，均绑定真实 session，未由自动化代填。
- 300 条最新 canonical annotations 全部通过对应 Schema；同一 sample/revision/stage
  的 review session 无重复。其余 79,636 条有效预标注继续保持 silver。
- 新增中文展示镜像导出：三场景各 100 条，翻译稳定字段名及已知枚举，保留自由文本
  与 canonical JSON，并用 SHA-256 绑定原人工标注，不能反向覆盖训练数据。
- 新增 Spartan 对话生成 sbatch 及严格 resume identity。目标仍为 10,000 条自动候选，
  之后固定抽样 100 条由本人验收；当前候选和人工合格对话均为 0。
- 本地定向测试 7/7、完整 `unittest` 319/319、Slurm shell 语法与
  `git diff --check` 通过。精确 OOD shell 恢复后，Spartan checkout 已 fast-forward
  到 `3396c41`。300 条人工/QC 归档已传入项目限定目录，归档 SHA-256 为
  `e288c8ef21a1eff59e836516e8b380665b481cd21080e33309d57f48f0cc9967`；旧 27 条逐行
  均包含于新文件，安装前副本已保留，六个安装后 JSONL 哈希与本地一致。
- 唯一 Week 5 对话候选作业 `29226849` 已提交到 `gpu-l40s`，run ID 为
  `week5_dialogues_qwen3_vl_4b_20260815_3396c41_a`，目标 10,000 条；首次状态为
  `PENDING (Priority)`，没有提交竞争作业。

## 2026-08-12：Spartan 容器启动错误修复

- `29114276` 确认因 vLLM 镜像仅提供 `python3` 而失败；入口已改为可预检的
  `python3`，并为每个 Slurm job 使用不可覆盖的版本化 vLLM 日志。
- `29116649`/`29116828` 越过 Python 入口后暴露第二个根因：FlashInfer 仍尝试写入
  已满的 `/home/yzhang3504/.cache`，两次均在模型请求前 `FAILED 4:0`，未产生结果。
- 提交 `3600a7b` 将 Apptainer `--home`、XDG cache 和 FlashInfer workspace 强制绑定到
  Trip 专属 GPFS；登录节点预检确认容器 HOME 可写且位于专属目录。
- `29116943` 已证明 vLLM 健康端点返回 200，随后因远端缺少 Week 4 development
  few-shot manifest 而在首个模型请求前失败。40 个冻结依赖文件以不覆盖方式补齐，归档
  SHA-256 为 `216b458546cbcc61e326c56f3b38517f1f06a96c9f7cdbec85078bee469ba0ff`；
  容器内三场景 manifest 校验为 12/12/12。
- 唯一替代 benchmark `29117353` 已提交到 `gpu-l40s`，当前 `PD(Resources)`；最新
  `squeue --start` 预计为 2026-08-13 00:26:54 AEST。尚无可计数结果，剩余分片未提交。定向测试 3/3、
  完整 `unittest` 300/300、shell 语法和 `git diff --check` 通过。

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
- 2026-08-24：建立三个隔离 worktree 并合并系统修复：Qwen3-VL fail-closed 运行时、
  Week 5 v2/新数据锁、CLIP/Milvus 和统一发布封装。完整测试 482/482 通过。
- 2026-08-24：实际生成 1,000 x 512 CLIP 向量；Milvus 1,000 条 CRUD、HNSW/COSINE、
  100 查询实测完成，平均/P95 2.2355/2.4097 ms，Recall@10=1.0。
- 2026-08-24：Week 5 v2 审计为 80,000 候选、64 条修复队列、五维评测冲突 0；因本轮
  Spartan 尚未登录，64 条重推理、Prompt pilot、继续 SFT 和模型门禁保持未完成。
- 2026-08-24：三路临时 worktree 已合并并删除；旧 closeout 的 11,037 个输出文件与主
  项目逐文件 SHA-256 一致后移除。当前本地与远端长期分支仅 `dev/stg/main`。
- 2026-08-25：Prompt pilot 在固定 development 集比较三种候选，商品选择 compact、
  售后选择 evidence、行程选择 current；发布 Prompt 分别锁定为
  `system_repair_product_compact_v3`、`system_repair_after_sales_evidence_v3` 和
  `system_repair_itinerary_structured_v4`。
- 2026-08-25：Spartan job `29560346` 完成 Week 5 v2 的 64/64 条 Qwen3-VL-8B
  修复，最终合并 80,000/80,000 Schema-valid silver，人工 accepted 统计不变。
- 2026-08-25：continuation SFT job `29562078` 在单个 L40S 上 `COMPLETED 0:0`，
  `04:48:36`；step 112 按 patience=2 早停并回载最佳 checkpoint-87。最终 adapter
  SHA-256 `c2fbb5c7...eaa2a`，adapter-only 磁盘回载验证通过。
- 2026-08-25：候选 checkpoint-87 总体加权 0.920725、核心三场景加权 0.905382，
  商品/售后/行程为 0.716146/1.000000/1.000000，对话自动综合 0.982097，失败率 0。
- 2026-08-25：job `29565493` 完成旧 unified 与 zero-shot 后暴露单场景汇总和失败前
  原始输出未持久化问题；代码修复后 job `29567157` 完成 Week 6 routed。同集四路总体
  加权为候选 0.920725、旧 unified 0.750034、zero-shot 0.084010、Week 6 routed
  0.061806。开发门禁 `PASS`，失败项 0，SHA-256 `e7ba5bc7...0402`。
- 2026-08-25：唯一一次 fresh test job `29569338` 已通过 `spartan-trip` SSH 提交，
  申请一个 L40S 或 A100、12 CPU、96 GB、1 小时，等待调度期间未读取 test 内容。
- 2026-08-25：job `29569338` 在 A100 上 `COMPLETED 0:0`，`00:42:49`；120/120、
  失败率 0，总体/核心加权 0.936170/0.926880，商品/售后/行程
  0.780639/1.000000/1.000000，对话自动综合 0.973330，三场景 JSON/Schema 均 1.0。
- 2026-08-25：fresh-test raw/metrics SHA-256 `34446498...eb19`/`853bd67e...1018`
  与 completed 消费标记一致；final gate `PASS`、失败项 0，SHA-256
  `9574b05b...a77d`。发布配置切换到 checkpoint-87，四层本地私有包复验通过。
- 2026-08-25：生产 smoke 前两轮分别暴露任务输入契约和对话纠错契约问题；第三轮
  `29571065` 进一步确认 `lm-format-enforcer` 不支持 arbitrary-object 对话状态，失败
  JSON 与原始两次输出完整保留。修复未改变三类固定业务 Schema 约束解码。
- 2026-08-25：真实四场景生产模型 smoke job `29571134` 在 A100 20 GB MIG 上
  `COMPLETED 0:0`，`00:01:22`；三场景首轮 Schema-valid，对话一次纠错后达到
  `DIALOGUE_BETA`，结果 SHA-256 `a256c64a...8f32`。
- 2026-08-25：Compose 新增幂等且 fail-closed 的 1,000 向量 `retrieval-init`；最终本地
  私有包四层及 12 份 evidence 校验通过。完整 unittest 和全新 checkout 均为
  513/513。OSS 创建、上传和下载复验仍等待即时确认，因此未进入 `stg`。
- 2026-08-25：导师交接口径修正为只需下一位能够接手模型，不要求 Spartan、OSS 或逐周
  全量数据留存。新增本地 handoff verifier，四层归档、adapter、release config、final
  gate、真实 smoke 和 Milvus 基准复验 `PASS`。
- 2026-08-25：安全清理 21 个 ignored 目标，释放 71,735,466,519 字节；删除 Yelp、公开
  基座缓存、中间输出/checkpoint 和迁移目录，仅保留 59.9 MB 唯一交接包、代码文档、
  轻量样例和未交付凭据。新增 Week 8 优化方向候选供导师选择，不写成已确定任务。

## 2026-08-26：Week 8 四方向并行优化

- 在独立 worktree 和 `feature/week8-product-understanding` 上执行；主 `dev` 工作树既有文档
  改动未暂存、覆盖或格式化。
- 本地/Spartan/阿里云审计后确认旧 65K Yelp 图片池不足；从官方 ZIP 重建 business
  150,346、photos 200,100。商品 fresh source 从 3,000 候选中保留 800 个三维历史隔离
  silver 身份，历史 source/group/image 重叠均为 0。
- v4 商品锁 train/dev/test=`400/60/60`，五维隔离 PASS；dev/test 风格非空支持
  `51/58`、设施 `60/60`，价位应 unknown `60/60`。
- Prompt dev job `29632502` 完成，`week8_product_field_check_v1` 以
  `0.815131` 对 `0.766765` 胜出；格式/失败/支持不回退，因此 SFT 未执行。
- 唯一商品 final job `29632815` 完成：综合 `0.804239→0.861085`，业态
  `0.900000→0.933333`，风格 micro-F1 `0.787402→0.796875`，设施 micro-F1
  `0.675159→0.806630`，price unknown `0.05→1.0`，JSON/Schema `1/1`，失败率 `0`。
- 对话 v2 在固定 4 条真实模型样本上首轮合规 `0→0.5`、纠错 `1→0.5`、失败
  `0.75→0.25`、上下文召回 `0.2727→0.8182`；v3 严格 schema 失败率 `1.0`，被拒绝。
- 固定商品延迟 512→384 cap 的 mean `1907.79→1903.28 ms`，质量完全一致但收益仅
  `0.24%`，不宣称实质改善。冷启动 `24.67 s`，峰值 allocated/reserved
  `6.69/8.43 GB`。
- 检索唯一 final job `29628157`：metadata rerank 的 NDCG@10
  `0.125654→0.506740`，Recall@10 `0.018090→0.133046`，失败率 `0`，可追溯率 `1`。
- 完整 unittest `561/561 PASS`。详细证据见 Week 8 报告。

## 2026-08-27：Week 8 全自动扩展优化

- 用户明确后续不再有人工标注、人工复核或人工验收；新增 target、review 与 acceptance
  一律保留 `programmatic_silver` 身份，三类人工计数均为 `0`。
- 从官方 Yelp Photos ZIP 的 6,000 个分层候选重建 fresh source v3：1,291 个通过图片
  哈希/可读性验证，锁定 1,000 个；与历史 source/group/image 重叠均为 `0`。合法可用
  业态上限为餐饮/景点/酒店 `992/7/1`，没有虚构酒店支持。
- 商品 v7 锁 train/development/test=`400/60/60`，五维隔离 PASS，内部 lock SHA-256
  `321bea49...b0301`；test 在候选选择期间保持 `LOCKED_UNCONSUMED`。
- fresh development job `29637779` 在 A100 20 GB MIG 上 `COMPLETED 0:0`：紧凑字段
  Prompt 的商品综合 `0.782941→0.836536`，业态 `0.883333→0.916667`，风格/设施
  micro-F1 `0.714286→0.734375` / `0.709677→0.820809`，JSON/Schema `1/1`、失败率
  `0`、支持不变；selection SHA-256 `35abf1b6...c4fae6`。
- 对话确定性三键契约在 5 条固定样本上将合规率 `0.4→1.0`、纠错率 `0.6→0`、
  失败率 `0.6→0`、状态召回/值准确率/精确率/整状态准确率提高到 `1/1/1/1`；明确的
  预算、天数、城市、偏好与节奏更新不再调用模型，真正模糊更新才进入安全 fallback。
- 600x400 固定真实图片基准中，最终选择的 384 token cap 与 512 cap 输出 5/5 完全一致，
  mean/P95 `5006.81/5028.50→5000.55/5009.02 ms`；图片 cap + processor cache 候选
  mean 反而增加 `2.84 ms`，因此最终 release 不启用这两个未证明有收益的开关。
- Milvus Lite v3 的 development 从 CLIP 中锁定 `hybrid_weighted`；唯一 final 的
  NDCG@10 `0.125654→0.564459`、Recall@10 `0.018090→0.142734`，失败率 `0`、过滤
  正确率和可追溯率均为 `1`，P95 增加约 `2.15 ms`。未回退到离线 NumPy fallback。
- 两阶段 development job `29637921` 的 composite/evidence Schema pass/failure 为
  `0.352974/0.266667/0.733333`。SFT job `29637514` 在首个 10% checkpoint-5 为
  `0.369804/0.316667/0.683333`，结合 hard-slice 正标签支持不足于 step 10 主动停止；
  checkpoint-5 adapter-only 回载通过但被拒绝，正式 checkpoint-87 保持最终选择。
- v7 唯一 final job `29638144` `COMPLETED 0:0`：composite
  `0.819003→0.857729`，设施 micro-F1 `0.695652→0.834286`，price unknown
  `0.033333→1.0`，完整性 `0.749167→0.819722`；业态 `0.966667→0.950000`、风格
  micro-F1 `0.755906→0.753846` 的轻微回退如实保留。JSON/Schema `1/1`、失败率 `0`，
  comparison SHA-256 `5dc83953...f3829`，没有根据 test 继续调参。
- release v7 四场景真实 smoke job `29638236` `COMPLETED 0:0`：商品/售后/行程首轮
  Schema-valid，对话确定性三键路径直接达到 `DIALOGUE_BETA`；证据 SHA-256
  `086133ec...85030`，绑定 release config `9defb3e7...ef749` 和正式 adapter。
- 终态验证：完整 `python -m unittest discover -s tests -v` 为 `594/594 PASS`；远端 v7
  数据锁复验为 `PASS`，唯一 test consumption marker 为 `COMPLETED`。`compileall`、
  `git diff --check`、tracked secret signature scan 和大于 10 MiB 的 tracked file scan
  均为 `PASS`；正式 adapter 文件哈希复算为 `c2fbb5c7...eaa2a`。

## 2026-08-27：Week 8 剩余优化续行

- 新增 development-only Prompt overlay，绑定 v7 development lock 且禁用 final。job
  `29643869` 完成：当前 v7/字段检查 v2/保守证据约束 composite 为
  `0.836536/0.701144/0.703235`；两个新 Prompt 均回退并被拒绝，未读取或重跑 final。
- 未消费 silver/OCR 审计 job `29643962` 完成。45 个 pre-hash 候选经历史及 v7 图片哈希
  排除后剩 8 个；确认可见金额/tier/正价位支持为 `0/0/0`。480 张未使用 v7 图全部为
  restaurant，caption style/facility 支持仅 `0/12`；因此没有启动缺少正支持的新 SFT。
- 商品 prepared-input cache job `29643870` 的 10 次 mean/P95
  `4845.46/4877.90→4868.88/4920.32 ms`，输出/tokens/Schema 一致但延迟回退，候选拒绝且
  release 保持关闭。
- 检索 v5 job `29644063` 使用真实 Milvus Lite、LRU512 和独立 development-only 锁；
  NDCG@10/Recall@10 均保持 `0.584776/0.172498`，P95
  `9.6339→8.4247 ms`（-12.55%）。预计算 `1602.23 ms`，entries `393/512`、eviction `0`、
  measurement hit/miss=`2484/0`；未运行 final，也未接入正式 API/release。
- 续行终态验证：定向 `76/76`、完整 unittest `609/609 PASS`；compileall、三份新增
  Slurm 脚本语法、diff、tracked secret/large-file scan 和 README 新命令帮助均通过。
  release/adapter SHA-256 仍为 `9defb3e7...ef749`/`c2fbb5c7...eaa2a`。

## 2026-08-27：全项目复审与商品优先修复

- 复审确认 v7 development 的 60/60 条商品参考混有商家 metadata，56/60 条业态
  known/unknown 矛盾。新增逐行审计与 SFT 输入防护；不修改冻结标签，也不把此前 silver
  匹配分解释为视觉准确率。固定四图自动定性检查发现无停车场的图仍输出 `parking`。
- 修复商品两阶段模型可见 Schema、失败纠错上下文、负证据映射，以及两种训练内存 backend
  的缓存初始化。修复失败占位 JSON 获分、价位支持写死、样本缺失/重复及非有限选优指标。
- development 身份校验不再打开 test 标签；final 校验完整选择证据与五维身份。已消费
  final 不重新评估、不用于调参。新增离线重计分入口，不重跑模型、不覆盖原输出。
- 修复 VLM/CLIP 并发加载和半初始化、HTTP 图片缓存、对话局部否定/重复/部分更新/数值
  范围，生产示例 fallback 与旧示例 planner 关闭。对话格式/状态指标不等同于推荐质量。
- 本轮最终复验完整 unittest `654/654 PASS`（33.380 s）；新增三组测试模块共 45 条，
  单独运行 `45/45 PASS`。真实模型暴露的取消预算、非法负天数被 fallback 改写问题再次修复。
  人工 annotation/review/acceptance 仍为 `0/0/0`；真实 GPU 证据单独记录在商品报告第 12 节。
- 商品 job `29664584`、解码消融 job `29666004` 均正常完成：完整 development 各角色
  60 条；旧证据链→修复契约失败 `17/60→0/60`，mean `17.631→7.999 s`，但设施银标
  micro F1 `0.661972→0.467836`，且仍有视觉反例，故不替换 v7 RC。基座取消受约束解码
  后失败 `33/60→2/60`；剩余重复事实违反 Schema，不能宣称零失败或视觉能力提升。
- 发现默认 cafe 图片是 64×64 占位图；前两作业对应 smoke/重复延迟只能证明模型连通性。
  新作业 `29666837`（`f58707c`，3 分钟）锁定 533×400 真实 development 图片，三场景
  Schema 和对话契约通过。10 条对话状态 exact `0.4→1.0`、首轮格式 `0.9→1.0`、失败
  `0.1→0`；商品输出上限 512/384 的 mean `3894.280→3901.957 ms`，没有明确提速。
- 两轮商品原始输出只读校验后另存重计分；新增模型请求为 0，原证据不覆盖。正式及 RC
  manifest、adapter 不变，已消费 final 不重跑；真实 smoke 的 Schema PASS 不等于语义
  正确，停车场猜测、行程模板复述及一般对话任务完成度仍是未解决项。

### 2026-08-28：c01b732 审查九项修复与真实复测

- `327f764` 修复标签解析/否定/子串、无效参考选优、对话实际分派、行程业务检查、金额、
  图片轮次、查询侧隔离与生产检索、输入 422、统一 release 配置；补充 25 条审查回归。
- 新诊断身份 `week8_audit_repair_20260827_v1` 保留原 60 条 development；caption-only
  silver 的业态/风格/设施/价位正支持为 3/0/3/0，旧 parking 58→新 caption parking 0。
  这不是视觉提升，三组旧 raw 重计分均不允许锁定 Prompt；最终 test 未读取或重跑。
- 真实 Milvus Lite + 生产路由 5 查询结果 5/5/0/5/5，过滤正确，换城市改变结果；使用
  已有身份绑定 CLIP 向量，不宣称独立图片相关性提升或已部署 HNSW。
- GPU job `29667548` 在 A100 MIG 20GB 完成（2:46）：技术 smoke PASS、业务 FAIL；
  两日行程错天数/占位内容经一次纠错仍失败，已明确返回未完成。商品对话实际调用模型，
  但 parking 猜测仍在。商品 5 次 mean/P50/P95=4686.114/4684.040/4702.216 ms，失败 0/5，
  每次 input/output=713/57，不能与历史整卡结果直接比较。
- 完整 unittest 679 条通过；旧四层交接包、显式配置及 Compose 静态复验通过。CLI
  复测纠正候选 quality 说明字段误当运行契约的问题，未改 manifest 或 adapter。
- Week 8 保持 PARTIAL；无人工工作、无新增训练、无晋级。详细字段支持和证据哈希见商品报告第 13 节。

### 2026-08-28 持续复审与契约消融

- 登记用户持续修复至可晋级候选的要求。修复独立 Prompt 版本中的行程契约冲突，新增
  图片事实约束版本与 development-only 消融脚本；正式 manifest 和冻结结果保持不变。
- 定向测试 4/4、完整 unittest 683/683 通过。真实模型消融尚待运行，不提前宣称质量通过。
- 第一轮消融已完成，确认 adapter 的模板/元数据偏差；base 描述更具体但仍未通过完整
  业务复审。新增活动时间/必去禁去/交通核验与请求天数纠错；新商品观察协议进入独立 pilot。
- 完整 unittest 697/697 通过（31.205 秒）。首次新增教师测试因临时目录缺 Schema 失败，
  已补齐真实 Schema fixture 后全量重跑；不修改原失败日志。独立教师始终为 silver。
- 全 60 条 development 三组模型实测完成（job `29684981`，执行 `7047093`）。独立视觉
  silver 综合分正式/观察 v1/观察 v2 为 0.463794/0.619460/0.745493；v1 风格召回回退
  未选，v2 全字段不回退，仅成为 development 候选。价位支持 0，单列 N/A。
- 将观察协议和逐场景 adapter 开关接入真实服务，售后继续使用原 adapter；补充日期和
  等义约束检查，加入独立生产 smoke 与三种缓存重复基准。尚未执行新最终 test。
- 本轮真实探针发现对话行程仍用“某文化空间”充当地点，原探针 PASS 不作为完整验收。
  新增具体地点检查及行程 Prompt v3；保留所有旧输出，继续在相同固定请求上复测。
- 新最终集构建器按五维身份排除 Week 3/5/6/7、system repair、所有 Week 8 商品锁、
  hard-slice 训练及五版检索身份；检索裸 business_id 与商品命名空间统一后排除。
  最终标签尚未生成；执行器要求先通过 development 与业务复验再锁定，每个角色只运行一次。
- CPU job `29689536` 排除后剩余 428 张，封存 100 张无标签图片；复查发现无模板照片被
  赋予实验模板名，保留 v1 产物但不消费，v2 用同 seed/来源顺序保留真实 null 模板。
- 探针 v2 的北京请求仍漏城市检查；还发现未检索却生成线路和 source_evidence。新增
  对应业务错误、结构化输入规则和行程 v4，保持一次纠错，不自动伪造通过检查。
- 修复 runtime 包缺少 planning、data 辅助模块、scenarios 及商品/检索配置的问题；
  隔离解包导入与故意缺依赖的反例均通过。完整 unittest 729/729 通过（23.130 秒）。
