# Project Requirements

## Week 1: OTA Multimodal VLM Engineering Foundation

### Goals

- Establish the Python, Docker, FastAPI, and vLLM project foundation.
- Support deterministic fallback tests without requiring a live GPU service.
- Verify a live single-image multimodal request on the local GPU.
- Prepare a small Yelp OTA sample and reproducible experiment records.

### Acceptance Criteria

- API health and image-understanding routes return structured responses.
- Docker serves the selected VLM through an OpenAI-compatible endpoint.
- Sample POIs, reviews, and images are available without committing raw Yelp data.
- Unit tests and experiment logs record the verified Week 1 state.

## Week 2: Yelp Multimodal Dataset Processing Pipeline

### Background
Week 1 established the project scaffold and small Yelp subset workflow. Week 2 builds a reproducible pipeline for parsing, validating, aligning, and reporting Yelp multimodal data.

### Goals
- Normalize Yelp raw, interim, processed, logs, and report directories.
- Consume the Week 1 downloaded/extracted Yelp files from the normalized raw directory.
- Read business, review, and photo JSONL files line by line for the full processing pipeline.
- Validate local photo files and record image metadata.
- Build strong valid-image/non-empty-caption pairs, medium image-business pairs, and weak image-review pairs.
- Provide a reproducible CLIP denoising stage that runs in a dedicated GPU Docker task without adding torch or vLLM to the base data environment.
- Generate `reports/yelp_multimodal_data_processing_report_part1.md`.

### Non-goals
- Do not commit raw Yelp archives, extracted images, or large generated Parquet files.
- Do not require GPU or CLIP for the base pipeline.
- Do not build model training or a UI this week.

### Deliverables
- Config-driven parsing and alignment scripts.
- Reusable `src/data/` modules for archive extraction, JSONL parsing, validation, alignment, statistics, templates, and optional denoising.
- Interim Parquet outputs, processed alignment outputs, statistics JSON, and report draft for the full local Yelp run.
- Tests for parsers, validation, alignment, denoising skip behavior, and report generation.

### Acceptance Criteria
- `python -m unittest discover -s tests -v` passes.
- Week 2 data processing can be installed with `pip install -r requirements-data.txt` without installing `vllm`.
- Parsing script writes business, review, photo, image-index, review-stats, and validation-summary outputs.
- Alignment script writes strong, medium, weak, and dataset-statistics outputs.
- Denoising task writes a row-level denoised table, a summary, and similarity distribution; disabled/dependency-unavailable paths still write an explicit skipped summary.
- Report generator writes the Week 2 report using real statistics or explicit `TODO` markers.

### Risks / Questions
- Official Yelp archives may need manual extraction before parsing unless archive extraction is added.
- The Yelp download/extraction step is treated as a completed prerequisite for this Week 2 processing review.
- Full review/photo data is large, so review output uses chunked table writing to keep memory bounded.
- CLIP requires the `clip-denoising` Docker profile and exclusive GPU access. Stop vLLM before full CLIP inference on the local 8GB GPU.
- Local environments without `pyarrow` will run with a CSV fallback at the configured output path until dependencies are installed.
- The default config sets `processing_limits.max_reviews` to `null` and uses chunked review writes for full-dataset parsing.
- vLLM should not be installed in native Windows Python by default; use Docker or WSL2 for live LLM serving dependencies.

## Week 3: Zero-Shot Business Baselines and Standardized Prompts

### Background

Week 3 measures the unoptimized multimodal model on three OTA business scenarios before prompt optimization or model fine-tuning. The delivery must establish reproducible evaluation data contracts, objective metrics, raw-output traceability, standardized prompts, and backend-compatible JSON Schemas. The report is an engineering and business decision record, not a paper; use concise Markdown tables and evidence.

### Goals

- Define independent evaluation sets for image-to-product search, intelligent after-sales understanding, and multimodal itinerary planning.
- Enforce separation between evaluation samples and any future training candidates.
- Define reproducible scenario metrics, normalization rules, invalid-output handling, and batch scoring.
- Run the simplest possible zero-shot baseline without a role, JSON constraint, chain-of-thought instruction, few-shot example, or prompt optimization.
- Preserve every input, raw model output, latency, parse result, validation result, model version, and prompt version.
- Define a four-layer standardized prompt architecture and one JSON Schema per scenario.
- Report measured weaknesses and evidence-backed prompt or fine-tuning priorities without inventing results.

### Non-goals

- Do not build a training or fine-tuning pipeline.
- Do not implement embedding retrieval, a UI, or an unrequested future-week feature.
- Do not auto-generate completed manual annotations or use model-generated labels as ground truth.
- Do not add a second annotator, adjudication, or generated gold. The mentor's latest clarification permits the existing annotator to label only the separately versioned v2 replacement rows needed after low-quality candidates are removed; evidence-supported product `unknown` labels are reused without forced completion.
- Do not require a full live-model run when validated data or compute is unavailable; a verified framework, dry-run, and honest partial evaluation are acceptable interim states.
- Do not expose or request private chain-of-thought. Standard prompts may request concise observable evidence and field-level checks only.

### Evaluation Set Targets and Status Counts

Target sizes describe the required evaluation design, not the current completion state:

| Scenario | Target | Required coverage |
| --- | ---: | --- |
| Image-to-product search | 200 images | Hotels, attractions, and restaurants |
| Intelligent after-sales | 150 evidence images | Hygiene stains, facility damage, attraction closure, and transport delay |
| Multimodal itinerary planning | 100 paired samples | Reference style image(s), text constraints, parsed requirements, and itinerary constraints |

Every scenario summary must report these counts separately:

- `target_count`: mentor-required target size.
- `candidate_count`: collected candidates before human annotation.
- `annotated_count`: candidates with completed human annotations.
- `validated_count`: annotations and files that pass validation.
- `tested_count`: validated samples with a persisted inference record for the selected run.

No report may present a target as completed work. Only validated samples may enter inference, and `tested_count` must come from actual run records.

### Annotation and Manifest Requirements

All manifests use versioned JSONL. Each record must include `sample_id`, `scenario`, `source_type`, `source_id`, `source_license`, `image_sha256` where applicable, `split`, `dataset_version`, `annotation_status`, `annotator`, legacy-compatible `review_status`, `annotation`, and `notes`. Paths are repository-relative. Missing values use JSON `null`; unknown semantic values use documented enums rather than fabricated labels. The legacy review fields do not control release eligibility.

Image-to-product annotations contain business category, style tags, visible core facilities, and price range. After-sales annotations contain issue type, severity, key information, and OCR ground truth when applicable. Itinerary annotations contain reference images, text constraints, style preferences, hard constraints, soft constraints, and required itinerary elements.

Candidate generation may use deterministic stratified sampling with a recorded seed, criteria, and per-stratum counts. Candidate generation must set annotations to pending; only records submitted by a human annotator may become completed. Public or synthetic after-sales sources must record provenance and usage rights.

Week 3 uses one human ground-truth annotator. The mentor does not require annotator B, dual independent annotation, a second-review gate, or adjudication. `annotation_status=completed` means the annotator inspected the image and supplied constraints and submitted the complete annotation. Model outputs and deterministic suggestions may assist inspection but must not directly create completed gold labels.

The v1 annotations, manifests, Prompts, Schemas, and runs remain immutable. The mentor's 2026-07-22 clarification supersedes the frozen-v1-only route and authorizes a separately named `week3_evaluation_v2` after low-quality-image removal. Product v1 labels are reused unchanged. Seventy replacement after-sales candidates require human submission after representative images are selected. All 100 itinerary rows require only a `style_preferences` supplement because the historical tool did not reliably expose or serialize that field; the prior text, hard/soft constraints, and required elements must be inherited unchanged. Until these v2 supplements are complete, v2 remains `PARTIAL` and no full live run may claim completion.

As of 2026-07-25, the approved v2 supplements, full baseline, full standardized run, and scoring are complete and traceable. The user approved the independent `baseline_semantic_coding_v1` deterministic lexical track for the minimal baseline. Its prediction stage accepts only `scenario`, `raw_output`, a fixed versioned codebook, and general normalization; human gold is joined only after prediction for scoring. This completes the previously unsupported baseline semantic metrics without changing the baseline Prompt, raw run, manifests, annotations, or strict JSON/Schema results.

V2 must preserve evaluation/training isolation, input hashes, source provenance, the public/synthetic after-sales source mix, and immutable v1 history. Candidate strata guide selection but do not automatically become human labels. Reasonable `unknown` remains valid when the image cannot support a field.

Eligibility to run does not prove complete mentor coverage. Report sampling-stratum coverage and human-gold distributions separately. When the frozen gold labels do not substantiate a required category or semantic field, the run may proceed for the supported metrics, but the affected metric and final delivery remain `PENDING` or `PARTIAL`; do not reinterpret sampling metadata as gold labels or claim `READY`.

Validation stays limited to what the mentor delivery needs: readable inputs, required JSON fields, scenario coverage, annotation completeness, and evaluation/training isolation. Use the documented unknown value when evidence is insufficient instead of guessing; report unknown and empty-field distributions as data limitations rather than adding an unrequested rejection system. Do not invent exact human-gold class quotas beyond the mentor's target sizes and required category coverage.

The intelligent after-sales set includes both public-scene evidence and business-synthetic evidence. The mentor has not specified a numeric source-type ratio. Low-score reviews may help candidate discovery, but the final selected images must remain relevant to the four required problem categories.

### Evaluation Isolation

The evaluation registry records stable `source_id` and `image_sha256` identifiers and generates `data/eval/registry/evaluation_exclusion_manifest.jsonl`. A reusable validator must reject any future training candidate whose source ID or image hash appears in the exclusion manifest. Week 3 does not need a training pipeline, but it must provide the exclusion interface and deterministic conflict tests. A registry without enforced collision checking is insufficient.

### Zero-Shot Baseline and Run Records

Create one versioned minimal baseline instruction per scenario. Each instruction states only the recognition task and must not contain a role definition, required fields, JSON or formatting constraint, chain-of-thought instruction, or example. Keep baseline prompts physically and logically separate from standardized prompts, and never overwrite an earlier run.

Each inference record must include `run_id`, `sample_id`, `scenario`, `model_name`, `model_config`, `prompt_version`, normalized input metadata, `raw_output`, `parsed_output`, `json_valid`, `schema_valid`, `latency_ms`, `error`, and `timestamp`. The batch runner reads the current verified model and serving settings from configuration and documentation; it must not hard-code or download a model merely because a prompt mentions one.

### Metrics and Scoring Rules

- Image-to-product: business-category and price-range accuracy; style and facility precision, recall, and F1; label completeness; JSON compliance; schema pass rate.
- Intelligent after-sales: issue and severity accuracy; key-information precision, recall, and F1; OCR field recall and exact match; JSON compliance; schema pass rate.
- Multimodal itinerary planning: constraint-recognition accuracy; hard- and soft-constraint recall; itinerary-element completeness; constraint-violation rate; JSON compliance; schema pass rate.

Metric specifications define the minimum normalization needed for reproducibility, including missing fields, synonyms, and multi-label scoring. Format compliance is scored separately from semantic task quality. Because the required minimal baseline has no JSON constraint, unparseable natural-language output must not automatically turn every semantic metric into zero; use a simple documented coding or extraction rule, or mark the unsupported semantic metric `PENDING`. Persist sample-level scores, scenario aggregates, latency, and representative errors.

### Standardized Prompt Architecture

All standardized scenario prompts use four layers: system role, task instruction, input context, and output constraint. The common role identifies a professional OTA travel-platform assistant and requires Chinese output, domain relevance, explicit unknown values, separation of observation from inference, no fabrication, privacy protection, and route/travel safety.

Scenario prompts cover structured product labels; after-sales anomaly localization, classification, severity, key fields, and OCR; and itinerary preference parsing, hard/soft constraints, itinerary elements, and constraint checking. They may instruct the model to check the task in the mentor's stated order, but the returned content is only the required JSON rather than a narrated private reasoning process.

### Deliverables

- Week 3 evaluation configuration, annotation specification, manifest contracts, and count definitions.
- Three minimal baseline prompts, common standardized prompt layers, and three standardized scenario prompts.
- Three JSON Schema files and prompt-rendering/schema-validation tests.
- A batch baseline runner that preserves inputs, raw outputs, latency, model identity, and errors.
- Scenario metrics, count aggregation, exclusion validation, and error-case export.
- A concise zero-shot baseline report and prompt architecture specification.
- Updated README, weekly records, experiment notes, and verification evidence.

### Acceptance Criteria

- All three evaluation formats and the five status counts are defined and validated.
- Evaluation samples produce an exclusion manifest; source-ID and image-hash collisions are rejected by tests.
- Baseline and standardized prompts are separately stored and versioned for all scenarios.
- Three JSON Schemas validate representative valid and invalid fixtures.
- Raw output, latency, model configuration, parse status, and schema status are persisted.
- Scenario metrics and baseline natural-language handling are documented and reproducible.
- Reports display target, candidate, annotated, validated, and tested counts separately.
- Any live or partial result is traceable to persisted output; unavailable results are marked `PENDING` rather than fabricated.
- Relevant unit tests and task-specific validators pass before delivery.

### Delivery Status Rules and Risks

- `READY`: required tests pass, the mentor-required real baseline and Prompt/Schema deliverables exist, reports match outputs, and no blocking evaluation/training leakage or data issue remains.
- `PARTIAL`: framework and tests pass, but real data, manual annotation, or full evaluation is incomplete; all missing work and five counts are explicit.
- `NOT READY`: a core flow fails, results are untraceable, evaluation leakage is possible, baseline and standardized prompts are mixed, or Git contains prohibited data.

Local GPU capacity may limit full inference, and public after-sales evidence may require licensing review. Existing annotation gaps may limit semantic metric coverage, but they do not create a new Week 3 labeling task. Any cloud or third-party runtime must be approved and documented before use. Phase completion requires a Project Control checkpoint; code defects found by Review return to Execution rather than being silently repaired during reporting.

## Week 4: Prompt Optimization and Milvus Vector Storage

### Goals

- Build scenario-specific few-shot examples from the completed Week 3 v2 gold data without adding a new annotation task or dataset version.
- Compare the existing `standardized_v2` zero-shot prompt with 4-shot and 7-shot variants, then run the selected prompt for each scenario on the full Week 3 v2 evaluation set.
- Measure business metrics, JSON and Schema compliance, token use, and inference latency against the immutable Week 3 baseline.
- Export evidence-based bad cases and provide a bounded JSON parsing and Schema-validation fallback.
- Deploy Milvus standalone with Docker Compose and design the `ota_business_image_vector` collection for filtered OTA image retrieval.
- Provide tested insert, search, delete, and index-management operations plus a small, reproducible performance baseline.

### Prompt Optimization Requirements

- Keep all Week 3 manifests, annotations, prompts, Schemas, runs, and score artifacts immutable.
- Continue inference with the repository's verified `Qwen/Qwen2-VL-2B-Instruct` vLLM configuration and existing generation settings. Do not switch to or download Qwen3-VL for this work.
- Select five representative positive examples and two boundary or negative examples per scenario from existing Week 3 v2 gold. Use three positives plus one boundary example for 4-shot and all seven examples for 7-shot.
- Compare `standardized_v2`, 4-shot, and 7-shot on a fixed pilot. Select each scenario's best tested variant using business quality, JSON and Schema compliance, token use, and latency, then run only that winner on the full v2 evaluation set.
- Prompt instructions may use concise field checks, `observed_evidence`, and `constraint_check`, but must not request or persist long-form private reasoning.
- Use a documented, reproducible metric interpretation for baseline and optimized outputs. Preserve the existing Week 3 scores and store Week 4 runs and comparisons separately.
- The selected prompt is best only among the tested candidates. Reports must not claim global optimality.

2026-07-26 审查澄清：上述 Few-Shot 示例来自最终测试集金标，因此现有
pilot 只能作为描述性运行证据，不能用于无偏泛化效果声明。当前范围仍禁止
新增人工标注或数据集版本，所以不临时构造新的 demo/dev pool。最终胜出的
`standardized_v2` 不使用示例，其 450 条全量结果不受该污染直接影响。
baseline 与 winner 的业务差值必须另存为共同确定性语义轨道：两组原始输出
使用同一编码器、同一 codebook、同一指标函数和相同 `sample_id` 成对评分；
不得覆盖 Week 3 原评分。

2026-07-26 用户后续直接授权完成全部未完成项，包括新建独立 demo/dev pool
及其人工标注，因此该授权只在此项上取代上一段“禁止新增”的临时限制。
独立池固定命名为 `week4_demo_dev_v1`、split 为 `development`，不得复用
最终 evaluation 金标；必须与 `week3_evaluation_v2` 在样本、来源、图片
哈希和来源组上隔离。每场景人工完成 12 条，从中选 5 个正例和 2 个边界例，
固定 evaluation pilot 不变。旧 test-gold Few-Shot 运行保留为历史证据，
新比较、全量 winner 和报告必须另行版本化。

The format fallback may remove an optional Markdown code fence, parse JSON, and validate the existing scenario Schema. It must preserve the raw output and return explicit errors. It must not invent fields, change enum values, infer missing labels, or call the model again.

### Milvus Requirements

- Use a pinned stable Milvus standalone Docker Compose deployment with etcd, MinIO, health checks, port mappings, resource bounds, and persistent storage.
- Keep PyMilvus in a separate dependency file so the API, data, and vLLM dependency groups remain unchanged.
- Define `ota_business_image_vector` with an auto-generated `vector_id`, `business_id`, `image_id`, `multimodal_vector`, `business_category`, `city`, `star_rating`, `price_range`, `image_type`, and `embedding_model`.
- The current Qwen2-VL vLLM endpoint remains the business inference model and is not treated as an embedding endpoint. Reuse the verified `openai/clip-vit-base-patch32` encoder and its actual 512-dimensional normalized vectors for the prototype.
- Use `COSINE` distance with a configurable HNSW index. Keep `M`, `efConstruction`, and query `ef` in configuration, and add scalar indexes for filtered fields.
- Provide five core operations: batch insert, single insert, vector search with allow-listed scalar filters, deletion, and index construction.
- Validate the deployment with a bounded set of real OTA image vectors that can run on the current machine. Record actual vector count, index build time, mean and P95 search latency, Recall@K, parameters, and environment. No production threshold or capability may be claimed unless it was specified and measured.
- Do not run local vLLM and CLIP GPU inference concurrently on the current 8 GB GPU.

### Deliverables

- Three selected scenario prompt templates and their versioned few-shot examples.
- Prompt optimization comparison report and bad-case analysis.
- JSON parsing and Schema-validation fallback script.
- Milvus Docker Compose assets and a multimodal collection design document.
- Vector operation SDK, unit tests, and real connectivity and CRUD evidence.
- Milvus deployment and performance validation report.
- Accurate README, weekly delivery, weekly log, and experiment updates.

### Non-Goals and Delivery Rules

- Do not add manual annotation, training, fine-tuning, a Web UI, new API routes, complex hybrid retrieval, cloud deployment, production monitoring, or future-week plans.
- Do not commit raw data, model files, generated vectors, run outputs, Milvus volumes, secrets, personal configuration, or Chat prompts.
- Execution proceeds end to end without phase approval gates unless a rule conflict, immutable-artifact overwrite, destructive operation, or new annotation requirement is discovered.
- Review and Report performs one consolidated review. It may correct evidence documents after its read-only review, but code defects are reported rather than silently refactored.

## Week 5：数据集标注与质检

### A. 三大业务场景指令数据集的标注与质检

#### 1. 标注规范与标注方案

- 分别制定以图搜商品、智能售后、多模态行程规划的标注规范，明确输入构成、
  输出字段、取值范围、格式标准和质量验收规则，统一全量标注口径。
- 设计各场景标准 JSON Schema，与前期 Prompt 工程输出格式保持一致，保证训练
  与推理结构统一。
- 标注流水线采用“模型预标注 + 人工修正 + 分层质检”，明确各环节责任与验收
  标准。
- 配置多模态标注模板，导入原始样本池并设置字段自动校验规则。

#### 2. 分场景样本池与预标注

- 从第 1 周 OTA 数据中分层抽样，与独立测试集严格隔离，避免数据泄漏。
- 以图搜商品：构建 5 万张高质量商家实景图样本池，覆盖酒店、餐饮、景点，
  兼顾风格、价位和城市分布。
- 智能售后：构建 2 万条公开场景凭证与业务合成样本，覆盖卫生污渍、设施损坏、
  景点关闭、交通延误，均衡不同严重等级。
- 多模态行程规划：构建 1 万组“风格参考图 + 出行约束”，覆盖不同出行人群、
  预算档位和行程天数。
- 复用前期沉淀的最优 Prompt，对全量样本批量预标注并生成初始结果。

#### 3. 全量人工标注修正

- 人工校验预标注准确性，补充遗漏字段、修正错误分类并统一标签颗粒度。
- 以图搜商品重点校准风格、设施、价位区间和业态类目。
- 智能售后重点校准严重等级、关键信息完整性和 OCR 标注。
- 多模态行程规划重点校验约束命中完整性、行程结构合理性和要素完整性。
- 将第 2-3 周梳理的 bad case 作为重点难例纳入标注范围。

#### 4. 多级质检与数据清洗

- 执行单人三级质检：人工修正保存时显式完成自审；确定性选中样本由同一标注人
  在不同 `review_session_id` 中进行同场景盲二次复核；其子集再执行核心样本抽检。
  不得伪造第二位审核人或把自动校验记为人工质检。
- 为最大程度降低唯一标注人的重复劳动并将全池额外质检控制在 500 次以下，商品
  交叉复核/核心抽检比例为 0.2%/0.05%，售后和行程为 0.5%/0.1%。选择使用
  `sample_id` SHA-256 固定，核心抽检是交叉复核的嵌套子集。未抽中样本在真实人工
  修正和内联自审通过后可 accepted。
- 剔除或修正标注错误、字段缺失、格式非法、图文不匹配和歧义样本，统一同类
  表述口径。
- 统计各场景标注合格率和问题分布，输出质检报告；不达标批次执行返工。
- 最终达标样本：以图搜商品不少于 4.5 万条，智能售后不少于 1.8 万条，
  行程规划不少于 0.9 万条。

### B. 多轮多模态对话数据集构建

#### 1. 对话场景与交互范式

- 三类场景为以图搜咨询、行程规划迭代、售后协商，典型对话为 3-8 轮。
- 对话结构包含对话 ID、场景类型、图片资源列表、多轮消息及其角色、内容和
  图片引用，并规范图片引用和指代规则。
- 对话应逻辑连贯、符合真实用户表达、上下文指代清晰、回复符合业务规范，
  禁止编造信息和违规承诺。

#### 2. 多轮对话样本生成

- 基于单轮指令样本，通过大模型辅助生成和人工校验扩展多轮对话。
- 覆盖首轮上传图片、轮中补充图片或条件、追问历史结果、修改约束重新生成、
  指代历史图片等交互模式。
- 构建不少于 1 万条多轮对话样本，三类场景分布均衡，平均轮次不少于 4 轮。

#### 3. 对话质检与归一化

- 校验逻辑连贯性、上下文一致性、图片指代正确性和业务合规性，剔除逻辑断裂、
  前后矛盾和不符合业务规则的样本。
- 助手回复统一为专业、规范、友好的 OTA 客服语气；用户表达保持真实口语化。
- 清洗过长、过短及无效对话，最终达标多轮对话样本不少于 0.9 万条。

### 交付要求

- 三场景标注规范、对应 JSON Schema 和标注工具配置。
- 三场景样本池、预标注结果、人工修正结果和三级质检结果。
- 多轮对话结构规范、生成结果和质检结果。
- 各场景数据统计与质检报告。

所有完成数量、人工标注、交叉互审、抽检和合格率必须来自实际结果，不得把目标
数量或模型预标注数量写成已完成人工标注或质检数量。数据集不得与独立测试集重叠。

### 2026-08-09 Project Control 执行澄清

- Qwen3-VL-4B 商品/售后 Prompt 固定为 `standardized_v2`/`fewshot_4_v2`；
  行程只允许在同一组最多 30 条候选上配对比较 `fewshot_4_v2` 与
  `standardized_v4`，不得扩展候选或用全量结果反向选 Prompt。没有有效结论时
  使用 `fewshot_4_v2`。
- 现有商品/售后/行程 50,000/20,000/10,000 候选保持不可覆盖。候选池与冻结
  评测集必须在 `sample_id/source_id/image_sha256/group_id/constraint_template_id`
  五维隔离；训练候选场景之间允许共享来源业务组，但不得共享 sample、source 或
  图片哈希。
- 不原地迁移候选中的旧 `pending`。workflow v2 使用 sidecar 绑定候选哈希，
  `model_preannotation.status` 与人工 `workflow_status` 分离。无人工修正时固定为
  `awaiting_human_annotation`；其余允许值为 `partial`、
  `awaiting_cross_review`、`awaiting_core_audit`、`accepted`、`rejected`。
- 多轮对话采用版本化 v2 Schema，必须包含 `schema_version/dialogue_id/scenario/
  image_resources/turns/source_sample_ids/generation/human_review/qc`。候选的
  `human_review` 为 `awaiting_human_annotation`，`qc` 为 `partial`；v1 保留不变。
- 全量预标注的前置条件为不可覆盖 run ID、独立目录、配置/候选哈希、输入和请求
  哈希、独立 raw 输出、逐次尝试和 retry、分片/checkpoint、独立 failures，以及
  元数据完全一致的显式 resume。
- 当前只授权受限行程 pilot：最多 30 个唯一样本、两个 Prompt、60 次总请求、
  1.0 GPU 小时、CNY 20；首 5 组后请求或基础设施失败率超过 20% 即停止。
  未授权 `preannotate-all`、80,000 条全量预标注、批量对话生成或任何训练。

### 2026-08-12 计算迁移与验收补充

- 用户后续直接授权的全量预标注继续有效。Week 5 模型固定为
  `Qwen/Qwen3-VL-4B-Instruct`，商品/售后/行程继续使用已冻结的
  `standardized_v2`/`fewshot_4_v2`/`standardized_v4` 映射，不得在同一全量
  数据中改用 8B 或反向重选 Prompt。
- 阿里云 A10 因欠费停止后，剩余预标注迁移到 Spartan。迁移必须使用独立版本、
  确定性互斥分片、不可覆盖 run ID、原始输出、checkpoint、失败记录和严格合并；
  A10 历史 run 保持只读。
- 模型预标注仍不是人工金标。单人实际人工处理预算控制在 3 小时以内、全部操作
  低于 500 次；降低人工预算只能减少 accepted 数量，不得把未人工确认的 silver
  数据改写为人工完成。
- 2026-08-14 用户进一步确认采用可完成的抽样验收：三场景各固定选择 100 条人工
  验证样本，并在该队列内固定包含每场景 10 条盲二次复核候选和 3 条核心抽检候选。
  已完成的 27 条真实修订计入三场景各 100 条目标，因此计划总操作为
  `300 + 30 + 9 = 339`。多轮对话先自动生成 10,000 条候选，再固定抽取 100 条由
  本人验收，因此总人工操作上限为 439，满足 3 小时与低于 500 次边界。未进入人工队列的成功
  预标注保持 `silver`，不得计为 `human_revised`、`accepted` 或人工合格数据。
- 原始 4.5 万/1.8 万/0.9 万单轮合格目标和 0.9 万人工合格对话目标，在当前单人
  预算下不作为人工完成声明；工程输出、silver 候选、人工抽样合格和对话候选/人工
  合格必须分别报告。自动生成的多轮对话可以作为候选，但未由本人检查的记录保持
  `awaiting_human_validation`；仅抽样验收通过的对话可计为人工 accepted，不得美化
  未抽样对话。
- Week 5 工程验收要求 80,000 条均具有成功预标注或明确未解决失败记录；人工
  `human_revised/self_reviewed/cross_reviewed/core_audited/accepted` 和对话 accepted
  继续按真实输入单独报告。
- Spartan 项目文件、缓存、日志、输出和虚拟环境只能写入
  `/data/gpfs/projects/punim2936/Trip_Project_yzhang3504/20260812a`；不得使用 home。
  Python 3.11 虚拟环境必须由 Slurm 作业创建并完成依赖校验，Week 5 Apptainer
  推理环境与 Week 6 宿主 venv 保持分离。
- 100 条 L40S benchmark 通过后，直接提交四个互斥分片的唯一 array 作业并持续监控，
  不再设置额外人工审批。出现明确可修复故障时自动修复、验证并只恢复失败或未完成
  分片；禁止盲目重试、成功样本重复请求、跨 run 双写或多分区竞争提交。
- 全量自动执行的完成条件是 80,000 个唯一候选均有成功结果或经安全恢复后仍明确保留的
  最终失败记录，并通过 merge、去重、隔离、配置/候选哈希和 JSONL 完整性验证。
- 2026-08-16 用户直接要求保留活动对话作业不变，同时对 10,000 条多轮对话使用
  独立运行目录的确定性互斥分片申请额外 GPU。分片必须绑定相同配置与 qualified
  sample 集合哈希，按 index 范围和 modulo 互斥，每个作业写独立 JSONL；最终只能
  通过显式 merge、Schema、图片引用、唯一 ID 和 10,000 条完整覆盖校验形成权威候选。
  该授权仅放宽多轮对话的“唯一活动作业”限制，不允许分片共同写文件或执行 Week 6。
- 2026-08-16 Week 5 最终完成事实：权威合并 run
  `week5_dialogues_merged_10000_20260816_522b4af` 含 10,000 个唯一候选，索引
  0–9999 完整，三场景分布 3334/3333/3333，消息数 8–12，角色交替、图片引用、
  配置与 qualified 集合哈希均通过验证，duplicate/conflict/missing 均为 0。固定
  100 条人工验收队列已由本人全部完成，五项 checks 完整且 100 条 decision 均为
  `pass`。只允许这 100 条计为人工 accepted；其余 9,900 条仍为未人工验收候选。
  至此 Week 5 已按单人预算内验收口径闭环，不表示执行或完成任何 Week 6 训练。

## Week 6：单场景 QLoRA 小样本链路与专项训练

### 当前授权边界

- Week 6 工程框架可与 Week 5 剩余计算并行建设；正式训练只有在 Week 5 数据版本、
  训练/验证切分和 manifest/split SHA-256 锁定后才能开始。
- 主基座为 `Qwen/Qwen3-VL-8B-Instruct`。售后和行程优先 8B；商品保留 4B 对照，
  先执行 8B 小样本链路验证，不因模型变大而重做 Week 5 全量预标注。
- 使用 transformers、PEFT、bitsandbytes；NF4、double quantization、bf16、基座
  参数冻结和 gradient checkpointing。LoRA 固定 `r=16`、`alpha=32`、
  `dropout=0.05`、`bias=none`，覆盖语言注意力投影和实际模型存在的视觉投影模块。
- adapter 独立保存，不合并基座。AdamW、cosine scheduler、warmup 0.03、weight
  decay 0.01；单 GPU batch 1、梯度累积 16，等效全局 batch 16。
- 模型预标注训练样本必须显式标记为 silver，权重不超过 0.5；人工修订样本可使用
  1.0 权重。任何自动校验、8B 二次输出或 Agent 行为都不能生成真人身份或金标状态。
- 冻结 Week 3 独立人工评测集只在训练参数锁定后进行最终效果评估，不参与训练、
  validation、early stopping 或反复调参。

### 验收

- 依赖版本、CUDA、bf16、4bit 量化、LoRA 目标层、反向传播、adapter 保存和断点
  checkpoint 在 Spartan 小样本试跑中真实通过后，才能标记链路完成。
- 正式三场景训练、最优 checkpoint 和效果评测必须按实际 Slurm 运行产物报告；
  只有配置和脚本时状态为 `READY FOR PILOT`，不是训练完成。

### 2026-08-19 终态后质量改进授权

- Week 6 已完成的 adapter、450 条冻结评测和归档保持不可覆盖；`week3_evaluation_v2`
  已被消费为终态测试，不得再用于方法选择、超参数选择、early stopping 或数据筛选。
- 后续提升先从未进入冻结集的来源建立新的 development/test 锁，按五维隔离、身份、
  支持数和 SHA-256 进行版本化；没有新锁时只能完成研究和工程准备，不能声称指标上升。
- 优先级固定为：错误切片数据质量与目标对齐、Schema 约束解码实验、场景平衡 SFT；
  只有可验证偏好对通过审计后才允许一次多模态 DPO 消融。不得默认换大模型、增加 epoch
  或使用高成本 RL。
- 详细证据与论文映射见 `reports/week6_post_training_improvement_review.md`；任何新运行
  必须在 `docs/experiments.md` 记录实际 commit、数据锁、命令、指标、失败和下一动作。

## Week 7：多任务混合微调与上下文搭建

### 2026-08-24 门禁与分支修复授权（v4 fix2）

- 用户要求修复 fix1 长期无法过门禁的根因，而不是降低阈值或回溯改写 fix1。fix1 的
  config、数据锁、raw、checkpoint、selector FAIL 和 test 未消费状态保持不可变。
- 新身份必须把 v3、首版 v4 和 fix1 的完整 identity manifest 都作为排除来源，建立
  fresh train/development/test；旧 development 只能用于定位评分缺陷，不得用于 fix2
  阈值选择、checkpoint 选择或 test 决策。
- 对话结构化值按叶子字段计分，避免单个嵌套字段错误把整个对象计 0。主观或自由文本
  caption 保持 programmatic silver，只作非门禁证据；不得把逐字 caption 匹配冒充
  视觉语义金标。逐轮协议完整性与逐轮语义准确率必须分开报告。
- 训练 early stopping 与最终 selector 必须使用同一 hard-gate-first 方向；任何通过
  全门禁的 checkpoint 必须优先于未通过候选，再按原加权综合分及最早 step 裁决。
  阈值不依据 fix1 结果下调。
- ADR-004 的长期分支只有 `dev`、`stg`、`main`；执行分支只是临时工作分支。修复代码、
  锁和真实证据进入 `dev` 后，应删除已合并的 `codex/*` 本地及远端分支。本次仍禁止
  进入 `stg`、打标签或进入 Week 8。

### 2026-08-22 用户直接修正授权（v4）

- 用户最新指令取代原 Week 7 对“v3 test 不可重开”和“必须继续人工抽样”的
  限制。v3 的 train/development/test 对话轮次构造有助手回复早于对应用户要求的
  系统性缺陷，因此 v3 数据、训练、原始输出和一次性 test 仅作不可改写的
  历史证据，不得用于声称已正确验证多轮能力。
- 新建 `week7_corrected_multitask_context_20260822_v4`，在保持核心场景分区边界和随机种子的
  前提下，替换 train/development/test 全部对话构造：每个用户轮必须先于回答，
  图片仅出现在首个用户轮，支持 5–8 个用户轮，最终结构化目标必须与当前
  上下文状态一致。v4 必须重建全数据锁、从原底座重训统一 adapter，不从 v3
  checkpoint 续训。
- v4 development 以确定性机器指标和对抗切片为主门禁；指标包含 JSON 格式、
  上下文键覆盖、上下文值准确率、任务结果键覆盖、上下文召回和失败率。
  已有 corrected development 24 条真人四维结果仅作辅助的历史描述；本轮不再
  等待新的人工输入，Agent 或机器指标不得改写为真人评分、人工验收或
  统计显著结论。
- 只有通过预注册自动门禁的 development checkpoint 才可锁定。锁定后允许对
  v4 corrected-dialogue test 执行一次新的自动评测，并同时对比锁定的 Week 6
  routed adapters 和零样本底座。开始消费 test 后无论作业成功或门禁失败都
  不得换参数重跑。
- 已完成的唯一一次 mDPO-style 消融因 validation 门禁失败而拒绝新 adapter；
  v4 不重试 DPO，不以自标注或机器自举冒充新的高质量偏好对。

### Week 6 交接与执行边界

- Week 6 已在提交 `132779b0f6d2929ce1cdbed18e62adf3ef9edd18` 完成交付并进入 `stg`。其训练数据锁、三个单场景 adapter、checkpoint、原始输出、冻结评测、报告和归档均作为历史终态保持不可变；Week 7 不得原地续训、覆盖或改写这些产物。
- `week3_evaluation_v2` 已被 Week 6 用于最终评测，只能作为历史对照，不得再用于 Week 7 的样本选择、训练、validation、early stopping、方法选择或超参数调整。
- Week 7 必须从未进入 `week3_evaluation_v2` 且未被 Week 6 训练消费的来源建立新的 development/test 身份锁。训练、development 和 test 必须互斥，并通过 `sample_id`、`source_id`、`image_sha256`、`group_id`、`constraint_template_id` 五维隔离校验。
- 新 development 集用于基线固定、错误切片、方法比较和参数选择；新 test 集在全部参数和方法锁定后只执行一次最终评测，不得根据 test 结果继续调参。
- 所有实验必须使用新的版本化数据身份、配置和 run ID，记录样本清单及哈希、模型与 adapter、生成参数、随机种子、失败记录和指标支持数。不得把模型生成或自动校验结果标记为人工金标。
- Week 5 多轮对话中只有真实完成人工验收的记录可以声明为 human accepted；其余记录如用于训练，必须保留真实的 silver/未人工验收身份及相应权重，不得自动补写人工审核状态。

### A. 多任务混合微调

#### 1. 多任务数据集融合与采样策略设计

- 整合以图搜商品、智能售后、多模态行程规划三类单任务指令数据集与通用领域正则数据，构建统一多任务训练集。
- 严格保证训练集与验证集无重叠，并与独立测试集完全隔离。
- 采用均衡采样策略：对样本量最大的以图搜商品场景执行降采样，对样本量较小的智能售后和行程规划场景执行加权上采样，避免大样本任务压制小样本任务。
- 在多任务训练集中加入 `8%-10%` 的通用多模态指令数据，覆盖通用图文问答、常识推理等场景，降低领域过拟合和灾难性遗忘风险。
- 输出多任务数据集配比方案与采样逻辑文档，并固定随机种子以保证训练可复现。

#### 2. 统一训练配置与防过拟合机制落地

- 沿用 QLoRA 4bit 训练框架与核心参数：`r=16`、`alpha=32`，训练注意力层与视觉投影层。
- 针对多任务训练将学习率调整为 `1.5e-4`，并增大权重衰减系数以提升泛化能力。
- 开启梯度裁剪、Dropout 和早停机制；当验证集综合指标连续 `2` 个评估步无提升时自动终止训练。
- 建立多任务综合评估指标体系，为三个场景设置权重，并以加权综合得分作为最优 checkpoint 的筛选标准，避免单任务最优而整体失衡。
- 保留验证集分场景独立统计能力，能够拆解各场景的效果变化。

#### 3. 多任务联合训练与过程管控

- 启动统一底座模型的全量训练，持续监控训练损失、各场景验证损失和综合得分，排查收敛异常与过拟合迹象。
- 每完成总训练步数的 `10%` 执行一次全场景验证评估，留存所有中间 checkpoint，以支持效果回溯和版本回退。
- 针对训练中暴露的薄弱场景，动态调整采样权重和学习率；所有调整必须记录触发依据、修改值和生效位置，且不得使用独立测试集参与调参。

#### 4. 效果对标验证与最优权重筛选

- 对训练完成的统一底座模型执行全量验证集测试，分别输出三个场景的核心业务指标。
- 与 Week 6 单任务 LoRA 效果基线横向对比，控制各单任务效果下降幅度不超过 `2%`。
- 与零样本基线对比，验证统一底座模型在三个场景的整体提升幅度。
- 选择加权综合得分最高的 checkpoint，作为统一领域 VLM 底座的基准 LoRA 权重。

### B. 对话上下文数据融入

#### 1. 多轮对话数据标准化与训练集融合

- 参考 Qwen3-VL 官方多轮对话模板，将 Week 5 构建的多轮对话数据转换为标准训练格式，统一系统提示、用户轮、助手轮的角色标记和图片占位符位置。
- 对多轮图片指代场景，图片仅在首次出现的用户轮中插入占位符，后续轮次通过文本指代关联历史图片。
- 将多轮对话数据按约 `15%` 的占比融入多任务混合训练集，与单轮指令数据共同训练，使模型同时具备任务执行和对话交互能力。
- 单独拆分对话验证集并独立统计对话效果，避免与单任务指标混淆。

#### 2. 对话能力专项训练优化

- 适当提高最大序列长度，支持完整输入 `5-8` 轮对话上下文。
- 优化截断策略，优先保留历史对话结构，避免因截断破坏上下文逻辑。
- 加入对话一致性约束，强化模型对历史信息的承接能力，减少前后回复矛盾和遗忘已确认需求的问题。
- 在对话数据中加入检索、规则引擎等工具调用格式样本，统一工具调用指令格式，为后续 Agent 对话能力保留兼容性。

#### 3. 多轮交互效果专项验证

- 构建对话专项评测集，覆盖历史图片指代、需求迭代调整、上下文信息承接和多轮逻辑连贯性四个维度。
- 采用自动化指标与人工抽样评估结合的方式：自动化评估输出格式合规率和上下文信息召回率；人工抽样评估对话连贯性、逻辑合理性和 OTA 业务专业性。
- 与纯单任务模型的对话效果进行对比，量化多轮能力提升，并整理对话场景仍存在的短板与优化方向。

### C. 受控实验顺序

1. 建立新的 development/test 数据锁，完成五维隔离、数据身份、Schema、图片引用和哈希验证。
2. 在固定 development 集上复现 Week 6 三个单场景 adapter 的基线，预先锁定三场景业务指标、JSON/Schema 合规率、指标支持数、综合权重和非回退门禁。
3. 单独比较自由生成与 Schema constrained decoding。该实验只评估格式稳定性、后端 Schema 覆盖、延迟和失败回退，不得修补旧输出，也不得把格式提升解释为语义提升。
4. 基于 development 集建立商品多标签、售后严重度与关键信息、行程约束及多轮上下文错误切片，再构建场景均衡的多任务混合训练数据。
5. 执行多任务 QLoRA/SFT，并在相同 development 口径下与 Week 6 三个单场景 adapter 公平比较。任何采样权重或学习率调整均生成新的配置和 run ID，不得原地覆盖实验。
6. 只有 chosen/rejected 偏好对的业务正确性、视觉证据、Schema 和来源身份通过真实审核后，才允许执行一次 mDPO/HDPO 风格的小规模消融；条件不满足时明确记为 `SKIPPED`，不得由同一模型自举结果冒充高质量偏好数据。
7. 参数、采样比例、上下文配置和解码方式全部锁定后，在新的未见 test 集上只评测一次，并据此生成 Week 7 最终报告。

### D. 交付与验收

- 输出新的 development/test 锁、五维隔离报告、多任务数据配比与可复现采样逻辑。
- 输出统一 QLoRA 配置、对话格式转换与截断配置、分场景和对话独立验证配置，以及版本化训练和 checkpoint 记录。
- 输出 Schema constrained decoding 独立对比结果、多任务模型与 Week 6 单场景 adapter/零样本基线的对比结果，以及真实 bad case 和限制。
- 综合门禁至少要求 JSON/Schema 不回退、主要业务指标按预先声明口径评估、指标支持数不得通过删除困难样本下降、延迟与失败率在记录的实验预算内；单任务效果相对 Week 6 基线的下降幅度不得超过 `2%`。
- 对话专项报告必须分别给出格式合规率、上下文信息召回率和真实人工抽样结果；没有人工输入时不得伪造连贯性、逻辑合理性或业务专业性结论。
- 仅在所有参数锁定后执行一次新 test 评测。任何未真实运行的训练、DPO、人工评估或指标均标记为 `PENDING` 或 `SKIPPED`，不得宣称完成或提升。
- 更新 README、周日志、交付清单、实验记录和 Week 7 报告；不得修改 Week 6 终态报告，不得生成 Week 8 计划。

## 系统收敛修复与统一封装（2026-08-24）

### 当前范围

- 对 Week 1-7 已发现的工程和模型问题执行实际修复，不以新增“已知限制”代替修复。
- 不新增人工标注；新构建标签只允许标记为 `silver` 或自动验证结果。Week 3、Week 6、
  Week 7 的冻结数据、运行和报告保持不可变。
- 默认业务模型为 `Qwen/Qwen3-VL-8B-Instruct` 加 Week 7 unified adapter；规范运行时为
  Transformers、PEFT 和 NF4，vLLM 仅保留为可选后端。
- 所有修复使用新的 config、dataset、run 和 test identity。开发门禁通过前不得读取新 test；
  test 只能消费一次。

### 必须修复

- API 生产模式禁止静默 fallback；模型、Schema 或检索失败必须返回明确错误。
- 提供三场景任务接口、多轮对话、CLIP/Milvus 视觉检索、`/health` 和严格 `/ready`。
  `/ready` 必须核验模型、adapter、Prompt、Schema、CLIP、Milvus 和 release identity。
- 每个结构化模型请求最多允许一次模型级 Schema 纠错；保留两次原始输出，脚本不得补字段。
- 新建 `week5_preannotation_repair_v2`，替换 44 个不可读输入并修复 20 个 JSON/Schema
  失败；原 80,000 候选、64 条失败和人工 accepted 统计保持不变。
- 建立 1,980 条自动修复训练集和独立 development/test：三个核心场景各 500、对话
  300、通用正则 180，silver 权重不超过 0.5；从 checkpoint-226 受控继续 SFT，旧
  adapter 不覆盖。
- 在每场景 48 条 fresh development 上比较当前 Week 7、`compact_schema_v1` 和
  `evidence_state_v1`；不根据 test 或全量运行反向选 Prompt。
- 使用 1,000 张真实 OTA 图片生成 CLIP 512 维向量，完成 Milvus HNSW/COSINE CRUD、
  过滤、延迟和 Recall@K 实测。
- 提供统一 Compose、release manifest、`tripctl doctor/validate/serve/smoke` 和本地分层
  交接包；不得打包密钥、模型缓存或未经许可的原始 Yelp 数据。

### 晋级门禁

- 核心场景主要业务指标不得低于同一 development 上最佳现有基线；JSON/Schema 目标均
  不低于 95%，失败率不高于 2%，平均延迟不超过基线 1.25 倍，支持数不得通过删除困难
  样本下降。
- 对话使用独立 `DIALOGUE_BETA` 门禁，同时必须超过 Week 6 routed 与 zero-shot；旧严格
  研究门禁结论不改写。
- 只有代码测试、Week 5 80,000/80,000 Schema-valid、Prompt pilot、修复 adapter、四场景
  smoke、Milvus 实测、Docker/本地交接包哈希和干净 checkout 全部通过，才允许快进 `dev` 与
  `stg`；否则代码和失败证据可进入 `dev`，但不得进入 `stg`。

### 2026-08-25 导师交接口径修正

- 交付目标是让下一位接手者能够验证、解压和运行当前模型，不要求使用 Spartan 或 OSS
  留存，也不要求保留每周全部原始数据、运行输出、checkpoint 和模型缓存。
- Git 保存代码、配置、Prompt、Schema、测试、交接说明和汇总报告；Git 外只保留一份
  通过 SHA-256 复验的 adapter/runtime/retrieval/evidence 本地交接包。
- 允许清理可下载或可再生成的 Yelp 数据、基座缓存、迁移目录及历史中间输出，但不得
  删除唯一交接包、修改冻结历史结论或把凭据纳入交付。
- 导师要求提出若干 Week 8 优化方向供其调整任务。当前只输出证据、候选方向和建议验收
  指标，不把任何候选写成已确定 Week 8 计划。
