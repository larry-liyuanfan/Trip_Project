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
  `300 + 30 + 9 = 339`，满足 3 小时与低于 500 次边界。未进入人工队列的成功
  预标注保持 `silver`，不得计为 `human_revised`、`accepted` 或人工合格数据。
- 原始 4.5 万/1.8 万/0.9 万单轮合格目标和 0.9 万人工合格对话目标，在当前单人
  预算下不作为人工完成声明；工程输出、silver 候选、人工抽样合格和对话候选/人工
  合格必须分别报告。自动生成的多轮对话可以作为候选，但未由本人检查的记录保持
  `awaiting_human_validation`，不得美化为人工 accepted。
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
