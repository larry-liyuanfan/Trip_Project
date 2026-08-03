# Technical Decisions

Record decisions that affect architecture, reproducibility, model serving, data handling, branching, or review scope.

## ADR-001: Keep API Tests Independent from Live vLLM

- **Date**: 2026-07-09
- **Status**: Accepted
- **Decision**: Maintain deterministic fallback responses for local image-understanding tests when live vLLM is not configured.
- **Reason**: Contributors can run core tests without GPU access, model downloads, or container startup.
- **Consequence**: Live model behavior must be validated separately through smoke tests and experiment records.

## ADR-002: Store Raw and Generated Yelp Data Outside Git

- **Date**: 2026-07-09
- **Status**: Accepted
- **Decision**: Keep raw Yelp archives, extracted images, and generated large subsets ignored locally.
- **Reason**: These files are large, external, and may have distribution restrictions.
- **Consequence**: Dataset preparation must be reproducible from documented commands and local source files.

## ADR-003: Use Experiment Files for Reproducibility

- **Date**: 2026-07-09
- **Status**: Accepted
- **Decision**: Track experiment commands, parameters, outcomes, and failures in `experiments/` and summarize them in `docs/experiments.md`.
- **Reason**: Weekly mentor review needs clear evidence of what was run and what changed.
- **Consequence**: Model, prompt, data, and serving changes should update experiment documentation before review.

## ADR-004: Use `dev`, `stg`, and `main` for Weekly Delivery

- **Date**: 2026-07-09
- **Status**: Accepted
- **Decision**: Use `feature/* -> dev -> stg -> main` as the promotion flow. Daily work happens on `dev` or `feature/*`; verified weekly deliverables promote to `stg`; milestone or mentor-confirmed stable versions promote from `stg` to `main`.
- **Reason**: This separates active development and experiments from mentor-reviewed weekly deliverables and milestone-level stable code.
- **Consequence**: Before merging into `stg`, provide a changed-files summary, verification commands and results, expected outputs, known limitations, updated documentation, and a proposed weekly tag such as `week2-yelp-data-processing`.

## ADR-005: Build Week 2 Yelp Processing as a Config-Driven Offline Pipeline

- **Date**: 2026-07-09
- **Status**: Accepted
- **Decision**: Add a reusable offline pipeline for Yelp JSONL parsing, local image validation, multimodal alignment generation, optional CLIP denoising, and report generation.
- **Reason**: Weekly review needs reproducible data preparation artifacts without requiring live VLM serving, GPU access, or committed raw Yelp files.
- **Consequence**: Raw and generated data stay under ignored `data/yelp/` paths; scripts must tolerate missing optional CLIP and Parquet dependencies while documenting the fallback.

## ADR-006: Keep One Canonical Weekly Delivery Record

- **Date**: 2026-07-11
- **Status**: Accepted
- **Decision**: Append each completed week to `docs/weekly_delivery.md`; keep `docs/weekly_log.md` as a concise timeline and avoid separate plan/delivery files per week.
- **Reason**: Separate Week 1 and Week 2 files drifted across branches and obscured earlier completed work.
- **Consequence**: Checklist state is finalized on `dev` before promotion, then inherited unchanged by `stg` and `main` through merge-based promotion.

## ADR-007: Freeze the Week 3 v1 Evaluation Labels

- **Date**: 2026-07-21
- **Status**: Superseded
- **Decision**: Keep the existing `week3_evaluation_v1` manifests and completed run artifacts immutable. Do not create `week3_gold_v2`, reopen the annotation UI, request supplemental labels, or perform v2 rescoring. Treat evidence-supported `unknown` values as completed labels rather than omissions. Keep itinerary image-style preference, after-sales facility-damage, and baseline natural-language semantic metrics `PENDING` where the frozen evidence does not support them.
- **Reason**: Project Control approved the frozen-v1 route after reviewing the annotation audit, historical UI backups, corrected product-facility statistics, and run provenance. The 100 empty itinerary style arrays are recorded as a probable historical field-exposure or serialization defect and are not attributed to the annotator.
- **Consequence**: Week 3 remains `PARTIAL`; reports must preserve support counts and limitations without modifying gold labels, rerunning equivalent inference, or converting sampling metadata into gold coverage.

## ADR-008: Build an Isolated Curated Week 3 v2 Evaluation Set

- **Date**: 2026-07-22
- **Status**: Accepted
- **Decision**: Preserve every v1 manifest, Prompt, Schema, and run as immutable history. Build `week3_evaluation_v2` in new ignored manifests and registry files, reuse all 200 product labels, retain 80 evidence-supported after-sales labels, replace 70 low-evidence after-sales candidates, and reopen the 100 itinerary pairs only to capture the previously unavailable image-style field. Use one human annotator; deterministic suggestions never become gold automatically.
- **Reason**: The mentor explicitly prioritized removal of low-quality images so the zero-shot baseline can serve as a fair reference for later comparison. The frozen-v1 set has unsupported facility-damage and itinerary-style dimensions.
- **Consequence**: V2 full baseline and standardized runs cannot begin until the 70 after-sales replacements and 100 itinerary style supplements are complete. Existing itinerary non-style labels are inherited, not re-entered. Standardized v2 may use a separately versioned bounded itinerary Schema and output-type skeleton to improve raw JSON/Schema compliance, while strict format compliance remains separate from semantic quality.

## ADR-009: Score Minimal-Baseline Semantics with a Gold-Independent Lexical Track

- **Date**: 2026-07-25
- **Status**: Accepted
- **Decision**: Use `baseline_semantic_coding_v1` to convert the immutable `baseline_minimal_v1` raw text into predictions before loading human gold. The encoder accepts only the scenario, raw output, a fixed versioned codebook derived from existing Schemas and annotation definitions, and general text normalization. Gold is joined only in the scoring stage.
- **Reason**: The mentor requires baseline business metrics, while the minimal Prompt intentionally produces unconstrained natural language. A deterministic lexical track measures supported semantics without changing the Prompt, rerunning inference, adding manual output coding, or treating JSON failure as semantic failure.
- **Consequence**: Store the lexical metrics as an independent scoring track with explicit support counts and codebook SHA-256. Preserve baseline JSON/Schema rates, raw output, latency, and run provenance. Do not compare this track to the standardized strict-structured track as if their difference were a pure Prompt effect.

## ADR-010：只从固定实测候选中选择 Week 4 Prompt

- **日期**：2026-07-25
- **状态**：Accepted
- **决策**：使用固定 v2 金标示例和不重叠 pilot，对比
  `standardized_v2`、4-shot、7-shot；按已提交的业务、JSON、Schema、
  token 和延迟加权分数选择每场景胜出版本，再只对胜出版本执行 v2 全量跑测。
- **原因**：在不修改金标、不猜标签、不扩展候选搜索的前提下满足导师要求。
- **影响**：`standardized_v2` 只是本次三个候选中的场景胜出版本。
  旧 Few-Shot v1 行程请求因上下文超限而失效；版本化 v2 在不改变模型和
  生成参数的前提下压缩重复上下文并完成有效重跑。新增 Few-Shot 候选仍未
  超过控制组，因此不称为新的“优化后最优 Prompt”。Week 3 产物保持不可变；
  baseline 词法编码与结构化严格评分不可直接比较，不计算业务差值。

## ADR-011：Milvus 和 CLIP 与业务推理解耦

- **日期**：2026-07-25
- **状态**：Accepted
- **决策**：使用固定版本 Milvus standalone 和独立 PyMilvus 依赖组，
  存储归一化的 512 维 `openai/clip-vit-base-patch32` 图片向量。
  Qwen2-VL 保持现有 vLLM 推理接口，不作为 embedding 端点。
- **原因**：完成真实向量 CRUD，同时不污染现有 API/data/vLLM 依赖，
  并遵守本地 8 GB GPU 资源边界。
- **影响**：运行 CLIP 前停止 vLLM；生成向量和 volumes 保持忽略。
  检索只支持固定标量白名单以及配置中的 HNSW/COSINE 参数。

## ADR-012：评估文本哈希跨平台稳定

- **日期**：2026-07-25
- **状态**：Accepted
- **决策**：通过 `.gitattributes` 强制评估 Prompt、Schema 和配置使用 LF；
  provenance 对文本换行归一化，并兼容既有运行曾按 LF 或 CRLF 原始字节
  记录的哈希。非换行字节变化仍必须导致验证失败。
- **原因**：Windows 自动换行转换不应使不可变运行证据失效。
- **影响**：Week 3 历史运行无需修改即可跨平台验证；未来运行使用统一的
  LF 文本哈希。

## ADR-013：共同语义轨道与 Few-Shot 证据边界

- **日期**：2026-07-26
- **状态**：Accepted
- **决策**：保留 Week 3 原词法评分和 Week 4 原严格结构化评分；另建
  `week4_common_semantic_coding_v1`，将两组冻结原始输出交给同一个
  `BaselineSemanticCoder.encode` 和 codebook，全部预测完成后再连接同一
  人工金标并执行同一指标与 paired bootstrap。现有 Few-Shot 示例来自最终
  测试集金标，其 pilot 仅作描述性证据，不支持无偏效果声明。
- **原因**：原业务指标使用不同预测转换，不能直接相减；同时示例与 pilot
  不重叠仍不能消除利用最终测试集金标设计 Prompt 的污染风险。
- **影响**：Week 3 原产物不覆盖。`standardized_v2` 的无示例全量运行仍可
  报告；Few-Shot 泛化比较保持 `PARTIAL`，除非以后获得明确授权的独立
  demo/dev pool，但本决策不创建该数据或未来任务。

> 2026-07-26 后续直接授权已满足上述条件；Few-Shot 数据边界由 ADR-014
> 接替。共同语义轨道部分继续有效。

## ADR-014：独立 demo/dev Few-Shot 证据

- **日期**：2026-07-26
- **状态**：Accepted
- **决策**：使用单独版本 `week4_demo_dev_v1` 和 `development` split
  保存 36 条人工金标；示例与最终 `week3_evaluation_v2` 在 sample、
  source、图片 SHA-256 和来源组四层隔离。选择文件升级为
  `week4_prompt_selection_v2`，旧 v1 不覆盖。
- **原因**：消除使用最终 test gold 设计 Prompt 的污染，使固定 pilot
  能支持本次候选内的无偏比较。
- **影响**：三组新 pilot 均须真实重跑且请求错误为 0；胜出版本只表示
  固定综合规则下的候选内最高分。全量结果不得反向用于重选 Prompt。

## ADR-015：Qwen3.7 行程输出使用紧凑 v4 Prompt

- **日期**：2026-08-02
- **状态**：Accepted
- **决策**：保留历史 v2/v3 产物，新增 `standardized_v4`。行程场景使用
  2560 token 独立输出预算，约束保持原文，活动证据不重复，Schema 枚举固定
  使用英文协议值；评估 CLI 允许在完整数据门禁后仅运行指定场景。
- **原因**：67/100 个旧输出因达到 1280 token 上限截断；v3 消除截断后，
  剩余失败全部来自 `required_itinerary_elements` 被翻译成中文。
- **影响**：最终 100 条行程 JSON/Schema 均通过，旧 Week 3/4 run、Prompt、
  Schema 和评分保持不可变。商品和售后配置不受影响。

## Decision Template

```markdown
## ADR-XXX: Title

- **Date**: YYYY-MM-DD
- **Status**: Proposed | Accepted | Superseded
- **Decision**:
- **Reason**:
- **Consequence**:
```
