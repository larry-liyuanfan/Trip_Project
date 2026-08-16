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

## ADR-016：Week 5 Qwen3-VL-4B 运行、状态与对话版本边界

- **日期**：2026-08-09
- **状态**：Accepted
- **决策**：Qwen3-VL-4B 的商品预标注固定使用 `standardized_v2`，售后固定
  使用 `fewshot_4_v2`。行程仅允许在同一组最多 30 条 Week 5 候选上配对比较
  `fewshot_4_v2` 与 `standardized_v4`；若没有有效结论，默认使用
  `fewshot_4_v2`。Week 5 候选只需与冻结评测集在样本、来源、图片、来源组和
  约束模板五维隔离，训练候选场景之间不要求 `group_id` 互斥。现有 80,000 条
  候选和 30 条历史 pilot 不覆盖。
- **决策**：新增 workflow v2 sidecar，以候选文件哈希和 `sample_id` 绑定原候选；
  模型状态与人工状态分离。新增 `multimodal_dialogue_v2`，使用
  `image_resources/turns/source_sample_ids/generation/human_review/qc`，保留 v1
  不变且禁止别名混用。
- **决策**：任何全量预标注前必须具备不可覆盖 run ID、独立运行目录、配置和
  候选哈希、逐请求输入/请求哈希、独立原始输出、尝试与 retry 记录、确定性分片、
  checkpoint、独立失败文件，以及仅在元数据哈希完全一致时允许的显式 resume。
- **成本边界**：本次仅授权行程配对 pilot：最多 30 个唯一样本、两个 Prompt、
  60 次总请求、1.0 GPU 小时和 CNY 20，任一上限先到即停止；首 5 组后基础设施
  或请求失败率超过 20% 立即停止。未授权 80,000 条全量预标注。
- **原因**：模型预标注不能替代人工金标；历史候选和运行必须保持可追溯且不可
  覆盖；GPU 成本需要显式上限。
- **影响**：工具链和测试通过后才可启动 ECS。pilot 后必须停止 vLLM、执行
  `sync`，并确认 ECS 为“已停止 + 节省停机模式”。Week 6 训练不在范围内。

## ADR-017：Week 5 全量模型预标注授权

- **日期**：2026-08-09
- **状态**：Accepted
- **决策**：用户直接批准执行现有 80,000 条候选的 Qwen3-VL-4B 全量模型预标注，
  因而仅替代 ADR-016 中“未授权全量预标注”的成本门结论；Prompt 映射、候选哈希、
  不可覆盖运行目录、逐请求审计、确定性分片、checkpoint、failure 导出和显式 resume
  要求继续有效。沿用 pilot 实测线性估算约 50.6 GPU 小时、CNY 927；实际费用与耗时
  以运行记录为准。用户同时要求暂不停止 ECS。
- **边界**：本授权只覆盖模型预标注，不把模型结果计为人工金标；真实人工修订、自审、
  交叉互审、核心抽检和多轮对话人工验收仍须等待真实人员输入，也不授权 Week 6 训练。
- **影响**：补齐全量运行审计入口并通过测试后，可启动新的唯一 run；历史 pilot、候选池、
  Week 3/4 冻结产物和用户现有工作区改动保持不变。

## ADR-018：Week 5 单人最小人工三级质检

- **日期**：2026-08-10
- **状态**：Accepted
- **决策**：用户确认只有一名人工操作者，并要求最大程度减少重复质检。每条最终数据
  仍须由该操作者查看原始输入、确认或修正模型结果，并在保存时显式完成自审；商品按
  1%/0.5%、售后和行程按 2%/1% 分别执行盲二次复核/核心抽检。两级抽样使用同一
  `sample_id` SHA-256 值形成嵌套集合。三阶段允许同一真实人员执行，但复核必须使用
  不同 `review_session_id`，且不得声称人员独立性。
- **原因**：原“所有样本交叉互审 + 5%/10% 抽检”在单人条件下既无法满足不同人员
  互审，也产生不可承受的重复劳动；确定性风险抽样能保留可审计的三级记录。
- **影响**：未抽中样本在真实人工修正和内联自审通过后可 accepted；抽中样本必须按
  当前 revision 完成相应后续阶段。模型输出、自动校验和 Agent 不能代替任何人工确认。

## ADR-019：Week 5 额外人工质检总量低于 500

- **日期**：2026-08-10
- **状态**：Accepted；替代 ADR-018 中的抽样比例，其余事实边界不变。
- **决策**：商品盲二次复核/核心抽检降至 0.2%/0.05%，售后和行程降至
  0.5%/0.1%。继续使用同一个 `sample_id` SHA-256 选择值，核心集合嵌套于盲复核
  集合，不允许人工换样。
- **实算结果**：现有 80,000 个候选对应商品 112/26、售后 102/21、行程 53/7，
  合计 321 次额外阶段操作，低于用户要求的 500 次。
- **边界**：降低的只是重复质检次数。每条最终 accepted 样本仍须由唯一操作者真实
  查看并确认或修正，且显式完成内联自审；不得把模型输出自动转为人工 accepted。

## ADR-020：Spartan 计算迁移、8B 训练基座与阿里云展示边界

- **日期**：2026-08-12
- **状态**：Accepted
- **决策**：欠费停机的阿里云 A10 不再作为活动计算节点，且不释放实例或数据盘。
  Week 5 剩余预标注迁移到墨尔本大学 Spartan，继续固定使用
  `Qwen/Qwen3-VL-4B-Instruct`、现有 Prompt 与 Schema，禁止将 4B/8B 输出混写进
  同一运行。迁移使用版本化 benchmark、确定性互斥分片、独立 run 和合并校验，
  不续写 A10 历史 run。
- **决策**：Week 6 QLoRA 主基座采用 `Qwen/Qwen3-VL-8B-Instruct`；售后和行程
  优先 8B，商品保留 4B 对照并只对 8B 做小样本验证。正式训练只能在 Week 5 数据
  版本、训练/验证切分和哈希锁定后开始；冻结 Week 3 评测集只用于参数锁定后的最终
  评估，不参与反复选参。
- **决策**：包月 CPU ECS `trip-api-sg` 只提供结果 API、静态报告和预计算示例，
  不部署本地 VLM、CUDA、vLLM、训练权重或实时 LoRA 推理。
- **身份边界**：用户于 2026-08-12 最新确认 `yzhang3504` 为本人持有并授权本项目使用的
  Spartan 账户，因此允许 Agent 代理核验资源并提交 Trip_Project 作业。密码不得写入
  文件、配置、命令、日志或 Git。所有项目文件必须位于新建的 Trip_Project 专属目录，
  只允许读取和管理本项目 Slurm job ID；不得读取、修改、取消或影响账户内既有的其他
  文件、目录、作业和进程。
- **原因**：降低阿里云 GPU 费用并缩短计算时间，同时保持历史运行不可变、任务审计
  清楚和第三方账户资源安全。
- **影响**：A10 最后远端观测只能作为历史线索；当前可独立验证的本地恢复点为
  15,166 条。若以后安全取得更完整的远端快照，必须生成新的迁移版本，不得覆盖当前
  migration。Spartan project、quota、scratch 和预计排队时间未核验前，只能交付可
  提交作业包，不能声称已经排队或运行。

## ADR-021：Spartan 项目存储与虚拟环境隔离

- **日期**：2026-08-12
- **状态**：Accepted
- **决策**：Trip 只使用 project GPFS 下的独立版本根目录
  `/data/gpfs/projects/punim2936/Trip_Project_yzhang3504/20260812a`。仓库、运行输出、
  Hugging Face/Apptainer/pip 缓存、临时文件和 Python 环境均置于该根目录；禁止向
  已满的 home 写入项目文件。Python 统一使用 `GCCcore/11.3.0` + `Python/3.11.3`，
  虚拟环境固定为 `envs/trip-week5-week6-py311`，并通过 Slurm 作业安装
  `requirements.txt` 与 `requirements-training.txt`。
- **容量事实**：`/data/gpfs` 文件系统实测总量 467 GiB、已用 375 GiB、可用 93 GiB；
  “500 GB”是共享 project 文件系统的标称总量，不是 Trip 独享配额。Trip 部署目录
  在本次核验时仅 99 MiB；后续模型、容器和环境缓存增长必须继续留在独立根目录，
  不得占用或整理同项目其他成员目录。
- **运行边界**：Week 5 vLLM 使用固定 Apptainer 镜像；项目 venv 用于 CPU 工具、
  校验及 Week 6 transformers/PEFT 训练入口。两者不混装。空间不足时先报告并清理
  本项目可重建缓存或申请项目配额，不移动、删除或覆盖其他成员文件。

## ADR-021：Spartan 项目存储与虚拟环境隔离

- **日期**：2026-08-12
- **状态**：Accepted
- **决策**：Trip 只使用 project GPFS 下的独立版本根目录
  `/data/gpfs/projects/punim2936/Trip_Project_yzhang3504/20260812a`。仓库、运行输出、
  Hugging Face/Apptainer/pip 缓存、临时文件和 Python 环境均置于该根目录；禁止向
  已满的 home 写入项目文件。Python 统一使用 `GCCcore/11.3.0` + `Python/3.11.3`，
  虚拟环境固定为 `envs/trip-week5-week6-py311`，并通过 Slurm 作业安装
  `requirements.txt` 与 `requirements-training.txt`。
- **容量事实**：`/data/gpfs` 文件系统实测总量 467 GiB、已用 375 GiB、可用 93 GiB；
  “500 GB”是共享 project 文件系统的标称总量，不是 Trip 独享配额。Trip 部署目录
  在本次核验时仅 99 MiB；后续模型、容器和环境缓存增长必须继续留在独立根目录，
  不得占用或整理同项目其他成员目录。
- **运行边界**：Week 5 vLLM 使用固定 Apptainer 镜像；项目 venv 用于 CPU 工具、
  校验及 Week 6 transformers/PEFT 训练入口。两者不混装。空间不足时先报告并清理
  本项目可重建缓存或申请项目配额，不移动、删除或覆盖其他成员文件。

## ADR-022：Week 5 Spartan 全量自动恢复执行

- **日期**：2026-08-12
- **状态**：Accepted
- **决策**：当前 100 条 L40S benchmark 通过身份、哈希、成功率和吞吐核验后，
  无需再次人工批准，立即提交唯一 `gpu-l40s` array `0-3`，并行处理四个确定性互斥
  分片。benchmark 或 shard 失败时，先取得明确根因，再自动修复本任务直接相关的
  脚本、依赖、路径、权限、容器、缓存、超时或恢复逻辑；验证通过后自动重新排队。
- **恢复约束**：禁止盲目原样重试。可恢复运行必须保持 run identity、配置和候选哈希
  一致，使用 `TRIP_RESUME=1`，只重提失败或未完成的 shard index；已成功 sample_id
  不重复请求。不得同时提交多个 GPU 分区竞争作业。
- **不变边界**：不得修改候选池、migration manifest、冻结 Week 3/4 产物、历史失败
  证据或人工状态。所有操作限于 Trip 专属 GPFS 和登记作业。80,000 条闭环后执行
  merge、去重、隔离、JSONL 与哈希验证，并停止监控。

## ADR-023：Week 5 单人预算内抽样验收

- **日期**：2026-08-14
- **状态**：Accepted；细化 ADR-019，并以最新用户指令解决“全量人工修订”与
  “3 小时、低于 500 次操作”的冲突。
- **决策**：在三场景 Schema-valid 预标注中各确定性选择 100 条进行真实人工验证，
  保留并计入已完成的商品 10、售后 8、行程 9 条。每个场景的 100 条队列固定包含
  10 条现行 SHA-256 规则选中的盲复核候选和其中 3 条核心抽检候选。因此完整预算为
  300 次人工修订与内联自审、30 次盲复核、9 次核心抽检。完成单轮队列后自动生成
  10,000 条多轮对话候选，并固定抽取 100 条由本人验收；人工操作总上限为 439。
- **边界**：其余有效预标注继续标记为 silver。自动校验、模型输出或 Agent 不得填写
  人工身份、确认、复核、抽检或 accepted；也不得通过改名、汇总口径或“美化结果”
  将 silver 伪装成人工数据。多轮对话允许自动生成候选，但只有本人真实检查的记录
  才可计为人工 accepted。
- **影响**：Week 5 工程覆盖仍按 79,936 成功与 64 最终失败验收；人工交付改为如实
  报告 300 条单轮抽样队列和 100 条对话抽样队列的完成率、问题分布及质检证据，
  不再声称达到原全量人工合格数量目标。

## ADR-024：Week 5 多轮对话保留原作业并独立并行分片

- **日期**：2026-08-16
- **状态**：Accepted；来自用户最新直接指令，仅适用于 Week 5 多轮对话。
- **决策**：保持活动串行作业 `29259879` 不变；新增对话生成的 bounded range、
  modulo shard 和有界客户端并发。每个 Slurm array task 使用独立 run ID、运行目录、
  candidates/failures 和 vLLM 日志，禁止多个进程写同一 JSONL。分片绑定相同配置与
  qualified sample 集合哈希，最终使用显式 merge 以源顺序去重，并验证确定性
  10,000 ID 全集、Schema 和图片引用。
- **原因**：活动链路 HTTP 500 为 0，但生成循环实际串行，配置中的并发未用于对话；
  单个 24 小时 L40S 配额按实测吞吐不足以一次完成 10,000 条。
- **影响**：ADR-022 的“不提交竞争作业”仍约束单轮预标注迁移，但不再禁止本次
  独立输出的多轮对话分片。现有作业不取消、不修改；额外 GPU 只处理版本化分片，
  未经 merge 完整校验的分片不能单独计为最终候选，也不授权任何 Week 6 工作。

## Decision Template

```markdown
## ADR-XXX: Title

- **Date**: YYYY-MM-DD
- **Status**: Proposed | Accepted | Superseded
- **Decision**:
- **Reason**:
- **Consequence**:
```
