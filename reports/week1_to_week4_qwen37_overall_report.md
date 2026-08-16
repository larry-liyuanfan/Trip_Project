# Week 1-4 Qwen3.7-Plus 整体报告

## 1. 总体结论

项目已将业务推理从本地 `Qwen/Qwen2-VL-2B-Instruct` 迁移到阿里云百炼
`qwen3.7-plus`，部署地域为新加坡，采用 OpenAI-compatible API，关闭
thinking。Week 1-4 的工程链路、数据处理、零样本评测、Prompt 优化和
Milvus 原型均已形成可复现交付。

| 周次 | 核心交付 | Qwen3.7 迁移后的状态 |
| --- | --- | --- |
| Week 1 | 工程结构、FastAPI 图像理解接口、容器化与数据样例 | 云端真实图片 smoke test 通过，无本地 fallback |
| Week 2 | Yelp 全量清洗、多模态对齐、CLIP 降噪 | 与生成模型无关，原实测结果继续有效 |
| Week 3 | 450 条人工金标、最简 baseline、标准化 Prompt 与量化评分 | Qwen3.7 全量重跑 450/450；请求错误 0 |
| Week 4 | Prompt 候选比较、bad case、Milvus 向量检索原型 | 按 Qwen3.7 重新选优；行程经 v4 修复后达到稳定结构化输出 |

本报告把三类结论分开：模型不参与的 Week 2 数据结果、同 Prompt/同数据的
模型替换结果、以及 Qwen3.7 上继续修改 Prompt 后的最终结果。最后一类不能
解释为纯模型提升。

## 2. 运行基线

| 项目 | 当前配置 |
| --- | --- |
| 业务推理模型 | `qwen3.7-plus` |
| 后端 | Alibaba Cloud Model Studio OpenAI-compatible API |
| 地域 | `ap-southeast-1` |
| thinking | `false` |
| temperature / top_p | 0.1 / 0.9 |
| 通用输出预算 | 1280 tokens |
| 行程修复输出预算 | 2560 tokens |
| 评测集 | `week3_evaluation_v2` |
| 评测规模 | 商品 200、售后 150、行程 100，共 450 |

API Key、业务空间端点、原始数据和运行输出均不进入 Git。ECS 只运行 API 与
依赖服务，模型推理由百炼托管端点完成。

## 3. Week 1：工程与云端推理链路

Week 1 已完成项目目录、Docker、FastAPI `/health` 与
`/v1/image-understanding`、图片 data URL 转换、Yelp 小样本准备和实验记录
规范。模型迁移后，真实 OTA 图片请求已通过阿里云部署链路返回
`qwen3.7-plus`，且云端配置禁止在请求失败时静默使用 deterministic
fallback。因此当前 smoke 结果能够证明真实模型链路可用。

本次没有用云模型重新执行 Week 1 的纯数据抽样，因为 POI 200、评论 1,000、
多模态条目 581 的生成过程不依赖生成模型。

## 4. Week 2：Yelp 多模态数据基线

Week 2 是确定性数据处理和 CLIP 向量任务，不受 Qwen2-VL 切换到 Qwen3.7
影响，因此不重复消耗模型调用。原全量结果继续作为当前数据基线。

| 项目 | 实测结果 |
| --- | ---: |
| 商家 | 150,346 |
| 有效评论 | 6,989,830 / 6,990,280 |
| 图片元数据 | 200,100 |
| 有效本地图片 | 199,994 |
| 损坏图片 | 106 |
| 强对齐 image-caption | 96,733 |
| 中粒度 image-business | 199,994 |
| 弱对齐商家组 | 36,673 |
| CLIP 候选 / 保留 | 555,459 / 131,146 |
| CLIP 保留率 | 23.61% |

CLIP 使用 `openai/clip-vit-base-patch32`、CUDA 和阈值 0.25。该分数用于弱
对齐降噪，不替代人工语义标注。

## 5. Week 3：Qwen3.7 零样本基线

### 5.1 数据与隔离

`week3_evaluation_v2` 包含 450 条人工金标：商品 200、售后 150、行程
100。baseline、standardized 和 Week 4 winner 使用同一 selected-sample
SHA-256。evaluation exclusion registry 继续用于拒绝未来训练候选中的
`source_id` 或图片 SHA-256 冲突。

### 5.2 最简 baseline

有效运行 `week3_qwen37_baseline_full_20260802_002` 完成 450/450，请求错误
为 0。baseline 不含角色、JSON 字段、格式限制、示例或思维链，因此
JSON/Schema 均为 0% 是预期格式结果，不能解释为业务语义为零。

| 场景 | Qwen3.7 baseline 主要指标 | JSON / Schema | 平均 / P95 延迟 |
| --- | --- | --- | --- |
| 商品 | 业态 40.00%；价位 10.00%；风格 F1 19.75%；设施 F1 41.58% | 0% / 0% | 14.61 / 19.01 s |
| 售后 | 分类 69.33%；严重度 2.67%；关键信息 F1 16.20%；OCR recall 33.33% | 0% / 0% | 12.64 / 20.92 s |
| 行程 | 约束识别 0%；行程要素完整度 96.00% | 0% / 0% | 23.86 / 29.76 s |

自然语言语义指标由冻结的 `baseline_semantic_coding_v1` 确定性词法编码器
产生。它对同义改写、隐含约束和否定表达的覆盖有限。

### 5.3 与原 Qwen2-VL baseline 的同口径变化

| 指标 | Qwen2-VL | Qwen3.7 | 变化 |
| --- | ---: | ---: | ---: |
| 商品业态准确率 | 45.45% | 40.00% | -5.45 pp |
| 商品价位准确率 | 2.00% | 10.00% | +8.00 pp |
| 商品风格 macro F1 | 28.28% | 19.75% | -8.53 pp |
| 商品设施 macro F1 | 53.22% | 41.58% | -11.64 pp |
| 售后问题分类准确率 | 60.00% | 69.33% | +9.33 pp |
| 售后严重等级准确率 | 0.00% | 2.67% | +2.67 pp |
| 售后关键信息 macro F1 | 29.67% | 16.20% | -13.47 pp |
| 售后 OCR recall | 14.22% | 33.33% | +19.11 pp |
| 行程要素完整度 | 77.20% | 96.00% | +18.80 pp |

Qwen3.7 并非在所有最简自然语言指标上都提升。其云端平均延迟也高于原本地
Qwen2-VL；两者运行硬件和网络不同，延迟不能作为纯模型速度比较。

## 6. 标准化 Prompt 与 Qwen3.7 行程修复

### 6.1 同一 `standardized_v2` 下的模型替换

| 场景/指标 | Qwen2-VL | Qwen3.7 | 变化 |
| --- | ---: | ---: | ---: |
| 商品 JSON / Schema | 79% / 75% | 100% / 100% | +21 / +25 pp |
| 商品业态准确率 | 60.00% | 84.55% | +24.55 pp |
| 商品价位准确率 | 17.00% | 35.00% | +18.00 pp |
| 售后 JSON / Schema | 96.67% / 96% | 100% / 100% | +3.33 / +4 pp |
| 售后问题分类准确率 | 71.33% | 92.67% | +21.34 pp |
| 售后严重等级准确率 | 29.33% | 63.33% | +34.00 pp |
| 售后 OCR recall | 1.33% | 100.00% | +98.67 pp |
| 行程 JSON / Schema | 90% / 88% | 33% / 33% | -57 / -55 pp |

商品和售后在相同 Prompt 下明显提升；行程出现严重回归。检查原始元数据后，
67/100 条行程输出恰好达到 1280 completion tokens 并被截断，且行程越长，
有效率越低。

### 6.2 `standardized_v4` 最终修复

修复没有覆盖历史 Prompt 或运行。`standardized_v4` 压缩重复字段、保留原文
约束、固定英文枚举，并将行程专用预算提高到 2560。最终运行
`itinerary_qwen37_repair_v4_full_20260802_001` 完成 100/100，所有
`finish_reason=stop`。

| 指标 | Qwen3.7 v2 | Qwen3.7 v4 | 变化 |
| --- | ---: | ---: | ---: |
| JSON / Schema | 33% / 33% | 100% / 100% | +67 / +67 pp |
| 约束识别准确率 | 0.14% | 89.95% | +89.81 pp |
| 硬约束 F1 | 0.40% | 96.33% | +95.93 pp |
| 软约束 F1 | 0.00% | 85.67% | +85.67 pp |
| 约束检查覆盖率 | 4.83% | 94.00% | +89.17 pp |
| 行程要素完整度 | 33.00% | 100.00% | +67.00 pp |
| 平均 / P95 延迟 | 24.72 / 27.63 s | 20.22 / 25.40 s | -4.50 / -2.23 s |

这是模型迁移后的最终行程配置效果，但同时包含 Prompt 与输出预算修改，不能
将全部增量归因于 Qwen3.7 模型本身。

## 7. Week 4：Prompt 优化与向量数据库

### 7.1 Qwen3.7 Prompt 候选重选

每场景在固定 15 条 pilot 上重新比较 `standardized_v2`、4-shot 和 7-shot，
不沿用 Qwen2-VL 的胜出结论。Qwen3.7 pilot 选择为：

| 场景 | 胜出候选 | 选择分数 |
| --- | --- | ---: |
| 商品 | `fewshot_4_v2` | 0.4997 |
| 售后 | `fewshot_4_v2` | 0.9292 |
| 行程 | `standardized_v2` | 0.3250 |

原全量 winner run 完成 450/450。商品和售后分别达到 JSON
100%/100%、Schema 98.5%/100%；行程 v2 因截断只有 33%/33%，随后由
版本化 v4 修复。`fewshot_4_v2` 只表示本次候选中的 pilot 胜出版本，不表示
全局最优 Prompt。

Qwen3.7 winner 相对其自身最简 baseline 的共同词法轨道主要变化为：商品
业态 +38.18 pp、设施 F1 +6.30 pp、风格 F1 +6.72 pp、价位 -9.00 pp；
售后问题分类 +6.00 pp、关键信息 F1 +20.80 pp、严重度 -2.67 pp。
这些结果说明 Prompt 优化具有字段差异，不能只用单一总分概括。

### 7.2 Milvus 原型

Milvus 不依赖生成模型，本次不重复部署性能实验。已验证 standalone + etcd +
MinIO、十字段 `ota_business_image_vector`、HNSW/COSINE、八个标量索引和
五类 SDK 操作。20 条真实 Yelp 图片使用 CLIP 512 维向量完成 CRUD：HNSW
构建 5.6621 s，10 次查询平均/P95 为 7.7982/10.7236 ms，Recall@5 为
1.0000。该规模只证明原型链路，不代表生产性能。

## 8. 最终能力与限制

当前可确认的结果：

- 阿里云 `qwen3.7-plus` 真实图片推理链路可用，失败不会静默降级。
- Week 2 全量数据处理、对齐和 CLIP 降噪结果继续有效。
- Qwen3.7 最简 baseline、标准化 Prompt 和 Week 4 候选均有真实全量证据。
- 商品与售后标准化结构合规达到 100%；最终行程 v4 达到 JSON/Schema
  100%，并显著恢复约束识别。
- Milvus 的结构、索引、过滤检索、CRUD 和小规模性能已验证。

已知限制：

- baseline 语义指标依赖确定性词法编码，对复杂自然语言的召回有限。
- Qwen2-VL 在本地 GPU、Qwen3.7 在云端运行，延迟不具备严格硬件可比性。
- Week 4 pilot 样本较小，“胜出”仅适用于已测试候选。
- Milvus 仅有 20 条向量性能基线，不宣称生产级容量或延迟。
- 行程 v4 的提升同时包含 Prompt 和 token 预算变化，不是纯模型消融结论。

## 9. 证据与复现入口

- 云部署：`docs/aliyun_deployment.md`
- Week 2 数据报告：`reports/yelp_multimodal_data_processing_report_part1.md`
- Qwen3.7 全量重跑：`reports/qwen37_previous_weeks_rerun_report.md`
- 行程修复：`reports/qwen37_itinerary_repair_report.md`
- Week 4 Prompt：`reports/week4_prompt_optimization_report.md`
- Week 4 Milvus：`reports/week4_milvus_deployment_performance_report.md`
- 实验记录：`docs/experiments.md`、`experiments/experiment_log.md`、
  `experiments/results.csv`

大规模数据、模型原始输出、评分明细、密钥和 Milvus volumes 均保持 Git
忽略。仓库只提交配置、实现、测试和可追溯报告。
