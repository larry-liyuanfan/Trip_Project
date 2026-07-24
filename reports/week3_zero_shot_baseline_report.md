# Week 3 零样本基线与标准化 Prompt 评测报告

## 交付结论

Week 3 状态为 `READY / COMPLETED`。`week3_evaluation_v2` 的 450 条人工金标、评测隔离、全量最简 baseline、全量 standardized v2、评分与报告均已完成并可追溯。

最简 baseline 的三条 `baseline_minimal_v1` 原文、原始模型输出、延迟及 JSON/Schema 判断均未改变。自然语言业务指标由独立的 `baseline_semantic_coding_v1` 确定性词法编码轨道产生；它不是重新人工标注，也不等价于模型原生结构化输出。

## 数据与五类数量

| 场景 | target | candidate | annotated | validated | tested |
| --- | ---: | ---: | ---: | ---: | ---: |
| 以图搜商品 | 200 | 200 | 200 | 200 | 200 |
| 智能售后 | 150 | 150 | 150 | 150 | 150 |
| 多模态行程规划 | 100 | 100 | 100 | 100 | 100 |

- 商品候选分层为酒店 67、景点 67、餐饮 66。
- 售后候选分层为卫生污渍 38、设施损坏 38、景点关闭 37、交通延误 37；人工金标分布为 42/34/37/37。
- 售后来源包含 `public_yelp=6` 与 `business_synthetic=144`。
- 行程 100 组均包含参考图和文本约束；99 条有至少一个人工风格偏好，1 条为空数组。
- v2 exclusion registry 包含 450 个唯一 source/image hash；`data/eval/` 和 `data/yelp/` 均不进入 Git。

## 运行与评分验签

| 项目 | 值 |
| --- | --- |
| dataset | `week3_evaluation_v2` |
| model | `Qwen/Qwen2-VL-2B-Instruct` |
| baseline run | `week3_v2_baseline_full_20260724_001` |
| standardized run | `week3_v2_standardized_full_20260724_001` |
| semantic score | `week3_v2_baseline_full_20260724_001__baseline_semantic_coding_v1` |
| semantic coding version | `baseline_semantic_coding_v1` |
| codebook SHA-256 | `563dc0747f92b6ccaa37466045cb0e74229787824013d59a5f6f26261bb033a6` |
| 每个 run selected/record | 450/450 |
| semantic score samples | 450（200/150/100） |
| sample-set SHA-256 | `3e900e64bb345df35343c8f14bfb1f8310ae597a57e4a4d9585bc01173ad648c` |

两个原始 run 均为 `completed/live/full`，使用相同样本集合、顺序、模型配置与非 Prompt 资产。评分未重新发送模型请求。

## 确定性语义编码方法

编码严格分为两个阶段：

1. 预测阶段只接收 `scenario`、模型 `raw_output`、固定版本 codebook 和通用文本归一化规则。
2. 所有预测完成后，评分阶段才按 `sample_id` 加载并连接现有人工金标。

编码接口不能接收 annotation、annotator、sampling stratum、source metadata、标注建议、图片业务分类或 standardized 输出。标量字段仅在唯一明确命中时输出枚举；冲突或未命中返回 `unknown`。多标签字段只返回固定词表中的明确命中。OCR 只提取模型文本中明确出现的有界可见 token。

## 最简 baseline：格式与延迟

| 场景 | N | JSON 合规 | Schema 通过 | 平均延迟 | P95 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 以图搜商品 | 200 | 0% | 0% | 4,456.84 ms | 7,927.61 ms |
| 智能售后 | 150 | 0% | 0% | 3,616.71 ms | 6,663.71 ms |
| 多模态行程规划 | 100 | 0% | 0% | 7,136.77 ms | 14,817.89 ms |

450 条输出均触发 `json_parse_error`。这只表示最简自然语言输出不满足后端 JSON 契约，不直接作为语义错误。

## 最简 baseline：确定性词法编码指标

### 以图搜商品

| 指标 | 结果 | support |
| --- | ---: | ---: |
| 业态分类准确率 | 45.45% | 110 |
| 价位区间准确率 | 2.00% | 100 |
| 风格 macro / micro F1 | 28.28% / 16.53% | 200 |
| 可见设施 macro / micro F1 | 53.22% / 46.41% | 200 |
| 标签完整度 | 23.50% | 169 |

### 智能售后

| 指标 | 结果 | support |
| --- | ---: | ---: |
| 问题分类准确率 | 60.00% | 150 |
| 严重等级准确率 | 0.00% | 150 |
| 关键信息 macro / micro F1 | 29.67% / 21.02% | 150 |
| OCR 召回率 | 14.22% | 75 |
| OCR exact match | 1.33% | 75 |

### 多模态行程规划

| 指标 | 结果 | support |
| --- | ---: | ---: |
| 约束识别准确率 | 0.00% | 100 |
| 硬约束 macro / micro F1 | 0.00% / 0.00% | 100 |
| 软约束 macro / micro F1 | 0.00% / 0.00% | 100 |
| 行程要素完整度 | 77.20% | 100 |
| 行程要素 macro / micro F1 | 71.09% / 71.81% | 100 |

## standardized v2 严格结构化结果

| 场景 | JSON | Schema | 主要严格业务指标 |
| --- | ---: | ---: | --- |
| 以图搜商品 | 79.00% | 75.00% | 业态准确率 60.00%（support 110）；价位准确率 17.00%（support 100）；风格 macro F1 6.77%；设施 macro F1 3.67% |
| 智能售后 | 96.67% | 96.00% | 问题分类 71.33%；严重度 29.33%；关键信息 macro F1 24.05%；OCR recall 1.33%（support 75） |
| 多模态行程规划 | 90.00% | 88.00% | 约束识别 0.41%；行程要素完整度 20.00%；硬约束 macro F1 1.09%；软约束 macro F1 0% |

baseline 语义指标来自词法编码，standardized v2 指标来自严格 JSON/Schema 结构化评分。两条轨道可以并列展示，但不能把它们的差值归因为纯 Prompt 效果，也没有生成新的 semantic paired comparison 或 bootstrap。

## 典型错误案例

- `image_product_search-62e4e15be59a89c7`：原始文本未唯一命中受控业态，编码返回 `business_category=unknown`，该样本业态得分为 0。说明自然语言描述常省略业务枚举名称。
- `after_sales-0d1b1ae14b0cd002`：文本同时命中“设施损坏”和“服务不可用”词法证据，标量冲突按规则返回 `unknown`。说明简单词法规则无法稳定处理复合描述和隐含严重度。
- `itinerary_planning-b3ebfed1c8435fec`：能识别预算、结束时间、交通及行程要素，但固定规范短语与人工原子约束未精确对齐，约束识别得分为 0。说明词法编码对改写、否定和隐含约束不敏感。

这些案例来自真实 baseline 原始输出和持久化 sample score，不是人工重新编码或挑选的模拟输出。

## 方法限制

- 固定词表对未登记同义词、否定、指代、隐含表达和复合语义不敏感。
- `unknown` 金标按既有指标规则排除，support 如实减少；不补造人工标签。
- OCR 仅处理模型文本中明确出现的可见 token，不能从图片或人工 OCR 金标复制。
- JSON/Schema 合规率仍为原始 baseline 的 0%，语义编码不会把自然语言包装成模型结构化输出。
- 本报告不包含 v3、LLM judge、人工输出复核、微调、训练流程或跨评分轨道的 Prompt 因果归因。

## 验证命令

```bash
python -m unittest discover -s tests -v
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3_v2.yaml
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3_v2.yaml --run-id week3_v2_baseline_full_20260724_001
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3_v2.yaml --run-id week3_v2_standardized_full_20260724_001
```

语义 score 已按不可覆盖规则生成并完成只读验签；同名命令不应再次执行。
