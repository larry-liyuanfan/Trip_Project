# Week 3 零样本基线与标准化 Prompt 评测报告

## 交付结论

`week3_evaluation_v2` 已完成 450 条人工标注、评测隔离、全量最简
baseline、全量 standardized v2、严格评分和同样本成对比较。两次运行均为
`completed/live/full`，各保留 450 条输入、模型原始输出、延迟和请求元数据。

Week 3 状态保持 `PARTIAL`：最简 baseline 按导师要求不含 JSON 或格式约束，
450 条输出均为自然语言。当前没有获批的确定性语义解析器或人工编码结果，
所以 baseline 的分类、OCR 和约束语义指标必须保持 `PENDING`，不能把格式解析
失败解释为语义得分 0。其余数据、运行、格式指标和 standardized v2 严格业务
指标均有持久化证据。

## 数据与五类数量

| 场景 | target | candidate | annotated | validated | tested |
| --- | ---: | ---: | ---: | ---: | ---: |
| 以图搜商品 | 200 | 200 | 200 | 200 | 200 |
| 智能售后 | 150 | 150 | 150 | 150 | 150 |
| 多模态行程规划 | 100 | 100 | 100 | 100 | 100 |

- 商品采样分层为酒店 67、景点 67、餐饮 66。
- 售后采样分层为卫生污渍 38、设施损坏 38、景点关闭 37、交通延误 37；
  人工金标为卫生污渍 42、设施损坏 34、景点关闭 37、交通延误 37。
- 售后来源包含 `public_yelp=6` 和 `business_synthetic=144`，满足公开场景与
  业务合成样本并存；不将采样分层冒充人工金标。
- 行程 100 组均包含参考图与文本约束；99 条具有至少一个人工风格偏好，
  1 条空数组是允许的无可确认风格结果。
- v2 exclusion registry 包含 450 个唯一 source/image hash。验证器拒绝跨场景
  `source_id`、图片 SHA-256 冲突以及与训练候选的碰撞。

## 运行验签

| 产物 | 值 |
| --- | --- |
| dataset | `week3_evaluation_v2` |
| model | `Qwen/Qwen2-VL-2B-Instruct` |
| baseline run | `week3_v2_baseline_full_20260724_001` |
| standardized run | `week3_v2_standardized_full_20260724_001` |
| comparison | `week3_v2_prompt_pair_20260724_001` |
| 每次 selected/record | 450/450 |
| request errors | 0/450，两个 run 均相同 |
| 样本集合 SHA-256 | `3e900e64bb345df35343c8f14bfb1f8310ae597a57e4a4d9585bc01173ad648c` |
| bootstrap | 2,000 次 |

两次运行使用同一数据版本、样本 ID 集合与顺序、图片和文本输入、模型配置及
非 Prompt 数据资产。Prompt、Schema 和运行目录均版本化并保留各自 SHA-256；
v1 manifest、Prompt、Schema 和历史 run 未被覆盖。

## 最简零样本 baseline

| 场景 | N | JSON 合规 | Schema 通过 | 平均延迟 | P95 延迟 | 原生语义指标 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 以图搜商品 | 200 | 0% | 0% | 4,456.84 ms | 7,927.61 ms | PENDING |
| 智能售后 | 150 | 0% | 0% | 3,616.71 ms | 6,663.71 ms | PENDING |
| 多模态行程规划 | 100 | 0% | 0% | 7,136.77 ms | 14,817.89 ms | PENDING |

三条 `baseline_minimal_v1` 均保持原始单句任务描述，不含角色设定、字段表、
JSON/格式约束、示例或思维链引导。450/450 条输出触发
`json_parse_error`，证明无格式约束时输出不可直接由后端结构化解析；这不证明
模型的业务分类、OCR 或约束理解能力为 0。评分产物将这些语义字段写为
`null`，并以 `format_only_unparsed_baseline` 标记评分轨道。

## standardized v2 严格业务结果

| 场景 | JSON | Schema | 主要受支持指标 |
| --- | ---: | ---: | --- |
| 以图搜商品 | 79.00% | 75.00% | 业态准确率 60.00%（support=110）；价位准确率 17.00%（support=100）；风格 macro F1 6.77%；设施 macro F1 3.67%；标签完整度 19.54%（support=174） |
| 智能售后 | 96.67% | 96.00% | 问题分类准确率 71.33%；严重度准确率 29.33%；关键信息 macro F1 24.05%；OCR exact/recall 1.33%（support=75） |
| 多模态行程规划 | 90.00% | 88.00% | 约束识别准确率 0.41%；行程要素完整度 20.00%；硬约束 macro F1 1.09%；软约束 macro F1 0% |

standardized v2 使用四层架构：系统角色、任务指令、输入上下文、输出约束；
请求包含有序图片 content parts、文字约束、完整 Schema 契约和仅 JSON 输出
要求。可观察证据、字段来源和 `constraint_check` 用于可验证检查，不要求或
保存模型内部推理过程。

## 成对比较

| 场景 | 指标 | baseline | standardized | 绝对变化 | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| 商品 | JSON 合规 | 0% | 79.00% | +79.00 pp | [+73.00, +84.50] pp |
| 商品 | Schema 通过 | 0% | 75.00% | +75.00 pp | [+69.00, +80.50] pp |
| 售后 | JSON 合规 | 0% | 96.67% | +96.67 pp | [+93.33, +99.33] pp |
| 售后 | Schema 通过 | 0% | 96.00% | +96.00 pp | [+92.67, +98.67] pp |
| 行程 | JSON 合规 | 0% | 90.00% | +90.00 pp | [+84.00, +95.00] pp |
| 行程 | Schema 通过 | 0% | 88.00% | +88.00 pp | [+81.00, +94.00] pp |

标准化平均延迟相对 baseline 增加：商品 6,112.19 ms、售后 722.41 ms、行程
793.56 ms。商品延迟差的 bootstrap 95% CI 为
[4,089.98, 8,265.61] ms；售后与行程的区间跨 0，不能据此断言稳定回归。
由于 baseline 语义指标未解析，成对比较只对格式、Schema 和延迟作归因，不把
standardized 的语义得分与不存在的 baseline 语义分数比较。

## 典型错误与能力边界

standardized v2 的 68 条严格失败由实际评分产物导出：

| 场景 | JSON 解析失败 | Schema 失败 | 主要原因 |
| --- | ---: | ---: | --- |
| 商品 | 42 | 8 | 长输出在字符串中截断；重复 style；evidence 超限；价位枚举越界 |
| 售后 | 5 | 1 | 字符串截断；单条 evidence 超长 |
| 行程 | 10 | 2 | 字符串截断；返回外层包装对象导致核心字段缺失 |

证据支持的当前短板：

- 小模型在复杂结构下会生成过长内容并在 JSON 字符串中截断，商品最明显。
- 商品风格、设施和价位的严格语义指标偏低，说明受控枚举映射和只依据可观察
  证据输出仍不稳定。
- 售后问题分类明显好于严重度、关键信息和 OCR；OCR 仍是最弱子任务。
- 行程能够较高比例生成 Schema 合规结构，但约束识别和约束集合匹配很低，
  说明“结构正确”不等于“约束理解正确”。

这些结论只来自实际原始输出和评分产物。baseline 原生语义能力仍未被当前方法
量化，不生成相应准确率或深层能力结论。

## 导师要求内的优化方向

- 保持 baseline 原文不变，用其 0% 格式合规和实测延迟作为未优化基准线。
- 标准化 Prompt 继续约束短证据、去重枚举和紧凑 JSON，重点减少截断及
  `maxItems`/`maxLength` 违规。
- 商品侧优先加强风格、设施、价位的证据到枚举映射；售后侧优先加强严重度、
  关键信息和 OCR；行程侧优先加强硬/软约束拆分、逐项覆盖和最终一致性检查。
- 后续任何模型微调或 Prompt 改动都应复用同一隔离测试集、样本顺序、评分规则
  和持久化产物，与本报告基准成对比较。

## 验证与生成命令

```bash
# 以下三条为当前不可变产物的只读验证命令。
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3_v2.yaml
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3_v2.yaml --run-id week3_v2_baseline_full_20260724_001
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3_v2.yaml --run-id week3_v2_standardized_full_20260724_001
python -m unittest discover -s tests -v
```

评分和比较命令用于首次生成产物，遵循不可覆盖规则：对应
`data/eval/scores/<run_id>` 或 `data/eval/comparisons/<comparison_id>` 目录必须
尚不存在。当前命名产物已经生成并验签，因此不应在现有工作区重复执行；如需从
保留的 run 在独立环境重建，应先确保目标输出目录为空，并保持原始 run、配置和
哈希资产不变。
