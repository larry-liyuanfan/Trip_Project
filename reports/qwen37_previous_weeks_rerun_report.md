# Qwen3.7-Plus 前期任务重跑报告

## 结论

2026-08-02 使用阿里云百炼 `qwen3.7-plus`、非思考模式和原 Week 3 v2 人工金标测试集完成重跑。有效 baseline、standardized 和 Week 4 winner run 均为 450/450，模型请求错误为 0，且样本集合 SHA-256 完全一致。

Week 2 数据处理和 Milvus/CLIP 向量结果与生成模型无关，本次未重复执行。Week 1 云端单图接口已通过真实图片 smoke test，返回模型为 `qwen3.7-plus`，未触发本地 fallback。

## 固定条件

| 项目 | 配置 |
| --- | --- |
| 模型 | `qwen3.7-plus` |
| 后端 | Alibaba Cloud Model Studio OpenAI-compatible API |
| 地域 | `ap-southeast-1` |
| thinking | `false` |
| temperature / top_p | 0.1 / 0.9 |
| max_tokens | 1280 |
| 数据集 | `week3_evaluation_v2` |
| 数量 | 商品 200；售后 150；行程 100 |
| 并发 | 4 |

## Week 3 Baseline

有效 run：`week3_qwen37_baseline_full_20260802_002`。

| 场景 | 核心语义指标 | JSON / Schema | 平均 / P95 延迟 |
| --- | --- | --- | --- |
| 商品理解 | 业态准确率 40.00%；价位 10.00%；风格 F1 19.75%；设施 F1 41.58% | 0% / 0% | 14.61 / 19.01 s |
| 智能售后 | 问题分类 69.33%；严重等级 2.67%；关键信息 F1 16.20%；OCR recall 33.33% | 0% / 0% | 12.64 / 20.92 s |
| 行程规划 | 约束识别 0%；行程要素完整度 96.00% | 0% / 0% | 23.86 / 29.76 s |

Baseline 继续遵守最简自然语言指令，因此 0% JSON/Schema 是预期格式结果，不是语义能力为零。语义指标由冻结的 `baseline_semantic_coding_v1` 确定性词法编码生成。

首次全量尝试 `week3_qwen37_baseline_full_20260802_001` 有 1 条 180 秒 ReadTimeout，已保留为失败证据，未用于任何正式指标。

## Week 3 Standardized Prompt

有效 run：`week3_qwen37_standardized_full_20260802_001`。

| 场景 | 核心业务指标 | JSON / Schema | 平均 / P95 延迟 |
| --- | --- | --- | --- |
| 商品理解 | 业态准确率 84.55%；价位 35.00% | 100% / 100% | 7.50 / 13.60 s |
| 智能售后 | 问题分类 92.67%；严重等级 63.33%；OCR recall 100% | 100% / 100% | 8.79 / 14.18 s |
| 行程规划 | 约束识别 0.14%；行程要素完整度 33.00% | 33% / 33% | 25.14 / 30.39 s |

相对旧 Qwen2-VL standardized run，商品和售后的格式及主要分类指标明显提高；行程 JSON/Schema 从 90%/88% 降至 33%/33%，是本次最明确的回归。

## Week 4 Prompt 重选

三个候选在固定 15 条 pilot 上重新比较，未沿用旧模型 winner：

| 场景 | Qwen3.7 胜出版本 | Pilot 选择分数 |
| --- | --- | ---: |
| 商品理解 | `fewshot_4_v2` | 0.4997 |
| 智能售后 | `fewshot_4_v2` | 0.9292 |
| 行程规划 | `standardized_v2` | 0.3250 |

全量 winner run：`week4_qwen37_winners_full_20260802_001`。

| 场景 | 核心业务指标 | JSON / Schema | 平均 / P95 延迟 |
| --- | --- | --- | --- |
| 商品理解 | 业态 77.27%；价位 2.00%；风格 F1 23.95%；设施 F1 43.00% | 100% / 98.5% | 9.91 / 14.15 s |
| 智能售后 | 问题分类 96.00%；严重等级 71.33%；关键信息 F1 98.00%；OCR recall 100% | 100% / 100% | 9.23 / 13.53 s |
| 行程规划 | 约束识别 0.14%；行程要素完整度 33.00% | 33% / 33% | 24.72 / 27.63 s |

真实 bad case 数量：分类错误 31、严重等级错误 43、约束遗漏 100、格式错误 70、字段遗漏或 Schema 错误 3。类别允许重叠。

## 同口径语义比较

共同语义比较 `qwen37_common_semantic_v1_20260802_001` 对 baseline 与 Week 4 winner 的 450 对原始输出使用同一确定性编码器，并执行 2,000 次 paired bootstrap。主要变化如下：

| 指标 | 绝对变化 |
| --- | ---: |
| 商品业态准确率 | +38.18 pp |
| 商品设施 F1 | +6.30 pp |
| 商品风格 F1 | +6.72 pp |
| 商品价位准确率 | -9.00 pp |
| 售后问题分类准确率 | +6.00 pp |
| 售后关键信息 F1 | +20.80 pp |
| 售后严重等级准确率 | -2.67 pp |
| 行程要素完整度 | -13.00 pp |
| 商品 / 售后 / 行程 JSON 合规率 | +100 / +100 / +33 pp |

行程约束指标在固定词法编码下仍为 0，不能据此宣称行程语义能力得到提升。

## 产物

- Week 3 runs：`data/eval/runs/week3_qwen37_*`
- Week 3 scores：`data/eval/scores/week3_qwen37_*`
- Week 3 paired comparison：`data/eval/comparisons/week3_qwen37_prompt_pair_20260802_001/`
- Week 4 runs and scores：`outputs/week4_qwen37_plus/`
- Common semantic：`outputs/week4_qwen37_plus/common_semantic/qwen37_common_semantic_v1_20260802_001/`

运行产物保持 Git 忽略；仓库仅提交配置、运行器兼容代码、测试和本报告。

## 行程修复补充

原行程结果已由版本化 `standardized_v4` 修复，历史 v2 结果保持不变。最终 100 条行程达到 JSON/Schema 100%/100%、约束识别 89.95%、行程要素完整度 100%。完整证据见 `reports/qwen37_itinerary_repair_report.md`。
