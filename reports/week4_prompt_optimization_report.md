# Week 4 Prompt 候选比较报告

## 测试范围

实验只使用不可变的 `week3_evaluation_v2` manifest 和人工金标。每个场景
固定选择 5 个正样本和 2 个边界样本；4-shot 使用 3 个正例和 1 个边界例，
7-shot 使用全部 7 个样本。没有修改 Week 3 标签、Prompt、Schema、运行、
原始输出或评分文件。

模型和生成设置保持为 `Qwen/Qwen2-VL-2B-Instruct`、vLLM、
temperature `0.1`、top-p `0.9`、repetition penalty `1.05` 和
max tokens `1280`。固定 pilot 每场景包含 5 个与示例不重叠的样本。

旧 `fewshot_4_v1` 和 `fewshot_7_v1` 的行程请求因上下文超过模型
4096-token 上限而全部返回 HTTP 400，这两组历史运行保留但不再作为有效
候选证据。版本化的 `fewshot_4_v2` 和 `fewshot_7_v2` 保留原有示例数量、
图片拼图、金标字段、模型和生成参数，仅删除示例中重复展开的长行程及完整
Schema 文本；最终输出仍按现有 Schema 校验。

## 有效 Pilot 选择

有效运行如下：

- `week4_pilot_standardized_v2_20260725_001`
- `week4_pilot_fewshot4_v2_20260725_001`
- `week4_pilot_fewshot7_v2_20260725_001`

三组运行均为 15/15，`model_request_error_count=0`。选择分数固定为：

`0.55 * business quality + 0.10 * JSON + 0.20 * Schema + 0.075 * token efficiency + 0.075 * latency efficiency`。

| 场景 | standardized_v2 | 4-shot v2 | 7-shot v2 | 胜出版本 |
| --- | ---: | ---: | ---: | --- |
| 商品理解 | 0.3280 | 0.1133 | 0.2419 | `standardized_v2` |
| 售后理解 | 0.5967 | 0.4848 | 0.4492 | `standardized_v2` |
| 行程规划 | 0.4775 | 0.0190 | 0.0015 | `standardized_v2` |

行程 4-shot v2 和 7-shot v2 已进入真实模型生成，不再是 HTTP 400；
两者实测 JSON/Schema 均为 0%，平均 token 分别为 2865.8 和 3217.8，
平均延迟分别为 42395.66 和 41648.37 ms。新增 Few-Shot 候选未超过
控制组，因此本次没有产生新的胜出 Prompt；三个场景均继续使用原有
`standardized_v2`。“胜出”只表示本次候选中的最佳版本。

## 与 Week 3 Baseline 的同口径比较

全量胜出运行 `week4_winners_full_20260725_001` 已完成 450/450，并与
Week 3 v2 使用相同样本哈希。这里只比较两边口径相同的原始 JSON、
Schema 和延迟。Week 3 baseline 运行未保存 token usage，因此明确记为
`PENDING / 不可获得`。

| 场景 | Baseline JSON/Schema | Week 4 JSON/Schema | Baseline 平均/P95 延迟 | Week 4 平均/P95 延迟 | Baseline token | Week 4 平均 token |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 商品理解 | 0% / 0% | 77.5% / 75.5% | 4456.84 / 7927.61 ms | 13658.11 / 49830.86 ms | `PENDING` | 1211.70 |
| 售后理解 | 0% / 0% | 96.67% / 96.67% | 3616.71 / 6663.71 ms | 6036.31 / 5663.14 ms | `PENDING` | 1047.41 |
| 行程规划 | 0% / 0% | 90.0% / 87.0% | 7136.77 / 14817.89 ms | 11062.42 / 52999.28 ms | `PENDING` | 1944.39 |

## 业务评分边界

Week 3 baseline 使用 `baseline_semantic_coding_v1` 确定性词法编码，
Week 4 使用结构化 JSON 严格评分。两者预测编码方式不同，不属于同一业务
评分轨道，因此不计算、不展示 `business_quality_delta`，也不据此声称
Prompt 带来业务提升。

Week 4 结构化轨道自身的业务综合值为商品 0.1565、售后 0.2977、行程
0.0508；Week 3 词法轨道的逐项指标继续以
`reports/week3_zero_shot_baseline_report.md` 为准。两组数值只在各自
轨道内解释。

## 审查修复

- runner 现在只要出现模型请求失败，就将整次运行标为 `failed`；错误响应体
  会保留在明确错误信息中。
- 统一验证器拒绝任何包含 `model_request_error` 的 pilot/full run，并检查
  九个候选摘要的请求错误计数必须为 0。
- 新增有效 Few-Shot v2 运行和 v2 比较产物；旧 HTTP 400 运行保持不变。
- baseline 比较产物只保留同口径格式、延迟和 token 可用性，不再计算跨轨道
  业务差值。
- `scripts/validate_week4_delivery.py` 当前验证为 `status=ok`。

## 复现命令

```bash
python scripts/run_week4_prompt_evaluation.py --config configs/evaluation_week4.yaml --run-id <pilot-id> --stage pilot --variant <standardized_v2|fewshot_4_v2|fewshot_7_v2>
python scripts/analyze_week4_prompts.py --config configs/evaluation_week4.yaml --pilot-run-id week4_pilot_standardized_v2_20260725_001 --pilot-run-id week4_pilot_fewshot4_v2_20260725_001 --pilot-run-id week4_pilot_fewshot7_v2_20260725_001
python scripts/analyze_week4_prompts.py --config configs/evaluation_week4.yaml --full-run-id week4_winners_full_20260725_001
python scripts/validate_week4_delivery.py --config configs/evaluation_week4.yaml
```
