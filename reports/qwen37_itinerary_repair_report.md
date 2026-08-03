# Qwen3.7-Plus 行程规划修复报告

## 结论

在不修改 `week3_evaluation_v2` 金标、历史 Prompt、Schema 和运行产物的前提下，
新增紧凑行程 Prompt `standardized_v4`，并将行程专用输出预算从 1280 提高到
2560 tokens。最终真实 run `itinerary_qwen37_repair_v4_full_20260802_001`
完成 100/100 条请求，无请求错误、截断、JSON 错误或 Schema 错误。

## 根因与修复

- 原 `standardized_v2` 的 67 条无效输出全部在 1280 completion tokens 处截断；
  3 天行程仅 1/33 JSON 有效，4 天行程 0/33 有效。
- `standardized_v3` 通过短字段、空 `source_evidence`、原文约束和简短检查消除
  截断，但 32 条输出把英文枚举翻译成中文，Schema 仅通过 68/100。
- `standardized_v4` 固定 `required_itinerary_elements` 的英文枚举和选择规则，
  最终 JSON 与 Schema 均达到 100/100。

历史 v2、v3 Prompt 和 run 均保留不变。首次 v3 live run `_001` 因本地 Docker
相对密钥路径不适用于仓库根目录而失败，仅作为环境失败证据保留，不计入指标。

## 指标对比

| 指标 | 修复前 v2 | 最终 v4 | 变化 |
| --- | ---: | ---: | ---: |
| JSON 合规率 | 33.00% | 100.00% | +67.00 pp |
| Schema 通过率 | 33.00% | 100.00% | +67.00 pp |
| 约束识别准确率 | 0.14% | 89.95% | +89.81 pp |
| 硬约束 F1 | 0.40% | 96.33% | +95.93 pp |
| 软约束 F1 | 0.00% | 85.67% | +85.67 pp |
| 约束检查覆盖率 | 4.83% | 94.00% | +89.17 pp |
| 行程要素完整度 | 33.00% | 100.00% | +67.00 pp |
| 平均延迟 | 24.72 s | 20.22 s | -4.50 s |
| P95 延迟 | 27.63 s | 25.40 s | -2.23 s |

## 分天数结果

| 请求天数 | 样本数 | JSON | Schema | 平均 completion tokens |
| --- | ---: | ---: | ---: | ---: |
| 2 天 | 34 | 34/34 | 34/34 | 793.7 |
| 3 天 | 33 | 33/33 | 33/33 | 943.7 |
| 4 天 | 33 | 33/33 | 33/33 | 1079.6 |

100 条结果的 `finish_reason` 均为 `stop`，没有输出达到 2560 token 上限。

## 产物与验证

- Prompt：`configs/evaluation/prompts/standardized_v4/`
- 推理配置：`configs/inference_qwen37_plus_itinerary.yaml`
- 评估配置：`configs/evaluation_itinerary_qwen37_repair.yaml`
- run：`data/eval/runs/itinerary_qwen37_repair_v4_full_20260802_001/`
- score：`data/eval/scores/itinerary_qwen37_repair_v4_full_20260802_001/`

run 和 score 保持 Git 忽略；仓库只提交可复现配置、代码、测试和本报告。
