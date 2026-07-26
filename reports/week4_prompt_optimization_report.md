# Week 4 Prompt 候选比较报告

## 结论

状态：`READY FOR MENTOR REVIEW`。

独立 `week4_demo_dev_v1` 包含 36 条人工金标，每场景 12 条；从中固定选择
5 个典型正例和 2 个边界例。整个 development 池与
`week3_evaluation_v2` 的 450 条 evaluation 数据在 `sample_id`、
`source_id`、图片 SHA-256 和来源组四层无重叠。4-shot 使用 3 正例 +
1 边界例，7-shot 使用 5 正例 + 2 边界例。

固定 15 条 pilot 的胜出版本为：

- 商品理解：`fewshot_4_v2`
- 售后理解：`standardized_v2`
- 行程规划：`standardized_v2`

“胜出”只表示预先约定综合分在本次三个候选中最高。商品 4-shot 的 pilot
业务质量和 Schema 合规率并不高，其胜出主要来自该 5 条 pilot 上的 token
和延迟优势；450 条全量的 Schema 合规率仅 20.5%。因此不能把它解释为
商品业务效果稳定提升，也不能宣称生产级最优。

## 实验一：独立 demo/dev Few-Shot pilot

模型与运行条件固定为 `Qwen/Qwen2-VL-2B-Instruct`、vLLM、
temperature `0.1`、top-p `0.9`、repetition penalty `1.05`、
max tokens `1280`。三次 pilot 使用完全相同的 15 个 evaluation 样本，
每场景 5 条；45/45 请求完成，`model_request_error_count=0`。

| 场景 | 候选 | 业务质量 | JSON | Schema | 平均 token | 平均延迟 ms | 综合分 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 商品 | `standardized_v2` | 0.1327 | 60% | 60% | 1384.2 | 20332.43 | 0.2530 |
| 商品 | `fewshot_4_v2` | 0.0000 | 100% | 20% | 1189.8 | 2398.33 | **0.2900** |
| 商品 | `fewshot_7_v2` | 0.0960 | 100% | 20% | 1356.6 | 2828.77 | 0.2766 |
| 售后 | `standardized_v2` | 0.3067 | 100% | 100% | 1018.4 | 3389.45 | **0.5624** |
| 售后 | `fewshot_4_v2` | 0.3200 | 100% | 40% | 1415.0 | 3140.87 | 0.4507 |
| 售后 | `fewshot_7_v2` | 0.2000 | 100% | 20% | 1556.2 | 3472.14 | 0.2500 |
| 行程 | `standardized_v2` | 0.0500 | 100% | 100% | 1828.8 | 4638.43 | **0.4775** |
| 行程 | `fewshot_4_v2` | 0.0000 | 0% | 0% | 2856.8 | 39903.27 | 0.0154 |
| 行程 | `fewshot_7_v2` | 0.0000 | 20% | 0% | 3121.4 | 38075.69 | 0.0239 |

旧的 test-gold Few-Shot v1/v2 运行仅作为历史证据保留，不参与本表选择。
本次示例选择文件为
`configs/evaluation/week4_prompt_selection_v2.json`。

## 实验二：胜出映射全量结果

全量运行 `week4_winners_full_20260726_002` 使用冻结的 450 条 Week 3 v2
样本，样本 SHA-256 为
`3e900e64bb345df35343c8f14bfb1f8310ae597a57e4a4d9585bc01173ad648c`。
450/450 请求完成，模型请求错误为 0。

| 场景 | 胜出版本 | JSON | Schema | 平均 token | 平均/P95 延迟 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 商品理解 | `fewshot_4_v2` | 82.0% | 20.5% | 1450.41 | 11001.37 / 47828.04 |
| 售后理解 | `standardized_v2` | 96.67% | 96.67% | 1048.17 | 5789.27 / 6188.43 |
| 行程规划 | `standardized_v2` | 91.0% | 88.0% | 1929.67 | 9513.91 / 47049.29 |

商品全量明显暴露了 5 条 pilot 的方差：大量输出遗漏必填 `confidence`，
导致 JSON 可解析但 Schema 不合规。该负结果保留，不使用全量测试结果反向
改选 Prompt。

## 实验三：Week 3 baseline 与 winner 的共同语义轨道

直接可比的格式和性能如下。Baseline 未保存 token，准确标为
`PENDING / 不可获得`。

| 场景 | Baseline JSON/Schema | Winner JSON/Schema | Baseline 平均/P95 延迟 ms | Winner 平均/P95 延迟 ms | Baseline token | Winner 平均 token |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 商品 | 0% / 0% | 82.0% / 20.5% | 4456.84 / 7927.61 | 11001.37 / 47828.04 | `PENDING` | 1450.41 |
| 售后 | 0% / 0% | 96.67% / 96.67% | 3616.71 / 6663.71 | 5789.27 / 6188.43 | `PENDING` | 1048.17 |
| 行程 | 0% / 0% | 91.0% / 88.0% | 7136.77 / 14817.89 | 9513.91 / 47049.29 | `PENDING` | 1929.67 |

共同轨道
`week4_common_semantic_coding_v1_20260726_003` 不覆盖 Week 3 原评分。
两组原始输出都先由同一 `baseline_semantic_coding_v1` codebook 转成
canonical prediction；编码器只接收 `scenario` 和 `raw_output`。全部预测
完成后才连接同一人工金标，并使用同一指标函数和 2,000 次 paired
bootstrap。

| 场景 | 指标 | Baseline | Winner | Delta | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| 商品 | 业态准确率 | 45.45% | 73.64% | +28.18 pp | [+18.18, +37.27] |
| 商品 | 价位准确率 | 2.00% | 0.00% | -2.00 pp | [-5.00, 0.00] |
| 商品 | 风格 macro F1 | 28.28% | 13.17% | -15.12 pp | [-22.18, -8.38] |
| 商品 | 设施 macro F1 | 53.22% | 25.17% | -28.05 pp | [-35.48, -20.03] |
| 售后 | 问题分类准确率 | 60.00% | 78.67% | +18.67 pp | [+9.33, +27.33] |
| 售后 | 严重等级准确率 | 0.00% | 0.00% | 0.00 pp | [0.00, 0.00] |
| 售后 | 关键信息 macro F1 | 29.67% | 30.33% | +0.67 pp | [-5.33, +6.67] |
| 售后 | OCR recall | 14.22% | 4.89% | -9.33 pp | [-14.22, -4.44] |
| 行程 | 约束识别准确率 | 0.00% | 0.00% | 0.00 pp | [0.00, 0.00] |
| 行程 | 行程要素完整度 | 77.20% | 20.60% | -56.60 pp | [-61.00, -52.20] |
| 行程 | 行程要素 macro F1 | 71.09% | 28.47% | -42.61 pp | [-45.63, -39.31] |

这些 delta 只表示固定词法编码器下的共同轨道结果。该 codebook 原为自然
语言 baseline 设计，对 JSON 标点、改写、否定和隐含约束识别有限，不能
替代盲法人工语义编码，也不能把所有差值解释为 Prompt 的因果效果。

## 验证与复现

- 统一验证器确认 45 条有效 pilot、450 条全量、0 个模型请求错误。
- 共同语义轨道确认 450 对预测、38 个聚合指标和 2,000 次 bootstrap。
- 独立 development 池为 36/36 人工完成；选入示例 21 条，未新增最终
  evaluation 样本。
- 原始运行、Week 3 评分、旧选择文件和历史比较产物均未覆盖。

```bash
python scripts/compare_week4_common_semantics.py \
  --winner-run-id week4_winners_full_20260726_002 \
  --output-dir outputs/week4/common_semantic/week4_common_semantic_coding_v1_20260726_003
python scripts/validate_week4_delivery.py --config configs/evaluation_week4.yaml
```
