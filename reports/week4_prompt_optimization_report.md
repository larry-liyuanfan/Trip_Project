# Week 4 Prompt 候选比较报告

## 结论

三场景最终均保留 `standardized_v2`。本周新增的 4-shot、7-shot 候选没有
产生新的胜出 Prompt；“最优”仅表示本次已测试候选中的最高分。

Few-Shot 示例来自最终测试集 `week3_evaluation_v2` 的人工金标。虽然示例
与 15 条 pilot 样本不重叠，但仍使用了最终测试集金标设计 Prompt，存在
test-gold contamination。因此 Few-Shot pilot 只保留为描述性运行证据，
不得用于无偏泛化效果声明。胜出的 `standardized_v2` 不含示例，450 条全量
结果不受该污染直接影响。

## 实验一：最简 baseline 与 standardized_v2

两组运行使用相同的 `week3_evaluation_v2`、450 个 `sample_id`、样本 SHA、
模型和生成参数。格式、Schema 与延迟直接读取原始运行；baseline token 未
保存，记为 `PENDING / 不可获得`。

| 场景 | Baseline JSON/Schema | standardized_v2 JSON/Schema | Baseline 平均/P95 延迟 | standardized_v2 平均/P95 延迟 | Baseline token | standardized_v2 平均 token |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 商品理解 | 0% / 0% | 77.5% / 75.5% | 4456.84 / 7927.61 ms | 13658.11 / 49830.86 ms | `PENDING` | 1211.70 |
| 售后理解 | 0% / 0% | 96.67% / 96.67% | 3616.71 / 6663.71 ms | 6036.31 / 5663.14 ms | `PENDING` | 1047.41 |
| 行程规划 | 0% / 0% | 90.0% / 87.0% | 7136.77 / 14817.89 ms | 11062.42 / 52999.28 ms | `PENDING` | 1944.39 |

### 共同确定性语义轨道

新增独立产物
`week4_common_semantic_coding_v1_20260726_001`，不覆盖 Week 3 原评分。
两组原始输出均先由同一个 `BaselineSemanticCoder.encode`、同一
`baseline_semantic_coding_v1` codebook 转成 canonical prediction；编码
阶段只接收 `scenario` 和 `raw_output`。450 对预测全部生成后才连接同一人工
金标，并使用同一指标函数及 2,000 次 paired bootstrap。

| 场景 | 指标 | Baseline | standardized_v2 | Delta | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| 商品 | 业态准确率 | 45.45% | 78.18% | +32.73 pp | [+22.73, +41.82] pp |
| 商品 | 价位准确率 | 2.00% | 15.00% | +13.00 pp | [+5.00, +21.00] pp |
| 商品 | 风格 macro F1 | 28.28% | 32.21% | +3.93 pp | [-2.29, +10.31] pp |
| 商品 | 设施 macro F1 | 53.22% | 33.33% | -19.88 pp | [-28.32, -11.33] pp |
| 售后 | 问题分类准确率 | 60.00% | 78.00% | +18.00 pp | [+8.00, +26.67] pp |
| 售后 | 严重等级准确率 | 0.00% | 0.00% | 0.00 pp | [0.00, 0.00] pp |
| 售后 | 关键信息 macro F1 | 29.67% | 30.33% | +0.67 pp | [-6.00, +6.67] pp |
| 售后 | OCR recall | 14.22% | 4.44% | -9.78 pp | [-14.22, -4.89] pp |
| 行程 | 约束识别准确率 | 0.00% | 0.00% | 0.00 pp | [0.00, 0.00] pp |
| 行程 | 行程要素完整度 | 77.20% | 21.80% | -55.40 pp | [-60.00, -50.60] pp |
| 行程 | 行程要素 macro F1 | 71.09% | 29.54% | -41.54 pp | [-44.83, -38.03] pp |

这些 delta 只表示固定词法编码器下的共同轨道结果。codebook 原本为自然语言
baseline 设计，例如严重度规则要求“severity high”等短语，不能稳定识别
JSON 中被标点分隔的 canonical value；它也不处理改写、否定、指代或隐含
约束。因此共同轨道解决了“评分入口不同”的问题，但不能替代盲法人工语义
编码，也不能把所有 delta 解释为模型真实语义能力变化。

原有两条业务评分轨道仍保留：Week 3 baseline 词法评分与 Week 4 严格
JSON/Schema 评分不直接相减。共同轨道是新增的、单独版本化的比较证据。

## 实验二：standardized_v2 与 Few-Shot

模型和生成设置保持为 `Qwen/Qwen2-VL-2B-Instruct`、vLLM、
temperature `0.1`、top-p `0.9`、repetition penalty `1.05`、
max tokens `1280`。有效运行均为 15/15，模型请求错误为 0。

| 场景 | standardized_v2 | 4-shot v2 | 7-shot v2 | 描述性最高分 |
| --- | ---: | ---: | ---: | --- |
| 商品理解 | 0.3280 | 0.1133 | 0.2419 | `standardized_v2` |
| 售后理解 | 0.5967 | 0.4848 | 0.4492 | `standardized_v2` |
| 行程规划 | 0.4775 | 0.0190 | 0.0015 | `standardized_v2` |

旧 `fewshot_4_v1`、`fewshot_7_v1` 的行程请求因超过 4096-token 上限而
全部 HTTP 400，保留为无效历史运行。压缩后的 v2 候选消除了请求错误，但
行程 4-shot/7-shot 的 JSON/Schema 仍均为 0%，平均 token 分别为
2865.8/3217.8，平均延迟为 42395.66/41648.37 ms。

以上分数只能描述这组受污染 pilot 上的表现，不能据此估计 Few-Shot 的无偏
泛化收益。由于当前范围禁止新增人工标注或数据集版本，本次不构造新的
demo/dev pool，也不重跑模型。

## 验证与产物

- 统一验证器确认 45 条有效 pilot、450 条全量、0 个模型请求错误。
- 共同语义轨道确认 450 对 canonical prediction、450 对 sample score、
  38 个聚合指标和 2,000 次 bootstrap 配置。
- 两组原始输出 SHA-256、codebook SHA-256
  `563dc074...033a6`、样本 SHA-256 `3e900e64...ad648c` 均已绑定。
- 原始运行、Week 3 评分和 v1/v2 历史比较产物均未覆盖。

复现命令：

```bash
python scripts/compare_week4_common_semantics.py
python scripts/validate_week4_delivery.py --config configs/evaluation_week4.yaml
```
