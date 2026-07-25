# Week 4 Prompt 优化报告

## 测试范围

实验只使用不可变的 `week3_evaluation_v2` manifest 和人工金标。每个场景
固定选择 5 个正样本和 2 个边界样本；4-shot 使用 3 个正例和 1 个边界例，
7-shot 使用全部 7 个样本。没有修改 Week 3 标签、Prompt、Schema、运行、
原始输出或评分文件。

模型和生成设置保持为 `Qwen/Qwen2-VL-2B-Instruct`、vLLM、
temperature `0.1`、top-p `0.9`、repetition penalty `1.05` 和
max tokens `1280`。固定 pilot 每场景包含 5 个与示例不重叠的样本。
每次运行在忽略目录 `outputs/week4/` 中保存 Prompt、输入和数据哈希、
示例 ID、拼图哈希、原始输出、解析/Schema 结果、延迟、token usage
和模型设置。

## Pilot 选择

选择分数固定为：

`0.55 * business quality + 0.10 * JSON + 0.20 * Schema + 0.075 * token efficiency + 0.075 * latency efficiency`。

| 场景 | standardized_v2 | 4-shot | 7-shot | 胜出版本 |
| --- | ---: | ---: | ---: | --- |
| 商品理解 | 0.3450 | 0.0400 | 0.2665 | `standardized_v2` |
| 售后理解 | 0.5967 | 0.0800 | 0.5208 | `standardized_v2` |
| 行程规划 | 0.4025 | 0.0750 | 0.0745 | `standardized_v2` |

“胜出”只表示本次候选中的最佳版本。商品场景的 `standardized_v2` 保留了
更高的业务质量和 Schema 合规性。售后 7-shot 的业务质量略高，但 Schema
和 token 效率较低。行程 4-shot、7-shot 均在生成前被后端拒绝，因此
JSON/Schema 合规率为 0。

## 全量胜出版本与 Baseline 对比

全量运行 `week4_winners_full_20260725_001` 已完成 450/450，并与
Week 3 v2 使用相同样本哈希。

| 场景 | 指标口径 | Week 3 baseline | Week 4 胜出版本 |
| --- | --- | ---: | ---: |
| 商品理解 | business quality / JSON / Schema | 0.3570 / 0% / 0% | 0.1565 / 77.5% / 75.5% |
| 售后理解 | business quality / JSON / Schema | 0.2500 / 0% / 0% | 0.2977 / 96.67% / 96.67% |
| 行程规划 | business quality / JSON / Schema | 0.1930 / 0% / 0% | 0.0508 / 90.0% / 87.0% |

业务综合指标差值依次为 -0.2005、+0.0477、-0.1423。商品、售后、行程的
平均 token 为 1211.70、1047.41、1944.39；平均延迟为
13658.11、6036.31、11062.42 ms，P95 为
49830.86、5663.14、52999.28 ms。

Week 3 baseline 业务值来自独立且不读取金标的
`baseline_semantic_coding_v1` 词法轨道。两类评分采用同一已说明字段，
但本报告不把差值解释为纯 Prompt 因果效果，也不覆盖 Week 3 评分文件。

## 审查问题修复

- 新增 `.gitattributes`，并在 provenance 中只归一化文本换行；既有
  LF/CRLF 两类历史运行哈希均可验证，其他字节变化仍会失败。
- Week 3 baseline 和 standardized 两个 run-bound 验证均已恢复为
  `status=ok`，没有修改不可变 Week 3 产物。
- 新增 `scripts/validate_week4_delivery.py`，统一只读检查 3 个 pilot、
  450 条全量运行、artifact/input/sample 哈希、记录数、token usage、
  score、比较和 bad case 产物；当前验证为 `status=ok`。

## 复现命令

```bash
python scripts/run_week4_prompt_evaluation.py --config configs/evaluation_week4.yaml --run-id <pilot-id> --stage pilot --product-variant <variant> --after-sales-variant <variant> --itinerary-variant <variant>
python scripts/analyze_week4_prompts.py --config configs/evaluation_week4.yaml --pilot-run-id <standardized-pilot> --pilot-run-id <4-shot-pilot> --pilot-run-id <7-shot-pilot>
python scripts/run_week4_prompt_evaluation.py --config configs/evaluation_week4.yaml --run-id <full-id> --stage full --product-variant standardized_v2 --after-sales-variant standardized_v2 --itinerary-variant standardized_v2
python scripts/analyze_week4_prompts.py --config configs/evaluation_week4.yaml --full-run-id <full-id>
python scripts/validate_week4_delivery.py --config configs/evaluation_week4.yaml
```
