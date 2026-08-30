# 评测与数据隔离

## 三场景契约

系统评测覆盖 `image_product_search`、`after_sales` 和 `itinerary_planning`。场景定义、
Prompt 渲染、Schema 校验、指标和错误分析位于 `src/evaluation/`；入口配置保留在
`configs/evaluation_week3*.yaml` 与 `configs/evaluation/`。

正式 release 使用的 Schema 为：

- `image_product_search_v1.schema.json`
- `after_sales_v1.schema.json`
- `itinerary_planning_v2.schema.json`

Week 3 v1/v2 的 manifest、人工标签、registry、原始输出和评分是历史冻结产物，位于 Git
外 `data/eval/`。不得覆盖，也不得用已消费测试结果继续调参。

## 隔离规则

评测 registry 为每条记录保存稳定来源和图片身份，并生成训练 exclusion manifest。训练或
预标注候选命中 `source_id` 或图片 SHA-256 时必须拒绝；可用时同时检查 `sample_id`、
`group_id` 和 `constraint_template_id`。`unknown` 是证据不足时的合法标签，不应被猜测值替换。

## 批量运行

```bash
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3_v2.yaml
python scripts/run_week3_evaluation.py --config configs/evaluation_week3_v2.yaml --run-id <run-id> --mode dry-run --prompt-version baseline_minimal_v1
python scripts/score_week3_evaluation.py --config configs/evaluation_week3_v2.yaml --run-id <run-id>
```

`mock`、`dry-run` 与 `live` 状态必须分开。只有真实 live 结果可形成模型指标；运行目录、
原始输出、延迟、token、配置哈希和输入身份均不可覆盖。

## 指标

- 商品：业态/价位准确率、风格与设施 F1、完整度、JSON/Schema 合规率。
- 售后：问题分类、严重度、关键信息 F1、OCR recall、JSON/Schema 合规率。
- 行程：约束识别、硬/软约束 recall、行程要素完整度、JSON/Schema 合规率。

所有指标必须同时报告支持数。最简 baseline 的自然语言词法评分与严格结构化评分是不同
轨道，不得直接把差值归因于 Prompt。商品当前参考为模型生成 silver，不能声明人工视觉
准确率；价位正支持不足时保持 `N/A/PENDING`。
