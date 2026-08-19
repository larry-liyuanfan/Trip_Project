# Week 6 Qwen3-VL-8B QLoRA 质量报告

## 结论

Week 6 的工程训练链、行程专项非回退门禁和一次性冻结评测均已完成。三场景最终运行
覆盖冻结的 `week3_evaluation_v2` 全部 450 条记录，没有把冻结集用于训练、early
stopping 或反复调参。商品的 JSON/Schema 均为 100%；售后 JSON 为 100%、Schema
为 96.67%；行程 JSON 为 95%、Schema 为 85%。因此本轮可判定为“训练与评测闭环
完成”，但不能把所有业务指标描述为优秀：商品设施/风格、售后严重度与关键信息、行程
约束识别仍有明确提升空间。

## 训练与获胜 adapter

固定消息规范化链使用提交
`3d6bc81df8c4afd496e1e78d41c6b4bfa07c7bf4`。pilot/gate/商品/售后/行程作业
`29312210`/`29312212`/`29312214`/`29312215`/`29312217` 均为
`COMPLETED 0:0`。三场景均为 NF4 double quant、bf16、LoRA
`r=16/alpha=32/dropout=0.05`，只保存 adapter，并完成磁盘回载。

| 场景 | 原训练最佳 checkpoint | eval loss | 最终评测 adapter SHA-256 |
| --- | --- | ---: | --- |
| 商品 | `checkpoint-5930` | 0.2234927863 | `8b57961542a5fd82767b6e71a470b08cecd7ed4c2a8c14beda0480b5acaa5df0` |
| 售后 | `checkpoint-2856` | 0.0083342027 | `36bcc21bedbb9aad4f58cb0cd1870d97a5d5e83aca2f0580d4e3750b298a2160` |
| 行程原版 | `checkpoint-1620` | 0.0056819413 | `18c5dfad0a423945f19b0d1ea863e82bda3934634aa4b5922023c3421ba114ac` |
| 行程专项获胜版 | `checkpoint-540` | 0.0013725368 | `7ab168a0f7073f2fad3369c028f744585362a0668f77c024098d9b27d92c9a6a` |

行程原 validation 的结构全通过为 `0/450`，所以低 loss 不能独立证明业务效果。专项
训练 `29375367` 从已验证的原 adapter 继续，在不可覆盖派生 silver 锁上完成
1791 steps/3 epochs，`train_loss=0.0016935773`。固定 64 条对照中，原 adapter 的
全通过为 0；获胜候选九项计数均为 64/64。门禁作业 `29412603` 返回
`status=passed`、`reasons=[]`，因此停止调参并锁定专项 adapter。

## 参数锁定后的独立最终评测

评测提交为 `bce0d8790f14e287657b21cfb0d9a1bd87ba770b`。冻结数据 validator
`29418805` 验证商品/售后/行程 `200/150/100`、总计 450/450；CPU preflight
`29418839` 完成 77 项定向测试、离线模型缓存和三个 adapter 哈希检查。三个 GPU 作业
严格串行，没有提交竞争作业。

| 场景 | Job / 用时 | 样本 | JSON | Schema | 平均延迟 |
| --- | --- | ---: | ---: | ---: | ---: |
| 商品 | `29418875` / 13:56 | 200 | 100% | 100% | 3973.41 ms |
| 售后 | `29419327` / 09:54 | 150 | 100% | 96.67% | 3669.90 ms |
| 行程 | `29422130` / 39:53 | 100 | 95% | 85% | 23502.63 ms |

### 商品理解

- 业态准确率 86.36%，support 110；价位准确率 46.00%，support 100。
- 标签完整度 33.07%，support 169。
- 设施 macro/micro F1 为 5.02%/4.29%；风格 macro/micro F1 为
  10.04%/10.93%。
- 200 条均通过 JSON 和 Schema，错误记录为 0；主要局限在多标签语义，而不是输出格式。

### 售后理解

- 问题类型准确率 86.67%，严重度准确率 34.67%。
- 关键信息 macro/micro F1 为 17.42%/31.71%。
- OCR exact match/recall 为 98.67%/99.56%，两者 support 均为 75。
- 150 条 JSON 全通过，145 条通过 Schema；5 条记录为 Schema validation error。

### 行程规划

- 约束识别准确率 30.33%，约束检查覆盖率 48.83%，违规率 0%。
- 硬约束 macro/micro F1 为 46.28%/50.91%；软约束为 29.33%/31.43%。
- 行程要素完整度 85.00%，要素 macro/micro F1 为 74.27%/80.49%。
- 100 条中 JSON 95、Schema 85；错误分类含 5 条 JSON parse error 和 10 条
  Schema validation error。专项门禁证明候选相对原 adapter 在派生结构集上不回退，
  但冻结人工集表明泛化仍未达到“优秀”。

## 方法与边界

本轮采用 QLoRA/NF4、LoRA adapter-only、best-checkpoint 回载、固定 holdout、严格
provenance 和非回退业务门禁。它们与参数高效微调和可靠评测的通行方法一致；最终冻结
集出现的弱项只记录为局限，不再据此调参。若后续导师授权新版本，应先从人工错误切片、
训练目标与 Schema 对齐、约束感知数据设计和多标签损失入手，并建立新的 development
集合；不得复用本次冻结结果进行选择。

## 可复现证据

- 冻结数据：`week3_evaluation_v2`，exclusion 450，validator `status=ok`。
- 运行：`week6_final_image_product_search_20260819_bce0d87_a`、
  `week6_final_after_sales_20260819_bce0d87_a`、
  `week6_final_itinerary_planning_20260819_bce0d87_a`。
- 原始 `metadata.json`、`results.jsonl`、逐样本评分、聚合评分和错误切片保留在忽略的
  `outputs/week6/final_evaluation/`，交付归档使用 SHA-256 覆盖清单。
- Spartan 最终评测时项目文件系统约余 35 GiB；三个 GPU 作业均由调度器正常回收，
  没有人工提前释放健康 allocation。
