# Week 3 v2 测试集重整与运行状态（历史快照，已被后续结果取代）

截至 2026-07-24，`week3_evaluation_v2` 的数据重整、人工标注和两组全量
真实推理均已完成。该文件只记录 v2 过程状态；正式指标、错误案例和结论见
`reports/week3_zero_shot_baseline_report.md`。

> 状态说明：本文末尾的 `PARTIAL / PENDING` 是 2026-07-24 当时的历史
> 状态。2026-07-25 完成获批的 `baseline_semantic_coding_v1` 后，Week 3
> 正式状态已更新为 `READY / COMPLETED`；以正式报告为准。

| 场景 | target | candidate | annotated | validated | tested |
| --- | ---: | ---: | ---: | ---: | ---: |
| 以图搜商品 | 200 | 200 | 200 | 200 | 200 |
| 智能售后 | 150 | 150 | 150 | 150 | 150 |
| 多模态行程规划 | 100 | 100 | 100 | 100 | 100 |

完成证据：

- v1 manifests、runs、Prompt 和 Schema 均保持不可变；v2 使用独立文件名和
  450 行 exclusion registry。
- 70 条低质量售后候选已替换为经人工查看的真实感证据图并完成标注；最终人工
  金标覆盖卫生污渍 42、设施损坏 34、景点关闭 37、交通延误 37。
- 100 条行程记录继承原有文本、硬/软约束和必要要素，只补充此前未可靠暴露的
  图片风格字段；99 条非空，1 条证据支持为空。
- baseline run `week3_v2_baseline_full_20260724_001` 与 standardized run
  `week3_v2_standardized_full_20260724_001` 均为 completed/live/full，
  selected_count 与 record_count 均为 450，模型请求错误均为 0。
- 两组运行的样本集合 SHA-256 都是
  `3e900e64bb345df35343c8f14bfb1f8310ae597a57e4a4d9585bc01173ad648c`。
- 比较产物 `week3_v2_prompt_pair_20260724_001` 包含 450 个成对样本和
  2,000 次 bootstrap。

## 后续状态

上述快照在 2026-07-24 仍为 `PARTIAL`：当时最简 baseline 自然语言输出
尚无获批的确定性解析结果。2026-07-25 随后新增
`baseline_semantic_coding_v1`，在不修改原始运行、Prompt、金标或严格
JSON/Schema 结果的前提下完成 450 条语义编码与评分。因此该历史
`PARTIAL / PENDING` 已被正式报告中的 `READY / COMPLETED` 取代。
