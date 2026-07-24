# Week 3 v2 测试集重整与运行状态

截至 2026-07-24，`week3_evaluation_v2` 的数据重整、人工标注和两组全量
真实推理均已完成。该文件只记录 v2 过程状态；正式指标、错误案例和结论见
`reports/week3_zero_shot_baseline_report.md`。

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

Week 3 仍标为 `PARTIAL`，唯一未量化部分是最简 baseline 自然语言输出的原生
语义指标。450 条 baseline 输出和延迟已完整留存，但没有获批的确定性解析或
人工编码结果；因此这些指标为 `PENDING`，不是数值 0。
