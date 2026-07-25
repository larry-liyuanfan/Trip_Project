# Week 4 Bad Case 报告

Bad case 仅从持久化的真实模型失败中导出，分类包括：分类错误、必填字段或
Schema 错误、JSON 格式错误、严重等级错误和约束遗漏。原始输出保留在
不可变的 Week 4 运行目录中；导出器只记录样本 ID 和错误证据，不修复输出，
也不重新标注。

运行 `week4_winners_full_20260725_001` 的统计如下：

| 类别 | 数量 | 代表样本 |
| --- | ---: | --- |
| 分类错误 | 86 | `image_product_search-0ddcb7ef11f23afb` |
| 必填字段或 Schema 错误 | 7 | `image_product_search-5d6823c5422a7bee` |
| JSON 格式错误 | 67 | `image_product_search-c0ed4d164daafeea` |
| 严重等级错误 | 105 | `after_sales-00b16886b9fe85f7` |
| 约束遗漏 | 100 | `itinerary_planning-b3ebfed1c8435fec` |

典型格式错误是字符串截断；典型 Schema 错误包括数组元素重复和
`price_range` 超出枚举。上述为类别计数，同一个样本可以进入多个类别。
机器可读明细位于忽略目录
`outputs/week4/bad_cases/week4_bad_cases_v1.jsonl`，统一只读验证器已核对
269 条 bad case 与全量比较产物中的分类计数一致。
