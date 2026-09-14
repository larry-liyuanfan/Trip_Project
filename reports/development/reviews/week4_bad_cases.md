# Week 4 Bad Case 报告

Bad case 仅从 `week4_winners_full_20260726_002` 的真实输出、既有 Week 3 v2 人工金标和同一评分产物导出，不修改输出或标签。

| 类别 | 数量 | 代表样本 |
| --- | ---: | --- |
| 分类错误 | 144 | `image_product_search-0ddcb7ef11f23afb` |
| 必填字段或 Schema 错误 | 126 | `image_product_search-2885a3c873d64e73` |
| 格式错误 | 176 | `image_product_search-2885a3c873d64e73` |
| 严重等级错误 | 106 | `after_sales-00b16886b9fe85f7` |
| 约束遗漏 | 100 | `itinerary_planning-b3ebfed1c8435fec` |

同一样本可以进入多个类别；机器可读 v5 明细共 376 条，位于忽略目录 `outputs/week4/bad_cases/week4_bad_cases_v5.jsonl`。

## 典型真实案例

### 1. 分类错误

- **样本：** `image_product_search-0ddcb7ef11f23afb`
- **金标：** 业态 `attraction`；价位 `budget`；风格 `casual,resort`；设施 `outdoor_seating`
- **模型预测：** 业态 `restaurant`；价位 `mid_range`；风格 `casual,artistic`；设施 `bar,restaurant`
- **错误原因：** 4-shot 模板化复制了 development 示例属性，未依据当前图片区分景点与餐厅。

### 2. Schema 错误

- **样本：** `image_product_search-2885a3c873d64e73`
- **金标：** 业态 `unknown`；价位 `mid_range`；风格 `modern,business`；设施 `bar`
- **模型预测：** JSON 可解析，但遗漏必填 `confidence`，并预测为 `restaurant`
- **错误原因：** Few-Shot 助手示例只展示业务字段，模型在最终完整 Schema 中遗漏尾部必填字段。

### 3. 格式错误

- **样本：** `image_product_search-2885a3c873d64e73`
- **金标：** 同上一案例
- **解析结果：** `schema_validation_error: missing confidence`
- **错误原因：** 本报告的 format 分类同时包括可解析 JSON 的 Schema 失败；兜底脚本不得补字段。

### 4. 严重等级错误

- **样本：** `after_sales-00b16886b9fe85f7`
- **金标：** `facility_damage / medium`
- **模型预测：** `facility_damage / high`
- **错误原因：** 问题类型正确，但没有支持 `high` 的安全风险或完全不可用证据。

### 5. 约束遗漏

- **样本：** `itinerary_planning-b3ebfed1c8435fec`
- **金标：** 2 天、预算不超过 2000 元、末日 17:00 前结束、每日用餐和交通；慢节奏、公共交通优先；5 个必要要素
- **模型预测：** 硬/软约束及 `constraint_check` 仍为占位内容，仅保留 `daily_schedule`
- **错误原因：** 未逐条复述原始文字约束，遗漏 `meals`、`transport`、`budget` 和 `end-time check`。

这些案例说明商品 4-shot 的 pilot 胜出没有在全量上转化为稳定 Schema 或业务收益；这是实测负结果，不据此追加 Prompt、重选 winner 或改动金标。
