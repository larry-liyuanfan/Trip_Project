# Week 4 Bad Case 报告

Bad case 只从 `week4_winners_full_20260725_001` 的真实输出、现有人工金标和
既有评分导出，不修改输出或标签。

| 类别 | 数量 | 代表样本 |
| --- | ---: | --- |
| 分类错误 | 86 | `image_product_search-0ddcb7ef11f23afb` |
| 必填字段或 Schema 错误 | 7 | `image_product_search-5d6823c5422a7bee` |
| JSON 格式错误 | 67 | `image_product_search-c0ed4d164daafeea` |
| 严重等级错误 | 105 | `after_sales-00b16886b9fe85f7` |
| 约束遗漏 | 100 | `itinerary_planning-b3ebfed1c8435fec` |

同一样本可以进入多个类别；机器可读 v3 明细共 269 条，位于忽略目录
`outputs/week4/bad_cases/week4_bad_cases_v3.jsonl`。

## 典型真实案例

| 类别 / sample_id | 金标 | 模型预测 | 错误原因 | 当前可核对的修复方向 |
| --- | --- | --- | --- | --- |
| 分类错误 `image_product_search-0ddcb7ef11f23afb` | `business_category=attraction`；`price_range=budget`；风格 `casual,resort`；设施 `outdoor_seating` | `business_category=restaurant`；`price_range=budget`；风格 `casual`；设施 `table,seating` | 把户外用餐画面直接归为餐饮，遗漏 resort 与受控设施枚举 | 先检查“画面内容”与“承载业态”证据，再映射受控枚举；无充分证据时返回 `unknown` |
| Schema 错误 `image_product_search-5d6823c5422a7bee` | `business_category=attraction`；风格 `casual,artistic`；设施 `play_area` | `style_tags` 和 `visible_facilities` 各重复输出 3 个占位符；Schema 报数组元素不唯一 | 模型照抄类型骨架占位符，且未执行数组去重检查 | 最终字段检查先拒绝占位符，再检查数组唯一性与枚举 |
| JSON 格式错误 `image_product_search-c0ed4d164daafeea` | `business_category=restaurant`；`price_range=mid_range`；风格 `modern,romantic`；设施 `stage` | 输出从 `style_tags` 开始重复扩张，最终字符串截断，无法解析 JSON | 重复标签消耗输出预算，导致 JSON 未闭合 | 输出前限制字段为短列表并去重，最后检查 JSON 闭合；兜底脚本不得修复截断内容 |
| 严重等级错误 `after_sales-00b16886b9fe85f7` | `issue_type=facility_damage`；`severity=medium` | `issue_type=facility_damage`；`severity=high` | 问题类型正确，但把水槽损坏的严重度上调，缺少支持 high 的安全/不可用证据 | 严重度在问题类型之后单独核对，只按现有严重度定义和可见影响选择 |
| 约束遗漏 `itinerary_planning-b3ebfed1c8435fec` | 2 天、预算不超过 2000 元、末日 17:00 前结束、每日用餐与交通；慢节奏、公共交通优先；5 个必要要素 | 硬/软约束均为占位符，只保留 `daily_schedule`；`constraint_check` 也是占位符 | 没有逐条复述输入约束，也未覆盖 meals、transport、budget/end-time check | 按输入顺序逐条提取硬/软约束，再逐项检查必要要素和 `constraint_check`，不生成占位文本 |

“修复方向”是对现有 Prompt 字段检查顺序的错误归纳，不代表新增标注、数据
流程或未来任务。
