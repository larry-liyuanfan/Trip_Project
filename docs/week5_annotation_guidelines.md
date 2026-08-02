# Week 5 三场景指令数据标注与质检规范

## 范围与证据边界

本规范只适用于 `week5_instruction_candidates_v1`。模型输出是预标注，不能写成
人工完成；`sampling_metadata` 只用于分层抽样，也不是人工金标。标注人员必须查看
原图和原始文字约束，证据不足时使用 Schema 允许的 `unknown`、`null` 或空数组，
不得根据 Yelp 类别、评论路由或模型置信度猜测。

固定流程为：模型预标注 → 人工修正 → 标注人员自审 → 同场景交叉互审 → 核心样本
抽检。售后和行程为核心场景，确定性抽检比例 10%；商品为一般场景，抽检比例 5%。
抽检选择由 `sample_id` 哈希固定，不允许人为换样。未抽中的样本仍须完成自审和
交叉互审。

## 公共输入与状态

每条候选是 UTF-8 JSONL 对象，包含：

- `sample_id`、`scenario`、`source_id`、`source_type`、`source_license`；
- `input.images[]` 中的仓库相对路径与真实 `sha256`；行程另有非空
  `input.text_constraints`；
- `sampling_metadata`，明确标记 `hints_are_gold=false` 或
  `route_is_gold=false`；
- `isolation`，记录已调用的 Week 3 v1/v2 exclusion manifest 以及
  source、图片哈希、来源组、约束模板四类冲突结果；
- `workflow`，分别记录预标注、人工修正、三级质检和最终状态。

自动校验拒绝路径缺失、图片字节哈希变化、评估集冲突、重复样本/图片、Schema
失败、交叉互审人与标注人相同、未入抽检集合却提交核心抽检、重复覆盖清单等情况。

## 以图搜商品

输入为一张商家实景图。人工输出必须严格通过
`configs/evaluation/schemas/image_product_search_v1.schema.json`，与当前推理
结构完全一致：

| 字段 | 类型与范围 | 验收口径 |
| --- | --- | --- |
| `business_category` | `hotel/attraction/restaurant/other/unknown` | 按画面主体判断业态；证据不足用 `unknown`。 |
| `style_tags` | 唯一字符串数组 | 只写可见装修、氛围或环境风格。 |
| `visible_facilities` | 唯一字符串数组 | 只写画面可见设施，不从商家资料补齐。 |
| `price_range` | `budget/mid_range/premium/luxury/unknown` | 仅直接价格证据可支持；外观不能单独证明。 |
| `observed_evidence` | 最多 10 条、每条 1–120 字符 | 可定位的直接视觉事实。 |
| `inferred_attributes` | 每条 1–120 字符 | 必要推断与观察分开。 |
| `unknown_fields` | 唯一字符串数组 | 明确列出无法判断的字段。 |
| `confidence` | `null` 或 0–1 数值 | 对整条人工判断的置信度。 |

重点修正风格、设施、价位和业态；Week 2–4 的分类错误、示例属性复制、必填字段
遗漏和 Schema 错误属于难例。不得把评估样本复制进本数据集。

## 智能售后

输入为一张公开场景凭证或项目自有合成凭证。人工输出严格通过
`configs/evaluation/schemas/after_sales_v1.schema.json`：

| 字段 | 类型与范围 | 验收口径 |
| --- | --- | --- |
| `issue_type` | `hygiene_stain/facility_damage/attraction_closure/transport_delay/other/unknown` | 只能选择一个问题类型。 |
| `severity` | `low/medium/high/critical/unknown` | 依据可见影响；安全风险或完全不可用才可提高等级。 |
| `issue_location` | `null` 或 1–120 字符 | 只写证据支持的位置/设施。 |
| `key_information` | 每条 1–120 字符的数组 | 记录对象、状态、时间、编号及影响。 |
| `ocr_text` | `null` 或字符串数组 | 按可见顺序逐字抄录，不补遮挡字符。 |
| `observed_evidence` | 最多 10 条 | 直接可见事实。 |
| `unknown_fields` | 唯一字符串数组 | 不可判断字段。 |
| `confidence` | `null` 或 0–1 | 整体置信度。 |

重点修正严重等级、关键信息和 OCR。评论关键词仅负责候选路由；如果图片不支持
路由问题，人工必须改为 `unknown/other` 或在质检中剔除。严重等级错误、OCR
遗漏、图文不匹配属于核心抽检问题。

## 多模态行程规划

输入为一张风格参考图和原始出行约束。人工输出严格通过
`configs/evaluation/schemas/itinerary_planning_v2.schema.json`。顶层字段为
`style_preferences`、`hard_constraints`、`soft_constraints`、
`required_itinerary_elements`、`itinerary`、`constraint_check`、
`observed_evidence`、`unknown_fields`、`confidence`。

- 图片只支持风格和观察证据，不能证明文字硬约束。
- 硬/软约束必须逐项覆盖原始文字，保留最短完整原意，不添加未给出的城市、预算、
  时间或特殊需求。
- `required_itinerary_elements` 只能使用现有英文枚举；行程天数必须匹配原始约束。
- `itinerary` 每天至少一个活动；无法从证据确定地点时 `place_name=null`。
- `constraint_check` 必须覆盖全部硬/软约束，状态只用
  `satisfied/violated/unknown`。

重点修正约束完整性、行程结构合理性和要素完整性；约束遗漏、占位约束、枚举翻译、
天数不符是核心难例。

## 人工包与三级质检记录

导出包保留只读的模型预标注，同时要求人工另填 `annotator`、
`human_annotation` 和 `corrected_at`。应用人工结果只产生“已修正”记录，并把旧
质检结论视为失效；不会自动通过自审。

质检记录字段为 `sample_id/scenario/annotation_revision/stage/decision/reviewer/
issues/notes/reviewed_at`。`stage` 取 `self_review/cross_review/core_audit`；
`decision` 取 `pass/rework/reject`。返工必须提交新的人工 revision，再重新完成
对应质检。最终合格要求当前 revision 自审、交叉互审均通过，且入抽检集合的样本
核心抽检通过。

## 多轮多模态对话

对话 Schema 为 `configs/week5/schemas/multimodal_dialogue_v1.schema.json`。
`dialogue_id` 唯一；`scenario` 为 `image_search_consultation`、
`itinerary_iteration` 或 `after_sales_negotiation`；`images` 保存
`image_id/path/sha256`；`messages` 按 user/assistant 严格交替，每条含
`role/content/image_refs`。

一条对话包含 3–8 轮，即 6–16 条消息。至少引用一张登记图片，引用只能使用本对话
的 `image_id`。候选应覆盖首次上传、补充图片或条件、追问历史结果、修改约束、
指代历史图片。助手使用专业友好的 OTA 语气，不作无证据判断、退款保证、价格承诺
或安全承诺；用户保持真实口语。人工对逻辑、上下文、图片指代、业务合规和语气五项
逐项给出 pass/fail；任何一项失败都不能计入最终合格。

## 操作命令

```powershell
python scripts/manage_week5_dataset.py build-pools
python scripts/manage_week5_dataset.py validate-pools
python scripts/manage_week5_dataset.py preannotate --scenario image_product_search
python scripts/manage_week5_dataset.py preannotate --scenario after_sales
python scripts/manage_week5_dataset.py preannotate --scenario itinerary_planning
python scripts/manage_week5_dataset.py export-annotations --scenario image_product_search --output outputs/week5/packets/product.jsonl
python scripts/manage_week5_dataset.py apply-human --scenario image_product_search --input <completed.jsonl>
python scripts/manage_week5_dataset.py apply-quality --scenario image_product_search --input <quality.jsonl>
python scripts/manage_week5_dataset.py generate-dialogues
python scripts/manage_week5_dataset.py apply-dialogue-quality --input <dialogue-quality.jsonl>
python scripts/manage_week5_dataset.py report
```

预标注和对话生成支持按已完成 ID 断点续跑；失败单独记录，`--retry-failures`
显式重试；已有候选池和导出包禁止覆盖。

