# Week 3 零样本基线与标准化 Prompt 状态报告

## 交付状态

| 范围 | 状态 |
| --- | --- |
| 评测数据结构、隔离、运行和评分框架 | PASS |
| 冻结人工标注与历史真实 run 验签 | PASS |
| 最简零样本 baseline 原始输出、格式和延迟 | PASS |
| baseline 原生语义指标 | PENDING |
| 标准化 Prompt 与三个 JSON Schema | PASS |
| Week 3 | PARTIAL |

Week 3 保持 `PARTIAL`。现有 450 条人工标注已冻结复用，不要求新增标注或人工编码。真实 baseline 已完成且可追溯；但最简 Prompt 的 450 条输出均为自然语言，当前没有可靠的自动语义编码规则，因此分类、OCR 和约束语义指标保持 `PENDING`。此外，冻结金标本身存在明确覆盖限制，不能由采样分层替代。

Project Control 已选择冻结 v1 路线：商品标注不重开，不创建 `week3_gold_v2`，不修复或重开历史标注台，不补充人工标注，也不执行 v2 重评分。本报告以冻结证据和实际支持数作为最终口径。

## 五类数量

以下 `tested_count` 绑定已验签的 completed full run `week3_baseline_full_20260721_003`：

| 场景 | target | candidate | annotated | validated | tested |
| --- | ---: | ---: | ---: | ---: | ---: |
| 以图搜商品 | 200 | 200 | 200 | 200 | 200 |
| 智能售后 | 150 | 150 | 150 | 150 | 150 |
| 多模态行程规划 | 100 | 100 | 100 | 100 | 100 |

运行资格只表示记录已人工完成、结构有效、图片可读、未明确拒绝且已纳入评测隔离；不等于人工金标覆盖完整，也不改变下述数据限制。

## 冻结数据与验签

| 产物 | SHA-256 |
| --- | --- |
| 商品 manifest | `cd85ce2926b3c9adee85c95dc166edd3b9905a844d4b9dd8fe76c224e133dd15` |
| 售后 manifest | `e1fdfc1b77db6519b311a6f846f4ff02df336e34661d841c1a5a42c725dc8a6e` |
| 行程 manifest | `584e2725459a88d48925077fe28239c77860f64b039fd410ed9199a0c6909fa8` |
| exclusion registry | `1430478f2af28c63025d017a806c3e8924900a168b39ca756eac8b0d776465c3` |

售后冻结集包含 `public_yelp=76` 和 `business_synthetic=74`，满足公开场景与业务合成来源并存，不设置导师未要求的数值比例。150 条售后 annotation payload 均与现有提交审计 SHA-256 和 annotator 匹配。当前 pending v3 售后 manifest 已备份，没有删除图片、历史 run 或审计记录。

两个真实 run 均通过仓库验证入口，且三个 manifest、exclusion、Prompt 和 Schema 哈希与各自 metadata 一致：

| run_id | Prompt | selected/record | 状态 |
| --- | --- | ---: | --- |
| `week3_baseline_full_20260721_003` | `baseline_minimal_v1` | 450/450 | completed、验签通过 |
| `week3_standardized_full_20260721_001` | `standardized_v1` | 450/450 | completed、验签通过；作为标准化设计的补充真实证据 |

两次运行的样本集合哈希均为 `5d244771ae4acd9eca46ad3937394232733d2526f2dde2255774ed2dcf9e96a7`。没有重复发送 live 请求。

## 冻结金标限制

采样分层证明候选设计覆盖，但以下统计来自人工 annotation，二者必须分开：

- 商品采样分层为酒店 67、景点 67、餐饮 66；人工金标为 `unknown=90`、餐饮 78、景点 16、酒店 16。
- 商品价位金标为 `unknown=100`、budget 40、mid_range 34、premium 23、luxury 3。这里的 `unknown` 是规范允许的有效人工判断：没有菜单、价目牌等直接价格证据时不得根据单份餐食或装修推断价位，不能解释为漏标。
- 商品设施的实际字段是 `visible_facilities`，不是 `core_facilities`。其中 128 条至少有一个可见设施标签，72 条为空数组；非空标签包含餐厅 61、舞台 17、户外座位 15、前台 15、酒吧 13、观景点 12 等。空数组表示图片中没有可确认的受控设施，不等于字段缺失。
- 售后采样分层为卫生 38、设施损坏 38、景点关闭 37、交通延误 37；人工问题金标为 `unknown=68`、关闭 37、延误 37、卫生 6、other 2，`facility_damage=0`。
- 售后严重度有 68 条 `unknown`；OCR 金标非空支持 85 条。
- 行程 100 条 `style_preferences` 均为空，因此图片风格偏好能力不受当前金标支持。专项检查表明这与商品的有效 `unknown` 不同，存在标注台字段暴露或提交序列化缺陷的高度可能，详见下节。

这些限制不阻断已冻结样本运行，但对应指标只能显示实际支持数、`PENDING` 或“不支持”，不得补齐、猜测或把 sampling stratum 当成 gold label。

### 行程风格字段专项检查

- 当前 100 条行程 annotation 均包含 `style_preferences` 键，但值全部为 `[]`；三个阶段性 UI manifest 备份从首次提交开始也始终没有非空风格值，因此不是后续恢复或报告统计造成的数据丢失。
- 100 条当前 annotation payload 均与提交审计 SHA-256 一致，说明空数组已经进入当时的提交 payload，而不是提交后被 manifest 覆盖。
- 100 条审计记录的 `suggestion_fields_accepted` 均只包含 `reference_images`、`text_constraints`、`hard_constraints`、`soft_constraints` 和 `required_itinerary_elements`；`style_preferences` 不在确定性建议范围内，本应由页面单独展示人工多选项。
- 设计规范列出了 15 个 `style_preferences` 选项。当前可恢复的最终词表字节码也包含这些选项，但其编译时间晚于行程标注提交，且当时实际运行的前端源码和静态资源未保留，因而不能反向证明标注时页面已展示并正确序列化该字段。
- 结合标注员确认“已完成当时页面列出的全部选项”和 100/100 一致为空，当前最合理的缺陷分类是 `probable_annotation_ui_field_exposure_or_serialization_defect`，而不是标注员漏标。商品标注不因此重新打开，现有行程值也不由规则或模型自动补造。

## 真实 baseline 结果

| 场景 | 样本 | JSON 合规 | Schema 通过 | 平均延迟 | P95 延迟 | 语义指标 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 以图搜商品 | 200 | 0% | 0% | 3463 ms | 5619 ms | PENDING，support=0 |
| 智能售后 | 150 | 0% | 0% | 2139 ms | 5000 ms | PENDING，support=0 |
| 多模态行程 | 100 | 0% | 0% | 3508 ms | 10195 ms | PENDING，support=0 |

最简 Prompt 按导师要求不含角色、JSON、字段表、示例、格式约束或推理引导。全部 450 条输出为自然语言，因此 0% JSON 合规是可复查的格式结果，不代表分类、OCR 或约束理解均为 0。当前评分器将这些 baseline 语义字段写为 `null`，并保留原始输出和延迟。

可追溯错误示例：`image_product_search-2885a3c873d64e73` 返回了关于酒瓶的自然语言分点描述，内容未形成 JSON；其错误类型为 `json_parse_error`，语义评分状态为 `pending`。这支持“最简指令下输出格式不稳定”的结论，但不足以证明其商品语义识别错误。

## 标准化 Prompt 补充证据

标准化 Prompt 保持系统角色、任务指令、输入上下文和输出约束四层结构，并使用三个独立 Schema。真实 run 的格式结果为：

| 场景 | JSON 合规 | Schema 通过 | 受支持标量指标 |
| --- | ---: | ---: | --- |
| 以图搜商品 | 68.5% | 29.5% | 业态 20.0%（110 条）；价位 13.0%（100 条） |
| 智能售后 | 98.0% | 2.0% | 问题类型 0%（82 条）；严重度 0%（82 条） |
| 多模态行程 | 28.0% | 0% | 风格偏好无金标支持 |

这是已有标准化 Prompt 的补充评测证据，不作为导师要求之外的强制 paired-comparison 门禁。可追溯错误显示部分商品输出因重复字段列表导致截断 JSON；行程场景没有 Schema-valid 输出，说明结构稳定性仍是当前 Prompt/小模型组合的主要工程限制。

## 结论与边界

- 已验证的工程能力：冻结数据验签、评测/训练隔离、真实批量请求、不可覆盖运行记录、原始输出和延迟留存、严格 JSON/Schema 处理、按金标支持数评分。
- 已证实的 baseline 短板：无格式约束时 450/450 为非 JSON 自然语言输出。
- 不能下结论的部分：baseline 的原生分类、OCR、约束识别能力，以及行程图片风格偏好能力。
- 当前导师范围内的 Prompt 优化依据：标准化四层架构显著提高部分场景 JSON 合规，但 Schema 稳定性仍不足；不得把冻结金标缺口解释成模型错误。
- Project Control 最终边界：冻结 v1，不创建 v2、不重开标注、不补标、不执行 v2 重评分；行程风格、售后 `facility_damage` 和 baseline 自然语言语义指标保持 `PENDING`。
- Project Control 已批准冻结 v1 的 `PARTIAL` 交付；Week 3 最终状态保持 `PARTIAL`。

## 验证命令

```bash
python -m unittest tests.test_evaluation_manifests tests.test_evaluation_runner tests.test_evaluation_metrics -v
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3.yaml
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3.yaml --run-id week3_baseline_full_20260721_003
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3.yaml --run-id week3_standardized_full_20260721_001
python -m unittest discover -s tests -v
git diff --check
git status --short
```
