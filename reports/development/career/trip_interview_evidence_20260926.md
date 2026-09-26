# Trip 简历与面试证据包

用途：为秋招面试复用现有、已验证的 Trip 证据，不启动模型、不重新消费 Fresh Test、也不把
silver 或 synthetic 标签改写成人工金标。本文是 `dev` 专属开发交接，不改变正式 release
`trip-qwen3-vl-8b-week8-final-v1`。

## 一页模型选型与系统架构

```text
图片 / 文本 / 对话历史
       |
       v
Qwen3-VL-8B + 场景 Prompt + JSON Schema
       |                     |
       |                     +-- 一次模型级纠错；失败关闭
       +-- 商品：base + 可观察事实/主体复查（adapter 关闭）
       +-- 售后：checkpoint-87 adapter 保留
       +-- 行程：base + 约束校验（adapter 关闭）
       +-- 对话：base + 显式状态、任务分派和工具协议（adapter 关闭）
       |
       v
CLIP 512 维 -> Milvus HNSW/COSINE -> 允许的标量过滤 -> FastAPI
```

### 为什么是这个组合

| 模块 | 选择 | 已有依据 | 不应夸大的结论 |
| --- | --- | --- | --- |
| 商品理解 | Qwen3-VL base + `product_visual_observation_v3` | 正式配置将 `image_product_search` 列入 `adapter_disabled_scenarios`；观察链把主体、可见事实和字段映射分开。 | 最终参考为 model-generated silver，不是人工视觉准确率。 |
| 智能售后 | checkpoint-87 adapter + `system_repair_after_sales_evidence_v3` | 正式配置没有禁用 `after_sales` adapter；历史 Fresh Test 的售后综合分为 1.0，JSON/Schema 为 1/1。 | 这不是新的未见测试，也不能代表所有售后类别。 |
| 行程规划 | Qwen3-VL base + `week8_itinerary_actionable_v5` + 业务约束校验 | 正式配置禁用 adapter；运行时检查日序、天数、占位文本、时限/地点约束，并保留一次模型纠错。 | 历史固定探针和 Fresh Test 结果不等于实时路线、库存或交通核验。 |
| 对话 | Qwen3-VL base + 显式 state/tool contract | 正式配置禁用 adapter；运行时拒绝只确认、不执行的模型答复。 | 交付为 beta，未通过 Week 7 strict research gate。 |
| 视觉检索 | CLIP 512 维 + Milvus HNSW/COSINE | 代码支持 city/category/price 标量过滤和未应用条件披露。 | ANN Recall 与工程延迟不等于独立人工业务相关性或生产 SLA。 |

运行时证据：[`configs/releases/qwen3_vl_system_final_v1.json`](../../../configs/releases/qwen3_vl_system_final_v1.json)
固定了模型、Prompt、Schema 和三项 adapter-disabled 场景；
[`src/inference/system_runtime.py`](../../../src/inference/system_runtime.py) 的 `_adapter_scope()`
实际使用 `disable_adapter()`，并在响应中回传 `adapter=none` 或 checkpoint 身份。

### 两分钟讲述

> 我做的是 OTA 多模态系统，不把它当成单一模型调用。输入图片、文本和对话上下文先进入
> Qwen3-VL，按商品、售后和行程使用独立 Prompt 与 JSON Schema。模型输出失败时只允许一次
> 模型级纠错，生产环境不允许静默 fallback。商品场景的主要问题不是 JSON，而是把图片里的
> 可见事实误推成商家属性，所以我把它改为 base model 的观察链：先识别主体和可见证据，再做
> 字段映射；证据不足就输出 unknown。行程和对话也采用 base 加规则契约，分别用约束校验和
> 显式状态处理。售后保留 QLoRA adapter，因为现有 release 的场景路由和历史评测都支持这一
> 选择。检索侧用 CLIP 生成 512 维向量，Milvus 负责 HNSW/COSINE 召回，业务过滤单独处理。
> 我把格式正确、自动 silver 匹配、历史 Fresh Test、人工金标和在线延迟严格分开记录：这让
> 系统可以交接，也让我能说明哪些结果只是开发诊断、哪些是正式交付边界。

## P0：商品主体/场景混淆与不支持属性

### 选择理由与数据边界

本轮选择该问题，而不是价位冲突或行程可执行性：

- 价位在当前商品 development 视觉正支持为 `0`，指标只能是 `N/A`；不能以商家 metadata
  替代图片可见价位。
- 历史 Fresh Test 没有保留 multi-subject 标签，无法把普通图片或模型输出反推为主体金标。
- 现有商品 development 有固定的 60 条图片、三角色同图同参考、五维身份隔离检查，以及
  “应当 abstain/unknown”的自动 silver 契约；它适合做可解释的工程对照，但不适合宣称人工
  视觉准确率。

数据与评分定义来自
[`reports/development/week8/week8_product_understanding_optimization_report.md`](../week8/week8_product_understanding_optimization_report.md)：

| 项目 | 已锁定定义 |
| --- | --- |
| 比较对象 | 同一 60 条图片上的“正式 Prompt + adapter”与“观察 v2 + base”。 |
| 标签来源 | qwen3.7-plus 图像 silver；`human annotation/review/acceptance=0`。 |
| 评分 | 已知业态 accuracy、style/facility micro P/R/F1、unknown accuracy、completeness、JSON/Schema、失败率和切片错误数；价位正支持为 0，记 `N/A`。 |
| 隔离 | `sample_id/source_id/image_sha256/group_id/constraint_template_id` 五维跨 split 冲突为 0；本表不读取或重跑 Fresh Test。 |
| 正确解释 | 表示同口径自动参考的路由/Prompt 诊断；不表示独立人工业务 gold 或生产泛化。 |

### 既有输出的同口径对照（不重跑）

| 指标 | 正式 Prompt + adapter | 观察 v2 + base | 解释 |
| --- | ---: | ---: | --- |
| 已知业态 accuracy（39） | 0.820513 | 0.820513 | 类别未改善。 |
| 含 unknown 业态 accuracy（60） | 0.533333 | 0.850000 | 主要来自证据不足时的 abstain。 |
| style F1 | 0.313725 | 0.613333 | 自动 silver 同口径改善。 |
| facility F1 | 0.257143 | 0.802632 | 需结合案例审查，不当作人工视觉结论。 |
| unknown accuracy | 0.470833 | 0.920833 | 降低无证据猜测。 |
| composite（3 个有支持字段） | 0.463794 | 0.745493 | 仅该 60 条自动 development。 |
| JSON / Schema / failure rate | 1 / 1 / 0 | 1 / 1 / 0 | 格式从来不是本问题的主要指标。 |
| mean / P95 latency（ms） | 4723.828 / 5015.134 | 6574.683 / 11249.417 | 质量诊断以更高延迟换取，不能称加速。 |

### 四个可解释案例/切片

以下是保留在报告中的已有错误事实，不复制原图或完整模型输出：

| 现象 | 既有样本/切片 | 正式 -> 观察 v2 | 能说明什么 |
| --- | --- | ---: | --- |
| 酿造设备被推成商家设施 | development suffix `0000` | 输出 `parking`，画面无停车场 | 场景线索不能直接变成商家属性。 |
| 食品特写被推成场所能力 | suffix `0015` | 输出 `parking`，无法由卷饼近景确认 | 主体不足时需要 `unknown`。 |
| 室内桌游/电脑被推成停车 | suffix `0030` | 输出 `parking`，画面无停车场 | 模型不能把上下文联想当可见证据。 |
| 多主体/主体不明 | 4 条切片 | 4 errors -> 3 errors | 有小幅改善，但支持太小且非人工标注，不能宣称解决。 |

同一 60 条上，食品特写错误为 `17 -> 1`，无价位依据却给价位为 `57 -> 0`，但这些也是
自动 silver 切片。它们适合面试中讲“如何限制 hallucination”，不能写作真人准确率。

### 本人可讲的实现范围

- `src/inference/system_runtime.py`：release 加载、场景 adapter 路由、一次模型级纠错、任务响应
  身份与失败关闭。
- `src/inference/product_observation.py`：商品可观察事实、主体/字段校验及 unknown 边界。
- `src/api/`：场景专属 Pydantic 输入、任务路由、`/health` 与 `/ready` 分离。
- `src/retrieval/`：CLIP 向量、Milvus HNSW/COSINE、标量过滤与未应用条件披露。
- `src/evaluation/`：字段级分母、切片与数据身份/哈希检查，避免用 JSON 合规替代业务指标。

## 数值归属：不要混用

| 数值/结论 | 正确归属 | 是否可和其他表横比 |
| --- | --- | --- |
| `0.780639` | 已消费 Fresh Test 120 的历史商品 composite；120/120、失败率 0，字段支持 style/facility/price=`25/30/5`。 | 不可与 Week 8 60/100 条 silver development 相减。 |
| Fresh Test 三场景 `0.780639/1.0/1.0` | A100 job `29569338` 的已完成历史 final；对话自动 composite=`0.973330`。 | 仅作历史发布描述；不可再选模或重跑。 |
| 商品 60 条 `0.745493` | Week 8 观察 v2 + base 的自动-silver development。 | 可与同表的正式 Prompt + adapter 对比；不是 Fresh Test。 |
| 商品 100 条 `0.736721` | 锁定 v9 的另一批自动-silver final comparison。 | 不能与 60 条或 Fresh Test 混算。 |
| `97.67%` | 历史 Fresh Test 120 中的 30 条对话样本，自动 **上下文词项召回**=`0.976667`；同表的对话自动 composite=`0.973330`。 | 可写为“30 条对话的自动上下文要素召回率 97.67%”，不是任务成功率、人工评分或当前正式 release 的新测量。 |

历史 Fresh Test 的离线审计还确认：商品字段 price F1=`0`（support=5），facility F1=`0.835`
（support=30），unknown 幻觉率=`0.1935`。这说明“JSON/Schema=100%”不能替代语义质量。

`97.67%` 的来源为
[`reports/development/reviews/system_consolidation_repair_report.md`](../reviews/system_consolidation_repair_report.md)
Fresh Test 表，以及 `src/training/week7_evaluation.py`：它把每条对话中期望上下文词项的命中数
除以该条的期望词项数，再在样本间取平均。历史报告保存 120 行 raw/metrics 的 SHA-256
`344464…eb19` / `853bd6…1018`，但本轮只读取报告与评分定义，未定位或重算 raw，因此这是
“有可追溯历史报告、当前工作包未原始重算”，不是新的测量。

## 最小复核入口

```powershell
git show dev:configs/releases/qwen3_vl_system_final_v1.json
git show dev:reports/development/week8/week8_product_understanding_optimization_report.md
git show dev:reports/development/reviews/search_algorithm_evidence_enhancement_report.md
```

不需要 GPU、训练或再次读取 Fresh Test。若将来获得新的、未见且标签来源明确的数据，必须新建
版本化 split、在运行前锁定 source/image/query identity 与评分代码；不能用本文自动 silver
结果反复调参追求正数。
