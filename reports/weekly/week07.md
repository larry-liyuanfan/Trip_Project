# Week 7 多任务混合微调与上下文搭建周报

- 报告日期：2026-08-24
- 代码基线：Week 6 终态 `132779b0f6d2929ce1cdbed18e62adf3ef9edd18`
- 当前结论：**三项核心业务场景通过；修正后的多轮对话一次性 test 未通过；模型不进入 `stg`。**

## 1. 本周结论

本周完成了新数据锁、多任务 QLoRA/SFT、checkpoint 选择、三场景正式评测、修正后的多轮对话评测、Schema 解码对照和一次 mDPO 消融。结果不是“整体通过”，而是：

- **核心三场景有效**：统一模型在商品、售后、行程的正式一次性 test 上全部通过预注册非回退门禁，且明显高于 Week 6 路由 adapter 与 zero-shot。
- **对话能力有提升但未达标**：fix2 checkpoint-226 在 corrected-dialogue test 上明显优于两个基线，但 14 项绝对门禁中只通过 4 项，因此最终状态为 `FAIL`，test 不重跑。
- **DPO 不采用**：唯一一次 mDPO-style 消融的 validation preference accuracy 为 0.3333，低于 0.5 门槛；新 adapter 已拒绝，主模型保持 SFT checkpoint。
- **发布决策**：Week 7 代码和证据进入 `dev`；不进入 `stg`，不打标签。

## 2. 本周实际完成内容

| 工作项 | 实际状态 | 可核验证据 |
| --- | --- | --- |
| fresh train/development/test 数据锁 | 完成 | 3000/114/114；五维跨 split 冲突 0 |
| 多任务 QLoRA/SFT | 完成 | Spartan job `29540085`，`COMPLETED 0:0` |
| development checkpoint 选择 | 完成 | 5/8 合格，锁定 checkpoint-226 |
| 三核心场景一次性 test | 完成并通过 | 历史正式 job `29459265`，`all_passed=true` |
| corrected-dialogue 一次性 test | 完成但未通过 | job `29544969`，最终 gate `FAIL` |
| Schema constrained decoding 对照 | 完成但服务不支持 primary | constrained 90/90 被服务端拒绝 |
| mDPO-style 消融 | 执行一次并拒绝 | job `29491859`，validation 门禁失败 |
| 人工对话 development 对比 | 完成 | 同一真实操作者各完成 24/24；不冒充 test |

## 3. 数据与训练配置

fix2 数据身份为 `week7_corrected_multitask_context_20260824_v4_fix2`，canonical lock SHA-256 为 `86a4360142c2517e46460cefc575131940989aa8129eca236c68eaaf71e5b14b`。

| 数据部分 | 数量 | 占 train 比例 |
| --- | ---: | ---: |
| 商品多标签 | 600 | 20% |
| 售后 | 840 | 28% |
| 行程 | 840 | 28% |
| 通用多模态正则 | 270 | 9% |
| 5–8 轮对话 | 450 | 15% |
| **train 合计** | **3000** | **100%** |

development/test 均为 114 条：商品、售后、行程各 30 条，对话 24 条。`sample_id`、`source_id`、`image_sha256`、`group_id`、`constraint_template_id` 五维跨分区碰撞均为 0；Week 3 v2、Week 6 训练来源及历史 Week 7 身份均被排除。

训练固定使用 Qwen3-VL-8B、NF4 4bit、LoRA `r=16/alpha=32/dropout=0.08`、学习率 `1.5e-4`、weight decay `0.03`、gradient clipping `1.0`、gradient checkpointing、有效 batch 16 和 patience=2。job `29540085` 运行 03:35:20，在 step 301 早停，train loss 0.160256；峰值 allocated/reserved GPU memory 为 15.17/25.07 GB。

## 4. 结果

### 4.1 三核心场景：通过

核心三场景沿用已锁定且只消费一次的正式 test。对话构造缺陷不影响这三项单场景结果。

| 模型 | 商品 composite | 售后 composite | 行程 composite | 加权综合分 | 平均延迟 | 失败率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 统一多任务 checkpoint-151 | 0.153846 | 1.000000 | 0.996667 | 0.744987 | 7173 ms | 0% |
| Week 6 路由 adapters | 0.056410 | 0.100000 | 0.028333 | 0.061840 | 8251 ms | 0% |
| zero-shot | 0.076923 | 0.100000 | 0.050000 | 0.075577 | 4788 ms | 0% |

统一模型相对 Week 6 的商品/售后/行程绝对变化为 +0.097436/+0.900000/+0.968333，JSON 与 Schema 合规率均为 100%，全局延迟为 Week 6 的 0.869 倍。需要注意：商品 test 的 gold-evaluable 支持数很稀疏（category=2、facility=0、label completeness=3、price=0、style=1），因此商品分数只能按现有支持范围解释，不能宣称全面解决商品识别。

### 4.2 修正后的多轮对话：有提升，但门禁失败

fix2 selector 在 development 的 8 个 checkpoint 中找到 5 个合格候选，按锁定规则选择 checkpoint-226：weighted composite 0.796113、dialogue automatic 0.995949、失败率 0。随后唯一 corrected-dialogue test job `29544969` 对三种模型角色各评 24 条：

| test 角色 | 自动综合分 | 格式合规 | 上下文召回 | 上下文值准确率 | 失败率 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| multitask checkpoint-226 | 0.793399 | 0.916667 | 0.750000 | 0.719444 | 0.041667 | 33084 ms |
| Week 6 routed | 0.152144 | 0.750000 | 0.116667 | 0.000000 | 0.125000 | 32557 ms |
| zero-shot | 0.174505 | 0.875000 | 0.108333 | 0.000000 | 0.125000 | 34826 ms |

相对 Week 6，multitask 自动综合分 +0.641255、上下文召回 +0.633333、失败率 -0.083333；但绝对门禁仍不合格：

| 未通过指标 | 实测 | 门槛 |
| --- | ---: | ---: |
| automatic composite | 0.793399 | ≥0.85 |
| format compliance | 0.916667 | ≥0.95 |
| context recall | 0.750000 | ≥0.85 |
| context-state value accuracy | 0.719444 | ≥0.75 |
| task-result key coverage | 0.787037 | ≥0.95 |
| task-result value accuracy | 0.681899 | ≥0.75 |
| tool protocol compliance | 0.666667 | ≥0.95 |
| dialogue/overall failure rate | 0.041667 | ≤0.02 |
| sequential-turn failure rate | 0.041667 | ≤0.02 |

通过的 4 项是 anchor retention 0.916667、initial stable value 0.834979、sequential protocol coverage 0.965278、sequential semantic accuracy 0.865410。最终 `FAIL` 是模型 test 结果未达到事前阈值，不是 selector 故障，也不通过降阈值或重跑来改写。

四维自动结果为：图片指代 0.333333、需求调整 0.833333、上下文承接 0.958333、逻辑连贯 0.719444。主要短板集中在图片历史指代、任务结果值和工具调用协议。

### 4.3 人工 development 对比：差异很小

同一真实操作者分别完成 multitask 与 Week 6 routed 的 24/24 条四维评分。总均值为 4.59375/4.56250，配对差 +0.03125，样本级 10 胜/7 平/7 负。图片指代、需求调整、上下文承接、逻辑连贯差值为 -0.125/+0.291667/-0.041667/0。该结果只说明 development 人工观感接近，不代表 test 通过，也不宣称统计显著。

### 4.4 Schema 与 DPO

- Schema 对照每种模式 90 条。free JSON 合规率 98.89%、Schema 覆盖 0%、平均延迟 3301 ms；constrained primary 90/90 被当前服务端拒绝，fallback free 90/90 成功，平均延迟 3360 ms。结论是当前服务端不支持该 constrained path，不能宣称格式或语义提升。
- mDPO-style 只执行一次：train preference accuracy 0.8、平均 margin +0.01861；隔离 validation 为 0.3333/-0.00981，未通过 0.5/>0 门禁。DPO adapter 不采用且不重试。

## 5. 本周修复的问题

| 原问题 | 修复 |
| --- | --- |
| v3 对话 assistant/user 顺序错误 | 新建独立 corrected-dialogue 身份，不改写历史 test |
| 嵌套 JSON 顶层全等放大局部错误 | fix2 改为叶子值准确率 |
| silver 自由文本被当作逐字 hard gold | 自由文本 evidence 不再进入 hard-gate 逐字比较 |
| early stopping 与 selector 目标错位 | 统一为 gate-first selection score |
| 分支过多 | 长期分支收敛为 `dev`、`stg`、`main` |

## 6. 交付与限制

- 数据锁、训练、checkpoint、raw、selector、test consumption marker 和报告均保留 SHA-256。
- fix2 定向测试 54/54、完整 unittest 454/454、数据锁、配置加载、两份 Slurm shell 语法和 `git diff --check` 均通过。
- corrected-dialogue test 已消费一次，禁止重跑；人工评分仅用于 development。
- 三核心场景通过不等于多轮对话通过。Week 7 的准确发布结论是：**核心能力通过，对话能力未验收，整体不晋级 `stg`。**

关键证据：训练 job `29540085`；对话 test job `29544969`；selection SHA-256 `cba44b4fe580dc47f7fbb332c12c46cae39fcd07c70bedd1a859a4793d0c3ac8`；final comparison SHA-256 `047d48bd40db7e06110063687e2fdb3b52801e856ae438fdaec02980b8a68e00`；consumption marker SHA-256 `3c9370b40f137d521f853f7534e57f57f50588a8ef938530fb9510e6ef50067b`。

## 7. 开会时可直接使用的表述

> 本周把三个单场景 adapter 合并成统一 Qwen3-VL-8B 多任务模型。核心商品、售后、行程一次性评测全部通过，综合分从 Week 6 路由基线的 0.0618 提升到 0.7450。修正后的多轮对话模型也明显超过两个基线，但绝对综合分 0.7934 未达到 0.85 门槛，格式、上下文召回、任务结果和工具协议仍有缺口，因此我们如实判定对话 FAIL，不重跑 test、不采用失败的 DPO adapter，也不进入 stg。
