# Week 7 多任务混合微调与上下文搭建执行报告

当前状态：`CORE_AUTOMATED_ACCEPTED / CORRECTED_DIALOGUE_DEV_HUMAN_COMPARISON_COMPLETED / V3_TEST_DIALOGUE_INVALID / V4_FIX1_DEVELOPMENT_GATE_FAILED_TEST_UNCONSUMED / V4_FIX2_ONE_SHOT_TEST_GATE_FAILED`。
三个核心场景通过唯一一次正式 test 的自动非回退门禁；后续审计发现 v3 多轮上下文构造
错位，对话自动指标只保留为历史输出，不作为真实多轮能力结论。独立 development-only
修复身份和新 raw 已恢复可信人工评分入口，真实单人操作者随后分别完成 multitask 与
Week 6 routed 各 24/24 条四维评分。

## Corrected-dialogue v4 fix2 门禁修复

fix1 的 selector 正确执行，但评分和训练目标存在两个实现缺陷：嵌套 JSON 使用顶层对象
全等比较，局部错误会把整块计为 0；自由文本轮次还把 Yelp 主观 caption 当作逐字视觉
金标。训练 early stopping 使用 `eval_weighted_composite`，而最终 selector 使用自动硬
门禁，导致 sequential 门禁在早停目标中的实际权重过低。这些问题与模型自身存在的
商品识别错误同时成立，不能通过直接降低 0.75 阈值解决。

fix2 配置为 `configs/week7/qwen3_vl_8b_multitask_context_v4_fix2.json`。新协议
`gate_aligned_v2` 按叶子字段计算稳定结构值，排除自由文本 evidence 的 hard-gate
逐字比较，并分别输出 `sequential_protocol_coverage` 与
`sequential_semantic_accuracy`。`eval_gate_selection_score` 保证全门禁通过候选优先，
未通过候选按最弱门禁进度排序；最终 selector 的通过后加权裁决保持不变。

新锁 `week7_corrected_multitask_context_20260824_v4_fix2` 为 3000/114/114，train 配比
600/840/840 + 通用 270（9%）+ 对话 450（15%）；五维跨分区碰撞为 0。canonical
lock SHA-256 为 `86a4360142c2517e46460cefc575131940989aa8129eca236c68eaaf71e5b14b`，
train/development/test SHA-256 分别为 `cc21a001...07ced`、`b157eace...025a4`、
`1c79407f...c8ede`。v3、首版 v4、fix1 identity manifest 均进入排除证据。

Spartan 训练 job `29540085` 在提交 `8a4fd0103694a4251b71f6fed159e9ee2f8d9c00` 上
`COMPLETED 0:0`，耗时 03:35:20，step 301 按 patience=2 早停；run identity/run summary
SHA-256 为 `b778df5d...f4fe9092`/`a07b67f9...db358ac7`，train loss 0.160256，峰值
allocated/reserved 显存为 15,166,590,464/25,071,452,160 bytes。不可覆盖 selector
对 8 个 checkpoint 重算，5 个通过全部 development 自动门禁，锁定最优 step 226；
adapter SHA-256 为 `ccc6062f7e451b9265c571c0df397903cbbc707a6bf2e894039079175e5f24ee`，
selection 文件 SHA-256 为 `cba44b4fe580dc47f7fbb332c12c46cae39fcd07c70bedd1a859a4793d0c3ac8`。

| fix2 development step | weighted composite | 全门禁 |
| ---: | ---: | --- |
| 38 | 0.363881 | FAIL |
| 76 | 0.702633 | FAIL |
| 113 | 0.782829 | FAIL |
| 151 | 0.794882 | PASS |
| 188 | 0.791937 | PASS |
| 226 | 0.796113 | PASS（selected） |
| 264 | 0.796113 | PASS |
| 301 | 0.796113 | PASS |

selected development 支持数为 114（商品/售后/行程各 30、对话 24），core weighted
0.746154、dialogue automatic 0.995949、格式与上下文召回均为 1、失败率为 0。

唯一 one-shot test job `29544969` `COMPLETED 0:0`，耗时 00:42:43；每个角色使用同一
corrected-dialogue 24 条，纯自动评估，未执行人工评分。comparison/consumption marker
SHA-256 为 `047d48bd40db7e06110063687e2fdb3b52801e856ae438fdaec02980b8a68e00`/
`3c9370b40f137d521f853f7534e57f57f50588a8ef938530fb9510e6ef50067b`。

| test 角色 | 自动综合分 | 格式 | 上下文召回 | 上下文值 | 失败率 | 平均延迟 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| multitask checkpoint-226 | 0.793399 | 0.916667 | 0.750000 | 0.719444 | 0.041667 | 33083.75 |
| Week 6 routed | 0.152144 | 0.750000 | 0.116667 | 0.000000 | 0.125000 | 32556.57 |
| zero-shot | 0.174505 | 0.875000 | 0.108333 | 0.000000 | 0.125000 | 34826.04 |

multitask 相对 Week 6 routed 的自动综合分、上下文召回、上下文值分别变化
+0.641255/+0.633333/+0.719444，失败率变化 -0.083333；四维自动差值（图片指代、需求
调整、上下文承接、逻辑连贯）为 +0.125/+0.750/+0.125/+0.719444。以上仅为描述性对比。
最终绝对门禁为 `FAIL`：automatic composite、context recall、context-state value、dialogue
failure、format、overall failure、sequential-turn failure、task-result key/value 和 tool
protocol 共 10 项未达阈值；anchor、initial stable、sequential protocol/semantic 4 项通过。
test 已消费且不得重跑，不能通过降阈值或选择其他 checkpoint 改写结果，因此 fix2 模型
未验收、未进入 `stg`、未打标签。DPO 保持既有一次 validation FAIL 后关闭。
immutable data-lock 验证摘要仍显示训练前字段 `test_consumed=false`；实际单次消费由独立
runtime marker 记录，二者职责不同，未改写锁文件。终态 fix2 定向 54/54、完整 unittest
454/454、数据锁与五维隔离、config loader、两份 v4 Slurm shell 语法和 diff 检查均通过。

## Corrected-dialogue v4 fix1 终态闭环

fix1 是用户直接授权的独立 gate-repair 身份，不改写 v3 或首版 v4。配置为
`configs/week7/qwen3_vl_8b_multitask_context_v4_fix1.json`，SHA-256
`42ac8657bf21dd0887ab53acbce68e0ab074aa5c5c9e0044b802d2f4a3003de6`；执行提交为
`6bb5322f9f0b1daa3004bab27c0884c4bd6971fd`。数据身份
`week7_corrected_multitask_context_20260823_v4_fix1` 的 canonical lock SHA-256 为
`7f66795c69f8cb35cafa712e7847155708a662b88d069824b60706f6903ea9a7`，
train/development/test=3000/114/114。train 实际配比为商品 600、售后 840、行程
840、通用多模态 270（9%）、对话 450（15%）；五维分区冲突为 0，并额外排除 v3 和
首版 v4 的完整 identity manifest。test 始终未读取。

首次 job `29526506` 因 `HF_HOME` 错指仅约 35 MiB 的 runtime-home 失败。安全修复只把
cache 指向已确认的 25 GiB `huggingface/hub` 并使用新非覆盖输出目录，没有改变
config/data/run/git identity。恢复 job `29526965` 在单张 L40S 上 `COMPLETED 0:0`，
耗时 02:43:24；运行至 step 226 后按 patience=2 早停。run identity 文件 SHA-256 为
`03663dad...69dbf`，run summary SHA-256 为 `6d5400fd...491d0`，train loss 为
0.182206，峰值 allocated/reserved 显存为 15,191,208,448/31,545,360,384 bytes。
best/final adapter 为 checkpoint-151，SHA-256 `b42aeeb...5131bc`。

| fix1 development step | weighted composite | core weighted | dialogue automatic | 格式 | context recall | sequential coverage | 全门禁 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 38 | 0.353427 | 0.391269 | 0.202059 | 0.708333 | 0.371528 | 0.334551 | FAIL |
| 76 | 0.729860 | 0.735654 | 0.706683 | 0.833333 | 0.826389 | 0.711825 | FAIL |
| 113 | 0.751086 | 0.725154 | 0.854817 | 0.958333 | 0.833333 | 0.724458 | FAIL |
| 151 | 0.764049 | 0.735654 | 0.877630 | 1.000000 | 0.854167 | 0.725585 | FAIL |
| 188 | 0.753292 | 0.725154 | 0.865844 | 0.958333 | 0.840278 | 0.733360 | FAIL |
| 226 | 0.752986 | 0.725154 | 0.864316 | 1.000000 | 0.854167 | 0.733084 | FAIL |

每次 development 支持数为商品/售后/行程各 30、对话 24。最佳 step 151 的三个核心
场景 composite 为 0.153846/0.970000/1.000000；context-state value、task-result
key/value、initial stable、anchor、tool protocol 分别为
0.791667/0.962384/0.820023/0.931548/1.0/1.0。overall、dialogue 与 sequential
failure rate 均为 0，平均延迟 11,503.48 ms。其自动门禁只有 sequential coverage
0.725585 未达到预注册 0.75；其余 checkpoint 至少一项门禁失败，不能用相邻 checkpoint
拼接指标。商品行级支持数为 30，但 gold-evaluable 子指标支持仍固定稀疏：category=3、
facility=2、label completeness=4、price=0、style=0；商品 JSON/Schema、售后各主指标、
行程各主指标支持数为 30，对话 tool-protocol 支持数为 3。没有删除低支持指标或将
unsupported 样本计为 0。

不可覆盖 selector 已重算全部六个候选并写出
`BLOCKED_NO_ELIGIBLE_CHECKPOINT`，eligible_count=0、selected checkpoint=null、
`test_read=false`。阻断 evidence 文件 SHA-256 为
`782e92ab673c8628861af8e1eb6247454f3c8c9f608c9888899ae3eec64cc104`，内部 canonical
selection SHA-256 为 `e2069003...c3572d`。因此唯一 fix1 one-shot test 没有提交，
consumption marker 不存在；没有 fix1 的 Week 6 routed/zero-shot test 对比，也没有将
development 数值或历史 v3 test 冒充本轮 test 结果。

## Corrected-dialogue v4 历史自动闭环

v4 身份 `week7_corrected_multitask_context_20260822_v4` 使用独立 3000/114/114
train/development/test 锁，canonical SHA-256 为
`000a2e57620428034da27e03ba3c92483e9c147032166ad273ed089fbb97c9fa`。训练实际比例为
三核心场景各 760、通用多模态 270（9%）、多轮对话 450（15%）；三分区五维冲突为 0，
v4 test 与 v3 完整 3228 行 identity manifest 五维重叠也为 0。

锁定 config SHA-256 为 `e5b76008e504e0775b62506acbeba3e38438cf14851493be512aa4325fd89b7c`。
最终同身份恢复 job `29506362` 在 clean commit `c002a78` 上 `COMPLETED 0:0`，耗时
02:37:50；从 checkpoint-38 恢复，完成 step 38/76/113/151/188/226 六次完整
development 评估，并在连续两次未改善后于 step 226 早停。run summary SHA-256 为
`5af980efc851e2e0c15d96ea13853e3728fa194618fcd737ea976e3926e2e5a5`，最佳综合分
0.833980 位于 step 151；该 checkpoint/final adapter SHA-256 为
`296ad3f362e559738b55d93e2164f549631994138f5acaed72d8b4b3b48d9d86`。

| v4 development step | weighted composite | automatic composite | 格式 | context recall | task value | sequential coverage | 全门禁 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 38 | 0.339991 | 0.231308 | 0.958333 | 0.354167 | 0.000000 | 0.306654 | FAIL |
| 76 | 0.787641 | 0.591224 | 0.625000 | 0.750000 | 0.468750 | 0.711481 | FAIL |
| 113 | 0.806420 | 0.769122 | 0.833333 | 0.809028 | 0.708333 | 0.721238 | FAIL |
| 151 | 0.833980 | 0.778585 | 0.833333 | 0.777778 | 0.741319 | 0.733081 | FAIL |
| 188 | 0.832830 | 0.775168 | 0.833333 | 0.750000 | 0.741319 | 0.726524 | FAIL |
| 226 | 0.817248 | 0.765102 | 0.833333 | 0.715278 | 0.741319 | 0.711618 | FAIL |

每次 development 支持数为商品/售后/行程各 30、对话 24。最佳 step 151 的三个核心
场景 composite 为 0.564706/0.970000/0.968333，对话 automatic 为 0.778585；step 226
相应为 0.547059/0.970000/0.933333/0.765102，114 条失败为 0，平均延迟
14,716.89 ms。step 226 的商品 JSON/Schema 为 1.0/0.966667，售后为
1.0/0.966667，行程为 0.933333/0.933333。这些都是 development 指标，不是 test 结果。

所有候选的 overall/dialogue/sequential failure rate 均为 0，但 0/6 同时达到预注册的
格式 0.95、context recall 0.85、context-state value 0.75、task key 0.95、task value
0.75、sequential coverage 0.75、automatic composite 0.85 门槛。不可覆盖 selector 已
实际执行并正确返回 `no v4 checkpoint passed the automatic development gate`，没有写出
selection。按既定隔离规则未提交 corrected-dialogue v4 test；test consumption marker
不存在且 test policy 仍为 `LOCKED_UNCONSUMED`，因此没有生成或宣称 v4 的 Week 6
routed/zero-shot test 对比。

## 数据锁和隔离结果

当前 fix1 锁及隔离结果见本报告首节：3000/114/114、canonical SHA-256
`7f66795c...9a7`，并对 v3、首版 v4、Week 3 与 Week 6 来源执行五维排除。以下段落保留
v3 历史身份和其一次性 test 消费记录，不代表 fix1 读取了 test。

执行分支从 Week 6 终态 `132779b0f6d2929ce1cdbed18e62adf3ef9edd18` 建立，旧
`agent/portfolio-positioning` 工作树未参与开发。以下历史 v3 身份为
`week7_fresh_multitask_context_20260820_v3`，配置 SHA-256 为
`d77d9f10b551f30c599572e974fba2c3c2af087f37ed35e93b9dc7ac2dc105fa`，数据锁
SHA-256 为 `8af2e2d13c22fb641fc7344b1e56e5827aa78b1ebde653c6e55c83b36d20504d`。
train/development/test 为 3000/114/114；sample_id、source_id、image_sha256、group_id、
constraint_template_id 五维跨分区碰撞均为 0。Week 3 v2、Week 6 训练来源与三分区保持
排除，Week 6 数据、adapter、checkpoint、raw、报告和归档均未修改。

早期 v2 锁因对话父任务全来自商品场景而作废，对应 GPU 作业 `29431992` 被取消，产物
禁止进入参数锁。v3 对话父任务在 train 为商品/售后/行程各 150 条，在 development/test
各为 8/8/8。test 仅在参数锁创建后由作业 `29459265` 读取一次；消费 marker 为
`COMPLETED`，`resume_count=0`、`failure_history=[]`。

## 实际数据配比

当前 fix1 train 的实际配比为商品/售后/行程 600/840/840，通用正则 270（9%）、
多轮对话 450（15%）；下述三核心场景各 760 是历史 v3/首版 v4 配比，保留用于解释
既有 test 与人工证据，不是 fix1 训练计数。

train 含商品、售后、行程各 760 条，共 2280 条；通用多模态正则 270 条（9%）；5–8 轮
对话 450 条（15%），其中工具调用格式 45 条（对话内 10%）。对话图片仅在首个用户轮
出现，并采用结构感知截断。全部 Week 7 标签仍为 programmatic silver；24 条固定人工
评估队列未产生人工分数。构造器在 450/24/24 条 train/development/test 对话中先追加
assistant 回复、再追加其对应的 user 问题；v3 锁、训练和正式 test 产物保持不可变，
不得在原身份上重排后继续计分。

修复身份 `week7_dialogue_review_20260821_v2` 不改写 v3：24 条按 5/6/7/8 轮各 6 条，
初始任务和每个 follow-up 均为 user→具体 assistant 回答，图片仅首次用户轮；其数据锁
canonical SHA 为 `0a3b65e9...7d39`。该身份仅允许 development 人工复核，禁止 train/test，
也不能改变已消费的一次性 final-test 结论。

## Schema 解码对照

作业 `29434316` 在 `Qwen/Qwen3-VL-8B-Instruct` 上完成，每个 mode 支持数 90。free 的
JSON 合规率 98.89%、Schema 覆盖 0%、请求失败率 0%、平均延迟 3300.72 ms。
constrained primary 90/90 被服务端拒绝，JSON/Schema 均为 0%；90 次真实 free fallback
全部成功，fallback 失败率 0%，包含 fallback 的平均延迟 3360.34 ms，延迟比 1.0181。
生产模式锁定 free。primary 与 fallback 分开保存和计分；该实验只说明格式和服务兼容性，
不解释为语义提升。

## 多任务训练和 checkpoint

统一 SFT 作业 `29434317` 在 L40S 上 `COMPLETED 0:0`，耗时 01:30:17。配置为 NF4
4bit、LoRA `r=16/alpha=32/dropout=0.08`、注意力层与视觉投影层、学习率 `1.5e-4`、
weight decay `0.03`、max grad norm `1.0`、gradient checkpointing、2 epochs、有效
batch 16。计划 376 个更新步；完成 38/76/113/151 四个预注册评估点后，综合分连续两次
无提升并按 patience=2 在 step 151 早停。所有 development raw、metrics 和 checkpoint
均已哈希绑定；峰值 allocated/reserved 显存为 14.82/21.52 GB。

独立 `evaluation_protocol_v5` 没有重训或新建数据锁，只绑定 v3 的同一 development 和
四个 checkpoint，并在同一 allocation 统一 BF16、static KV cache、Transformers
compile、32-token warm-up、CUDA 同步计时、结构感知截断与 gold-evaluable support。
有效作业 `29456896` 为 `COMPLETED 0:0`，耗时 01:28:19；四个候选均合格，selector
按最高综合分选择 checkpoint-151。此前 protocol-v4 的阻断结论作为旧测量协议历史保留，
不再作为终态选择依据。

| v3 checkpoint | v5 development 综合分 | 平均延迟（ms） | 相对 Week 6 延迟比 | 合格 |
| --- | ---: | ---: | ---: | --- |
| step 38 | 0.074359 | 8689.47 | 1.0405 | 是 |
| step 76 | 0.642718 | 7863.99 | 0.9417 | 是 |
| step 113 | 0.645237 | 7188.87 | 0.8609 | 是 |
| step 151 | 0.740904 | 7356.58 | 0.8809 | 是 |

参数锁 canonical SHA-256 为
`1b3f3ffafc2f549ca29034fcee505e346bcb70bc8ce974adcdbb83ad6d38adef`，完整绑定
protocol、selection、checkpoint-151、adapter 和最终推理 runtime。

## 三场景及对话指标

历史 v3 的唯一 final-test 作业 `29459265` 在 L40S 上 `COMPLETED 0:0`，耗时 00:40:50；marker
和 7 个结果 artifact 的 SHA-256 全部复验通过，`all_passed=true`。

| 模型角色 | 商品 composite | 售后 composite | 行程 composite | 加权综合分 | 平均延迟（ms） | 失败率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 统一多任务 checkpoint-151 | 0.153846 | 1.000000 | 0.996667 | 0.744987 | 7173.16 | 0% |
| Week 6 路由 adapters | 0.056410 | 0.100000 | 0.028333 | 0.061840 | 8250.70 | 0% |
| zero-shot | 0.076923 | 0.100000 | 0.050000 | 0.075577 | 4788.49 | 0% |

统一模型的三个核心场景 JSON 与 Schema 合规率均为 100%。商品 gold-evaluable 支持数按
metric 为 category 2、facility 0、label completeness 3、price 0、style 1；售后和行程
各业务指标支持数均为 30。低或为 0 的商品支持数是锁定 test gold 的可评估证据范围，
没有强制补标签或把 unsupported 计为 0。

历史自动输出中，统一模型对话支持数 24，格式合规率 1.0、字符串包含式上下文召回率 0.878472；Week 6 为
0.5/0.496528，zero-shot 为 0.5/0.600694。图片指代、需求调整、上下文承接和逻辑连贯性
的人工队列 24/24 被完整性门禁标记为 `BLOCKED_INVALID_SOURCE_CONTEXT`。这些自动值未检测
assistant/user 语义顺序或末轮回答相关性，因此不能证明真实多轮连贯性。

新的 corrected development run `29479822` 在 checkpoint-151 上 24/24 成功、失败 0，
raw SHA 为 `9cb8cafc...cd162`；自动格式合规率 0.875、字符串 context recall 0.583333。
这些自动值不是人工结论。真实单人操作者在同一会话完成 corrected 队列 24/24，最终
24 条决定均为 `pass`；历史图片指代、需求迭代、上下文承接、逻辑连贯均分分别为
4.541667/4.625000/4.500000/4.708333，四维未加权均值 4.59375。26 条 append-only
记录含 2 次真实修订，结果 SHA 为 `bdec2d18...af932`，Agent 未代填分数。

## Week 6 / 零样本对比

统一模型相对 Week 6 的商品/售后/行程绝对变化为 +0.097436/+0.900000/+0.968333，
相对变化为 +172.73%/+900.00%/+3417.65%；相对 zero-shot 的绝对变化为
+0.076923/+0.900000/+0.946667，相对变化为 +100.00%/+900.00%/+1893.33%。最终
multitask/Week 6 全局延迟比约 0.8694，失败率均为 0。三场景任务、支持、JSON/Schema、
全局延迟和失败率门禁全部通过；没有把 development 指标冒充 test 结果。

## DPO 执行状态

初始门禁因 0 条偏好对记为 `SKIPPED`。双模型真实四维评分完成后，确定性派生 16 个
非平局候选，并通过 chosen 各维 ≥4、JSON、视觉证据命中、来源身份、生成成功和反转
探针审计；锁为 10 train/6 validation。Agent 只执行可复现审计，没有生成真人身份或
复制分数，也没有把派生选择写成显式人工二选一。

唯一 mDPO-style job `29491859` 执行 5 次 optimizer update，train preference accuracy/
平均 policy-reference margin 为 0.8/+0.01861，隔离 validation 为 0.3333/-0.00981，
未通过预注册 0.5/>0 门禁。新 adapter SHA `3791896e...39b64` 不选用；按单次消融约束
不再重试，且未运行核心 development 生成或 test。checkpoint-151 保持最终选择。

## 测试结果

终态新增机器优先对抗审计：既有真人 development 评分只作辅助证据，Agent 不替换人工
身份或分数。基线替换、跨分区碰撞、比例漂移、Schema 语义洗白、test 重跑、支持数删除、
对话缺陷洗白、repair 读取 test、Agent 冒充人工、失败 DPO 晋级和 DPO 读取 test 共
11/11 个反事实均被拒绝。该历史审计曾允许 v3 实现进入 `dev` 保存；当前 v4 自动门禁
失败后不再满足本轮 `dev` 快进条件，也不允许晋级 `stg` 或绕过门禁运行 test。

当前 fix1 闭环实测定向 26/26、全部 Week 7 79/79、完整
`python -B -m unittest discover -s tests -v` 450/450 PASS。fix1 数据锁/五维隔离、
配置解析、两份 v4 Slurm 脚本 `bash -n` 和 `git diff --check` 均通过；历史远端
final-runtime 定向 22/22 继续保留。

## Commit / push 状态

protocol-v5 提交 `64a5a7a`、final runtime 修复 `8619b76`、对话修复提交
`bc299c3`/`3e5e767`/`7cf656a` 已推送至历史分支；首版 v4 恢复、训练与门禁证据保留在
历史 recovery 分支。fix1 实现提交 `6bb5322` 与本终态证据位于并推送至
`origin/codex/week7-v4-gate-repair`。由于 fix1 development 自动门禁 FAIL，当前分支
不快进 `dev`，不进入 `stg`，不打标签。

本地目录整理把 408,127,632 字节作废 v1/v2 锁和传输包、失败构建及临时预检脚本移入
Windows 回收站；保留 v3 锁/归档、corrected dialogue raw、真实人工记录、偏好锁和
mDPO 运行证据。该操作不改动任何 Week 6 终态产物。

## 未完成项和真实原因

corrected development 的 multitask 与 Week 6 routed 人工四维均完成 24/24。Week 6
三个冻结 adapter 在同一 corrected 24 条上按 8/8/8 路由生成，job `29491047` 完成且
失败 0。multitask/Week 6 四维总均值为 4.59375/4.56250，配对差 +0.03125，样本级
10 胜/7 平/7 负；图片指代、需求调整、上下文承接、逻辑连贯差值分别为
-0.125/+0.291667/-0.041667/0。两轮由同一真实操作者在同一 session identity 独立录入，
Agent 未复制或代填；该小差异只作描述性结果，不宣称统计显著。仍存在的限制是
v3 final-test 对话构造
缺陷不可逆；本次 development 人工完成不会重开 test 或改写历史 test 对话结论。
DPO 已执行一次但 validation 门禁失败并正确拒绝新 adapter。三个核心场景训练、
checkpoint 选择、参数锁和一次性 test 已完成，不存在伪造 GPU、人工审核或指标。
corrected-dialogue v4 fix1 的 GPU 训练已完成，但 development 自动门禁失败；selector
写出了不可覆盖的阻断 evidence 而没有 selected checkpoint，一次性 fix1 test 未提交。
这是真实终态，不以历史 v3 test、首版 v4 development 或人工 development 结果替代。
因此不存在 fix1 Week 6/zero-shot test 对比，分支不快进 `dev`，不进入 `stg`，不打标签。
