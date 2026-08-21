# Week 7 多任务混合微调与上下文搭建执行报告

终态：`CORE_AUTOMATED_ACCEPTED / DIALOGUE_CONTEXT_INVALID / HUMAN_REVIEW_BLOCKED`。
三个核心场景通过唯一一次正式 test 的自动非回退门禁；后续审计发现 v3 多轮上下文构造
错位，对话自动指标只保留为历史输出，不作为真实多轮能力结论，人工四维评分被阻断。

## 数据锁和隔离结果

执行分支从 Week 6 终态 `132779b0f6d2929ce1cdbed18e62adf3ef9edd18` 建立，旧
`agent/portfolio-positioning` 工作树未参与开发。活动身份为
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

train 含商品、售后、行程各 760 条，共 2280 条；通用多模态正则 270 条（9%）；5–8 轮
对话 450 条（15%），其中工具调用格式 45 条（对话内 10%）。对话图片仅在首个用户轮
出现，并采用结构感知截断。全部 Week 7 标签仍为 programmatic silver；24 条固定人工
评估队列未产生人工分数。构造器在 450/24/24 条 train/development/test 对话中先追加
assistant 回复、再追加其对应的 user 问题；v3 锁、训练和正式 test 产物保持不可变，
不得在原身份上重排后继续计分。

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

唯一 final-test 作业 `29459265` 在 L40S 上 `COMPLETED 0:0`，耗时 00:40:50；marker
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

## Week 6 / 零样本对比

统一模型相对 Week 6 的商品/售后/行程绝对变化为 +0.097436/+0.900000/+0.968333，
相对变化为 +172.73%/+900.00%/+3417.65%；相对 zero-shot 的绝对变化为
+0.076923/+0.900000/+0.946667，相对变化为 +100.00%/+900.00%/+1893.33%。最终
multitask/Week 6 全局延迟比约 0.8694，失败率均为 0。三场景任务、支持、JSON/Schema、
全局延迟和失败率门禁全部通过；没有把 development 指标冒充 test 结果。

## DPO 执行状态

真实质量与视觉证据审核通过的 chosen/rejected 偏好对为 0，唯一允许的 mDPO/HDPO
风格消融按门禁记为 `SKIPPED`。没有自举或伪造偏好对，不影响已完成的 SFT 自动验收。

## 测试结果

当前完整 `python -m unittest discover -s tests -v` 为 418/418 PASS；远端 final-runtime
定向测试 22/22 PASS。compileall、数据隔离和配置验证、七份 Week 7 Slurm 脚本
`bash -n`、`git diff --check` 均通过。

## Commit / push 状态

protocol-v5 提交 `64a5a7a` 与 final runtime 修复 `8619b76` 已推送至
`origin/codex/week7-multitask-context`；最终证据与文档由本次收尾提交继续推送。因真实
人工项未完成，不快进 `dev`；未进入 `stg`，未打标签。

## 未完成项和真实原因

24 条对话人工四维评分因源上下文错位保持 `HUMAN_REVIEW_BLOCKED`，不是等待用户继续给
无效样本打分。要恢复可信对话验收，必须建立新的版本化对话数据身份，修正首轮回答及
user→assistant 顺序，并对修正上下文重新生成模型输出；不得修改 v3 锁或复用旧 raw。
DPO 因 0 条真实审核偏好对正确 `SKIPPED`。三个核心场景训练、checkpoint 选择、参数锁和
一次性 test 已完成，不存在伪造 GPU、人工审核或指标。
