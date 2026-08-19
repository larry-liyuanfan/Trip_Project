# Week 7 多任务混合微调与上下文搭建执行报告

## 数据锁和隔离结果

执行分支从 Week 6 终态 `132779b0f6d2929ce1cdbed18e62adf3ef9edd18` 建立，旧
`agent/portfolio-positioning` 工作树未参与开发。活动数据身份为
`week7_fresh_multitask_context_20260820_v3`，配置 SHA-256 为
`d77d9f10b551f30c599572e974fba2c3c2af087f37ed35e93b9dc7ac2dc105fa`，数据锁
SHA-256 为 `8af2e2d13c22fb641fc7344b1e56e5827aa78b1ebde653c6e55c83b36d20504d`。
train/development/test 为 3000/114/114；sample_id、source_id、image_sha256、group_id、
constraint_template_id 五维跨分区碰撞均为 0。development 对话父任务为商品/售后/行程
各 8 条，train 为各 150 条；锁验证只读取 train/development，`test_consumed=false`。

v2 锁因 train/development/test 对话父任务全部来自商品场景而作废；对应 GPU 作业
`29431992` 在审计发现后取消，三个已生成 checkpoint 均禁止进入参数锁。Week 3 v2、
Week 6 训练来源与三分区保持排除；Week 5 的 100 条真实验收对话未继承人工身份。

## 实际数据配比

train 含三个核心场景各 760 条，共 2280 条；通用多模态正则 270 条（9%）；5–8 轮对话
450 条（15%），其中工具调用格式 45 条（对话内 10%）。对话图片只在首个用户轮出现，
采用结构感知截断。全部 Week 7 标签仍为 programmatic silver；24 条固定人工评估队列
保持 `PENDING_REAL_HUMAN_INPUT`。

## Schema 解码对照

作业 `29434316` 在锁定的 `Qwen/Qwen3-VL-8B-Instruct` 上完成，每个 mode 支持数 90。
free JSON 合规率 98.89%、Schema 覆盖 0%、请求失败率 0%、平均延迟 3300.72 ms。
constrained primary 的 90 次请求均被服务端拒绝，故 JSON 合规率和 Schema 覆盖均为 0%；
90 次真实 free fallback 全部成功，fallback 失败率 0%，包含 fallback 的平均延迟
3360.34 ms，配对延迟比 1.0181。生产模式因此固定为 free。该结果仅说明格式与服务端
约束兼容性，不代表任何语义提升；primary 与 fallback 原始输出分开计分和哈希绑定。

## 多任务训练和 checkpoint

统一 SFT 作业 `29434317` 正常完成，耗时 01:30:17。配置固定为 NF4 4bit、
LoRA `r=16/alpha=32/dropout=0.08`、注意力层与视觉投影层、学习率 `1.5e-4`、weight
decay `0.03`、max grad norm `1.0`、gradient checkpointing、2 epochs、有效 batch 16。
计划更新步 376；实际完成 38/76/113/151 四个评估点后，综合分连续两次未提升并按
patience=2 在 step 151 早停。四个 development raw/metrics/checkpoint 均已哈希绑定，
adapter-only 回载验证通过，峰值 allocated/reserved 显存为 14.82/21.52 GB。Trainer 的
最高综合分 checkpoint 为 step 76，但独立 selector 返回
`BLOCKED_NO_ELIGIBLE_CHECKPOINT`，没有参数锁可用。

## 三场景及对话指标

新 development 上 Week 6 路由 adapters 的三场景 composite（支持数均 30）为：商品
0.041176、售后 0.100000、行程 0.048333；全 114 条加权综合分 0.064270、失败率 0%、
平均延迟 5830.99 ms。24 条对话按三场景 8/8/8 路由，格式合规率 45.83%、上下文召回率
49.65%。统一模型最高综合分出现在 step 76：商品/售后/行程 composite 为
0.564706/1.000000/1.000000（支持数各 30），全 114 条综合分 0.869412、失败率 0%、
平均延迟 8324.26 ms；对话格式合规率 95.83%、上下文召回率 96.53%。对话人工四维
评分仍为 `PENDING_REAL_HUMAN_INPUT`，Agent 未代填。

## Week 6 / 零样本对比

development 零样本全 114 条加权综合分 0.070147、失败率 0%、平均延迟 1961.06 ms；
商品/售后/行程 composite 分别为 0.058824/0.100000/0.050000，对话格式合规率 54.17%、
上下文召回率 53.13%。step 76 相对 Week 6 的三场景 composite 均未回退，格式和失败率
门禁通过，但全局延迟比为 1.4276，超过 1.25 上限；商品 `label_completeness` 支持数也
低于基线，因此 selector 的 4 个候选均不合格。参数锁未创建，一次性 test 未读取，
故不存在可报告的 test 三方绝对/相对变化，也不能宣称最终 2% 门禁通过。

## DPO 执行状态

真实质量与视觉证据审核通过的 chosen/rejected 偏好对为 0，唯一允许的 mDPO/HDPO
风格消融按门禁记为 `SKIPPED`；没有自举或伪造偏好对，且不阻塞主要 SFT。

## 测试结果

当前代码完整 `python -m unittest discover -s tests -v` 为 401/401 PASS；compileall、
五份 Week 7 Slurm 脚本 `bash -n`、数据锁验证和 `git diff --check` 均通过。Spartan
训练环境实测为 L40S、torch 2.8.0+cu128、Transformers 4.57.1、PEFT 0.17.1、
bitsandbytes 0.47.0，环境门禁通过。

## Commit / push 状态

证据与恢复门禁提交 `bb6ecfeb2f31fdb0fb3d07bfcdde8efcb1ba443d` 已推送到
`origin/codex/week7-multitask-context`；最终 GPU 结果、阻断状态和文档也已在收尾提交
推送。因自动验收未全部通过，不快进 `dev`，未进入 `stg`，未打标签。

## 未完成项和真实原因

训练已完成，但 selector 因全部 4 个 checkpoint 的全局延迟和商品支持数门禁失败而
`BLOCKED_NO_ELIGIBLE_CHECKPOINT`。因此参数锁与一次性 test 按规则未执行，test 保持
未消费；这不是 GPU 或代码未完成，而是预注册验收未通过。24 条对话人工四维评分需要
真实用户输入，保持 `PENDING_REAL_HUMAN_INPUT`。不存在伪造 GPU、人工审核或指标。
