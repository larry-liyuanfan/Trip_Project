# Week 7 多任务混合微调与上下文搭建执行报告

## 数据锁和隔离结果

Week 7 从提交 `132779b0f6d2929ce1cdbed18e62adf3ef9edd18` 建立独立分支和新数据身份。
成功锁为 `week7_fresh_multitask_context_20260819_v1`，锁 SHA-256 为
`a0e39f56da3385f71353987f3053f23a46e95b5da28ead80d6ceda0bfbfa3916`。
train/development/test 分别为 3000/114/114；sample_id、source_id、image_sha256、
group_id、constraint_template_id 五维跨分区碰撞均为 0。无 test 验证实际读取 train
3000 和 development 114，返回 `test_consumed=false`。

Week 3 v1/v2 exclusion 和 Week 6 已消费 79,936 条身份均进入排除检查。Week 5 权威
10,000 条对话（含 100 条真实 human accepted）全部与 Week 6 已消费来源碰撞，Week 7
可用数为 0，因此没有继承这些数据或人工身份。Week 7 新数据全部标为
`programmatic_silver`，真实 human accepted 为 0。

## 实际数据配比

训练集三个核心场景各 760 条，共 2,280 条；通用多模态正则 270 条，占 9%；新多轮
对话 450 条，占 15%。development/test 各含三个核心场景各 30 条和对话 24 条。
对话为 5–8 轮，图片只在首次用户轮出现，含固定工具调用格式样本；24 条 development
人工队列保持 `PENDING_REAL_HUMAN_INPUT`。

## Schema 解码对照

已实现同一 development 身份上的 free/constrained OpenAI-compatible JSON Schema
对照，严格保留原始输出且不提取、不修补 JSON。输出口径仅含 JSON 合规率、Schema
覆盖、延迟和失败回退，代码禁止生成语义提升结论。实际 GPU/服务端对照尚未运行，状态为
`PENDING_EXTERNAL_GPU_ACCESS`，因此没有格式提升数字。

## 多任务训练和 checkpoint

配置已锁定为 Qwen3-VL-8B、NF4 4bit double quant bf16、LoRA
`r=16/alpha=32/dropout=0.08`，注意力层与视觉投影层，学习率 `1.5e-4`、weight decay
`0.03`、max grad norm `1.0`、gradient checkpointing、2 epochs、有效 batch 16。
每约 10% 更新步对完整 development 生成并评分一次，保存原始输出和 checkpoint；按
三场景 0.30/0.35/0.35 加权综合分选优，连续 2 次无提升早停。训练尚未提交到 GPU，
状态为 `PENDING_EXTERNAL_GPU_ACCESS`，没有 checkpoint、loss、显存或完成时长可报告。

## 三场景及对话指标

Week 6 adapters 的新 development 基线、统一模型指标和对话自动指标均未实际生成，状态
为 `PENDING_EXTERNAL_GPU_ACCESS`。对话格式合规率和上下文召回率只有实际推理后才会
输出；连贯性、逻辑合理性和 OTA 专业性必须等待真实用户填写 24 条固定队列，当前为
`PENDING_REAL_HUMAN_INPUT`。

## Week 6 与零样本对比

比较实现已绑定新 development/test 身份及支持数口径，但由于没有获得本轮 GPU 输出，
Week 6 单场景 adapter、统一多任务 adapter 和零样本基线之间没有可报告的绝对或相对
变化。2% 单任务非回退门禁尚未执行，不能宣称通过。

## DPO

真实质量、视觉证据和 Schema 审核通过的 chosen/rejected 偏好对为 0。按预注册门禁，
唯一允许的 mDPO/HDPO 风格消融记为 `SKIPPED`；未使用同一模型自举结果冒充偏好数据，
且该跳过不阻塞 SFT 工程链。

## 测试结果

定向测试 5/5、完整 `python -m unittest discover -s tests -v` 375/375、train/development
锁验证、两份 Slurm shell 的 `bash -n` 和 `git diff --check` 均通过。本机 GPU 环境检查
返回 `missing_dependencies`，缺少 torch、torchvision、transformers、accelerate、peft、
bitsandbytes 和 kernels，因此没有把本机计作训练环境通过。

## Commit / push 状态

初始执行链提交为 `31f7174ef7874da7084223aa71af0c59bad03983`，已推送到
`origin/codex/week7-multitask-context`。最终文档和锁摘要将以独立提交推送。由于 GPU
验收未完成，没有快进 `dev`；未进入 `stg`，未打标签。

## 未完成项和真实原因

既有 Spartan Trip OOD 终端在本次执行时被另一浏览器控制会话占用。按本项目终端保护
规则，不新建重复终端、不绕过既有会话、不提交竞争作业。因此 development 基线、Schema
实跑、统一训练、checkpoint、参数锁和一次性 test 均保持
`PENDING_EXTERNAL_GPU_ACCESS`；test 从未运行。对话人工四维评分保持
`PENDING_REAL_HUMAN_INPUT`。除此之外没有虚报 GPU、人工审核或指标。
