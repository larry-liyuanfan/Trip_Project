# 价格冲突 abstention：前置证据阻断记录

日期：2026-09-14。状态：`BLOCKED_PROTECTED_IDENTITY_COVERAGE`。
本轮完成身份登记审计、18 项定向测试和文档一致性修复；未构造新数据、未训练或推理，
不是价格识别改善，也不是一次模型效果负实验。已有 scorer v2 修复保持不变。

## 当前情况与中断判断

Iris SSH 正常。2026-09-14 04:54:32 UTC 的账户级 `squeue` 为空；`sacct` 核对旧作业：

| 作业 | 状态 | 耗时 | ExitCode |
| --- | --- | --- | --- |
| `30044630` / v9 | COMPLETED | 00:09:07 | 0:0 |
| `30211170` / v10 | COMPLETED | 00:07:19 | 0:0 |

旧作业不需重提交；本轮提交数为 0。没有恢复旧 heartbeat、后台监控或其他账户作业。
此前 v10 在已查看的 v9 development 上修复渲染后仍有价格回退，结论留在原报告，
不能将其改称新的独立开发集或测试提升。

## 已执行的前置审计

入口为 `scripts/audit_price_conflict_prerequisites.py`，配置为
`configs/evaluation/price_conflict_abstention_20260914.json`。检查只接受绑定 SHA 的平面身份
元数据，不允许 gold、prompt、prediction、嵌套样本或 image bytes；不扫描 raw 文件作为 fallback。
来源、图片、query 不可相互冒充；sample/group/template ID 不能替代内容 SHA。
工具输出只保留计数与哈希，不复制身份值。

12 项登记范围中，10 份现有文件哈希校验通过，但 0/12 满足声明的完整内容维度覆盖。
两项尚无完整可用登记。10 份文件共 96,788 行是登记行数之和，存在跨历史版本重复，
不是新增数据、唯一样本数或模型评测分母。

| 已有登记 | 行数 | 具体缺失 |
| --- | ---: | --- |
| system-repair fresh v2 | 2,268（train 1,980 / dev 168 / 已消费 test 120） | query SHA、source record 内容 SHA |
| evaluation exclusion v1 / v2 | 450 / 450 | query SHA、source record 内容 SHA |
| Week 6 split manifest | 79,936 | 只有 sample/scenario/split，缺 source、image、query 及 source 内容 SHA |
| Week 7 v3 / v4 / fix1 / fix2 | 各 3,228 | query SHA、source record 内容 SHA |
| Week 8 已有 audit exclusion 子集 | 460 | query SHA、source record 内容 SHA；不代表全部 Week 8 数据 |
| v4 image asset registry | 312 | 图片登记不含 query/source 内容 SHA，且不覆盖对话查询 |

v5 的完整逐行身份索引未在允许范围内找到；提交的 v5 lock 只有聚合 SHA，不能用来计算
任意新 query 的重叠。v4 对话、v7/v9/v10 已看开发池及其余 Week 8/弱查询池的完整覆盖
没有得到认证；发现上述上游缺口后停止继续构造完整管线。

查找范围是当前 91ac、旧 b8bb 的项目 outputs、正式本地 outputs，以及 Iris 已有
`Trip_Project_yzhang3504/20260812a` 项目内有界的 identity/registry/归档文件名。没有查找
其他账户、穷举服务器，或打开/重建封存 final。v5 历史脚本将 pool 放在 job-local 空间，
持久化 run 目录未找到该 registry；这只说明当前允许范围未找到，不声称全服务器已删除。

Fresh Test 只读取原有 identity manifest 和 dataset lock，不读取样本、标签或预测。
原 lock 的 `LOCKED_UNCONSUMED` 是历史状态，不覆盖后来的已消费记录。
所有新数据 overlap 数保持 `null/NOT_RUN`，不是 0。

## 运行前协议与事实边界

预注册固定 v9 adapter `6332a2baa03d88751c022cb6efdd9dbfb21e6f88f32f31f4d9cb775a0bf38f17`，
保留基座 revision、Prompt、生成和训练超参；若之后取得允许的完整身份证据，才可另经
总控调度继续落实数据锁、实际图像 QA 和同一新 development 的配对对照。

比较名称限定为“包含冲突样本的续训方案 vs 原 v9”。候选多出优化步骤，即使其他超参
固定也不能严格归因为样本组成单因素；本轮不增加等步数对照或第二次训练。
计划规模和质量阈值均标为尚未生成/未运行的预注册目标，不是运行产物或模型指标。
复用同一 renderer/template 家族也不支持真实图片或人类金标上的泛化结论。

未执行项：新 train/dev 生成、source/image/query 内容排重、新图片目视 QA、训练、逐样本
推理、质量门判定及 GPU 性能。本轮不开展新的性能矩阵，也不改变正式 adapter/release。

## 验证与提交

- 文档独立提交：`c5d59556f3bbdf658f9dacae74fde3369e4b8f4e`，仅六份获准 Markdown；旧报告、
  旧实验 JSON、评分源码不动。旧 gate、历史 521 项、scorer v2 固定源码 1013 项分时点标记；
  发布决策不改变检索标注来源，亦不抹去其他历史 VLM 人标集。
- 审计源码提交：`f2eccec376b12fbb72c93e68e071803d8af19e39`。
- 定向 `unittest`：18/18 passed，失败 0、skip 0，0.198 秒；没有重跑历史 1013 项全量。
- 独立 `hashlib` 与字段计数复核：10/10 文件一致；post-commit 审计产物字节完全一致。
- CLI 的 Python exit code 为 2，代表预期前置阻断；不是运行成功或模型 gate FAIL。
- 本地 CPU 为 i7-14700HX（20 核/28 线程），Windows 11 10.0.26200，Python 3.13.13；无 GPU。

精确 SHA、配置、分母与命令见
[机器证据](../../../experiments/price_conflict_abstention_20260914.json)、
[身份审计](../../price_conflict_abstention_20260914/identity_inventory.json)及
[交接](../../price_conflict_abstention_20260914/HANDOFF.md)。CPU 槽已释放，无运行中子进程或待恢复 GPU 作业。

## 解除阻断所需材料

需要数据保管侧提供经过原 manifest/lock 绑定的 **identity-only** 导出，覆盖缺失的
source 内容、query 内容、图片/纯文本适用性及全部受保护池；须记录内容规范化和导出来源。
不能靠改 ID、目录、卡片标题或噪点获取“独立”资格；不能修改旧锁后复用已消费测试。
本轮不自行打开 sealed final 补索引。材料与后续调度到位前不提交训练，未声称价格回退已经修复。
