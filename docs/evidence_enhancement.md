# 搜索、VLM 与端到端证据增强协议

本协议属于 `dev` 开发证据，不修改正式 release
`trip-qwen3-vl-8b-week8-final-v1`，也不重新消费已锁定的 Fresh Test 120。所有新输出必须
写入不存在的目标文件；正式四层包只读，归档或成员哈希不匹配时立即失败。

## v4 与 v5 已完成证据

v4 使用 `configs/evaluation/automated_evidence_v4.json` 和三向锁
`configs/evaluation/evidence_enhancement/exploration_pool_lock_v4.json`。搜索与 VLM 的
training/development/final 均要求 source ID、image SHA 及 query/sample ID 两两零重叠；
final 在 development gate 通过后先写 exclusive marker，再第一次读取。该轮全部
标签是 deterministic synthetic，human support=0，不用于正式 release 晋级。

v4 搜索固定比较 exact CLIP、CLIP+Milvus、结构化硬过滤+CLIP、硬过滤+
轻量重排和带业态 guard 的候选。它将 ANN-vs-exact Recall@10 另列为实现
保真，业务指标另报 Recall@K、MRR@10、nDCG@10、no-result slice、filter
correctness 和各类别分母。最终 synthetic 一次性集 n=24（ranking=12、
no-result=12、hard-filter=16）上，业态 guard 候选的 MRR@10、nDCG@10、
no-result accuracy 和 filter correctness 均为 1.0；ANN-vs-exact Recall@10=0.9917。
这些数字只是 synthetic 规则标签结果。

v4 原报告中的 query-vs-index byte collision 曾因正式 retrieval 压缩包不含原图而记为
`NOT_RUN_MISSING_INDEX_IMAGE_SHA`。后续只读审计使用与该 archive metadata SHA 完全一致的
Git 外 Yelp overlay，对正式索引 1000/1000 原图及 v4 training/development/已消费 final
各 24 张 synthetic 查询图重新哈希。job `30046716` 得到查询覆盖 72/72、索引覆盖
1000/1000，byte collision=0、source-identity collision=0，独立 verifier=`PASS`。
它未读取 annotation、ranking 或 Fresh Test，只把数据隔离状态从未知补成可验证 PASS；
机器摘要见 `experiments/retrieval_query_leakage_evidence_v4.json`。

v4 VLM 在同一新 development 锁上比较 zero-shot、旧 unified adapter、checkpoint-87
和 targeted adapter，每角色 n=36（商品 24、对话 12）。targeted adapter 的业态/
风格/设施/价位 F1 分别为 1.0/1.0/1.0/0.6667，unknown abstention=1.0、
unsupported hallucination=0、first-attempt JSON=1.0；但 context recall=0.4167，低于
预锁 0.6，因此 v4 VLM final 保持未消费。

v4 服务性能使用真实 loopback HTTP FastAPI/Uvicorn、外部单节点 Milvus 2.6.18
standalone 和一张 L40S，顺序比较 checkpoint-87 与 targeted adapter。每角色 1 次
cold、2 次 warmup，并发 1/2/4 各 8 个 batch，steady 分母每角色 56 请求。
候选 c=1 HTTP P95=1210.95 ms，基线=558.54 ms，比值 2.168>1.25，故保留为
负实验。Milvus 查询约 2.5 ms 只是阶段耗时，不是 HTTP 端到端延迟；
该部署也不是 multi-node distributed Milvus。

v5 上下文专项在任何运行前固定于
`configs/evaluation/automated_evidence_v5.json` 和
`configs/evaluation/evidence_enhancement/context_focus_pool_lock_v5.json`。它从 v4 adapter 继续
训练，保持基座 revision、Prompt、优化器、学习率、epoch 和 seed 不变，只改变
主要因素“上下文专项 synthetic 训练数据组成与支持数”。新锁为 training
528（商品 144、对话 384）、development/final 各 48（商品 24、对话 24）；
source、image、sample 及 dialogue-text SHA 三向零重叠。开发门槛包含 context
recall 至少 0.6 且相对 v4 提升至少 0.1，其他对话指标回退不超过 0.1。
只有开发质量通过，才会在同一硬件上运行 v4→v5 真 HTTP/Milvus 延迟对比；
只有质量和 c=1 HTTP P95 比值≤1.25 同时通过，才允许消费 v5 final。

实际 job `29998754` 严格按以上顺序完成。development（每角色 n=48）上 v5 的 context
recall 从 v4 的 7/24 提升至 24/24，state、task key/value 与 first route 也均为 24/24；
价位 F1 从 9/12 提升至 12/12，unknown abstention=36/36、unsupported hallucination=0/36。
同一 A100 80GB 上，真实 HTTP + 外部单节点 Milvus 的 c=1 steady P95 为 v4
1046.03 ms、v5 1030.30 ms，比值=.985；c=1/2/4 共 56 steady 请求/角色，失败率均为 0。
两个门均通过后，v5 final 只消费一次；n=48 的协议指标全部通过。完整分母、阶段延迟、
硬件和产物 SHA 见 `experiments/context_focus_evidence_v5.json`。全部质量标签仍是 synthetic，
human support=0；结果不代表人工视觉准确率、真实用户业务相关性、分布式 Milvus 或生产 SLA。

## v6、v7、v8 与 v9 后续证据

v7 使用 `configs/evaluation/automated_evidence_v7.json` 和独立数据锁，只改变主要因素
“语义鲁棒性 synthetic training 数据”。training=512（商品/对话各 256）；新的 development
每角色 96（商品/对话各 48），与 training、v5 development/final 在 source、image、sample、
query/dialogue identity 上零重叠。候选相对固定 v5 baseline 的四字段 F1、exact、unknown、
hallucination 和五项对话指标均改善；但 multi-subject conflict abstention 为 5/8=.625，低于
预锁 .75，故 development gate=`FAIL`。它必须作为负实验保留；服务 c=1/2/4 按串行门控为
`NOT_RUN_DEVELOPMENT_GATE_FAILED`，没有 final，也没有新的性能结论。机器摘要见
`experiments/semantic_robustness_evidence_v7.json`。

v8 使用与索引图分离的 40 条 synthetic calibration 和 40 条一次性 validation，并与 v4
training 24 条做 source/image/query 三向碰撞审计。固定 v4 margin guard 与新 dual-centroid
guard 在 validation 上得到相同的 no-result 17/20、business-positive acceptance 18/20、
nDCG@10=.9 和 filter correctness=1；两者均通过预锁压力门，但新候选相对固定基线增益为 0，
因此结论是“既有 guard 获得新的 synthetic 复验，新方法为中性实验”，不是算法提升。
该轮不测 ANN fidelity，也没有 final。机器摘要见 `experiments/no_result_stress_evidence_v8.json`。

v9 在任何运行前固定于 `configs/evaluation/automated_evidence_v9.json` 和
`configs/evaluation/evidence_enhancement/semantic_robustness_pool_lock_v9.json`。它以 v7
Adapter SHA `a06742eb…b5b1` 为固定基线，保持 Qwen revision、Prompt、generation、optimizer、
学习率、epoch、seed、质量阈值与 objective 不变，只增加并强化 multi-subject counterexample
training 构成。training=640（商品 384、对话 256），全新 development=132（商品 84、对话
48，其中 multi-subject=24）；两者互相隔离，并同时与 v5 training 528 条及 v7 training+
development 608 条在 source/image/sample/dialogue-text 身份上零重叠。

job `30044630` 已完成。候选相对 v7 baseline 的 objective 从 .96189 提至 .97517，业态 F1
.9895→1、价位 .8936→1、设施 .9074→.9815，unsupported hallucination 2/192→0/192，
对话 state value 42/48→48/48；但 style F1 由 1.0 降至 .8571，回退 .1429 超过预锁
.05，固定 development gate=`FAIL`。新显式 multi-subject slice 上两角色均为 24/24，因而没有
可归因的相对提升。程序按串行门控未运行候选单机 HTTP，也没有 final。完整分母与 SHA 见
`experiments/semantic_robustness_evidence_v9.json`；正向观察不能覆盖整体负实验结论。

v6 首次双节点 A100 / distributed Milvus 作业 `30004826` 为工程负实验：两个节点及五类
Milvus 角色均启动，但 Milvus 自动选择并通告了控制节点上一张不可跨节点路由的接口；worker
到 mixcoord 的内部 gRPC 持续超时，`create_collection` 阻塞至 30 分钟作业时限。该 run 没有
产生 `summary.json`、`raw.jsonl` 或任何 HTTP 请求，所以 c=1/2/4 仍是 `NOT_RUN`，不能从
cluster identity 的 `READY` 或 61.7 秒启动观测推导性能。修复版显式使用 Slurm 主机名解析的
IPv4，并在写入 `READY` 前执行五个双向跨节点 TCP probe；它必须写入新 run，旧失败不覆盖。
repair-1 job `30042086` 在两个 A100 节点上 `COMPLETED 0:0`。五个跨节点 probe 全部通过，
Milvus 2.6.18 的 mixcoord/proxy 在 control，querynode/streamingnode/datanode 在 worker；
cluster cold startup=65.06 s，1000 向量 collection build/load=15.67 s。固定 synthetic
training 请求上，v4/v5 每角色 1 cold、2 warmup，c=1/2/4 各 8 batch，steady 分母每角色
8/16/32、共 56（raw rows=112），失败率均为 0。v5/v4 c=1 HTTP P95 比值为
1.019≤1.25，固定性能门通过。v5 c=1 steady 的 HTTP P50/P95 为 1055.00/1059.76 ms，
CLIP 为 7.17/8.64 ms、Milvus 为 3.65/3.82 ms、rerank 为 .031/.036 ms、VLM 为
1038.60/1042.68 ms。独立 verifier=`PASS`，机器摘要见
`experiments/distributed_milvus_http_evidence_v6.json`。这补齐了双节点实跑与分阶段计时，
但仍不支持生产 SLA，也不把 3.82 ms 向量阶段写成约 1.06 s 的端到端延迟。

## 自动化 v2 / weak v3 预运行锁

第二轮自动化证据使用 `configs/evaluation/automated_evidence_v2.json` 和
`configs/evaluation/evidence_enhancement/automated_pool_lock_v2.json`。实现从
`03c23b4f0597d2f2fef073951fd7670e0cb51c87` 开始；修正后的搜索 gate 绑定 `77dd052`，最终
固定长度性能运行绑定实现提交 `85eb519ef074065f26ca9d3d3c184fc03e363719`。每一阶段都先从
对应提交生成并逐文件验证 source snapshot；任何结果均不得回写阈值、切分或数据锁。该轮
明确排除人工标注/仲裁，human support 固定为 0。

`scripts/build_automated_evidence_pool_v2.py` 以确定性 PPM 字节生成 32 条搜索查询（calibration
与 holdout 各 16 条）和 18 条 VLM 样例（12 商品、6 对话）。两个搜索 split 的 source ID、
image SHA、query ID 必须完全不重叠；生成后分别核对 query/annotation/file canonical SHA。
holdout 只在 calibration 选出 no-result 阈值和轻量 star-rating 权重后写入 exclusive consumption
marker，并只执行一次。v1 的 10 条 Commons 查询只作为历史开发证据，不参与 v2 选择。

搜索 v2 固定比较 exact CLIP、Milvus、结构化过滤、hard-filter-before-light-rerank。calibration
网格和选择目标在配置中预先固定；holdout 报 Recall/MRR/nDCG、no-result、filter、失败率与
切片分母，负结果照常保留。Milvus 运行形态是本地文件 Milvus Lite，不代表分布式服务。

VLM weak v3 的三角色仍只允许 adapter 变化，新增可见数值价位、等显著多主体冲突、证据不足
和对话状态样例。价位映射规则、Prompt、generation config、图片字节与 manifest 均在运行前
锁定。它是 synthetic/weak development probe，不能替代 Fresh Test 或人工视觉真值。

性能矩阵按 `short_32`、`medium_64`、`long_128` 三个固定 profile 分别启动独立进程；输出通过
相同的 `min_new_tokens=max_new_tokens` 强制为 32/64/128 token，输入通过预锁定 padding 形成三个
不同的实际 token 长度。矩阵记录真实 process cold（含模型/索引 startup）和 warmup 后 steady
的分阶段 P50/P95/min/max、VRAM、QPS、失败率与实际 input/output token 数。当前隔离环境没有
standalone Milvus、vLLM、FastAPI 或 Uvicorn，因此只运行 concurrency=1 的 transformers +
Milvus Lite component pipeline；concurrency=2/4 及 distributed Milvus/HTTP 单元必须为
`NOT_RUN`，不得形成生产 SLA 声明。早期仅改变 `max_new_tokens`、但实际都提前停止在相同长度的
六个作业标记为 `SUPERSEDED_IDENTICAL_REALIZED_LENGTHS`，不计入最终矩阵。

泄漏审计由 `scripts/audit_retrieval_query_leakage_v2.py` 完成：按 formal metadata 的 image ID 和
source path 对封闭 1000 图 overlay 原位哈希，只输出不含图片字节的 registry；再将可用的
`project/repo` 副本作为重叠交叉核验。正式 overlay 的 1000/1000 原图全部原位哈希后，主碰撞
结论才可为 `PASS_COMPLETE_NO_QUERY_INDEX_COLLISION`；副本覆盖不足单独记录
`UNKNOWN_INCOMPLETE`，它只是交叉核验边界，不降低完整 overlay 主分母。已覆盖部分一旦字节
不一致则整体失败。

## 四条互不替代的证据轨道

| 轨道 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| ANN-vs-exact | HNSW 返回集合对精确余弦 TopK 的保真度 | 结果对用户是否有用 |
| 业务语义相关性 | 独立查询在弱标注或人工分级下的 Recall、MRR、nDCG、过滤与无结果行为 | 人工视觉准确率，除非标签协议真的完成双人标注与仲裁 |
| VLM/SFT 语义 | 固定数据、Prompt、基座、生成参数后，仅改变 adapter 的字段和对话指标 | 正式发布晋级；小弱池不能替代冻结测试 |
| 端到端性能 | 同硬件上 CLIP 编码、Milvus、轻量重排、VLM 和总耗时 | 历史 2.41 ms 向量查询不能替代该结果 |

历史 Milvus `Recall@10=1.0` 的固定口径是 100 个自查询/原型查询上的
`ANN-vs-exact`；历史 P95 `2.409699955 ms` 只覆盖 Milvus vector query。两者不得写成独立
业务相关性或端到端延迟。

## 独立搜索查询池

查询锁位于 `configs/evaluation/evidence_enhancement/query_manifest_v1.jsonl`，包含 10 个查询、
5 个独立 Wikimedia Commons 图片资产和以下切片：

- 同类视觉检索；
- 城市、业态、设施和价位组合；
- 视觉相似但业务不相关；
- 无结果；
- 图片语义与显式过滤冲突。

每条记录保存来源页面、稳定文件解析入口、许可、作者、图片 SHA-256、来源记录 SHA-256 和
完整查询 SHA-256；`query_asset_source_registry_v1.jsonl` 另将精确 960px 字节 URL、尺寸和
同一 SHA-256 绑定。查询来源明确为 `independent_public_source_not_yelp`。正式检索包只有向量和
metadata，没有索引原图 SHA，因此来源隔离可以验证，逐字节索引/查询碰撞检查只能记录为
`NOT_RUN_MISSING_INDEX_IMAGE_SHA`，不能记为通过。

当前 `search_annotations_weak_v1.jsonl` 是 metadata 规则弱标注：相关性为 0–3 级，等级 2
及以上计为相关。它不是人工标注。若将来晋级为人工证据，每条查询至少需要两名人工标注者；
无分歧记 `none`，有分歧必须记 `adjudicated`。程序会拒绝单人标签冒充 human，也会拒绝查询
与标签支持数不一致。

搜索比较固定为：

1. CLIP 精确余弦；
2. Milvus HNSW/COSINE；
3. 城市、业态、价位结构化过滤后 CLIP；
4. 固定 Top-50 候选上的轻量 metadata 重排。

设施不在正式检索 metadata 中，必须作为 `unsupported_constraints_unapplied` 披露。报告同时
给出 Recall@5/10、MRR@10、nDCG@10、无结果率/准确率、过滤正确率、失败率、切片支持数和
实际分母。

## VLM/SFT 角色与数据锁

历史 168 条 development 原始输出只用于 recomputation/audit；它已经参与历史选择，不得
称为本任务新 development。历史主角色身份固定为：

- `zero_shot`：无 adapter；
- `old_unified_adapter`：Week 7 checkpoint-226，adapter SHA-256
  `ccc6062f7e451b9265c571c0df397903cbbc707a6bf2e894039079175e5f24ee`；
- `current_system_repair_checkpoint_87`：从 checkpoint-226 continuation 得到，adapter SHA-256
  `c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a`。

Week 6 image-product 单任务 adapter 只能作为额外历史诊断，不能冒充旧 unified 或当前多任务。

本任务的新 `vlm_manifest_weak_v2.jsonl` 与历史五维 identity 分离，含 5 条 Commons 商品弱标签
和 3 条 synthetic 对话协议样例。三个主角色必须共享完全相同的 manifest hash、基座 revision、
Prompt hash 和 generation hash，只允许 adapter 因素变化。该池规模很小、标签为 weak/synthetic，
所以只能形成诊断结果，不能用于晋级。

早期 v1 manifest 因只绑定相对路径、未在 VLM 锁内绑定图片 SHA，被保留为
`REJECTED_MISSING_ASSET_BYTE_BINDING`；它的运行结果不得与 v2 混分。

商品指标包括业态、风格、可见设施、价位的微平均 precision/recall/F1、逐样本 exact match、
unknown 字段幻觉率、首次严格 JSON 合规和纠错触发率。对话指标包括上下文召回、state/value、
task key/value 和首轮路由。历史 raw 只保存最终输出时，首次尝试与纠错必须标为
`NOT_RECORDED`；最终 JSON 合规不得冒充首次尝试合规。

历史 Fresh Test 商品综合 `0.780639` 在报告中保留为已发布聚合值。当前本地 Git 外正式
evidence 包没有对应 120 行 raw/metrics，因此本地 handoff 状态仍是
`EVIDENCE_GAP_RAW_SAMPLE_OUTPUTS_NOT_IN_LOCAL_HANDOFF`。只读限域检查确认 Iris 历史
system-repair 项目保留了已消费一次的 120 行 raw 和 metrics；
`scripts/audit_system_repair_final_test.py` 只对该固定产物做离线身份、逐字段与错误切片重算，
不加载模型、不生成新输出、不调参，也不产生任何新的晋级资格。

## 端到端性能

性能 raw 每行必须包含 `clip_encode_ms`、`milvus_ms`、`rerank_ms`、`vlm_ms`、
`end_to_end_ms`、峰值 VRAM、失败状态和硬件身份。协议固定 1 次冷请求、3 次不计分 warmup、
30 次稳态请求；报告 P50/P95、范围、吞吐和失败率。冷请求额外包含模型/索引 startup，不能与
稳态混合。

候选和旧 unified baseline 只有在硬件身份、冷/稳态支持完全一致时才比较。固定门禁为：稳态
端到端 P95 不超过 12 秒、失败率不超过 2%、峰值 VRAM 不超过 8192 MiB、候选/基线 P95
比不超过 1.25。门禁在运行前锁定；缺阶段、缺支持或换硬件一律失败关闭。

## 运行入口

本地协议与正式只读包核验：

```bash
python scripts/run_relevance_evidence.py validate-pool \
  --asset-dir <query-assets> \
  --retrieval-archive <formal-release>/retrieval.tar.gz \
  --release-manifest <formal-release>/release_manifest.json
```

搜索、VLM、历史审计和端到端脚本分别为：

```text
scripts/run_relevance_evidence.py
scripts/run_vlm_semantic_evidence.py
scripts/audit_system_repair_development.py
scripts/run_end_to_end_relevance_benchmark.py
```

Spartan 作业位于 `scripts/spartan/`。它们不含私钥或用户主目录，要求调用者显式传入 Iris
项目根、独立 venv、任务 cache、输出和只读资产路径，并验证可写路径没有逃逸项目根。
四个实际评估作业还强制要求 `TRIP_SOURCE_MANIFEST` 与
`TRIP_SOURCE_SNAPSHOT_SHA256`；`scripts/verify_relevance_source_snapshot.py` 会在模型或数据
处理启动前验证 canonical 清单哈希及清单内每个 runner、配置和传递依赖的 byte SHA。最终
证据必须同时记录真实 `git_base_sha`、先于运行创建的 `implementation_commit_sha`、source
snapshot SHA 和 Slurm job ID；不得把作业完成后才创建的提交冒充运行来源。
