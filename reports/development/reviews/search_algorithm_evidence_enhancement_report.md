# 搜索算法、VLM/SFT 与端到端证据增强报告

日期：2026-09-04。当前结论：v4 synthetic 搜索 development/final gate 通过；v4 VLM
因对话 context recall 未达 0.6、且相对 checkpoint-87 的真实 HTTP 延迟比为 2.168 而保留
为负实验。随后预注册的 v5 上下文专项在新 development 上通过质量门和相对 v4 的延迟门，
并只消费一次独立 synthetic final。正式 release 不变、Fresh Test 120 未读取/未重跑，
human support=0。此后 v7 鲁棒性候选因多主体门槛失败保留为负实验；v8 no-result 压力门
通过但新方法与固定基线持平；首次双节点 Milvus 因网卡通告错误超时且没有 HTTP 分母。
后文 v2/v3/v1 均为历史证据，不得覆盖本节。

## v4 当前最新结果

机器可读摘要为 `experiments/search_algorithm_evidence_v4.json`（文件 SHA-256
`49d4716b5d68fdfa33c1e245fc633c4d5ba7a95a6432d06a50f76ca23940c295`）。搜索绑定实现提交
`3c6beecae5c555372ad2585eedee0fd0efe9cde7`，source snapshot
`b66e91562f66cc13ff01aaff8be78f1747169bb65eba13843688372363d12010`；训练提交
`938c8fd3abc9cc3a530f76e4dd82c26816156f89`，候选后续评测绑定
`5af48aee5aa3df3f8d79b9f6543c82f8c524f8b2`。所有作业在运行前验证 source manifest
及逐文件 SHA。

### 搜索语义相关性

training/development/final 各 24 条 synthetic 查询，source ID、image SHA 和 query ID
三向零重叠。最终集只在开发门槛通过后消费一次。其分母为 query=24、
ranking=12、no-result=12、hard-filter=16。

| 方法 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | no-result slice | filter correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIP exact | .00598 | .00855 | .625 | .8435 | 0 | .3333 |
| CLIP + Milvus | .00598 | .00855 | .625 | .8435 | 0 | .3333 |
| structured filter + CLIP | .06998 | .07945 | 1.0 | 1.0 | .3333 | 1.0 |
| hard filter + light rerank | .06998 | .07945 | 1.0 | 1.0 | .3333 | 1.0 |
| hard filter + CLIP business guard | .06998 | .07945 | 1.0 | 1.0 | 1.0 | 1.0 |

选择参数只在 development 的 14 个预定组合中比较，得到 business-guard threshold=0、
star weight=0。因此不宣称 star-rating rerank 有独立收益；实际收益来自硬过滤和
business guard。最终 ANN-vs-exact Recall@10=0.9917（development=1.0）只代表 HNSW
返回集与精确余弦 Top10 的一致性，不代表业务相关性。由于正式检索归档未保存
索引原图 SHA，query-vs-index 字节碰撞仍为 `NOT_RUN_MISSING_INDEX_IMAGE_SHA`。

### VLM/SFT 语义与训练

新 development 同锁比较 zero-shot、旧 unified checkpoint-226、当前 checkpoint-87 和 v4
targeted adapter，每角色 n=36（商品 24、对话 12），唯一比较因素为 adapter。v4 训练
仅读 288 条 training（商品 192、对话 96），1 epoch/18 step，训练 67.13 s，
loss=.17585，峰值 allocated/reserved 显存约 10.30/12.66 GiB，得到 adapter SHA
`529026d7e704bb3e2e7761a641df765f88b69f071dd970e339497aa2af108c77`。

| 角色 | category/style/facility/price F1 | exact | unknown abstain | hallucination | first JSON / correction |
| --- | --- | ---: | ---: | ---: | ---: |
| zero-shot | 1/1/1/.25 | .625 | 1.0 | 0 | 1.0 / 0 |
| 旧 unified | .7917/.9091/1/.3333 | .500 | 1.0 | 0 | 1.0 / 0 |
| checkpoint-87 | .625/.6667/.6667/.3333 | .4583 | .3333 | .6667 | .8889 / .1111 |
| v4 targeted | 1/1/1/.6667 | .8333 | 1.0 | 0 | 1.0 / 0 |

| 角色（对话 n=12） | context | state | task key | task value | first route |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero-shot | 0 | .3333 | 0 | .75 | 1.0 |
| 旧 unified | .0833 | 0 | 0 | .25 | 1.0 |
| checkpoint-87 | 0 | 0 | 0 | .25 | .6667 |
| v4 targeted | .4167 | 1.0 | .8333 | 1.0 | 1.0 |

v4 targeted 选择目标从 checkpoint-87 的 .4477 提高到 .9167，但 context=.4167<.6，
预锁门槛整体 `FAIL`。原始错误显示了合并原子事实、漏业态、加字段前缀和误将当轮
新值塞入历史上下文四类问题。本轮不降低事后门槛，v4 VLM final 未打开。

### 真实 HTTP + 外部 Milvus 端到端性能

job `29996972` 在同一 L40S 上顺序比较 checkpoint-87 和 v4 targeted；每角色
1 cold、2 warmup，c=1/2/4 各 8 batch，steady 总分母 56，失败率均为 0。运行使用
FastAPI/Uvicorn loopback HTTP 和独立 Milvus 2.6.18 standalone 进程，插入并加载
1000 条正式向量。Milvus cold startup=1845.50 ms，collection build/load=8359.16 ms。

| 角色 | c | HTTP P50/P95 ms | queue P50 ms | throughput req/s | peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| checkpoint-87 | 1 | 555.72/558.54 | .0007 | 1.800 | 6864.3 |
| checkpoint-87 | 2 | 828.96/1105.41 | 271.18 | 1.811 | 6872.4 |
| checkpoint-87 | 4 | 1384.36/2214.97 | 826.14 | 1.805 | 6888.7 |
| v4 targeted | 1 | 1206.56/1210.95 | .0007 | .829 | 6864.3 |
| v4 targeted | 2 | 1806.97/2426.84 | 597.97 | .829 | 6872.4 |
| v4 targeted | 4 | 3016.81/4827.35 | 1801.64 | .829 | 6888.7 |

v4 targeted 的 c=1 HTTP P95/基线比值为 2.168，超过 1.25，性能 gate `FAIL`。
其 c=1 稳态分阶段 P50 为 CLIP 4.44 ms、Milvus 2.54 ms、rerank .021 ms、VLM
1194.97 ms、HTTP 1206.56 ms。这正是为什么 2.41 ms 向量查询绝不能写成整个
多模态搜索服务延迟。单节点 standalone 不是 multi-node distributed Milvus，本结果
也不支持生产 SLA。

## v5 上下文专项（已完成）

v5 以 v4 targeted adapter 为固定基线，保持基座、Prompt、训练超参和 seed 不变，
只改变上下文专项 synthetic 训练数据组成/支持数。新锁为 training 528（product=144、
dialogue=384）、development/final 各 48（product=24、dialogue=24），source/image/
sample/dialogue-text SHA 三向零重叠。开发 context 必须≥0.6 且相对 v4 提升≥0.1，
其他对话指标不得回退超过 .1；随后的同硬件 c=1 HTTP P95 比值必须≤1.25。
只有两个门槛同时通过才消费 v5 final，不根据 v4 已见开发结果降门槛。机器证据为
`experiments/context_focus_evidence_v5.json`（文件 SHA-256
`24ef6b1e581dc3730f629adc3a77b6f09e5581d6777fee20d030800a1ae8c397`），实现提交为
`15fe56eb871e731df98e11b8207a08583dce84a4`，source snapshot 为
`802cc2e3e9466bcd6f55e709c010a52f98b64edfa18907c1c8d5af4ac325fa5e`。

### v5 训练与新 development

Iris job `29998754` 使用一张 A100 80GB PCIe。v5 从 v4 adapter 继续训练 528 条
synthetic training 样本，1 epoch/33 step，loss=.03446、runtime=187.26 s、
2.82 samples/s；峰值 allocated/reserved 显存约 10.29/12.66 GiB。adapter SHA 为
`b8a1ce69bbce2be99d02a7445a7aaa1fe36a954c42fb1404e26b2244f825cac4`。首次 job
`29998410` 在基座加载阶段失败且未打开 development/final，保留为工程失败，不产生科学结论。
成功 run 的不可变 `run_summary` 存在 schema 名被 identity 字段覆盖的元数据缺陷；指标键和
SHA 完整保留，writer 已修正，历史产物不回写。

v4 与 v5 在同一新 development 锁上各评 48 条（商品 24、对话 24），只改变上下文专项
训练数据组成与支持数：

| 角色 | category/style/facility/price P=R=F1 | exact | unknown | hallucination | first JSON / correction |
| --- | --- | ---: | ---: | ---: | ---: |
| v4 baseline | .9583/1/1/.75 | .875 | 1.0（36/36） | 0/36 | 1.0 / 0 |
| v5 candidate | .9583/1/1/1 | .9583 | 1.0（36/36） | 0/36 | 1.0 / 0 |

| 角色（对话 n=24） | context | state | task key | task value | first route |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 baseline | .2917 | .625 | .2083 | .500 | 1.0 |
| v5 candidate | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

context 从 7/24 提升到 24/24，超过预锁的绝对 0.6 与相对 +0.1 门槛；其他门槛也全部
通过。该开发结果只证明确定性 synthetic 协议拟合改善，不是人工视觉或真实用户相关性。

### v5 真实 HTTP 延迟门与一次性 final

同一 job、同一 A100、同一 training 请求顺序比较 v4/v5；使用真实 loopback
FastAPI/Uvicorn、外部 Milvus 2.6.18 single-node standalone、1000 个可见向量。Milvus
cold startup=1936.34 ms，collection build/load=8216.36 ms；每角色 1 cold、2 warmup，
c=1/2/4 各 8 batch，steady 分母 8/16/32（合计 56），失败均为 0。

| 角色 | c | HTTP P50/P95 ms | queue P50 ms | throughput req/s | peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 baseline | 1 | 1024.10/1046.03 | .0008 | .973 | 6864.3 |
| v4 baseline | 2 | 1528.97/2045.82 | 501.54 | .982 | 6872.4 |
| v4 baseline | 4 | 2534.31/4076.09 | 1520.77 | .985 | 6888.7 |
| v5 candidate | 1 | 1025.40/1030.30 | .0008 | .976 | 6864.3 |
| v5 candidate | 2 | 1527.60/2052.88 | 504.95 | .979 | 6872.4 |
| v5 candidate | 4 | 2528.99/4097.31 | 1517.80 | .985 | 6888.7 |

v5/v4 的 c=1 HTTP P95 比值=.985≤1.25，性能门通过。v5 c=1 分阶段 P50 为 CLIP
7.00 ms、Milvus 3.21 ms、rerank .029 ms、VLM 1009.82 ms、HTTP 1025.40 ms；
模型服务 cold startup=36.04 s，首次请求 HTTP=1.596 s。质量与性能门均通过后，程序写入
exclusive marker 并首次读取 v5 final。final n=48（商品 24、对话 24）上四字段 P/R/F1、
exact、unknown abstention（36/36）、全部五项对话指标和首轮 JSON 均为 1.0，unsupported
hallucination 与 correction trigger 均为 0。final raw canonical SHA 为
`ceb1f4e991050ce96c0c695749ed503965adfc5477c7d235f202a63d1c4b9ca1`。

这组满分只能标作 `synthetic exploration`：没有人工标注，不修改正式 adapter/release，
也没有读取冻结 Fresh Test。服务是单节点 standalone Milvus，不是 multi-node distributed
集群，不支持生产 SLA。

## v7 语义鲁棒性（负实验）

Iris job `30005386` 在一张 L40S、8 CPU、64 GiB 上 `COMPLETED 0:0`，耗时 6 分 56 秒。
实现提交 `84c9b500b303e76c7338d463999a1581df94ffbb`，source snapshot 为
`5f7797423ebd6f911625c06b0e4dae4058c60b2621bdd0221520ee18e5c02e91`。训练从 v5 adapter
继续，保持模型 revision、Prompt 和 generation config 不变，只改变 robustness synthetic
training 数据：512 条（商品/对话各 256）、32 step、112.36 秒；候选 adapter SHA 为
`a06742ebf6344567650e3ae95e67c56116db212d07eeb5eb5c31fd6ae519b5b1`，峰值 allocated/
reserved 显存为 10.27/12.66 GiB。

新的 development 与 v5 development/final 不重叠；每角色 96 条（商品/对话各 48），
unknown opportunity=96、证据不足=16、多主体冲突=8。以下 P/R/F1 分母依次为业态 32、
风格 24、设施 24、价位 16：

| 角色 | category P/R/F1 | style P/R/F1 | facility P/R/F1 | price P/R/F1 |
| --- | --- | --- | --- | --- |
| 固定 v5 baseline | .718/.875/.789 | .577/.750/.652 | .681/.889/.771 | .552/1/.711 |
| v7 candidate | .914/1/.955 | .949/.925/.937 | .943/.917/.930 | .889/1/.941 |

| 角色 | exact | unknown abstain | unsupported hallucination | evidence-low | multi-subject |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 v5 baseline | 33/48 | 49/96 | 47/96 | 7/16 | 0/8 |
| v7 candidate | 43/48 | 91/96 | 5/96 | 13/16 | 5/8 |

两角色 first-attempt JSON 均为 96/96，correction trigger 均为 0/96。候选对话 context、
state value、task key、task value、first route 分别为 48/48、43/48、48/48、48/48、
48/48；固定 v5 baseline 为 42/48、18/48、30/48、46/48、48/48。选择目标由 .7311
提升到 .9548，但预锁 multi-subject 门槛要求至少 6/8，实际只有 5/8，因此 gate=`FAIL`。
程序按约定没有运行候选 HTTP c=1/2/4，也没有 final；正向变化不能越过失败门包装成胜出。
独立 verifier 只确认产物/SHA/指标重算完整，不把质量 gate 改为通过。机器证据为
`experiments/semantic_robustness_evidence_v7.json`（文件 SHA-256
`dfcdd0633bed56fd528309ce2bb8118ca6dd4a3c1c0ce0b51c0af09a5dbba2e1`）。

## v8 no-result 一次性压力验证（通过但中性）

Iris job `30005527` 在一张 L40S、8 CPU、48 GiB 上 `COMPLETED 0:0`。40 条 calibration
只用于选择阈值；写入 exclusive marker 后才读取 40 条 validation（ranking=20、no-result=20、
business-positive=20）。两者及 v4 training 24 条在 source/image/query identity 上零重叠。
全部标签仍为 synthetic，human support=0。

| 方法（validation） | R@5 | R@10 | MRR@10 | nDCG@10 | no-result | positive accept | filter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hard-filter CLIP | .04142 | .04763 | .8 | 1.0 | 8/20 | 20/20 | 40/40 |
| 固定 v4 margin guard | .03938 | .04559 | .7 | .9 | 17/20 | 18/20 | 40/40 |
| 新 dual-centroid guard | .03938 | .04559 | .7 | .9 | 17/20 | 18/20 | 40/40 |

新候选通过 no-result≥.75、positive acceptance≥.9、nDCG≥.75 等预锁门槛，但与固定 v4
baseline 的三个 gate 指标增益均为 0。因此正结论是“既有 v4 guard 在新的一次性 synthetic
压力集上通过”；负/中性结论是“dual-centroid 未带来改进”。该实验未测 ANN-vs-exact，
不得补写 ANN 指标。机器证据为 `experiments/no_result_stress_evidence_v8.json`（文件 SHA-256
`5689fbf4a3c1e7e06171655e18479ceff75967b24edb34a7fcd1875d97b5148a`）。

## v4 查询/正式索引字节隔离复核

v4 搜索报告原先只能证明 synthetic query 三向隔离；由于 formal retrieval archive 不含原图，
query-vs-index 字节碰撞记为 `NOT_RUN_MISSING_INDEX_IMAGE_SHA`。后续只读审计定位到一个与 archive
中 `clip_metadata_1000.jsonl` SHA `7a7989…42d` 完全一致的 Git 外 Yelp overlay，并在
Spartan CPU job `30046716` 上重新哈希索引原图 1000/1000、v4 query 72/72（training/
development/已消费 final 各 24）。结果如下：

| 检查 | 分母 | 结果 |
| --- | ---: | ---: |
| formal index primary byte coverage | 1000 | 1000/1000=1.0 |
| formal index replica byte crosscheck | 1000 | 1000/1000=1.0，mismatch=0 |
| synthetic query byte coverage | 72 | 72/72=1.0 |
| query-vs-index byte collision | 72×1000 身份集合 | 0 |
| query-vs-index source identity collision | 72×1000 身份集合 | 0 |

固定 acceptance 全部通过，独立 verifier=`PASS`。实现提交为
`9452ceb2b0a797a2663e1847c787b2736a18a792`，source snapshot SHA 为
`138f43eed5b1ef9a6f895aa0a29516184d70cf637992a7039ebde4827cd5c14b`；raw registry/report/
verification SHA 分别为 `07b8fc…d014`、`d11f1b…07d7`、`64419b…7854`。审计不读取
annotation、ranking、阈值、模型输出或 Fresh Test，因此只支持“字节与来源身份零碰撞”，不支持
语义相关性结论。机器摘要为 `experiments/retrieval_query_leakage_evidence_v4.json`（文件
SHA-256 `f40076572a3e1e4e174c13af8587574f955ccfb4dd8ab69504bfcd5daa0cb75e`）。

两次失败尝试均保留：`30046002` 因 Iris home 配额满导致默认 Slurm stdout 截断，仅写出
35 bytes traceback；`30046374` 因 Windows/Spartan 的 CRLF/LF raw lock SHA 不同而 fail closed。
两者都没有产生证据。成功修复不放宽内容锁：canonical JSON lock SHA 继续预锁，具体 raw
字节仍由每个 source snapshot 的逐文件 SHA 绑定。

## v6 双节点 Milvus 首次运行（工程负实验）

job `30004826` 固定请求两台 A100 节点、8 CPU、128 GiB 总内存，最终状态
`TIMEOUT 0:0`，耗时 30 分 17 秒。source snapshot
`1f6d6aabadef3bdf209faa3682fd6c6491f3ec37c65f0ffde52a81fbe4282a9f` 已通过逐文件验证；
etcd、MinIO、mixcoord/proxy、querynode/streamingnode、datanode 都启动并注册。

失败原因不是 VLM 延迟：控制节点 Milvus 自动通告了与 Slurm 主机名不同、不可跨节点路由的
接口地址。worker 对 mixcoord gRPC 持续 `i/o timeout`，客户端 `create_collection` 阻塞直到
Slurm 时限取消。该 run 只有 cluster identity 和服务日志，没有 summary/raw/chain，也没有
任何 HTTP 请求；所以 cold、c=1/2/4、吞吐、失败率均为 `NOT_RUN`，不能把 identity 中的
61.7 秒启动观测或旧 2.41 ms vector query 写成端到端性能。

旧 run 保持不可变。修复提交 `8d27e48b8c28470b16b5d51ca3377dd914737a6a` 显式绑定
DNS-resolved inter-node IPv4，并要求 worker→mixcoord/proxy 与 control→querynode/
streamingnode/datanode 五个 probe 全部通过后才允许写 `READY`。修复运行必须使用新的 source
snapshot 和 run 目录；结果未完成前，分布式 HTTP 性能仍为 `NOT_RUN`。

## 自动化 v2 / weak v3 最终补充

以下为历史 v2/v3 结果，不能覆盖前述 v4/v5 当前结论；后文 v1 同样只保留为历史开发证据。
机器可读汇总为 `experiments/search_evidence_enhancement_v2.json`。该轮没有读取或重跑 Fresh Test，没有人工
标注或仲裁，human support=0；所有搜索和 VLM 质量数字均为 synthetic/weak development
evidence，不能宣称人工业务相关性、人工视觉准确率或正式晋级资格。

### Source、数据和作业边界

搜索/VLM/泄漏运行绑定实现提交 `485f706eeddf86455998a409df45a7c49520aac2`，source snapshot
canonical SHA-256 为 `347464ca7f0bdb3bad21b0ac10c8045f56b9e62e15fcb90602f808b13741506e`；
修正搜索分母的离线 gate 作业绑定 `77dd052`，snapshot 为
`e91a8ca74c5c830715661a196efdc48f633c6560e4fb2e9d5575f83f17e66f54`；最终固定长度性能
绑定 `85eb519ef074065f26ca9d3d3c184fc03e363719`，snapshot 为
`1666a6ef31d4f8af428823359cfabab56ceee89f1bf988f0c667e840758e02a7`。三者都在作业内逐文件
验证，不接受事后替换 source。

预运行锁包含 44 个确定性图像资产：搜索 calibration/holdout 各 16 条，VLM 商品 12 条；
另有 6 条不带图片的 synthetic 对话。搜索两 split 的 query ID、source ID、image SHA 均为
0 重叠。配置 SHA-256 为 `fc6982…f4a3`，pool lock SHA-256 为 `1494f1…2fe5`。

### 1000 图字节注册和泄漏审计

source-bound job `29960516` 对 formal overlay 的 1000/1000 source image 原位计算 SHA-256；
metadata、唯一 image ID、唯一 source path 和已哈希字节分母均为 1000。54 张查询图（历史 v1
10、calibration 16、holdout 16、VLM 商品 12）与索引图的 byte collision 和 source-identity
collision 均为 0，主结论为 `PASS_COMPLETE_NO_QUERY_INDEX_COLLISION`。

索引内部只有 976 个唯一 byte SHA，发现 19 个重复字节组、24 个别名；这只是索引内部重复，
不是 query leakage。`project/repo` 比较副本仅覆盖 339/1000，已覆盖部分 mismatch=0，因此该
辅助交叉核验单独标记 `UNKNOWN_INCOMPLETE`，不替代完整 overlay 主分母。registry 未包含图片
字节；file/canonical SHA-256 分别为 `f93929…fc09` / `ebd77f…e50d`。

### 搜索 calibration、一次性 holdout 与修正 gate

job `29960700` 只用 16 条 calibration 选择 `no_result_similarity_threshold=0.12`、
`star_rating_weight=0.0`，随后只消费一次 16 条 holdout。历史 v1 查询没有参与选择。holdout
ranking support=8，ANN-vs-exact Recall@10=1.0 仍只代表近邻保真。

| 方法 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | aggregate no-result | filter correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact CLIP / Milvus | 0.00723 | 0.00977 | 0.50 | 0.9073 | 0.50 | 0.25 |
| structured filter + CLIP | 0.05964 | 0.06741 | 1.00 | 1.0000 | 0.75 | 1.00 |
| hard filter + light rerank | 0.05964 | 0.06741 | 1.00 | 1.0000 | 0.75 | 1.00 |

每个切片的分母和关键指标如下；格式为 `R5/R10/MRR/nDCG/no-result/filter`。ranking support=0
时前四项为 N/A。exact 与 Milvus 完全一致；被选 star weight=0 时 structured 与 hard-filter
light-rerank 也完全一致，所以表内合并显示，但机器证据仍保留四个方法的总指标。

| 切片 | support / ranking | exact = Milvus | structured = hard-filter rerank |
| --- | ---: | --- | --- |
| city_business_facility_price | 4 / 4 | .00723/.00963/.5/.9442/1/0 | .09068/.10354/1/1/1/1 |
| filter_conflict | 4 / 4 | .00723/.00990/.5/.8704/1/0 | .02860/.03127/1/1/1/1 |
| hard_filter_before_rerank | 12 / 8 | .00723/.00977/.5/.9073/.6667/0 | .05964/.06741/1/1/1/1 |
| image_similar | 4 / 4 | .00723/.00963/.5/.9442/1/0 | .09068/.10354/1/1/1/1 |
| no_result | 8 / 0 | N/A/N/A/N/A/N/A/0/.5 | N/A/N/A/N/A/N/A/.5/1 |
| visual_similar_business_irrelevant | 4 / 0 | N/A/N/A/N/A/N/A/0/1 | N/A/N/A/N/A/N/A/0/1 |

被选 star weight 为 0，所以不能声称 light rerank 产生独立收益。原 summary 错把 aggregate
no-result 数值用于固定 gate，记录的 `PASS` 标记为
`INVALID_SUPERSEDED_AGGREGATE_DENOMINATOR`。job `29961100` 仅离线重算冻结结果，没有再次搜索、
运行模型或消费 holdout：真正 no-result slice support=8、accuracy=0.50，低于门槛 0.75；
hard-filter slice support=12（ranking=8），filter correctness/nDCG 均为 1.0。修正后的最终搜索
gate 为 `FAIL`，选择参数保持不变。

### VLM weak v3 三角色比较

job `29960745/29960746/29960747` 比较 current/old/zero，只改变 adapter。每角色 support=18
（商品 12、对话 6），三角色总 support=54；known-price support=4、multi-subject support=4、
insufficient-evidence support=8。结果 canonical SHA-256 为 `fc59ff…3144`。

| 字段 | 每角色 evaluable/unknown | zero P/R/F1；unknown abstain | 旧 unified P/R/F1；unknown abstain | checkpoint-87 P/R/F1；unknown abstain |
| --- | ---: | --- | --- | --- |
| business category | 12 / 0 | .6667/.6667/.6667；N/A | .75/.75/.75；N/A | .6667/.6667/.6667；N/A |
| price range | 4 / 8 | .25/.25/.25；1.00 | 0/0/0；1.00 | 0/0/0；1.00 |
| style tags | 4 / 8 | 0/0/0；1.00 | 0/0/0；1.00 | 0/0/0；0 |
| visible facilities | 4 / 8 | 0/0/0；1.00 | 0/0/0；1.00 | 0/0/0；0.125 |

| 角色 | supported exact | unknown opportunity / abstain | known price exact | multi abstain | insufficient abstain | hallucination | first JSON / correction | dialogue route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero-shot | .6667 | 24 / 1.00 | .25 | 1.00 | 1.00 | 0 | 1.0000 / 0 | 1.0000 |
| 旧 unified | .6667 | 24 / 1.00 | 0 | 1.00 | 1.00 | 0 | 1.0000 / 0 | 1.0000 |
| checkpoint-87 | .6667 | 24 / .375 | 0 | 0 | 0 | .625 | .7778 / .2222 | .3333 |

对话分母每角色 n=6：

| 角色 | context recall | state/value | task key | task value | first-turn route |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero-shot | 0 | .3333 | 0 | .6667 | 1.0000 |
| 旧 unified | 0 | .1667 | 0 | .3333 | 1.0000 |
| checkpoint-87 | 0 | 0 | 0 | 0 | .3333 |

checkpoint-87 只有商品/对话最小 support 检查通过；价位、多主体、证据不足、幻觉、首次 JSON
与全部对话指标均未通过固定 gate。VLM 联合质量结论为 `FAIL`，不允许晋级。

### 真实固定输入/输出长度组件性能

job `29961579–29961584` 在同一 `NVIDIA A100 80GB PCIe MIG 1g.20gb` 规格上完成旧/当前
adapter × 3 profile。每格为一个独立进程、1 条真实 process cold、1 次 warmup、5 条 steady；
失败率均为 0。实际输入 231/327/615 token 互不相同，实际输出由
`min_new_tokens=max_new_tokens` 强制为 32/64/128，不再用最大上限冒充实际长度。

| profile | 实际 input/output | 旧 P95 | current P95 | current/old | current peak VRAM | gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| short_32 | 231 / 32 | 2,600.51 ms | 2,590.26 ms | 0.9961 | 6,884.70 MiB | PASS |
| medium_64 | 327 / 64 | 5,003.46 ms | 4,925.37 ms | 0.9844 | 6,904.36 MiB | PASS |
| long_128 | 615 / 128 | 9,349.75 ms | 9,646.08 ms | 1.0317 | 6,968.59 MiB | PASS |

| profile | old cold startup / e2e；steady QPS | current cold startup / e2e；steady QPS |
| --- | --- | --- |
| short_32 | 53,683.27 / 57,038.11 ms；.3861 | 29,959.98 / 32,888.75 ms；.3878 |
| medium_64 | 53,699.31 / 59,426.10 ms；.2003 | 29,902.69 / 35,216.84 ms；.2039 |
| long_128 | 30,034.63 / 39,813.40 ms；.1074 | 29,888.02 / 39,804.71 ms；.1045 |

完整矩阵位于 `experiments/search_evidence_enhancement_v2.json` 的
`performance_v2.measured_cells`：每格保留 cold startup/e2e、steady CLIP/Milvus/rerank/VLM/
e2e 的 P50/P95、throughput、failure rate、VRAM、raw SHA 和 job ID。上表只做紧凑展示；cold
每格仅 n=1，且并发提交造成的共享节点启动差异不能解释为 adapter 性能差异。

三个 current 绝对 P95 都低于预锁定 15/22/35 秒门槛，三个相对比也低于 1.25，组件延迟 gate
为 `PASS`。但 quality gate 为 `FAIL`，所以 joint quality+latency gate 为 `FAIL`。matrix 文件
SHA-256 为 `e0dcf9…c2bc`，36 条 measured row 的 canonical SHA-256 为 `2dfd45…caf3`。

该 v2 轮 concurrency=2/4 的单模型 Milvus Lite 单元因未声明线程安全而 `NOT_RUN`；
concurrency=1/2/4 的 distributed Milvus + HTTP 服务因没有隔离、安全的服务端点而 `NOT_RUN`。
后续 v4/v5 已补真实 HTTP + 外部单节点 Milvus 的 c=1/2/4，但 multi-node distributed 仍未运行。因此这里的 v2 结果只能说明
单进程组件性能，不是生产服务吞吐或 SLA。

早期 job `29960754–29960759` 虽完成，但三个 profile 实际都为相同输入/输出长度，统一标记
`SUPERSEDED_IDENTICAL_REALIZED_LENGTHS`；10GB probe `29960861` 被取消，均不进入最终矩阵。

### v2 决策

硬过滤在该 synthetic holdout 上提高 filter correctness 和 nDCG，但 no-result 真正切片未过门；
checkpoint-87 未过 weak v3 质量门；固定长度组件延迟过门但没有生产 SLA 资格。最终不修改正式
release、adapter、Prompt、阈值或 Fresh Test 状态。

## v1 结论先行（历史）

本任务建立了可运行、失败关闭的四轨证据：ANN-vs-exact、独立查询业务语义、VLM/SFT
one-factor 语义和端到端性能。最重要的结论不是“所有指标都变好”，而是找到了可以成立和
不能成立的事实边界：

- 新 10 条 Commons 弱标注查询上，Milvus 对 exact 的 Recall@10 仍为 1.0；这只证明 ANN
  保真，不证明业务相关。
- 结构化过滤相对纯 CLIP 将 MRR@10 从 0.40 提至 0.65、nDCG@10 从 0.8020 提至
  0.8253、过滤正确率从 0.50 提至 1.00、无结果准确率从 0.80 提至 0.90。轻量重排提升
  MRR/Recall，但未修复过滤正确率，且 nDCG 略降。
- 历史 168 条同锁 development 重算确认 checkpoint-87 明显优于旧 unified/zero-shot；但该
  数据已参与历史选择，只能称 audit，不是本任务新提升。
- Iris 保存的历史 Fresh Test 120 raw 与 metrics 可离线对账：商品 composite
  `0.7806388889` 与发布值 `0.780639` 一致；字段审计暴露 price F1=0、设施 F1=0.835 和
  unknown 幻觉率 0.1935。该测试已消费，只做错误审计，不产生新晋级资格。
- 新的字节绑定 v2 弱池只有 5 条商品和 3 条 synthetic 对话。checkpoint-87 没有形成联合
  质量优势：首次 JSON 0.75、设施 F1 0、对话首轮路由 0.333，均低于旧 unified 的
  1.00、0.20、1.00。该负结果不允许晋级。
- 同 A100 20GB MIG、1 cold + 3 warmup + 30 steady 的短 hotel probe 上，checkpoint-87
  稳态端到端 P95 为 4121.38 ms，旧 unified 为 4767.04 ms，比例 0.8646；性能门禁通过，
  但不能覆盖上述质量失败。

## 运行来源与提交边界

最终采用的远端结果不是把事后 commit SHA 补写到旧作业上。实现先在基线
`5456f477c80e54a0764051206898fccd48db6237` 上提交为
`4cb78787974214d20f8c6d0bbb4dffcd84376d36`，再通过该提交的 `git archive` 将 23 个运行时
文件物化到 Iris 独立项目。逐文件 SHA 清单位于
`experiments/relevance_evidence_source_snapshot_v1.json`，其 canonical
`run_source_snapshot_sha256` 为
`ec422d421b05749a1617004c5aa4d7ff1f0341e283902880c78df8a83e485dea`。
其中固定 evaluation config 的 byte SHA-256 为
`bd83f69b6714d624316ac82a49fe4463a282b66ad37c0492d4479f2375fa6130`。

最终 search、历史 audit、三角色 VLM 与两角色 performance 作业都在运行主程序前重新计算
清单和每个文件的 SHA；不一致即退出。最终 source-bound job 为
`29926868/29926869/29926870/29926871/29926872/29926873/29926874`。此前未绑定最终 source
snapshot 的 job 只保留为被取代的诊断运行，不再作为本报告的最终 raw provenance。

## 数据、来源与标签等级

### 独立搜索池

搜索 manifest 为 10 条查询、5 张 Wikimedia Commons 图片，覆盖：同类视觉、城市/业态/
设施/价位组合、视觉相似但业务无关、无结果、图片/过滤冲突。查询 manifest SHA-256 为
`a259f3dde00efc4225afcaf9f43e00d34345a977605347a655cfcb249ee33125`；精确 960px
来源 registry SHA-256 为
`ad7130d83f1636053048a1298a1a373456679d15a937378a76f5318fec02682f`。

5 张图片的 byte SHA 均已验证，来源明确不是 Yelp。正式检索包未保存索引原图 SHA，所以
只能证明来源隔离，字节级索引/查询碰撞检查为 `NOT_RUN_MISSING_INDEX_IMAGE_SHA`。

10 条相关性等级来自 metadata 规则，annotation SHA-256 为
`e9bc8fa6be25cff65d7ff4fcdc2f352bf4e6d0c410366b3d9a75bc9792eb8349`；human support
为 0。所有业务指标都是 weak evidence，不能写成人工业务相关性。

### VLM 新弱池

v2 锁包含 5 条 Commons 商品弱标签和 3 条 synthetic 对话，data lock SHA-256 为
`bc67a10ef27892cba19ba66cba95c7c9d8eb31ff2632abc741146cdf98ed2f2e`。基座、revision、
Prompt 和 generation config 在三角色间完全一致，只允许 adapter 变化。

早期 v1 缺图片 SHA/source_id 的锁内绑定，运行虽完成，但判为
`REJECTED_MISSING_ASSET_BYTE_BINDING`，没有与 v2 混分。v2 没有可信多主体图片，相关切片
support=0、状态 `PENDING_NO_CREDIBLE_SAMPLE`。价位 5/5 都是 unknown，known-price
support=0，价位 P/R/F1 为 `N/A`；只报告 unknown abstention accuracy。

## 正式检索包与历史口径审计

正式 retrieval archive SHA-256 为
`3cdb98f4d50bc72ae53c4e7e96d823ea5b08af93f41df5d14ff1118d12d1a15b`，其向量、metadata、
benchmark 成员 SHA 分别为 `021f09…59ee`、`7a7989…42d`、`21b296…d90`。

历史 benchmark 的正确描述是：100 个 self/prototype query、Top10、HNSW/COSINE、
M=16、efConstruction=128、ef=64；ANN-vs-exact Recall@10=1.0，Milvus vector query
mean/P95=2.23545/2.40970 ms。它没有独立查询，没有业务分级，也不包含 CLIP 编码、重排或
VLM，因此不能外推为搜索质量或端到端性能。

## 新搜索实验

运行环境为 A100 20GB MIG、Python 3.11.3、Torch 2.8.0+cu128、PyMilvus 2.6.16；source-bound
job `29926869` 为 `COMPLETED 0:0`。结果 canonical SHA-256 为
`09ab01d2e74424b9aa126245eeedd2187b7479ca1c2a23077c97387ad4373cb6`，raw 文件 SHA-256
为 `ef2ed32c41ebb0c7938a8cebd13cc40ded9e621fc9d88fe769fd668d97cb87e1`，summary 文件
SHA-256 为 `94d6626af72762d01660fb9561730979efc2f734ab9a9eb889981c21b47cfa94`。

| 方法 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | no-result acc | filter correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIP exact | 0.00455 | 0.00883 | 0.400 | 0.8020 | 0.80 | 0.50 |
| CLIP Milvus | 0.00455 | 0.00883 | 0.400 | 0.8020 | 0.80 | 0.50 |
| structured filter + CLIP | 0.04622 | 0.05063 | 0.650 | 0.8253 | 0.90 | 1.00 |
| lightweight rerank | 0.02538 | 0.02980 | 0.525 | 0.7905 | 0.80 | 0.50 |

ranking support 为 8；2 条 no-result 不进入 ranking 分母。这里的 Recall 分母是 1000 条
metadata 中满足弱规则的全部记录，因此数值会远低于只看返回 Top10 的 nDCG，不应混用。
所有方法失败率为 0，未支持的 facility 约束披露率为 1.0。

正实验：结构化过滤对组合约束和 filter-conflict 两个切片的过滤正确率与 nDCG 都达到 1.0；
它是当前最明确的算法收益。

负实验：纯 exact/Milvus 的 no-result 切片准确率为 0；结构化过滤也只有 0.5，因为无显式
过滤的 private living-room 仍被低阈值接受。轻量重排没有保证硬过滤，尽管 MRR/Recall 上升，
整体 nDCG 从 0.8020 降到 0.7905。下一版应把硬过滤置于重排之前，并在独立 development
上重新锁定 no-result calibration；本次不根据该弱池修改正式阈值。

搜索阶段 P50/P95 为：CLIP 16.67/381.14 ms、Milvus 3.13/3.97 ms、重排
0.13/0.24 ms、search path 28.42/412.73 ms。首条冷图片编码抬高 P95，不能与历史
2.41 ms vector-only 数值直接比较。

## 历史 168 条 development 重算

该段是 `historical development recomputation/audit`。数据已经参与 checkpoint-87 的历史选择，
不能称为本任务新的 development 或新的独立提升。168 条均为 `programmatic_silver`，场景支持
为商品/售后/行程/对话=48/48/48/24，三角色 sample/config/data lock 完全一致。最终
source-bound audit job `29926868` 为 `COMPLETED 0:0`，audit 文件 SHA-256 为
`19497e293081b3c8494965abc17907f427dad195770c16f176c5a3b20088585b`。

| 角色 | preserved raw SHA-256 | recorded mean latency |
| --- | --- | ---: |
| zero-shot | `86895ababdca937bccd3ccbe40b8c63b8fa4a3cf54aee04c627a67b80dff98c2` | 3,242.23 ms |
| 旧 unified checkpoint-226 | `7ba40d30824ce1f1608928b6757d7e8dfd4823cc6c5a01ea750e1c1e25ceb421` | 9,628.89 ms |
| checkpoint-87 | `6b123981fe6b86d99a75ac731cc865e28887b5755176a0269a57c1184d05dee3` | 10,707.54 ms |

| 角色 | adapter SHA | 商品 category F1 | style F1 | facility F1 | price F1 / support | exact | unknown hallucination | 对话综合 | 失败率 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| zero-shot | 无 | 0 | 0 | 0 | 0 / 5 | 0 | 0 | 0.1534 | 0.0179 |
| 旧 unified checkpoint-226 | `ccc606…24ee` | 0 | 0 | 0 | 0 / 5 | 0 | 0 | 0.9529 | 0 |
| checkpoint-87 | `c2fbb5…eaa2` | 1.000 | 0.6139 | 0.8701 | 0 / 5 | 0.3333 | 0.1042 | 0.9821 | 0 |

旧 raw 只保存最终输出，没有 attempt 序列。最终输出 JSON compliance 可以重算，但首次尝试
合规和 correction trigger 为 `NOT_RECORDED`，不能由最终输出反推。

## 新 VLM/SFT v2 实验

三角色 source-bound job `29926870/29926871/29926872` 均为 `COMPLETED 0:0`。三者
product support=5、dialogue support=3；这是跨来源小弱池诊断，不用于晋级。

| 角色 | result SHA-256 | raw 文件 SHA-256 | mean latency |
| --- | --- | --- | ---: |
| zero-shot | `213523e26336d71cb492a03e310f2506b1f6d7bc0d59637b77a04538ab074fbc` | `478e454754ecd82dccbe7fb9f6717637e79f975100eb64e0387fa578c30dcf57` | 5,059.35 ms |
| 旧 unified | `c7b257f8b94bcaa372fb2e04b080fb2a145a78f706669620918260b97506f8ed` | `ef0c782177380f5f62843f33514d6c94848f242d31fe2351b7b03569b057f7eb` | 3,186.12 ms |
| checkpoint-87 | `b7ac06390a57126e91bd6d5d44fb1ea7b88b2242125d58c3dd8af10f634a750d` | `500bbf2dffd63c1036290bef94a6263f83ae9f2d26e9bf4ced632c1a02e5e814` | 5,327.13 ms |

三角色合并语义 score 文件 SHA-256 为
`d33b3ab867742f3ba88d8ad98e2ab475eb8fbba025b117d19dc4ba626a583b40`。

| 角色 | category F1 | style F1 | facility F1 | price | price unknown abstain | supported exact | first JSON | correction | hallucination |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| zero-shot | 0.80 | 0.1176 | 0.1667 | N/A (n=0) | 0.80 | 0 | 1.00 | 0 | 0.20 |
| 旧 unified | 0.80 | 0 | 0.20 | N/A (n=0) | 1.00 | 0 | 1.00 | 0 | 0 |
| checkpoint-87 | 0.80 | 0.1429 | 0 | N/A (n=0) | 1.00 | 0 | 0.75 | 0.25 | 0 |

| 角色 | context recall | state/value | task key | task value | first route |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero-shot | 0 | 0 | 0 | 0.667 | 1.000 |
| 旧 unified | 0 | 0 | 0 | 0 | 1.000 |
| checkpoint-87 | 0 | 0 | 0 | 0 | 0.333 |

checkpoint-87 相对旧 unified 仅在这个小池的 style F1 上提高 0.1429；facility F1 下降 0.20、
first JSON 下降 0.25、首次路由下降 0.667，且两条对话纠错后仍没有可评分对象。联合质量结论
因此为 FAIL。它和历史 168 条的正结果不矛盾：一个是已参与选择的同分布 programmatic
silver audit，一个是 8 条跨来源弱/合成探针，二者不混分。

## 端到端性能

基线与候选分别由 source-bound job `29926873/29926874` 在同一 A100 20GB MIG 规格完成；
每个角色固定 1 cold、3 warmup（不计分）、30 steady，失败率均为 0。基线/候选 raw 文件
SHA-256 分别为
`e2122d63698ace1a529d30e55aae43163c8292af6d571e1ae857a5d66bcac7e9`/
`66b411ee24e5700246ba353bde1ae7029c28098117ce09badcbd151a2d7f891c`，比较文件 SHA-256 为
`c0b005a3b123ea8246f01d29e5d167d256fab765eca5cff4b8c2d72e052c416f`。

| 指标 | 旧 unified | checkpoint-87 |
| --- | ---: | ---: |
| cold startup（n=1） | 27,616.19 ms | 25,970.98 ms |
| cold CLIP / Milvus / rerank / VLM（n=1） | 216.72 / 5.89 / 0.033 / 4,965.84 ms | 162.57 / 6.10 / 0.030 / 4,291.32 ms |
| cold end-to-end（含 startup） | 32,815.28 ms | 30,440.22 ms |
| steady end-to-end P50 | 4,722.29 ms | 4,082.50 ms |
| steady end-to-end P95 | 4,767.04 ms | 4,121.38 ms |
| steady CLIP P50 | 16.14 ms | 15.86 ms |
| steady CLIP P95 | 17.58 ms | 16.80 ms |
| steady Milvus P50 | 3.74 ms | 3.40 ms |
| steady Milvus P95 | 4.76 ms | 4.29 ms |
| steady rerank P50 | 0.023 ms | 0.023 ms |
| steady rerank P95 | 0.031 ms | 0.026 ms |
| steady VLM P50 | 4,699.31 ms | 4,059.56 ms |
| steady VLM P95 | 4,741.17 ms | 4,098.89 ms |
| peak VRAM | 7,036.59 MiB | 7,036.59 MiB |
| throughput | 0.2116 qps | 0.2448 qps |

checkpoint-87/旧 unified 的稳态 P95 比为 0.8646，低于固定 1.25 门禁；候选 P95 也低于
12 秒、峰值 VRAM 低于 8192 MiB。性能 gate 为 PASS。VLM 占稳态总耗时约 99%，Milvus
约 3–4 ms，进一步说明 2.41 ms 不是系统延迟。

cold 每个角色只有 1 条，不计算有意义的 P50/P95；上表直接报告该次观测并明确包含 startup。
steady 的每个阶段均以 30 条分别计算 P50/P95。

该结果只适用于一个固定、短输出的 hotel-search probe，不是通用生产 SLA。历史 168 条上
checkpoint-87 平均延迟反而是旧 unified 的 1.112 倍，说明延迟强烈依赖输出长度和任务分布。

## 0.780639 审计与 v1 历史缺口的当前状态

正式本地 evidence 包没有对应 120 行 raw/metrics，所以本地 handoff 的初始状态确实是
`EVIDENCE_GAP_RAW_SAMPLE_OUTPUTS_NOT_IN_LOCAL_HANDOFF`。随后只读、限域检查 Iris 的
`system-repair-20260824/outputs/system_repair`，发现已消费一次的历史目录
`final_test/system_repair_fresh_test_once_20260825_v4` 保存了 120 行 raw 与 metrics。

source-bound 离线 audit job `29927144` 为 `COMPLETED 0:0`；它只解析既有文件，不加载模型、
不生成新预测、不重跑 test，也没有反馈到 Prompt、阈值或候选选择。audit 实现提交为
`496f067d36b3d2b79041dcf268d2e041045c280b`，source snapshot SHA-256 为
`6bc1efd779dadbe60c016fa4e1229cb33497f6019fe6e4f6f90de19569000bed`，audit 文件 SHA-256 为
`6deb77b113b1ece1b0cc69df46ca8faca7571aee5405ed9b8d07bb895f4eaf98`。

raw、原 metrics、test dataset、gate、consumption SHA-256 分别为
`344464…eb19`、`853bd6…1018`、`f31519…f456`、`9574b0…77d`、`2a86c9…9082`；120 个
sample_id 与 dataset 完全一致。原 metrics 的商品 composite 为 `0.7806388888888889`，六位
小数正好是 `0.780639`。

| 商品字段 | support | precision | recall | F1 | 不完全匹配样本数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| business category | 29 | 1.000 | 1.000 | 1.000 | 0 |
| style | 25 | 0.960 | 0.889 | 0.923 | 3 |
| visible facility | 30 | 0.860 | 0.811 | 0.835 | 14 |
| price range | 5 | 0 | 0 | 0 | 5 |

商品逐样本 exact match 为 0.40；unknown 机会 31 个，其中 6 个出现非空预测，幻觉率
0.1935。最终记录 JSON compliance=1.0、operational failure=0。保留 raw 没有 attempt 序列，
首次 JSON 与 correction trigger 仍是 `NOT_RECORDED`。这是对历史已用测试的误差审计，不是
新的独立提升，也不改变正式 gate 或 release。

锁定 dataset 与 raw 中没有显式 multi-subject/multiple-subject/多主体标签；多主体冲突
support=0，状态为 `NOT_SCORABLE_NO_PRESERVED_MULTI_SUBJECT_LABEL`。本审计不从普通图片或
模型输出反推多主体真值。

以下为截至 v9 预注册和 v4 byte-audit repair-2 的补齐情况与仍存边界；v2 的旧失败状态仅是历史负实验：

- **仍未完成**：人工相关性双人标注与仲裁，human support 仍为 0；
- **已自动化补齐**：job `30046716` 通过 Git 外 formal overlay 对 1000/1000 索引原图与
  v4 query 72/72 原位哈希，byte/source collision 都为 0；正式 retrieval 压缩包本身仍不含原图；
- **已补齐但仍是 weak/synthetic**：v4/v5/v7 对价位、unknown、多主体、证据不足与对话状态
  提供新分母；v7 将多主体冲突从 0/8 提至 5/8，但未过预锁 6/8 门槛，仍是负实验；
- **已预注册未出结果**：v9 以 v7 为固定基线，在新 development 只改变多主体反例训练构成，
  不下调门槛且不定义 final；job `30044630` 完成前不得声称提升；
- **synthetic 已补齐**：v4 在三向隔离的 24 条一次性 final 上通过 no-result 与 hard-filter；
  v8 又在 40 条一次性 validation 上得到 17/20 no-result，但新 dual-centroid 与固定 v4 baseline
  持平；v2 的 no-result 4/8 失败继续作为历史负实验；
- **部分补齐**：已完成真实 loopback HTTP、外部单节点 Milvus 及 concurrency=1/2/4 基准；
  首次 multi-node distributed Milvus 因接口通告错误超时且没有 HTTP 分母，修复运行未完成前
  仍为 `NOT_RUN`，不支持生产 SLA；
- **保持冻结**：没有新的未消费人工/真实用户最终集；Fresh Test 120 未读取、未调参、未重跑。

## 决策与可复现入口

当前机器证据以 `experiments/search_algorithm_evidence_v4.json`、
`experiments/context_focus_evidence_v5.json`、`experiments/semantic_robustness_evidence_v7.json`、
`experiments/no_result_stress_evidence_v8.json` 和
`experiments/retrieval_query_leakage_evidence_v4.json` 为最新入口；v1/v2 文件只保留为历史开发证据。
固定配置和预运行数据锁分别位于 `configs/evaluation/automated_evidence_v4.json`、
`configs/evaluation/automated_evidence_v5.json`、`configs/evaluation/automated_evidence_v9.json` 与
`configs/evaluation/evidence_enhancement/`；
协议与运行命令见 `docs/evidence_enhancement.md`。

最终决策：v4 hard-filter + business guard 在新 synthetic final 上形成正向排序、过滤与
no-result 证据；v5 上下文专项依次通过新 development 质量门和同机 HTTP 延迟门，再一次性
通过 synthetic final。v7 虽有显著开发改善但未过多主体门槛；v8 通过新压力门却与固定基线
持平；两者分别记为负实验和中性实验。v4 context/延迟失败、v2 no-result 失败和 v6 首次
分布式部署超时及 byte-audit 的两次基础设施/跨平台失败同样保留。v4 query/index 字节隔离现已
用 1000×72 完整分母补齐，但这不改变语义指标。由于没有人工标签或真实用户最终集，且
分布式修复运行尚未形成有效 HTTP 分母，仍不修改正式 release、Prompt、adapter、阈值或
Fresh Test 状态。

## 验证

- `python -m unittest discover -s tests -v`：978 项通过，2 项既有跳过；
- `python scripts/tripctl.py validate`：`status=ok`；
- 正式 Git 外 release 的历史 `scripts/verify_final_delivery.py` 记录为 `PASS`（包内 948 项测试）；
  本 worktree 当前未挂载该外部包，故本轮复核为 `NOT_RUN_MISSING_EXTERNAL_RELEASE_PACKAGE`；
- `docker compose ... config --quiet`：通过；
- 本地独立查询池、来源 registry、正式 retrieval/release 哈希核验：`PASS`；
- VLM weak v3 对冻结 raw 重评分：score SHA-256 保持 `b039c5…7458`；
- v5 job `29998754`：source 验证、训练、development、真实 HTTP/Milvus c=1/2/4、一次性 final 完成；
- v7 job `30005386`：source、训练、development 与独立重算完整，固定质量 gate `FAIL`；
- v8 job `30005527`：calibration 后一次性 validation 与独立重算完整，固定压力 gate `PASS`；
- v4 byte audit job `30046716`：索引 1000/1000、query 72/72 字节覆盖完整，零碰撞，独立 verifier `PASS`；
- 固定长度 performance scorer：历史实际 input/output 矩阵验证通过，v2 joint gate `FAIL`；
- `git diff --check`：通过。

## 简历候选表述

只有在同时保留 development/weak/synthetic 标签限定时，建议使用以下两条：

1. 构建 training/development/final 三向隔离的 synthetic 多模态搜索评测，在一次性 final
   24 条（ranking=12、no-result=12）上将 CLIP 的 MRR@10/nDCG@10/filter correctness
   从 .625/.8435/.333 提至 1.0/1.0/1.0，并将 ANN-vs-exact Recall@10=.9917 单列为检索保真。
2. 在 Spartan A100 80GB 上完成上下文专项 LoRA 的单因素训练与质量/延迟串行门控：新
   synthetic development 的 context recall 从 7/24 提至 24/24；真实 HTTP + 外部单节点
   Milvus 的 c=1 P95 为 1.030 s（相对 v4=.985），c=1/2/4 共 56 稳态请求/角色零失败。
