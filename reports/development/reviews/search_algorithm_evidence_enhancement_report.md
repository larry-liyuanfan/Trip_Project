# 搜索算法、VLM/SFT 与端到端证据增强报告

日期：2026-09-03 结论：`COMPLETED_DEVELOPMENT_DIAGNOSTIC`；正式 release 不变；性能门禁
`PASS`，质量与延迟联合门禁 `FAIL_NOT_ELIGIBLE_WEAK_LABEL`。

## 结论先行

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

## 0.780639 审计与未完成项

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

未完成项：

- 人工相关性双人标注与仲裁尚未执行，human support=0；
- 索引原图 SHA 不在正式 retrieval 包，字节级 collision audit 未运行；
- new VLM v2 的 known-price support=0、多主体 support=0，历史 Fresh Test 也没有可评分的
  多主体冲突标签；
- no-result 阈值没有可用于晋级的人工 development calibration；
- 当前性能只有单一短 probe，不覆盖生产请求分布或并发。

## 决策与可复现入口

机器证据位于 `experiments/search_evidence_enhancement_v1.json`。固定配置为
`configs/evaluation/evidence_enhancement_v1.json`；协议与运行命令见
`docs/evidence_enhancement.md`。

最终决策：结构化过滤形成正向弱证据；checkpoint-87 性能门禁通过；新 VLM 联合质量门禁
失败且标签等级不足以晋级。因此不修改正式 release、Prompt、adapter、阈值或 Fresh Test
状态。

## 验证

- `python -m unittest discover -s tests -v`：926 项通过，2 项既有跳过；
- `python scripts/tripctl.py validate`：`status=ok`；
- 正式 Git 外 release 的 `scripts/verify_final_delivery.py`：`PASS`，包内记录 948 项测试；
- `docker compose ... config --quiet`：通过；
- 本地独立查询池、来源 registry、正式 retrieval/release 哈希核验：`PASS`；
- `git diff --check`：通过。

## 简历候选表述

只有在同时保留 development/weak 标签限定时，建议使用以下两条：

1. 构建独立视觉搜索评测与失败关闭评分框架，在 10 条 Commons 弱标注查询上比较 exact
   CLIP、Milvus、结构化过滤和轻量重排；结构化过滤将 MRR@10 从 0.40 提至 0.65，并把过滤
   正确率从 50% 提至 100%，同时将 ANN Recall 与业务相关性严格分轨。
2. 在 Spartan A100 20GB MIG 上实现 CLIP→Milvus→重排→Qwen3-VL 的分阶段基准；固定
   30 次稳态下 checkpoint-87 端到端 P95 为 4.12 秒、相对旧 unified 为 0.865 倍，并通过
   数据锁、adapter SHA 和联合质量门禁阻止“只快不准”的候选晋级。
