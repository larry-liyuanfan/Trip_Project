# 系统收敛修复与统一封装报告

日期：2026-08-25
当前状态：`PARTIAL`  
发布结论：代码、检索、Prompt pilot 与 Week 5 修复已通过；模型修复门禁尚未完成，不进入 `stg`

## 已完成修复

| 范围 | 实际结果 |
| --- | --- |
| 跨平台配置哈希 | 配置文本统一按 LF canonical bytes；Windows/Linux 回归通过 |
| 生产 API | Qwen3-VL + PEFT/NF4 运行时、三场景接口、对话接口、视觉检索接口已实现 |
| 失败处理 | 生产模式关闭静默 fallback；Schema 最多一次模型级纠错并保留两次原始输出 |
| 就绪检查 | `/health` 仅检查进程；`/ready` 核验 adapter、模型、Prompt、Schema、CLIP、Milvus |
| Prompt pilot | 三场景各 48 条 development 样本完成三候选比较并锁定场景胜出 Prompt |
| Week 5 v2 | 80,000 条均有 Schema-valid silver 结果；44 条同层替换，五维评测冲突 0 |
| 新数据锁 | train/development/test=1,980/168/120，五维跨 split 冲突 0，test 未消费 |
| continuation SFT | Spartan job `29562078` 完成并早停于 step 112；回载最佳 checkpoint-87 |
| 检索 | 1,000 张真实 OTA 图片完成 CLIP 512 维编码和 Milvus 实测 |
| 封装 | 统一 Compose、release manifest、四层私有 OSS 打包器和 `tripctl` 已实现 |
| 测试 | 当前完整 `unittest` 509/509；`git diff --check` 通过 |
| 仓库整理 | 仅保留 `dev/stg/main`；旧 closeout 证据迁入并校验 11,037 个 SHA-256 后移除 |

## Week 5 修复池

| 指标 | 结果 |
| --- | ---: |
| 候选总数 | 80,000 |
| 商品/售后/行程 | 50,000 / 20,000 / 10,000 |
| 历史 Schema-valid | 79,936 |
| Qwen3-VL-8B 修复成功 | 64 / 64 |
| 不可读输入替换 | 44 |
| Schema/JSON 修复 | 19 / 1 |
| sample/source/image SHA 唯一数 | 80,000 / 80,000 / 80,000 |
| 五维评测集冲突 | 0 |

最终不可覆盖合并产物为 `schema_valid_silver_80000.jsonl`，80,000/80,000 均通过
JSON Schema；SHA-256 为
`86b0a158567da3e3b683fd73476d51f1608ad6f59ae5219e7f52354180ff5926`。
新增结果仍为 `silver`，历史人工 accepted 保持商品/售后/行程各 100、对话 100，没有增加。

## Prompt pilot

固定 development 集对 `current`、`compact_schema_v1` 和 `evidence_state_v1` 完成比较，
不可覆盖选择记录位于 run `system_repair_prompt_pilot_20260824_v8`。场景胜出结果为：

| 场景 | 胜出候选 | 发布 Prompt |
| --- | --- | --- |
| 商品理解 | compact | `system_repair_product_compact_v3` |
| 智能售后 | evidence | `system_repair_after_sales_evidence_v3` |
| 行程规划 | current | `system_repair_itinerary_structured_v4` |

## Milvus 真实验证

运行环境：本地 RTX 4070 Laptop GPU、Milvus `2.6.20`、PyMilvus `2.6.16`。

| 指标 | 实测 |
| --- | ---: |
| CLIP 向量 | 1,000 x 512 |
| 向量范数范围 | 0.99999988 - 1.00000012 |
| HNSW/COSINE | M=16，efConstruction=128，ef=64 |
| 索引构建 | 4.6205 秒 |
| 查询数 / TopK | 100 / 10 |
| 平均 / P95 延迟 | 2.2355 / 2.4097 ms |
| Recall@10 | 1.0 |
| CRUD | 999 批量 + 1 单条；删除 1；删除后命中 0 |

运行产物 SHA-256：

- vectors：`021f09d764038a3ce53d28d348b4c1b6f5b50ba82f51d69ccd2b1acfeee059ee`
- metadata：`7a79894fb027e2f0e6e6aa943a5af21c10c72b084f0dd866c931405debfcd42d`
- benchmark：`21b296fea2ffbc3422c0863eeb87e5f6151de7cddecba18997ebdf9218ac1d90`

## continuation SFT

Spartan job `29562078` 在单个 L40S 上 `COMPLETED 0:0`，耗时 `04:48:36`。
训练从 Week 7 checkpoint-226 adapter 继续，学习率 `5e-5`、最多 1 epoch、
patience=2；step 100 和 step 112 连续未刷新最佳后自动早停并回载 checkpoint-87。

| 指标 | checkpoint-87 development 结果 |
| --- | ---: |
| 总体加权综合 | 0.920725 |
| 核心三场景加权综合 | 0.905382 |
| 商品理解综合 | 0.716146 |
| 智能售后综合 | 1.000000 |
| 行程规划综合 | 1.000000 |
| 对话自动综合 | 0.982097 |
| 请求失败率 | 0.000000 |

最终 adapter SHA-256 为
`c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a`；
adapter-only 保存和磁盘回载验证通过。以上只是候选自身的 development 结果，仍需与
旧 unified、zero-shot 和 Week 6 routed 在同一 development 集比较后才能进入 test。

## Development 对比与门禁

同一 168 条 development 集的真实对比如下；商品、售后、行程各 48 条，对话 24 条，
没有删除困难样本：

| 模型路径 | 总体加权 | 核心加权 | 对话自动综合 | 失败率 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| repair checkpoint-87 | 0.920725 | 0.905382 | 0.982097 | 0.000000 | 10,707.54 ms |
| 旧 unified adapter | 0.750034 | 0.699306 | 0.952946 | 0.000000 | 9,628.89 ms |
| zero-shot | 0.084010 | 0.066667 | 0.153384 | 0.017857 | 3,242.23 ms |
| Week 6 routed | 0.061806 | 不适用 | 0.212302 | 0.000000 | 3,625.33 ms |

候选平均延迟是旧 unified 的 1.112 倍，低于 1.25 倍门禁。不可覆盖门禁
`system_repair_development_gate_20260825_v4` 已重算为 `PASS`，失败项 0，允许消费一次
fresh test；门禁 SHA-256 为
`e7ba5bc7e300c10be6ce933eb05e5d7a0a723aac67cdaf3edd92e3d5cc4d0402`。

首个对比 job `29564321` 因错误 Hugging Face 缓存路径在加载前失败；job `29565493`
完成旧 unified 与 zero-shot 后发现单场景汇总错误要求对话证据，且评分失败前未落盘原始
输出。修复并补充回归测试后，job `29567157` 完成 Week 6 routed，历史失败日志保持不变。

## 尚未完成的最终门禁

- 唯一一次 fresh test job `29569338` 已通过 SSH 提交，等待 Spartan GPU 调度；提交前
  已确认输出目录不存在，脚本会再次核对候选、数据锁与门禁哈希。
- 候选 adapter 已下载到本地 ignored 输出目录并核对 SHA-256；发布配置仍保持旧 adapter，
  只有 fresh test 通过后才允许切换并构建最终包。
- 四场景生产 API 模型 smoke 尚未形成最终证据。
- OSS 目标和真实 adapter 尚不可用，因此私有 OSS 上传与下载哈希复验未执行。

以上任一模型或发布门禁未完成时均不得进入 `stg`。本报告不使用历史 test 结果生成新标签，
不新增人工标注，也不修改 Week 3、Week 6、Week 7 冻结产物。

当前实现基线为 `a11d082`，已推送 `origin/dev`；`origin/stg` 仍为 `132779b`，
`origin/main` 仍为 `1e3cdf7`。
