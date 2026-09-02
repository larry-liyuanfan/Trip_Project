# 搜索、VLM 与端到端证据增强协议

本协议属于 `dev` 开发证据，不修改正式 release
`trip-qwen3-vl-8b-week8-final-v1`，也不重新消费已锁定的 Fresh Test 120。所有新输出必须
写入不存在的目标文件；正式四层包只读，归档或成员哈希不匹配时立即失败。

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
