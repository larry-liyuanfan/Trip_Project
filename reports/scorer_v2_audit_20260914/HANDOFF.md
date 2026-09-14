# Scorer v2 audit — 2026-09-14

## 完成内容与提交边界

本工作包完成搜索评分契约修复与原预测离线重算；没有训练、模型推理、阈值重选或 Fresh Test
120 访问。全部重算标签仍是 synthetic/weak，human support=0。本次不形成新的独立测试提升。

- 工作区：独立 Codex-managed `91ac/Trip_Project`；分支 `codex/trip-scorer-audit-20260914`。
- fetch 后确认 main/origin/main 为 `d6e0d80077e21dffca29d27cd2401b14620ec0b8`。
- 固定依赖：`43bed689cbc4c3dc670c884857dc146494528afc`；旧 `b8bb` worktree 与旧分支
  `codex/trip-relevance-eval` 只读、干净、未改动。
- 独立依赖 merge：`7706670c484c68af53b35854487ac0587cb33c9f`，无冲突。
- 评分代码/测试/契约提交：`315c469276a9b249f0e83523a1684d6954c864c7`。
- 本目录证据位于后续独立 evidence commit；用 `git log --oneline -- reports/scorer_v2_audit_20260914`
  定位。只推送本工作分支，不自行合入 dev/stg/main。
- 正式 release、四层包、旧配置/模型/Prompt/历史实验 JSON 不变。未改求职主简历。

## 修复的真实缺陷

| 缺陷 | 修复 |
| --- | --- |
| IDCG 只对返回列表排序，漏掉高相关商品仍可能满分 | IDCG 由完整有效 qrels 生成；反例 `[返回 grade1] / [qrels grade3, grade1]` 得 nDCG@1=1/7 |
| Recall 分母由结果里的 `relevant_total` 或已命中正例提供 | 分母始终来自完整 qrels 的 binary positives；漏召回的 gold 不会消失 |
| 无结果率是 `no_result` 切片占比 | 统计实际成功空预测；另列 no-positive-qrels、dataset slice 比例和两种准确率 |
| 同一 annotator 重复可冒充双人 | 标注者 ID（含大小写变体）去重校验；每对 query-document 要求独立评分及一致/仲裁 |
| human 模式仍调用 metadata 弱规则 | 人工声明模式只读明确 qrels，缺失/冲突拒绝，无 metadata fallback |
| 失败查询从 ranking macro 中排除、过滤统计混入无过滤查询 | 失败在可评价查询中记零；过滤指标仅以有支持过滤条件的查询为分母 |

完整规则与输入结构见 [docs/search_scorer_v2.md](../../docs/search_scorer_v2.md)。默认 binary
threshold=2、graded gain=`2**grade-1`、K=5/10。无 binary positive 的 Recall/MRR 为 null；只有
IDCG=0 才使 nDCG 为 null。unjudged 保留名次、记零并计数；重复标识拒绝。所有 macro 与切片
指标有独立分母。人工协议合法仅得到 `human_protocol_complete`，不自动获得真实人标认证；
`promotion_eligible_as_human_ground_truth` 保持 false，身份和原图字节仍待外部核验。

旧 `_aggregate_search_method_v1` 只为不可变旧证据的完整性验证而保留，明确标记旧无结果率
误名；当前评分不调用它。旧推理配置在新 scorer 下失败关闭，必须另行预注册新版本，不能
给旧已看过的 holdout 加一个字段后重跑。旧 gate 不自动继承为 scorer-v2 gate。

## 离线重算与正负结论

读取 Iris `yzhang3504` 的既有检索产物；没有访问另一个账户，也没有 sbatch、重启或恢复训练。
v4 原始查询文件仍在；v8 查询资产不在当前远端 assets 目录，使用原始提交
`2acf4018aeba3cc717edcac8d2d7836ae15fe1de` 的生成器重建，完整 bundle 与原 lock 完全一致。
这只是原查询字节/标签身份恢复，不是重新构建独立评测。v4/v8 原预测、原 metrics、query 和
annotation 的文件 SHA 及 canonical 绑定均通过。正式 metadata 共 1000 个 document ID，
v4/v8 qrels 分别为 24,000 / 40,000 个程序规则判断。

### 在原排名查询子集上只纠正评分

| 原运行/方案 | 同一旧查询分母 | 旧 nDCG@10 | 完整 qrels 修正后 nDCG@10 |
| --- | ---: | ---: | ---: |
| v4 CLIP exact / Milvus | 各 12 | 0.843503 | 0.355818 |
| v4 结构化过滤 / 轻量重排 / business guard | 各 12 | 1.000000 | 0.828990 |
| v8 hard-filter CLIP | 20 | 1.000000 | 0.652563 |
| v8 fixed margin guard / dual-centroid candidate | 各 20 | 0.900000 | 0.619412 |

这些是评分纠错，不是模型退化或新模型效果。在固定旧预测上，旧“满分”不能继续当作完整
语义相关性证据。v4 轻量重排没有相对结构化过滤的 ranking 增益；v8 candidate 仍与固定
baseline 相同，是中性/无增益结果。没有候选因此获得新的质量或延迟资格。

### 新契约全查询宇宙结果

| 数据/方案 | Recall@10 | MRR@10 | nDCG@10 | 实际空结果 | 支持过滤正确 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 exact / Milvus | 各 0.010299 | 各 0.656250 | 各 0.297172 | 各 0/24 | 各 0/16 |
| v4 structured / light rerank | 各 0.059589 | 各 0.750000 | 各 0.414495 | 各 4/24 | 各 16/16 |
| v4 business guard | 0.059589 | 0.750000 | 0.414495 | 12/24 | 16/16 |
| v8 hard-filter CLIP | 0.034023 | 0.571429 | 0.326282 | 8/40 | 16/16 |
| v8 fixed guard / candidate | 各 0.032565 | 各 0.500000 | 各 0.309706 | 各 19/40 | 各 16/16 |

v4 Recall/MRR query denominator=16，nDCG denominator=24；v8 分别为 28 和 40。原因是
binary relevance 与 graded gain 独立，且不再按旧 dataset `no_result` 标签排除 ranking。
每个查询的 Recall 分母为完整 qrels 的正例数，macro sum/count 和所有切片见 JSON。
以上全宇宙数字**不能**直接与原 12/20 分母相减后称为模型变化。

进一步发现：v4 的 dataset no-result=12/24，但 qrels 无 binary positives=8/24；v8 为
20/40 vs 12/40。例如“城市无结果/过滤冲突”的 metadata 规则仍可能给同业态 grade2。
v4 guard 的 dataset no-result accuracy=24/24，而 qrels 口径仅 20/24；v8 两 guard 分别是
35/40 与 27/40。差异证明弱规则不能替代真实用户相关性判断。本轮不修改这些旧标签来追分。

## 证据索引与不可重算项

- [v4_final_offline_recompute_v2.json](v4_final_offline_recompute_v2.json)：原 job `29996439`、
  L40S、24 queries、1000 documents；包括旧/新指标、分母、切片、完整输入/来源/产物 SHA。
- [v8_validation_offline_recompute_v2.json](v8_validation_offline_recompute_v2.json)：原 job
  `30005527`、L40S、40 queries、1000 documents；同样保留完整身份与边界。
- [verification.json](verification.json)：当前源码提交的 CPU 回归、CLI 配置、正式包只读校验、
  Compose 静态配置和 diff check 的实际结果与日志 SHA。
- [artifact_index.json](artifact_index.json)：本次追加证据的文件 SHA 与 canonical SHA。

| 历史记录 | 本轮状态 | 原因/限制 |
| --- | --- | --- |
| Commons v1 搜索 | NOT_RECOMPUTABLE_FROM_RETAINED_SCOPED_INPUTS | 旧本地只找到 summary，没有逐 query 预测；限域远端检查也未定位 `search_results_v1.jsonl`。无法从汇总 nDCG 恢复 IDCG |
| automated v2 搜索 | NOT_RECOMPUTABLE_FROM_RETAINED_SCOPED_INPUTS | 两个已知 run 目录与当前资产范围未定位 `holdout_results.jsonl`，无完整预测身份链。保留旧汇总，不伪造重算 |
| v4 development | NOT_RECOMPUTED_IN_THIS_PACKAGE | 已拿到文件，但本包仅选择有已提交原始预测/metrics 文件 SHA 的 v4 final；不把新获取的文件哈希冒充历史预绑定 |
| Fresh Test 120、VLM、延迟 | NOT_IN_SCOPE | 不读取/重跑 Fresh Test；没有新 VLM 语义或端到端延迟结论 |

“未定位”仅描述本轮检查的旧本地 outputs 与 Trip 已知远端目录，不声称全盘/全服务器已删除。
原本负实验的结论保留；历史1.0 Recall仍是 ANN-vs-exact；2.4097 ms仍是向量查询阶段耗时。

## 当前提交的实际验证

源码提交 `315c469276a9b249f0e83523a1684d6954c864c7` 的最终验证：

- `python -m unittest discover -s tests -v`：1009 项，1007 passed、2 skipped，失败/错误 0，
  约 125.82 秒。两项 skip 是既有 Week8 Spartan launch-script 检查（本地交接刻意不含对应
  launch scripts），不是评分器测试跳过。新增评分/CLI 用例全部运行。
- 定向七个测试文件共 57 项曾全部通过；最终完整回归已覆盖相同用例及最后的 provenance 修订。
- `python scripts/tripctl.py validate`：PASS，无 errors。
- `python scripts/verify_final_delivery.py <formal-package>`：PASS，四层 SHA、runtime 隔离导入
  和正式包身份一致。包内 948 项是历史封装证据，不计入此次 1009 项。
- Compose `config --quiet`：exit 0。当前沙箱无法读取用户 Docker config，产生 warning；这里只
  证明静态配置通过，不声明 live Docker/Milvus 服务验证。
- `git diff --check`：PASS。新增报告在提交前另行执行 staged diff check。
- 原分支 `43bed689...` 仍干净，main 仍为 `d6e0d800...`；无共享 checkout 写入。

精确命令、运行 CPU/OS/Python、耗时、skip 原因及日志 SHA 见 `verification.json`。
日志保留于 Git 忽略的 `outputs/scorer-v2-verification-final`。无 live GPU/Milvus 推理与新
端到端 benchmark。本工作包 CPU 重验证已结束，**重验证槽已释放**；不再后台计算或监控。

## 重现入口

输入文件仅保留在 Git 忽略的 `outputs/scorer-v2-audit-inputs`，不提交原始 Yelp metadata、
图片或逐条预测。持有相同历史输入者可运行：

```bash
python scripts/audit_search_scorer_v2.py --case v4_final \
  --input-root outputs/scorer-v2-audit-inputs --retrieval-archive <formal-package>/retrieval.tar.gz \
  --output <new-versioned-v4-report.json>
python scripts/audit_search_scorer_v2.py --case v8_validation \
  --input-root outputs/scorer-v2-audit-inputs --retrieval-archive <formal-package>/retrieval.tar.gz \
  --output <new-versioned-v8-report.json>
python scripts/verify_search_scorer_v2_audit.py \
  --output-dir <new-verification-dir> --release-dir <formal-package>
```

v8 重建必须使用上文原提交，不是未来修改后的生成器；导出限定的
`scripts/build_no_result_stress_pool_v8.py`、`scripts/build_exploration_pool_v4.py`、
`src/evaluation/relevance_evidence.py` 后运行生成器，并传原 `--prior-v4-lock` 与
`--expected-lock`。若 lock 不同立即拒绝。不要恢复旧工作区或改写既有文件。

产物内 `scorer_files_sha256` 是此次实际执行文件的 byte hash；跨平台重现应同时使用 Git SHA
和报告的 canonical 输入/指标身份，不能把 CRLF/LF 差异冒充实验差异。报告没有新的性能计时。

## 尚未完成与下一步边界

- 真人相关性标注及身份/原图字节真实性核验仍未完成。`judged_pool` 完整只指声明候选宇宙，
  不能推出全库无相关商品；full-index weak qrels 同样不证明真实业务语义。
- 新 scorer 下的新数据、质量/延迟门槛与晋级需另行预注册；本轮不重选参数或消费旧测试。
- 新 Spartan 实验由总控验收后另派。本轮没有待恢复/重提交的 Slurm 作业，也未创建监控任务。
- 可建议的最小后续实验：在全新、source/image/query 与历史 split 隔离的数据上，仅改变
  “price conflict/unknown abstention 的训练样本组成”；固定基座、adapter 起点、Prompt、
  渲染器、训练超参和推理参数，先比较 development 的价位 F1、冲突 abstention、unsupported
  hallucination 与其他字段非退化，再走同硬件延迟门。门槛与数据锁必须在运行前另行提交；
  不使用已看过的 v10 paired development 调参。本段只是建议，未提交任何训练。

## 可用于简历的候选表述（不修改主简历）

1. 建立 OTA 多模态检索的可追溯离线评测链，分离 ANN 一致性、弱标签相关性与服务时延，
   实现完整 qrels 的 Recall/MRR/nDCG、空结果与过滤指标，并通过反例测试修复评分高估。
2. 在保留原预测与数据哈希的前提下审计 CLIP、Milvus、结构化过滤及轻量重排，揭示旧
   nDCG 归一化偏差和 synthetic 无结果标签冲突，保留无增益实验与明确的证据边界。

不能写“人工搜索准确率提升”“新独立测试提升”“nDCG=1 的业务搜索效果”或“2.41 ms
端到端检索”。本轮的正结果是评测正确性与证据完整性，不是新的模型性能提升。
