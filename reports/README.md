# 报告索引

当前搜索评分入口优先读取 [scorer v2 审计交接](scorer_v2_audit_20260914/HANDOFF.md)。
以下 v4/v8/v9 等旧报告和机器 JSON 是当时旧协议历史证据，原文件保持不变；
旧 gate 不继承为 v2 门禁，追加重算也不是新模型提升。

最新价格冲突续训前置审计：[交接与阻断记录](price_conflict_abstention_20260914/HANDOFF.md)。
身份内容维度不全，未生成数据、未训练/推理；不把 inventory 阻断当作模型负实验。

最新输入质量修复与配对诊断：`development/reviews/card_render_repair_v10.md`。
该诊断使用已查看的 v9 development，不构成独立测试提升。
已完成配对实跑，风格改善但价位回退导致诊断门 FAIL；完整结果与 SHA 在报告中。

本目录按分支保存不同粒度的证据：

- `main`：`project_summary.md` 与 `final_delivery_status.md`，作为最终接手入口。
- `stg`：在最终报告之外，通过 `weekly/week01.md` 至 `weekly/week08.md` 保存每周一份的稳定总结。
- `dev`：在 `stg` 内容之外保存详细优化报告、bad case 和经审查的轻量权衡证据。

当前正式发布身份、已优化项和待优化项以 `final_delivery_status.md` 为准。历史报告中的
阶段性选择和指标不得覆盖当前 release 配置；需要重现实验时再结合 `experiments/` 中的
机器可读身份与 Git 提交读取。

旧协议时点的 `dev` 搜索算法证据入口为
`development/reviews/search_algorithm_evidence_enhancement_report.md`；对应最新机器证据为
`../experiments/search_algorithm_evidence_v4.json` 与
`../experiments/context_focus_evidence_v5.json`、
`../experiments/semantic_robustness_evidence_v7.json` 和
`../experiments/semantic_robustness_evidence_v9.json`、
`../experiments/no_result_stress_evidence_v8.json`、
`../experiments/distributed_milvus_http_evidence_v6.json`，以及只证明字节/来源隔离的
`../experiments/retrieval_query_leakage_evidence_v4.json`。质量证据均为 synthetic/weak，
human support=0；v7 是门槛失败的负实验，v8 通过压力门但新候选与固定基线持平。
v9 同样是门槛失败的负实验；双节点 Milvus HTTP 性能证据通过固定探索门，但不支持生产 SLA。
这些旧协议证据不覆盖正式 release、冻结 Fresh Test 或当前 scorer v2 结论。

