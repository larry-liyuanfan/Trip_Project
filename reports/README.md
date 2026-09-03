# 报告索引

本目录按分支保存不同粒度的证据：

- `main`：`project_summary.md` 与 `final_delivery_status.md`，作为最终接手入口。
- `stg`：在最终报告之外，通过 `weekly/week01.md` 至 `weekly/week08.md` 保存每周一份的稳定总结。
- `dev`：在 `stg` 内容之外保存详细优化报告、bad case 和经审查的轻量权衡证据。

当前正式发布身份、已优化项和待优化项以 `final_delivery_status.md` 为准。历史报告中的
阶段性选择和指标不得覆盖当前 release 配置；需要重现实验时再结合 `experiments/` 中的
机器可读身份与 Git 提交读取。

当前 `dev` 搜索算法证据入口为
`development/reviews/search_algorithm_evidence_enhancement_report.md`；对应最新机器证据为
`../experiments/search_algorithm_evidence_v4.json` 与
`../experiments/context_focus_evidence_v5.json`。二者均为 synthetic/weak 开发证据，
human support=0，不覆盖正式 release 或冻结 Fresh Test。

