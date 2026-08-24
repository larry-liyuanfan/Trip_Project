# 项目报告索引

本目录只保存导师可直接阅读的交付报告。当前会议优先阅读
`system_consolidation_repair_report.md`；模型原始输出、逐条评分、checkpoint 和大规模数据
位于 Git 忽略的 `outputs/`，实验过程与失败证据位于 `docs/experiments.md` 和
`experiments/`。报告中的数字必须能由这些产物追溯。

## 综合报告

- `system_consolidation_repair_report.md`：Week 1-7 问题实际修复、统一运行时、Milvus
  实测、模型门禁和发布状态。
- `week1_to_week5_qwen3_vl_4b_rerun_report.md`：使用单 A10 上的
  `Qwen3-VL-4B-Instruct` 对 Week 1-5 进行有边界的重跑、验证与成本评估。
- `week1_to_week4_qwen37_overall_report.md`：迁移到阿里云
  `qwen3.7-plus` 后的 Week 1-4 整体交付与效果总结。

## 分周报告

| 周次 | 报告 | 说明 |
| --- | --- | --- |
| Week 1 | `../docs/internship_weekly_summary.md` | 工程基础、API、初始数据准备与验证记录 |
| Week 2 | `yelp_multimodal_data_processing_report_part1.md` | Yelp 全量处理、对齐、CLIP 降噪与质量统计 |
| Week 3 | `week3_zero_shot_baseline_report.md` | 原 Qwen2-VL 基线和标准化 Prompt 评测 |
| Week 3/4 重跑 | `qwen37_previous_weeks_rerun_report.md` | Qwen3.7 baseline、standardized 与 Prompt 候选重跑 |
| 行程修复 | `qwen37_itinerary_repair_report.md` | Qwen3.7 行程截断根因和 `standardized_v4` 结果 |
| Week 4 | `week4_prompt_optimization_report.md` | 原 Prompt 候选比较与共同评分口径 |
| Week 4 | `week4_bad_cases.md` | 真实错误案例分类 |
| Week 4 | `week4_milvus_deployment_performance_report.md` | Milvus CRUD 与小规模性能基线 |
| Week 5 | `week5_dataset_quality_report.md` | 全量预标注、单人预算内人工验收与多轮候选质量证据 |
| Week 6 | `week6_qlora_quality_report.md` | 三场景 QLoRA 终态、行程业务门禁与专项优化结论 |
| Week 6 后评审 | `week6_post_training_improvement_review.md` | 完成度判断、冻结指标短板、前沿方法映射与非污染改进门禁 |
| Week 7 | `week7_multitask_context_report.md` | 会议版周报：核心三场景 PASS、corrected-dialogue FAIL、DPO 拒绝与证据索引 |

`week3_v2_recuration_status.md` 是历史过程状态，不代表当前最终结论。
