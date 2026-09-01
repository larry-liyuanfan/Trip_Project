# 开发证据索引

本目录仅在 `dev` 保留，用于解释模型迁移、错误分析和未晋级方案。它不是当前 release 的
配置来源；正式身份始终以 `configs/releases/qwen3_vl_system_final_v1.json` 为准。

## 分类

- `model_migration/`：Qwen3.7-Plus 与 Qwen3-VL 迁移、重跑及跨周对比。
- `reviews/`：数据修订、Week 4 bad case、Milvus 验证和训练后评审。
- `week8/`：Week 8 优化方向及完整商品理解专项报告。

Week 8 商品报告引用的设施路由权衡原始证据位于
`outputs/week8/review/week8_facility_routing_tradeoff_20260829_v1.json`。该 JSON 只记录
聚合指标、哈希和晋级决策，不包含图片、密钥、模型权重或个人数据。报告与证据均保留
原始阶段性口径，不能据此把 development-only 路由声明为正式发布能力。

