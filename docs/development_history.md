# Dev 开发与复现索引

本文件仅用于 `dev`。正式发布身份仍以
`configs/releases/qwen3_vl_system_final_v1.json` 为准，历史配置不得作为默认运行配置。

## 当前保留范围

- `reports/weekly/`：Week 1 至 Week 8 每周一份完整报告。
- `reports/development/`：模型迁移、bad case、训练后评审和 Week 8 商品专项报告。
- `outputs/week8/review/week8_facility_routing_tradeoff_20260829_v1.json`：报告直接引用的
  轻量聚合权衡证据。
- 40 个恢复的非云端历史测试文件及其最小脚本、Prompt、release 和实验配置依赖。
- 历史 requirements、decisions、experiments、Prompt 与评测设计文档。

恢复后的 `dev` 测试集为 912 项通过、2 项跳过。两个跳过项只检查 Spartan 作业脚本；
项目已决定不再依赖 Spartan，因此不恢复对应 `.sbatch` 文件，也不把跳过解释为模型通过。
`main/stg` 的精简交付测试集仍为 521 项。

## 未恢复内容

- Spartan 作业、迁移工具和专用依赖。
- OSS 上传工具、Bucket 配置和云端副本。
- 旧阿里云/GPU ECS 部署文件及密钥路径。
- 模型缓存、adapter、原始 Yelp 图片、生成数据集和大体量运行输出。

这些内容不是当前运行依赖。清理前的完整文本仍可在 Git 提交 `2eb51d4` 查看；不要为了
复现旧环境而重新启用云资源。正式四层包仅存在于本机 Git 外目录
`outputs/releases/trip-qwen3-vl-8b-week8-final-v1`，GitHub 仓库本身不包含模型资产。

## 使用方式

1. 当前系统运行与交接读取根目录 `README.md`、`docs/model_handoff.md` 和正式 release。
2. 复查 Week 8 商品权衡时读取商品专项报告、设施路由 JSON 和对应恢复测试。
3. 复现实验时显式指定历史配置，不修改或覆盖正式 release config。
4. 新代码提交可选择性晋级到 `stg/main`；本目录及 development-only 证据留在 `dev`。
