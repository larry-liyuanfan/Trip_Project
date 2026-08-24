# 系统收敛修复与统一封装报告

日期：2026-08-24  
当前状态：`PARTIAL`  
发布结论：代码与检索工程已通过；模型修复门禁尚未完成，不进入 `stg`

## 已完成修复

| 范围 | 实际结果 |
| --- | --- |
| 跨平台配置哈希 | 配置文本统一按 LF canonical bytes；Windows/Linux 回归通过 |
| 生产 API | Qwen3-VL + PEFT/NF4 运行时、三场景接口、对话接口、视觉检索接口已实现 |
| 失败处理 | 生产模式关闭静默 fallback；Schema 最多一次模型级纠错并保留两次原始输出 |
| 就绪检查 | `/health` 仅检查进程；`/ready` 核验 adapter、模型、Prompt、Schema、CLIP、Milvus |
| Week 5 v2 | 新池 80,000 条、44 条同层替换、64 条修复队列、五维评测冲突 0 |
| 新数据锁 | train/development/test=1,980/168/120，五维跨 split 冲突 0，test 未消费 |
| 检索 | 1,000 张真实 OTA 图片完成 CLIP 512 维编码和 Milvus 实测 |
| 封装 | 统一 Compose、release manifest、四层私有 OSS 打包器和 `tripctl` 已实现 |
| 测试 | 当前工作树及全新 checkout 的完整 `unittest` 均 482/482；`git diff --check` 通过 |

## Week 5 修复池

| 指标 | 结果 |
| --- | ---: |
| 候选总数 | 80,000 |
| 商品/售后/行程 | 50,000 / 20,000 / 10,000 |
| 历史 Schema-valid | 79,936 |
| 待模型修复 | 64 |
| 不可读输入替换 | 44 |
| Schema/JSON 修复 | 19 / 1 |
| sample/source/image SHA 唯一数 | 80,000 / 80,000 / 80,000 |
| 五维评测集冲突 | 0 |

当前 v2 状态为 `AWAITING_MODEL_REPAIR`。64 条尚未使用 Qwen3-VL-8B unified adapter
重新推理，因此不能声明 80,000/80,000 Schema-valid；历史人工 accepted 统计没有增加。

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

## 未通过的最终门禁

- Spartan 尚未在本轮完成登录，新的 Prompt pilot、继续 SFT 和 fresh development/test
  评测尚未运行。
- 本地没有 checkpoint-226 adapter 文件，`tripctl doctor` 正确返回 `not_ready`；不能
  构建含真实 adapter 的发布包或完成四场景模型 smoke。
- Week 5 的 64 条模型修复未运行，最终 80,000/80,000 Schema-valid 门禁未通过。
- OSS 目标和真实 adapter 尚不可用，因此私有 OSS 上传与下载哈希复验未执行。

以上任一模型或发布门禁未完成时均不得进入 `stg`。本报告不使用历史 test 结果生成新标签，
不新增人工标注，也不修改 Week 3、Week 6、Week 7 冻结产物。
