# 当前架构决策

本文件只记录接手者需要遵守的当前决策。逐周 ADR 原文可从 Git 历史 `2eb51d4` 查阅。

1. 正式版本固定为 `trip-qwen3-vl-8b-week8-final-v1`；配置、四层包、Prompt、Schema 和
   已记录哈希不可覆盖。行为变化必须创建新版本。
2. 基座固定为 `Qwen/Qwen3-VL-8B-Instruct` revision
   `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，运行时使用 Transformers + PEFT；adapter
   model SHA-256 固定为 `c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a`。
3. 商品采用 v12 观察链，行程采用 v13/v5；商品参考为自动 silver，human=0，价位指标保持
   `N/A/PENDING`，不得改写为人工视觉准确率。
4. 生产模式 fail closed。模型、adapter、Schema 或检索失败必须返回明确错误；Schema 失败
   最多进行一次模型级纠错，脚本不得猜字段或修改模型语义。
5. `/health` 只表示进程存活，`/ready` 检查 release、模型、adapter、Schema、CLIP 与
   Milvus。业务接口保持 Pydantic 边界和稳定 OpenAPI 路径。
6. CLIP 与 VLM 解耦；检索向量固定 512 维，Milvus 使用 HNSW/COSINE 和标量白名单过滤。
7. 原始 Yelp、生成数据、冻结评测、模型、adapter、向量库、运行输出和密钥不进入 Git。
   自动 silver、人工标签、metadata 和模型输出必须保持独立身份。
8. Git 外只保留 `outputs/releases/trip-qwen3-vl-8b-week8-final-v1`。本地验证替代 OSS、
   Spartan 或旧云部署依赖。
9. 分支仅保留 `dev`、`stg`、`main`；开发使用短期 `feature/*`，验证后依次晋级，不直接在
   `main` 开发。
