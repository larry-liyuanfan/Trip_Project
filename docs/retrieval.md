# 视觉检索与 Milvus

## 当前结构

图片由 `openai/clip-vit-base-patch32` 编码为 L2 归一化 512 维向量。Milvus collection 为
`ota_business_image_vector`，向量索引使用 HNSW 与 COSINE，标量过滤限定在明确白名单。

核心字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `vector_id` | INT64 auto id | 主键 |
| `business_id` / `image_id` | VARCHAR | 业务与图片身份 |
| `multimodal_vector` | FLOAT_VECTOR(512) | CLIP 图像向量 |
| `business_category` / `city` / `price_range` / `image_type` | scalar | 过滤字段 |
| `star_rating` | FLOAT | 星级过滤 |
| `embedding_model` | VARCHAR | 强制向量模型身份 |

HNSW 的 `M`、`efConstruction` 和查询 `ef` 由配置提供，不在业务代码硬编码。用户文本中无法
映射到允许字段的条件必须作为未应用约束返回，不能假装参与排序。

## 入口

```bash
python scripts/build_week4_clip_vectors.py --config configs/milvus_week4.yaml
python scripts/benchmark_week4_milvus.py --config configs/milvus_week4.yaml
python scripts/load_system_retrieval.py --config docker/system/milvus_system.yaml --vectors <vectors.npz> --metadata <metadata.jsonl>
```

`src/retrieval/milvus_store.py` 封装建表、索引、批量/单条写入、过滤检索和删除；
`src/retrieval/visual_search.py` 连接 CLIP 与 Milvus；`src/retrieval/query_inputs.py` 负责有界查询
解析。生产模式检索不可用时显式失败，不返回样例 keyword 结果。

本地交付包的 retrieval 层包含 1,000 条向量、metadata 与工程基准。该 Recall 结果验证索引
实现，不等同于人工业务相关性结论。
