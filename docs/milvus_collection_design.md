# Week 4 Milvus 集合设计

## 范围

Week 4 使用官方 Milvus standalone 拓扑完成有界的本地 OTA 图片向量验证。Qwen2-VL 继续承担业务推理，不作为 embedding 接口。图片向量来自 `openai/clip-vit-base-patch32`，固定为 512 维，入库前必须完成 L2 归一化。

## 集合

集合名：`ota_business_image_vector`

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `vector_id` | INT64 | 主键、自增 |
| `business_id` | VARCHAR(128) | 必填、标量索引 |
| `image_id` | VARCHAR(128) | 必填、标量索引 |
| `multimodal_vector` | FLOAT_VECTOR(512) | 必填、HNSW/COSINE |
| `business_category` | VARCHAR(64) | 必填、标量索引 |
| `city` | VARCHAR(128) | 必填、标量索引 |
| `star_rating` | FLOAT | 必填、0 至 5、标量索引 |
| `price_range` | VARCHAR(32) | 必填、标量索引 |
| `image_type` | VARCHAR(64) | 必填、标量索引 |
| `embedding_model` | VARCHAR(128) | 必填、标量索引 |

动态字段关闭，一致性级别为 `Strong`。HNSW 的 `M`、`efConstruction`、查询 `ef` 和标量索引字段列表均从 `configs/milvus_week4.yaml` 读取。当前配置为 `M=16`、`efConstruction=128`、`ef=64`。

## SDK 边界

`src/retrieval/milvus_vectors.py` 提供导师要求的五项操作：

1. 有界批量入库；
2. 单条新增；
3. 使用固定标量白名单进行等值或 `IN` 过滤检索；
4. 按白名单条件删除；
5. 构建向量索引和标量索引。

SDK 拒绝缺失或额外字段、非有限值、未归一化向量、原始过滤表达式、未授权过滤字段和无条件删除，不猜测元数据，也不修改向量。

## 部署

`docker/milvus/docker-compose.yml` 固定 Milvus `v2.6.20`、etcd `v3.5.18` 和 MinIO 版本，包含命名持久化卷、健康检查、重启策略、CPU/内存限制和仅回环地址开放的 Milvus 端口。凭据通过未跟踪的 `docker/milvus/.env` 注入；仓库只保留 `.env.example` 占位模板。

```bash
python -m pip install -r requirements-milvus.txt
Copy-Item docker/milvus/.env.example docker/milvus/.env
# 替换本机 .env 中的占位值
docker compose --env-file docker/milvus/.env -f docker/milvus/docker-compose.yml config
docker compose --env-file docker/milvus/.env -f docker/milvus/docker-compose.yml up -d
```

生成向量、运行报告和 Milvus volumes 均不进入 Git。本地 8 GB GPU 在运行 CLIP 向量构建前必须停止 vLLM：

```bash
docker compose -f docker/docker-compose.yml stop vllm
python scripts/build_week4_clip_vectors.py --config configs/milvus_week4.yaml
python scripts/benchmark_week4_milvus.py --config configs/milvus_week4.yaml
```
