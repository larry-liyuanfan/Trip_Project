# Week 4 Milvus 部署与性能报告

## 部署

- Compose：`docker/milvus/docker-compose.yml`
- Milvus：`milvusdb/milvus:v2.6.20`
- PyMilvus：`2.6.16`
- 拓扑：standalone + etcd + MinIO
- 持久化：3 个命名卷
- 网络暴露：Milvus `19530` 和健康端口 `9091`，仅回环地址
- 集合：`ota_business_image_vector`
- 向量索引：HNSW/COSINE，`M=16`、`efConstruction=128`、查询 `ef=64`
- 标量索引：8 个可过滤元数据字段

三个容器均达到 healthy。真实 SDK 连接创建并验证了固定十字段 Schema、
一个 HNSW 索引和八个标量索引，没有用 Qwen2-VL 充当 embedding 接口。

## CRUD 与性能

基准使用 20 张真实 Yelp OTA 图片及 CUDA 生成的归一化
`openai/clip-vit-base-patch32` 向量。启动 CLIP 前已停止 vLLM。

| 实测项 | 结果 |
| --- | ---: |
| 入库向量 | 20 |
| 批量入库 | 19 |
| 单条新增 | 1 |
| 过滤检索命中 | 1 |
| 删除 | 1 |
| 删除后命中 | 0 |
| 逻辑可见行 | 19 |
| HNSW 构建耗时 | 5.6621 s |
| 查询次数 / K | 10 / 5 |
| 平均延迟 | 7.7982 ms |
| P95 延迟 | 10.7236 ms |
| Recall@5 | 1.0000 |

环境为 Windows 11（`10.0.26200`）、Python 3.13.13、Intel64 Family 6
Model 183。以上只是本机小规模实测，不宣称生产级性能。

## 审查问题修复

- `configs/milvus_week4.yaml` 只保存环境变量名；Compose 只引用
  `MINIO_ROOT_USER` 和 `MINIO_ROOT_PASSWORD`。仓库提交
  `docker/milvus/.env.example`，真实 `.env` 保持忽略。
- 已使用不落盘的新随机凭据重建 MinIO 和 Milvus 容器；三个服务恢复
  healthy，原有 19 条逻辑可见向量仍可查询。
- 基准脚本在写入前同时拒绝既有输出和非空集合，避免重复运行累积数据。
- `actual_vector_count_*` 现在来自 Milvus `count(*)` 逻辑可见行查询，
  不再根据输入列表长度推断；插入后必须为 20，删除后必须为 19。

## 复现命令

```bash
python -m pip install -r requirements-milvus.txt
Copy-Item docker/milvus/.env.example docker/milvus/.env
# 替换本机 .env 中的占位值
docker compose --env-file docker/milvus/.env -f docker/milvus/docker-compose.yml config
docker compose --env-file docker/milvus/.env -f docker/milvus/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml stop vllm
python scripts/build_week4_clip_vectors.py --config configs/milvus_week4.yaml
python scripts/benchmark_week4_milvus.py --config configs/milvus_week4.yaml
```

基准命令要求全新的空集合和不存在的输出路径；不会自动删除既有数据。
