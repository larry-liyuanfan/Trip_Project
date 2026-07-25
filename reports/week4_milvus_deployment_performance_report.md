# Week 4 Milvus Deployment and Performance Report

## Deployment

- Compose: `docker/milvus/docker-compose.yml`
- Milvus: `milvusdb/milvus:v2.6.20`
- PyMilvus: `2.6.16`
- Topology: standalone + etcd + MinIO
- Persistence: three named volumes
- Network exposure: Milvus `19530` and health `9091`, loopback only
- Collection: `ota_business_image_vector`
- Vector index: HNSW/COSINE, `M=16`, `efConstruction=128`, query `ef=64`
- Scalar indexes: all eight filterable metadata fields

The three containers reached healthy state. A real SDK connection created the
fixed collection, verified its ten-field Schema, and built one HNSW plus eight
scalar indexes without replacing existing data.

## CRUD and Performance

The benchmark used 20 real Yelp OTA images and CUDA-generated, normalized
`openai/clip-vit-base-patch32` vectors. vLLM was stopped before CLIP started.

| Measurement | Result |
| --- | ---: |
| Inserted vectors | 20 |
| Batch insert | 19 |
| Single insert | 1 |
| Filtered-search hits | 1 |
| Deleted rows | 1 |
| Post-delete hits | 0 |
| Remaining rows | 19 |
| HNSW build time | 5.6621 s |
| Search queries / K | 10 / 5 |
| Mean latency | 7.7982 ms |
| P95 latency | 10.7236 ms |
| Recall@5 | 1.0000 |

Environment: Windows 11 (`10.0.26200`), Python 3.13.13, Intel64 Family 6
Model 183. These are small local measurements only; no production-performance
claim is made. Reproduce after stopping vLLM:

```bash
docker compose -f docker/docker-compose.yml stop vllm
python scripts/build_week4_clip_vectors.py --config configs/milvus_week4.yaml
python scripts/benchmark_week4_milvus.py --config configs/milvus_week4.yaml
```
