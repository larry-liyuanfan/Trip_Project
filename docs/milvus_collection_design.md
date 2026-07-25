# Week 4 Milvus Collection Design

## Scope

Week 4 uses the official Milvus standalone topology for a bounded local OTA
image-vector test. Qwen2-VL remains the business inference model; it is not
treated as an embedding endpoint. Image vectors come from
`openai/clip-vit-base-patch32`, are 512-dimensional, and must be L2-normalized
before insertion.

## Collection

Collection: `ota_business_image_vector`

| Field | Type | Constraint |
| --- | --- | --- |
| `vector_id` | INT64 | primary key, auto increment |
| `business_id` | VARCHAR(128) | required, scalar indexed |
| `image_id` | VARCHAR(128) | required, scalar indexed |
| `multimodal_vector` | FLOAT_VECTOR(512) | required, HNSW/COSINE |
| `business_category` | VARCHAR(64) | required, scalar indexed |
| `city` | VARCHAR(128) | required, scalar indexed |
| `star_rating` | FLOAT | required, 0 to 5, scalar indexed |
| `price_range` | VARCHAR(32) | required, scalar indexed |
| `image_type` | VARCHAR(64) | required, scalar indexed |
| `embedding_model` | VARCHAR(128) | required, scalar indexed |

Dynamic fields are disabled and consistency is `Strong`. HNSW parameters
`M`, `efConstruction`, query `ef`, and the scalar-index list are read from
`configs/milvus_week4.yaml`. The checked-in defaults are `M=16`,
`efConstruction=128`, and `ef=64`.

## SDK Boundary

`src/retrieval/milvus_vectors.py` exposes the five required operations:

1. bounded batch insertion;
2. single-row insertion;
3. vector search with equality or `IN` filters on the fixed scalar whitelist;
4. filtered deletion;
5. vector and scalar index construction.

The SDK rejects missing or extra entity fields, non-finite or non-normalized
vectors, raw filter expressions, unsupported filter fields, and unbounded
deletes. It never guesses metadata or modifies vectors.

## Deployment

`docker/milvus/docker-compose.yml` fixes Milvus `v2.6.20`, etcd `v3.5.18`,
and a fixed MinIO release. It provides named persistent volumes, health checks,
restart policies, CPU/memory limits, and loopback-only Milvus ports. Install
the separate client dependency group and start the stack with:

```bash
python -m pip install -r requirements-milvus.txt
docker compose -f docker/milvus/docker-compose.yml config
docker compose -f docker/milvus/docker-compose.yml up -d
```

Generated vectors, reports, and Milvus volumes are ignored. On the local 8 GB
GPU, stop vLLM before running the CLIP vector builder:

```bash
docker compose -f docker/docker-compose.yml stop vllm
python scripts/build_week4_clip_vectors.py --config configs/milvus_week4.yaml
python scripts/benchmark_week4_milvus.py --config configs/milvus_week4.yaml
```
