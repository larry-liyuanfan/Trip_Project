#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

test -f docker/aliyun/.env
test -f docker/milvus/.env
test -f secrets/dashscope_api_key
test -s secrets/dashscope_api_key

docker compose --env-file docker/aliyun/.env \
  -f docker/aliyun/docker-compose.yml config --quiet
docker compose --env-file docker/milvus/.env \
  -f docker/milvus/docker-compose.yml config --quiet

docker compose --env-file docker/milvus/.env \
  -f docker/milvus/docker-compose.yml up -d
docker compose --env-file docker/aliyun/.env \
  -f docker/aliyun/docker-compose.yml up -d --build

for _ in $(seq 1 30); do
  if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
    docker compose --env-file docker/aliyun/.env \
      -f docker/aliyun/docker-compose.yml ps
    docker compose --env-file docker/milvus/.env \
      -f docker/milvus/docker-compose.yml ps
    exit 0
  fi
  sleep 2
done

docker compose --env-file docker/aliyun/.env \
  -f docker/aliyun/docker-compose.yml logs --tail=100 api
exit 1
