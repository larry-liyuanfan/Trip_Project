#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${project_root}/docker/gpu/docker-compose.yml"
env_file="${GPU_ENV_FILE:-${project_root}/docker/gpu/.env}"

if [[ ! -f "${env_file}" ]]; then
  echo "缺少 ${env_file}；请从 docker/gpu/.env.example 创建并复核。" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

if [[ "${VLLM_BIND_ADDRESS:-}" != "127.0.0.1" ]]; then
  echo "拒绝启动：VLLM_BIND_ADDRESS 必须为 127.0.0.1。" >&2
  exit 3
fi

case "${MODEL_NAME:-}" in
  Qwen/Qwen3-VL-2B-Instruct|Qwen/Qwen3-VL-4B-Instruct)
    ;;
  *)
    echo "拒绝启动：仅允许已批准的 Qwen3-VL 2B/4B Instruct 检查点。" >&2
    exit 4
    ;;
esac

if [[ "${HF_HOME:-}" != /data/* ]]; then
  echo "拒绝启动：HF_HOME 必须位于 /data。" >&2
  exit 5
fi

mkdir -p "${HF_HOME}"

docker_command=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    docker_command=(sudo -n docker)
  else
    echo "无法访问 Docker；当前用户和免密 sudo 均不可用。" >&2
    exit 6
  fi
fi

"${docker_command[@]}" compose --env-file "${env_file}" -f "${compose_file}" config --quiet

case "${1:-up}" in
  up)
    exec "${docker_command[@]}" compose --env-file "${env_file}" -f "${compose_file}" up -d
    ;;
  down)
    exec "${docker_command[@]}" compose --env-file "${env_file}" -f "${compose_file}" down
    ;;
  logs)
    exec "${docker_command[@]}" compose --env-file "${env_file}" -f "${compose_file}" logs -f vllm-qwen3-vl
    ;;
  status)
    exec "${docker_command[@]}" compose --env-file "${env_file}" -f "${compose_file}" ps
    ;;
  *)
    echo "用法：$0 [up|down|logs|status]" >&2
    exit 64
    ;;
esac
