#!/usr/bin/env bash
#SBATCH --job-name=trip-w5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?set TRIP_PROJECT_ROOT to the isolated Trip_Project checkout}"
: "${TRIP_CONTAINER_CACHE:?set TRIP_CONTAINER_CACHE inside the isolated Trip project root}"

if [[ -n "${TRIP_MIGRATION_DIR:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  shard_index=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")
  export TRIP_CONFIG="${TRIP_MIGRATION_DIR}/shards/${shard_index}/config.json"
  export TRIP_RUN_ID
  TRIP_RUN_ID=$(python -c 'import json,os; m=json.load(open(os.path.join(os.environ["TRIP_MIGRATION_DIR"],"migration_manifest.json"))); print(m["run_ids"]["shards"][int(os.environ["SLURM_ARRAY_TASK_ID"])])')
fi
: "${TRIP_CONFIG:?set TRIP_CONFIG to one generated migration config}"
: "${TRIP_RUN_ID:?set TRIP_RUN_ID to the matching immutable run id}"

cd "${TRIP_PROJECT_ROOT}"
if [[ ! -f "${TRIP_CONFIG}" ]]; then
  echo "missing config: ${TRIP_CONFIG}" >&2
  exit 2
fi
if [[ -e "$(python -c 'import json,sys; from pathlib import Path; c=json.load(open(sys.argv[1])); print(Path(c["paths"]["output_dir"])/"runs"/sys.argv[2])' "${TRIP_CONFIG}" "${TRIP_RUN_ID}")" ]]; then
  echo "run already exists; set TRIP_RESUME=1 only after identity verification" >&2
  [[ "${TRIP_RESUME:-0}" == "1" ]] || exit 3
fi

export HF_HOME="${TRIP_HF_HOME:?set TRIP_HF_HOME to the isolated project cache}"
export WEEK5_MODEL_BASE_URL_OVERRIDE="http://127.0.0.1:8001/v1"
export PYTHONUNBUFFERED=1
export APPTAINER_CACHEDIR="${TRIP_CONTAINER_CACHE}"
export APPTAINER_TMPDIR="${TRIP_CONTAINER_CACHE}/tmp"
export TRIP_VLLM_IMAGE="${TRIP_VLLM_IMAGE:-docker://vllm/vllm-openai:v0.11.0}"
export APPTAINERENV_HF_HOME="${TRIP_HF_HOME}"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_WEEK5_MODEL_BASE_URL_OVERRIDE="${WEEK5_MODEL_BASE_URL_OVERRIDE}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

module purge
module load Apptainer/1.3.3

container_args=(
  exec --nv --cleanenv
  --bind "${TRIP_PROJECT_ROOT}:${TRIP_PROJECT_ROOT}"
  --bind "${TRIP_HF_HOME}:${TRIP_HF_HOME}"
  "${TRIP_VLLM_IMAGE}"
)

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" || true
  fi
  sync
}
trap cleanup EXIT INT TERM

apptainer "${container_args[@]}" python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --served-model-name Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  >"${TRIP_RUN_ID}.vllm.log" 2>&1 &
VLLM_PID=$!

healthy=0
for _ in $(seq 1 90); do
  if curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null; then
    healthy=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before health check passed" >&2
    exit 4
  fi
  sleep 10
done
[[ "${healthy}" == "1" ]] || { echo "vLLM health timeout" >&2; exit 5; }

args=(--config "${TRIP_CONFIG}" preannotate-all --run-id "${TRIP_RUN_ID}")
if [[ "${TRIP_RESUME:-0}" == "1" ]]; then
  args+=(--resume)
fi
apptainer "${container_args[@]}" python scripts/manage_week5_dataset.py "${args[@]}"
