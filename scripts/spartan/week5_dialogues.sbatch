#!/usr/bin/env bash
#SBATCH --job-name=trip-w5-dialogues
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?set the isolated Trip_Project checkout}"
: "${TRIP_CONFIG:?set the single-operator Week 5 config}"
: "${TRIP_DIALOGUE_RUN_ID:?set a new immutable dialogue run id}"
: "${TRIP_HF_HOME:?set the isolated project model cache}"
: "${TRIP_CONTAINER_CACHE:?set the isolated Apptainer cache}"

cd "${TRIP_PROJECT_ROOT}"
[[ -f "${TRIP_CONFIG}" ]] || { echo "missing config: ${TRIP_CONFIG}" >&2; exit 2; }

export HF_HOME="${TRIP_HF_HOME}"
export WEEK5_MODEL_BASE_URL_OVERRIDE="http://127.0.0.1:8001/v1"
export PYTHONUNBUFFERED=1
export APPTAINER_CACHEDIR="${TRIP_CONTAINER_CACHE}"
export APPTAINER_TMPDIR="${TRIP_CONTAINER_CACHE}/tmp/${SLURM_JOB_ID:-manual}"
export TRIP_VLLM_IMAGE="${TRIP_VLLM_IMAGE:-docker://vllm/vllm-openai:v0.11.0}"
export TRIP_RUNTIME_HOME="${TRIP_RUNTIME_HOME:-${TRIP_HF_HOME}/runtime-home}"
export TRIP_RUNTIME_CACHE="${TRIP_RUNTIME_CACHE:-${TRIP_RUNTIME_HOME}/.cache}"
export APPTAINERENV_HF_HOME="${TRIP_HF_HOME}"
export APPTAINERENV_HOME="${TRIP_RUNTIME_HOME}"
export APPTAINERENV_XDG_CACHE_HOME="${TRIP_RUNTIME_CACHE}"
export APPTAINERENV_FLASHINFER_WORKSPACE_DIR="${TRIP_RUNTIME_CACHE}/flashinfer"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_WEEK5_MODEL_BASE_URL_OVERRIDE="${WEEK5_MODEL_BASE_URL_OVERRIDE}"
export APPTAINERENV_TRIP_DIALOGUE_CONCURRENCY="${TRIP_DIALOGUE_CONCURRENCY:-1}"
mkdir -p "${APPTAINER_TMPDIR}" "${TRIP_RUNTIME_CACHE}/flashinfer"

dialogue_run_id="${TRIP_DIALOGUE_RUN_ID}"
dialogue_shard_index="${TRIP_DIALOGUE_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  dialogue_run_id="${TRIP_DIALOGUE_RUN_ID}-s${SLURM_ARRAY_TASK_ID}"
fi

module purge
module load GCCcore/11.3.0
module load Apptainer/1.3.3

container_args=(
  exec --nv --cleanenv
  --home "${TRIP_RUNTIME_HOME}"
  --bind "${TRIP_PROJECT_ROOT}:${TRIP_PROJECT_ROOT}"
  --bind "${TRIP_HF_HOME}:${TRIP_HF_HOME}"
  "${TRIP_VLLM_IMAGE}"
)
if [[ "${TRIP_APPTAINER_DISABLE_CACHE:-0}" == "1" ]]; then
  # Spartan 的 Apptainer 不接受全局 --disable-cache；环境变量跨版本兼容。
  export APPTAINER_DISABLE_CACHE=true
fi
container_python="${TRIP_CONTAINER_PYTHON:-python3}"
apptainer "${container_args[@]}" /bin/sh -c \
  "test -w \"\${HOME}\" && command -v '${container_python}' >/dev/null && '${container_python}' -c 'import vllm'"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" || true
  fi
  sync
}
trap cleanup EXIT INT TERM

vllm_log="dialogue-${dialogue_run_id}.${SLURM_JOB_ID:-manual}.vllm.log"
[[ ! -e "${vllm_log}" ]] || { echo "vLLM log already exists: ${vllm_log}" >&2; exit 3; }
structured_outputs_config='{"backend":"xgrammar","disable_any_whitespace":true}'
apptainer "${container_args[@]}" "${container_python}" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --served-model-name Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --allowed-local-media-path "${TRIP_PROJECT_ROOT}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --structured-outputs-config "${structured_outputs_config}" \
  >"${vllm_log}" 2>&1 &
VLLM_PID=$!

healthy=0
for _ in $(seq 1 90); do
  if curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null; then
    healthy=1
    break
  fi
  kill -0 "${VLLM_PID}" 2>/dev/null || { echo "vLLM exited before health check" >&2; exit 4; }
  sleep 10
done
[[ "${healthy}" == "1" ]] || { echo "vLLM health timeout" >&2; exit 5; }

dialogue_args=(
  --config "${TRIP_CONFIG}" generate-dialogues
  --run-id "${dialogue_run_id}" --limit "${TRIP_DIALOGUE_LIMIT:-10000}"
  --start-index "${TRIP_DIALOGUE_START_INDEX:-0}"
  --shard-index "${dialogue_shard_index}"
  --shard-count "${TRIP_DIALOGUE_SHARD_COUNT:-1}"
)
if [[ -n "${TRIP_DIALOGUE_END_INDEX:-}" ]]; then
  dialogue_args+=(--end-index "${TRIP_DIALOGUE_END_INDEX}")
fi
if [[ "${TRIP_DIALOGUE_RESUME:-0}" == "1" ]]; then
  dialogue_args+=(--resume)
fi
apptainer "${container_args[@]}" "${container_python}" \
  scripts/manage_week5_dataset.py "${dialogue_args[@]}"
sync
