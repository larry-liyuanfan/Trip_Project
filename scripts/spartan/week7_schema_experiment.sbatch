#!/usr/bin/env bash
#SBATCH --job-name=trip-w7-schema
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated Week 7 checkout}"
: "${TRIP_VENV:?set the verified environment}"
: "${TRIP_HF_HOME:?set the existing isolated model cache}"
: "${TRIP_CONTAINER_CACHE:?set the isolated Apptainer cache}"
: "${TRIP_SERVED_MODEL:?set the exact served Qwen3-VL-8B model name}"
: "${TRIP_OUTPUT_DIR:?set a new schema experiment directory}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load Apptainer/1.3.3
export HF_HOME="${TRIP_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_PYTHONUNBUFFERED=1
mkdir -p "${APPTAINER_TMPDIR}" "${TRIP_RUNTIME_CACHE}/flashinfer"
source "${TRIP_VENV}/bin/activate"
python scripts/manage_week7.py validate-lock

container_args=(
  exec --nv --cleanenv
  --home "${TRIP_RUNTIME_HOME}"
  --bind "${TRIP_PROJECT_ROOT}:${TRIP_PROJECT_ROOT}"
  --bind "${TRIP_HF_HOME}:${TRIP_HF_HOME}"
  "${TRIP_VLLM_IMAGE}"
)
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

endpoint="http://127.0.0.1:8001"
mkdir -p "$(dirname "${TRIP_OUTPUT_DIR}")"
vllm_log="${TRIP_OUTPUT_DIR}.vllm.${SLURM_JOB_ID:-manual}.log"
[[ ! -e "${vllm_log}" ]] || { echo "vLLM log already exists: ${vllm_log}" >&2; exit 3; }
structured_outputs_config='{"backend":"xgrammar","disable_any_whitespace":true}'
apptainer "${container_args[@]}" "${container_python}" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name "${TRIP_SERVED_MODEL}" \
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
  if curl -fsS --max-time 5 "${endpoint}/v1/models" >/dev/null; then
    healthy=1
    break
  fi
  kill -0 "${VLLM_PID}" 2>/dev/null || { echo "vLLM exited before health check" >&2; exit 4; }
  sleep 10
done
[[ "${healthy}" == "1" ]] || { echo "vLLM health timeout" >&2; exit 5; }

python scripts/run_week7.py schema-experiment \
  --output-dir "${TRIP_OUTPUT_DIR}" \
  --endpoint "${endpoint}" \
  --served-model "${TRIP_SERVED_MODEL}"
sync
