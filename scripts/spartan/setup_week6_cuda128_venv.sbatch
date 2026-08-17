#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-env-cu128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_DEPLOY_ROOT:?set isolated Trip deployment root}"
: "${TRIP_PROJECT_ROOT:?set Trip repository root}"
: "${TRIP_VENV:?set a new versioned Week 6 environment}"

case "${TRIP_PROJECT_ROOT}" in
  "${TRIP_DEPLOY_ROOT}"/*) ;;
  *) echo "TRIP_PROJECT_ROOT must be below TRIP_DEPLOY_ROOT" >&2; exit 2 ;;
esac
case "${TRIP_VENV}" in
  "${TRIP_DEPLOY_ROOT}"/envs/*) ;;
  *) echo "TRIP_VENV must be below TRIP_DEPLOY_ROOT/envs" >&2; exit 2 ;;
esac
if [[ -e "${TRIP_VENV}" ]]; then
  echo "refusing to mutate an existing environment: ${TRIP_VENV}" >&2
  exit 2
fi

module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export PIP_CACHE_DIR="${TRIP_DEPLOY_ROOT}/pip-cache"
export PIP_NO_CACHE_DIR=1
export XDG_CACHE_HOME="${TRIP_DEPLOY_ROOT}/xdg-cache"
export HF_HOME="${TRIP_DEPLOY_ROOT}/huggingface"
export TMPDIR="${TRIP_DEPLOY_ROOT}/tmp"
mkdir -p "${TRIP_DEPLOY_ROOT}/envs" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" \
  "${HF_HOME}" "${TMPDIR}"

python -m venv "${TRIP_VENV}"
"${TRIP_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${TRIP_VENV}/bin/python" -m pip install \
  'torch==2.8.0+cu128' --index-url https://download.pytorch.org/whl/cu128
"${TRIP_VENV}/bin/python" -m pip install \
  -r "${TRIP_PROJECT_ROOT}/requirements-training-spartan-cu128.txt"
"${TRIP_VENV}/bin/python" -m pip check
"${TRIP_VENV}/bin/python" -c \
  'import importlib.metadata, accelerate, bitsandbytes, kernels, peft, torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__, peft.__version__, bitsandbytes.__version__, importlib.metadata.version("kernels"))'
sync
