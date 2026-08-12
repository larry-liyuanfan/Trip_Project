#!/usr/bin/env bash
#SBATCH --job-name=trip-env-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${TRIP_DEPLOY_ROOT:?set the isolated Trip deployment root}"
: "${TRIP_PROJECT_ROOT:?set the isolated Trip repository root}"
: "${TRIP_VENV:?set TRIP_VENV below the isolated Trip deployment root}"

case "${TRIP_PROJECT_ROOT}" in
  "${TRIP_DEPLOY_ROOT}"/*) ;;
  *) echo "TRIP_PROJECT_ROOT must be below TRIP_DEPLOY_ROOT" >&2; exit 2 ;;
esac
case "${TRIP_VENV}" in
  "${TRIP_DEPLOY_ROOT}"/envs/*) ;;
  *) echo "TRIP_VENV must be below TRIP_DEPLOY_ROOT/envs" >&2; exit 2 ;;
esac

module purge
module load GCCcore/11.3.0
module load Python/3.11.3

export PIP_CACHE_DIR="${TRIP_DEPLOY_ROOT}/pip-cache"
export XDG_CACHE_HOME="${TRIP_DEPLOY_ROOT}/xdg-cache"
export HF_HOME="${TRIP_DEPLOY_ROOT}/huggingface"
export TMPDIR="${TRIP_DEPLOY_ROOT}/tmp"
mkdir -p "${TRIP_DEPLOY_ROOT}/envs" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TMPDIR}"

if [[ ! -x "${TRIP_VENV}/bin/python" ]]; then
  python -m venv "${TRIP_VENV}"
fi

"${TRIP_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${TRIP_VENV}/bin/python" -m pip install \
  -r "${TRIP_PROJECT_ROOT}/requirements.txt" \
  -r "${TRIP_PROJECT_ROOT}/requirements-training.txt"
"${TRIP_VENV}/bin/python" -m pip check
"${TRIP_VENV}/bin/python" -c 'import accelerate, bitsandbytes, peft, torch, transformers; print(torch.__version__, transformers.__version__)'
sync
