#!/usr/bin/env bash
#SBATCH --job-name=trip-render-v10
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_RUN_DIR:?}"
: "${TRIP_V7_ADAPTER:?}"
: "${TRIP_V9_ADAPTER:?}"
: "${TRIP_IMPLEMENTATION_COMMIT:?}"
test ! -e "${TRIP_RUN_DIR}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HF_HOME="${TRIP_CACHE_ROOT}/huggingface"
export XDG_CACHE_HOME="${TRIP_CACHE_ROOT}/xdg"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd "${TRIP_PROJECT_ROOT}"
"${TRIP_VENV}/bin/python" scripts/run_card_render_diagnostic_v10.py \
  --output-dir "${TRIP_RUN_DIR}" --v7-adapter "${TRIP_V7_ADAPTER}" \
  --v9-adapter "${TRIP_V9_ADAPTER}" --implementation-commit "${TRIP_IMPLEMENTATION_COMMIT}"
