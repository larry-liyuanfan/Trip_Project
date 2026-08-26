#!/usr/bin/env bash
#SBATCH --job-name=trip-system-model-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated system-repair checkout}"
: "${TRIP_VENV:?set the verified CUDA virtual environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_ADAPTER_DIR:?set the final-gate adapter directory}"
: "${TRIP_SMOKE_OUTPUT:?set the immutable smoke evidence path}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HOME="${TRIP_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
source "${TRIP_VENV}/bin/activate"

release_config="${TRIP_RELEASE_CONFIG:-configs/releases/qwen3_vl_system_v1.json}"
smoke_image="${TRIP_SMOKE_IMAGE:-data/samples/images/cafe_001.jpg}"

python scripts/run_week7.py \
  --config configs/system_repair/qwen3_vl_8b_system_repair_v2.json \
  check-environment
python scripts/run_system_model_smoke.py \
  --adapter-dir "${TRIP_ADAPTER_DIR}" \
  --release-config "${release_config}" \
  --image "${smoke_image}" \
  --output "${TRIP_SMOKE_OUTPUT}"
sync
