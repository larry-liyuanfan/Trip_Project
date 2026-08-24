#!/usr/bin/env bash
#SBATCH --job-name=trip-system-repair-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated system-repair checkout}"
: "${TRIP_VENV:?set the verified CUDA virtual environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_ADAPTER_DIR:?set the verified checkpoint-226 adapter}"
: "${TRIP_PROMPT_OUTPUT_DIR:?set a new immutable prompt pilot directory}"
: "${TRIP_WEEK5_RUN_ID:?set a new immutable Week 5 repair run ID}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HOME="${TRIP_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export TRIP_RELEASE_CONFIG="configs/releases/qwen3_vl_system_v1.json"
source "${TRIP_VENV}/bin/activate"

python scripts/manage_system_repair.py \
  --config configs/system_repair/qwen3_vl_8b_system_repair_v1.json \
  validate-config
python scripts/manage_week7.py \
  --config configs/system_repair/qwen3_vl_8b_system_repair_v1.json \
  validate-lock

arguments=(
  python scripts/manage_system_repair.py \
  --config configs/system_repair/qwen3_vl_8b_system_repair_v1.json \
  run-inference-repair \
  --output-dir "${TRIP_PROMPT_OUTPUT_DIR}" \
  --run-id "${TRIP_WEEK5_RUN_ID}" \
  --adapter-dir "${TRIP_ADAPTER_DIR}"
)
if [[ "${TRIP_RESUME:-0}" == "1" ]]; then
  arguments+=(--resume)
fi
"${arguments[@]}"
sync
