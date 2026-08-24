#!/usr/bin/env bash
#SBATCH --job-name=trip-system-repair-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated system-repair checkout}"
: "${TRIP_VENV:?set the verified CUDA virtual environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_INITIAL_ADAPTER_DIR:?set the verified checkpoint-226 adapter}"
: "${TRIP_OUTPUT_ROOT:?set a new development evaluation root}"

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

config="configs/system_repair/qwen3_vl_8b_system_repair_v1.json"
python scripts/manage_week7.py --config "${config}" validate-lock
python scripts/run_week7.py --config "${config}" check-environment

python scripts/run_week7.py --config "${config}" evaluate-development \
  --output-dir "${TRIP_OUTPUT_ROOT}/existing_adapter" \
  --run-id system_repair_20260824_v1_existing_adapter_development \
  --adapter-dir "${TRIP_INITIAL_ADAPTER_DIR}" \
  --model-role multitask_existing \
  --max-new-tokens 3072

python scripts/run_week7.py --config "${config}" evaluate-development \
  --output-dir "${TRIP_OUTPUT_ROOT}/zero_shot" \
  --run-id system_repair_dev_zero_shot_20260824_v4 \
  --model-role zero_shot \
  --max-new-tokens 3072
sync
