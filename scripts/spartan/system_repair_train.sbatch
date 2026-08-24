#!/usr/bin/env bash
#SBATCH --job-name=trip-system-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated system-repair checkout}"
: "${TRIP_VENV:?set the verified CUDA virtual environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_INITIAL_ADAPTER_DIR:?set the verified checkpoint-226 adapter}"
: "${TRIP_OUTPUT_DIR:?set a new immutable run directory}"
: "${TRIP_RUN_ID:=system_repair_continuation_sft_20260824_v4}"

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

config="${TRIP_CONFIG:-configs/system_repair/qwen3_vl_8b_system_repair_v2.json}"
python scripts/manage_system_repair.py --config "${config}" validate-config
arguments=(
  python scripts/manage_system_repair.py --config "${config}" train
  --output-dir "${TRIP_OUTPUT_DIR}"
)
if [[ -n "${TRIP_RESUME_FROM_CHECKPOINT:-}" ]]; then
  arguments+=(--resume-from-checkpoint "${TRIP_RESUME_FROM_CHECKPOINT}")
fi
"${arguments[@]}"
sync
