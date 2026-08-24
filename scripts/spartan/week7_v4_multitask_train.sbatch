#!/usr/bin/env bash
#SBATCH --job-name=trip-w7-v4-multitask
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated Week 7 checkout}"
: "${TRIP_VENV:?set the verified CUDA virtual environment}"
: "${TRIP_HF_HOME:?set the existing isolated model cache}"
: "${TRIP_CONFIG:?set the locked Week 7 v4 config}"
: "${TRIP_OUTPUT_DIR:?set a new Week 7 v4 run directory}"
: "${TRIP_RUN_ID:?set the locked Week 7 v4 run ID}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HOME="${TRIP_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export TRIP_GIT_COMMIT="$(git rev-parse HEAD)"
source "${TRIP_VENV}/bin/activate"

# validate-lock 只读取 train/development，不消费 test。
python scripts/manage_week7.py --config "${TRIP_CONFIG}" validate-lock
python scripts/run_week7.py --config "${TRIP_CONFIG}" check-environment
arguments=(
  python scripts/run_week7.py --config "${TRIP_CONFIG}" train-multitask
  --output-dir "${TRIP_OUTPUT_DIR}"
  --confirm-dataset-lock
)
if [[ -n "${TRIP_RESUME_FROM_CHECKPOINT:-}" ]]; then
  arguments+=(--resume-from-checkpoint "${TRIP_RESUME_FROM_CHECKPOINT}")
fi
"${arguments[@]}"
sync

