#!/usr/bin/env bash
#SBATCH --job-name=trip-w7-dev-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated Week 7 checkout}"
: "${TRIP_VENV:?set the verified CUDA environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_OUTPUT_DIR:?set a new development output directory}"
: "${TRIP_RUN_ID:?set a unique development run ID}"
: "${TRIP_MODEL_ROLE:?set zero_shot or the exact Week 6 adapter role}"

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

python scripts/manage_week7.py validate-lock
python scripts/run_week7.py check-environment
arguments=(
  python scripts/run_week7.py evaluate-development
  --output-dir "${TRIP_OUTPUT_DIR}"
  --run-id "${TRIP_RUN_ID}"
  --model-role "${TRIP_MODEL_ROLE}"
)
if [[ -n "${TRIP_ADAPTER_DIR:-}" ]]; then
  arguments+=(--adapter-dir "${TRIP_ADAPTER_DIR}")
fi
if [[ -n "${TRIP_SCENARIO:-}" ]]; then
  arguments+=(--scenario "${TRIP_SCENARIO}")
fi
"${arguments[@]}"
sync
