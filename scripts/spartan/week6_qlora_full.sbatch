#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-qlora-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set isolated project root}"
: "${TRIP_CONFIG:?set the versioned Week 6 config}"
: "${TRIP_SCENARIO:?set Week 6 scenario}"
: "${TRIP_TRAIN_INPUT:?set locked training JSONL}"
: "${TRIP_EVAL_INPUT:?set locked validation JSONL}"
: "${TRIP_OUTPUT_DIR:?set a new output directory}"
: "${TRIP_HF_HOME:?set isolated model cache}"
: "${TRIP_VENV:?set the project-scoped Spartan virtual environment}"
: "${TRIP_RUN_ID:?set a unique formal run ID}"

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
python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" check-environment
python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" validate-data \
  --scenario "${TRIP_SCENARIO}" --input "${TRIP_TRAIN_INPUT}"
python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" validate-data \
  --scenario "${TRIP_SCENARIO}" --input "${TRIP_EVAL_INPUT}"
train_args=(
  python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" train-full
  --scenario "${TRIP_SCENARIO}"
  --train-input "${TRIP_TRAIN_INPUT}"
  --eval-input "${TRIP_EVAL_INPUT}"
  --output-dir "${TRIP_OUTPUT_DIR}"
  --confirm-dataset-lock
)
if [[ -n "${TRIP_RESUME_FROM_CHECKPOINT:-}" ]]; then
  train_args+=(--resume-from-checkpoint "${TRIP_RESUME_FROM_CHECKPOINT}")
fi
"${train_args[@]}"
sync
