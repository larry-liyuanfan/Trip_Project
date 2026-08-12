#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-qlora-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set isolated project root}"
: "${TRIP_SCENARIO:?set Week 6 scenario}"
: "${TRIP_TRAIN_INPUT:?set locked training JSONL}"
: "${TRIP_EVAL_INPUT:?set locked validation JSONL}"
: "${TRIP_OUTPUT_DIR:?set a new output directory}"
: "${TRIP_HF_HOME:?set isolated model cache}"
: "${TRIP_VENV:?set the project-scoped Spartan virtual environment}"

cd "${TRIP_PROJECT_ROOT}"
export HF_HOME="${TRIP_HF_HOME}"
export PYTHONUNBUFFERED=1
source "${TRIP_VENV}/bin/activate"
python scripts/train_week6_qlora.py check-environment
python scripts/train_week6_qlora.py validate-data \
  --scenario "${TRIP_SCENARIO}" --input "${TRIP_TRAIN_INPUT}"
python scripts/train_week6_qlora.py train-pilot \
  --scenario "${TRIP_SCENARIO}" \
  --train-input "${TRIP_TRAIN_INPUT}" \
  --eval-input "${TRIP_EVAL_INPUT}" \
  --output-dir "${TRIP_OUTPUT_DIR}" \
  --confirm-dataset-lock
sync
