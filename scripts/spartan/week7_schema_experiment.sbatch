#!/usr/bin/env bash
#SBATCH --job-name=trip-w7-schema
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated Week 7 checkout}"
: "${TRIP_VENV:?set the verified environment}"
: "${TRIP_SCHEMA_ENDPOINT:?set an already-started OpenAI-compatible endpoint}"
: "${TRIP_SERVED_MODEL:?set the exact served Qwen3-VL-8B model name}"
: "${TRIP_OUTPUT_DIR:?set a new schema experiment directory}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export PYTHONUNBUFFERED=1
source "${TRIP_VENV}/bin/activate"
python scripts/manage_week7.py validate-lock
python scripts/run_week7.py schema-experiment \
  --output-dir "${TRIP_OUTPUT_DIR}" \
  --endpoint "${TRIP_SCHEMA_ENDPOINT}" \
  --served-model "${TRIP_SERVED_MODEL}"
sync
