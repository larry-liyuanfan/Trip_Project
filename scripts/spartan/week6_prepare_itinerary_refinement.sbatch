#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-itin-refine-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set isolated project root}"
: "${TRIP_SOURCE_TRAIN:?set the immutable source training JSONL}"
: "${TRIP_SOURCE_VALIDATION:?set the immutable source validation JSONL}"
: "${TRIP_OUTPUT_DIR:?set a new refinement lock directory}"
: "${TRIP_DATASET_VERSION:?set the refinement dataset version}"
: "${TRIP_VENV:?set the project-scoped Spartan virtual environment}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export PYTHONUNBUFFERED=1
source "${TRIP_VENV}/bin/activate"

if [[ -e "${TRIP_OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing refinement lock: ${TRIP_OUTPUT_DIR}" >&2
  exit 20
fi

python scripts/prepare_week6_itinerary_refinement.py \
  --train-input "${TRIP_SOURCE_TRAIN}" \
  --validation-input "${TRIP_SOURCE_VALIDATION}" \
  --output-dir "${TRIP_OUTPUT_DIR}" \
  --dataset-version "${TRIP_DATASET_VERSION}" \
  --silver-weight 0.5

python scripts/train_week6_qlora.py \
  --config configs/week6/qwen3_vl_8b_qlora_itinerary_refinement_v1.json \
  validate-data --scenario itinerary_planning \
  --input "${TRIP_OUTPUT_DIR}/itinerary_planning/train.jsonl"
python scripts/train_week6_qlora.py \
  --config configs/week6/qwen3_vl_8b_qlora_itinerary_refinement_v1.json \
  validate-data --scenario itinerary_planning \
  --input "${TRIP_OUTPUT_DIR}/itinerary_planning/validation.jsonl"
