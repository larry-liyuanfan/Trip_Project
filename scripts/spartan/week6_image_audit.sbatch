#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-image-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set isolated project root}"
: "${TRIP_IMAGE_MANIFEST:?set immutable image manifest}"
: "${TRIP_AUDIT_OUTPUT_DIR:?set new shard output directory}"
: "${TRIP_AUDIT_SHARD_COUNT:?set array shard count}"
: "${TRIP_VENV:?set project virtual environment}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as an array}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
source "${TRIP_VENV}/bin/activate"
mkdir -p "${TRIP_AUDIT_OUTPUT_DIR}"
python scripts/week6_image_manifest.py audit-shard \
  --manifest "${TRIP_IMAGE_MANIFEST}" \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --shard-count "${TRIP_AUDIT_SHARD_COUNT}" \
  --output "${TRIP_AUDIT_OUTPUT_DIR}/audit-${SLURM_ARRAY_TASK_ID}.json"
