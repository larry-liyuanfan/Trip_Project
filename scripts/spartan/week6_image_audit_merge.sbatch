#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-image-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set isolated project root}"
: "${TRIP_AUDIT_OUTPUT_DIR:?set shard output directory}"
: "${TRIP_AUDIT_SHARD_COUNT:?set expected shard count}"
: "${TRIP_AUDIT_MERGED_OUTPUT:?set merged audit output}"
: "${TRIP_VENV:?set project virtual environment}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
source "${TRIP_VENV}/bin/activate"
python scripts/week6_image_manifest.py merge-audits \
  --input-dir "${TRIP_AUDIT_OUTPUT_DIR}" \
  --shard-count "${TRIP_AUDIT_SHARD_COUNT}" \
  --output "${TRIP_AUDIT_MERGED_OUTPUT}"
