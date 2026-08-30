#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-retrieval-photos
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-trip-w8-retrieval-photos-%j.out
#SBATCH --error=slurm-trip-w8-retrieval-photos-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_VENV:?TRIP_VENV is required}"
: "${TRIP_RETRIEVAL_METADATA:?TRIP_RETRIEVAL_METADATA is required}"
: "${TRIP_YELP_PHOTOS_ZIP:?TRIP_YELP_PHOTOS_ZIP is required}"
: "${TRIP_RETRIEVAL_PHOTO_ROOT:?TRIP_RETRIEVAL_PHOTO_ROOT is required}"

cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
source "$TRIP_VENV/bin/activate"

python scripts/extract_week8_retrieval_photos.py \
  --metadata "$TRIP_RETRIEVAL_METADATA" \
  --photos-zip "$TRIP_YELP_PHOTOS_ZIP" \
  --output-root "$TRIP_RETRIEVAL_PHOTO_ROOT"
