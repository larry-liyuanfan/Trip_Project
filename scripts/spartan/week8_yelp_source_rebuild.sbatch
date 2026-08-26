#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-yelp-source
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-trip-w8-yelp-source-%j.out
#SBATCH --error=slurm-trip-w8-yelp-source-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_DEPLOY_ROOT:?TRIP_DEPLOY_ROOT is required}"
: "${TRIP_DATA_VENV:?TRIP_DATA_VENV is required}"
: "${TRIP_YELP_ARCHIVE_ROOT:?TRIP_YELP_ARCHIVE_ROOT is required}"
: "${TRIP_EXISTING_IMAGE_ROOT:?TRIP_EXISTING_IMAGE_ROOT is required}"
: "${TRIP_OUTPUT_YELP_ROOT:?TRIP_OUTPUT_YELP_ROOT is required}"
: "${TRIP_REBUILD_MANIFEST:?TRIP_REBUILD_MANIFEST is required}"

case "$TRIP_YELP_ARCHIVE_ROOT" in "$TRIP_DEPLOY_ROOT"/*) ;; *) exit 2 ;; esac
case "$TRIP_OUTPUT_YELP_ROOT" in "$TRIP_PROJECT_ROOT"/*) ;; *) exit 2 ;; esac

module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export PYTHONUNBUFFERED=1
source "$TRIP_DATA_VENV/bin/activate"
mkdir -p "$TRIP_YELP_ARCHIVE_ROOT"

json_zip="$TRIP_YELP_ARCHIVE_ROOT/Yelp-JSON-20250115.zip"
photos_zip="$TRIP_YELP_ARCHIVE_ROOT/Yelp-Photos-20250115.zip"
curl --fail --location --retry 5 --continue-at - \
  --output "$json_zip" \
  https://business.yelp.com/external-assets/files/Yelp-JSON.zip
curl --fail --location --retry 5 --continue-at - \
  --output "$photos_zip" \
  https://business.yelp.com/external-assets/files/Yelp-Photos.zip

test "$(stat -c %s "$json_zip")" = 4345335132
test "$(stat -c %s "$photos_zip")" = 7447210067

cd "$TRIP_PROJECT_ROOT"
python scripts/rebuild_week8_yelp_sources.py \
  --json-zip "$json_zip" \
  --photos-zip "$photos_zip" \
  --existing-image-root "$TRIP_EXISTING_IMAGE_ROOT" \
  --output-yelp-root "$TRIP_OUTPUT_YELP_ROOT" \
  --manifest "$TRIP_REBUILD_MANIFEST" \
  --workers "$SLURM_CPUS_PER_TASK"
