#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-product-source
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-trip-w8-product-source-%j.out
#SBATCH --error=slurm-trip-w8-product-source-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_DEPLOY_ROOT:?TRIP_DEPLOY_ROOT is required}"
: "${TRIP_DATA_VENV:?TRIP_DATA_VENV is required}"
: "${TRIP_YELP_ARCHIVE_ROOT:?TRIP_YELP_ARCHIVE_ROOT is required}"
: "${TRIP_REBUILT_YELP_ROOT:?TRIP_REBUILT_YELP_ROOT is required}"
: "${TRIP_HISTORICAL_ROOT:?TRIP_HISTORICAL_ROOT is required}"

case "$TRIP_YELP_ARCHIVE_ROOT" in "$TRIP_DEPLOY_ROOT"/*) ;; *) exit 2 ;; esac
case "$TRIP_REBUILT_YELP_ROOT" in "$TRIP_PROJECT_ROOT"/*) ;; *) exit 2 ;; esac

module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export PYTHONUNBUFFERED=1
source "$TRIP_DATA_VENV/bin/activate"

photos_zip="$TRIP_YELP_ARCHIVE_ROOT/Yelp-Photos-20250115.zip"
test "$(stat -c %s "$photos_zip")" = 7447210067

cd "$TRIP_PROJECT_ROOT"
python scripts/build_week8_product_fresh_sources.py \
  --config "${TRIP_CONFIG:-configs/week8/product_understanding_v2.json}" \
  --rebuilt-yelp-root "$TRIP_REBUILT_YELP_ROOT" \
  --historical-root "$TRIP_HISTORICAL_ROOT" \
  --photos-zip "$photos_zip"
sync
