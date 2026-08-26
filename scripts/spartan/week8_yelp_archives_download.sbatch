#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-yelp-download
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-trip-w8-yelp-download-%j.out
#SBATCH --error=slurm-trip-w8-yelp-download-%j.err

set -euo pipefail

: "${TRIP_DEPLOY_ROOT:?TRIP_DEPLOY_ROOT is required}"
: "${TRIP_YELP_ARCHIVE_ROOT:?TRIP_YELP_ARCHIVE_ROOT is required}"
case "$TRIP_YELP_ARCHIVE_ROOT" in "$TRIP_DEPLOY_ROOT"/*) ;; *) exit 2 ;; esac

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

