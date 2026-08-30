#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-silver-audit
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-trip-w8-silver-audit-%j.out
#SBATCH --error=slurm-trip-w8-silver-audit-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_DATA_VENV:?TRIP_DATA_VENV is required}"
: "${TRIP_PHOTOS_ZIP:?TRIP_PHOTOS_ZIP is required}"
: "${TRIP_OUTPUT_DIR:?TRIP_OUTPUT_DIR is required}"
cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
module load tesseract/5.3.0
source "$TRIP_DATA_VENV/bin/activate"
python scripts/audit_week8_product_silver_source_v8.py \
  --config "${TRIP_CONFIG:-configs/week8/product_silver_source_audit_v8.json}" \
  audit \
  --output-dir "$TRIP_OUTPUT_DIR" \
  --photos-zip "$TRIP_PHOTOS_ZIP" \
  --run-ocr
sync
