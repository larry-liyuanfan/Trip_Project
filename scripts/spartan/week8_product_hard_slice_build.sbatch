#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-hard-silver
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-trip-w8-hard-silver-%j.out
#SBATCH --error=slurm-trip-w8-hard-silver-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_DATA_VENV:?TRIP_DATA_VENV is required}"
cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
source "$TRIP_DATA_VENV/bin/activate"
python scripts/manage_week8_product_two_stage.py \
  --config "${TRIP_CONFIG:-configs/week8/product_two_stage_v1.json}" \
  build-hard-slice
sync
