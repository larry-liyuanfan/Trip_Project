#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-product-lock
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-trip-w8-product-lock-%j.out
#SBATCH --error=slurm-trip-w8-product-lock-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_SOURCE_PROJECT_ROOT:?TRIP_SOURCE_PROJECT_ROOT is required}"
: "${TRIP_VENV:?TRIP_VENV is required}"

cd "$TRIP_PROJECT_ROOT"
source "$TRIP_VENV/bin/activate"
python -m unittest tests.test_week8_product -v
python scripts/manage_week8_product.py build-lock \
  --source-project-root "$TRIP_SOURCE_PROJECT_ROOT"
python scripts/manage_week8_product.py validate-lock

