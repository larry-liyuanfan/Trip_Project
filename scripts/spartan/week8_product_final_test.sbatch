#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-product-test
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s,gpu-a100
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-trip-w8-product-test-%j.out
#SBATCH --error=slurm-trip-w8-product-test-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_VENV:?TRIP_VENV is required}"
: "${TRIP_ADAPTER_DIR:?TRIP_ADAPTER_DIR is required}"
: "${HF_HOME:?HF_HOME is required}"

cd "$TRIP_PROJECT_ROOT"
source "$TRIP_VENV/bin/activate"
python scripts/manage_week8_product.py validate-lock
python scripts/manage_week8_product.py final-test-once \
  --development-dir outputs/week8/development/week8_product_prompt_development_20260826_v1 \
  --output-dir outputs/week8/final_test/week8_product_final_test_once_20260826_v1
