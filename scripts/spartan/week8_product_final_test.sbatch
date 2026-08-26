#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-product-test
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s,gpu-a100
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-trip-w8-product-test-%j.out
#SBATCH --error=slurm-trip-w8-product-test-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_VENV:?TRIP_VENV is required}"
: "${TRIP_ADAPTER_DIR:?TRIP_ADAPTER_DIR is required}"
: "${HF_HOME:?HF_HOME is required}"

cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
source "$TRIP_VENV/bin/activate"
config="${TRIP_CONFIG:-configs/week8/product_understanding_v1.json}"
development_dir="${TRIP_DEVELOPMENT_DIR:-outputs/week8/development/week8_product_prompt_development_20260826_v1}"
final_test_dir="${TRIP_FINAL_TEST_DIR:-outputs/week8/final_test/week8_product_final_test_once_20260826_v1}"
python scripts/manage_week8_product.py --config "$config" validate-lock
python scripts/manage_week8_product.py --config "$config" final-test-once \
  --development-dir "$development_dir" \
  --output-dir "$final_test_dir"
