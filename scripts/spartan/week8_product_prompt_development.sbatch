#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-product-dev
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s,gpu-a100
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=slurm-trip-w8-product-dev-%j.out
#SBATCH --error=slurm-trip-w8-product-dev-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_VENV:?TRIP_VENV is required}"
: "${TRIP_ADAPTER_DIR:?TRIP_ADAPTER_DIR is required}"
: "${HF_HOME:?HF_HOME is required}"

cd "$TRIP_PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
source "$TRIP_VENV/bin/activate"
python scripts/manage_week8_product.py validate-lock
python scripts/manage_week8_product.py prompt-development \
  --output-dir outputs/week8/development/week8_product_prompt_development_20260826_v1
