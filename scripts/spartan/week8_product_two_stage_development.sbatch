#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-two-stage
#SBATCH --account=punim2936
#SBATCH --partition=deeplearn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=slurm-trip-w8-two-stage-%j.out
#SBATCH --error=slurm-trip-w8-two-stage-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_VENV:?TRIP_VENV is required}"
: "${TRIP_HF_HOME:?TRIP_HF_HOME is required}"
: "${TRIP_INITIAL_ADAPTER_DIR:?TRIP_INITIAL_ADAPTER_DIR is required}"
: "${TRIP_OUTPUT_DIR:?TRIP_OUTPUT_DIR is required}"
cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HOME="$TRIP_HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TRIP_ADAPTER_DIR="$TRIP_INITIAL_ADAPTER_DIR"
source "$TRIP_VENV/bin/activate"
python scripts/manage_week8_product_two_stage.py \
  --config "${TRIP_CONFIG:-configs/week8/product_two_stage_v1.json}" \
  development --output-dir "$TRIP_OUTPUT_DIR"
sync
