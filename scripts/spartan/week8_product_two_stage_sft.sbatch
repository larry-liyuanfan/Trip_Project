#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-two-sft
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s,gpu-a100
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:30:00
#SBATCH --output=slurm-trip-w8-two-sft-%j.out
#SBATCH --error=slurm-trip-w8-two-sft-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_VENV:?TRIP_VENV is required}"
: "${TRIP_HF_HOME:?TRIP_HF_HOME is required}"
: "${TRIP_INITIAL_ADAPTER_DIR:?TRIP_INITIAL_ADAPTER_DIR is required}"
: "${TRIP_OUTPUT_DIR:?TRIP_OUTPUT_DIR is required}"
: "${TRIP_RUN_ID:=week8_product_two_stage_continuation_sft_20260827_v1}"
cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HOME="$TRIP_HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
source "$TRIP_VENV/bin/activate"
arguments=(
  python scripts/manage_week8_product_two_stage.py
  --config "${TRIP_CONFIG:-configs/week8/product_two_stage_v1.json}"
  train --output-dir "$TRIP_OUTPUT_DIR"
)
if [[ -n "${TRIP_RESUME_FROM_CHECKPOINT:-}" ]]; then
  arguments+=(--resume-from-checkpoint "$TRIP_RESUME_FROM_CHECKPOINT")
fi
"${arguments[@]}"
sync
