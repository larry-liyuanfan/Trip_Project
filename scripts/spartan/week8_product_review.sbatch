#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-review
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s,gpu-a100
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=slurm-trip-w8-review-%j.out
#SBATCH --error=slurm-trip-w8-review-%j.err
set -euo pipefail
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_HF_HOME:?}"
: "${TRIP_ADAPTER_DIR:?}"
: "${TRIP_OUTPUT_DIR:?}"
cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HF_HOME="$TRIP_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$TRIP_PROJECT_ROOT/outputs/week8/review_cache/$SLURM_JOB_ID"
export TMPDIR="$XDG_CACHE_HOME/tmp"
mkdir -p "$TMPDIR"
source "$TRIP_VENV/bin/activate"
python scripts/review_week8_product.py \
  --config "${TRIP_REVIEW_CONFIG:-configs/week8/product_review_v1.json}" \
  --output-dir "$TRIP_OUTPUT_DIR"
python scripts/run_system_model_smoke.py --adapter-dir "$TRIP_ADAPTER_DIR" \
  --release-config configs/releases/qwen3_vl_system_week8_v7.json \
  --output "$TRIP_OUTPUT_DIR/model_smoke.json"
python scripts/benchmark_week8_runtime.py --adapter-dir "$TRIP_ADAPTER_DIR" \
  --release-config configs/releases/qwen3_vl_system_week8_v7.json \
  --benchmark-config "${TRIP_RUNTIME_CONFIG:-configs/week8/runtime_review_v1.json}" \
  --output "$TRIP_OUTPUT_DIR/runtime.json"
