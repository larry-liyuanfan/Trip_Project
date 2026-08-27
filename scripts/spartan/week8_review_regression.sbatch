#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-review-regression
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s,gpu-a100
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_HF_HOME:?}"
: "${TRIP_ADAPTER_DIR:?}"
: "${TRIP_OUTPUT_DIR:?}"
: "${TRIP_SMOKE_IMAGE:?set the locked real development photo}"
: "${TRIP_SMOKE_IMAGE_SHA256:?set its verified SHA-256}"
cd "$TRIP_PROJECT_ROOT"
if [[ -e "$TRIP_OUTPUT_DIR" ]]; then
  printf '%s\n' 'Refusing to overwrite regression evidence' >&2
  exit 1
fi
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HF_HOME="$TRIP_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$TRIP_PROJECT_ROOT/outputs/week8/review_cache/$SLURM_JOB_ID"
export TMPDIR="$XDG_CACHE_HOME/tmp"
mkdir -p "$TMPDIR"
source "$TRIP_VENV/bin/activate"
python scripts/run_system_model_smoke.py --adapter-dir "$TRIP_ADAPTER_DIR" \
  --image "$TRIP_SMOKE_IMAGE" --expected-image-sha256 "$TRIP_SMOKE_IMAGE_SHA256" \
  --release-config configs/releases/qwen3_vl_system_week8_v7.json \
  --output "$TRIP_OUTPUT_DIR/model_smoke.json"
python scripts/benchmark_week8_runtime.py --adapter-dir "$TRIP_ADAPTER_DIR" \
  --release-config configs/releases/qwen3_vl_system_week8_v7.json \
  --benchmark-config configs/week8/runtime_review_v3.json \
  --output "$TRIP_OUTPUT_DIR/runtime.json"
