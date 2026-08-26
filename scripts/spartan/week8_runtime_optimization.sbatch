#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-runtime-opt
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s,gpu-a100
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated Week 8 checkout}"
: "${TRIP_VENV:?set the verified CUDA virtual environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_ADAPTER_DIR:?set the formal release adapter directory}"
: "${TRIP_RUNTIME_OUTPUT:?set the immutable Week 8 runtime evidence path}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HOME="${TRIP_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/trip-w8-runtime-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/trip-w8-runtime-triton-${SLURM_JOB_ID}"
source "${TRIP_VENV}/bin/activate"

python scripts/run_week7.py \
  --config configs/system_repair/qwen3_vl_8b_system_repair_v2.json \
  check-environment
python scripts/benchmark_week8_runtime.py \
  --adapter-dir "${TRIP_ADAPTER_DIR}" \
  --release-config configs/releases/qwen3_vl_system_week8_v1.json \
  --output "${TRIP_RUNTIME_OUTPUT}"
sync
