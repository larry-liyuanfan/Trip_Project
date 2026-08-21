#!/usr/bin/env bash
#SBATCH --job-name=trip-w7-dialogue-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated Week 7 checkout}"
: "${TRIP_VENV:?set the verified CUDA environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_DIALOGUE_DATASET_DIR:?set the corrected development-only dialogue directory}"
: "${TRIP_SELECTED_CHECKPOINT:?set the locked checkpoint-151 directory}"
: "${TRIP_OUTPUT_DIR:?set a new dialogue-review output directory}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HOME="${TRIP_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/trip-w7-dialogue-review-v2-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/trip-w7-dialogue-review-v2-triton-${SLURM_JOB_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
source "${TRIP_VENV}/bin/activate"

python scripts/run_week7.py dialogue-review-v2 \
  --review-config configs/week7/dialogue_review_v2.json \
  --dataset-dir "${TRIP_DIALOGUE_DATASET_DIR}" \
  --adapter-dir "${TRIP_SELECTED_CHECKPOINT}" \
  --output-dir "${TRIP_OUTPUT_DIR}"
sync
