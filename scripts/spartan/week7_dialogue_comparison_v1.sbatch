#!/usr/bin/env bash
#SBATCH --job-name=trip-w7-dialogue-w6
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
: "${TRIP_WEEK6_PRODUCT_ADAPTER:?set the locked Week 6 product adapter}"
: "${TRIP_WEEK6_AFTER_SALES_ADAPTER:?set the locked Week 6 after-sales adapter}"
: "${TRIP_WEEK6_ITINERARY_ADAPTER:?set the locked Week 6 itinerary adapter}"
: "${TRIP_OUTPUT_DIR:?set a new dialogue-comparison output directory}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HOME="${TRIP_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/trip-w7-dialogue-w6-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/trip-w7-dialogue-w6-triton-${SLURM_JOB_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
source "${TRIP_VENV}/bin/activate"

python scripts/run_week7.py dialogue-week6-baseline-v1 \
  --comparison-config configs/week7/dialogue_comparison_v1.json \
  --dataset-dir "${TRIP_DIALOGUE_DATASET_DIR}" \
  --product-adapter "${TRIP_WEEK6_PRODUCT_ADAPTER}" \
  --after-sales-adapter "${TRIP_WEEK6_AFTER_SALES_ADAPTER}" \
  --itinerary-adapter "${TRIP_WEEK6_ITINERARY_ADAPTER}" \
  --output-dir "${TRIP_OUTPUT_DIR}"
sync
