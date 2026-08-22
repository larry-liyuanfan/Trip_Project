#!/usr/bin/env bash
#SBATCH --job-name=trip-w7-v4-dialogue-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated Week 7 checkout}"
: "${TRIP_VENV:?set the verified CUDA environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_CONFIG:?set the locked Week 7 v4 config}"
: "${TRIP_SELECTION:?set the passed v4 development selection artifact}"
: "${TRIP_OUTPUT_DIR:?set the new one-shot corrected-dialogue test output directory}"
: "${TRIP_WEEK6_PRODUCT_ADAPTER:?set the locked Week 6 product adapter}"
: "${TRIP_WEEK6_AFTER_SALES_ADAPTER:?set the locked Week 6 after-sales adapter}"
: "${TRIP_WEEK6_ITINERARY_ADAPTER:?set the locked Week 6 itinerary adapter}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HOME="${TRIP_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
source "${TRIP_VENV}/bin/activate"

# 此检查不读取 test；下一个命令会先原子占用 marker，再读取 corrected dialogue test。
python scripts/manage_week7.py --config "${TRIP_CONFIG}" validate-lock
python scripts/run_week7.py --config "${TRIP_CONFIG}" check-environment
python scripts/run_week7.py --config "${TRIP_CONFIG}" v4-dialogue-test \
  --selection "${TRIP_SELECTION}" \
  --output-dir "${TRIP_OUTPUT_DIR}" \
  --week6-product-adapter "${TRIP_WEEK6_PRODUCT_ADAPTER}" \
  --week6-after-sales-adapter "${TRIP_WEEK6_AFTER_SALES_ADAPTER}" \
  --week6-itinerary-adapter "${TRIP_WEEK6_ITINERARY_ADAPTER}"
sync

