#!/usr/bin/env bash
#SBATCH --job-name=trip-w7-latency-v5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated Week 7 checkout}"
: "${TRIP_VENV:?set the verified CUDA environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_OUTPUT_DIR:?set a new immutable protocol-v5 output directory}"
: "${TRIP_PROTOCOL_CONFIG:?set the locked evaluation protocol v5 config}"
: "${TRIP_TRAINING_DIR:?set the completed v3 training directory}"
: "${TRIP_TRAINING_SUMMARY:?set the completed v3 training summary}"
: "${TRIP_WEEK6_BASELINE:?set the bound v3 combined baseline}"
: "${TRIP_WEEK6_BASELINE_EVIDENCE:?set the checked-in v3 baseline evidence}"
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

# 只验证 train/development；使用 BF16 + static cache + compile，不读取 test。
python scripts/manage_week7.py validate-lock
python scripts/run_week7.py check-environment
python scripts/run_week7.py latency-protocol-v5 \
  --output-dir "${TRIP_OUTPUT_DIR}" \
  --protocol-config "${TRIP_PROTOCOL_CONFIG}" \
  --training-dir "${TRIP_TRAINING_DIR}" \
  --training-summary "${TRIP_TRAINING_SUMMARY}" \
  --week6-baseline "${TRIP_WEEK6_BASELINE}" \
  --week6-baseline-evidence "${TRIP_WEEK6_BASELINE_EVIDENCE}" \
  --week6-product-adapter "${TRIP_WEEK6_PRODUCT_ADAPTER}" \
  --week6-after-sales-adapter "${TRIP_WEEK6_AFTER_SALES_ADAPTER}" \
  --week6-itinerary-adapter "${TRIP_WEEK6_ITINERARY_ADAPTER}"
sync
