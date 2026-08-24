#!/usr/bin/env bash
#SBATCH --job-name=trip-system-repair-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated system-repair checkout}"
: "${TRIP_VENV:?set the verified CUDA virtual environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_INITIAL_ADAPTER_DIR:?set the verified checkpoint-226 adapter}"
: "${TRIP_OUTPUT_ROOT:?set a new development evaluation root}"
: "${TRIP_WEEK6_PRODUCT_ADAPTER:?set the verified Week 6 product adapter}"
: "${TRIP_WEEK6_AFTER_SALES_ADAPTER:?set the verified Week 6 after-sales adapter}"
: "${TRIP_WEEK6_ITINERARY_ADAPTER:?set the verified Week 6 itinerary adapter}"

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

config="${TRIP_CONFIG:-configs/system_repair/qwen3_vl_8b_system_repair_v2.json}"
python scripts/manage_week7.py --config "${config}" validate-lock
python scripts/run_week7.py --config "${config}" check-environment

python scripts/run_week7.py --config "${config}" evaluate-development \
  --output-dir "${TRIP_OUTPUT_ROOT}/existing_adapter" \
  --run-id system_repair_20260825_v2_existing_adapter_development \
  --adapter-dir "${TRIP_INITIAL_ADAPTER_DIR}" \
  --model-role multitask_existing

python scripts/run_week7.py --config "${config}" evaluate-development \
  --output-dir "${TRIP_OUTPUT_ROOT}/zero_shot" \
  --run-id system_repair_dev_zero_shot_20260825_v4 \
  --model-role zero_shot

python scripts/run_week7.py --config "${config}" evaluate-development \
  --output-dir "${TRIP_OUTPUT_ROOT}/week6_product" \
  --run-id system_repair_dev_existing_product_20260825_v4 \
  --adapter-dir "${TRIP_WEEK6_PRODUCT_ADAPTER}" \
  --model-role week6_single_task_adapter \
  --scenario image_product_search

python scripts/run_week7.py --config "${config}" evaluate-development \
  --output-dir "${TRIP_OUTPUT_ROOT}/week6_after_sales" \
  --run-id system_repair_dev_existing_after_sales_20260825_v4 \
  --adapter-dir "${TRIP_WEEK6_AFTER_SALES_ADAPTER}" \
  --model-role week6_single_task_adapter \
  --scenario after_sales

python scripts/run_week7.py --config "${config}" evaluate-development \
  --output-dir "${TRIP_OUTPUT_ROOT}/week6_itinerary" \
  --run-id system_repair_dev_existing_itinerary_20260825_v4 \
  --adapter-dir "${TRIP_WEEK6_ITINERARY_ADAPTER}" \
  --model-role week6_single_task_adapter \
  --scenario itinerary_planning

python scripts/run_week7.py --config "${config}" evaluate-week6-dialogue-development \
  --output-dir "${TRIP_OUTPUT_ROOT}/week6_dialogue" \
  --product-adapter "${TRIP_WEEK6_PRODUCT_ADAPTER}" \
  --after-sales-adapter "${TRIP_WEEK6_AFTER_SALES_ADAPTER}" \
  --itinerary-adapter "${TRIP_WEEK6_ITINERARY_ADAPTER}"

python scripts/run_week7.py --config "${config}" combine-week6-development \
  --product-metrics "${TRIP_OUTPUT_ROOT}/week6_product/metrics.json" \
  --after-sales-metrics "${TRIP_OUTPUT_ROOT}/week6_after_sales/metrics.json" \
  --itinerary-metrics "${TRIP_OUTPUT_ROOT}/week6_itinerary/metrics.json" \
  --dialogue-metrics "${TRIP_OUTPUT_ROOT}/week6_dialogue/metrics.json" \
  --output "${TRIP_OUTPUT_ROOT}/week6_routed_metrics.json"
sync
