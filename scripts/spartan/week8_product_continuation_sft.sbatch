#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-product-sft
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s,gpu-a100
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated Week 8 checkout}"
: "${TRIP_VENV:?set the verified CUDA virtual environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_INITIAL_ADAPTER_DIR:?set the verified release checkpoint-87 adapter}"
: "${TRIP_PROMPT_DEVELOPMENT_DIR:?set the completed Prompt development directory}"
: "${TRIP_OUTPUT_DIR:?set a new immutable SFT run directory}"
: "${TRIP_RUN_ID:=week8_product_continuation_sft_20260826_v1}"

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

config="${TRIP_CONFIG:-configs/week8/product_continuation_sft_v1.json}"
python scripts/manage_week8_product_sft.py --config "${config}" validate-config
python scripts/manage_week8_product_sft.py --config "${config}" validate-eligibility \
  --prompt-development-dir "${TRIP_PROMPT_DEVELOPMENT_DIR}"
arguments=(
  python scripts/manage_week8_product_sft.py --config "${config}" train
  --prompt-development-dir "${TRIP_PROMPT_DEVELOPMENT_DIR}"
  --output-dir "${TRIP_OUTPUT_DIR}"
)
if [[ -n "${TRIP_RESUME_FROM_CHECKPOINT:-}" ]]; then
  arguments+=(--resume-from-checkpoint "${TRIP_RESUME_FROM_CHECKPOINT}")
fi
"${arguments[@]}"
python scripts/manage_week8_product_sft.py --config "${config}" select-candidate \
  --prompt-development-dir "${TRIP_PROMPT_DEVELOPMENT_DIR}" \
  --training-dir "${TRIP_OUTPUT_DIR}"
sync
