#!/usr/bin/env bash
#SBATCH --job-name=trip-system-repair-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set the isolated system-repair checkout}"
: "${TRIP_VENV:?set the verified CUDA virtual environment}"
: "${TRIP_HF_HOME:?set the existing model cache}"
: "${TRIP_SELECTION:?set the hash-bound candidate selection}"
: "${TRIP_GATE:?set the passed development gate}"
: "${TRIP_OUTPUT_DIR:?set the immutable one-shot test output}"

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
python scripts/manage_system_repair.py --config "${config}" run-final-test \
  --selection "${TRIP_SELECTION}" \
  --gate "${TRIP_GATE}" \
  --output-dir "${TRIP_OUTPUT_DIR}"
sync
