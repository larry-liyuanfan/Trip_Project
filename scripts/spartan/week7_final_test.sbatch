#!/usr/bin/env bash
#SBATCH --job-name=trip-w7-final-test
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
: "${TRIP_PARAMETER_LOCK:?set the immutable parameter lock}"
: "${TRIP_OUTPUT_DIR:?set the single final-test output directory}"

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

# 这里的 validate-lock 故意不读取 test；final-test 会先原子占用一次性门禁再读取。
python scripts/manage_week7.py validate-lock
python scripts/run_week7.py check-environment
arguments=(
  python scripts/run_week7.py final-test
  --parameter-lock "${TRIP_PARAMETER_LOCK}"
  --output-dir "${TRIP_OUTPUT_DIR}"
)
if [[ "${TRIP_FINAL_TEST_RESUME:-0}" == "1" ]]; then
  arguments+=(--resume)
fi
"${arguments[@]}"
sync
