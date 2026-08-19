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

recover_current_signal() {
  local signal_name="${1}"
  trap - TERM INT
  set +e
  python scripts/run_week7.py recover-final-test \
    --parameter-lock "${TRIP_PARAMETER_LOCK}" \
    --output-dir "${TRIP_OUTPUT_DIR}" \
    --slurm-job-id "${SLURM_JOB_ID}" \
    --slurm-job-state "SIGNAL_TERM"
  exit 143
}
trap 'recover_current_signal TERM' TERM
trap 'recover_current_signal INT' INT

# 这里的 validate-lock 故意不读取 test；final-test 会先原子占用一次性门禁再读取。
python scripts/manage_week7.py validate-lock
python scripts/run_week7.py check-environment
if [[ "${TRIP_FINAL_TEST_RECOVER_STALE:-0}" == "1" ]]; then
  : "${TRIP_FINAL_TEST_STALE_JOB_ID:?set the exact interrupted Slurm job ID}"
  observed_state="$({
    sacct -X -n -P -j "${TRIP_FINAL_TEST_STALE_JOB_ID}" --format=JobIDRaw,State \
      | awk -F'|' -v job_id="${TRIP_FINAL_TEST_STALE_JOB_ID}" '$1 == job_id {print $2; exit}'
  } || true)"
  if [[ -z "${observed_state}" ]]; then
    echo "Cannot verify interrupted Slurm job state" >&2
    exit 2
  fi
  python scripts/run_week7.py recover-final-test \
    --parameter-lock "${TRIP_PARAMETER_LOCK}" \
    --output-dir "${TRIP_OUTPUT_DIR}" \
    --slurm-job-id "${TRIP_FINAL_TEST_STALE_JOB_ID}" \
    --slurm-job-state "${observed_state}"
  export TRIP_FINAL_TEST_RESUME=1
fi
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
