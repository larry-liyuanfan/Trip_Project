#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-candidate-probe
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
: "${TRIP_PROJECT_ROOT:?}" "${TRIP_VENV:?}" "${TRIP_HF_HOME:?}" "${TRIP_ADAPTER_DIR:?}"
cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HF_HOME="$TRIP_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$TRIP_PROJECT_ROOT/outputs/week8/review_cache/$SLURM_JOB_ID"
export TMPDIR="$XDG_CACHE_HOME/tmp"
mkdir -p "$TMPDIR"
source "$TRIP_VENV/bin/activate"
set +e
python scripts/verify_week8_candidate_runtime.py --config "${TRIP_RUNTIME_PROBE_CONFIG:-configs/week8/candidate_runtime_probe_v1.json}"
status=$?
if [[ "$status" != 0 ]]; then
  printf '%s\n' 'Application failed; allocation retained for a bounded 180-second diagnosis window.'
  sleep 180
fi
exit "$status"
