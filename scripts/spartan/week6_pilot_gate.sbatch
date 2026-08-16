#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-pilot-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_PROJECT_ROOT:?set isolated project root}"
: "${TRIP_CONFIG:?set the versioned Week 6 config}"
: "${TRIP_PILOT_SUMMARY:?set the completed pilot summary}"
: "${TRIP_GATE_OUTPUT:?set a new gate output path}"
: "${TRIP_EXPECTED_PILOT_COMMIT:?set the exact pilot commit}"
: "${TRIP_GPU_TOTAL_MEMORY_GB:?set observed GPU memory in GiB}"
: "${TRIP_VENV:?set the project-scoped Spartan virtual environment}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
source "${TRIP_VENV}/bin/activate"
python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" gate-pilot \
  --summary "${TRIP_PILOT_SUMMARY}" \
  --scenario after_sales \
  --expected-git-commit "${TRIP_EXPECTED_PILOT_COMMIT}" \
  --gpu-total-memory-gb "${TRIP_GPU_TOTAL_MEMORY_GB}" \
  --output "${TRIP_GATE_OUTPUT}"
