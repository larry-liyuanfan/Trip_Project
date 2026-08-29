#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-retrieval-v8
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
: "${TRIP_PROJECT_ROOT:?}" "${TRIP_VENV:?}" "${TRIP_RETRIEVAL_CONFIG:?}"
cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0 Python/3.11.3
source "$TRIP_VENV/bin/activate"
python scripts/verify_week8_retrieval_routing.py --config "$TRIP_RETRIEVAL_CONFIG"
