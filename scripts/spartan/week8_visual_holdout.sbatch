#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-visual-holdout
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
: "${TRIP_PROJECT_ROOT:?}" "${TRIP_DATA_VENV:?}"
cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0 Python/3.11.3
source "$TRIP_DATA_VENV/bin/activate"
python scripts/build_week8_visual_holdout.py --config "${TRIP_FINAL_CONFIG:-configs/week8/visual_final_v1.json}"
