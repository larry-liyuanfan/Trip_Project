#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-retrieval-test
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-trip-w8-retrieval-test-%j.out
#SBATCH --error=slurm-trip-w8-retrieval-test-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_RETRIEVAL_DIR:?TRIP_RETRIEVAL_DIR is required}"
: "${TRIP_RETRIEVAL_LOCK_DIR:?TRIP_RETRIEVAL_LOCK_DIR is required}"
: "${TRIP_RETRIEVAL_SELECTION:?TRIP_RETRIEVAL_SELECTION is required}"
: "${TRIP_RETRIEVAL_FINAL_DIR:?TRIP_RETRIEVAL_FINAL_DIR is required}"
: "${TRIP_RETRIEVAL_TEST_MARKER:?TRIP_RETRIEVAL_TEST_MARKER is required}"
: "${TRIP_VENV:?TRIP_VENV is required}"

cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
source "$TRIP_VENV/bin/activate"

python scripts/manage_week8_retrieval.py validate-lock \
  --lock-dir "$TRIP_RETRIEVAL_LOCK_DIR"
python scripts/manage_week8_retrieval.py evaluate-final-test \
  --retrieval-dir "$TRIP_RETRIEVAL_DIR" \
  --lock-dir "$TRIP_RETRIEVAL_LOCK_DIR" \
  --selection "$TRIP_RETRIEVAL_SELECTION" \
  --output-dir "$TRIP_RETRIEVAL_FINAL_DIR" \
  --consumption-marker "$TRIP_RETRIEVAL_TEST_MARKER" \
  --retrieval-backend auto
